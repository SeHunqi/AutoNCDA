import argparse
import importlib
import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


MASTER_PATH = "data/disease_master_final_merged.csv"
OBO_PATH = "data/doid.obo"
N_FOLDS = 5
EMBED_DIM = 128


def parse_fold_list(raw: str):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    folds = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError(f"fold must be positive, got {v}")
        folds.append(v)
    if not folds:
        raise ValueError("empty fold list")
    return sorted(set(folds))


def parse_lambda_list(raw: str):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    lambdas = []
    for p in parts:
        v = float(p)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"lambda must be in [0,1], got {v}")
        lambdas.append(round(v, 2))
    if not lambdas:
        raise ValueError("empty lambda list")
    return sorted(set(lambdas))


def get_ancestors(graph, node):
    if node not in graph:
        return set()
    return nx.descendants(graph, node) | {node}


def load_ontology(obo_path):
    try:
        obonet = importlib.import_module("obonet")
    except ImportError as e:
        raise ImportError("obonet is required. Install with: pip install obonet") from e

    print("Loading Disease Ontology...")
    graph = obonet.read_obo(obo_path)
    node_ancestors = {}
    for node in tqdm(graph.nodes(), desc="Precompute ancestors"):
        node_ancestors[node] = get_ancestors(graph, node)
    return node_ancestors


def calc_jaccard_matrix(doids, node_ancestors):
    n = len(doids)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    doid_anc_sets = [node_ancestors.get(d, set()) for d in doids]

    print("Computing semantic Jaccard matrix...")
    for i in tqdm(range(n)):
        set_i = doid_anc_sets[i]
        if not set_i:
            continue
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            set_j = doid_anc_sets[j]
            if not set_j:
                continue
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            score = intersection / union if union > 0 else 0.0
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score
    return sim_matrix


def calc_gip_sim_from_csv(train_csv_path, all_rnas, all_doids):
    df = pd.read_csv(train_csv_path)

    if "Association_Label" in df.columns:
        pos_df = df[df["Association_Label"] == 1]
    elif "Label" in df.columns:
        pos_df = df[df["Label"] == 1]
    else:
        raise ValueError(f"CSV {train_csv_path} missing label column")

    n_r = len(all_rnas)
    n_d = len(all_doids)

    rna_map = {r: i for i, r in enumerate(all_rnas)}
    doid_map = {d: i for i, d in enumerate(all_doids)}

    valid_rows = []
    valid_cols = []
    skipped = 0

    urs_ids = pos_df["URS_ID"].values
    do_ids = pos_df["DO_ID"].values

    for r, d in zip(urs_ids, do_ids):
        if r in rna_map and d in doid_map:
            valid_rows.append(rna_map[r])
            valid_cols.append(doid_map[d])
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Warning: skipped {skipped} unmatched associations")

    adj = np.zeros((n_r, n_d), dtype=np.float32)
    if valid_rows:
        adj[valid_rows, valid_cols] = 1.0

    x = adj.T
    norm_sq = np.sum(x ** 2, axis=1)
    avg_norm = np.mean(norm_sq)
    gamma = 1.0 / avg_norm if avg_norm > 0 else 1.0

    dot_prod = np.dot(x, x.T)
    dist_sq = norm_sq[:, np.newaxis] + norm_sq[np.newaxis, :] - 2 * dot_prod
    dist_sq = np.maximum(dist_sq, 0)

    return np.exp(-gamma * dist_sq)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Generate fold-specific DSS (Disease Semantic Similarity) features."
    )
    parser.add_argument(
        "--lambdas", type=str, default="0.5",
        help="Comma-separated lambda values in [0,1]. "
             "DSS = λ·Semantic + (1-λ)·GIP. Default: 0.5",
    )
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM,
                        help=f"SVD embedding dimension (default: {EMBED_DIM}).")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--folds", type=str, default=None,
        help="Comma-separated fold indices to process, e.g. '1,3,5'. "
             "Default: all 1..n-folds.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    split_dir = os.path.join(repo_root, "fold_splits")
    out_base_dir = os.path.join(repo_root, "fold_dss_features")
    lambdas = parse_lambda_list(args.lambdas)

    print(f"Generating DSS features: lambdas={lambdas}")

    df_master = pd.read_csv(os.path.join(repo_root, MASTER_PATH))
    all_doids = sorted(df_master["DO_ID"].dropna().unique())
    all_rnas = sorted(df_master["URS_ID"].dropna().unique())
    print(f"  Diseases: {len(all_doids)}, RNAs: {len(all_rnas)}")

    ensure_dir(out_base_dir)

    sem_cache_path = os.path.join(out_base_dir, "semantic_sim_matrix.npy")
    if os.path.exists(sem_cache_path):
        print(f"  Loading cached semantic similarity matrix...")
        sem_sim = np.load(sem_cache_path)
    else:
        ancestors_map = load_ontology(os.path.join(repo_root, OBO_PATH))
        sem_sim = calc_jaccard_matrix(all_doids, ancestors_map)
        np.save(sem_cache_path, sem_sim)

    fold_list = (
        parse_fold_list(args.folds)
        if args.folds else list(range(1, args.n_folds + 1))
    )
    if args.folds and any(f > args.n_folds for f in fold_list):
        raise ValueError(f"Some folds exceed --n-folds={args.n_folds}")

    for fold_idx in fold_list:
        print(f"\nFold {fold_idx}/{args.n_folds}")
        train_csv = os.path.join(split_dir, f"disease_fold_{fold_idx}_train.csv")

        if not os.path.exists(train_csv):
            print(f"  Missing split file: {train_csv}, skip")
            continue

        gip_sim = calc_gip_sim_from_csv(train_csv, all_rnas, all_doids)

        for lam in lambdas:
            fused_matrix = lam * sem_sim + (1.0 - lam) * gip_sim
            svd = TruncatedSVD(n_components=args.embed_dim, random_state=42)
            dss_emb = svd.fit_transform(fused_matrix)

            out_dir = os.path.join(out_base_dir, lambda_tag(lam))
            ensure_dir(out_dir)
            pd.Series(all_doids).to_csv(
                os.path.join(out_dir, "doid_order.csv"),
                index=False,
                header=["DO_ID"],
            )
            save_path = os.path.join(out_dir, f"dss_fold_{fold_idx}.npy")
            np.save(save_path, dss_emb)
            print(f"  lambda={lam:.2f} -> {save_path} shape={dss_emb.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()

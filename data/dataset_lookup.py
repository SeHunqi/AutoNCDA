
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SafeFastLookupDataset(Dataset):
   
    def __init__(
        self,
        split_csv_path: str,
        rna_feat_path: str,
        rna_map_path: str,
        dss_feat_path: str,
        doid_map_path: str,
        task: str = 'association',
        target_dim: int = 960,
        rna_type_filter: str = None,
        dynamic_negatives: bool = False,
    ):
        self.task = task
        self.target_dim = target_dim
        self.dynamic_negatives = dynamic_negatives
        self.rna_type_filter = rna_type_filter

        # 1. Load ID → index maps
        df_rmap = pd.read_csv(rna_map_path)
        self.rna_map = dict(zip(df_rmap['URS_ID'], df_rmap['index']))

        df_dmap = pd.read_csv(doid_map_path)
        doid_col = 'DO_ID' if 'DO_ID' in df_dmap.columns else df_dmap.columns[0]
        self.doid_map = {d: i for i, d in enumerate(df_dmap[doid_col])}
        self.num_diseases = len(self.doid_map)
        self.all_doid_indices = list(self.doid_map.values())

        # 2. Memory-map feature matrices (lazy load)
        if not os.path.exists(rna_feat_path):
            raise FileNotFoundError(f"RNA feature file not found: {rna_feat_path}")
        self.rna_features = np.load(rna_feat_path, mmap_mode='r')

        if not os.path.exists(dss_feat_path):
            raise FileNotFoundError(f"DSS feature file not found: {dss_feat_path}")
        self.dss_features = np.load(dss_feat_path, mmap_mode='r')

        # 3. Load and optionally filter the split CSV
        self.df_split = pd.read_csv(split_csv_path)

        if rna_type_filter:
            type_cols = [c for c in self.df_split.columns if 'type' in c.lower()]
            if type_cols:
                col = type_cols[0]
                self.df_split = self.df_split[self.df_split[col] == rna_type_filter]
            else:
                import warnings
                warnings.warn(
                    f"rna_type_filter='{rna_type_filter}' requested but no RNA type "
                    f"column found in {os.path.basename(split_csv_path)}."
                )

        # 4. Parse samples
        self.samples = []       # static mode: list of (r_idx, d_idx, label)
        self.pos_indices = []   # dynamic mode: positive (r_idx, d_idx) pairs
        self.known_pairs = set()
        seen_rna_indices = set()

        lbl_col = 'Association_Label' if 'Association_Label' in self.df_split.columns else 'Label'

        for _, row in self.df_split.iterrows():
            urs = row['URS_ID']
            doid = row['DO_ID']
            label = float(row[lbl_col])
            is_pos = (label == 1.0)

            r_idx = self.rna_map.get(urs, -1)
            d_idx = self.doid_map.get(doid, -1)
            if r_idx == -1 or d_idx == -1:
                continue

            if self.dynamic_negatives:
                if is_pos:
                    self.pos_indices.append((r_idx, d_idx))
                    self.known_pairs.add((r_idx, d_idx))
                    seen_rna_indices.add(r_idx)
            else:
                self.samples.append((r_idx, d_idx, label))

        if self.dynamic_negatives:
            self.rna_pool = list(seen_rna_indices)
        else:
            pass  # static mode

    def __len__(self) -> int:
        if self.dynamic_negatives:
            return len(self.pos_indices) * 2  # 1:1 positive-to-negative ratio
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        # --- Dynamic negative sampling ---
        if self.dynamic_negatives:
            n_pos = len(self.pos_indices)

            if idx < n_pos:
                # First half: fixed positive samples
                r_idx, d_idx = self.pos_indices[idx]
                label = 1.0
            else:
                # Second half: sample a negative pair via random corruption
                anchor_idx = idx % n_pos
                r_anchor, d_anchor = self.pos_indices[anchor_idx]
                label = 0.0
                r_neg, d_neg = r_anchor, d_anchor

                for _ in range(100):
                    if random.random() < 0.5:
                        # Strategy A: keep RNA, swap disease
                        d_cand = random.choice(self.all_doid_indices)
                        if (r_anchor, d_cand) not in self.known_pairs:
                            r_neg, d_neg = r_anchor, d_cand
                            break
                    else:
                        # Strategy B: keep disease, swap RNA (within same type)
                        r_cand = random.choice(self.rna_pool)
                        if (r_cand, d_anchor) not in self.known_pairs:
                            r_neg, d_neg = r_cand, d_anchor
                            break

                r_idx, d_idx = r_neg, d_neg

        # --- Static read ---
        else:
            r_idx, d_idx, label = self.samples[idx]

        # --- Feature extraction ---
        feat_rna = self.rna_features[r_idx].copy()
        feat_dss = self.dss_features[d_idx].copy()
        feat = np.concatenate([feat_rna, feat_dss])

        # Zero-pad if needed
        if feat.shape[0] < self.target_dim:
            pad = np.zeros(self.target_dim - feat.shape[0], dtype=feat.dtype)
            feat = np.concatenate([feat, pad])

        return {
            'features': torch.from_numpy(feat).float(),
            'association_label': torch.tensor(label, dtype=torch.float32),
            'doid_idx': torch.tensor(d_idx, dtype=torch.long),
        }

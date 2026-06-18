import argparse
import itertools
import json
import math
import os
import random
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.dataloader_factory import SafeFastDataLoaderFactory
from models.model import MMoEModel, StaticMMoEModel
from training.losses import search_best_threshold
from training.metrics import _clean_for_json, _pad_or_truncate
from training.search_trainer import DARTSLikeSearcherV1
from training.trainer import DecoupledTrainer, set_family_class_weight
from utils.arch_utils import CANDIDATE_NAMES


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_folder(config: dict) -> str:
    """Create a timestamped experiment directory encoding key hyperparameters."""
    from datetime import datetime

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_dir = os.path.join(repo_root, "training_results")
    if config.get("results_dir"):
        override = str(config["results_dir"])
        base_dir = override if os.path.isabs(override) else os.path.join(repo_root, override)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = (
        f"fold{config['fold_idx']}"
        f"_seed{config.get('seed', 42)}_{ts}"
    )
    exp_dir = os.path.join(base_dir, name)
    for sub in ("", "checkpoints", "best_models", "logs"):
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir



def run_nas_search(
    config: dict,
    loaders: dict,
    device: str,
    actual_input_dim: int,
    experiment_dir: str):
    fam_train_dl, fam_val_dl, _ = loaders['family']
    asc_train_dl, asc_val_dl, _ = loaders['association']

    logs_dir = os.path.join(experiment_dir, "search_logs")
    os.makedirs(logs_dir, exist_ok=True)

    search_model_config = {
        'input_dim': actual_input_dim,
        'num_families': config['num_families'],
        'mmoe': config['mmoe'],
        'nas': {**config['nas'], 'micro_search': True, 'adapter_search': True,
                'micro_expert_choices': None},
        'family_dim': config.get('family_dim', 832),
    }
    model = MMoEModel(search_model_config).to(device)

    nas_cfg = config.get('nas', {})
    searcher = DARTSLikeSearcherV1(
        model, device, target_input_dim=actual_input_dim,
        w_lr=nas_cfg.get('w_lr', 5e-4),
        w_wd=nas_cfg.get('w_wd', 2e-4),
        alpha_lr=nas_cfg.get('alpha_lr', 4e-3),
        alpha_wd=nas_cfg.get('alpha_wd', 5e-3),
        entropy_reg=nas_cfg.get('entropy_reg', 8e-3),
        diversity_reg=nas_cfg.get('diversity_reg', 5e-2),
    )

    # pos_weight for association BCE loss
    pos_cnt = neg_cnt = 0
    for batch in asc_train_dl:
        labels = batch['association_label']
        pos_cnt += int((labels == 1).sum())
        neg_cnt += int((labels == 0).sum())
    pos_weight = neg_cnt / max(pos_cnt, 1)
    searcher.bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    fam_tr_it = iter(itertools.cycle(fam_train_dl))
    asc_tr_it = iter(itertools.cycle(asc_train_dl))
    fam_vl_it = iter(itertools.cycle(fam_val_dl))
    asc_vl_it = iter(itertools.cycle(asc_val_dl))

    steps_train = max(len(fam_train_dl), len(asc_train_dl))
    steps_val = max(1, math.ceil(steps_train * 0.5))
    search_epochs = int(nas_cfg.get('search_epochs', 20))
    warmup_epochs = int(nas_cfg.get('warmup_epochs', 10))

    print(f"[NAS] Starting search: {search_epochs} epochs "
          f"(warmup={warmup_epochs})")

    for ep in range(search_epochs):
        update_alpha = (ep >= warmup_epochs)
        phase = "Warmup" if not update_alpha else "Search"
        print(f"  Search epoch {ep + 1}/{search_epochs} [{phase}]")
        searcher.train_epoch(
            fam_train_iter=fam_tr_it, assoc_train_iter=asc_tr_it,
            fam_val_iter=fam_vl_it, assoc_val_iter=asc_vl_it,
            steps_train=steps_train, steps_val=steps_val,
            task_weights=(
                config.get('family_loss_weight', 1.0),
                config.get('association_loss_weight', 1.0),
            ),
            update_alpha=update_alpha,
        )
        searcher.log_alpha(logs_dir, ep + 1)
        searcher.anneal_temperature(factor=0.96, min_t=0.20)

    arch = searcher.export_arch()

    if arch.get('micro_expert_choices'):
        def _name(idx):
            return CANDIDATE_NAMES[idx] if isinstance(idx, int) and idx < len(CANDIDATE_NAMES) else str(idx)
        arch['micro_expert_names'] = [
            [_name(x) for x in item] if isinstance(item, list) else _name(item)
            for item in arch['micro_expert_choices']
        ]

    arch_path = os.path.join(logs_dir, "exported_arch.json")
    with open(arch_path, "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2, ensure_ascii=False)
    print(f"[NAS] Architecture saved: {arch_path}")

    cfg = dict(config)
    cfg.setdefault('nas', {})
    cfg['nas'].update({
        'micro_search': False,
        'adapter_search': False,
        'share_search': False,
        'micro_expert_choices': arch.get('micro_expert_choices'),
        'adapter_family_choice': arch.get('adapter_family_choice'),
        'adapter_association_choice': arch.get('adapter_association_choice'),
    })
    return cfg



def main(
    fold_idx: int = 1,
    seed: int = 42,
    arch_file: str = None,
    force_search: bool = False,
    search_epochs: int = None,
    inner_repeats: int = 1,
    results_dir: str = None,
    num_epochs: int = 100,
    micro_layers: int = 2,
    hidden_dim: int = 256,
    num_experts: int = 6,
):
    set_global_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    feature_config = {
        'rna_fm_dim': 640,
        'kmer_vocab': 64,
        'structure_dim': 128,
        'dss_dim': 128,
        'total_dim': 960,
    }
    family_dim = (feature_config['rna_fm_dim']
                  + feature_config['kmer_vocab']
                  + feature_config['structure_dim'])  # 832

    config = {
        'input_dim': 960,
        'family_dim': family_dim,
        'feature_config': feature_config,

        'dataloader_config': {
            'batch_size': 1024,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True,
        },

        'nas': {
            'micro_search': True,
            'adapter_search': True,
            'share_search': True,
            'micro_expert_choices': None,
            'micro_layers': int(micro_layers),
            'micro_drop': 0.1,
            'micro_tau': 1.5,
            'micro_hard': False,
            'search_epochs': 20,
            'warmup_epochs': 10,
            'entropy_reg': 8e-3,
            'diversity_reg': 5e-2,
            'val_ratio': 0.18,
            'w_lr': 5e-4,
            'alpha_lr': 4e-3,
            'alpha_wd': 5e-3,
            'adapter_candidates': [0, 16, 32, 48, 64],
            'forbid_identity_first_layer': True,
        },

        'optimizer': {
            'shared_lr': 1e-4,
            'family_lr': 2e-4,
            'association_lr': 5e-5,
            'weight_decay': 2e-2,
            'warmup_epochs': 10,
        },

        'router': {
            'tau': 1.0,
            'tau_warmup': 2.5,
            'eps_warmup': 0.10,
            'noise_std': 0.10,
            'noise_warmup_std': 0.20,
        },

        'w_bal': 0.05,
        'w_bal_warmup': 0.20,
        'w_bal_decay_epochs': 30,
        'w_con': 0.1,

        'use_manual_task_weights': True,
        'family_loss_weight': 1.0,
        'association_loss_weight': 1.0,

        'num_families': 6,
        'hidden_dim': hidden_dim,
        'weight_decay': 2e-2,
        'max_grad_norm': 0.5,
        'use_amp': False,
        'use_mixup': True,
        'training_dropout': 0.3,

        'num_epochs': num_epochs,
        'early_stop_patience': 10,
        'family_patience': 10,
        'association_patience': 10,

        'fold_idx': fold_idx,
        'arch_file': arch_file,
        'seed': seed,
        'results_dir': results_dir,
        'dss_lambda': 0.5,
    }

    config.setdefault('mmoe', {
        'num_experts': num_experts,
        'expert_hidden_dim': hidden_dim,
        'expert_output_dim': hidden_dim // 2,
        'tower_hidden_dim': hidden_dim // 4,
    })

    if search_epochs is not None:
        config['nas']['search_epochs'] = int(search_epochs)

  
    if arch_file and not force_search:
        with open(arch_file, 'r') as f:
            arch_data = json.load(f)
        searched = arch_data.get('micro_expert_choices', [])
        if searched:
            config['nas']['micro_expert_choices'] = [
                searched[i % len(searched)] for i in range(num_experts)
            ]
        config['nas']['adapter_family_choice'] = arch_data.get('adapter_family_choice', 0)
        config['nas']['adapter_association_choice'] = arch_data.get('adapter_association_choice', 0)
        config['nas']['micro_search'] = False
        config['nas']['adapter_search'] = False
        config['nas']['share_search'] = False


    try:
        factory = SafeFastDataLoaderFactory(config)
        (fam_train_loader, fam_val_loader, fam_test_loader,
         asc_train_loader, asc_val_loader, asc_test_loader) = factory.create_fold_loaders(
            fold_idx=fold_idx,
            val_ratio=config['nas'].get('val_ratio', 0.18),
        )
    except Exception as e:
        print(f"[Error] DataLoader creation failed: {e}")
        traceback.print_exc()
        return

 
    exp_dir = create_experiment_folder(config)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dims = factory.get_feature_dims(fold_idx)
    actual_input_dim = max(dims['family'], dims['association'])
    config['input_dim'] = actual_input_dim


    if not arch_file and not force_search:
        raise ValueError(
            "Error: You must EITHER provide a searched architecture file (--arch-file) "
            "OR enable architecture search (--force-search) before training."
        )

  
    should_search = force_search
    if should_search:
        robust_loaders = {
            'family': (fam_train_loader, fam_val_loader, None),
            'association': (asc_train_loader, asc_val_loader, None),
        }
        config = run_nas_search(config, robust_loaders, device, actual_input_dim, exp_dir)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

  
    print(f"[Fold {fold_idx}] Building model (input_dim={actual_input_dim})...")
    try:
        if not config['nas']['micro_search']:
            model_dims = {
                'expert_output_dim': config['mmoe']['expert_output_dim'],
                'tower_hidden_dim': config['mmoe']['tower_hidden_dim'],
                'dropout_rate': config.get('training_dropout', 0.3),
                'expert_hidden_dim': config['mmoe']['expert_hidden_dim'],
            }
            model = StaticMMoEModel.from_nas_choices(
                input_dim=actual_input_dim,
                num_families=config['num_families'],
                nas_cfg=config['nas'],
                dims=model_dims,
                family_dim=dims['family'],
                router_cfg=config.get('router', {}),
                disable_mmoe=False,
            )
        else:
            model = MMoEModel(config)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}  |  Type: {type(model).__name__}")
    except Exception as e:
        print(f"[Error] Model creation failed: {e}")
        traceback.print_exc()
        return

    trainer = DecoupledTrainer(model, config, device=device)
    trainer.target_input_dim = actual_input_dim
    set_family_class_weight(trainer, fam_train_loader, config['num_families'])


    pos_cnt = neg_cnt = 0
    for b in asc_train_loader:
        y = b['association_label']
        pos_cnt += int((y == 1).sum())
        neg_cnt += int((y == 0).sum())
    if pos_cnt > 0:
        ratio = neg_cnt / pos_cnt
        trainer.association_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(ratio, device=device, dtype=torch.float32)
        )
        print(f"  Association pos_weight={ratio:.3f}")


    best_family_acc = -float('inf')
    best_assoc_auc = -float('inf')
    patience_family = 0
    patience_assoc = 0
    best_val_metrics = None
    fam_pat_th = int(config.get('family_patience', 10))
    asso_pat_th = int(config.get('association_patience', 10))
    early_patience = int(config.get('early_stop_patience', 10))
    family_frozen = False
    assoc_frozen = False

    log_file = os.path.join(exp_dir, "logs", "training.log")
    with open(log_file, 'w') as f:
        f.write("epoch,family_acc,assoc_auc,loss\n")

    print(f"\n[Fold {fold_idx}] Training for {config['num_epochs']} epochs "
          f"(early-stop patience={early_patience})...\n")

    for epoch in range(config['num_epochs']):
        try:
            trainer.apply_warmup()
            trainer.train_epoch(fam_train_loader, asc_train_loader)
            trainer.step_scheduler()

            eval_res = trainer.evaluate(fam_val_loader, asc_val_loader)
            trainer.update_schedulers(eval_res)

            family_acc = eval_res.get('family_acc', float('nan'))
            assoc_auc = eval_res.get('association_auc', float('nan'))

            # Threshold search on validation predictions
            if '_raw_assoc_probs' in eval_res and '_raw_assoc_labels' in eval_res:
                thr_info = search_best_threshold(
                    eval_res['_raw_assoc_probs'],
                    eval_res['_raw_assoc_labels'],
                    metric='youden',
                    n_points=200,
                    min_specificity=0.6,
                )
                eval_res['association_best_thr'] = thr_info['thr']
                eval_res['association_f1_best_thr'] = thr_info['f1']

            # Print one-line epoch summary
            print(
                f"  Epoch {epoch+1:3d}/{config['num_epochs']} | "
                f"Fam ACC={family_acc:.4f} | "
                f"Assoc AUC={assoc_auc:.4f} | "
                f"Loss={trainer.last_epoch_loss:.4f} | "
                f"Pat F={patience_family} A={patience_assoc}"
            )

            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{family_acc:.4f},{assoc_auc:.4f},"
                        f"{trainer.last_epoch_loss:.4f}\n")

            # Family early-stopping
            if family_acc > best_family_acc:
                best_family_acc = family_acc
                patience_family = 0
                fam_best_path = os.path.join(exp_dir, "best_models", "best_by_family.pth")
                torch.save(model.state_dict(), fam_best_path)
            else:
                patience_family += 1

            # Association early-stopping
            if assoc_auc > best_assoc_auc:
                best_assoc_auc = assoc_auc
                patience_assoc = 0
                best_val_metrics = eval_res.copy()
                best_val_metrics['best_thr'] = eval_res.get('association_best_thr', 0.5)
                best_model_path = os.path.join(exp_dir, "best_models", "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                with open(os.path.join(exp_dir, "metrics_best.json"), "w") as f:
                    json.dump(_clean_for_json({'fold': fold_idx, **eval_res}), f, indent=2)
            else:
                patience_assoc += 1

            # Freeze converged task towers
            if not family_frozen and patience_family >= fam_pat_th:
                print(f"  [Fold {fold_idx}] Family task converged — freezing tower.")
                for p in model.family_tower.parameters():
                    p.requires_grad = False
                if hasattr(model, 'family_adapter'):
                    for p in model.family_adapter.parameters():
                        p.requires_grad = False
                family_frozen = True

            if not assoc_frozen and patience_assoc >= asso_pat_th:
                print(f"  [Fold {fold_idx}] Association task converged — freezing tower.")
                for p in model.association_tower.parameters():
                    p.requires_grad = False
                if hasattr(model, 'assoc_adapter'):
                    for p in model.assoc_adapter.parameters():
                        p.requires_grad = False
                assoc_frozen = True

            if patience_family >= early_patience and patience_assoc >= early_patience:
                print(f"  [Fold {fold_idx}] Both tasks converged — early stopping.")
                break
            if family_frozen and assoc_frozen:
                print(f"  [Fold {fold_idx}] Both towers frozen — stopping.")
                break

        except Exception as e:
            print(f"  [Warning] Epoch {epoch+1} error: {e}")
            traceback.print_exc()
            continue


    print(f"\n[Fold {fold_idx}] Final test evaluation...")
    final_test = {}

    fam_best_path = os.path.join(exp_dir, "best_models", "best_by_family.pth")
    best_model_path = os.path.join(exp_dir, "best_models", "best_model.pth")

    # Family task test
    if os.path.exists(fam_best_path) and fam_test_loader is not None:
        state = torch.load(fam_best_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in fam_test_loader:
                feats = _pad_or_truncate(batch['features'].to(device), model.input_dim)
                logits, _ = model(feats, task='family')
                preds.append(torch.argmax(logits, dim=1).cpu())
                labels.append(batch['family_label'].cpu())
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        final_test['family_acc'] = accuracy_score(labels, preds)
        final_test['family_mcc'] = matthews_corrcoef(labels, preds)
        final_test['family_f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
        final_test['family_f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        final_test['family_prec_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0)
        final_test['family_rec_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0)

    # Association task test
    if os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        val_best_thr = float(best_val_metrics.get('best_thr', 0.5)) if best_val_metrics else 0.5
        T = float(best_val_metrics.get('temp_T', 1.0)) if best_val_metrics else 1.0

        test_probs, test_labels = [], []
        with torch.no_grad():
            for batch in asc_test_loader:
                feats = _pad_or_truncate(batch['features'].to(device), model.input_dim)
                logit, _ = model(feats, task='association')
                logit = logit.squeeze(-1) / T
                test_probs.append(torch.sigmoid(logit).cpu().numpy())
                test_labels.append(batch['association_label'].cpu().numpy())

        test_probs = np.concatenate(test_probs) if test_probs else np.array([])
        test_labels = np.concatenate(test_labels) if test_labels else np.array([])
        np.savez(os.path.join(exp_dir, "test_predictions.npz"),
                 probs=test_probs, labels=test_labels)

        if test_probs.size > 0 and len(np.unique(test_labels)) >= 2:
            final_test['association_auc'] = roc_auc_score(test_labels, test_probs)
            final_test['association_auprc'] = average_precision_score(test_labels, test_probs)

            pred_best = (test_probs >= val_best_thr).astype(int)
            final_test['association_f1_best'] = f1_score(test_labels, pred_best, zero_division=0)
            final_test['association_mcc_best'] = matthews_corrcoef(test_labels, pred_best)
            final_test['association_prec_best'] = precision_score(test_labels, pred_best, zero_division=0)
            final_test['association_rec_best'] = recall_score(test_labels, pred_best, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(test_labels, pred_best).ravel()
            final_test['association_spec_best'] = tn / (tn + fp + 1e-8)

            pred_05 = (test_probs >= 0.5).astype(int)
            final_test['association_f1_at_0_5'] = f1_score(test_labels, pred_05, zero_division=0)
            final_test['association_acc_at_0_5'] = accuracy_score(test_labels, pred_05)

    # Print final summary
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Fold {fold_idx} | Results")
    print(sep)
    print(f"  [Family]")
    print(f"    ACC       = {final_test.get('family_acc', float('nan')):.4f}")
    print(f"    MCC       = {final_test.get('family_mcc', float('nan')):.4f}")
    print(f"    F1 (W)    = {final_test.get('family_f1_weighted', float('nan')):.4f}")
    print(f"    F1 (Mac)  = {final_test.get('family_f1_macro', float('nan')):.4f}")
    print(f"  [Association]  (thr={val_best_thr:.3f})")
    print(f"    AUC       = {final_test.get('association_auc', float('nan')):.4f}")
    print(f"    AUPRC     = {final_test.get('association_auprc', float('nan')):.4f}")
    print(f"    F1        = {final_test.get('association_f1_best', float('nan')):.4f}")
    print(f"    MCC       = {final_test.get('association_mcc_best', float('nan')):.4f}")
    print(f"    Precision = {final_test.get('association_prec_best', float('nan')):.4f}")
    print(f"    Recall    = {final_test.get('association_rec_best', float('nan')):.4f}")
    print(f"    Spec.     = {final_test.get('association_spec_best', float('nan')):.4f}")
    print(f"    F1@0.5    = {final_test.get('association_f1_at_0_5', float('nan')):.4f}")
    print(sep)
    print(f"  Results dir: {exp_dir}")
    print(sep + "\n")

    # Save final metrics JSON
    with open(os.path.join(exp_dir, "metrics_final.json"), "w", encoding="utf-8") as f:
        json.dump(
            _clean_for_json({
                'fold': fold_idx,
                'seed': seed,
                'dss_lambda': config.get('dss_lambda', 0.5),
                'family_loss_weight': config.get('family_loss_weight', 1.0),
                'micro_layers': config.get('nas', {}).get('micro_layers', None),
                'val_best': best_val_metrics,
                'test': final_test,
            }),
            f, indent=2, ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ncRNA family classification + disease association prediction."
    )
    parser.add_argument("--fold", type=str, default="1",
                        help="Fold index (1-5) or 'all' to run all folds.")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds, e.g. '42,1337'.")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--micro-layers", type=int, default=2,
                        help="Number of operation layers per micro expert.")

    # NAS / architecture
    nas_group = parser.add_argument_group("NAS")
    nas_group.add_argument("--arch-file", type=str, default=None,
                           help="Path to exported_arch.json from a previous search run.")
    nas_group.add_argument("--force-search", action="store_true",
                           help="Run NAS search even when --arch-file is provided.")
    nas_group.add_argument("--inner-repeats", type=int, default=1)
    nas_group.add_argument("--search-epochs", type=int, default=None)

    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override base output directory.")

    args = parser.parse_args()

    folds = list(range(1, 6)) if args.fold.lower() == "all" else [int(args.fold)]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for fold in folds:
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"  Fold={fold}  Seed={seed}")
            print(f"{'='*50}")
            main(
                fold_idx=fold,
                seed=seed,
                arch_file=args.arch_file,
                force_search=args.force_search,
                search_epochs=args.search_epochs,
                inner_repeats=args.inner_repeats,
                results_dir=args.results_dir,
                num_epochs=args.num_epochs,
                micro_layers=args.micro_layers,
                hidden_dim=args.hidden_dim,
                num_experts=args.num_experts,
            )

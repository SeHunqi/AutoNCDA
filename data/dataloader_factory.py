import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset import SafeFastRNADataset
from data.dataset_lookup import SafeFastLookupDataset


def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class SafeFastDataLoaderFactory:


    def __init__(self, config: dict):
        self.config = config
        self.dss_lambda = float(config.get('dss_lambda', 0.5))

        # Repo root is one level above the 'data/' package directory
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.fold_splits_dir = os.path.join(self.repo_root, "fold_splits")
        self.fold_features_dir = os.path.join(self.repo_root, "fold_features")
        self.fold_dss_dir = os.path.join(
            self.repo_root, "fold_dss_features"
        )

        dl_cfg = config['dataloader_config']
        self.batch_size = dl_cfg.get('batch_size', 1024)
        self.num_workers = dl_cfg.get('num_workers', 4)
        self.pin_memory = dl_cfg.get('pin_memory', True)
        self.persistent_workers = dl_cfg.get('persistent_workers', True) and self.num_workers > 0

        # Global RNA feature files shared across all folds
        self.global_rna_feat = os.path.join(self.fold_features_dir, "disease_rna_features.npy")
        self.global_rna_map = os.path.join(self.fold_features_dir, "disease_rna_map.csv")
        self.doid_map_path = os.path.join(self.fold_dss_dir, "doid_order.csv")
        self.num_diseases = None

    def _get_dataset(
        self,
        task: str,
        fold_idx: int,
        split_type: str,
        rna_type_filter: str = None,
    ):
        """Construct a Dataset for the given task, fold, and split."""
        target_dim = self.config['feature_config'].get('total_dim', 960)

        if task == 'family':
            feat_path = os.path.join(self.fold_features_dir,
                                     f'family_fold_{fold_idx}_{split_type}.npy')
            lbl_path = os.path.join(self.fold_features_dir,
                                    f'family_fold_{fold_idx}_{split_type}_labels.npy')
            return SafeFastRNADataset(
                feature_npy_path=feat_path,
                label_npy_path=lbl_path,
                meta_csv_path=None,
                task=task,
                target_dim=target_dim,
                dss_matrix=None,
            )

        # Association task — Lookup mode
        csv_path = os.path.join(
            self.fold_splits_dir,
            f"disease_fold_{fold_idx}_{split_type}.csv"
        )
        dss_feat_path = os.path.join(self.fold_dss_dir, f'dss_fold_{fold_idx}.npy')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        if not os.path.exists(self.global_rna_feat):
            raise FileNotFoundError(f"Global RNA feature matrix not found: {self.global_rna_feat}")
        if not os.path.exists(dss_feat_path):
            raise FileNotFoundError(
                f"DSS feature file not found: {dss_feat_path}\n"
                "Please run scripts/generate_dss.py first."
            )
        if not os.path.exists(self.doid_map_path):
            raise FileNotFoundError(f"DOID order file not found: {self.doid_map_path}")

        dataset = SafeFastLookupDataset(
            split_csv_path=csv_path,
            rna_feat_path=self.global_rna_feat,
            rna_map_path=self.global_rna_map,
            dss_feat_path=dss_feat_path,
            doid_map_path=self.doid_map_path,
            task='association',
            target_dim=target_dim,
            rna_type_filter=rna_type_filter,
            dynamic_negatives=(
                self.config.get('dynamic_negatives', False) and split_type == 'train'
            ),
        )
        if self.num_diseases is None:
            self.num_diseases = int(getattr(dataset, 'num_diseases', 0)) or None
        return dataset

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=_worker_init_fn if shuffle else None,
        )

    def create_fold_loaders(
        self,
        fold_idx: int,
        val_ratio: float = 0.18,
        rna_type_filter: str = None,
    ):
        """
        Create train / val / test loaders for a single fold.

        The validation set is carved out of the training split using a fixed seed,
        ensuring that the same val indices are used regardless of global random state.

        Returns:
            Tuple of six DataLoaders:
            (family_train, family_val, family_test,
             assoc_train, assoc_val, assoc_test)
        """
        # --- Association ---
        assoc_train_full = self._get_dataset('association', fold_idx, 'train', rna_type_filter)
        assoc_test = self._get_dataset('association', fold_idx, 'test', rna_type_filter)

        if val_ratio > 0:
            total = len(assoc_train_full)
            n_val = int(total * val_ratio)
            n_train = total - n_val
            assoc_train_sub, assoc_val_sub = torch.utils.data.random_split(
                assoc_train_full, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            assoc_train_loader = self._make_loader(assoc_train_sub, shuffle=True)
            assoc_val_loader = self._make_loader(assoc_val_sub, shuffle=False)
        else:
            assoc_train_loader = self._make_loader(assoc_train_full, shuffle=True)
            assoc_val_loader = None

        assoc_test_loader = self._make_loader(assoc_test, shuffle=False)

        # --- Family ---
        fam_train_full = self._get_dataset('family', fold_idx, 'train')
        fam_test = self._get_dataset('family', fold_idx, 'test')

        if val_ratio > 0:
            ft = len(fam_train_full)
            fv = int(ft * val_ratio)
            fam_train_sub, fam_val_sub = torch.utils.data.random_split(
                fam_train_full, [ft - fv, fv],
                generator=torch.Generator().manual_seed(42),
            )
            fam_train_loader = self._make_loader(fam_train_sub, shuffle=True)
            fam_val_loader = self._make_loader(fam_val_sub, shuffle=False)
        else:
            fam_train_loader = self._make_loader(fam_train_full, shuffle=True)
            fam_val_loader = None

        fam_test_loader = self._make_loader(fam_test, shuffle=False)

        return (fam_train_loader, fam_val_loader, fam_test_loader,
                assoc_train_loader, assoc_val_loader, assoc_test_loader)

    def get_feature_dims(self, fold_idx: int = 1) -> dict:
        """Return the expected feature dimensions for each task."""
        return {
            'family': 832,       # RNA-FM(640) + k-mer(64) + structure(128)
            'association': 960,  # family(832) + DSS(128)
        }
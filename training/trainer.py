

from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from training.losses import FocalLoss, mixup_criterion, mixup_data
from training.metrics import _pad_or_truncate, THR_FIXED


def _unique_params(params):
    seen, out = set(), []
    for p in params:
        if id(p) not in seen:
            out.append(p)
            seen.add(id(p))
    return out


def set_family_class_weight(trainer: 'DecoupledTrainer', family_loader, num_classes: int):

    cnt = Counter()
    for b in family_loader:
        cnt.update(b['family_label'].tolist())
    freq = torch.tensor([cnt.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    w = torch.sqrt(1.0 / (freq + 1e-6))
    w = w / w.mean()
    trainer.family_criterion = nn.CrossEntropyLoss(weight=w.to(trainer.device))


class DecoupledTrainer:


    def __init__(self, model: nn.Module, config: dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device

        opt_cfg = config['optimizer']
        self.w_bal = float(config.get('w_bal', 0.05))
        self.w_bal_warmup = float(config.get('w_bal_warmup', self.w_bal))
        self.w_bal_decay_epochs = int(config.get('w_bal_decay_epochs', 30))
        self.warmup_epochs = int(opt_cfg.get('warmup_epochs', 0))

        # ── Parameter groups (differential LR) ───────────────────────────
        shared_fn = getattr(model, "shared_parameters", None)
        fam_fn = getattr(model, "family_tower_parameters", None)
        asc_fn = getattr(model, "association_tower_parameters", None)

        self.log_vars = nn.Parameter(torch.zeros(2, device=device))

        if callable(shared_fn) and callable(fam_fn) and callable(asc_fn):
            shared = list(shared_fn()) + [self.log_vars]
            family = list(fam_fn())
            assoc = list(asc_fn())

            # Deduplicate across groups
            seen = set()
            def _dedup(group):
                out = []
                for p in group:
                    if id(p) not in seen:
                        out.append(p)
                        seen.add(id(p))
                return out

            param_groups = [
                {'params': _dedup(shared), 'lr': opt_cfg['shared_lr']},
                {'params': _dedup(family), 'lr': opt_cfg['family_lr']},
                {'params': _dedup(assoc),  'lr': opt_cfg['association_lr']},
            ]
        else:
            param_groups = [{'params': _unique_params(list(model.parameters()) + [self.log_vars]),
                             'lr': opt_cfg.get('shared_lr', 1e-3)}]

        # Prefer fused AdamW on Volta+ GPUs
        fused_kw = {}
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_capability()[0] >= 7:
                    fused_kw['fused'] = True
            except Exception:
                pass

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=opt_cfg['weight_decay'], **fused_kw
        )

        self.arch_optimizer: Optional[torch.optim.Optimizer] = None
        if hasattr(model, "arch_parameters") and callable(model.arch_parameters):
            arch_params = list(model.arch_parameters())
            if arch_params:
                nas_cfg = config.get("nas", {})
                self.arch_optimizer = torch.optim.AdamW(
                    arch_params,
                    lr=nas_cfg.get("alpha_lr", 4e-3),
                    weight_decay=nas_cfg.get("alpha_wd", 5e-3),
                )

        self.base_lrs = [g['lr'] for g in self.optimizer.param_groups]
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        self.association_criterion = FocalLoss(alpha=0.5, gamma=2.0)
        self.family_criterion = nn.CrossEntropyLoss()  # replaced by set_family_class_weight

        self.epoch_idx = 0

    # ── LR helpers ────────────────────────────────────────────────────────

    def apply_warmup(self):
        """Linearly ramp learning rates from 0 during warmup epochs."""
        if self.warmup_epochs > 0 and self.epoch_idx < self.warmup_epochs:
            frac = (self.epoch_idx + 1) / self.warmup_epochs
            for i, g in enumerate(self.optimizer.param_groups):
                g['lr'] = self.base_lrs[i] * frac

    def current_w_bal(self, epoch_idx: int) -> float:
        """Load-balance weight, linearly decayed from w_bal_warmup to w_bal."""
        if epoch_idx < self.warmup_epochs:
            return self.w_bal_warmup
        t = min(1.0, (epoch_idx - self.warmup_epochs) / max(1, self.w_bal_decay_epochs))
        return (1.0 - t) * self.w_bal_warmup + t * self.w_bal

    # ── Training ──────────────────────────────────────────────────────────

    def train_epoch(self, family_loader, assoc_loader) -> float:

        self.model.train()
        epoch_idx = self.epoch_idx
        is_warmup = epoch_idx < self.warmup_epochs

        real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(real_model, 'warmup'):
            real_model.warmup = is_warmup

        total_loss = 0
        fam_iter = iter(family_loader)
        asc_iter = iter(assoc_loader)
        steps = max(len(family_loader), len(assoc_loader))
        pbar = tqdm(range(steps), desc=f"Epoch {epoch_idx}", leave=False)

        for _ in pbar:
            try:
                batch_fam = next(fam_iter)
            except StopIteration:
                fam_iter = iter(family_loader)
                batch_fam = next(fam_iter)

            try:
                batch_asc = next(asc_iter)
            except StopIteration:
                asc_iter = iter(assoc_loader)
                batch_asc = next(asc_iter)

            xf = batch_fam['features'].to(self.device)
            yf = batch_fam['family_label'].to(self.device)
            xa = batch_asc['features'].to(self.device)
            ya = batch_asc['association_label'].float().to(self.device)

            if hasattr(self, 'target_input_dim'):
                xf = _pad_or_truncate(xf, self.target_input_dim)
                xa = _pad_or_truncate(xa, self.target_input_dim)

            self.optimizer.zero_grad()
            if self.arch_optimizer is not None:
                self.arch_optimizer.zero_grad()

            # Contrastive alignment loss
            z_rna, z_dis = self.model(xa, task='contrastive')
            cos_sim = F.cosine_similarity(F.normalize(z_rna, dim=1), F.normalize(z_dis, dim=1))
            loss_con = ((1 - cos_sim) * ya + F.relu(cos_sim - 0.3) * (1 - ya)).mean()

            # Family task
            out_fam = self.model(xf, task='family')
            fam_logits, fam_bal = out_fam if isinstance(out_fam, tuple) else (out_fam, 0.0)
            loss_family = self.family_criterion(fam_logits, yf)

            # Association task (with Mixup after warmup)
            use_mixup = self.config.get('use_mixup', False) and not is_warmup
            if use_mixup:
                xa_m, ya_a, ya_b, lam = mixup_data(xa, ya, alpha=0.2, device=self.device)
                out_asc = self.model(xa_m, task='association')
                asc_logits, asc_bal = out_asc if isinstance(out_asc, tuple) else (out_asc, 0.0)
                loss_assoc = mixup_criterion(
                    self.association_criterion, asc_logits.squeeze(-1), ya_a, ya_b, lam
                )
            else:
                out_asc = self.model(xa, task='association')
                asc_logits, asc_bal = out_asc if isinstance(out_asc, tuple) else (out_asc, 0.0)
                loss_assoc = self.association_criterion(asc_logits.squeeze(-1), ya)

            w_bal = self.current_w_bal(epoch_idx)
            w_con = self.config.get('w_con', 0.1)
            w_fam = self.config.get('family_loss_weight', 1.0)
            w_asc = self.config.get('association_loss_weight', 1.0)
            loss_bal = 0.5 * (fam_bal + asc_bal)

            loss = w_fam * loss_family + w_asc * loss_assoc + w_con * loss_con + w_bal * loss_bal

            if torch.isnan(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
            self.optimizer.step()
            if not is_warmup and self.arch_optimizer is not None:
                self.arch_optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'Lf': f"{loss_family.item():.3f}",
                'La': f"{loss_assoc.item():.3f}",
            })

        self.last_epoch_loss = total_loss / steps
        return self.last_epoch_loss

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, family_loader, assoc_loader) -> dict:
 
        self.model.eval()
        results = {}

        with torch.no_grad():
            # Family
            preds, labels = [], []
            for batch in tqdm(family_loader, desc="Eval Family", leave=False):
                feats = _pad_or_truncate(batch['features'].to(self.device), self.model.input_dim)
                logits, _ = self.model(feats, task='family')
                preds.append(torch.argmax(logits, dim=1).cpu())
                labels.append(batch['family_label'].cpu())
            preds = torch.cat(preds).numpy()
            labels = torch.cat(labels).numpy()
            results['family_acc'] = accuracy_score(labels, preds)
            results['family_f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
            results['family_f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
            results['family_f1_micro'] = f1_score(labels, preds, average='micro', zero_division=0)

            # Association
            probs_list, labels_list = [], []
            for batch in tqdm(assoc_loader, desc="Eval Assoc", leave=False):
                feats = _pad_or_truncate(batch['features'].to(self.device), self.model.input_dim)
                feats = torch.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-1e6, 1e6)
                logits, _ = self.model(feats, task='association')
                probs_list.append(torch.sigmoid(logits.squeeze(-1)).cpu())
                labels_list.append(batch['association_label'].cpu())

            asc_probs = torch.cat(probs_list).numpy() if probs_list else np.array([])
            asc_labels = torch.cat(labels_list).numpy() if labels_list else np.array([])
            results['_raw_assoc_probs'] = asc_probs
            results['_raw_assoc_labels'] = asc_labels

            valid = np.isfinite(asc_probs)
            if valid.sum() >= 2 and len(np.unique(asc_labels[valid])) >= 2:
                pv, lv = asc_probs[valid], asc_labels[valid]
                results['association_auc'] = roc_auc_score(lv, pv)
                results['association_auprc'] = average_precision_score(lv, pv)
                pred_05 = (pv >= THR_FIXED).astype(int)
                results['association_acc_at_fixed'] = accuracy_score(lv, pred_05)
                results['association_precision_at_fixed'] = precision_score(lv, pred_05, zero_division=0)
                results['association_recall_at_fixed'] = recall_score(lv, pred_05, zero_division=0)
                results['association_f1_at_fixed'] = f1_score(lv, pred_05, zero_division=0)
            else:
                for k in ['association_auc', 'association_auprc', 'association_acc_at_fixed',
                          'association_precision_at_fixed', 'association_recall_at_fixed',
                          'association_f1_at_fixed']:
                    results[k] = float('nan')

        return results

    # ── Scheduler / bookkeeping ───────────────────────────────────────────

    def update_schedulers(self, metrics: dict = None):
        """Step the cosine LR scheduler."""
        self.scheduler.step()

    def step_scheduler(self):
        """Increment the internal epoch counter (call after each epoch)."""
        self.epoch_idx += 1

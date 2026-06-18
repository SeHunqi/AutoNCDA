import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

try:
    from models.nas.mix_expert import OP_NAMES, build_micro_expert
except Exception:
    OP_NAMES = None

def _pad_or_truncate(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = x.size(-1)
    if cur == target_dim: return x
    if cur < target_dim:
        pad = torch.zeros(x.size(0), target_dim - cur, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)
    return x[..., :target_dim]

def _entropy(v: torch.Tensor) -> torch.Tensor:
    """H(softmax(v))"""
    w = torch.softmax(v, dim=0)
    return -(w * (w.clamp_min(1e-8).log())).sum()

def _experts_alpha_matrix(model):
   
    rows = []
    for e in getattr(model, 'experts', []):
        vecs = []
        for blk in getattr(e, 'blocks', []):
            if hasattr(blk, 'alpha'):
                vecs.append(F.softmax(blk.alpha, dim=0))
        if len(vecs) > 0:
            rows.append(torch.stack(vecs, 0).mean(0))  # [K]
    return torch.stack(rows, 0) if rows else None

def _diversity_loss_experts_alpha(P: torch.Tensor) -> torch.Tensor:
  
    if P is None or P.size(0) < 2:
        return torch.tensor(0.0, device=P.device if P is not None else 'cpu')
    G = P @ P.t()                       # [E, E]
    off_diag = G - torch.diag_embed(G.diag())
    return off_diag.sum() / (P.size(0) * (P.size(0) - 1))  # 平均相似度

class FocalLoss(nn.Module):
    """
    Binary Focal Loss (copied from train_model.py to avoid circular import)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()
        if inputs.dim() == 2 and inputs.size(1) == 1 and targets.dim() == 1:
            targets = targets.view(-1, 1)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        if self.reduction == 'mean': return loss.mean()
        return loss

class DARTSLikeSearcherV1(nn.Module):
    def __init__(self, model, device, target_input_dim,
                 w_lr=6e-4, w_wd=2e-4,
                 alpha_lr=8e-3, alpha_wd=2e-3,
                 entropy_reg=3e-3, diversity_reg=2e-3):
        super().__init__()
        self.model = model
        self.device = device
        self.target_input_dim = target_input_dim
        self.entropy_reg = entropy_reg
        self.diversity_reg = diversity_reg
        self.task_weights = (1.0, 1.0)
        self.bce = FocalLoss(alpha=0.25, gamma=2.0) 

        alpha_params_raw = list(model.arch_parameters())
        alpha_params, seen = [], set()
        for p in alpha_params_raw:
            if id(p) not in seen:
                alpha_params.append(p); seen.add(id(p))
        alpha_param_ids = {id(p) for p in alpha_params}
        w_params = [p for p in model.parameters() if p.requires_grad and id(p) not in alpha_param_ids]
        self.w_optim = AdamW(w_params, lr=w_lr, weight_decay=w_wd)
        assert len(alpha_params) > 0, "未检测到架构参数 alpha"
        self.alpha_optim = AdamW(alpha_params, lr=alpha_lr, weight_decay=alpha_wd)
        self._alpha_list = alpha_params

    def _entropy_regularizer(self):
       
        rows = []
        for e in getattr(self.model, 'experts', []):
            for blk in getattr(e, 'blocks', []):
                if hasattr(blk, 'alpha'):
                    probs = torch.softmax(blk.alpha, dim=0)
                    rows.append(probs)
        if not rows:
            return torch.tensor(0.0, device=self.device)
        mat = torch.stack(rows, 0)  # [N,K]

        ent = -(mat * (mat.clamp_min(1e-8).log())).sum(dim=1).mean()
        return ent

    def _diversity_regularizer(self):
        P = _experts_alpha_matrix(self.model)  # [E,K] or None
        return _diversity_loss_experts_alpha(P)

    def _compute_losses(self, batch_family, batch_assoc):
        xf = batch_family['features'].to(self.device)
        yf = batch_family['family_label'].to(self.device)
        xa = batch_assoc['features'].to(self.device)
        ya = batch_assoc['association_label'].float().to(self.device)
        
       
        out_fam = self.model(xf, task='family')
      
        if isinstance(out_fam, tuple):
            fam_logits, fam_bal = out_fam
        else:
            fam_logits, fam_bal = out_fam, 0.0
            
        fam_loss = F.cross_entropy(fam_logits, yf)

      
        out_asc = self.model(xa, task='association')
        if isinstance(out_asc, tuple):
            asc_logits, asc_bal = out_asc
        else:
            asc_logits, asc_bal = out_asc, 0.0
            
        dis_loss = self.bce(asc_logits.squeeze(-1), ya)

       
        z_rna, z_dis = self.model(xa, task='contrastive')
        mask_pos = (ya == 1).squeeze()
        if mask_pos.sum() > 0:
            loss_con = (1 - F.cosine_similarity(z_rna[mask_pos], z_dis[mask_pos])).mean()
        else:
            loss_con = torch.tensor(0.0, device=xf.device)

        ent   = self._entropy_regularizer()
        l_div = self._diversity_regularizer()
        
        
        return fam_loss, dis_loss, ent, l_div, loss_con, (fam_bal + asc_bal)

    def train_epoch(self, fam_train_iter, assoc_train_iter,
                    fam_val_iter, assoc_val_iter,
                    steps_train, steps_val,
                    task_weights=(1.0, 1.0),
                    update_alpha=True):
        self.task_weights = task_weights
        w_f, w_d = self.task_weights
        w_con = 0.1  
        w_bal = 0.05 

        # W-step
        for _ in tqdm(range(steps_train), desc="W-step", leave=False):
            bf = next(fam_train_iter); bd = next(assoc_train_iter)
            fam_l, dis_l, ent, l_div, con_l, bal_l = self._compute_losses(bf, bd)
            
            loss = (w_f * fam_l + w_d * dis_l + 
                    w_con * con_l + w_bal * bal_l - 
                    self.entropy_reg * ent + self.diversity_reg * l_div)
            
            self.w_optim.zero_grad(); loss.backward(); self.w_optim.step()

        # α-step
        if update_alpha:
            for _ in tqdm(range(steps_val), desc="α-step", leave=False):
                bf = next(fam_val_iter); bd = next(assoc_val_iter)
                # 使用验证集计算 Losses
                fam_l, dis_l, ent, l_div, con_l, bal_l = self._compute_losses(bf, bd)
                
                loss = (w_f * fam_l + w_d * dis_l + 
                        w_con * con_l + w_bal * bal_l - 
                        self.entropy_reg * ent + self.diversity_reg * l_div)
                        
                self.alpha_optim.zero_grad(); loss.backward(); self.alpha_optim.step()
       

    def anneal_temperature(self, factor=0.95, min_t=0.30):
      
        for e in getattr(self.model, 'experts', []):
            for blk in getattr(e, 'blocks', []):
                if hasattr(blk, 'tau'):
                    blk.tau = max(min_t, float(blk.tau) * factor)
     
        if hasattr(self.model, 'adapter_mix') and hasattr(self.model.adapter_mix, 'temperature'):
            self.model.adapter_mix.temperature = max(min_t, self.model.adapter_mix.temperature * factor)
        if hasattr(self.model, 'family_share_mixer') and hasattr(self.model.family_share_mixer, 'temperature'):
            self.model.family_share_mixer.temperature = max(min_t, self.model.family_share_mixer.temperature * factor)
        if hasattr(self.model, 'assoc_share_mixer') and hasattr(self.model.assoc_share_mixer, 'temperature'):
            self.model.assoc_share_mixer.temperature = max(min_t, self.model.assoc_share_mixer.temperature * factor)

    @torch.no_grad()
    def export_choices(self):
        choices = []
        for m in self.model.experts:
            if hasattr(m, 'alpha'):
                choices.append(int(torch.argmax(m.alpha).item()))
            else:
                choices.append(-1)
        return choices

    @torch.no_grad()
    def log_alpha(self, log_dir: str, epoch: int):
    
        payload = {'experts': [], 'adapter': None, 'family_share': None, 'assoc_share': None}
        for ei, e in enumerate(getattr(self.model, 'experts', [])):
            blk_rows = []
            for bi, blk in enumerate(getattr(e, 'blocks', [])):
                if hasattr(blk, 'alpha'):
                    dist = torch.softmax(blk.alpha.detach(), dim=0)
                    blk_rows.append([float(x) for x in dist.tolist()])
                    row = [float(x) for x in dist.tolist()]
                    if OP_NAMES and len(row) == len(OP_NAMES):
                        blk_rows.append({'probs': row, 'names': OP_NAMES})
                    else:
                        blk_rows.append(row)
            payload['experts'].append({'idx': ei, 'blocks': blk_rows})
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"alpha_epoch_{epoch}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @torch.no_grad()
    def export_arch(self):
        if hasattr(self.model, "export_architecture"):
            return self.model.export_architecture()
        micro, micro_names = [], []
        for e in getattr(self.model, 'experts', []):
            idx_row, name_row = [], []
            for blk in getattr(e, 'blocks', []):
                if hasattr(blk, 'alpha'):
                    idx = int(torch.argmax(blk.alpha).item())
                    idx_row.append(idx)
                    name_row.append(OP_NAMES[idx] if OP_NAMES and 0 <= idx < len(OP_NAMES) else str(idx))
            micro.append(idx_row); micro_names.append(name_row)
        return {'micro_expert_choices': micro, 'micro_expert_names': micro_names}
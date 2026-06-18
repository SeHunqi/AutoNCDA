

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
  

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits, any shape.
            targets: Binary labels {0, 1}, same shape or broadcastable.
        """
        targets = targets.float()

        # Ensure shapes match (common case: inputs [B,1], targets [B])
        if inputs.dim() == 2 and inputs.size(1) == 1 and targets.dim() == 1:
            targets = targets.view(-1, 1)

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)

        # p_t: probability assigned to the ground-truth class
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # alpha_t: class-wise weighting
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    device: str = 'cuda'
):
    """
    Apply Mixup augmentation to a batch.
    
    Returns:
        mixed_x: Mixed inputs.
        y_a, y_b: Pairs of targets.
        lam: Mixup interpolation coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss by interpolating between two target losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """
    Fit a temperature scaling parameter T on validation data to reduce 
    test-set threshold drift.
    
    Args:
        val_logits: Raw logits from the model on the validation set.
        val_labels: Ground-truth binary labels for the validation set.
        
    Returns:
        Calibrated temperature T (clamped to [0.2, 5.0]).
    """
    z = torch.tensor(val_logits, dtype=torch.float32)
    y = torch.tensor(val_labels, dtype=torch.float32)
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / T, y)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        T.clamp_(0.2, 5.0)
    return float(T.item())


def search_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = 'youden',
    n_points: int = 200,
    min_specificity: float = 0.6
) -> dict:
    """
    Search for the optimal classification threshold on a set of predicted probabilities.
    
    Args:
        probs: Predicted probabilities for the positive class.
        labels: Ground-truth binary labels.
        metric: Optimization target - 'f1' | 'acc' | 'mcc' | 'youden' (recommended).
        n_points: Number of threshold candidates to evaluate.
        min_specificity: Minimum required specificity for a threshold to be considered.
        
    Returns:
        dict with keys: thr, f1, acc, mcc, prec, rec, spec, youden.
    """
    from sklearn.metrics import confusion_matrix

    eps = 1e-8
    thresholds = np.linspace(0.01, 0.99, n_points)

    best = {
        'thr': 0.5, 'f1': -1, 'acc': -1, 'mcc': -1,
        'prec': 0, 'rec': 0, 'spec': 0, 'youden': -1
    }

    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()

        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        numerator = (float(tp) * tn) - (float(fp) * fn)
        denom_sq = (float(tp) + fp) * (float(tp) + fn) * (float(tn) + fp) * (float(tn) + fn)
        denominator = np.sqrt(denom_sq) + eps
        mcc = numerator / denominator

        youden = rec + spec - 1

        current = {
            'thr': t, 'f1': f1, 'acc': acc, 'mcc': mcc,
            'prec': prec, 'rec': rec, 'spec': spec, 'youden': youden
        }

        if spec < min_specificity:
            continue

        if metric == 'youden':
            if youden > best['youden']:
                best = current
        elif metric == 'mcc':
            if mcc > best['mcc']:
                best = current
        elif metric == 'f1':
            if f1 > best['f1']:
                best = current
        else:
            if acc > best['acc']:
                best = current

    return best

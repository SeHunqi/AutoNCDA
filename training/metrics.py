

from typing import Any

import numpy as np
import torch

THR_FIXED = 0.5  


def _to_scalar(x: Any):
    """Convert numpy/tensor scalar to a plain Python scalar."""
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().item() if x.numel() == 1 else None
    if isinstance(x, (float, int, bool, str)) or x is None:
        return x
    return None


def _clean_for_json(obj: Any) -> Any:
    """
    Recursively convert dict/list/tuple/numpy/tensor objects to JSON-serializable types.
    Keys starting with '_' (e.g. _raw_* temporary fields) are filtered out.
    """
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def _pad_or_truncate(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Adjust the last dimension of a 2-D tensor to ``target_dim``.

    Zero-pads if the current dimension is smaller; truncates if larger.
    """
    import torch.nn.functional as F

    d = tensor.shape[-1]
    if d == target_dim:
        return tensor
    if d < target_dim:
        return F.pad(tensor, (0, target_dim - d), "constant", 0)
    return tensor[..., :target_dim]

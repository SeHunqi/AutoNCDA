"""
Utilities for NAS architecture sampling and export.
"""

import random
from typing import List
CANDIDATE_NAMES = ["SwiGLU", "GatedMLP", "ResNeXt", "Attention", "Identity"]


def sample_random_micro_choices(
    num_experts: int,
    micro_layers: int,
    n_ops: int,
    arch_seed: int,
    forbid_identity_first_layer: bool = True,
    identity_idx: int = 4,
):
    rng = random.Random(int(arch_seed))
    choices = []
    for _ in range(int(num_experts)):
        row = []
        for li in range(int(micro_layers)):
            pool = list(range(int(n_ops)))
            if forbid_identity_first_layer and li == 0 and identity_idx in pool:
                pool.remove(identity_idx)
            row.append(rng.choice(pool))
        choices.append(row)
    return choices

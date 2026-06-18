import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityAdapter(nn.Module):
    def forward(self, x): return x

class BottleneckAdapter(nn.Module):
    def __init__(self, in_dim, b):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, b)
        self.fc2 = nn.Linear(b, in_dim)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class AdapterMix(nn.Module):
    def __init__(self, in_dim, choices=(0,16,32,64), temperature=1.0, mode='soft'):
        super().__init__()
        ops = []
        for b in choices:
            ops.append(IdentityAdapter() if b == 0 else BottleneckAdapter(in_dim, b))
        self.ops = nn.ModuleList(ops)
        self.alpha_adapter = nn.Parameter(torch.zeros(len(self.ops)))
        self.temperature = float(temperature)
        self.mode = mode
        with torch.no_grad():
            if self.alpha_adapter.numel() >= 2:
                self.alpha_adapter[0] = -0.5
                self.alpha_adapter[1:] = 0.2

    def forward(self, x, search=True, choice_idx=None):
        if search:
            if self.mode == 'gumbel':
                g = torch.rand_like(self.alpha_adapter).clamp_(1e-6, 1-1e-6)
                g = -torch.log(-torch.log(g))
                logits = (self.alpha_adapter + g) / max(self.temperature, 1e-6)
                w_soft = F.softmax(logits, dim=0)
                idx = int(torch.argmax(w_soft).item())
                w_hard = torch.zeros_like(w_soft); w_hard[idx] = 1.0
                w = (w_hard - w_soft).detach() + w_soft
            else:
                w = F.softmax(self.alpha_adapter / max(self.temperature, 1e-6), dim=0)
            outs = [op(x) for op in self.ops]
            return torch.stack(outs, 0).mul_(w.view(-1,1,1)).sum(0)
        else:
            idx = 0 if choice_idx is None else int(choice_idx)
            return self.ops[idx](x)

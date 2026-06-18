import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskPrivateProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x): return F.gelu(self.fc(x))

class GatedFuse(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj_t = nn.Linear(in_dim, out_dim)
        self.proj_s = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(out_dim*2, out_dim)
    def forward(self, x_s, x_t):
        s = F.gelu(self.proj_s(x_s))
        t = F.gelu(self.proj_t(x_t))
        g = torch.sigmoid(self.gate(torch.cat([s, t], dim=-1)))
        return g * t + (1 - g) * s

class SharePatternMixer(nn.Module):
    def __init__(self, in_dim, out_dim=None, temperature=1.0):
        super().__init__()
        out_dim = out_dim or in_dim
        self.shared_passthrough = nn.Identity()          
        self.task_private = TaskPrivateProj(in_dim, out_dim)  
        self.gated = GatedFuse(in_dim, out_dim)          
        self.alpha_share = nn.Parameter(torch.zeros(3))
        self.temperature = temperature
        self.out_dim = out_dim

    def forward(self, x_shared, x_task_in, search=True, choice_idx=None):
        if search:
            w = F.softmax(self.alpha_share / max(self.temperature, 1e-6), dim=0)
            o0 = self.shared_passthrough(x_shared)
            o1 = self.task_private(x_task_in)
            o2 = self.gated(x_shared, x_task_in)
            return w[0]*o0 + w[1]*o1 + w[2]*o2
        else:
            idx = int(0 if choice_idx is None else choice_idx)
            if idx == 0: return self.shared_passthrough(x_shared)
            if idx == 1: return self.task_private(x_task_in)
            return self.gated(x_shared, x_task_in)

    @torch.no_grad()
    def export_choice(self):
        return int(torch.argmax(self.alpha_share).item())

class SharePatternMixerHard(nn.Module):
    def __init__(self, d, tau=1.5, hard=False):
        super().__init__()
        self.alpha_share = nn.Parameter(0.01 * torch.randn(3))
        self.tau = tau; self.hard = hard
    def forward(self, fam, dis):
        if self.training:
            w = F.gumbel_softmax(self.alpha_share, tau=self.tau, hard=self.hard)
        else:
            w = F.one_hot(self.alpha_share.argmax(), num_classes=3).float().to(self.alpha_share.device)
        out_f = w[0]*fam + w[1]*((fam+dis)/2) + w[2]*(0.5*dis + 0.5*fam)
        out_d = w[0]*dis + w[1]*((fam+dis)/2) + w[2]*(0.5*fam + 0.5*dis)
        return out_f, out_d

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpResNeXt(nn.Module):
 
    def __init__(self, d, groups=4, drop=0.1):
        super().__init__()
        mid = int(d * 2)
        self.proj_in = nn.Linear(d, mid)
      
        self.group_transform = nn.Conv1d(mid, mid, kernel_size=1, groups=groups)
        self.proj_out = nn.Linear(mid, d)
        
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        h = self.act(self.proj_in(x))
        h = self.group_transform(h.unsqueeze(-1)).squeeze(-1)
        return self.drop(self.proj_out(h))

class OpIdentity(nn.Module):
    def __init__(self, d, drop=0.0): 
        super().__init__()
    def _init_weights(self): pass
    def forward(self, x): return x

class OpSEBlock(nn.Module):
    def __init__(self, d, reduction=4, drop=0.1):
        super().__init__()
        mid = max(d // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(d, mid),
            nn.ReLU(),
            nn.Linear(mid, d),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d, d) 
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        scale = self.fc(x)
        recalibrated = x * scale
        return self.drop(self.proj(recalibrated))

class OpSE_MLP(nn.Module):
  
    def __init__(self, d, b=48, drop=0.1):
        super().__init__()
        mid = int(d * 2) 
        self.net = nn.Sequential(
            nn.Linear(d, mid),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.se = OpSEBlock(mid, reduction=4, drop=0.0)
        self.proj = nn.Linear(mid, d)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.net(x)
        x = self.se(x) 
        return self.drop(self.proj(x))
    
class OpGatedMLP(nn.Module):

    def __init__(self, d, drop=0.1):
        super().__init__()
        # 🔥 微调：从 2.0 改为 1.33，严格对齐参数量
        mid = int(d * 4 / 3) 
        self.fc_feat = nn.Linear(d, mid)
        self.fc_gate = nn.Linear(d, mid)
        self.fc_out = nn.Linear(mid, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        feat = self.act(self.fc_feat(x))
        gate = torch.sigmoid(self.fc_gate(x))
        h = feat * gate
        return self.drop(self.fc_out(h))
    

class OpSEBlock(nn.Module):

    def __init__(self, d, reduction=4, drop=0.1):
        super().__init__()
        mid = max(d // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(d, mid),
            nn.ReLU(),
            nn.Linear(mid, d),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d, d) 
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
       
        scale = self.fc(x)
     
        recalibrated = x * scale
       
        return self.drop(self.proj(recalibrated))


class OpBilinear(nn.Module):
    
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)

    def forward(self, x):
        return self.drop(self.linear1(x) * self.linear2(x))


class OpSimpleMLP(nn.Module):

    def __init__(self, d, drop=0.1):
        super().__init__()
        # 与 FixedMicroBlock 保持一致的包裹结构
        self.ln1 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)   # 无门控，无分组，无注意力
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(drop)
        
        self.ln2 = nn.LayerNorm(d)
        self.fc2 = nn.Linear(d, d)
        self.drop2 = nn.Dropout(drop)
        self._init_weights()
    def _init_weights(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = x + self.drop1(self.act(self.fc1(self.ln1(x))))
        return x


class OpSwiGLU(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        hidden = int(d * 4 / 3)
        self.w1 = nn.Linear(d, hidden)
        self.w2 = nn.Linear(d, hidden)
        self.w3 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)

    def forward(self, x):
        x1 = F.silu(self.w1(x))
        x2 = self.w2(x)
        return self.drop(self.w3(x1 * x2))

class OpSimpleAttention(nn.Module):
   
    def __init__(self, d, heads=4, drop=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, D = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, D)
        return self.drop(self.proj(x))

# ==============================================================================

def build_component_pool(d, cfg):
    drop = cfg.get('micro_drop', 0.1)
    
    
    return [
        OpSwiGLU(d, drop),                 
        OpGatedMLP(d, drop),              
        OpResNeXt(d, groups=8, drop=drop),
        OpSimpleAttention(d, heads=8, drop=drop),
        OpIdentity(d, drop)              
    ]

OP_NAMES = ["SwiGLU", "GatedMLP", "ResNeXt", "Attention", "Identity"]

class FixedMicroBlock(nn.Module):
    def __init__(self, d, op):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.op = op
    
    def forward(self, x):
        return x + self.op(self.ln(x))

class MicroBlock(nn.Module):
    def __init__(self, d, op_list, tau, hard):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ops = nn.ModuleList(op_list)
        self.alpha = nn.Parameter(1e-3 * torch.randn(len(op_list)))
        self.tau = tau
        self.hard = hard

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        
        if self.training:
            weights = F.gumbel_softmax(self.alpha, tau=self.tau, hard=self.hard)
        else:
            idx = torch.argmax(self.alpha)
            weights = F.one_hot(idx, num_classes=len(self.ops)).float()
        
        out = sum(w * op(x_norm) for w, op in zip(weights, self.ops) if w > 1e-6)
        return residual + out

class MicroExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nas_cfg, specific_choices=None):
        super().__init__()
        self.nas_cfg = nas_cfg
        self.input_dim = input_dim
        self.bn_dim = hidden_dim 
        drop = nas_cfg.get('micro_drop', 0.1)
        self.num_layers = int(nas_cfg.get('micro_layers', nas_cfg.get('num_layers', 2)))

        self.proj_in = nn.Linear(input_dim, self.bn_dim)
        
        self.blocks = nn.ModuleList()
        
        if specific_choices is not None:
            actual_layers = len(specific_choices)
            for i in range(actual_layers):
                op_idx = specific_choices[i]
                pool = build_component_pool(self.bn_dim, nas_cfg)
                chosen_op = pool[op_idx]
                self.blocks.append(FixedMicroBlock(self.bn_dim, chosen_op))
        else:
            tau = nas_cfg.get('tau', 1.0)
            hard = nas_cfg.get('hard_gumbel', True)
            for _ in range(self.num_layers):
                pool = build_component_pool(self.bn_dim, nas_cfg)
                self.blocks.append(MicroBlock(self.bn_dim, pool, tau, hard))
        
        self.proj_out = nn.Linear(self.bn_dim, output_dim)
        self.dropout = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.proj_in(x))
        for block in self.blocks:
            x = block(x)
        return self.dropout(self.proj_out(x))


def build_micro_expert(input_dim, hidden_dim, output_dim, nas_cfg, expert_choices=None):
    return MicroExpert(input_dim, hidden_dim, output_dim, nas_cfg, specific_choices=expert_choices)
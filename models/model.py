import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from models.nas.mix_expert import build_micro_expert, OpResNeXt, OpSEBlock, OpSimpleMLP, FixedMicroBlock
from models.nas.mix_share import SharePatternMixerHard


class _SimpleMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), 
            nn.ReLU(),
            nn.Dropout(drop), 
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x):
        # return self.linear(x)
        return self.net(x)


class BioEncoderFusion(nn.Module):

    def __init__(self, rna_dim, dis_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        # 1. 维度对齐
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        self.dis_proj = nn.Sequential(
            nn.LayerNorm(dis_dim),         
            nn.Linear(dis_dim, hidden_dim),
            nn.GELU(),                      
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)      
        )
        
   

        self.rna_encoder = nn.Identity()
        self.dis_encoder = nn.Identity()
        
   
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )


        self.fusion_proj = nn.Linear(hidden_dim*2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.out_dim = hidden_dim 

    def forward(self, x_rna, x_dis):
   
        if x_rna.shape[1] == 0:
            h_rna = torch.zeros(x_rna.shape[0], self.out_dim, device=x_rna.device)
        else:
            feat_rna = F.gelu(self.rna_proj(x_rna))
            h_rna = feat_rna + self.rna_encoder(feat_rna) 
             
    


       
        if x_dis.shape[1] == 0:
            h_dis = torch.zeros(x_dis.shape[0], self.out_dim, device=x_dis.device)
           
            gamma = torch.zeros_like(h_rna)
            beta  = torch.zeros_like(h_rna)
        else:
            feat_dis = F.gelu(self.dis_proj(x_dis))
            h_dis = feat_dis + self.dis_encoder(feat_dis)
            
        
            film_params = self.film_generator(h_dis) # [B, 2*hidden]
            gamma, beta = torch.chunk(film_params, 2, dim=-1) # [B, hidden], [B, hidden]
        
    
        interact = torch.cat([h_rna, h_dis], dim=-1)
        
    
        fused = self.fusion_proj(interact)        
        fused = self.layer_norm(self.dropout(fused))

        return h_rna, fused, h_dis

class BottleneckAdapter(nn.Module):
    def __init__(self, d_in, bottleneck, drop=0.1):
        super().__init__()
        if bottleneck == 0: self.net = nn.Identity()
        else:
            h = max(4, min(bottleneck, d_in))
            self.net = nn.Sequential(
                nn.Linear(d_in, h), nn.GELU(),
                nn.Linear(h, d_in), nn.Dropout(drop)
            )
    def forward(self, x): return x + self.net(x)

class AdapterMix(nn.Module):
  
    def __init__(self, d, bottleneck_list, drop=0.1, tau=1.5, hard=False, id_bias=0.0):
        super().__init__()
        self.ops = nn.ModuleList(
            [nn.Identity()] + [BottleneckAdapter(d, b, drop) for b in bottleneck_list]
        )
        self.alpha = nn.Parameter(0.01 * torch.randn(len(self.ops)))
        self.tau = tau; self.hard = hard
        if id_bias != 0.0:
            with torch.no_grad(): self.alpha[0].add_(float(id_bias))

    def forward(self, x):
        if self.training:
            w = F.gumbel_softmax(self.alpha, tau=self.tau, hard=self.hard, dim=-1)
        else:
            idx = self.alpha.argmax()
            w = F.one_hot(idx, len(self.ops)).float().to(x.device)
        return sum(w[i] * op(x) for i, op in enumerate(self.ops))


class BaseMMoE(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.input_dim = int(config['input_dim'])

    
        if 'family_dim' in config:
            self.family_dim = int(config['family_dim'])
        else:
            print(" Warning: 'family_dim' missing in config. Fallback to 832 (640+64+128).")
            self.family_dim = 640 + 64 + 128

        self.num_families = int(config['num_families'])

       
        self.rna_dim = self.family_dim
        self.dis_dim = self.input_dim - self.family_dim
      
        if 'mmoe' in config:
            self.inner_dim = int(config['mmoe'].get('expert_hidden_dim', 256))
        else:
            self.inner_dim = int(config.get('expert_hidden_dim', 256))
            
     
        self.interaction = BioEncoderFusion(
            self.rna_dim, self.dis_dim,
            hidden_dim=self.inner_dim,
            dropout=config.get('dropout_rate', 0.1)
        )
        self.expert_input_dim = self.interaction.out_dim

        self.contrastive_head = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, 128)
        )

    def _process_inputs(self, x):
        x_rna = x[:, :self.family_dim]
        x_dis = x[:, self.family_dim:]
        
      
        h_rna, h_fused, h_dis = self.interaction(x_rna, x_dis)
      
        return h_rna, h_fused, h_dis


class MMoEModel(BaseMMoE):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nas_cfg = dict(config.get('nas', {}))
        mmoe_cfg = dict(config.get('mmoe', {}))
        
        self.num_experts = int(mmoe_cfg.get('num_experts', 4))
        self.expert_output_dim = int(mmoe_cfg.get('expert_output_dim', 128))
        self.tower_hidden_dim = int(mmoe_cfg.get('tower_hidden_dim', 64))
        self.dropout_rate = float(config.get('dropout_rate', 0.3))
        
    
        self.disable_mmoe = config.get('disable_mmoe', False)

        
        self.experts = nn.ModuleList([
            build_micro_expert(self.expert_input_dim, self.inner_dim, 
                               self.expert_output_dim, self.nas_cfg)
            for _ in range(self.num_experts)
        ])

     
        self.family_gate = nn.Linear(self.expert_input_dim, self.num_experts)
        self.association_gate = nn.Linear(self.expert_input_dim, self.num_experts)

       
        if self.nas_cfg.get('adapter_search', False):
            cands = self.nas_cfg.get('adapter_candidates', [16,32,48])
            self.family_adapter = AdapterMix(self.expert_output_dim, cands, drop=0.1, id_bias=-0.5)
            self.assoc_adapter  = AdapterMix(self.expert_output_dim, cands, drop=0.1, id_bias=-1.0)
        else:
            self.family_adapter = BottleneckAdapter(self.expert_output_dim, 32, 0.1)
            self.assoc_adapter  = BottleneckAdapter(self.expert_output_dim, 32, 0.1)

     
        self.dropout = nn.Dropout(self.dropout_rate)
        self.family_tower = nn.Sequential(
            nn.Linear(self.expert_output_dim, self.tower_hidden_dim),
            nn.GELU(), nn.Dropout(self.dropout_rate),
            nn.Linear(self.tower_hidden_dim, self.num_families)
        )
        self.association_tower = nn.Sequential(
            nn.Linear(self.expert_output_dim, self.tower_hidden_dim),
            nn.GELU(), nn.Dropout(self.dropout_rate),
            nn.Linear(self.tower_hidden_dim, 1)
        )
        
     
        self.warmup = False 
        router_cfg = dict(config.get('router', {}))
        self.router_tau = float(router_cfg.get('tau', 1.0))
        self.router_tau_warmup = float(router_cfg.get('tau_warmup', 2.5))
        self.router_eps_warmup = float(router_cfg.get('eps_warmup', 0.10))

        self.router_noise_std = float(router_cfg.get('noise_std', 0.10))
        self.router_noise_warmup_std = float(router_cfg.get('noise_warmup_std', 0.20))

    def _run_experts(self, x):
     
        return torch.stack([e(x) for e in self.experts], dim=1)

    def _mix(self, x_gate, gate, experts_out, shared_gate=None):
     
        if getattr(self, 'disable_mmoe', False) and shared_gate is not None:
            gate = shared_gate

        logits = gate(x_gate)

        if self.training:
            std = self.router_noise_warmup_std if self.warmup else self.router_noise_std
            if std and std > 0:
                logits = logits + torch.randn_like(logits) * std

   
        tau = self.router_tau_warmup if self.warmup else self.router_tau
        tau = max(0.05, float(tau))

        w = torch.softmax(logits / tau, dim=-1)  # [B, E]

        if self.warmup:
            eps = float(self.router_eps_warmup)
            if eps > 0:
                w = (1.0 - eps) * w + eps / self.num_experts

        mixed = torch.sum(experts_out * w.unsqueeze(-1), dim=1)
        return mixed, w

    def forward(self, x, task='both', return_embed=False):
     
        h_rna, h_fused, h_dis = self._process_inputs(x)
        
     
        if task == 'contrastive':
        
            z_rna = F.normalize(self.contrastive_head(h_rna), dim=1)
            z_dis = F.normalize(self.contrastive_head(h_dis), dim=1)
            return z_rna, z_dis

      
        experts_out = self._run_experts(h_fused) 
        
      
        fam_feat, fam_w = self._mix(h_rna, self.family_gate, experts_out,
                                     shared_gate=self.association_gate)
        asc_feat, asc_w = self._mix(h_fused, self.association_gate, experts_out,
                                     shared_gate=self.association_gate)

        fam_feat = self.family_adapter(fam_feat)  
        asc_feat = self.assoc_adapter(asc_feat)  

     
        if self.training:
      
            avg_usage = (fam_w.mean(dim=0) + asc_w.mean(dim=0)) / 2
         
            target = 1.0 / self.num_experts
            balance_loss = ((avg_usage - target) ** 2).mean() * self.num_experts # Scale it
        else:
            balance_loss = 0.0

     
        fam_logits = self.family_tower(self.dropout(fam_feat))
        asc_logits = self.association_tower(self.dropout(asc_feat))
        
        if return_embed:
            return fam_logits, asc_logits, h_dis, balance_loss
            
        if task == 'family': 
            return fam_logits, balance_loss
        
        
        if task == 'association': 
            return asc_logits, balance_loss # 修正此处
        
    
        return fam_logits, asc_logits

    def arch_parameters(self):
        for e in self.experts:
            for blk in e.blocks:
                if hasattr(blk, 'alpha'): yield blk.alpha
        if isinstance(self.family_adapter, AdapterMix): yield self.family_adapter.alpha
        if isinstance(self.assoc_adapter, AdapterMix): yield self.assoc_adapter.alpha

    def shared_parameters(self):
        for e in self.experts: yield from e.parameters()
        yield from self.family_gate.parameters()
        yield from self.association_gate.parameters()
        yield from self.family_adapter.parameters()
        yield from self.assoc_adapter.parameters()
        yield from self.interaction.parameters()
        yield from self.contrastive_head.parameters()
    
    def family_tower_parameters(self): yield from self.family_tower.parameters()
    def association_tower_parameters(self): yield from self.association_tower.parameters()
    
    @torch.no_grad()
    def export_architecture(self):
        micro, micro_names = [], []
        for e in self.experts:
            row_idx, row_name = [], []
            for blk in e.blocks:
                idx = int(torch.argmax(blk.alpha))
                row_idx.append(idx); row_name.append(str(idx))
            micro.append(row_idx); micro_names.append(row_name)
       
        arch = {'micro_expert_choices': micro}
        if hasattr(self, 'family_adapter') and isinstance(self.family_adapter, AdapterMix):
            arch['adapter_family_choice'] = int(self.family_adapter.alpha.argmax().item())
        if hasattr(self, 'assoc_adapter') and isinstance(self.assoc_adapter, AdapterMix):
            arch['adapter_association_choice'] = int(self.assoc_adapter.alpha.argmax().item())
        return arch


class StaticMMoEModel(BaseMMoE):

    def __init__(self, input_dim, num_families, num_experts, expert_output_dim, 
                 tower_hidden_dim, dropout_rate, micro_choices, nas_cfg, 
                 family_dim=None, expert_hidden_dim=256,
                 router_cfg: Dict[str, Any] = None, disable_mmoe=False):
        
        config = {
            'input_dim': input_dim, 'num_families': num_families,
            'family_dim': family_dim, 'expert_hidden_dim': expert_hidden_dim,
            'dropout_rate': dropout_rate, 'disable_mmoe': disable_mmoe
        }
        router_cfg = dict(router_cfg or {})
        config['router'] = router_cfg

        super().__init__(config)
        
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim
        self.dropout_rate = dropout_rate
        
      
        self.disable_mmoe = disable_mmoe

     
        self.experts = nn.ModuleList()
        use_simple_mlp = nas_cfg.get('use_simple_mlp_expert', False)

        if use_simple_mlp:
         
            drop = nas_cfg.get('micro_drop', 0.1)
            for _ in range(num_experts):
                simple_expert = _SimpleMLP(
                    self.expert_input_dim, self.inner_dim, self.expert_output_dim, drop=drop
                )
                self.experts.append(simple_expert)
        else:
            if micro_choices is None:
                micro_choices = [[0]*2 for _ in range(num_experts)]
            for i in range(num_experts):
                row_choices = micro_choices[i] if i < len(micro_choices) else [0]*2
                expert = build_micro_expert(
                    self.expert_input_dim, self.inner_dim, self.expert_output_dim,
                    nas_cfg,
                    expert_choices=row_choices
                )
                self.experts.append(expert)
        
        self.family_gate = nn.Linear(self.expert_input_dim, num_experts)
        self.association_gate = nn.Linear(self.expert_input_dim, num_experts)
        
    
        fam_sel = nas_cfg.get('adapter_family_choice', 0)
        asc_sel = nas_cfg.get('adapter_association_choice', 0)
        cands = nas_cfg.get('adapter_candidates', [16,32,48,64])
        
        def _build_adp(sel):
            if not sel: return nn.Identity()
            idx = max(0, min(sel-1, len(cands)-1))
            return BottleneckAdapter(expert_output_dim, cands[idx], 0.1)
            
        self.family_adapter = _build_adp(fam_sel)
        self.assoc_adapter = _build_adp(asc_sel)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.family_tower = nn.Sequential(
            nn.Linear(expert_output_dim, tower_hidden_dim),
            nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(tower_hidden_dim, num_families)
        )
        self.association_tower = nn.Sequential(
            nn.Linear(expert_output_dim, tower_hidden_dim),
            nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(tower_hidden_dim, 1)
        )
        self.warmup = False

        router_cfg = dict(config.get('router', {}))
        self.router_tau = float(router_cfg.get('tau', 1.0))
        self.router_tau_warmup = float(router_cfg.get('tau_warmup', 2.5))
        self.router_eps_warmup = float(router_cfg.get('eps_warmup', 0.10))

        self.router_noise_std = float(router_cfg.get('noise_std', 0.10))
        self.router_noise_warmup_std = float(router_cfg.get('noise_warmup_std', 0.20))

    def _run_experts(self, x):
        return torch.stack([e(x) for e in self.experts], dim=1)

    def _mix(self, x_gate, gate, experts_out, shared_gate=None,share_input=None):
        if getattr(self, 'disable_mmoe', False):
            if shared_gate is not None:
                gate = shared_gate
            if share_input is not None:
                x_gate = share_input

        logits = gate(x_gate)

        if self.training:
            std = self.router_noise_warmup_std if self.warmup else self.router_noise_std
            if std and std > 0:
                logits = logits + torch.randn_like(logits) * std

        tau = self.router_tau_warmup if self.warmup else self.router_tau
        tau = max(0.05, float(tau))

        w = torch.softmax(logits / tau, dim=-1)  # [B, E]

        if self.warmup:
            eps = float(self.router_eps_warmup)
            if eps > 0:
                w = (1.0 - eps) * w + eps / self.num_experts

        mixed = torch.sum(experts_out * w.unsqueeze(-1), dim=1)
        return mixed, w

    def forward(self, x, task='both', return_embed=False):
        h_rna, h_fused, h_dis = self._process_inputs(x)
        
        if task == 'contrastive':
            z_rna = F.normalize(self.contrastive_head(h_rna), dim=1)
            z_dis = F.normalize(self.contrastive_head(h_dis), dim=1)
            return z_rna, z_dis

        experts_out = self._run_experts(h_fused)
        
        fam_feat, fam_w = self._mix(h_rna, self.family_gate, experts_out,
                                     shared_gate=self.association_gate,
                                     share_input=h_fused)
        asc_feat, asc_w = self._mix(h_fused, self.association_gate, experts_out,
                                     shared_gate=self.association_gate,
                                     share_input=h_fused)

        fam_feat = self.family_adapter(fam_feat)  
        asc_feat = self.assoc_adapter(asc_feat)  

        if self.training:
            avg_usage = (fam_w.mean(dim=0) + asc_w.mean(dim=0)) / 2
            target = 1.0 / self.num_experts
            balance_loss = ((avg_usage - target) ** 2).mean() * self.num_experts
        else:
            balance_loss = 0.0
        
        fam_logits = self.family_tower(self.dropout(fam_feat))
        asc_logits = self.association_tower(self.dropout(asc_feat))
        
        if return_embed:
            return fam_logits, asc_logits, h_dis

       
        if task == 'family': 
            return fam_logits, balance_loss
        
       
        if task == 'association': 
            return asc_logits, balance_loss
            
        return fam_logits, asc_logits

    def shared_parameters(self):
        for e in self.experts: yield from e.parameters()
        yield from self.family_gate.parameters()
        yield from self.association_gate.parameters()
        if hasattr(self, 'family_adapter'): yield from self.family_adapter.parameters()
        if hasattr(self, 'assoc_adapter'): yield from self.assoc_adapter.parameters()
        yield from self.interaction.parameters()
        yield from self.contrastive_head.parameters()

    def family_tower_parameters(self): yield from self.family_tower.parameters()
    def association_tower_parameters(self): yield from self.association_tower.parameters()

    @staticmethod
    def from_nas_choices(input_dim, num_families, nas_cfg, dims, family_dim=None,router_cfg: Dict[str, Any] = None, disable_mmoe=False):
        return StaticMMoEModel(
            input_dim=input_dim, num_families=num_families,
            num_experts=len(nas_cfg.get('micro_expert_choices') or [[0]*2]*4),
            expert_output_dim=dims['expert_output_dim'],
            tower_hidden_dim=dims['tower_hidden_dim'],
            dropout_rate=dims['dropout_rate'],
            micro_choices=nas_cfg.get('micro_expert_choices'),
            nas_cfg=nas_cfg, family_dim=family_dim,
            expert_hidden_dim=dims.get('expert_hidden_dim', 256),
            router_cfg=router_cfg, disable_mmoe=disable_mmoe
        )


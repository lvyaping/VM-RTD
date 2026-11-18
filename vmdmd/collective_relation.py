"""Training-only relation mixture; teacher never enters the gate."""
import math
import torch
from torch import nn
from .objectives import local_relations

class CollectiveRelation(nn.Module):
    def __init__(self, mode='uniform', eta=.5, radius=1, auxiliary_ratio=None):
        super().__init__()
        if mode not in ('uniform','learned') or not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError('Expected uniform/learned and finite eta in [0,1]')
        if auxiliary_ratio is not None and (not math.isfinite(auxiliary_ratio) or auxiliary_ratio<0):
            raise ValueError('Auxiliary ratio must be finite and nonnegative')
        self.auxiliary_ratio=auxiliary_ratio
        self.mode,self.eta,self.radius=mode,eta,radius
        self.gate=None
        if mode=='learned':
            # Identical training RNG and initially identical weights to uniform control.
            with torch.random.fork_rng(devices=[]):
                self.gate=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,4))
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)
        self.diagnostics=None

    def mix(self, relations):
        weights=(torch.full_like(relations,.25) if self.gate is None else
                 self.gate(relations.detach()).softmax(dim=-1))
        return (weights*relations).sum(-1),weights

    def forward(self, branches, teacher):
        if len(branches)!=4 or any(x.shape[0]!=teacher.shape[0] or x.shape[-2:]!=teacher.shape[-2:] for x in branches):
            raise ValueError('Expected four branches with matching teacher batch/grid')
        target=local_relations(teacher.detach(),self.radius)
        student=[local_relations(x,self.radius) for x in branches]
        branch=teacher.new_zeros((),dtype=torch.float32)
        collective=branch.clone(); entropy=branch.clone(); usage=branch.new_zeros(4)
        count=sum(t.numel() for t in target)
        for k,t in enumerate(target):
            r=torch.stack([x[k] for x in student],dim=-1)
            mixed,a=self.mix(r)
            branch=branch+(r-t.unsqueeze(-1)).square().sum()/4
            collective=collective+(mixed-t).square().sum()
            with torch.no_grad():
                usage+=a.sum(tuple(range(a.ndim-1)))
                entropy+=(-(a*a.clamp_min(1e-12).log()).sum(-1)).sum()/math.log(4)
        branch,collective=branch/count,collective/count
        if self.auxiliary_ratio is not None:
            # Same reduction order as original independent branch MSE.
            branch=torch.stack([sum((s-t).square().sum() for s,t in zip(row,target))/count for row in student]).mean()
        self.diagnostics=torch.cat([torch.stack([branch.detach(),collective.detach(),entropy/count]),usage/count])
        if self.auxiliary_ratio is not None:
            return branch+self.auxiliary_ratio*collective
        return (1-self.eta)*branch+self.eta*collective

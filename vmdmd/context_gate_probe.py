"""Frozen stage-3 probes; no integration into segmentation training."""
import torch
from torch import nn
OFFSETS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def restore_pairs(r,t,h=16,w=16):
    if r.shape!=(sum((h-abs(dy))*(w-abs(dx)) for dy,dx in OFFSETS),4) or t.shape!=(len(r),):
        raise ValueError('Unexpected cache layout: expected stage3 size256 radius1, single image')
    x=r.new_zeros(h,w,8,4);y=t.new_zeros(h,w,8);mask=torch.zeros(h,w,8,dtype=torch.bool)
    pos=0
    for k,(dy,dx) in enumerate(OFFSETS):
        y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
        n=(y1-y0)*(x1-x0)
        x[y0:y1,x0:x1,k]=r[pos:pos+n].reshape(y1-y0,x1-x0,4)
        y[y0:y1,x0:x1,k]=t[pos:pos+n].reshape(y1-y0,x1-x0)
        mask[y0:y1,x0:x1,k]=True;pos+=n
    return x.reshape(-1,8,4),y.reshape(-1,8),mask.reshape(-1,8)

class GateProbe(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind not in ('context','point'):raise ValueError('Unknown probe')
        self.kind=kind
        # 2368 context vs 2362 pointwise parameters; same depth/activation.
        self.net=nn.Sequential(nn.Linear(40,32),nn.GELU(),nn.Linear(32,32)) if kind=='context' else nn.Sequential(nn.Linear(4,262),nn.GELU(),nn.Linear(262,4))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,r,mask):
        clean=r.detach()*mask.unsqueeze(-1)
        logits=self.net(torch.cat([clean.flatten(1),mask.to(r.dtype)],-1)).reshape(-1,8,4) if self.kind=='context' else self.net(clean)
        weights=logits.softmax(-1)
        return (weights*r).sum(-1),weights

def masked_mse(pred,target,mask):
    return (pred-target).square()[mask].mean()

"""Pair-weighted mean/std/shape relation KD. No adapters or new parameters."""
import math
import torch
import torch.nn.functional as F

def validate_component_weights(weights):
    if len(weights)!=3 or any(not math.isfinite(w) or w<0 for w in weights) or not any(w>0 for w in weights):
        raise ValueError('Three finite nonnegative component weights required, at least one positive')
    return [float(w) for w in weights]

def relations(x,radius):
    x=F.normalize(x.float(),dim=1,eps=1e-6)
    b,c,h,w=x.shape
    if not 1<=radius<min(h,w):raise ValueError('Invalid relation radius')
    padded=F.pad(x,(radius,)*4)
    ones=x.new_ones(b,1,h,w);border=F.pad(ones,(radius,)*4)
    values=[];valid=[]
    for dy,dx in [(-radius,-radius),(-radius,0),(-radius,radius),(0,-radius),(0,radius),(radius,-radius),(radius,0),(radius,radius)]:
        values.append((x*padded[:,:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]).sum(1))
        valid.append(border[:,0,radius+dy:radius+dy+h,radius+dx:radius+dx+w])
    return torch.stack(values,1),torch.stack(valid,1)

def decompose_relations(student,teacher,valid):
    # vector_norm has a finite zero subgradient: no unstable sqrt(var) or rho division.
    n=valid.sum(1)
    ms=(student*valid).sum(1)/n;mt=(teacher*valid).sum(1)/n
    cs=(student-ms[:,None])*valid;ct=(teacher-mt[:,None])*valid
    ss=torch.linalg.vector_norm(cs,dim=1)/n.sqrt()
    st=torch.linalg.vector_norm(ct,dim=1)/n.sqrt()
    bias=(ms-mt).square()
    scale=(ss-st).square()
    shape=2*(ss*st-(cs*ct).sum(1)/n)
    # No epsilon/correlation cutoff or clipping: preserve the exact MSE identity.
    return torch.stack([(v*n).sum()/n.sum() for v in (bias,scale,shape)])

def component_relation_loss(student,teacher,weights,radius=1):
    if student.shape[0]!=teacher.shape[0] or student.shape[-2:]!=teacher.shape[-2:]:
        raise ValueError('Batch and spatial grids must match')
    s,valid=relations(student,radius);t,_=relations(teacher.detach(),radius)
    components=decompose_relations(s,t,valid)
    return (components*components.new_tensor(weights)).sum()

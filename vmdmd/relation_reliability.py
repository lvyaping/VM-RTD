"""Teacher flip consistency weighting. No student-dependent weights."""
import torch
from .objectives import local_relations

def validate_config(cfg):
    mode=cfg.get('relation_reliability')
    if mode is None:return
    if mode not in ('aligned','shuffled') or cfg.get('mode')!='branch' or cfg.get('adapter')!='relation' or cfg.get('kd_stages')!=[2,3] or cfg.get('kd_stage_weights')!=[.5,.5]:
        raise ValueError('Reliability requires equal stage34 branch relation')
    if any(cfg.get(k) is not None for k in ('relation_curriculum','collective_mode','collective_aux_weight','cross_scale_weight','kd_direction_weights','relation_component_weights')) or cfg.get('task_gradient',False):
        raise ValueError('Reliability cannot be combined with other KD extensions')

@torch.no_grad()
def rank_weights(u):
    # Per-image ranks over all valid offsets/pairs; ties share their average rank.
    rows=[]
    for row in u:
        if not torch.isfinite(row).all():raise FloatingPointError('Nonfinite teacher discrepancy')
        n=row.numel()
        if n<2:rows.append(torch.ones_like(row));continue
        vals,order=row.sort()
        _,counts=torch.unique_consecutive(vals,return_counts=True)
        ends=counts.cumsum(0);starts=ends-counts
        ranks=torch.repeat_interleave((starts+ends-1).to(row.dtype)/2,counts)
        w=torch.empty_like(row);w[order]=1.5-ranks/(n-1);rows.append(w)
    return torch.stack(rows)

@torch.no_grad()
def make_weights(target,restored,mode,generator):
    lengths=[t[0].numel() for t in target]
    u=torch.cat([(t-v).abs().flatten(1) for t,v in zip(target,restored)],1)
    weights=rank_weights(u);parts=list(weights.split(lengths,dim=1))
    if mode=='shuffled':
        if generator is None:raise ValueError('Dedicated shuffle generator required')
        parts=[torch.stack([row[torch.randperm(len(row),generator=generator).to(row.device)] for row in part]) for part in parts]
    elif mode!='aligned':raise ValueError('Unknown reliability mode')
    flat=torch.cat(parts,1)
    return [w.reshape_as(t) for w,t in zip(parts,target)],torch.stack([u.mean(),u.std(unbiased=False),flat.std(unbiased=False),(flat*u).sum()/flat.sum()])

def reliability_loss(branches,teacher,restored_teacher,mode='aligned',generator=None,radius=1):
    if teacher.shape!=restored_teacher.shape or len(branches)!=4 or any(b.shape[0]!=teacher.shape[0] or b.shape[-2:]!=teacher.shape[-2:] for b in branches):raise ValueError('Matching grids and four branches required')
    tr=local_relations(teacher.detach(),radius);other=local_relations(restored_teacher.detach(),radius)
    weights,diag=make_weights(tr,other,mode,generator)
    sr=[local_relations(b,radius) for b in branches]
    denom=sum(w.sum() for w in weights)
    loss=torch.stack([sum((s-t).square().mul(w).sum() for s,t,w in zip(row,tr,weights))/denom for row in sr]).mean()
    with torch.no_grad():
        raw=torch.stack([sum((s-t).square().sum() for s,t in zip(row,tr))/sum(t.numel() for t in tr) for row in sr]).mean()
    return loss,torch.cat([diag,torch.stack([raw,loss.detach()])])

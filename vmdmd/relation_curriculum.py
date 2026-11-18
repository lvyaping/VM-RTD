"""Detached reachable-target curriculum and matched scalar KD warmup."""
import math
import torch
from .objectives import local_relations

def progress(epoch,warmup_epochs):
    if type(warmup_epochs) is not int or warmup_epochs<2 or type(epoch) is not int or epoch<1:
        raise ValueError('Positive epoch and warmup_epochs >=2 required')
    return min(1.,(epoch-1)/(warmup_epochs-1))

def validate_config(cfg):
    mode=cfg.get('relation_curriculum')
    if mode is None:return
    if mode not in ('target','weight') or cfg.get('mode')!='branch' or cfg.get('adapter')!='relation' or cfg.get('kd_stages')!=[2,3] or cfg.get('kd_stage_weights')!=[.5,.5]:
        raise ValueError('Curriculum requires equal stage34 branch relation')
    if any(cfg.get(k) is not None for k in ('kd_direction_weights','relation_component_weights','collective_mode','collective_aux_weight','cross_scale_weight')) or cfg.get('task_gradient',False):
        raise ValueError('Curriculum cannot be combined with other KD extensions')
    if not math.isfinite(cfg['beta']) or cfg['beta']<=0:raise ValueError('Positive finite beta required')
    progress(1,cfg.get('curriculum_epochs',10))

def detached_target(relations,teacher,lam):
    if not math.isfinite(lam) or not 0<=lam<=1:raise ValueError('Lambda must be in [0,1]')
    with torch.no_grad():
        t=teacher.detach();r=relations.detach()
        lower=r.min(-1).values;upper=r.max(-1).values
        reachable=torch.maximum(lower,torch.minimum(t,upper))
        target=t if lam==1 else reachable+lam*(t-reachable)
    return target,reachable,(t<lower)|(t>upper)

def curriculum_loss(branches,teacher,lam,radius=1):
    if len(branches)!=4 or any(b.shape[0]!=teacher.shape[0] or b.shape[-2:]!=teacher.shape[-2:] for b in branches):
        raise ValueError('Four matching branch grids required')
    tr=local_relations(teacher.detach(),radius)
    sr=[local_relations(b,radius) for b in branches]
    targets=[];outside=tr[0].new_zeros(());shift=outside.clone()
    for k,t in enumerate(tr):
        target,reachable,out=detached_target(torch.stack([v[k] for v in sr],-1),t,lam)
        targets.append(target);outside+=out.sum();shift+=(target-t).square().sum()
    count=sum(t.numel() for t in tr)
    loss=torch.stack([sum((s-t).square().sum() for s,t in zip(row,targets))/count for row in sr]).mean()
    with torch.no_grad():
        raw=torch.stack([sum((s-t).square().sum() for s,t in zip(row,tr))/count for row in sr]).mean()
        diagnostics=torch.stack([loss.detach(),raw,outside/count,shift/count])
    return loss,diagnostics

"""Fixed 2x2 feature pooling, then cross-stage relation differences."""
import torch
import torch.nn.functional as F
from .objectives import local_relations

def relation_delta(fine,coarse,radius=1):
    if fine.shape[0]!=coarse.shape[0] or fine.shape[-2:]!=(coarse.shape[-2]*2,coarse.shape[-1]*2):
        raise ValueError('Expected aligned batch and exactly 2x spatial grid')
    a=local_relations(F.avg_pool2d(fine.float(),2,2),radius)
    b=local_relations(coarse,radius)
    return [y-x for x,y in zip(a,b)]

def cross_scale_relation_loss(branches,teacher,radius=1):
    if len(branches[2])!=4 or len(branches[3])!=4:raise ValueError('Expected four branches at both stages')
    target=relation_delta(teacher[2].detach(),teacher[3].detach(),radius)
    losses=[]
    for fine,coarse in zip(branches[2],branches[3]):
        if fine.shape[0]!=teacher[2].shape[0] or fine.shape[-2:]!=teacher[2].shape[-2:] or coarse.shape[-2:]!=teacher[3].shape[-2:]:
            raise ValueError('Teacher/student grid mismatch')
        student=relation_delta(fine,coarse,radius)
        losses.append(sum((s-t).square().sum() for s,t in zip(student,target))/sum(t.numel() for t in target))
    return torch.stack(losses).mean()

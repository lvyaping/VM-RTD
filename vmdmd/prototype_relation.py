"""GT-conditioned pixel-to-prototype cosine-map distillation (binary 2D)."""
import math
import torch
import torch.nn.functional as F


def validate_config(cfg):
    if 'prototype_weight' not in cfg:
        if any(k in cfg for k in ('prototype_mode','prototype_boundary_radius','prototype_stages','prototype_stage_weights')):raise ValueError('Prototype options require prototype_weight')
        return
    validate_options(cfg.get('prototype_mode','binary'),cfg.get('prototype_boundary_radius',3))
    validate_prototype_stages(cfg.get('prototype_stages'),cfg.get('prototype_stage_weights'))
    w=cfg['prototype_weight']
    if not math.isfinite(w) or w<0 or not math.isfinite(cfg['beta']) or cfg['beta']<=0:
        raise ValueError('Prototype weight must be finite/nonnegative; beta positive')
    if cfg.get('mode')!='branch' or cfg.get('adapter')!='relation' or cfg.get('kd_stages')!=[2,3] or cfg.get('kd_stage_weights')!=[.5,.5]:
        raise ValueError('Prototype KD requires equal stage34 branch relation')
    if any(cfg.get(k) is not None for k in ('relation_curriculum','relation_reliability','collective_mode','collective_aux_weight','cross_scale_weight','kd_direction_weights','relation_component_weights')) or cfg.get('task_gradient',False):
        raise ValueError('Prototype experiment cannot combine other KD extensions')


def prototype_relation_loss(student,teacher,mask):
    """Equal image mean; within each image equal mean over present GT classes.

    Soft occupancy preserves nonzero small lesions during feature-grid reduction.
    It does not recover detail already lost by the encoder. Maps cover all pixels.
    Teacher and masks are detached; student prototypes remain differentiable.
    """
    if student.ndim!=4 or teacher.ndim!=4 or student.shape[0]!=teacher.shape[0] or student.shape[-2:]!=teacher.shape[-2:]:
        raise ValueError('Expected BCHW student/teacher on matching spatial grids')
    if mask is None or mask.ndim!=4 or mask.shape[:2]!=(student.shape[0],1):
        raise ValueError('Expected B1HW binary segmentation mask')
    mask=mask.detach().float()
    if not torch.isfinite(mask).all() or (mask<0).any() or (mask>1).any():
        raise ValueError('Mask must be finite in [0,1]')
    occupancy=F.adaptive_avg_pool2d(mask,student.shape[-2:])
    weights=torch.cat([1-occupancy,occupancy],1).flatten(2)
    mass=weights.sum(-1)
    valid=mass>0
    def maps(feature):
        x=feature.float().flatten(2)
        prototypes=torch.bmm(weights,x.transpose(1,2))/mass.clamp_min(1e-6).unsqueeze(-1)
        return torch.bmm(F.normalize(prototypes,dim=-1,eps=1e-6),F.normalize(x,dim=1,eps=1e-6))
    sm=maps(student)
    with torch.no_grad():tm=maps(teacher.detach())
    errors=(sm-tm).square().mean(-1)
    loss=((errors*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
    # Foreground and background present-image fractions, respectively.
    return loss,valid.float().mean(0).flip(0).detach()


def validate_options(mode,radius):
    if mode not in ('binary','regions','binary_region_balanced'):raise ValueError('Unknown prototype mode')
    if type(radius) is not int or radius<1:raise ValueError('Boundary radius must be a positive integer')


@torch.no_grad()
def region_occupancy(mask,size,radius=3):
    """Interior, two-sided boundary, far background; replicate image edges.
    Radius is measured on the training mask, before area pooling.
    """
    validate_options('regions',radius)
    if mask.ndim!=4 or mask.shape[1]!=1 or not torch.isfinite(mask).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Regions require finite binary B1HW labels')
    m=mask.detach().float()
    padded=F.pad(m,(radius,)*4,mode='replicate')
    k=2*radius+1
    dilation=F.max_pool2d(padded,k,1)
    erosion=1-F.max_pool2d(1-padded,k,1)
    return F.adaptive_avg_pool2d(torch.cat([erosion,dilation-erosion,1-dilation],1),size)


def region_prototype_relation_loss(student,teacher,mask,radius=3):
    """Mean images -> valid query regions -> valid prototype channels.
    Both query-region means and prototypes use soft region occupancy.
    No mask enters inference. Teacher detached, student prototypes differentiable.
    """
    if student.ndim!=4 or teacher.ndim!=4 or student.shape[0]!=teacher.shape[0] or student.shape[-2:]!=teacher.shape[-2:]:
        raise ValueError('Expected BCHW features with matching spatial grids')
    if mask is None or mask.shape[0]!=student.shape[0]:raise ValueError('Matching labels required')
    weights=region_occupancy(mask,student.shape[-2:],radius).flatten(2)
    mass=weights.sum(-1);valid=mass>0
    def maps(f):
        x=f.float().flatten(2)
        p=torch.bmm(weights,x.transpose(1,2))/mass.clamp_min(1e-6).unsqueeze(-1)
        return torch.bmm(F.normalize(p,dim=-1,eps=1e-6),F.normalize(x,dim=1,eps=1e-6))
    sm=maps(student)
    with torch.no_grad():tm=maps(teacher.detach())
    # B x prototype x query-region: average pixels within each query region.
    errors=torch.bmm((sm-tm).square(),weights.transpose(1,2))/mass.clamp_min(1e-6).unsqueeze(1)
    pairs=valid.unsqueeze(2)&valid.unsqueeze(1)
    loss=((errors*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)).mean()
    return loss,valid.float().mean(0).detach()


def validate_prototype_stages(stages=None,weights=None):
    if stages is None:
        if weights is not None:raise ValueError('Prototype stage weights require explicit stages')
        return [3],[1.]
    stages=list(stages)
    if not stages or any(type(s) is not int or s not in (2,3) for s in stages) or len(set(stages))!=len(stages):
        raise ValueError('Prototype stages must be distinct indices 2/3 (physical stages 3/4)')
    weights=[1.]*len(stages) if weights is None else list(weights)
    if len(weights)!=len(stages) or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('One positive finite prototype weight per stage required')
    scaled=[w/max(weights) for w in weights]
    return stages,[w/sum(scaled) for w in scaled]


def binary_region_balanced_loss(student,teacher,mask,radius=3):
    """Two class prototypes, three equally weighted spatial query regions.
    Prototype construction is identical to binary KD; only spatial reduction changes.
    Returns query-region present fractions in interior/boundary/far-background order.
    """
    if student.ndim!=4 or teacher.ndim!=4 or student.shape[0]!=teacher.shape[0] or student.shape[-2:]!=teacher.shape[-2:]:
        raise ValueError('Expected BCHW features with matching spatial grids')
    if mask is None or mask.ndim!=4 or mask.shape[:2]!=(student.shape[0],1):
        raise ValueError('Expected matching B1HW labels')
    query=region_occupancy(mask,student.shape[-2:],radius).flatten(2)
    occupancy=F.adaptive_avg_pool2d(mask.detach().float(),student.shape[-2:])
    weights=torch.cat([1-occupancy,occupancy],1).flatten(2)
    mass=weights.sum(-1);qmass=query.sum(-1)
    valid=mass>0;qvalid=qmass>0
    def maps(feature):
        x=feature.float().flatten(2)
        p=torch.bmm(weights,x.transpose(1,2))/mass.clamp_min(1e-6).unsqueeze(-1)
        return torch.bmm(F.normalize(p,dim=-1,eps=1e-6),F.normalize(x,dim=1,eps=1e-6))
    sm=maps(student)
    with torch.no_grad():tm=maps(teacher.detach())
    errors=torch.bmm((sm-tm).square(),query.transpose(1,2))/qmass.clamp_min(1e-6).unsqueeze(1)
    pairs=valid.unsqueeze(2)&qvalid.unsqueeze(1)
    loss=((errors*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)).mean()
    return loss,qvalid.float().mean(0).detach()

"""Fixed teacher targets; shared student-only adapters; equal scan-branch weights."""
import torch
from torch import nn
import torch.nn.functional as F


def segmentation_loss(logits, target):
    p, y = logits.sigmoid().flatten(1), target.flatten(1)
    dice = (2*(p*y).sum(1)+1)/(p.sum(1)+y.sum(1)+1)
    return F.binary_cross_entropy_with_logits(logits,target)+1-dice.mean()


class Adapter(nn.Module):
    """No teacher inputs. Context = residual depthwise local convolution, not attention."""
    def __init__(self, channels, kind):
        super().__init__()
        self.context = nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False) if kind=='context' else None
        self.projection = nn.Conv2d(channels,channels,1,bias=False)
        nn.init.eye_(self.projection.weight[:,:,0,0])
        if self.context is not None:
            nn.init.zeros_(self.context.weight)

    def forward(self,x):
        if self.context is not None:
            x=x+self.context(x)
        return self.projection(x)


def local_relations(x, radius=1):
    """Eight ordered offsets on a Chebyshev ring; valid pairs only, no padding.

    Each output is [B, H-|dy|, W-|dx|]. Coordinates refer to the feature grid.
    Cosine affinity is descriptive similarity, not an attention/causal weight.
    """
    if x.ndim != 4 or radius < 1 or min(x.shape[-2:]) <= radius:
        raise ValueError('Relation radius must be positive and smaller than H and W')
    x=F.normalize(x.float(),p=2,dim=1,eps=1e-6)
    h,w=x.shape[-2:]
    relations=[]
    for dy,dx in [(-radius,-radius),(-radius,0),(-radius,radius),
                  (0,-radius),(0,radius),(radius,-radius),(radius,0),(radius,radius)]:
        y0,y1=max(0,-dy),min(h,h-dy)
        x0,x1=max(0,-dx),min(w,w-dx)
        a=x[:,:,y0:y1,x0:x1]
        b=x[:,:,y0+dy:y1+dy,x0+dx:x1+dx]
        relations.append((a*b).sum(1))
    return relations


def relation_mse(student, teacher_relations, radius=1):
    student_relations=local_relations(student,radius)
    # Average over all valid directed pairs, not over padded pixels or offset means.
    return sum((s-t).square().sum() for s,t in zip(student_relations,teacher_relations))/sum(t.numel() for t in teacher_relations)


class RelationContext(nn.Module):
    """Training-only shared residual depthwise 3x3, identity initialization."""
    def __init__(self, channels):
        super().__init__()
        # Preserve RNG so adding an identity adapter does not shift training RNG.
        with torch.random.fork_rng(devices=[]):
            self.local=nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
            nn.init.zeros_(self.local.weight)

    def forward(self,x):
        return x+self.local(x)


class Distillation(nn.Module):
    def __init__(self,mode,stage,adapter,relation_radius=1,relation_purity=.9):
        super().__init__()
        self.mode,self.stage=mode,stage
        self.relation_purity=relation_purity
        self.kind,self.relation_radius=adapter,relation_radius
        self.adapter=Adapter([96,192,384,768][stage],adapter) if mode!='none' and adapter not in ('relation','relation_context') else None
        if mode!='none' and adapter=='relation_context':
            self.adapter=RelationContext([96,192,384,768][stage])

    def forward(self,features,branches,teacher,mask=None):
        if self.mode in ('native_label','native_matched'):
            view=features[self.stage]
            targets=None if self.mode=='native_label' else local_relations(teacher[self.stage].detach(),self.relation_radius)
            if targets is not None and teacher[self.stage].shape[-2:] != view.shape[-2:]:
                raise ValueError('Teacher/student grids must match')
            loss=label_relation_loss(view,mask,targets,self.relation_radius,self.relation_purity)
            return loss,list(view.new_zeros(4,4).unbind())
        if self.mode in ('branch_pair','branch_gap','branch_gap_label'):
            from .gap_relation import gap_loss
            views=branches[self.stage]
            if len(views)!=4:raise ValueError('Expected four coordinate-restored branches')
            t=None if teacher is None else teacher[self.stage].detach()
            if t is not None and t.shape[-2:]!=views[0].shape[-2:]:raise ValueError('Grid mismatch')
            loss=gap_loss(views,t,mask,self.mode,self.relation_radius,self.relation_purity)
            w=views[0].new_zeros(4,4);w[self.stage]=.25
            return loss,list(w.unbind())
        target=teacher[self.stage].detach()
        # Real post-block encoder output, including residual paths, before downsample.
        views=[features[self.stage]] if self.mode=='native' else branches[self.stage]
        if self.mode!='native' and len(views)!=4:
            raise ValueError('Expected four SS2D branches in original spatial coordinates')
        if self.mode=='fused':
            # Matched extraction point: mean of the SAME four post-norm/gate/projection views.
            # This is not the residual stage output or the native sum-before-LayerNorm output.
            views=[torch.stack(views).mean(0)]
        losses=[]
        target_relations=local_relations(target,self.relation_radius) if self.kind in ('relation','relation_context') else None
        for view in views:
            if self.kind in ('relation','relation_context'):
                if view.shape[0]!=target.shape[0] or view.shape[-2:]!=target.shape[-2:]:
                    raise ValueError('Relation KD requires matching batch and spatial grids; no silent resizing')
                if self.kind=='relation_context':view=self.adapter(view)
                losses.append(relation_mse(view,target_relations,self.relation_radius))
                continue
            if view.shape[-2:]!=target.shape[-2:]:
                view=F.interpolate(view,target.shape[-2:],mode='bilinear',align_corners=False)
            view=self.adapter(view)
            if view.shape!=target.shape:
                raise ValueError('Teacher/student feature shapes differ after adaptation')
            losses.append((1-F.cosine_similarity(view.float(),target.float(),dim=1)).mean())
        weights=target.new_zeros(4,4)
        if self.mode=='branch':weights[self.stage]=.25
        return torch.stack(losses).mean(),list(weights.unbind())


class TrainingSystem(nn.Module):
    def __init__(self,student,teacher,kd):
        super().__init__()
        self.student,self.teacher,self.kd=student,teacher,kd
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        if teacher is not None:teacher.requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(mode)
        if self.teacher is not None:self.teacher.eval()
        return self

    def normalize(self,x):return (x-self.mean)/self.std

    def predict(self,x):return self.student(self.normalize(x))

    def forward(self,image,target,alpha=1.,beta=.1):
        image=self.normalize(image)
        kd=image.new_zeros(());weights=[]
        if self.kd.mode!='none' and beta>0:
            if self.kd.mode in ('native','native_label','native_matched'):
                logits,features=self.student(image,return_stage_features=True)
                branches=[]
            else:
                logits,features,branches=self.student(image,return_stage_features=True,return_directional_features=True)
            with torch.no_grad():teacher=self.teacher(image) if self.teacher is not None else None
            if getattr(self.kd,'reliability_mode',None) is not None:
                with torch.no_grad():
                    flipped=self.teacher(image.flip(-1))
                    restored=[v.flip(-1) for v in flipped]
                kd,weights=self.kd(features,branches,teacher,target,restored_teacher=restored)
            else:
                kd,weights=self.kd(features,branches,teacher,target)
        else:logits=self.student(image)
        task=segmentation_loss(logits,target)
        return alpha*task+beta*kd,task.detach(),kd.detach(),weights


def label_relation_loss(feature, mask, teacher_relations=None, radius=1, purity=.9):
    """Same pure-cell pairs for GT and teacher controls; global valid-pair mean.
    GT targets: same binary class 1, different class 0 (orthogonality convention).
    Training-mask occupancy, not native-resolution diagnostic mask occupancy.
    """
    if not .5 < purity <= 1:
        raise ValueError('Purity must be in (.5,1]')
    occupancy=F.adaptive_avg_pool2d(mask.detach().float(),feature.shape[-2:])[:,0]
    pure=(occupancy>=purity)|(occupancy<=1-purity)
    foreground=occupancy>=purity
    h,w=feature.shape[-2:]
    values=local_relations(feature,radius)
    total=feature.float().sum()*0
    count=feature.new_zeros((),dtype=torch.float32)
    offsets=[(-radius,-radius),(-radius,0),(-radius,radius),(0,-radius),(0,radius),(radius,-radius),(radius,0),(radius,radius)]
    for i,(dy,dx) in enumerate(offsets):
        y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
        a=(slice(None),slice(y0,y1),slice(x0,x1))
        b=(slice(None),slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))
        valid=pure[a]&pure[b]
        target=(foreground[a]==foreground[b]).float() if teacher_relations is None else teacher_relations[i].detach()
        total=total+((values[i]-target).square()*valid).sum()
        count=count+valid.sum()
    # DDP averages gradients: compensate to optimize the global pair mean even
    # when individual ranks contain different numbers of valid pairs.
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        global_count=count.clone();dist.all_reduce(global_count)
        global_total=total.detach().clone();dist.all_reduce(global_total)
        gradient_loss=total*dist.get_world_size()/global_count.clamp_min(1)
        return gradient_loss+(global_total/global_count.clamp_min(1)-gradient_loss.detach())
    return total/count.clamp_min(1)

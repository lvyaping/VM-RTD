"""Fixed normalized stage weights for branch relation KD."""
import math
import torch
from torch import nn
from .objectives import Distillation

def validate_stages(stages,weights):
    if not stages or any(type(s) is not int or s not in range(4) for s in stages) or len(set(stages))!=len(stages):
        raise ValueError('Stages must be distinct indices in 0..3')
    if len(weights)!=len(stages) or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('One positive finite weight per stage is required')
    total=sum(weights)
    return [w/total for w in weights]

def validate_direction_weights(stages,weights):
    if len(weights)!=len(stages):raise ValueError('One direction-weight row per selected stage required')
    result=[]
    for row in weights:
        if len(row)!=4 or any(not math.isfinite(w) or w<0 for w in row) or max(row)<=0:
            raise ValueError('Four finite nonnegative direction weights required; row cannot be all zero')
        scaled=[w/max(row) for w in row]
        result.append([w/sum(scaled) for w in scaled])
    return result

class MultiStageDistillation(nn.Module):
    def __init__(self,stages,weights,radius=1,component_weights=None,mode='branch',direction_weights=None,cross_scale_ratio=0.,collective_mode=None,collective_eta=.5,collective_auxiliary_ratio=None,target_curriculum=False,reliability_mode=None,prototype_ratio=0.,prototype_mode='binary',prototype_boundary_radius=3,prototype_stages=None,prototype_stage_weights=None):
        super().__init__()
        if mode not in ('branch','fused'):raise ValueError('Expected branch or fused')
        if mode=='fused' and component_weights is not None:raise ValueError('Fused control uses original MSE only')
        if direction_weights is not None and (mode!='branch' or component_weights is not None):
            raise ValueError('Custom directions require branch mode and original relation MSE')
        self.direction_weights=None if direction_weights is None else validate_direction_weights(stages,direction_weights)
        if not math.isfinite(cross_scale_ratio) or cross_scale_ratio<0:raise ValueError('Invalid cross-scale ratio')
        if cross_scale_ratio and (list(stages)!=[2,3] or mode!='branch' or component_weights is not None or direction_weights is not None):
            raise ValueError('Cross-scale requires original equal-direction stage34 branch KD')
        if target_curriculum and (list(stages)!=[2,3] or mode!='branch' or component_weights is not None or direction_weights is not None or cross_scale_ratio or collective_mode is not None):
            raise ValueError('Target curriculum requires original stage34 branch KD')
        if reliability_mode is not None and (reliability_mode not in ('aligned','shuffled') or list(stages)!=[2,3] or mode!='branch' or component_weights is not None or direction_weights is not None or cross_scale_ratio or collective_mode is not None or target_curriculum):
            raise ValueError('Reliability requires original stage34 branch KD')
        if not math.isfinite(prototype_ratio) or prototype_ratio<0:raise ValueError('Invalid prototype ratio')
        if prototype_ratio and (list(stages)!=[2,3] or list(weights)!=[.5,.5] or mode!='branch' or component_weights is not None or direction_weights is not None or cross_scale_ratio or collective_mode is not None or target_curriculum or reliability_mode is not None):
            raise ValueError('Prototype requires original equal stage34 branch KD')
        from .prototype_relation import validate_options
        validate_options(prototype_mode,prototype_boundary_radius)
        from .prototype_relation import validate_prototype_stages
        self.prototype_stages,self.prototype_stage_weights=validate_prototype_stages(prototype_stages,prototype_stage_weights)
        self.prototype_stage_diagnostics=None
        self.prototype_mode=prototype_mode
        self.prototype_boundary_radius=prototype_boundary_radius
        self.prototype_ratio=prototype_ratio
        self.prototype_diagnostics=None
        self.reliability_mode=reliability_mode
        self.reliability_generator=torch.Generator().manual_seed(42) if reliability_mode else None
        self.reliability_diagnostics=None
        self.target_curriculum=target_curriculum
        self.curriculum_lambda=0.
        self.curriculum_diagnostics=None
        self.collective=None
        self.collective_diagnostics=None
        if collective_mode is not None:
            if list(stages)!=[2,3] or mode!='branch' or component_weights is not None or direction_weights is not None or cross_scale_ratio:
                raise ValueError('Collective KD requires stage34 branch relation without other extensions')
            from .collective_relation import CollectiveRelation
            self.collective=nn.ModuleList([CollectiveRelation(collective_mode,collective_eta,radius,auxiliary_ratio=collective_auxiliary_ratio) for _ in stages])
        self.cross_scale_ratio=cross_scale_ratio
        self.cross_scale_diagnostics=None
        self.mode=mode
        self.component_weights=component_weights
        self.radius=radius
        if component_weights is not None:
            from .relation_components import validate_component_weights
            self.component_weights=validate_component_weights(component_weights)
        self.stages=list(stages)
        self.weights=validate_stages(stages,weights)
        self.losses=nn.ModuleList([Distillation(mode,s,'relation',radius) for s in stages])

    def forward(self,features,branches,teacher,mask=None,restored_teacher=None):
        if self.reliability_mode is not None:
            if restored_teacher is None:raise ValueError('Restored flipped teacher features required')
            from .relation_reliability import reliability_loss
            outputs=[reliability_loss(branches[s],teacher[s],restored_teacher[s],self.reliability_mode,self.reliability_generator,self.radius) for s in self.stages]
            terms=[v[0] for v in outputs]
            self.reliability_diagnostics=torch.stack([v[1] for v in outputs])
        elif self.target_curriculum:
            from .relation_curriculum import curriculum_loss
            outputs=[curriculum_loss(branches[s],teacher[s],self.curriculum_lambda,self.radius) for s in self.stages]
            terms=[v[0] for v in outputs]
            self.curriculum_diagnostics=torch.stack([v[1] for v in outputs])
        elif self.collective is not None:
            terms=[loss(branches[s],teacher[s]) for s,loss in zip(self.stages,self.collective)]
            self.collective_diagnostics=torch.stack([loss.diagnostics for loss in self.collective])
        elif self.direction_weights is not None:
            from .objectives import local_relations,relation_mse
            terms=[]
            for s,row in zip(self.stages,self.direction_weights):
                target=teacher[s].detach()
                if len(branches[s])!=4:raise ValueError('Expected four branches')
                tr=local_relations(target,self.radius)
                losses=[]
                for v in branches[s]:
                    if v.shape[0]!=target.shape[0] or v.shape[-2:]!=target.shape[-2:]:raise ValueError('Grid mismatch')
                    losses.append(relation_mse(v,tr,self.radius))
                values=torch.stack(losses)
                terms.append((values*values.new_tensor(row)).sum())
        elif self.component_weights is None:
            terms=[loss(features,branches,teacher,mask)[0] for loss in self.losses]
        else:
            from .relation_components import component_relation_loss
            terms=[]
            for s in self.stages:
                if len(branches[s])!=4:raise ValueError('Expected four restored branches')
                terms.append(torch.stack([component_relation_loss(v,teacher[s],self.component_weights,self.radius)
                                          for v in branches[s]]).mean())
        result=sum(w*x for w,x in zip(self.weights,terms))
        if self.prototype_ratio:
            from .prototype_relation import prototype_relation_loss
            outputs=[]
            for s in self.prototype_stages:
                if self.prototype_mode=='binary_region_balanced':
                    from .prototype_relation import binary_region_balanced_loss
                    outputs.append(binary_region_balanced_loss(features[s],teacher[s],mask,self.prototype_boundary_radius))
                elif self.prototype_mode=='regions':
                    from .prototype_relation import region_prototype_relation_loss
                    outputs.append(region_prototype_relation_loss(features[s],teacher[s],mask,self.prototype_boundary_radius))
                else:
                    outputs.append(prototype_relation_loss(features[s],teacher[s],mask))
            proto=sum(w*v[0] for w,v in zip(self.prototype_stage_weights,outputs))
            present=sum(w*v[1] for w,v in zip(self.prototype_stage_weights,outputs))
            self.prototype_stage_diagnostics=torch.stack([v[0].detach() for v in outputs])
            self.prototype_diagnostics=torch.cat([torch.stack([result.detach(),proto.detach()]),present])
            result=result+self.prototype_ratio*proto
        if self.cross_scale_ratio:
            from .cross_scale_relation import cross_scale_relation_loss
            delta=cross_scale_relation_loss(branches,teacher,self.radius)
            self.cross_scale_diagnostics=torch.stack([terms[0].detach(),terms[1].detach(),delta.detach()])
            result=result+self.cross_scale_ratio*delta
        # Log effective stage-times-direction weights; total is 1.
        weights=result.new_zeros(4,4)
        if self.mode=='branch' and self.collective is None:
            for i,(s,w) in enumerate(zip(self.stages,self.weights)):
                row=[.25]*4 if self.direction_weights is None else self.direction_weights[i]
                weights[s]=weights.new_tensor(row)*w
        # Fused supervision has no separate per-direction losses.
        return result,list(weights.unbind())

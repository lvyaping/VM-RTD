"""First-order task-gradient guided branch weighting; no learned weight network."""
import math
import torch
import torch.distributed as dist
from .objectives import TrainingSystem,segmentation_loss,local_relations,relation_mse

def guided_weights(task_grad,kd_grads,temperature=.5,uniform_mix=.5):
    if not math.isfinite(temperature) or temperature<=0 or not 0<=uniform_mix<=1:
        raise ValueError('temperature > 0 and uniform_mix in [0,1] required')
    # Per-image cosine, then batch mean. Zero gradient has neutral score zero.
    g=task_grad.detach().float().flatten(1)
    scores=[]
    for grad in kd_grads:
        h=grad.detach().float().flatten(1)
        denom=g.norm(dim=1)*h.norm(dim=1)
        scores.append(torch.where(denom>1e-12,(g*h).sum(1)/denom.clamp_min(1e-12),torch.zeros_like(denom)).mean())
    scores=torch.stack(scores).clamp(-1,1)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scores);scores/=dist.get_world_size()
    if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite gradient cosine')
    weights=uniform_mix/4+(1-uniform_mix)*torch.softmax(scores/temperature,dim=0)
    return weights.detach(),scores.detach()

class TaskGradientSystem(TrainingSystem):
    def __init__(self,student,teacher,kd,temperature=.5,uniform_mix=.5):
        super().__init__(student,teacher,kd)
        if not math.isfinite(temperature) or temperature<=0 or not 0<=uniform_mix<=1:
            raise ValueError('Invalid task-gradient settings')
        self.temperature,self.uniform_mix=temperature,uniform_mix
        self.gradient_diagnostics=None

    def forward(self,image,target,alpha=1.,beta=.03):
        if beta<=0:return super().forward(image,target,alpha,beta)
        image=self.normalize(image)
        shared=[]
        # Both segmentation and every branch KD depend on this same tensor.
        hook=self.student.layers[self.kd.stage].register_forward_pre_hook(lambda m,args:shared.append(args[0]))
        try:
            logits,features,branches=self.student(image,return_stage_features=True,return_directional_features=True)
        finally:hook.remove()
        if len(shared)!=1 or not shared[0].requires_grad:
            raise RuntimeError('Expected one differentiable stage input')
        with torch.no_grad():teacher=self.teacher(image)[self.kd.stage]
        views=branches[self.kd.stage]
        if len(views)!=4 or any(v.shape[-2:]!=teacher.shape[-2:] for v in views):
            raise ValueError('Four branches and matching teacher grid required')
        targets=local_relations(teacher.detach(),self.kd.relation_radius)
        terms=[relation_mse(v,targets,self.kd.relation_radius) for v in views]
        task=segmentation_loss(logits,target)
        # Stop at an intermediate activation; do not accumulate parameter grads.
        # create_graph=False avoids second-order differentiation/custom-kernel double backward.
        tg=torch.autograd.grad(task,shared[0],retain_graph=True,create_graph=False)[0]
        kg=[torch.autograd.grad(term,shared[0],retain_graph=True,create_graph=False)[0] for term in terms]
        w,c=guided_weights(tg,kg,self.temperature,self.uniform_mix)
        kd=sum(a*b for a,b in zip(w,terms))
        self.gradient_diagnostics=torch.cat([c,w,-(w*w.clamp_min(1e-12).log()).sum().view(1)]).detach()
        weights=task.new_zeros(4,4);weights[self.kd.stage]=w
        return alpha*task+beta*kd,task.detach(),kd.detach(),list(weights.unbind())

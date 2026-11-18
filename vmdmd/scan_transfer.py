"""Training-only local teacher-relation supervision of SS2D transfer coefficients."""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .objectives import TrainingSystem, segmentation_loss


def validate_config(cfg):
    variant=cfg.get('scan_transfer_variant','full')
    if variant not in ('full','output','endpoint'):raise ValueError('Unknown scan_transfer_variant')
    weight=cfg.get('scan_transfer_weight',0.)
    if not math.isfinite(weight) or weight<0:
        raise ValueError('scan_transfer_weight must be finite and nonnegative')
    if cfg.get('scan_transfer_only',False) and (weight<=0 or variant!='full'):
        raise ValueError('Transfer-only requires positive fixed weight and full variant')
    if not weight:return
    required=(cfg.get('mode')=='branch' and cfg.get('adapter')=='relation'
              and cfg.get('student_arch')=='shallow' and cfg.get('kd_stages')==[2,3]
              and cfg.get('kd_stage_weights')==[.5,.5] and cfg.get('beta')==.1
              and cfg.get('relation_radius',1)==1)
    if not required:raise ValueError('Scan transfer requires shallow original stage34 equal branch relation KD, beta=.1')
    for key in ('prototype_weight','cross_scale_weight','collective_mode','collective_aux_weight',
                'relation_curriculum','relation_reliability','kd_direction_weights',
                'relation_component_weights','task_gradient'):
        if cfg.get(key):raise ValueError('Scan transfer cannot be combined with '+key)


def neighbor_indices(h,w,direction,device):
    """Return scan indices and spatial query/source indices; raw scan order RF,CF,RB,CB."""
    spatial=torch.arange(h*w,device=device).reshape(h,w)
    order=spatial.flatten() if direction%2==0 else spatial.t().flatten()
    if direction>=2:order=order.flip(0)
    inverse=torch.empty_like(order);inverse[order]=torch.arange(h*w,device=device)
    sources=[];valid=[]
    yy=order//w;xx=order%w
    for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        y,x=yy+dy,xx+dx
        q=y.clamp(0,h-1)*w+x.clamp(0,w-1)
        j=inverse[q]
        ok=(y>=0)&(y<h)&(x>=0)&(x<w)&(j<torch.arange(h*w,device=device))
        sources.append(j);valid.append(ok)
    js=torch.stack(sources,1);mask=torch.stack(valid,1)
    keep=mask.sum(1)>=2
    ii=torch.arange(h*w,device=device)[keep]
    return ii,js[keep],mask[keep],order[ii],order[js[keep]]


def coefficient_strength(prefix,delta,A,B,C,ii,jj):
    """Fixed-parameter scan kernel: [batch,query,neighbor], excluding self D skip.
    prefix/delta [batch,channel,L]; A [channel,state]; B,C [batch,state,L].
    """
    elapsed=prefix[:,:,ii].unsqueeze(-1)-prefix[:,:,jj]
    decay=torch.exp(elapsed.unsqueeze(2)*A[None,:,:,None,None])
    write=B[:,:,jj].unsqueeze(1)*delta[:,:,jj].unsqueeze(2)
    read=C[:,:,ii][:,None,:,:,None]
    kernel=(read*decay*write).sum(2)
    return kernel.abs().mean(1)


def endpoint_strength(prefix,delta,A,B,C,ii,jj):
    # Identity path transition. A is intentionally unused, not artificially connected.
    write=B[:,:,jj].unsqueeze(1)*delta[:,:,jj].unsqueeze(2)
    read=C[:,:,ii][:,None,:,:,None]
    return (read*write).sum(2).abs().mean(1)


def output_relation_loss(branches,teacher):
    """Matched KL control; restored branches RF,RB,CF,CB -> raw RF,CF,RB,CB."""
    if len(branches)!=4:raise ValueError('Expected four restored output branches')
    h,w=teacher.shape[-2:]
    t=F.normalize(teacher.detach().float(),dim=1,eps=1e-6).flatten(2)
    losses=[]
    for d,b in enumerate([branches[k] for k in (0,2,1,3)]):
        if b.shape[0]!=teacher.shape[0] or b.shape[-2:]!=(h,w):raise ValueError('Output grid mismatch')
        _,_,valid,pi,pj=neighbor_indices(h,w,d,teacher.device)
        if not len(pi):raise ValueError('No queries with at least two predecessors')
        z=F.normalize(b.float(),dim=1,eps=1e-6).flatten(2)
        ts=(t[:,:,pi].unsqueeze(-1)*t[:,:,pj]).sum(1).masked_fill(~valid[None],float('-inf'))
        ss=(z[:,:,pi].unsqueeze(-1)*z[:,:,pj]).sum(1).masked_fill(~valid[None],float('-inf'))
        target=ts.softmax(-1)
        logprob=ss.log_softmax(-1).masked_fill(~valid[None],0.)
        losses.append((target*(target.clamp_min(1e-30).log()-logprob)).sum(-1).mean())
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite output KL')
    # Nearzero transfer mass is not defined for output cosine supervision.
    return loss,torch.stack(losses).detach(),loss.new_zeros(4)


def transfer_loss(capture,teacher,chunk=16,recompute=True,variant='full'):
    if variant not in ('full','endpoint'):raise ValueError('Expected full or endpoint')
    strength_fn=coefficient_strength if variant=='full' else endpoint_strength
    h,w=capture['shape'];batch=teacher.shape[0]
    if tuple(teacher.shape[-2:])!=(h,w):raise ValueError('Teacher/student grid mismatch')
    t=F.normalize(teacher.detach().float(),dim=1,eps=1e-6).flatten(2)
    delta=F.softplus(capture['dts'].float()+capture['bias'].float()[None,:,:,None])
    A=capture['A'].float();B=capture['B'].float();C=capture['C'].float()
    losses=[];nearzero=[]
    for d in range(4):
        ii,jj,valid,sp_i,sp_j=neighbor_indices(h,w,d,teacher.device)
        if not len(ii):raise ValueError('No queries with at least two predecessors')
        prefix=delta[:,d].cumsum(-1)
        total=delta.new_zeros(());zeros=delta.new_zeros(());count=0
        for start in range(0,len(ii),chunk):
            end=min(start+chunk,len(ii));i,j=ii[start:end],jj[start:end]
            mask=valid[start:end]
            # Padded future indices are replaced before exp, avoiding overflow on invalid entries.
            j=torch.where(mask,j,i[:,None])
            args=(prefix,delta[:,d],A[d],B[:,d],C[:,d],i,j)
            if recompute and torch.is_grad_enabled():
                strength=checkpoint(strength_fn,*args,use_reentrant=False)
            else:strength=strength_fn(*args)
            scores=(t[:,:,sp_i[start:end]].unsqueeze(-1)*t[:,:,sp_j[start:end]]).sum(1)
            scores=scores.masked_fill(~mask[None],float('-inf'))
            target=scores.softmax(-1)  # fixed temperature 1
            mass=(strength+1e-12)*mask[None]
            prob=mass/mass.sum(-1,keepdim=True)
            # Stable log, zero-mass padded entries contribute exactly zero.
            kl=target*(target.clamp_min(1e-30).log()-prob.clamp_min(1e-30).log())
            total=total+kl.sum()
            zeros=zeros+((strength.detach()<1e-12)&mask[None]).sum()
            count+=batch*int(mask.sum())
        losses.append(total/(batch*len(ii)))
        nearzero.append(zeros/max(count,1))
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite scan transfer KL')
    return loss,torch.stack(losses).detach(),torch.stack(nearzero).detach()


class ScanTransferSystem(TrainingSystem):
    def __init__(self,student,teacher,kd,weight,variant='full',transfer_only=False):
        super().__init__(student,teacher,kd)
        self.transfer_only=transfer_only
        self.scan_transfer_variant=variant
        self.scan_transfer_weight=float(weight)
        self.scan_diagnostics=None
        self.keep_calibration_terms=False
        self.calibration_terms=None

    def forward(self,image,target,alpha=1.,beta=.1):
        if not self.scan_transfer_weight:
            return super().forward(image,target,alpha,beta)
        if self.transfer_only:beta=0.
        elif beta<=0:raise ValueError('Scan transfer requires positive relation KD beta')
        scan=self.student.layers[2].blocks[-1].self_attention
        scan.capture_transfer=self.scan_transfer_variant!='output'
        try:
            image=self.normalize(image)
            logits,features,branches=self.student(image,return_stage_features=True,return_directional_features=True)
            capture=getattr(scan,'transfer_capture',None)
        finally:
            scan.capture_transfer=False
            scan.transfer_capture=None
        with torch.no_grad():teacher=self.teacher(image)
        if self.transfer_only:
            relation=logits.new_zeros(())
            weights=list(logits.new_zeros(4,4).unbind())
        else:
            relation,weights=self.kd(features,branches,teacher,target)
        if self.scan_transfer_variant=='output':
            extra,per_direction,nearzero=output_relation_loss(branches[2],teacher[2])
        else:
            extra,per_direction,nearzero=transfer_loss(capture,teacher[2],variant=self.scan_transfer_variant)
        task=segmentation_loss(logits,target)
        combined=self.scan_transfer_weight*extra if self.transfer_only else relation+self.scan_transfer_weight/beta*extra
        self.scan_diagnostics=torch.cat([torch.stack([relation.detach(),extra.detach(),extra.detach()*self.scan_transfer_weight]),per_direction,nearzero])
        if self.keep_calibration_terms:self.calibration_terms=(relation,extra)
        return alpha*task+beta*relation+self.scan_transfer_weight*extra,task.detach(),combined.detach(),weights


def calibrate(system,cfg,device):
    """Four deterministic real train batches, no optimizer, no validation selection."""
    if system.transfer_only:
        raise ValueError('Transfer-only uses the fixed full-method weight; do not recalibrate against disabled relation KD')
    import random
    import numpy as np
    from torch.utils.data import DataLoader,Subset
    from .runner import dataset
    state=random.getstate();npstate=np.random.get_state();was_training=system.training
    train=dataset(cfg,'train');train.epoch=0
    ids=list(range(min(16,len(train))))
    loader=DataLoader(Subset(train,ids),batch_size=4,shuffle=False,num_workers=0)
    scan=system.student.layers[2].blocks[-1].self_attention
    params=[scan.A_logs,scan.x_proj_weight,scan.dt_projs_weight,scan.dt_projs_bias]
    rows=[]
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
            system.train();system.keep_calibration_terms=True
            for image,target,sids in loader:
                system(image.to(device),target.to(device),1.,cfg['beta'])
                rel,aux=system.calibration_terms
                gr=torch.autograd.grad(cfg['beta']*rel,params,retain_graph=True,allow_unused=False)
                ga=torch.autograd.grad(aux,params,allow_unused=True)
                nr=torch.stack([v.float().square().sum() for v in gr]).sum().sqrt().item()
                na=torch.stack([v.float().square().sum() if v is not None else aux.new_zeros(()) for v in ga]).sum().sqrt().item()
                if not math.isfinite(nr+na) or min(nr,na)<=1e-12:
                    raise ValueError('Degenerate calibration gradients; inspect implementation instead of increasing weight')
                rows.append(dict(sample_ids=list(sids),relation_gradient_norm=nr,transfer_gradient_norm=na,
                                 relation_loss=rel.detach().item(),transfer_loss=aux.detach().item()))
                system.calibration_terms=None
    finally:
        system.keep_calibration_terms=False;system.calibration_terms=None;system.train(was_training)
        random.setstate(state);np.random.set_state(npstate)
    if not rows:raise ValueError('No train calibration samples')
    recommended=.1*sum(r['relation_gradient_norm'] for r in rows)/sum(r['transfer_gradient_norm'] for r in rows)
    if not math.isfinite(recommended) or recommended<=0:raise ValueError('Invalid calibrated weight')
    return dict(recommended_weight=recommended,gradient_ratio=.1,batches=rows,variant=system.scan_transfer_variant,
                calibration_seed=cfg['seed'],
                definition='lambda=0.1*mean(norm(grad(0.1*relation)))/mean(norm(grad(transfer)))',
                limitation='Initial train-batch scale heuristic; not an optimal weight or a validation search')

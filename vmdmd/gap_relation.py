"""Label-defined local relation gaps; no adapters or teacher target learning."""
import torch
import torch.nn.functional as F
import torch.distributed as dist

def neighborhood(x,r):
    x=F.normalize(x.float(),dim=1,eps=1e-6)
    b,c,h,w=x.shape
    padded=F.pad(x,(r,r,r,r))
    return torch.stack([(x*padded[:,:,r+dy:r+dy+h,r+dx:r+dx+w]).sum(1)
        for dy,dx in [(0,-r),(0,r),(-r,0),(r,0),(-r,-r),(-r,r),(r,-r),(r,r)]],1)

def gap_loss(views,teacher,mask,kind,radius=1,purity=.9):
    # Axis and diagonal neighbors are handled separately to control 2-D distance.
    h,w=views[0].shape[-2:]
    occ=F.adaptive_avg_pool2d(mask.detach().float(),(h,w))
    pure=(occ>=purity)|(occ<=1-purity); fg=occ>=purity
    offsets=[(0,-radius),(0,radius),(-radius,0),(radius,0),(-radius,-radius),(-radius,radius),(radius,-radius),(radius,radius)]
    p=F.pad(pure.float(),(radius,)*4);g=F.pad(fg.float(),(radius,)*4)
    valid=torch.stack([pure[:,0]&p[:,0,radius+dy:radius+dy+h,radius+dx:radius+dx+w].bool() for dy,dx in offsets],1)
    same=torch.stack([fg[:,0]==g[:,0,radius+dy:radius+dy+h,radius+dx:radius+dx+w].bool() for dy,dx in offsets],1)
    tr=None if teacher is None else neighborhood(teacher.detach(),radius)
    total=views[0].float().sum()*0; count=total.detach().clone()
    for view in views:
        sr=neighborhood(view,radius)
        for lo,hi in [(0,4),(4,8)]:
            pos=valid[:,lo:hi]&same[:,lo:hi];neg=valid[:,lo:hi]&~same[:,lo:hi]
            np=pos.sum(1);nn=neg.sum(1);eligible=(np>0)&(nn>0)
            def avg(z,m,n):return (z[:,lo:hi]*m).sum(1)/n.clamp_min(1)
            ds=avg(sr,pos,np)-avg(sr,neg,nn)
            if kind=='branch_pair':
                err=(sr-tr).square()
                value=.5*(avg(err,pos,np)+avg(err,neg,nn))
            elif kind=='branch_gap':
                dt=avg(tr,pos,np)-avg(tr,neg,nn)
                value=(ds-dt).square()
            elif kind=='branch_gap_label':
                # Label-only reference: margin 1 is a convention, not teacher knowledge.
                value=F.relu(1-ds).square()
            else:raise ValueError(kind)
            total=total+(value*eligible).sum();count=count+eligible.sum()
    if dist.is_available() and dist.is_initialized():
        n=count.clone();dist.all_reduce(n)
        t=total.detach().clone();dist.all_reduce(t)
        loss=total*dist.get_world_size()/n.clamp_min(1)
        return loss+(t/n.clamp_min(1)-loss.detach())
    return total/count.clamp_min(1)

"""Two-process CPU/Gloo compatibility check; not a CUDA performance test."""
import os
import tempfile
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from vmdmd.task_gradient import TaskGradientSystem
from vmdmd.objectives import Distillation

class Student(nn.Module):
 def __init__(self):
  super().__init__();self.stem=nn.Conv2d(3,6,1);self.layers=nn.ModuleList([nn.Identity() for _ in range(3)]+[nn.Conv2d(6,6,1)]);self.head=nn.Conv2d(6,1,1)
 def forward(self,x,return_stage_features=False,return_directional_features=False):
  f=self.layers[3](self.stem(x));b=[f,f*.8,f+f.square()*.1,f-f.square()*.1]
  logits=self.head(f)
  if return_directional_features:return logits,[f]*4,[b]*4
  return logits
class Teacher(nn.Module):
 def __init__(self):super().__init__();self.conv=nn.Conv2d(3,7,1)
 def forward(self,x):return [self.conv(x)]*4
def worker(rank,path):
 torch.set_num_threads(1)
 dist.init_process_group('gloo',init_method='file:///'+path.replace('\\','/'),rank=rank,world_size=2)
 try:
  torch.manual_seed(42)
  system=TaskGradientSystem(Student(),Teacher(),Distillation('branch',3,'relation'))
  model=DDP(system,find_unused_parameters=True)
  optimizer=torch.optim.SGD(model.parameters(),lr=.01)
  for step in range(2):
   torch.manual_seed(100+rank+step);x=torch.rand(2,3,4,4);y=(torch.rand(2,1,4,4)>.5).float()
   optimizer.zero_grad();loss,*_=model(x,y);loss.backward();optimizer.step()
   assert torch.isfinite(loss)
   weights=system.gradient_diagnostics[4:8];out=[torch.empty_like(weights) for _ in range(2)];dist.all_gather(out,weights)
   torch.testing.assert_close(out[0],out[1])
  if rank==0:print('Two-rank DDP forward/backward and synchronized weights PASS')
 finally:dist.destroy_process_group()
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as d:mp.spawn(worker,args=(os.path.join(d,'init'),),nprocs=2,join=True)

import os
import unittest
import torch
from torch import nn
from vmdmd.task_gradient import guided_weights,TaskGradientSystem
from vmdmd.objectives import Distillation,TrainingSystem
os.environ['VMDMD_SCAN_BACKEND']='reference'
torch.set_num_threads(2)

class GradientTests(unittest.TestCase):
 def test_scores(self):
  g=torch.tensor([[1.,0.]])
  w,c=guided_weights(g,[g,-g,torch.tensor([[0.,1.]]),torch.zeros_like(g)])
  torch.testing.assert_close(c,torch.tensor([1.,-1.,0.,0.]))
  self.assertGreater(w[0],w[2]);self.assertGreater(w[2],w[1])
  self.assertTrue((w>=.125).all());self.assertTrue((w<=.625).all())
  torch.testing.assert_close(w.sum(),torch.tensor(1.))
  self.assertFalse(w.requires_grad)
  z,_=guided_weights(torch.zeros_like(g),[g]*4)
  torch.testing.assert_close(z,torch.full((4,),.25))

 def test_real_backbone_first_order_and_uniform_reference(self):
  from vmdmd.models.vmamba import VSSM
  s=VSSM(num_classes=1,depths=[1]*4,depths_decoder=[1]*4,dims=[8,16,32,64],dims_decoder=[64,32,16,8],d_state=2)
  class Teacher(nn.Module):
   def __init__(self):super().__init__();self.conv=nn.Conv2d(3,7,1)
   def forward(self,x):
    f=self.conv(x);return [torch.nn.functional.adaptive_avg_pool2d(f,(64//(4*2**i),)*2) for i in range(4)]
  t=Teacher();kd=Distillation('branch',3,'relation')
  model=TaskGradientSystem(s,t,kd,uniform_mix=1.).eval()
  x=torch.rand(1,3,64,64);y=(torch.rand(1,1,64,64)>.5).float()
  expected=TrainingSystem(s,t,kd).eval()(x,y)[0]
  loss,*_=model(x,y,beta=.1)
  torch.testing.assert_close(loss,expected)
  self.assertTrue(all(p.grad is None for p in s.parameters()))
  loss.backward()
  self.assertGreater(s.layers[3].blocks[-1].self_attention.x_proj_weight.grad.abs().sum().item(),0)
  self.assertTrue(all(p.grad is None for p in t.parameters()))
  self.assertTrue(torch.isfinite(model.gradient_diagnostics).all())
  self.assertEqual(len(s.layers[3]._forward_pre_hooks),0)
  s.zero_grad(set_to_none=True);model.uniform_mix=.5
  loss,*_=model(x,y);loss.backward()
  self.assertTrue(torch.isfinite(loss))

if __name__=='__main__':unittest.main()

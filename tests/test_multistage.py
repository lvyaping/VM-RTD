import unittest
import torch
from vmdmd.multistage import MultiStageDistillation,validate_stages
from vmdmd.objectives import Distillation

class MultiTests(unittest.TestCase):
 def test_sum_and_gradient(self):
  torch.set_num_threads(2)
  b=[[],[],[torch.randn(1,6,4,4,requires_grad=True) for _ in range(4)],[torch.randn(1,9,2,2,requires_grad=True) for _ in range(4)]]
  t=[None,None,torch.randn(1,5,4,4,requires_grad=True),torch.randn(1,7,2,2,requires_grad=True)]
  kd=MultiStageDistillation([2,3],[1.,1.]);loss,w=kd([],b,t)
  expected=sum(Distillation('branch',s,'relation')([],b,t)[0] for s in (2,3))/2
  torch.testing.assert_close(loss,expected)
  loss.backward()
  self.assertTrue(all(x.grad.abs().sum()>0 for s in (2,3) for x in b[s]))
  self.assertTrue(all(t[s].grad is None for s in (2,3)))
  torch.testing.assert_close(torch.stack(w)[2:],torch.full((2,4),.125))
  self.assertEqual(sum(p.numel() for p in kd.parameters()),0)
  torch.testing.assert_close(MultiStageDistillation([3],[1.])([],b,t)[0],Distillation('branch',3,'relation')([],b,t)[0])

 def test_bad_config(self):
  for s,w in [([2,2],[1,1]),([4],[1]),([2,3],[1]),([2,3],[0,1]),([2],[float('nan')])]:
   with self.assertRaises(ValueError):validate_stages(s,w)

if __name__=='__main__':unittest.main()

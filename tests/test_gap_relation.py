import unittest
import torch
from vmdmd.gap_relation import gap_loss
from vmdmd.objectives import Distillation

class GapTests(unittest.TestCase):
 def test_targets_and_gradients(self):
  torch.set_num_threads(2)
  y=torch.zeros(1,1,4,4);y[:,:,:,2:]=1
  t=torch.randn(1,7,4,4,requires_grad=True)
  for mode in ('branch_pair','branch_gap','branch_gap_label'):
   xs=[torch.randn(1,5,4,4,requires_grad=True) for _ in range(4)]
   loss,w=Distillation(mode,3,'relation')([], [[],[],[],xs],None if mode.endswith('label') else [None]*3+[t],y)
   loss.backward()
   self.assertTrue(all(x.grad.abs().sum()>0 for x in xs))
   self.assertIsNone(t.grad)
   torch.testing.assert_close(w[3],torch.full((4,),.25))
  for mode in ('branch_pair','branch_gap'):
   self.assertEqual(gap_loss([t]*4,t,y,mode).item(),0)
  x=torch.cat([1-y,y],1)
  self.assertEqual(gap_loss([x]*4,None,y,'branch_gap_label').item(),0)

 def test_empty_pairs(self):
  xs=[torch.randn(1,3,4,4,requires_grad=True) for _ in range(4)]
  loss=gap_loss(xs,None,torch.zeros(1,1,4,4),'branch_gap_label')
  loss.backward();self.assertEqual(loss.item(),0)
  self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in xs))

if __name__=='__main__':unittest.main()

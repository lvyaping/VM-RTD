import os
import unittest
import torch
from torch import nn
from vmdmd.objectives import Distillation,TrainingSystem,relation_mse,local_relations

os.environ['VMDMD_SCAN_BACKEND']='reference'
torch.set_num_threads(2)

class NativeTests(unittest.TestCase):
    def test_uses_native_not_branch_mean(self):
        torch.manual_seed(1)
        t=torch.randn(1,8,3,5,requires_grad=True)
        x=torch.randn(1,8,3,5,requires_grad=True)
        kd=Distillation('native',3,'relation')
        loss,w=kd([None,None,None,x],[],[None,None,None,t])
        torch.testing.assert_close(loss,relation_mse(x,local_relations(t.detach())))
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertEqual(sum(p.numel() for p in kd.parameters()),0)
        self.assertEqual(torch.stack(w).abs().sum().item(),0)
        same=kd([None,None,None,t.detach()],[],[None,None,None,t])[0]
        self.assertEqual(same.item(),0)

    def test_real_student_native_only_kd_gradient(self):
        from vmdmd.models.vmamba import VSSM
        model=VSSM(num_classes=1,depths=[1]*4,depths_decoder=[1]*4,
                   dims=[8,16,32,64],dims_decoder=[64,32,16,8],d_state=2).eval()
        class Teacher(nn.Module):
            def __init__(self):super().__init__();self.conv=nn.Conv2d(3,9,1)
            def forward(self,x):
                v=self.conv(x)
                return [torch.nn.functional.adaptive_avg_pool2d(v,(64//(4*2**s),)*2) for s in range(4)]
        t=Teacher();system=TrainingSystem(model,t,Distillation('native',3,'relation')).eval()
        x=torch.rand(1,3,64,64);y=torch.zeros(1,1,64,64)
        with torch.no_grad():
            pred=model(system.normalize(x))
            extracted,features=model(system.normalize(x),return_stage_features=True)
            torch.testing.assert_close(pred,extracted)
            self.assertEqual(features[3].shape[-2:],(2,2))
        loss,_,_,_=system(x,y,alpha=0.,beta=1.)
        loss.backward()
        grad=model.layers[3].blocks[-1].self_attention.x_proj_weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in t.parameters()))

if __name__=='__main__':unittest.main()

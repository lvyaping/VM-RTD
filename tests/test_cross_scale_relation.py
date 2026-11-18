import unittest
import torch
from vmdmd.cross_scale_relation import cross_scale_relation_loss,relation_delta
from vmdmd.multistage import MultiStageDistillation

class CrossScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.b={s:[torch.randn(2,5,h,h,requires_grad=True) for _ in range(4)] for s,h in ((2,8),(3,4))}
        self.t={s:torch.randn(2,7,h,h,requires_grad=True) for s,h in ((2,8),(3,4))}
    def test_total_and_gradients(self):
        base=MultiStageDistillation([2,3],[.5,.5])({},self.b,self.t)[0]
        kd=MultiStageDistillation([2,3],[.5,.5],cross_scale_ratio=.5)
        total=kd({},self.b,self.t)[0]
        delta=cross_scale_relation_loss(self.b,self.t)
        torch.testing.assert_close(.1*total,.1*base+.05*delta)
        delta.backward()
        for row in self.b.values():
            for v in row:self.assertTrue(torch.isfinite(v.grad).all() and v.grad.abs().sum()>0)
        self.assertTrue(all(t.grad is None for t in self.t.values()))
        self.assertEqual(len(kd.cross_scale_diagnostics),3)
    def test_identical_zero(self):
        b={s:[t.clone() for _ in range(4)] for s,t in self.t.items()}
        self.assertEqual(cross_scale_relation_loss(b,self.t).item(),0)
    def test_invalid(self):
        with self.assertRaises(ValueError):relation_delta(torch.randn(1,3,7,7),torch.randn(1,4,4,4))
        with self.assertRaises(ValueError):MultiStageDistillation([2,3],[.5,.5],mode='fused',cross_scale_ratio=.5)
    def test_zero_compatibility(self):
        a=MultiStageDistillation([2,3],[.5,.5])({},self.b,self.t)[0]
        b=MultiStageDistillation([2,3],[.5,.5],cross_scale_ratio=0)({},self.b,self.t)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':unittest.main()

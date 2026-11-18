import unittest
import torch
import torch.nn.functional as F
from vmdmd.prototype_relation import binary_region_balanced_loss,prototype_relation_loss,region_occupancy
from vmdmd.multistage import MultiStageDistillation

class BalancedTests(unittest.TestCase):
    def test_manual_two_prototypes_three_queries(self):
        torch.manual_seed(4)
        s=torch.randn(1,3,8,8);t=torch.randn(1,5,8,8)
        m=torch.zeros(1,1,8,8);m[:,:,2:6,2:6]=1
        q=region_occupancy(m,(8,8),1)[0].flatten(1)
        w=torch.cat([1-m,m],1)[0].flatten(1)
        def maps(x):
            x=x[0].flatten(1)
            p=torch.stack([(x*w[c]).sum(1)/w[c].sum() for c in range(2)])
            return F.normalize(p,dim=1)@F.normalize(x,dim=0)
        e=(maps(s)-maps(t)).square()
        manual=torch.stack([(e[c]*q[k]).sum()/q[k].sum() for c in range(2) for k in range(3)]).mean()
        actual,_=binary_region_balanced_loss(s,t,m,1)
        torch.testing.assert_close(actual,manual)
        self.assertGreater(abs(actual.item()-prototype_relation_loss(s,t,m)[0].item()),1e-5)
    def test_missing_small_and_gradients(self):
        s=torch.randn(3,3,4,4,requires_grad=True);t=torch.randn(3,5,4,4,requires_grad=True)
        m=torch.zeros(3,1,16,16);m[1]=1;m[2,0,8,8]=1;m.requires_grad_()
        loss,p=binary_region_balanced_loss(s,t,m,1)
        torch.testing.assert_close(p,torch.tensor([1/3,1/3,2/3]))
        single=torch.stack([binary_region_balanced_loss(s[i:i+1],t[i:i+1],m[i:i+1],1)[0] for i in range(3)]).mean()
        torch.testing.assert_close(loss,single)
        for i in [0,1]:torch.testing.assert_close(binary_region_balanced_loss(s[i:i+1],t[i:i+1],m[i:i+1],1)[0],prototype_relation_loss(s[i:i+1],t[i:i+1],m[i:i+1])[0])
        loss.backward();self.assertTrue(torch.isfinite(s.grad).all());self.assertGreater(s.grad.abs().sum().item(),0);self.assertIsNone(t.grad);self.assertIsNone(m.grad)
    def test_invariance_and_integration(self):
        f=[torch.randn(1,3,4,4,requires_grad=True) for _ in range(4)]
        t=[torch.randn(1,5,4,4,requires_grad=True) for _ in range(4)]
        b=[[torch.randn_like(x,requires_grad=True) for _ in range(4)] for x in f]
        m=torch.zeros(1,1,16,16);m[:,:,4:12,4:12]=1
        same=torch.cat([f[3][:,[2,0,1]],torch.zeros(1,2,4,4)],1)
        self.assertLess(binary_region_balanced_loss(f[3],same,m)[0].item(),1e-12)
        base=MultiStageDistillation([2,3],[.5,.5])
        kd=MultiStageDistillation([2,3],[.5,.5],prototype_ratio=.1,prototype_mode='binary_region_balanced')
        total=kd(f,b,t,m)[0];p=binary_region_balanced_loss(f[3],t[3],m)[0]
        torch.testing.assert_close(.1*total,.1*base(f,b,t,m)[0]+.01*p)
        self.assertEqual(kd.prototype_diagnostics.numel(),5)
        total.backward();self.assertIsNotNone(f[3].grad);self.assertIsNone(f[2].grad)
        self.assertTrue(all(x.grad is None for x in t))
if __name__=='__main__':unittest.main()

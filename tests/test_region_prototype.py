import unittest
import torch
from vmdmd.prototype_relation import region_occupancy,region_prototype_relation_loss,validate_config
from vmdmd.multistage import MultiStageDistillation

class RegionPrototypeTests(unittest.TestCase):
    def test_partition_and_geometry(self):
        m=torch.zeros(1,1,9,9);m[:,:,2:7,2:7]=1
        w=region_occupancy(m,(9,9),1)
        self.assertEqual(w.sum((0,2,3)).tolist(),[9,40,32])
        torch.testing.assert_close(w.sum(1),torch.ones(1,9,9))
        torch.testing.assert_close(region_occupancy(m,(3,3),1).sum(1),torch.ones(1,3,3))
    def test_missing_regions_and_gradients(self):
        m=torch.zeros(3,1,16,16);m[1]=1;m[2,0,8,8]=1
        s=torch.randn(3,3,4,4,requires_grad=True);t=torch.randn(3,5,4,4,requires_grad=True)
        loss,p=region_prototype_relation_loss(s,t,m,1)
        torch.testing.assert_close(p,torch.tensor([1/3,1/3,2/3]))
        singles=[region_prototype_relation_loss(s[i:i+1],t[i:i+1],m[i:i+1],1)[0] for i in range(3)]
        torch.testing.assert_close(loss,torch.stack(singles).mean())
        loss.backward();self.assertTrue(torch.isfinite(s.grad).all());self.assertGreater(s.grad.abs().sum().item(),0);self.assertIsNone(t.grad)
    def test_manual_balanced_formula_and_invariance(self):
        m=torch.zeros(1,1,8,8);m[:,:,2:6,2:6]=1
        s=torch.randn(1,3,4,4);t=torch.randn(1,5,4,4)
        w=region_occupancy(m,(4,4),1).flatten(2)[0]
        def maps(f):
            x=f[0].flatten(1);p=torch.stack([(x*w[c]).sum(1)/w[c].sum() for c in range(3)])
            return torch.nn.functional.normalize(p,dim=1)@torch.nn.functional.normalize(x,dim=0)
        e=(maps(s)-maps(t)).square()
        expected=torch.stack([(e[c]*w[q]).sum()/w[q].sum() for c in range(3) for q in range(3)]).mean()
        torch.testing.assert_close(region_prototype_relation_loss(s,t,m,1)[0],expected)
        same=torch.cat([s[:,[2,0,1]],torch.zeros(1,2,4,4)],1)
        self.assertLess(region_prototype_relation_loss(s,same,m,1)[0].item(),1e-12)
    def test_integration_and_validation(self):
        f=[torch.randn(1,3,4,4,requires_grad=True) for _ in range(4)]
        t=[torch.randn(1,5,4,4) for _ in range(4)]
        b=[[torch.randn(1,3,4,4,requires_grad=True) for _ in range(4)] for _ in range(4)]
        m=torch.zeros(1,1,16,16);m[:,:,4:12,4:12]=1
        base=MultiStageDistillation([2,3],[.5,.5]);kd=MultiStageDistillation([2,3],[.5,.5],prototype_ratio=.1,prototype_mode='regions',prototype_boundary_radius=1)
        v=kd(f,b,t,m)[0];p=region_prototype_relation_loss(f[3],t[3],m,1)[0]
        torch.testing.assert_close(.1*v,.1*base(f,b,t,m)[0]+.01*p)
        self.assertEqual(kd.prototype_diagnostics.numel(),5)
        v.backward();self.assertIsNotNone(f[3].grad);self.assertIsNone(f[2].grad)
        with self.assertRaises(ValueError):validate_config({'prototype_mode':'regions'})
        with self.assertRaises(ValueError):region_occupancy(m,(4,4),0)
if __name__=='__main__':unittest.main()

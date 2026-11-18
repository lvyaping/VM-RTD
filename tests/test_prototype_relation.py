import unittest
import torch
from vmdmd.prototype_relation import prototype_relation_loss,validate_config
from vmdmd.multistage import MultiStageDistillation

class PrototypeTests(unittest.TestCase):
    def test_invariance_and_channels(self):
        torch.manual_seed(1)
        x=torch.randn(2,3,4,4);mask=torch.randint(2,(2,1,16,16)).float()
        t=torch.cat([x[:,[2,0,1]],torch.zeros(2,2,4,4)],1)
        loss,_=prototype_relation_loss(x,t,mask)
        self.assertLess(loss.item(),1e-12)
    def test_gradients_absent_and_small(self):
        s=torch.randn(2,3,4,4,requires_grad=True);t=torch.randn(2,5,4,4,requires_grad=True)
        m=torch.zeros(2,1,32,32);m[0,0,0,0]=1;m.requires_grad_()
        loss,present=prototype_relation_loss(s,t,m);loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all());self.assertGreater(s.grad.abs().sum().item(),0)
        self.assertIsNone(t.grad);self.assertIsNone(m.grad)
        self.assertEqual(present.tolist(),[.5,1.])
        a,_=prototype_relation_loss(s[:1],t[:1],m[:1]);b,_=prototype_relation_loss(s[1:],t[1:],m[1:])
        torch.testing.assert_close(loss,(a+b)/2)
    def test_additive_and_native_path(self):
        f=[torch.randn(2,3,4,4,requires_grad=True) for _ in range(4)]
        t=[torch.randn(2,5,4,4) for _ in range(4)]
        b=[[torch.randn(2,3,4,4,requires_grad=True) for _ in range(4)] for _ in range(4)]
        m=torch.randint(2,(2,1,16,16)).float()
        base=MultiStageDistillation([2,3],[.5,.5]);new=MultiStageDistillation([2,3],[.5,.5],prototype_ratio=.1)
        raw,_=base(f,b,t,m);total,_=new(f,b,t,m);p,_=prototype_relation_loss(f[3],t[3],m)
        torch.testing.assert_close(.1*total,.1*raw+.01*p)
        total.backward();self.assertIsNotNone(f[3].grad);self.assertIsNone(f[2].grad)
        self.assertEqual(list(base.state_dict()),list(new.state_dict()))
        zero=MultiStageDistillation([2,3],[.5,.5],prototype_ratio=0)
        torch.testing.assert_close(zero(f,b,t,m)[0],raw,rtol=0,atol=0)
    def test_invalid_config(self):
        cfg=dict(prototype_weight=.01,beta=.1,mode='branch',adapter='relation',kd_stages=[2,3],kd_stage_weights=[.5,.5])
        validate_config(cfg)
        for extra in [dict(prototype_weight=float('nan')),dict(relation_reliability='aligned'),dict(relation_curriculum='weight')]:
            with self.assertRaises(ValueError):validate_config(dict(cfg,**extra))

if __name__=='__main__':unittest.main()

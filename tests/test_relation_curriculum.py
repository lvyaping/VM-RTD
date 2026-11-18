import unittest
import torch
from vmdmd.relation_curriculum import progress,detached_target,curriculum_loss,validate_config
from vmdmd.objectives import local_relations,relation_mse
from vmdmd.multistage import MultiStageDistillation
class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.b=[torch.randn(2,5,6,7,requires_grad=True) for _ in range(4)]
        self.t=torch.randn(2,8,6,7,requires_grad=True)
    def test_schedule(self):
        self.assertEqual(progress(1,10),0);self.assertEqual(progress(10,10),1);self.assertEqual(progress(30,10),1)
        self.assertAlmostEqual(progress(5,10),4/9)
        with self.assertRaises(ValueError):progress(1,1)
    def test_projection_and_detachment(self):
        r=torch.tensor([[0.,.2,.5,1.],[0.,.2,.5,1.],[.2,.2,.2,.2]],requires_grad=True)
        t=torch.tensor([-.4,.7,.8],requires_grad=True)
        q,_,out=detached_target(r,t,.5)
        torch.testing.assert_close(q,torch.tensor([-.2,.7,.5]));self.assertFalse(q.requires_grad)
        self.assertEqual(out.tolist(),[True,False,True])
    def test_full_teacher_loss_and_gradient_identity(self):
        old=torch.stack([relation_mse(b,local_relations(self.t.detach())) for b in self.b]).mean()
        new,d=curriculum_loss(self.b,self.t,1.)
        torch.testing.assert_close(old,new,rtol=0,atol=0)
        ga=torch.autograd.grad(old,self.b,retain_graph=True);gb=torch.autograd.grad(new,self.b)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(d[3].item(),0.)
    def test_no_target_gradient(self):
        loss,d=curriculum_loss(self.b,self.t,.3);loss.backward()
        self.assertIsNone(self.t.grad);self.assertFalse(d.requires_grad)
        self.assertTrue(all(b.grad is not None and torch.isfinite(b.grad).all() for b in self.b))
    def test_multistage_restoration(self):
        b={2:self.b,3:self.b};t={2:self.t,3:self.t}
        plain=MultiStageDistillation([2,3],[.5,.5]);new=MultiStageDistillation([2,3],[.5,.5],target_curriculum=True)
        new.curriculum_lambda=1.
        torch.testing.assert_close(plain({},b,t)[0],new({},b,t)[0],rtol=0,atol=0)
        self.assertEqual(new.curriculum_diagnostics.shape,(2,4))
    def test_invalid_config(self):
        cfg=dict(mode='branch',adapter='relation',kd_stages=[2,3],kd_stage_weights=[.5,.5],beta=.1,relation_curriculum='target',curriculum_epochs=10)
        validate_config(cfg)
        for updates in [dict(collective_mode='learned'),dict(beta=float('nan')),dict(curriculum_epochs=1),dict(mode='fused')]:
            with self.assertRaises(ValueError):validate_config(dict(cfg,**updates))
    def test_detached_bounds_do_not_receive_extra_gradient(self):
        r=torch.tensor([[0.,.3,.6,1.]],requires_grad=True);t=torch.tensor([2.])
        q,_,_=detached_target(r,t,0.)
        ((r-q[:,None])**2).mean().backward()
        torch.testing.assert_close(r.grad,(r.detach()-1.)/2)
if __name__=='__main__':unittest.main()

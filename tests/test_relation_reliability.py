import unittest
import torch
from vmdmd.relation_reliability import rank_weights,make_weights,reliability_loss
from vmdmd.objectives import local_relations,relation_mse
class Tests(unittest.TestCase):
    def test_rank_ties_and_mean(self):
        u=torch.tensor([[0.,0.,1.,2.],[1.,1.,1.,1.]],requires_grad=True)
        w=rank_weights(u)
        torch.testing.assert_close(w.mean(1),torch.ones(2))
        torch.testing.assert_close(w[1],torch.ones(4))
        self.assertEqual(w[0,0],w[0,1]);self.assertFalse(w.requires_grad)
        self.assertTrue((w>=.5).all() and (w<=1.5).all())
    def test_identical_teacher_restores_mse_gradients(self):
        torch.manual_seed(1)
        b=[torch.randn(2,3,5,6,requires_grad=True) for _ in range(4)];t=torch.randn(2,4,5,6,requires_grad=True)
        new,d=reliability_loss(b,t,t)
        old=torch.stack([relation_mse(v,local_relations(t.detach())) for v in b]).mean()
        torch.testing.assert_close(new,old)
        x=torch.autograd.grad(new,b,retain_graph=True);y=torch.autograd.grad(old,b)
        for a,c in zip(x,y):torch.testing.assert_close(a,c)
        self.assertFalse(d.requires_grad)
    def test_shuffle_preserves_per_image_offset_histogram_and_rng(self):
        t=local_relations(torch.randn(2,3,5,6));v=local_relations(torch.randn(2,3,5,6))
        state=torch.get_rng_state().clone()
        a,_=make_weights(t,v,'aligned',None)
        b,_=make_weights(t,v,'shuffled',torch.Generator().manual_seed(5))
        c,_=make_weights(t,v,'shuffled',torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for x,y,z in zip(a,b,c):
            torch.testing.assert_close(x.flatten(1).sort(1).values,y.flatten(1).sort(1).values)
            self.assertTrue(torch.equal(y,z))
    def test_teacher_detached(self):
        b=[torch.randn(1,3,5,5,requires_grad=True) for _ in range(4)]
        t=torch.randn(1,5,5,5,requires_grad=True);f=torch.randn_like(t,requires_grad=True)
        loss,_=reliability_loss(b,t,f);loss.backward()
        self.assertIsNone(t.grad);self.assertIsNone(f.grad)
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in b))
if __name__=='__main__':unittest.main()

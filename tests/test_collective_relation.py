import unittest
import torch
from vmdmd.collective_relation import CollectiveRelation
from vmdmd.objectives import local_relations,relation_mse
from vmdmd.multistage import MultiStageDistillation

class CollectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.b=[torch.randn(2,5,6,7,requires_grad=True) for _ in range(4)]
        self.t=torch.randn(2,8,6,7,requires_grad=True)
    def test_uniform_formula_and_gradients(self):
        tr=local_relations(self.t.detach())
        rr=[local_relations(b) for b in self.b]
        branch=sum(relation_mse(b,tr) for b in self.b)/4
        mixed=sum((sum(r[k] for r in rr)/4-t).square().sum() for k,t in enumerate(tr))/sum(t.numel() for t in tr)
        loss=CollectiveRelation('uniform')(self.b,self.t)
        torch.testing.assert_close(loss,.5*branch+.5*mixed)
        loss.backward()
        self.assertIsNone(self.t.grad)
        self.assertTrue(all(b.grad is not None and torch.isfinite(b.grad).all() and b.grad.abs().sum()>0 for b in self.b))
    def test_initialization_rng_and_learning(self):
        state=torch.get_rng_state().clone();m=CollectiveRelation('learned')
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.testing.assert_close(m(self.b,self.t),CollectiveRelation('uniform')(self.b,self.t),rtol=0,atol=0)
        opt=torch.optim.AdamW(m.parameters(),lr=.01)
        loss=m(self.b,self.t);loss.backward();opt.step()
        self.assertGreater(m.gate[-1].weight.grad.abs().sum().item(),0)
        m(self.b,self.t)
        self.assertFalse(torch.allclose(m.diagnostics[3:],torch.full((4,),.25)))
    def test_gate_input_detached_and_weights_normalized(self):
        m=CollectiveRelation('learned')
        with torch.no_grad():m.gate[-1].weight.normal_()
        r=torch.randn(2,3,4,requires_grad=True);mix,a=m.mix(r)
        mix.sum().backward()
        torch.testing.assert_close(r.grad,a.detach())
        torch.testing.assert_close(a.sum(-1),torch.ones(2,3))
    def test_eta_zero_and_multistage(self):
        b={2:self.b,3:[x.clone() for x in self.b]};t={2:self.t,3:self.t}
        old=MultiStageDistillation([2,3],[.5,.5])({},b,t)[0]
        m=MultiStageDistillation([2,3],[.5,.5],collective_mode='uniform',collective_eta=0)
        torch.testing.assert_close(old,m({},b,t)[0])
        self.assertEqual(m.collective_diagnostics.shape,(2,7))
    def test_invalid(self):
        for eta in [-1,float('nan'),2]:
            with self.assertRaises(ValueError):CollectiveRelation('learned',eta)
        with self.assertRaises(ValueError):MultiStageDistillation([2,3],[.5,.5],mode='fused',collective_mode='learned')
    def test_additive_formula_and_gradient(self):
        ratio=.5
        m=CollectiveRelation('learned',auxiliary_ratio=ratio)
        loss=m(self.b,self.t)
        tr=local_relations(self.t.detach())
        branch=torch.stack([relation_mse(b,tr) for b in self.b]).mean()
        replacement=CollectiveRelation('learned',eta=1)(self.b,self.t)
        torch.testing.assert_close(.1*loss,.1*branch+.05*replacement)
        actual=torch.autograd.grad(loss,self.b,retain_graph=True)
        expected=torch.autograd.grad(branch+ratio*replacement,self.b)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b)
        loss.backward()
        self.assertGreater(m.gate[-1].weight.grad.abs().sum().item(),0)
        self.assertIsNone(self.t.grad)
    def test_additive_zero_restores_original(self):
        b={2:self.b,3:self.b};t={2:self.t,3:self.t}
        old=MultiStageDistillation([2,3],[.5,.5])({},b,t)[0]
        new=MultiStageDistillation([2,3],[.5,.5],collective_mode='learned',collective_auxiliary_ratio=0)({},b,t)[0]
        torch.testing.assert_close(old,new,rtol=0,atol=0)
    def test_invalid_auxiliary(self):
        for value in [-.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):CollectiveRelation(auxiliary_ratio=value)
    def test_checkpoint_roundtrip(self):
        a=CollectiveRelation('learned');b=CollectiveRelation('learned')
        b.load_state_dict(a.state_dict(),strict=True)
        torch.testing.assert_close(a(self.b,self.t),b(self.b,self.t))

if __name__=='__main__':unittest.main()

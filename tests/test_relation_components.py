import unittest
import torch
from vmdmd.objectives import local_relations,relation_mse
from vmdmd.relation_components import component_relation_loss,decompose_relations,validate_component_weights
from vmdmd.multistage import MultiStageDistillation

class ComponentsTests(unittest.TestCase):
    def test_original_value_and_gradient(self):
        torch.manual_seed(12)
        for radius in (1,2):
            s=torch.randn(2,5,6,7,requires_grad=True);t=torch.randn(2,9,6,7,requires_grad=True)
            old=relation_mse(s,local_relations(t.detach(),radius),radius)
            new=component_relation_loss(s,t,[1,1,1],radius)
            torch.testing.assert_close(old,new,rtol=2e-6,atol=2e-7)
            ga=torch.autograd.grad(old,s)[0];gb=torch.autograd.grad(new,s)[0]
            torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-7)
            self.assertIsNone(t.grad)
    def test_zero_variance_gradients(self):
        s=torch.ones(1,8,3,4,requires_grad=True);t=torch.zeros_like(s);valid=torch.ones_like(s)
        parts=decompose_relations(s,t,valid)
        torch.testing.assert_close(parts,torch.tensor([1.,0.,0.]))
        (parts*torch.tensor([.5,1.,2.])).sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())
    def test_multistage(self):
        branches={s:[torch.randn(1,4,5,5,requires_grad=True) for _ in range(4)] for s in (2,3)}
        teachers={s:torch.randn(1,7,5,5) for s in (2,3)}
        old=MultiStageDistillation([2,3],[.5,.5])({},branches,teachers)[0]
        new=MultiStageDistillation([2,3],[.5,.5],component_weights=[1,1,1])({},branches,teachers)[0]
        torch.testing.assert_close(old,new)
    def test_invalid(self):
        for w in ([0,0,0],[-1,1,1],[float('nan'),1,1],[1,1]):
            with self.assertRaises(ValueError):validate_component_weights(w)

if __name__=='__main__':unittest.main()

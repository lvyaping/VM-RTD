import unittest
import torch
from vmdmd.multistage import MultiStageDistillation
from vmdmd.objectives import local_relations,relation_mse

class FusedTest(unittest.TestCase):
    def test_fused_loss_and_grad(self):
        torch.manual_seed(4)
        b={s:[torch.randn(1,4,5,5,requires_grad=True) for _ in range(4)] for s in (2,3)}
        t={s:torch.randn(1,7,5,5) for s in (2,3)}
        kd=MultiStageDistillation([2,3],[.5,.5],mode='fused')
        actual,_=kd({},b,t)
        expected=sum(.5*relation_mse(torch.stack(b[s]).mean(0),local_relations(t[s])) for s in (2,3))
        torch.testing.assert_close(actual,expected)
        xs=b[2]+b[3]
        ga=torch.autograd.grad(actual,xs);gb=torch.autograd.grad(expected,xs)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
        separate,_=MultiStageDistillation([2,3],[.5,.5])({},b,t)
        self.assertFalse(torch.isclose(actual,separate))
    def test_reject_components(self):
        with self.assertRaises(ValueError):MultiStageDistillation([2,3],[.5,.5],component_weights=[.5,1,1],mode='fused')

if __name__=='__main__':unittest.main()

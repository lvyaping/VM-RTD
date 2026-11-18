import unittest
import torch
from vmdmd.multistage import MultiStageDistillation,validate_direction_weights
from vmdmd.objectives import local_relations,relation_mse

class DirectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.b={s:[torch.randn(1,4,5,6,requires_grad=True) for _ in range(4)] for s in (2,3)}
        self.t={s:torch.randn(1,7,5,6,requires_grad=True) for s in (2,3)}
    def test_equal_compatibility(self):
        old,_=MultiStageDistillation([2,3],[.5,.5])({},self.b,self.t)
        new,w=MultiStageDistillation([2,3],[.5,.5],direction_weights=[[1]*4]*2)({},self.b,self.t)
        torch.testing.assert_close(old,new)
        xs=self.b[2]+self.b[3]
        for x,y in zip(torch.autograd.grad(old,xs),torch.autograd.grad(new,xs)):torch.testing.assert_close(x,y)
        torch.testing.assert_close(w[2],torch.full((4,),.125))
    def test_custom_and_zero(self):
        rows=[[4,3,2,1],[0,0,0,1]]
        actual,w=MultiStageDistillation([2,3],[.5,.5],direction_weights=rows)({},self.b,self.t)
        expected=sum(.5*weight*relation_mse(v,local_relations(self.t[s].detach())) for s,row in zip((2,3),validate_direction_weights([2,3],rows)) for v,weight in zip(self.b[s],row))
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertEqual(self.b[3][0].grad.abs().sum().item(),0)
        self.assertIsNone(self.t[2].grad)
        torch.testing.assert_close(w[2],torch.tensor([.2,.15,.1,.05]))
        torch.testing.assert_close(w[3],torch.tensor([0.,0.,0.,.5]))
    def test_invalid(self):
        for rows in ([[0]*4]*2,[[-1,1,1,1]]*2,[[float('nan'),1,1,1]]*2,[[1]*4]):
            with self.assertRaises(ValueError):validate_direction_weights([2,3],rows)
        with self.assertRaises(ValueError):MultiStageDistillation([2,3],[.5,.5],mode='fused',direction_weights=[[1]*4]*2)

if __name__=='__main__':unittest.main()

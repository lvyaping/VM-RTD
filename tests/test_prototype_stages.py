import unittest
import torch
from vmdmd.multistage import MultiStageDistillation
from vmdmd.prototype_relation import region_prototype_relation_loss,validate_prototype_stages

class StageTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(validate_prototype_stages(),([3],[1.]))
        self.assertEqual(validate_prototype_stages([2,3],[2,2]),([2,3],[.5,.5]))
        for s,w in [([],None),([2,2],None),([1],None),([2,3],[1]),([2],[0]),([2],[float('nan')]),(None,[1])]:
            with self.assertRaises(ValueError):validate_prototype_stages(s,w)
    def test_stages_budget_gradients_and_default(self):
        torch.manual_seed(10)
        for selected,weights in [([2],[1.]),([2,3],[.5,.5]),([3],[1.])]:
            f=[torch.randn(2,3,n,n,requires_grad=True) for n in [16,12,8,4]]
            t=[torch.randn(2,5,n,n,requires_grad=True) for n in [16,12,8,4]]
            b=[[torch.randn_like(x,requires_grad=True) for _ in range(4)] for x in f]
            m=torch.zeros(2,1,32,32);m[0,:,8:24,8:24]=1
            base=MultiStageDistillation([2,3],[.5,.5])
            kd=MultiStageDistillation([2,3],[.5,.5],prototype_ratio=.1,prototype_mode='regions',prototype_stages=selected,prototype_stage_weights=weights)
            raw=base(f,b,t,m)[0];total=kd(f,b,t,m)[0]
            expected=sum(w*region_prototype_relation_loss(f[s],t[s],m,3)[0] for s,w in zip(selected,weights))
            torch.testing.assert_close(.1*total,.1*raw+.01*expected)
            torch.testing.assert_close(kd.prototype_stage_diagnostics,torch.stack([region_prototype_relation_loss(f[s],t[s],m,3)[0].detach() for s in selected]))
            total.backward()
            for s in [2,3]:self.assertEqual(f[s].grad is not None,s in selected)
            self.assertTrue(all(x.grad is None for x in t))
            if selected==[3]:
                default=MultiStageDistillation([2,3],[.5,.5],prototype_ratio=.1,prototype_mode='regions')
                torch.testing.assert_close(default(f,b,t,m)[0],total,rtol=0,atol=0)
if __name__=='__main__':unittest.main()

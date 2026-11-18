import unittest
import torch
from vmdmd.objectives import Distillation,local_relations,relation_mse

class FusedContextTests(unittest.TestCase):
    def test_mean_before_shared_adapter(self):
        torch.set_num_threads(2)
        xs=[torch.randn(1,768,4,4,requires_grad=True) for _ in range(4)]
        t=torch.randn(1,19,4,4,requires_grad=True)
        kd=Distillation('fused',3,'relation_context')
        branch=Distillation('branch',3,'relation_context')
        self.assertEqual(sum(p.numel() for p in kd.parameters()),6912)
        self.assertEqual(sum(p.numel() for p in branch.parameters()),6912)
        args=([],[[],[],[],xs],[None,None,None,t])
        loss,_=kd(*args)
        torch.testing.assert_close(loss,Distillation('fused',3,'relation')(*args)[0])
        with torch.no_grad():kd.adapter.local.weight.fill_(.01)
        inputs=[]
        h=kd.adapter.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach()))
        loss,_=kd(*args);h.remove()
        self.assertEqual(len(inputs),1)
        torch.testing.assert_close(inputs[0],torch.stack(xs).mean(0))
        expected=relation_mse(kd.adapter(torch.stack(xs).mean(0)),local_relations(t.detach()))
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertTrue(all(x.grad.abs().sum()>0 for x in xs))
        self.assertGreater(kd.adapter.local.weight.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()

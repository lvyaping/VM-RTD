import unittest
import torch
from vmdmd.objectives import Distillation

class ContextTests(unittest.TestCase):
    def test_identity_shared_gradient_and_checkpoint(self):
        torch.set_num_threads(2)
        state=torch.get_rng_state().clone()
        kd=Distillation('branch',3,'relation_context')
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        xs=[torch.randn(1,768,8,8,requires_grad=True) for _ in range(4)]
        target=torch.randn(1,24,8,8,requires_grad=True)
        teacher=[None]*3+[target];branches=[[]]*3+[xs]
        expected,_=Distillation('branch',3,'relation')([],branches,teacher)
        calls=[]
        hook=kd.adapter.register_forward_hook(lambda *args:calls.append(1))
        loss,w=kd([],branches,teacher)
        hook.remove()
        torch.testing.assert_close(loss,expected)
        self.assertEqual(len(calls),4)
        self.assertEqual(sum(p.numel() for p in kd.parameters()),6912)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(all(x.grad.abs().sum()>0 for x in xs))
        self.assertGreater(kd.adapter.local.weight.grad.abs().sum().item(),0)
        torch.testing.assert_close(w[3],torch.full((4,),.25))
        torch.optim.SGD(kd.parameters(),lr=.1).step()
        restored=Distillation('branch',3,'relation_context')
        restored.load_state_dict(kd.state_dict())
        torch.testing.assert_close(restored([],branches,teacher)[0],kd([],branches,teacher)[0])

if __name__=='__main__':unittest.main()

import unittest,torch
from vmdmd.context_gate_probe import restore_pairs,GateProbe,masked_mse
from vmdmd.objectives import local_relations
class Tests(unittest.TestCase):
    def test_restore_matches_coordinates(self):
        x=torch.randn(1,5,16,16);rr=local_relations(x)
        flat=torch.cat([v.flatten() for v in rr]);r=flat[:,None].repeat(1,4)
        a,t,m=restore_pairs(r,flat)
        self.assertEqual(int(m.sum()),len(flat))
        from vmdmd.context_gate_probe import OFFSETS
        grid=t.reshape(16,16,8)
        for k,(dy,dx) in enumerate(OFFSETS):
            torch.testing.assert_close(grid[max(0,-dy):min(16,16-dy),max(0,-dx):min(16,16-dx),k],rr[k][0])
        torch.testing.assert_close(a[...,0][m],t[m])
    def test_initial_and_gradients(self):
        r=torch.randn(12,8,4);m=torch.ones(12,8,dtype=torch.bool);m[0,0]=False;t=torch.randn(12,8)
        for kind,n in [('point',2362),('context',2368)]:
            model=GateProbe(kind);pred,w=model(r,m)
            self.assertEqual(sum(p.numel() for p in model.parameters()),n)
            torch.testing.assert_close(pred,r.mean(-1));torch.testing.assert_close(w.sum(-1),torch.ones(12,8))
            loss=masked_mse(pred,t,m);loss.backward()
            self.assertGreater(model.net[-1].weight.grad.abs().sum().item(),0)
            self.assertIsNone(r.grad)
    def test_invalid_layout(self):
        with self.assertRaises(ValueError):restore_pairs(torch.zeros(10,4),torch.zeros(10))
if __name__=='__main__':unittest.main()

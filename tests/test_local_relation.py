import os
import unittest
import torch
from vmdmd.objectives import Distillation, local_relations

os.environ['VMDMD_SCAN_BACKEND']='reference'
torch.set_num_threads(2)


class LocalRelationTests(unittest.TestCase):
    def test_matches_explicit_rectangular_pixel_pairs(self):
        torch.manual_seed(7)
        t=torch.randn(2,5,3,5)
        s=torch.randn(2,7,3,5)
        terms=[]
        for n in range(2):
            for y in range(3):
                for x in range(5):
                    for dy in [-1,0,1]:
                        for dx in [-1,0,1]:
                            if (dy or dx) and 0<=y+dy<3 and 0<=x+dx<5:
                                rt=torch.nn.functional.cosine_similarity(t[n,:,y,x],t[n,:,y+dy,x+dx],dim=0)
                                rs=torch.nn.functional.cosine_similarity(s[n,:,y,x],s[n,:,y+dy,x+dx],dim=0)
                                terms.append((rt-rs).square())
        kd=Distillation('branch',0,'relation')
        actual,_=kd([],[[s]*4],[t])
        torch.testing.assert_close(actual,torch.stack(terms).mean())
        self.assertEqual(sum(r.numel() for r in local_relations(t)),len(terms))

    def test_channel_basis_invariance_and_spatial_mismatch(self):
        torch.manual_seed(10)
        t=torch.randn(1,8,4,5)
        # A channel permutation changes feature vectors but preserves pair affinities.
        s=t[:,torch.randperm(8)]*3
        kd=Distillation('branch',0,'relation')
        self.assertLess(kd([],[[s]*4],[t])[0].item(),1e-10)
        self.assertGreater(kd([],[[s.flip(-1)]*4],[t])[0].item(),.01)
        self.assertEqual(sum(p.numel() for p in kd.parameters()),0)

    def test_fused_and_branch_are_distinct_with_same_views(self):
        torch.manual_seed(11)
        t=torch.randn(1,6,4,5)
        views=[torch.randn_like(t) for _ in range(4)]
        fused=Distillation('fused',0,'relation')
        branch=Distillation('branch',0,'relation')
        mean=torch.stack(views).mean(0)
        f=fused([],[views],[t])[0]
        torch.testing.assert_close(f,branch([],[[mean]*4],[t])[0])
        b=branch([],[views],[t])[0]
        expected=torch.stack([branch([],[[v]*4],[t])[0] for v in views]).mean()
        torch.testing.assert_close(b,expected)
        self.assertGreater(abs((b-f).item()),1e-5)

    def test_all_branch_gradients_and_frozen_teacher(self):
        t=torch.randn(2,5,3,5,requires_grad=True)
        views=[torch.randn(2,7,3,5,requires_grad=True) for _ in range(4)]
        loss,weights=Distillation('branch',0,'relation')([],[views],[t])
        loss.backward()
        self.assertIsNone(t.grad)
        for v in views:
            self.assertTrue(torch.isfinite(v.grad).all())
            self.assertGreater(v.grad.abs().sum().item(),0)
        torch.testing.assert_close(weights[0],torch.full((4,),.25))

    def test_radius_zero_vectors_and_shape_checks(self):
        kd=Distillation('branch',0,'relation',2)
        z=torch.zeros(1,4,3,5,requires_grad=True)
        loss,_=kd([],[[z]*4],[torch.zeros(1,7,3,5)])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaises(ValueError):local_relations(z,3)
        with self.assertRaises(ValueError):local_relations(z,0)
        with self.assertRaises(ValueError):kd([],[[z]*4],[torch.randn(1,7,4,5)])

    def test_real_backbones_kd_only_gradient(self):
        from vmdmd.models.vmamba import VSSM
        from vmdmd.models.swin import SwinTransformer
        from vmdmd.objectives import TrainingSystem
        s=VSSM(num_classes=1,depths=[2,2,6,2],depths_decoder=[2,6,2,2],drop_path_rate=.2)
        t=SwinTransformer(drop_path_rate=.3)
        t.requires_grad_(False)
        t.eval()
        for mode in ['fused','branch']:
            print('Real 32x32 relation-only backward:',mode,flush=True)
            s.zero_grad(set_to_none=True)
            system=TrainingSystem(s,t,Distillation(mode,2,'relation')).train()
            x=torch.rand(1,3,32,32)
            y=torch.zeros(1,1,32,32)
            # alpha=0: no segmentation gradient can hide a disconnected KD path.
            loss,_,_,_=system(x,y,alpha=0.,beta=1.)
            loss.backward()
            grad=s.layers[2].blocks[-1].self_attention.x_proj_weight.grad
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)
            self.assertTrue(all(p.grad is None for p in t.parameters()))


if __name__=='__main__':unittest.main()

import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('metrics',Path(__file__).resolve().parents[1]/'vmdmd/metrics.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class MetricsTests(unittest.TestCase):
    def test_empty_and_exact(self):
        z=torch.zeros(1,1,32,32)
        self.assertEqual(m.boundary_metrics(z,z)['boundary_f1'],1.)
        p=z.clone(); p[:,:,8:20,8:20]=1
        self.assertEqual(m.boundary_metrics(p,z)['boundary_f1'],0.)
        self.assertEqual(m.boundary_metrics(p,p)['boundary_f1'],1.)

    def test_distance_and_border(self):
        p=torch.zeros(32,32); p[8,8]=1
        q=torch.zeros_like(p); q[8,9]=1
        self.assertEqual(m.boundary_metrics(p,q)['boundary_f1'],1.)
        q.zero_(); q[8,10]=1
        self.assertEqual(m.boundary_metrics(p,q)['boundary_f1'],0.)
        p.fill_(1)
        self.assertEqual(m.boundary_metrics(p,p)['boundary_f1'],1.)

    def test_region_counts(self):
        a=m.BinaryMetrics()
        a.update(torch.tensor([1,1,0,0]),torch.tensor([1,0,1,0]))
        r=a.compute()
        self.assertEqual(r['dice'],.5)
        self.assertAlmostEqual(r['iou'],1/3)
        self.assertEqual(r['recall'],r['sensitivity'])


if __name__=='__main__': unittest.main()

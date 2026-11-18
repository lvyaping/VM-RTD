import os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['VMDMD_SCAN_BACKEND']='reference'
import torch
from vmdmd.scan_transfer import coefficient_strength,neighbor_indices,transfer_loss,validate_config,ScanTransferSystem


class ScanTransferTests(unittest.TestCase):
    def test_kernel_matches_impulse_recurrence(self):
        torch.manual_seed(5)
        b,c,n,l=2,3,4,8
        dt=torch.rand(b,c,l,dtype=torch.float64)+.1
        A=-torch.rand(c,n,dtype=torch.float64)
        B=torch.randn(b,n,l,dtype=torch.float64);C=torch.randn_like(B)
        ii=torch.tensor([3,7]);jj=torch.tensor([[0,2],[1,6]])
        actual=coefficient_strength(dt.cumsum(-1),dt,A,B,C,ii,jj)
        for qi,i in enumerate(ii):
            for qj,j in enumerate(jj[qi]):
                state=torch.zeros(b,c,n,dtype=torch.float64)
                for k in range(int(i)+1):
                    state=torch.exp(dt[:,:,k,None]*A)*state
                    if k==j:state=state+dt[:,:,k,None]*B[:,None,:,k]
                expected=(state*C[:,None,:,i]).sum(-1).abs().mean(1)
                torch.testing.assert_close(actual[:,qi,qj],expected,rtol=1e-10,atol=1e-10)

    def test_direction_geometry(self):
        for d in range(4):
            i,j,mask,pi,pj=neighbor_indices(3,4,d,'cpu')
            self.assertTrue((j[mask]<i[:,None].expand_as(j)[mask]).all())
            self.assertTrue((mask.sum(1)>=2).all())
            dy=(pi[:,None]//4-pj//4).abs();dx=(pi[:,None]%4-pj%4).abs()
            self.assertTrue(((dy<=1)&(dx<=1)&((dy+dx)>0))[mask].all())

    def make_capture(self):
        torch.manual_seed(10)
        return dict(shape=(3,4),dts=torch.randn(2,4,3,12,requires_grad=True),
            bias=torch.randn(4,3,requires_grad=True),A=(-torch.rand(4,3,2)).requires_grad_(),
            B=torch.randn(2,4,2,12,requires_grad=True),C=torch.randn(2,4,2,12,requires_grad=True))

    def test_checkpoint_gradient_parity_teacher_detached(self):
        cap=self.make_capture();teacher=torch.randn(2,5,3,4,requires_grad=True)
        params=[cap[k] for k in ('dts','bias','A','B','C')]
        x,_,_=transfer_loss(cap,teacher,chunk=3,recompute=False)
        gx=torch.autograd.grad(x,params)
        y,_,_=transfer_loss(cap,teacher,chunk=3,recompute=True)
        gy=torch.autograd.grad(y,params)
        torch.testing.assert_close(x,y)
        for a,b in zip(gx,gy):
            torch.testing.assert_close(a,b)
            self.assertTrue(torch.isfinite(a).all());self.assertGreater(a.abs().sum().item(),0)
        self.assertIsNone(teacher.grad)

    def test_underflow_is_finite(self):
        cap=self.make_capture();cap['A']=torch.full_like(cap['A'],-1e6,requires_grad=True)
        loss,_,nz=transfer_loss(cap,torch.randn(2,5,3,4),recompute=False)
        self.assertTrue(torch.isfinite(loss));self.assertTrue((nz==1).all())
        loss.backward()
        self.assertTrue(torch.isfinite(cap['A'].grad).all())

    def test_validation(self):
        validate_config({})
        with self.assertRaises(ValueError):validate_config(dict(scan_transfer_weight=.01))
        with self.assertRaises(ValueError):validate_config(dict(scan_transfer_weight=float('nan')))

    def test_real_scan_capture_preserves_output_and_gradients(self):
        from vmdmd.models.vmamba import SS2D
        torch.manual_seed(4)
        m=SS2D(d_model=8,d_state=3).eval()
        x=torch.randn(1,3,4,8)
        before=set(m.state_dict())
        ordinary=m(x)
        m.capture_transfer=True
        captured=m(x)
        torch.testing.assert_close(ordinary,captured,rtol=0,atol=0)
        self.assertEqual(before,set(m.state_dict()))
        loss,_,_=transfer_loss(m.transfer_capture,torch.randn(1,5,3,4))
        loss.backward()
        for p in (m.A_logs,m.x_proj_weight,m.dt_projs_weight,m.dt_projs_bias):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)

    def test_system_zero_equivalence_and_full_backward(self):
        from vmdmd.models.vmamba import VSSM
        from vmdmd.multistage import MultiStageDistillation
        from vmdmd.objectives import TrainingSystem
        class Teacher(torch.nn.Module):
            def forward(self,x):
                return [torch.nn.functional.adaptive_avg_pool2d(x,(64//(4*2**s),)*2) for s in range(4)]
        student=VSSM(num_classes=1,depths=[1]*4,depths_decoder=[1]*4,
                     dims=[8,16,32,64],dims_decoder=[64,32,16,8],d_state=2,drop_path_rate=0.)
        teacher=Teacher();kd=MultiStageDistillation([2,3],[.5,.5])
        normal=TrainingSystem(student,teacher,kd).eval()
        system=ScanTransferSystem(student,teacher,kd,0.).eval()
        x=torch.rand(1,3,64,64);y=(torch.rand(1,1,64,64)>.5).float()
        old=normal(x,y);zero=system(x,y)
        for a,b in zip(old[:3],zero[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(set(normal.state_dict()),set(system.state_dict()))
        system.scan_transfer_weight=.01
        loss,_,_,_=system(x,y)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(student.layers[2].blocks[-1].self_attention.capture_transfer)
        self.assertIsNone(student.layers[2].blocks[-1].self_attention.transfer_capture)
        loss.backward()
        self.assertEqual(system.scan_diagnostics.numel(),11)
        from unittest.mock import patch
        from vmdmd.scan_transfer import calibrate
        class Data(torch.utils.data.Dataset):
            def __len__(self):return 4
            def __getitem__(self,index):return x[0],y[0],str(index)
        saved={k:v.clone() for k,v in system.state_dict().items()}
        rng=torch.get_rng_state().clone()
        with patch('vmdmd.runner.dataset',return_value=Data()):
            report=calibrate(system,dict(seed=42,beta=.1),torch.device('cpu'))
        self.assertGreater(report['recommended_weight'],0)
        self.assertEqual(len(report['batches']),1)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for k,v in system.state_dict().items():torch.testing.assert_close(v,saved[k],rtol=0,atol=0)
        self.assertIsNone(system.calibration_terms)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()

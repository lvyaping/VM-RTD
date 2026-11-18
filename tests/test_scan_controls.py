import os,sys,unittest,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['VMDMD_SCAN_BACKEND']='reference'
import torch
import torch.nn.functional as F
from vmdmd.scan_transfer import *

class Controls(unittest.TestCase):
    def capture(self):
        torch.manual_seed(7)
        return dict(shape=(3,4),dts=torch.randn(2,4,3,12,requires_grad=True),bias=torch.randn(4,3,requires_grad=True),A=(-torch.rand(4,3,2)).requires_grad_(),B=torch.randn(2,4,2,12,requires_grad=True),C=torch.randn(2,4,2,12,requires_grad=True))

    def test_endpoint_identity_path_and_unused_A(self):
        cap=self.capture(); t=torch.randn(2,5,3,4,requires_grad=True)
        endpoint=transfer_loss(cap,t,variant='endpoint',recompute=False)[0]
        zero=dict(cap,A=torch.zeros_like(cap['A']))
        torch.testing.assert_close(endpoint,transfer_loss(zero,t,recompute=False)[0])
        grads=torch.autograd.grad(endpoint,[cap[k] for k in ('dts','bias','A','B','C')],allow_unused=True)
        self.assertIsNone(grads[2])
        for g in [grads[i] for i in (0,1,3,4)]:
            self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum(),0)
        self.assertIsNone(t.grad)
        checkpointed=transfer_loss(cap,t,variant='endpoint',recompute=True)[0]
        torch.testing.assert_close(endpoint,checkpointed)

    def test_output_manual_geometry_and_direction_mapping(self):
        torch.manual_seed(3)
        t=torch.randn(1,3,3,4,requires_grad=True)
        branches=[torch.randn(1,5,3,4,requires_grad=True) for _ in range(4)]
        loss,directions,nz=output_relation_loss(branches,t)
        spatial=torch.arange(12).reshape(3,4)
        expected=[]
        for d,bi in enumerate((0,2,1,3)):
            order=(spatial if d%2==0 else spatial.t()).flatten().tolist()
            if d>=2:order=order[::-1]
            vals=[]
            for k,i in enumerate(order):
                candidates=[j for j in order[:k] if max(abs(i//4-j//4),abs(i%4-j%4))==1]
                if len(candidates)<2:continue
                teacher=F.cosine_similarity(t.flatten(2)[:,:,i,None],t.flatten(2)[:,:,candidates],dim=1).softmax(-1)
                student=F.cosine_similarity(branches[bi].flatten(2)[:,:,i,None],branches[bi].flatten(2)[:,:,candidates],dim=1).log_softmax(-1)
                vals.append((teacher*(teacher.log()-student)).sum())
            expected.append(torch.stack(vals).mean())
        torch.testing.assert_close(directions,torch.stack(expected))
        loss.backward();self.assertIsNone(t.grad)
        for b in branches:self.assertGreater(b.grad.abs().sum(),0)
        identical=output_relation_loss([t.detach()]*4,t)[0]
        self.assertLess(abs(identical.item()),1e-6)

    def test_control_system_backward_and_calibration(self):
        from vmdmd.models.vmamba import VSSM
        from vmdmd.multistage import MultiStageDistillation
        from unittest.mock import patch
        class Teacher(torch.nn.Module):
            def forward(self,x):return [F.adaptive_avg_pool2d(x,(64//(4*2**s),)*2) for s in range(4)]
        class Data(torch.utils.data.Dataset):
            def __len__(self):return 4
            def __getitem__(self,i):return torch.rand(3,64,64),(torch.rand(1,64,64)>.5).float(),str(i)
        for variant in ('output','endpoint'):
            student=VSSM(num_classes=1,depths=[1]*4,depths_decoder=[1]*4,dims=[8,16,32,64],dims_decoder=[64,32,16,8],d_state=2,drop_path_rate=0.)
            system=ScanTransferSystem(student,Teacher(),MultiStageDistillation([2,3],[.5,.5]),.01,variant)
            before={k:v.clone() for k,v in system.state_dict().items()}
            loss=system(torch.rand(1,3,64,64),torch.zeros(1,1,64,64))[0]
            loss.backward();self.assertTrue(torch.isfinite(loss))
            with patch('vmdmd.runner.dataset',return_value=Data()):
                report=calibrate(system,{'seed':42,'beta':.1},torch.device('cpu'))
            self.assertGreater(report['recommended_weight'],0)
            self.assertEqual(report['variant'],variant)
            for k,v in system.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

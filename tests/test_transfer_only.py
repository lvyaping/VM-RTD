import os,sys,unittest,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['VMDMD_SCAN_BACKEND']='reference'
import torch
import torch.nn.functional as F
from vmdmd.scan_transfer import ScanTransferSystem,validate_config
from vmdmd.objectives import segmentation_loss
from vmdmd.models.vmamba import VSSM
from vmdmd.runner import coefficients
class Teacher(torch.nn.Module):
 def forward(self,x):return [F.adaptive_avg_pool2d(x,(64//(4*2**s),)*2) for s in range(4)]
class NoRelation(torch.nn.Module):
 def forward(self,*args):raise AssertionError('Relation KD must not run')
class TransferOnly(unittest.TestCase):
 def test_backward_and_roundtrip(self):
  torch.manual_seed(42)
  student=VSSM(num_classes=1,depths=[1]*4,depths_decoder=[1]*4,dims=[8,16,32,64],dims_decoder=[64,32,16,8],d_state=2,drop_path_rate=0.)
  system=ScanTransferSystem(student,Teacher(),NoRelation(),.015216837498380707,transfer_only=True)
  image=torch.rand(1,3,64,64);target=torch.zeros(1,1,64,64)
  loss,task,kd,weights=system(image,target,beta=0.)
  torch.testing.assert_close(loss.detach(),task+kd)
  self.assertEqual(system.scan_diagnostics[0].item(),0.)
  self.assertEqual(sum(w.abs().sum().item() for w in weights),0.)
  loss.backward()
  scan=student.layers[2].blocks[-1].self_attention
  for p in (scan.A_logs,scan.x_proj_weight,scan.dt_projs_weight):
   self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0.)
  system.eval()
  with torch.no_grad():expected=system.predict(image)
  buf=io.BytesIO();torch.save({'student':student.state_dict()},buf);buf.seek(0)
  student.load_state_dict(torch.load(buf)['student'],strict=True)
  with torch.no_grad():torch.testing.assert_close(system.predict(image),expected)
 def test_config(self):
  cfg=dict(mode='branch',adapter='relation',student_arch='shallow',kd_stages=[2,3],kd_stage_weights=[.5,.5],beta=.1,scan_transfer_weight=.015,scan_transfer_only=True)
  validate_config(cfg);self.assertEqual(coefficients(cfg,1),(1.,0.))
  for updates in ({'scan_transfer_weight':0},{'scan_transfer_variant':'endpoint'}):
   with self.assertRaises(ValueError):validate_config(dict(cfg,**updates))
if __name__=='__main__':unittest.main()

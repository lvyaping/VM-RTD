import os,unittest,tempfile,json
from pathlib import Path
import torch
from torch import nn
from unittest.mock import patch
from vmdmd.objectives import Distillation,Adapter,TrainingSystem
from vmdmd.models.vmamba import SS2D,VSSM
from vmdmd.checkpoints import load_teacher,load_student_pretrained

os.environ['VMDMD_SCAN_BACKEND']='reference'
torch.set_num_threads(2)

class Tests(unittest.TestCase):
    def test_scan_coordinates_rectangular(self):
        import vmdmd.models.vmamba as vm
        old=vm.select_backend
        vm.select_backend=lambda:lambda u,*a,**k:u
        try:
            x=torch.arange(120.).reshape(1,8,3,5)
            for y in SS2D(d_model=4,d_state=2).forward_corev0(x):torch.testing.assert_close(y.reshape_as(x),x)
        finally:vm.select_backend=old

    def test_branch_loss_gradients_and_checkpoint(self):
        for kind in ['project','context']:
            kd=Distillation('branch',0,kind)
            t=torch.randn(2,96,3,5,requires_grad=True)
            b=[[torch.randn_like(t,requires_grad=True) for _ in range(4)]]
            loss,w=kd([],b,[t]);loss.backward()
            self.assertIsNone(t.grad)
            for view in b[0]:self.assertGreater(view.grad.abs().sum().item(),0)
            for p in kd.parameters():self.assertGreater(p.grad.abs().sum().item(),0)
            torch.testing.assert_close(torch.stack(w)[0],torch.full((4,),.25))
            other=Distillation('branch',0,kind);other.load_state_dict(kd.state_dict())
            torch.testing.assert_close(other([],b,[t])[0],loss)

    def test_fusion_is_same_views_and_no_teacher_rotation(self):
        t=torch.randn(1,96,3,5)
        kd=Distillation('branch',0,'project')
        self.assertLess(abs(kd([],[[t]*4],[t])[0].item()),1e-6)
        self.assertGreater(kd([],[[t,t.flip(-1),t,t]],[t])[0].item(),.01)
        views=[torch.randn_like(t) for _ in range(4)]
        fused=Distillation('fused',0,'project')
        expect=(1-torch.nn.functional.cosine_similarity(torch.stack(views).mean(0),t,dim=1)).mean()
        torch.testing.assert_close(fused([],[views],[t])[0],expect)

    def test_context_starts_identical_to_projection(self):
        x=torch.randn(2,96,5,7)
        torch.testing.assert_close(Adapter(96,'project')(x),Adapter(96,'context')(x))

    def test_extraction_does_not_change_student_prediction(self):
        s=VSSM(num_classes=1,depths=[1]*4,depths_decoder=[1]*4,
               dims=[8,16,32,64],dims_decoder=[64,32,16,8],d_state=2).eval()
        x=torch.randn(1,3,32,32)
        with torch.no_grad():
            pred=s(x);extra=s(x,return_stage_features=True,return_directional_features=True)
        torch.testing.assert_close(pred,extra[0])
        self.assertEqual(len(extra[2]),4)

    def test_teacher_strict_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'t.pt';t=nn.Linear(3,2)
            torch.save({'state_dict':{'backbone.'+k:v for k,v in t.state_dict().items()}},path)
            self.assertEqual(load_teacher(nn.Linear(3,2),path)['coverage'],1)
            torch.save({'state_dict':{}},path)
            with self.assertRaises(ValueError):load_teacher(t,path)

    def test_real_vmamba_pretrained_coverage(self):
        model=VSSM(num_classes=1,depths=[2,2,6,2],depths_decoder=[2,6,2,2],drop_path_rate=.2)
        report=load_student_pretrained(model,Path(__file__).resolve().parents[1]/'weights/vmamba_tiny_e292.pth')
        self.assertEqual(report['encoder_coverage'],1.)

    def test_real_backbones_all_kd_modes(self):
        from vmdmd.models.swin import SwinTransformer
        student=VSSM(num_classes=1,depths=[2,2,6,2],depths_decoder=[2,6,2,2],drop_path_rate=.2)
        teacher=SwinTransformer(drop_path_rate=.3)
        x=torch.rand(1,3,32,32);y=(torch.rand(1,1,32,32)>.5).float()
        for mode in ['fused','branch']:
            for adapter in ['project','context']:
                print('Real backbone gradient check:',mode,adapter,flush=True)
                student.zero_grad(set_to_none=True)
                system=TrainingSystem(student,teacher,Distillation(mode,2,adapter)).train()
                self.assertFalse(teacher.training)
                loss,_,_,_=system(x,y,1.,.1);loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertTrue(all(p.grad is not None for p in system.kd.parameters()))
                self.assertGreater(student.layers[2].blocks[-1].self_attention.x_proj_weight.grad.abs().sum().item(),0)

    def test_train_resume_and_separate_test(self):
        from argparse import Namespace
        from PIL import Image
        from vmdmd import runner
        from vmdmd.data import make_manifest
        def tiny(cfg,initialize=True):
            model=VSSM(num_classes=1,depths=[1]*4,depths_decoder=[1]*4,
                       dims=[8,16,32,64],dims_decoder=[64,32,16,8],d_state=2)
            return TrainingSystem(model,None,Distillation('none',2,'project')),{}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';run=root/'run'
            for split in ['train','val','test']:
                for sub in ['image','mask']:(data/split/sub).mkdir(parents=True)
                Image.new('RGB',(32,32),100).save(data/split/'image'/(split+'.png'))
                Image.new('L',(32,32),255).save(data/split/'mask'/(split+'.png'))
            manifest=root/'manifest.csv';make_manifest(data,manifest)
            cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/isic17.json').read_text())
            cfg.update(mode='none',data_root=str(data),manifest=str(manifest),epochs=1,workers=0,batch_size=1,size=32)
            # Store absolute optional paths so loading saved config is stable.
            cfg.update(student_pretrained=str(root/'unused.pth'),teacher_checkpoint=str(root/'unused_teacher.pth'))
            args=Namespace(device='cpu',run_dir=str(run),resume=False)
            with patch.object(runner,'build_system',side_effect=tiny):
                runner.run_train(args,cfg)
                self.assertTrue((run/'best.pt').is_file());self.assertTrue((run/'latest.pt').is_file())
                self.assertFalse((run/'test_metrics.json').exists())
                args.resume=True;runner.run_train(args,cfg)
                with patch('sys.argv',['run.py','evaluate','--config',str(run/'config.json'),'--checkpoint',str(run/'best.pt'),'--device','cpu','--scan-backend','reference','--split','test','--output',str(run/'test_metrics.json')]):runner.main()
                result=json.loads((run/'test_metrics.json').read_text())
                self.assertEqual(result['split'],'test');self.assertEqual(result['per_image'][0]['sample_id'],'test')

if __name__=='__main__':unittest.main()

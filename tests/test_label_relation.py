import unittest
import torch
from torch import nn
from vmdmd.objectives import label_relation_loss,local_relations,Distillation,TrainingSystem

class LabelTests(unittest.TestCase):
    def test_exact_label_geometry(self):
        y=torch.tensor([[[[0.,1.],[0.,1.]]]])
        x=torch.cat([1-y,y],1).requires_grad_()
        self.assertEqual(label_relation_loss(x,y).item(),0)
        self.assertEqual(label_relation_loss(x,y,local_relations(x.detach())).item(),0)
        wrong=torch.ones_like(x,requires_grad=True)
        self.assertGreater(label_relation_loss(wrong,y).item(),0)

    def test_empty_mask_pairs(self):
        x=torch.randn(2,3,2,2,requires_grad=True)
        loss=label_relation_loss(x,torch.full((2,1,8,8),.5))
        loss.backward()
        self.assertEqual(loss.item(),0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_student_only_and_teacher_detached(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__();self.conv=nn.Conv2d(3,4,1);self.head=nn.Conv2d(4,1,1)
            def forward(self,x,return_stage_features=False):
                f=self.conv(x);logits=self.head(f)
                return (logits,[f]*4) if return_stage_features else logits
        model=TrainingSystem(Student(),None,Distillation('native_label',3,'relation'))
        x=torch.rand(2,3,4,4);y=(torch.rand(2,1,4,4)>.5).float()
        loss,*_=model(x,y,alpha=0,beta=1);loss.backward()
        self.assertGreater(model.student.conv.weight.grad.abs().sum().item(),0)
        f=torch.randn(2,4,4,4,requires_grad=True);t=torch.randn(2,7,4,4,requires_grad=True)
        kd=Distillation('native_matched',3,'relation')
        loss,_=kd([f]*4,[],[t]*4,y);loss.backward()
        self.assertIsNone(t.grad)
        self.assertGreater(f.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()

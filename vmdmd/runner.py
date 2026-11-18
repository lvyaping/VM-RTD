import argparse
import copy
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from .checkpoints import sha256, load_teacher, load_student_pretrained
from .data import make_manifest, audit_manifest, SegmentationDataset
from .metrics import BinaryMetrics, boundary_metrics, BOUNDARY_PROTOCOL
from .objectives import Distillation, TrainingSystem, segmentation_loss

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key in ('data_root','manifest','student_pretrained','teacher_checkpoint'):
        if cfg.get(key):
            p = Path(os.path.expandvars(cfg[key])).expanduser()
            cfg[key] = str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    if cfg['size'] < 32 or cfg['size'] % 32:
        raise ValueError('Input size must be a positive multiple of 32')
    if cfg['epochs'] <= 0 or cfg['batch_size'] <= 0:
        raise ValueError('Epochs and global batch size must be positive')
    if not 0 < cfg['threshold'] < 1 or cfg['rotation_degrees'] < 0:
        raise ValueError('Invalid threshold or rotation range')
    if cfg['mode'] not in ('none','fused','branch','native','native_label','native_matched','branch_pair','branch_gap','branch_gap_label') or cfg['adapter'] not in ('project','context','relation','relation_context'):
        raise ValueError('Unsupported mainline experiment')
    if cfg['mode'] in ('native','native_label','native_matched') and cfg['adapter']!='relation':
        raise ValueError('Native output currently supports relation KD only')
    if cfg['stage'] not in range(4) or cfg['beta'] < 0:
        raise ValueError('Invalid stage or KD weight')
    if cfg['mode'] != 'none' and cfg['beta'] == 0:
        raise ValueError('KD experiments require beta > 0')
    if cfg['workers'] < 0 or cfg['lr'] <= 0 or not 0 <= cfg['min_lr'] <= cfg['lr']:
        raise ValueError('Invalid workers or learning rate')
    if cfg['adapter']=='relation_context' and cfg['mode'] not in ('none','branch','fused'):
        raise ValueError('Relation context supports branch or fused KD')
    if cfg['adapter'] in ('relation','relation_context'):
        radius=cfg.get('relation_radius',1)
        if type(radius) is not int or not 1 <= radius < cfg['size']//(4*2**cfg['stage']):
            raise ValueError('Invalid relation radius for selected feature grid')
    if cfg.get('student_arch','deep') not in ('deep','shallow'):
        raise ValueError('Unknown student architecture')
    if 'kd_stages' in cfg:
        from .multistage import validate_stages
        validate_stages(cfg['kd_stages'],cfg['kd_stage_weights'])
        if cfg['mode'] not in ('none','branch','fused') or cfg['adapter']!='relation':
            raise ValueError('Multi-stage supports branch or fused relation')
        if any(not 1 <= cfg.get('relation_radius',1) < cfg['size']//(4*2**s) for s in cfg['kd_stages']):
            raise ValueError('Radius invalid for selected stages')
    if cfg.get('task_gradient',False):
        if cfg['mode'] not in ('none','branch') or cfg['adapter']!='relation' or 'kd_stages' in cfg:
            raise ValueError('Task gradient requires single-stage branch relation')
    if 'collective_aux_weight' in cfg:
        import math
        if cfg.get('collective_mode') not in ('uniform','learned') or cfg['mode']!='branch' or cfg['adapter']!='relation' or cfg.get('kd_stages')!=[2,3]:
            raise ValueError('Collective auxiliary requires stage34 branch relation')
        if not math.isfinite(cfg['collective_aux_weight']) or cfg['collective_aux_weight']<0 or not math.isfinite(cfg['beta']) or cfg['beta']<=0:
            raise ValueError('Invalid collective auxiliary weight/beta')
    from .relation_curriculum import validate_config as validate_curriculum
    validate_curriculum(cfg)
    from .relation_reliability import validate_config as validate_reliability
    validate_reliability(cfg)
    from .prototype_relation import validate_config as validate_prototype
    validate_prototype(cfg)
    from .scan_transfer import validate_config as validate_scan
    validate_scan(cfg)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_system(cfg, initialize=True):
    from .models.vmamba import VSSM
    from .models.swin import SwinTransformer
    arch=cfg.get('student_arch','deep')
    if arch not in ('deep','shallow'):raise ValueError('Unknown student architecture')
    depths,decoder=([2,2,2,2],[2,2,2,1]) if arch=='shallow' else ([2,2,6,2],[2,6,2,2])
    student = VSSM(num_classes=1, depths=depths, depths_decoder=decoder,
                   drop_path_rate=.2)
    reports = {'architecture':dict(name=arch,depths=depths,depths_decoder=decoder)}
    if initialize:
        reports['student'] = load_student_pretrained(student, cfg['student_pretrained'])
    teacher = None
    if cfg['mode'] not in ('none','native_label','branch_gap_label'):
        teacher = SwinTransformer(drop_path_rate=.3)
        if initialize:
            reports['teacher'] = load_teacher(teacher, cfg['teacher_checkpoint'])
    kd = Distillation(cfg['mode'],cfg['stage'],cfg['adapter'],cfg.get('relation_radius',1),cfg.get('relation_purity',.9))
    if 'kd_stages' in cfg and cfg['mode']!='none':
        from .multistage import MultiStageDistillation
        kd=MultiStageDistillation(cfg['kd_stages'],cfg['kd_stage_weights'],cfg.get('relation_radius',1),cfg.get('relation_component_weights'),mode=cfg['mode'],direction_weights=cfg.get('kd_direction_weights'),cross_scale_ratio=cfg.get('cross_scale_weight',0.)/cfg['beta'] if cfg.get('cross_scale_weight',0.) else 0.,collective_mode=cfg.get('collective_mode'),collective_eta=cfg.get('collective_eta',.5),collective_auxiliary_ratio=cfg['collective_aux_weight']/cfg['beta'] if 'collective_aux_weight' in cfg else None,target_curriculum=cfg.get('relation_curriculum')=='target',reliability_mode=cfg.get('relation_reliability'),prototype_ratio=cfg.get('prototype_weight',0.)/cfg['beta'],prototype_mode=cfg.get('prototype_mode','binary'),prototype_boundary_radius=cfg.get('prototype_boundary_radius',3),prototype_stages=cfg.get('prototype_stages'),prototype_stage_weights=cfg.get('prototype_stage_weights'))
        reports['distillation_stages']=dict(stages=kd.stages,weights=kd.weights)
        if cfg.get('relation_curriculum'):
            reports['distillation_stages'].update(relation_curriculum=cfg['relation_curriculum'],curriculum_epochs=cfg.get('curriculum_epochs',10))
        if 'prototype_weight' in cfg:reports['distillation_stages'].update(prototype_stages=[s+1 for s in kd.prototype_stages],prototype_stage_weights=kd.prototype_stage_weights,prototype_weight=cfg['prototype_weight'],prototype_mode=cfg.get('prototype_mode','binary'),prototype_boundary_radius=cfg.get('prototype_boundary_radius',3))
        if kd.reliability_mode:
            reports['distillation_stages']['relation_reliability']=kd.reliability_mode
        if kd.collective is not None:
            reports['distillation_stages'].update(collective_mode=cfg['collective_mode'],collective_eta=cfg.get('collective_eta',.5))
        if 'collective_aux_weight' in cfg:
            reports['distillation_stages'].update(collective_form='additive',collective_aux_weight=cfg['collective_aux_weight'])
        if kd.cross_scale_ratio:
            reports['distillation_stages']['cross_scale_weight']=cfg['cross_scale_weight']
        if kd.direction_weights is not None:
            reports['distillation_stages']['direction_weights']=kd.direction_weights
        if kd.component_weights is not None:
            reports['distillation_stages']['relation_component_weights']=kd.component_weights
    if cfg.get('task_gradient',False) and cfg['mode']!='none':
        from .task_gradient import TaskGradientSystem
        return TaskGradientSystem(student,teacher,kd,cfg.get('gradient_temperature',.5),cfg.get('gradient_uniform_mix',.5)),reports
    if cfg.get('scan_transfer_weight',0)>0 and cfg['mode']!='none':
        from .scan_transfer import ScanTransferSystem,validate_config
        validate_config(cfg)
        reports['scan_transfer']=dict(variant=cfg.get('scan_transfer_variant','full'),stage=3,weight=cfg['scan_transfer_weight'],temperature=1.,radius=1,
                                     raw_direction_order=['row_forward','col_forward','row_backward','col_backward'])
        return ScanTransferSystem(student,teacher,kd,cfg['scan_transfer_weight'],cfg.get('scan_transfer_variant','full'),transfer_only=cfg.get('scan_transfer_only',False)),reports
    return TrainingSystem(student,teacher,kd),reports


def coefficients(cfg, epoch):
    beta=cfg['beta'] if cfg['mode']!='none' and not cfg.get('scan_transfer_only',False) else 0.
    if cfg.get('relation_curriculum')=='weight':
        from .relation_curriculum import progress
        beta*=progress(epoch,cfg.get('curriculum_epochs',10))
    return 1.,beta


def environment():
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'platform': platform.platform(), 'numpy': np.__version__,
            'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'source_hashes': {p.relative_to(ROOT).as_posix(): sha256(p)
                              for p in sorted([ROOT/'run.py', *(ROOT/'vmdmd').rglob('*.py'), *(ROOT/'scripts').glob('*.py'), *(ROOT/'scripts').glob('*.sh')])},
            'scan_backend': os.environ.get('VMDMD_SCAN_BACKEND','cuda')}


def dataset(cfg, split):
    return SegmentationDataset(cfg['data_root'], cfg['manifest'], split, cfg['size'],
                               cfg['seed'], cfg['augmentation'], cfg['rotation_degrees'])


@torch.no_grad()
def evaluate(system, loader, device, threshold, save_dir=None):
    system.eval()
    aggregate, per_image = BinaryMetrics(), []
    losses = []
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for image, target, ids in loader:
        image, target = image.to(device), target.to(device)
        logits = system.predict(image)
        logits = F.interpolate(logits, target.shape[-2:], mode='bilinear', align_corners=False)
        losses.append(float(segmentation_loss(logits, target)))
        pred = logits.sigmoid() >= threshold
        aggregate.update(pred, target)
        single = BinaryMetrics()
        single.update(pred, target)
        per_image.append({'sample_id': ids[0], **single.compute(), **boundary_metrics(pred, target)})
        if save_dir:
            Image.fromarray(pred[0,0].cpu().numpy().astype(np.uint8)*255).save(Path(save_dir)/(ids[0]+'.png'))
    global_metrics = aggregate.compute()
    global_metrics['boundary_f1'] = float(np.mean([x['boundary_f1'] for x in per_image]))
    return {'global': global_metrics, 'boundary_protocol': BOUNDARY_PROTOCOL, 'mean_image_dice': float(np.mean([x['dice'] for x in per_image])),
            'loss': float(np.mean(losses)), 'n_images': len(per_image), 'per_image': per_image,
            'resolution': 'original mask resolution; logits bilinearly resized before thresholding'}


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def run_train(args, cfg):
    rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    local_rank = int(os.environ.get('LOCAL_RANK',0))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if world > 1 and args.device != 'cuda':
        raise ValueError('Distributed training is supported on Linux CUDA/NCCL only')
    device = torch.device(f'cuda:{local_rank}' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl')
    if cfg['batch_size'] % world:
        raise ValueError('Global batch size must be divisible by world size')
    run = Path(args.run_dir).resolve()
    if rank == 0:
        if args.resume:
            if not (run/'latest.pt').exists():
                raise FileNotFoundError('Resume requested without latest.pt')
            saved_cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
            if saved_cfg != cfg:
                raise ValueError('Resume config differs from the recorded run')
        elif run.exists() and any(run.iterdir()):
            raise FileExistsError('Choose a new empty run directory; old experiments are never overwritten')
        run.mkdir(parents=True, exist_ok=True)
        audit = audit_manifest(cfg['data_root'],cfg['manifest'],check_duplicates=cfg.get('check_duplicates',True))
        write_json(run/('resume_data_audit.json' if args.resume else 'data_audit.json'), audit)
        if not args.resume:
            write_json(run/'config.json', cfg)
        write_json(run/'environment.json' if not args.resume else run/'resume_environment.json', environment())
        (run/'pip_freeze.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
        import shutil
        if not args.resume:
            shutil.copytree(ROOT/'vmdmd',run/'source'/'vmdmd',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT/'run.py',run/'source'/'run.py')
            shutil.copytree(ROOT/'scripts',run/'source'/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(cfg['manifest'],run/'manifest.csv')
    if world > 1:
        dist.barrier()
    audit = json.loads((run/('resume_data_audit.json' if args.resume else 'data_audit.json')).read_text(encoding='utf-8'))
    seed_all(cfg['seed'])
    system, load_report = build_system(cfg, initialize=not args.resume)
    system.to(device)
    if rank == 0 and not args.resume:
        write_json(run/'weight_load.json',load_report)
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],
                                  lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['min_lr'])
    start, best, step = 1, -1., 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(run/'latest.pt',map_location='cpu')
        if checkpoint['config'] != cfg or checkpoint['world_size'] != world:
            raise ValueError('Exact resume requires unchanged config and world size')
        if checkpoint['data_fingerprint'] != audit['data_fingerprint']:
            raise ValueError('Dataset changed since checkpoint')
        if checkpoint['source_hashes'] != environment()['source_hashes']:
            raise ValueError('Source changed since checkpoint; start a new experiment instead')
        system.load_state_dict(checkpoint['system'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best, step = checkpoint['epoch']+1, checkpoint['best_dice'], checkpoint['step']
        if rank==0 and (run/'epochs.jsonl').exists():
            lines=(run/'epochs.jsonl').read_text(encoding='utf-8').splitlines()
            kept=[line for line in lines if json.loads(line)['epoch'] < start]
            (run/'epochs.jsonl').write_text('\n'.join(kept)+'\n',encoding='utf-8')
    model = DDP(system,device_ids=[local_rank],find_unused_parameters=True) if world > 1 else system
    train_set, val_set = dataset(cfg,'train'), dataset(cfg,'val')
    sampler = DistributedSampler(train_set,num_replicas=world,rank=rank,shuffle=True,seed=cfg['seed']) if world>1 else None
    generator = torch.Generator()
    loader = DataLoader(train_set,batch_size=cfg['batch_size']//world,shuffle=sampler is None,
                        sampler=sampler,num_workers=cfg['workers'],pin_memory=device.type=='cuda',
                        generator=generator,persistent_workers=False)
    val_loader = DataLoader(val_set,batch_size=1,num_workers=cfg['workers']) if rank==0 else None
    seed_all(cfg['seed']+rank)
    if checkpoint is not None:
        restore_rng(checkpoint['rng'][rank])
        del checkpoint
    source_hashes = environment()['source_hashes']
    for epoch in range(start,cfg['epochs']+1):
        model.train()
        train_set.epoch = epoch
        generator.manual_seed(cfg['seed']+epoch)
        if sampler:
            sampler.set_epoch(epoch)
        alpha,beta = coefficients(cfg,epoch)
        scan_stats=torch.zeros(12,device=device,dtype=torch.float64) if cfg.get('scan_transfer_weight',0)>0 else None
        prototype_stats=torch.zeros((6 if cfg.get('prototype_mode') in ('regions','binary_region_balanced') else 5)+len(system.kd.prototype_stages),device=device,dtype=torch.float64) if cfg.get("prototype_weight",0)>0 else None
        reliability_stats=torch.zeros(13,device=device,dtype=torch.float64) if cfg.get('relation_reliability') else None
        if reliability_stats is not None:
            system.kd.reliability_generator.manual_seed(cfg['seed']+epoch*100000+rank)
        curriculum_progress=None
        if cfg.get('relation_curriculum'):
            from .relation_curriculum import progress
            curriculum_progress=progress(epoch,cfg.get('curriculum_epochs',10))
            system.kd.curriculum_lambda=curriculum_progress
        curriculum_stats=torch.zeros(9,device=device,dtype=torch.float64) if curriculum_progress is not None else None
        accum = torch.zeros(4,device=device,dtype=torch.float64)
        directions = torch.zeros(4,4,device=device,dtype=torch.float64)
        direction_batches = 0
        collective_stats=torch.zeros(15,device=device,dtype=torch.float64)
        cross_stats=torch.zeros(4,device=device,dtype=torch.float64)
        gradient_stats=torch.zeros(9,device=device,dtype=torch.float64)
        gradient_batches=0
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        begin=time.perf_counter()
        for image,target,_ in loader:
            optimizer.zero_grad(set_to_none=True)
            image,target=image.to(device),target.to(device)
            loss,task,kd,weights=model(image,target,alpha,beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
            loss.backward()
            optimizer.step()
            n=image.shape[0]
            accum += torch.tensor([loss.item()*n,task.item()*n,kd.item()*n,n],device=device)
            if weights:
                directions += torch.stack(weights)
                direction_batches += 1
            if scan_stats is not None:
                scan_stats[:11]+=system.scan_diagnostics.to(dtype=torch.float64)*n
                scan_stats[11]+=n
            proto=getattr(system.kd,'prototype_diagnostics',None)
            if prototype_stats is not None and proto is not None:
                prototype_stats[:-1]+=torch.cat([proto,system.kd.prototype_stage_diagnostics]).to(dtype=torch.float64)*n
                prototype_stats[-1]+=n
            reliability=getattr(system.kd,'reliability_diagnostics',None)
            if reliability_stats is not None and reliability is not None:
                reliability_stats[:12]+=reliability.flatten().to(dtype=torch.float64)*n
                reliability_stats[12]+=n
            curriculum=getattr(system.kd,'curriculum_diagnostics',None)
            if curriculum_stats is not None and curriculum is not None:
                curriculum_stats[:8]+=curriculum.flatten().to(dtype=torch.float64)*n
                curriculum_stats[8]+=n
            collective=getattr(system.kd,'collective_diagnostics',None)
            if collective is not None:
                collective_stats[:14]+=collective.flatten().to(dtype=torch.float64)*n
                collective_stats[14]+=n
            cross=getattr(system.kd,'cross_scale_diagnostics',None)
            if cross is not None:
                cross_stats[:3]+=cross.to(dtype=torch.float64)*n
                cross_stats[3]+=n
            diagnostic=getattr(system,'gradient_diagnostics',None)
            if diagnostic is not None:
                gradient_stats+=diagnostic
                gradient_batches+=1
            step+=1
        if device.type=='cuda':
            torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
        if world>1:
            dist.all_reduce(accum)
            if scan_stats is not None:dist.all_reduce(scan_stats)
            if prototype_stats is not None:dist.all_reduce(prototype_stats)
            if reliability_stats is not None:dist.all_reduce(reliability_stats)
            if curriculum_stats is not None:dist.all_reduce(curriculum_stats)
            dist.all_reduce(cross_stats)
            dist.all_reduce(collective_stats)
        scheduler.step()
        if rank==0:
            result=evaluate(system,val_loader,device,cfg['threshold'])
            improved=result['global']['dice']>best
            if improved:
                best=result['global']['dice']
                atomic_save({'student':system.student.state_dict(),'config':cfg,'epoch':epoch,
                             'val_dice':best,'data_fingerprint':audit['data_fingerprint']},run/'best.pt')
                write_json(run/'best_val_metrics.json',result)
            entry={'epoch':epoch,'step':step,'train_loss':float(accum[0]/accum[3]),
                   'task_loss':float(accum[1]/accum[3]),'kd_loss':float(accum[2]/accum[3]),
                   'alpha':alpha,'beta':beta,'next_lr':scheduler.get_last_lr()[0],
                   'val_dice':result['global']['dice'],'best_val_dice':best,
                   'val_iou':result['global']['iou'],
                   'val_precision':result['global']['precision'],
                   'val_recall':result['global']['recall'],
                   'val_boundary_f1':result['global']['boundary_f1'],
                   'rank0_train_seconds':elapsed,'rank0_peak_train_bytes':peak,
                   'rank0_direction_weights':(directions/max(direction_batches,1)).tolist()}
            if scan_stats is not None:
                sv=(scan_stats[:11]/scan_stats[11]).tolist()
                entry.update(scan_transfer_only=cfg.get('scan_transfer_only',False),local_relation_loss=sv[0],scan_transfer_loss=sv[1],weighted_scan_transfer_loss=sv[2],
                    scan_transfer_weight=cfg['scan_transfer_weight'],scan_transfer_stage=3,
                    scan_transfer_variant=cfg.get('scan_transfer_variant','full'),
                    scan_transfer_nearzero_applicable=cfg.get('scan_transfer_variant','full')!='output',
                    scan_transfer_direction_losses=sv[3:7],scan_transfer_nearzero_fractions=sv[7:11],
                    val_mean_image_dice=result['mean_image_dice'])
            if prototype_stats is not None and prototype_stats[-1]>0:
                pv=(prototype_stats[:-1]/prototype_stats[-1]).tolist()
                entry.update(local_relation_loss=pv[0],prototype_loss=pv[1],prototype_weight=cfg['prototype_weight'],
                    weighted_prototype_loss=cfg['prototype_weight']*pv[1],
                    prototype_stages=[s+1 for s in system.kd.prototype_stages],
                    prototype_stage_weights=system.kd.prototype_stage_weights,
                    prototype_mode=cfg.get('prototype_mode','binary'),
                    val_mean_image_dice=result['mean_image_dice'])
                entry['prototype_definition']='regions' if cfg.get('prototype_mode')=='regions' else 'foreground_background'
                entry['prototype_spatial_reduction']='pixel_mean' if cfg.get('prototype_mode','binary')=='binary' else 'equal_valid_regions'
                stage_values=pv[-len(system.kd.prototype_stages):]
                entry['prototype_stage_losses']={str(s+1):v for s,v in zip(system.kd.prototype_stages,stage_values)}
                entry['weighted_prototype_stage_losses']={str(s+1):cfg['prototype_weight']*w*v for s,w,v in zip(system.kd.prototype_stages,system.kd.prototype_stage_weights,stage_values)}
                if len(system.kd.prototype_stages)==1:entry['prototype_stage']=system.kd.prototype_stages[0]+1
                if cfg.get('prototype_mode') in ('regions','binary_region_balanced'):
                    entry.update(prototype_boundary_radius=cfg.get('prototype_boundary_radius',3),
                        prototype_region_present_fractions=dict(zip(['interior','boundary','far_background'],pv[2:])))
                else:
                    entry.update(prototype_foreground_present_fraction=pv[2],prototype_background_present_fraction=pv[3])
            if reliability_stats is not None and reliability_stats[12]>0:
                vals=(reliability_stats[:12]/reliability_stats[12]).reshape(2,6).tolist()
                entry.update(relation_reliability=cfg['relation_reliability'],val_mean_image_dice=result['mean_image_dice'],
                    reliability_stage_diagnostics={str(s):dict(mean_disagreement=v[0],batch_disagreement_std=v[1],batch_weight_std=v[2],weighted_disagreement=v[3],unweighted_mse=v[4],weighted_mse=v[5]) for s,v in zip([3,4],vals)})
            if curriculum_progress is not None:
                entry.update(relation_curriculum=cfg['relation_curriculum'],curriculum_progress=curriculum_progress,
                             target_lambda=curriculum_progress if cfg['relation_curriculum']=='target' else 1.,
                             val_mean_image_dice=result['mean_image_dice'])
                if curriculum_stats[8]>0:
                    vals=(curriculum_stats[:8]/curriculum_stats[8]).reshape(2,4).tolist()
                    entry['curriculum_stage_diagnostics']={str(s):dict(used_mse=v[0],full_teacher_mse=v[1],outside_fraction=v[2],target_shift_mse=v[3]) for s,v in zip([3,4],vals)}
            if collective_stats[14]>0:
                vals=(collective_stats[:14]/collective_stats[14]).reshape(2,7).tolist()
                entry.update(collective_mode=cfg['collective_mode'],collective_eta=cfg.get('collective_eta',.5),
                             collective_stage_diagnostics={str(s+1):dict(branch_loss=v[0],mixture_loss=v[1],normalized_entropy=v[2],mean_gate_weights=v[3:]) for s,v in zip([2,3],vals)})
                auxiliary='collective_aux_weight' in cfg
                bw=beta if auxiliary else beta*(1-cfg.get('collective_eta',.5))
                mw=beta*cfg['collective_aux_weight']/cfg['beta'] if auxiliary else beta*cfg.get('collective_eta',.5)
                entry.update(collective_form='additive' if auxiliary else 'replacement',
                             effective_branch_coefficient=bw,effective_mixture_coefficient=mw,
                             weighted_branch_loss=bw*sum(w*v[0] for w,v in zip(cfg['kd_stage_weights'],vals)),
                             weighted_mixture_loss=mw*sum(w*v[1] for w,v in zip(cfg['kd_stage_weights'],vals)))
            if cross_stats[3]>0:
                vals=(cross_stats[:3]/cross_stats[3]).tolist()
                entry.update(relation_stage3_loss=vals[0],relation_stage4_loss=vals[1],cross_scale_relation_loss=vals[2],cross_scale_weight=cfg['cross_scale_weight'])
            if gradient_batches:
                g=(gradient_stats/gradient_batches).tolist()
                entry.update(task_gradient_cosines=g[:4],task_gradient_weights=g[4:8],task_gradient_entropy=g[8],
                             gradient_temperature=system.temperature,gradient_uniform_mix=system.uniform_mix)

            with (run/'epochs.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(entry)+'\n')
            print(json.dumps(entry),flush=True)
        states=[None]*world
        if world>1:
            dist.all_gather_object(states,rng_state())
        else:
            states[0]=rng_state()
        if rank==0:
            atomic_save({'system':system.state_dict(),'optimizer':optimizer.state_dict(),
                         'scheduler':scheduler.state_dict(),'epoch':epoch,'best_dice':best,'step':step,
                         'rng':states,'config':cfg,'world_size':world,
                         'data_fingerprint':audit['data_fingerprint'],'source_hashes':source_hashes},run/'latest.pt')
        if world>1:
            dist.barrier()
    if world>1:
        dist.destroy_process_group()
    if rank==0:
        print('Training finished. Test was NOT evaluated; run evaluate explicitly with best.pt.')


def main():
    parser=argparse.ArgumentParser(description='Clean binary VM-DMD experiments')
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--data-root',required=True)
    prep.add_argument('--output',required=True)
    audit=sub.add_parser('audit')
    audit.add_argument('--config',required=True)
    audit.add_argument('--output',required=True)
    for command in ('train','preflight','evaluate','benchmark'):
        p=sub.add_parser(command)
        p.add_argument('--config',required=True)
        p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        p.add_argument('--scan-backend',choices=['cuda','reference'],default='cuda')
        if command=='train':
            p.add_argument('--run-dir',required=True)
            p.add_argument('--resume',action='store_true')
        else:
            p.add_argument('--output',required=True)
        if command in ('evaluate','benchmark'):
            p.add_argument('--checkpoint',required=True)
        if command=='evaluate':
            p.add_argument('--split',choices=['val','test'],default='test')
            p.add_argument('--save-predictions')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.command=='prepare':
        make_manifest(args.data_root,args.output)
        print('Manifest created; run audit before training.')
        return
    cfg=load_config(args.config)
    if args.command=='audit':
        report=audit_manifest(cfg['data_root'],cfg['manifest'],strict=False,check_duplicates=cfg.get('check_duplicates',True))
        write_json(args.output,report)
        print(json.dumps(report,ensure_ascii=False))
        if report['errors']:
            raise SystemExit(2)
        return
    os.environ['VMDMD_SCAN_BACKEND']=args.scan_backend
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    seed_all(cfg['seed'])
    if args.command=='train':
        run_train(args,cfg)
        return
    device=torch.device(args.device)
    if args.command=='preflight':
        system,reports=build_system(cfg)
        system.to(device).train()
        if cfg.get('scan_transfer_only',False):
            reports['scan_transfer_fixed_weight']=cfg['scan_transfer_weight']
        if cfg.get('scan_transfer_weight',0)>0 and not cfg.get('scan_transfer_only',False):
            from .scan_transfer import calibrate
            reports['scan_transfer_calibration']=calibrate(system,cfg,device)
            print('SCAN_TRANSFER_WEIGHT='+str(reports['scan_transfer_calibration']['recommended_weight']),flush=True)
        if cfg.get('relation_curriculum'):
            reports['curriculum_preflight']='target lambda=0; weight control checked with full beta (training epoch1 uses beta=0)'
        # A small real-backbone gradient smoke check, not a segmentation experiment.
        x=torch.rand(1,3,64,64,device=device)
        y=(torch.rand(1,1,64,64,device=device)>.5).float()
        loss,_,_,_=system(x,y,1.,.1 if cfg['mode']!='none' else 0.)
        loss.backward()
        learned_grads=[p.grad for p in system.kd.parameters() if p.requires_grad]
        if learned_grads and not any(g is not None and g.abs().sum()>0 for g in learned_grads):
            raise AssertionError('Adapter receives no gradients')
        if system.teacher and any(p.grad is not None for p in system.teacher.parameters()):
            raise AssertionError('Teacher unexpectedly receives gradients')
        reports.update(environment=environment(),loss=float(loss),
                       student_parameters=sum(p.numel() for p in system.student.parameters()),
                       kd_parameters=sum(p.numel() for p in system.kd.parameters()),
                       status='64x64 real-backbone forward/backward passed',
                       limitation='Not a training run; CUDA custom kernel verified only when backend=cuda')
        write_json(args.output,reports)
        print(json.dumps({k:v for k,v in reports.items() if k not in ('environment','student','teacher')},ensure_ascii=False))
        return
    ckpt=torch.load(args.checkpoint,map_location='cpu')
    if 'student' not in ckpt:
        raise ValueError('Evaluation requires new-format best.pt, not an old or latest checkpoint')
    relocation_keys={'data_root','manifest','student_pretrained','teacher_checkpoint'}
    if {k:v for k,v in ckpt['config'].items() if k not in relocation_keys}!={k:v for k,v in cfg.items() if k not in relocation_keys}:
        raise ValueError('Evaluate with the saved run config.json; do not change preprocessing or threshold')
    eval_cfg=copy.deepcopy(cfg)
    eval_cfg.update(mode='none')
    system,_=build_system(eval_cfg,initialize=False)
    system.student.load_state_dict(ckpt['student'],strict=True)
    system.to(device).eval()
    if args.command=='evaluate':
        audit=audit_manifest(cfg['data_root'],cfg['manifest'],check_duplicates=cfg.get('check_duplicates',True))
        if audit['data_fingerprint']!=ckpt['data_fingerprint']:
            raise ValueError('Dataset differs from training checkpoint')
        result=evaluate(system,DataLoader(dataset(cfg,args.split),batch_size=1),device,cfg['threshold'],args.save_predictions)
        result.update(split=args.split,checkpoint_sha256=sha256(args.checkpoint),epoch=ckpt['epoch'],data_audit=audit)
        write_json(args.output,result)
        print(json.dumps(result['global']))
    else:
        if args.scan_backend!='cuda' or device.type!='cuda':
            raise ValueError('Publishable latency benchmark requires CUDA custom scan backend')
        x=torch.rand(1,3,cfg['size'],cfg['size'],device=device)
        with torch.no_grad():
            for _ in range(20):
                system.predict(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations=[]
            for _ in range(100):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                system.predict(x)
                end.record()
                torch.cuda.synchronize()
                durations.append(start.elapsed_time(end))
        write_json(args.output,{'environment':environment(),'batch_size':1,'size':cfg['size'],
                   'precision':'float32','latency_median_ms':float(np.median(durations)),
                   'latency_p95_ms':float(np.percentile(durations,95)),
                   'throughput_images_per_second':1000/float(np.mean(durations)),
                   'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'student_parameters':sum(p.numel() for p in system.student.parameters()),
                   'scope':'student forward plus normalization; excludes disk IO and output resize',
                   'flops':'not reported: selective scan needs an explicit complete counting convention'})


if __name__=='__main__':
    main()

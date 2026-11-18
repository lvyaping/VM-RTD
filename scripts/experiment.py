"""One explicit parameter entry for feature and local-relation comparisons."""
import argparse,json,os,re,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vmdmd.data import make_manifest

EXPERIMENTS={'branch_pair':('branch_pair','relation'),'branch_gap':('branch_gap','relation'),'branch_gap_label':('branch_gap_label','relation'),'branch_task_gradient':('branch','relation'),'baseline':('none','project'),'fused_project':('fused','project'),
             'branch_project':('branch','project'),'fused_context':('fused','context'),
             'branch_context':('branch','context'),
             'fused_relation':('fused','relation'),'branch_relation':('branch','relation'),'branch_relation_context':('branch','relation_context'),'fused_relation_context':('fused','relation_context'),'native_relation':('native','relation'),'label_relation':('native_label','relation'),'teacher_matched_relation':('native_matched','relation')}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','preflight','test'])
    p.add_argument('--dataset',choices=['isic17','isic18','bus','busi'],required=True)
    p.add_argument('--experiment',choices=EXPERIMENTS,default='branch_project')
    p.add_argument('--data-root',required=True);p.add_argument('--teacher',required=True)
    p.add_argument('--student-pretrained',default=str(ROOT/'weights/vmamba_tiny_e292.pth'))
    p.add_argument('--run',default='mainline_v1');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--gpus',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--beta',type=float,default=.1)
    p.add_argument('--stage',type=int,choices=range(4),default=2)
    p.add_argument('--relation-radius',type=int,default=1)
    p.add_argument('--relation-purity',type=float,default=.9)
    p.add_argument('--student-arch',choices=['deep','shallow'],default='deep')
    p.add_argument('--kd-stages',type=int,nargs='+')
    p.add_argument('--kd-stage-weights',type=float,nargs='+')
    p.add_argument('--kd-direction-weights',type=float,nargs='+')
    p.add_argument('--scan-transfer-only',action='store_true',help='Disable output relation loss; retain fixed full transfer supervision')
    p.add_argument('--scan-transfer-weight',type=float)
    p.add_argument('--scan-transfer-variant',choices=['full','output','endpoint'])
    p.add_argument('--prototype-weight',type=float)
    p.add_argument('--prototype-stages',type=int,nargs='+')
    p.add_argument('--prototype-stage-weights',type=float,nargs='+')
    p.add_argument('--prototype-mode',choices=['binary','regions','binary_region_balanced'])
    p.add_argument('--prototype-boundary-radius',type=int)
    p.add_argument('--relation-reliability',choices=['aligned','shuffled'])
    p.add_argument('--relation-curriculum',choices=['target','weight'])
    p.add_argument('--curriculum-epochs',type=int,default=10)
    p.add_argument('--collective-mode',choices=['uniform','learned'])
    p.add_argument('--collective-eta',type=float,default=.5)
    p.add_argument('--collective-aux-weight',type=float)
    p.add_argument('--cross-scale-weight',type=float,default=None)
    p.add_argument('--relation-component-weights',type=float,nargs=3,metavar=('MEAN','STD','SHAPE'))
    p.add_argument('--gradient-temperature',type=float,default=.5)
    p.add_argument('--gradient-uniform-mix',type=float,default=.5)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run):p.error('Run name must contain letters/numbers/_/- only')
    if a.gpus<1 or a.batch_size<1 or a.batch_size%a.gpus:p.error('Global batch must divide evenly across GPUs')
    if a.resume and a.action!='train':p.error('Resume only applies to train')
    name='{}_{}_seed{}'.format(a.run,a.experiment,a.seed)
    run=ROOT/'runs'/a.dataset/name
    if a.action=='test':
        command=[sys.executable,str(ROOT/'run.py'),'evaluate','--config',str(run/'config.json'),'--checkpoint',str(run/'best.pt'),'--split','test','--output',str(run/'test_metrics.json')]
    else:
        cfg=json.loads((ROOT/'configs'/('{}.json'.format(a.dataset))).read_text())
        mode,adapter=EXPERIMENTS[a.experiment]
        # Unique manifest per run; no shared mutable generated configs between Slurm jobs.
        manifest=ROOT/'data'/a.dataset/(name+'.csv')
        cfg.update(data_root=str(Path(a.data_root).resolve()),manifest=str(manifest),
                   teacher_checkpoint=str(Path(a.teacher).resolve()),student_pretrained=str(Path(a.student_pretrained).resolve()),
                   mode=mode,adapter=adapter,stage=a.stage,beta=a.beta,epochs=a.epochs,
                   seed=a.seed,batch_size=a.batch_size,workers=a.workers,lr=a.lr)
        if a.kd_stages is not None:
            if a.experiment not in ('branch_relation','fused_relation'):p.error('Multi-stage requires branch_relation or fused_relation')
            from vmdmd.multistage import validate_stages
            weights=a.kd_stage_weights or [1.]*len(a.kd_stages)
            try: weights=validate_stages(a.kd_stages,weights)
            except ValueError as exc:p.error(str(exc))
            if any(not 1 <= a.relation_radius < cfg['size']//(4*2**s) for s in a.kd_stages):p.error('Invalid multi-stage radius')
            cfg.update(kd_stages=a.kd_stages,kd_stage_weights=weights)
        elif a.kd_stage_weights is not None:p.error('Stage weights require stages')
        if a.collective_aux_weight is not None and a.collective_mode is None:
            p.error('Auxiliary weight requires collective mode')
        if a.collective_mode is not None:
            import math
            if a.kd_stages!=[2,3] or a.experiment!='branch_relation' or a.cross_scale_weight is not None or a.kd_direction_weights is not None or a.relation_component_weights is not None:
                p.error('Collective requires stage34 branch relation without other extensions')
            if not math.isfinite(a.collective_eta) or not 0 <= a.collective_eta <= 1:
                p.error('Collective eta must be in [0,1]')
            cfg.update(collective_mode=a.collective_mode,collective_eta=a.collective_eta)
            if a.collective_aux_weight is not None:
                if not math.isfinite(a.collective_aux_weight) or a.collective_aux_weight<0 or not math.isfinite(a.beta) or a.beta<=0:
                    p.error('Auxiliary weight must be finite/nonnegative and beta positive')
                cfg['collective_aux_weight']=a.collective_aux_weight
        if a.cross_scale_weight is not None:
            import math
            if not math.isfinite(a.cross_scale_weight) or a.cross_scale_weight<0 or not math.isfinite(a.beta) or a.beta<=0:
                p.error('Cross-scale weight must be finite and nonnegative; beta must be positive')
            if a.kd_stages!=[2,3] or a.experiment!='branch_relation' or a.kd_direction_weights is not None or a.relation_component_weights is not None:
                p.error('Cross-scale requires original stage34 branch relation without extra weighting')
            cfg['cross_scale_weight']=a.cross_scale_weight
        if a.kd_direction_weights is not None:
            if a.kd_stages is None or a.experiment!='branch_relation' or a.relation_component_weights is not None:
                p.error('Custom directions require multi-stage branch_relation without component weights')
            if len(a.kd_direction_weights)!=4*len(a.kd_stages):p.error('Need four weights per selected stage')
            from vmdmd.multistage import validate_direction_weights
            raw=[a.kd_direction_weights[i:i+4] for i in range(0,len(a.kd_direction_weights),4)]
            try: cfg['kd_direction_weights']=validate_direction_weights(a.kd_stages,raw)
            except ValueError as exc:p.error(str(exc))
            cfg['kd_direction_weights_raw']=raw
        if a.relation_component_weights is not None:
            if a.kd_stages is None or a.experiment!='branch_relation':
                p.error('Component weights require multi-stage branch_relation')
            from vmdmd.relation_components import validate_component_weights
            try: cfg['relation_component_weights']=validate_component_weights(a.relation_component_weights)
            except ValueError as exc:p.error(str(exc))
        if a.experiment=='branch_task_gradient':
            import math
            if not math.isfinite(a.gradient_temperature) or a.gradient_temperature<=0 or not 0<=a.gradient_uniform_mix<=1:
                p.error('Invalid task-gradient temperature or uniform mix')
            cfg.update(task_gradient=True,gradient_temperature=a.gradient_temperature,gradient_uniform_mix=a.gradient_uniform_mix)
        if a.student_arch=='shallow':cfg['student_arch']='shallow'
        if mode in ('native_label','native_matched','branch_pair','branch_gap','branch_gap_label'):
            if not .5 < a.relation_purity <= 1:p.error('Purity must be in (.5,1]')
            cfg['relation_purity']=a.relation_purity
        if adapter in ('relation','relation_context'):
            if not 1 <= a.relation_radius < cfg['size']//(4*2**a.stage):p.error('Relation radius must be smaller than selected feature grid')
            cfg['relation_radius']=a.relation_radius
        if a.prototype_stages is not None:cfg['prototype_stages']=a.prototype_stages
        if a.prototype_stage_weights is not None:cfg['prototype_stage_weights']=a.prototype_stage_weights
        if a.prototype_mode is not None:cfg['prototype_mode']=a.prototype_mode
        if a.prototype_boundary_radius is not None:cfg['prototype_boundary_radius']=a.prototype_boundary_radius
        if a.prototype_weight is not None:
            cfg["prototype_weight"]=a.prototype_weight
        if a.relation_curriculum is not None:
            from vmdmd.relation_curriculum import validate_config
            cfg.update(relation_curriculum=a.relation_curriculum,curriculum_epochs=a.curriculum_epochs)
            try:validate_config(cfg)
            except ValueError as exc:p.error(str(exc))
        if a.relation_reliability is not None:
            from vmdmd.relation_reliability import validate_config
            cfg['relation_reliability']=a.relation_reliability
            try:validate_config(cfg)
            except ValueError as exc:p.error(str(exc))
        if a.scan_transfer_only:cfg['scan_transfer_only']=True
        if a.scan_transfer_weight is not None:cfg['scan_transfer_weight']=a.scan_transfer_weight
        if a.scan_transfer_variant is not None:cfg['scan_transfer_variant']=a.scan_transfer_variant
        from vmdmd.scan_transfer import validate_config as validate_scan
        try:validate_scan(cfg)
        except ValueError as exc:p.error(str(exc))
        from vmdmd.prototype_relation import validate_config as validate_prototype
        try:validate_prototype(cfg)
        except ValueError as exc:p.error(str(exc))
        if a.action=='train' and not a.resume and run.exists() and any(run.iterdir()):p.error('Run exists; choose a new RUN or exact RESUME=true')
        config=ROOT/'configs/generated'/a.dataset/(name+'.json')
        if config.exists() and json.loads(config.read_text())!=cfg:p.error('Inputs changed; choose a new RUN')
        for path in [Path(cfg['student_pretrained'])]+([Path(cfg['teacher_checkpoint'])] if mode not in ('none','native_label','branch_gap_label') else []):
            if not path.is_file():p.error('Missing weights: '+str(path))
        if not manifest.exists():make_manifest(a.data_root,manifest)
        config.parent.mkdir(parents=True,exist_ok=True)
        config.write_text(json.dumps(cfg,indent=2)+'\n')
        command=[sys.executable]
        if a.action=='train' and a.gpus>1:
            command+=['-m','torch.distributed.run','--nproc_per_node',str(a.gpus),'--master_port',str(20000+int(os.environ.get('SLURM_JOB_ID','9501'))%20000)]
        command += [str(ROOT/'run.py'),a.action,'--config',str(config)]
        if a.action=='train':
            command+=['--run-dir',str(run)]
            if a.resume:command+=['--resume']
        else:command+=['--output',str(ROOT/'reports'/(name+'_preflight.json'))]
    print(json.dumps(dict(dataset=a.dataset,experiment=a.experiment,run_dir=str(run),command=command)),flush=True)
    subprocess.run(command,check=True,cwd=ROOT)

if __name__=='__main__':main()

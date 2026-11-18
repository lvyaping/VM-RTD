"""Figure 4: original-resolution paired A/D segmentation and whole-split Dice deltas."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def crop_box(gt,a,d):
    """Fixed quarter-size crop; most A/D disagreement, then GT boundary fallback."""
    h,w=gt.shape
    focus=a!=d
    if not focus.any():
        pad=np.pad(gt,1,constant_values=False)
        interior=np.logical_and.reduce([pad[y:y+h,x:x+w] for y in range(3) for x in range(3)])
        focus=gt&~interior
    if focus.any():
        heat=F.adaptive_avg_pool2d(torch.from_numpy(focus.astype(np.float32))[None,None],(32,32))
        scores=F.avg_pool2d(heat,8,stride=1)[0,0].numpy()
        y,x=np.unravel_index(scores.argmax(),scores.shape)
    else:y=x=12
    x0=int(x*w/32);y0=int(y*h/32)
    return [x0,y0,min(w,max(x0+1,int((x+8)*w/32))),min(h,max(y0+1,int((y+8)*h/32)))]


def plot_cases(cases,output,formats=('png','pdf','svg')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2*len(cases),4,figsize=(12,5.3*len(cases)),squeeze=False,constrained_layout=True)
    for case_index,case in enumerate(cases):
        rgb,gt,a,d=case['rgb'],case['gt'],case['a'],case['d'];row=case['metrics']
        x0,y0,x1,y1=case['crop']
        titles=['Image / '+case.get('category','Case'),'Ground truth',
                'A: relation KD | Dice {:.2f}%'.format(row['dice_A']*100),
                'D: full method | Dice {:.2f}%'.format(row['dice_D']*100)]
        for col,(array,title) in enumerate(zip([rgb,gt,a,d],titles)):
            ax=axes[case_index*2,col];ax.imshow(array,cmap=None if col==0 else 'gray',vmin=0,vmax=None if col==0 else 1)
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#f09b26',linewidth=1.3))
            ax.add_patch(Rectangle((-.5,-.5),array.shape[1],array.shape[0],fill=False,edgecolor='#888888',linewidth=.5))
            ax.set_title(title,fontsize=10);ax.axis('off')
            ax=axes[case_index*2+1,col];ax.imshow(array[y0:y1,x0:x1],cmap=None if col==0 else 'gray',vmin=0,vmax=None if col==0 else 1)
            ax.add_patch(Rectangle((-.5,-.5),x1-x0,y1-y0,fill=False,edgecolor='#888888',linewidth=.5))
            ax.set_title('Same ROI: '+['image','GT','A','D'][col],fontsize=10);ax.axis('off')
        axes[case_index*2+1,0].text(0,-.12,'{} | D - A = {:+.2f} pp'.format(row['sample_id'],row['delta_dice_pp']),transform=axes[case_index*2+1,0].transAxes,fontsize=10)
    fig.suptitle('Paired segmentation results | white: foreground | orange: shared zoom region',fontsize=12)
    for ext in formats:fig.savefig(str(output)+'.'+ext,dpi=250)
    plt.close(fig)


def plot_deltas(rows,output,epsilon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=np.array([r['delta_dice_pp'] for r in rows])
    ordered=np.sort(values)
    colors=np.where(ordered>epsilon,'#138b82',np.where(ordered < -epsilon,'#c65b4a','#999999'))
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    axes[0].scatter(np.arange(1,len(values)+1),ordered,c=colors,s=9)
    axes[0].axhline(0,color='black',lw=.8);axes[0].set_xlabel('Images ranked by paired Dice difference')
    axes[0].set_ylabel('Dice(D) - Dice(A) (percentage points)')
    axes[0].set_title('All {} evaluated images'.format(len(values)))
    axes[1].hist(values,bins=min(40,max(5,int(np.sqrt(len(values))))),color='#328f9b',edgecolor='white')
    axes[1].axvline(0,color='black',lw=.8)
    axes[1].axvline(values.mean(),color='#d07817',ls='--',label='Mean = {:+.3f} pp'.format(values.mean()))
    axes[1].set_xlabel('Dice(D) - Dice(A) (percentage points)');axes[1].set_ylabel('Number of images');axes[1].legend()
    fig.suptitle('Paired per-image differences (not a significance test)',fontsize=12)
    for ext in ('png','pdf','svg'):fig.savefig(str(output)+'.'+ext,dpi=300)
    plt.close(fig)


def select_cases(rows,epsilon):
    groups={'Improvement':sorted([r for r in rows if r['delta_dice_pp']>epsilon],key=lambda r:(r['delta_dice_pp'],r['sample_id'])),
            'Similar':sorted([r for r in rows if abs(r['delta_dice_pp'])<=epsilon],key=lambda r:(abs(r['delta_dice_pp']),r['sample_id'])),
            'Regression':sorted([r for r in rows if r['delta_dice_pp'] < -epsilon],key=lambda r:(r['delta_dice_pp'],r['sample_id']))}
    chosen=[]
    for category,items in groups.items():
        if items:chosen.append((category,items[0] if category=='Similar' else items[len(items)//2]))
    return groups,chosen


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--dataset',choices=['isic17','isic18','bus','busi'],default='isic18')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--run-a',default='');parser.add_argument('--run-d',default='')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--split',choices=['test','val'],default='test')
    parser.add_argument('--output',required=True)
    parser.add_argument('--near-zero-pp',type=float,default=.1,help='Display grouping tolerance in percentage points; not a significance threshold')
    parser.add_argument('--render',choices=['all','selected'],default='all')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if not np.isfinite(args.near_zero_pp) or args.near_zero_pp<0:parser.error('near-zero-pp must be finite and nonnegative')
    root=Path(args.project_root).resolve();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'scripts'))
    # Reuse strict A/D identity checks and legacy run-name resolution from Figure 3.
    from visualize_transfer import resolve_run,check_config
    os.environ['VMDMD_SCAN_BACKEND']='cuda' if args.device=='cuda' else 'reference'
    from vmdmd.models.vmamba import VSSM
    from vmdmd.checkpoints import sha256
    from vmdmd.data import SegmentationDataset,read_manifest
    from vmdmd.metrics import BinaryMetrics
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={};configs={};provenance={};manifests={}
    for role,explicit in [('A',args.run_a),('D',args.run_d)]:
        run=resolve_run(root,args.dataset,args.seed,explicit,role)
        ck=torch.load(run/'best.pt',map_location='cpu');cfg=ck['config'];check_config(cfg,role)
        if cfg['seed']!=args.seed:raise ValueError('Wrong seed: '+role)
        load=json.loads((run/'weight_load.json').read_text(encoding='utf-8-sig'))
        model=VSSM(num_classes=1,depths=[2,2,2,2],depths_decoder=[2,2,2,1],drop_path_rate=.2)
        model.load_state_dict(ck['student'],strict=True);model.to(args.device);model.eval()
        models[role]=model;configs[role]=cfg;manifests[role]=read_manifest(run/'manifest.csv')
        provenance[role]=dict(run=str(run),checkpoint_sha256=sha256(run/'best.pt'),epoch=ck['epoch'],
                              data_fingerprint=ck.get('data_fingerprint'),pretrained_sha256=load['student']['sha256'],teacher_sha256=load['teacher']['sha256'])
        del ck
    for key in ('size','threshold','seed','epochs','batch_size','lr','min_lr','weight_decay','augmentation','rotation_degrees','student_arch'):
        if configs['A'].get(key)!=configs['D'].get(key):raise ValueError('Training config mismatch: '+key)
    for key in ('data_fingerprint','pretrained_sha256','teacher_sha256'):
        if provenance['A'][key]!=provenance['D'][key]:raise ValueError('Run provenance mismatch: '+key)
    canonical=lambda rows:sorted((r['split'],r['sample_id'],r['image'],r['mask']) for r in rows)
    if canonical(manifests['A'])!=canonical(manifests['D']):raise ValueError('Manifest mismatch')
    dataset=SegmentationDataset(args.data_root,Path(provenance['A']['run'])/'manifest.csv',args.split,size=configs['A']['size'],augment=False)
    dataset.rows.sort(key=lambda row:row['sample_id'])
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output folder: '+str(out))
    for folder in ('masks_A','masks_D','cases'): (out/folder).mkdir(parents=True,exist_ok=True)
    metadata=dict(arguments=vars(args),models=provenance,n_images=len(dataset),complete=False,
                  metric_protocol='Original-resolution bilinear logit resize, align_corners=False, sigmoid >= saved threshold; VM-DMD BinaryMetrics; empty/empty Dice=1',
                  delta_unit='percentage points: 100*(Dice_D-Dice_A)',
                  crop_rule='Fixed quarter-size image ROI maximizing A/D disagreement on 32x32 pooled grid; fallback GT inner boundary or image center',
                  category_rule='Improved/regressed beyond +/- near_zero_pp; otherwise similar. Descriptive grouping, not statistical significance.',
                  case_selection='Median delta in improvement and regression groups; closest to zero in similar group; empty groups omitted')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    mean=torch.tensor([.485,.456,.406],device=args.device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=args.device).view(1,3,1,1)
    aggregates={role:BinaryMetrics() for role in ('A','D')};rows=[];mapping={}
    with (out/'paired_metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream,torch.no_grad():
        writer=None
        for index in range(len(dataset)):
            image,target,sid=dataset[index];record=dataset.rows[index];mapping[sid]=record
            normalized=(image[None].to(args.device)-mean)/std
            metrics={};predictions={}
            for role in ('A','D'):
                logits=models[role](normalized)
                logits=F.interpolate(logits,size=target.shape[-2:],mode='bilinear',align_corners=False)
                pred=(logits.sigmoid()>=configs[role]['threshold']).cpu()[0]
                single=BinaryMetrics();single.update(pred,target);aggregates[role].update(pred,target)
                metrics[role]=single.compute();predictions[role]=pred[0].numpy()
                Image.fromarray(predictions[role].astype(np.uint8)*255).save(out/('masks_'+role)/(sid+'.png'))
            gt=target[0].numpy().astype(bool);box=crop_box(gt,predictions['A'],predictions['D'])
            row=dict(sample_id=sid,dice_A=metrics['A']['dice'],dice_D=metrics['D']['dice'],
                     delta_dice_pp=100*(metrics['D']['dice']-metrics['A']['dice']),iou_A=metrics['A']['iou'],iou_D=metrics['D']['iou'],
                     crop_x0=box[0],crop_y0=box[1],crop_x1=box[2],crop_y1=box[3],
                     image_sha256=sha256(Path(args.data_root)/record['image']),mask_sha256=sha256(Path(args.data_root)/record['mask']))
            row['category']='Improvement' if row['delta_dice_pp']>args.near_zero_pp else ('Regression' if row['delta_dice_pp'] < -args.near_zero_pp else 'Similar')
            rows.append(row)
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);stream.flush()
            if args.render=='all':
                with Image.open(Path(args.data_root)/record['image']) as im:rgb=np.asarray(im.convert('RGB'))
                if rgb.shape[:2]!=gt.shape:raise ValueError('Image/mask grid mismatch: '+sid)
                plot_cases([dict(rgb=rgb,gt=gt,a=predictions['A'],d=predictions['D'],crop=box,metrics=row,category=row['category'])],out/'cases'/('figure4_'+sid),formats=('png',))
            print('[{}/{}] {}: A={:.2f}% D={:.2f}% delta={:+.3f} pp'.format(index+1,len(dataset),sid,row['dice_A']*100,row['dice_D']*100,row['delta_dice_pp']),flush=True)
    groups,chosen=select_cases(rows,args.near_zero_pp)
    write_csv(out/'ranked_cases.csv',sorted(rows,key=lambda r:(-r['delta_dice_pp'],r['sample_id'])))
    combined=[]
    for category,row in chosen:
        sid=row['sample_id'];record=mapping[sid]
        with Image.open(Path(args.data_root)/record['image']) as im:rgb=np.array(im.convert('RGB'))
        with Image.open(Path(args.data_root)/record['mask']) as im:gt=np.array(im.convert('L'))>0
        with Image.open(out/'masks_A'/(sid+'.png')) as im:a=np.array(im)>0
        with Image.open(out/'masks_D'/(sid+'.png')) as im:d=np.array(im)>0
        case=dict(rgb=rgb,gt=gt,a=a,d=d,crop=[row[k] for k in ('crop_x0','crop_y0','crop_x1','crop_y1')],metrics=row,category=category)
        combined.append(case);plot_cases([case],out/('example_'+category.lower()))
    plot_cases(combined,out/'figure4a_representative_cases')
    plot_deltas(rows,out/'figure4b_all_image_differences',args.near_zero_pp)
    summary=dict(n_images=len(rows),global_A=aggregates['A'].compute(),global_D=aggregates['D'].compute(),
                 mean_image_dice_A=float(np.mean([r['dice_A'] for r in rows])),mean_image_dice_D=float(np.mean([r['dice_D'] for r in rows])),
                 mean_paired_delta_pp=float(np.mean([r['delta_dice_pp'] for r in rows])),category_counts={k:len(v) for k,v in groups.items()},
                 exact_positive=sum(r['delta_dice_pp']>0 for r in rows),exact_negative=sum(r['delta_dice_pp']<0 for r in rows),exact_zero=sum(r['delta_dice_pp']==0 for r in rows),
                 grouping_tolerance_pp=args.near_zero_pp,selected_cases=[dict(category=c,sample_id=r['sample_id'],delta_dice_pp=r['delta_dice_pp']) for c,r in chosen])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    metadata['complete']=True;(out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Finished: '+str(out),flush=True)


if __name__=='__main__':main()

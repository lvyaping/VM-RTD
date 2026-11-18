"""Figure 3: fixed teacher q versus internal scan distributions of paper A/D.
No training; calls the exact neighbor/strength functions used by scan_transfer.py.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DIRECTIONS={'row_forward':0,'column_forward':1,'row_backward':2,'column_backward':3}
OFFSETS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def resolve_run(root,dataset,seed,explicit,role):
    if explicit:
        p=Path(explicit)
        return p.resolve() if p.is_absolute() else (root/'runs'/dataset/p).resolve()
    names=(['stage34_relation_300_control_v1','sab_A_relation_300_v1'] if role=='A'
           else ['scan_transfer_stage3_300_v1','sab_B_transfer_300_v1'])
    found=[root/'runs'/dataset/(n+'_branch_relation_seed'+str(seed)) for n in names]
    found=[p for p in found if (p/'best.pt').is_file()]
    if len(found)!=1:
        raise ValueError('Set FIG3_RUN_'+role+' to the full result folder name. Candidates: '+str(found))
    return found[0]


def check_config(cfg,role):
    if cfg.get('student_arch')!='shallow' or cfg.get('mode')!='branch' or cfg.get('adapter')!='relation' or cfg.get('kd_stages')!=[2,3] or cfg.get('kd_stage_weights')!=[.5,.5]:
        raise ValueError(role+': expected shallow stage34 branch-relation experiment')
    if cfg.get('scan_transfer_only',False):raise ValueError('Transfer-only is not paper D')
    if cfg.get('beta')!=.1 or cfg.get('relation_radius',1)!=1:
        raise ValueError('Expected relation beta=0.1 and radius=1')
    weight=cfg.get('scan_transfer_weight',0.)
    if (role=='A' and weight!=0) or (role=='D' and (weight<=0 or cfg.get('scan_transfer_variant','full')!='full')):
        raise ValueError(role+': wrong experiment; D must be FULL transfer, not endpoint/output')
    for key in ('prototype_weight','cross_scale_weight','collective_mode','collective_aux_weight','relation_curriculum','relation_reliability','kd_direction_weights','relation_component_weights','task_gradient'):
        if cfg.get(key):raise ValueError('Unexpected extra supervision: '+key)


def choose_queries(mask,h,w,valid_spatial):
    """Deterministic GT-only selection; units are feature cells, not input pixels."""
    resized=F.interpolate(torch.from_numpy(mask)[None,None].float(),size=(h*16,w*16),mode='nearest')
    coverage=F.adaptive_avg_pool2d(resized,(h,w))[0,0].numpy()
    fg=coverage>=.9; bg=coverage<=.1
    hard=coverage>=.5
    boundary=~(fg|bg)
    for y in range(h):
        for x in range(w):
            for dy,dx in OFFSETS:
                yy,xx=y+dy,x+dx
                if 0<=yy<h and 0<=xx<w and hard[yy,xx]!=hard[y,x]:boundary[y,x]=True
    valid=np.zeros((h,w),dtype=bool);valid.flat[valid_spatial]=True
    # Display queries avoid the outer grid border, so neighborhoods are complete.
    valid[[0,-1],:]=False;valid[:,[0,-1]]=False
    coords=np.stack(np.meshgrid(np.arange(h),np.arange(w),indexing='ij'),-1)
    chosen=[]
    for region,allowed in [('Lesion interior',fg&~boundary),('Near boundary',boundary),('Background',bg&~boundary)]:
        points=coords[allowed&valid]
        if not len(points):return []
        if region=='Near boundary':
            score=-((points-np.array([(h-1)/2,(w-1)/2]))**2).sum(-1)
        else:
            edge=coords[boundary]
            score=((points[:,None]-edge[None])**2).sum(-1).min(-1) if len(edge) else np.zeros(len(points))
        y,x=points[np.argmax(score)]
        chosen.append(dict(region=region,row=int(y),col=int(x),foreground_fraction=float(coverage[y,x])))
    return chosen


def distributions(capture,teacher,direction,spatial_queries):
    from vmdmd.scan_transfer import neighbor_indices,coefficient_strength
    h,w=capture['shape']
    ii,jj,valid,pi,pj=neighbor_indices(h,w,direction,teacher.device)
    indices=[]
    for query in spatial_queries:
        found=torch.nonzero(pi==query,as_tuple=False).flatten()
        if len(found)!=1:raise ValueError('Query must have at least two valid predecessors')
        indices.append(int(found[0]))
    select=torch.tensor(indices,device=teacher.device)
    ii,jj,valid,pi,pj=(v[select] for v in (ii,jj,valid,pi,pj))
    delta=F.softplus(capture['dts'].float()+capture['bias'].float()[None,:,:,None])[:,direction]
    safe_j=torch.where(valid,jj,ii[:,None])
    strength=coefficient_strength(delta.cumsum(-1),delta,capture['A'][direction].float(),
                                  capture['B'][:,direction].float(),capture['C'][:,direction].float(),ii,safe_j)
    mass=(strength+1e-12)*valid[None]
    p=mass/mass.sum(-1,keepdim=True)
    t=F.normalize(teacher.detach().float(),dim=1,eps=1e-6).flatten(2)
    scores=(t[:,:,pi].unsqueeze(-1)*t[:,:,pj]).sum(1).masked_fill(~valid[None],float('-inf'))
    q=scores.softmax(-1)
    kl=(q*(q.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1)
    if not torch.isfinite(p).all() or not torch.isfinite(kl).all():raise FloatingPointError('Nonfinite transfer distribution')
    return dict(p=p[0].cpu().numpy(),q=q[0].cpu().numpy(),kl=kl[0].cpu().numpy(),
                valid=valid.cpu().numpy(),source_spatial=pj.cpu().numpy(),query_scan=ii.cpu().numpy(),
                source_scan=jj.cpu().numpy(),strength=strength[0].cpu().numpy())


def capture_scan(model,image):
    scan=model.layers[2].blocks[-1].self_attention
    scan.capture_transfer=True
    try:
        model(image)
        capture=scan.transfer_capture
    finally:
        scan.capture_transfer=False
        scan.transfer_capture=None
    return capture


def draw(rgb,mask,queries,a,d,direction,sample_id,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(len(queries),4,figsize=(11,2.8*len(queries)),squeeze=False,constrained_layout=True)
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#dedede');norm=Normalize(0,1)
    h,w=mask.shape
    grid_h,grid_w=a['grid']
    for r,query in enumerate(queries):
        ax=axes[r,0];ax.imshow(rgb)
        if mask.min()!=mask.max():ax.contour(mask,levels=[.5],colors=['#18c7ad'],linewidths=.8)
        x=query['col']*w/grid_w;y=query['row']*h/grid_h
        ax.add_patch(Rectangle((x,y),w/grid_w,h/grid_h,fill=False,edgecolor='#e38312',linewidth=2))
        ax.set_title(query['region']+'\ncell (row={}, col={})'.format(query['row'],query['col']),fontsize=10)
        ax.axis('off')
        for c,(title,values) in enumerate([('Teacher target q',a['q'][r]),('A: relation KD',a['p'][r]),('D: full method',d['p'][r])],start=1):
            ax=axes[r,c];matrix=np.full((3,3),np.nan)
            for k,(dy,dx) in enumerate(OFFSETS):
                if a['valid'][r,k]:matrix[dy+1,dx+1]=values[k]
            im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,norm=norm)
            for k,(dy,dx) in enumerate(OFFSETS):
                if a['valid'][r,k]:
                    value=values[k];ax.text(dx+1,dy+1,'{:.2f}'.format(value),ha='center',va='center',color='white' if value>.6 else '#14283a')
            ax.add_patch(Rectangle((.5,.5),1,1,facecolor='#f5d38a',edgecolor='white'))
            ax.text(1,1,'Q',ha='center',va='center',weight='bold')
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
            ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',length=0)
            ax.set_title(title)
            if c>1:ax.set_xlabel('KL(q || p) = {:.5f}'.format((a if c==2 else d)['kl'][r]))
    fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),shrink=.75,label='Relative weight (shared scale)')
    fig.suptitle('Stage 3 | {} | {}\nTeal: GT boundary; orange: query cell; gray: excluded candidate'.format(sample_id,direction.replace('_',' ')),fontsize=12)
    for suffix in ('png','pdf','svg'):fig.savefig(str(output)+'.'+suffix,dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--dataset',choices=['isic17','isic18','bus','busi'],default='isic18')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-a',default='');p.add_argument('--run-d',default='')
    p.add_argument('--teacher',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--direction',choices=list(DIRECTIONS),default='row_forward')
    p.add_argument('--sample-id',nargs='*',default=[])
    p.add_argument('--limit',type=int,default=0,help='0: all eligible images; positive: first N eligible images')
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    if args.limit<0:p.error('limit must be nonnegative (0 means all eligible images)')
    root=Path(args.project_root).resolve();sys.path.insert(0,str(root))
    import os
    os.environ['VMDMD_SCAN_BACKEND']='cuda' if args.device=='cuda' else 'reference'
    from vmdmd.models.vmamba import VSSM
    from vmdmd.models.swin import SwinTransformer
    from vmdmd.checkpoints import load_teacher,sha256
    from vmdmd.data import read_manifest,SegmentationDataset
    from vmdmd.scan_transfer import neighbor_indices
    device=torch.device(args.device)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher_hash=sha256(args.teacher)
    models={};configs={};manifests={};info={}
    for role,explicit in [('A',args.run_a),('D',args.run_d)]:
        run=resolve_run(root,args.dataset,args.seed,explicit,role)
        ck=torch.load(run/'best.pt',map_location='cpu')
        cfg=ck['config'];check_config(cfg,role)
        report=json.loads((run/'weight_load.json').read_text(encoding='utf-8-sig'))
        if report['teacher']['sha256']!=teacher_hash:raise ValueError(role+': teacher checkpoint mismatch')
        if cfg['seed']!=args.seed:raise ValueError(role+': unexpected seed')
        configs[role]=cfg;manifests[role]=read_manifest(run/'manifest.csv')
        model=VSSM(num_classes=1,depths=[2,2,2,2],depths_decoder=[2,2,2,1],drop_path_rate=.2)
        model.load_state_dict(ck['student'],strict=True);models[role]=model.to(device).eval()
        info[role]=dict(run=str(run),checkpoint_sha256=sha256(run/'best.pt'),epoch=ck['epoch'],
                        data_fingerprint=ck.get('data_fingerprint'),pretrained_sha256=report['student']['sha256'])
        del ck
    for key in ('size','threshold','seed','epochs','batch_size','lr','min_lr','weight_decay','augmentation','rotation_degrees','student_arch'):
        if configs['A'].get(key)!=configs['D'].get(key):raise ValueError('Protocol differs: '+key)
    if info['A']['pretrained_sha256']!=info['D']['pretrained_sha256']:raise ValueError('Student initialization differs')
    if info['A']['data_fingerprint']!=info['D']['data_fingerprint']:raise ValueError('Training dataset fingerprints differ')
    canonical=lambda rows:sorted((r['split'],r['sample_id'],r['image'],r['mask']) for r in rows)
    if canonical(manifests['A'])!=canonical(manifests['D']):raise ValueError('Manifest mismatch')
    size=configs['A']['size'];h=w=size//16;direction=DIRECTIONS[args.direction]
    valid=neighbor_indices(h,w,direction,torch.device('cpu'))[3].numpy()
    dataset=SegmentationDataset(args.data_root,info['A']['run']+'/manifest.csv',args.split,size=size,augment=False)
    by_id={r['sample_id']:i for i,r in enumerate(dataset.rows)}
    ids=args.sample_id or sorted(by_id)
    selected=[];skipped=[]
    for checked,sample_id in enumerate(ids,start=1):
        if checked%100==0:print('Checking GT eligibility: {}/{}'.format(checked,len(ids)),flush=True)
        if sample_id not in by_id:raise ValueError('Sample not in requested split: '+sample_id)
        index=by_id[sample_id];image,mask,_=dataset[index]
        queries=choose_queries(mask[0].numpy(),h,w,valid)
        if not queries:
            skipped.append(sample_id)
            if args.sample_id:raise ValueError(sample_id+': missing one of the three valid GT regions at Stage 3')
            continue
        selected.append((index,queries))
        if not args.sample_id and args.limit>0 and len(selected)>=args.limit:break
    if not selected:raise ValueError('No images with three valid query regions; try another split or direction')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use a new output folder: '+str(out))
    selection_report=dict(split=args.split,split_images=len(dataset.rows),checked_images=checked,
                          eligible_selected=len(selected),skipped_images=len(skipped),
                          selected_ids=[dataset.rows[index]['sample_id'] for index,_ in selected],
                          skipped=[dict(sample_id=s,reason='Missing at least one GT region with a valid non-border Stage-3 query') for s in skipped],
                          limit=args.limit,all_split_images_checked=(not args.sample_id and checked==len(ids)))
    (out/'selection_report.json').write_text(json.dumps(selection_report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Checked {} images; selected {}; skipped {}.'.format(checked,len(selected),len(skipped)),flush=True)
    teacher=SwinTransformer(drop_path_rate=.3)
    load_teacher(teacher,args.teacher)
    # This legacy Swin overrides train() without returning self, so eval()
    # returns None. Keep the model reference instead of assigning that result.
    teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    rows=[];selection=[]
    with torch.no_grad():
        for completed,(index,queries) in enumerate(selected,start=1):
            image,mask,sample_id=dataset[index];record=dataset.rows[index]
            x=(image[None].to(device)-mean)/std
            target=teacher(x)[2]
            arrays={}
            for role in ('A','D'):
                capture=capture_scan(models[role],x)
                arrays[role]=distributions(capture,target,direction,[q['row']*w+q['col'] for q in queries])
                arrays[role]['grid']=(h,w)
                del capture
            a,d=arrays['A'],arrays['D']
            assert np.array_equal(a['valid'],d['valid']) and np.allclose(a['q'],d['q'])
            with Image.open(Path(args.data_root)/record['image']) as im:rgb=np.asarray(im.convert('RGB'))
            stem=out/('figure3_'+sample_id)
            draw(rgb,mask[0].numpy(),queries,a,d,args.direction,sample_id,stem)
            np.savez_compressed(str(stem)+'.npz',**{role+'_'+k:v for role,result in arrays.items() for k,v in result.items()})
            for i,q in enumerate(queries):
                rows.append(dict(sample_id=sample_id,direction=args.direction,**q,kl_A=float(a['kl'][i]),kl_D=float(d['kl'][i]),
                                 kl_D_minus_A=float(d['kl'][i]-a['kl'][i]),valid_candidates=int(a['valid'][i].sum())))
            selection.append(dict(sample_id=sample_id,queries=queries,image_sha256=sha256(Path(args.data_root)/record['image']),
                                  mask_sha256=sha256(Path(args.data_root)/record['mask'])))
            print('[{}/{}] Saved {}'.format(completed,len(selected),stem),flush=True)
    with (out/'query_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata=dict(arguments=vars(args),models=info,teacher_sha256=teacher_hash,selection=selection,skipped_missing_regions=skipped,
                  selection_report=selection_report,
                  selection_rule='GT only; explicit IDs if supplied, otherwise sorted IDs; limit=0 selects all eligible images; no selection by KL or segmentation gain',
                  scope='Individual selected queries in one direction; NOT whole-test-set mean KL',
                  region_rule='GT occupancy >=0.9 foreground / <=0.1 background; boundary=mixed cells or 8-neighbor foreground-majority transition; interior/background exclude boundary; display queries exclude outer grid ring',
                  transfer='Original coefficient_strength; state sum then abs then channel mean; epsilon 1e-12; teacher softmax T=1',
                  stage=3,grid=[h,w],offsets=OFFSETS,coordinate_convention='zero-based feature-grid row,col')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Finished. Outputs: PNG/PDF/SVG, NPZ, query_metrics.csv, metadata.json',flush=True)


if __name__=='__main__':main()

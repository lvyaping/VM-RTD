"""Offline, image-paired spatial affinity diagnostics. No training or threshold tuning."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

VIEWS=('fused','row_forward','row_backward','col_forward','col_backward')

def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def save_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def choose_samples(rows,split,limit,seed):
    rows=[r for r in rows if r['split']==split]
    rows.sort(key=lambda r:hashlib.sha256((str(seed)+':'+r['sample_id']).encode()).hexdigest())
    if limit:rows=rows[:limit]
    if not rows:raise ValueError('No samples selected')
    return rows

def pair_grid(h,w,radii):
    # Four canonical offsets enumerate each undirected eight-neighbor pair ONCE.
    # Cosine is symmetric; duplicating the opposite offsets adds no evidence.
    p,q,rr,dy,dx=[],[],[],[],[]
    for r in radii:
        if r<1 or r>=min(h,w):raise ValueError('Radius outside feature grid')
        for yy,xx in [(0,r),(r,0),(r,r),(r,-r)]:
            for y in range(h-yy):
                for x in range(max(0,-xx),min(w,w-xx)):
                    p.append(y*w+x);q.append((y+yy)*w+x+xx)
                    rr.append(r);dy.append(yy);dx.append(xx)
    p,q,rr,dy,dx=map(lambda v:np.asarray(v,dtype=np.int64),(p,q,rr,dy,dx))
    return dict(p=p,q=q,radius=rr,dy=dy,dx=dx,distance2=dy*dy+dx*dx,
                row_distance=np.abs(q-p),
                col_distance=np.abs((q%w)*h+q//w-((p%w)*h+p//w)))

def classes(occupancy,p,q,purity):
    o=occupancy.reshape(-1)
    fg=o>=purity;bg=o<=1-purity
    # 0 mixed/ambiguous, 1 foreground-foreground, 2 foreground-background, 3 background-background
    return np.where(fg[p]&fg[q],1,np.where((fg[p]&bg[q])|(bg[p]&fg[q]),2,np.where(bg[p]&bg[q],3,0)))

def auc(scores,positive):
    scores=np.asarray(scores);positive=np.asarray(positive,dtype=bool)
    n1=int(positive.sum());n0=len(positive)-n1
    if not n1 or not n0:return None
    order=np.argsort(scores,kind='mergesort');v=scores[order]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];ends=np.r_[starts[1:],len(v)]
    ranks=np.empty(len(v),dtype=np.float64)
    for a,b in zip(starts,ends):ranks[a:b]=(a+1+b)/2
    return float((ranks[positive[order]].sum()-n1*(n1+1)/2)/(n1*n0))

def measures(scores,labels):
    ff=scores[labels==1];cross=scores[labels==2];bb=scores[labels==3]
    keep=(labels==1)|(labels==2)
    return dict(n_fg_fg=len(ff),n_cross=len(cross),n_bg_bg=len(bb),n_mixed=int((labels==0).sum()),
                mean_fg_fg=float(ff.mean()) if len(ff) else None,
                mean_cross=float(cross.mean()) if len(cross) else None,
                mean_bg_bg=float(bb.mean()) if len(bb) else None,
                gap=float(ff.mean()-cross.mean()) if len(ff) and len(cross) else None,
                auc=auc(scores[keep],labels[keep]==1))

def groups(grid,view):
    n=len(grid['p']);yield 'all',np.ones(n,dtype=bool)
    for r in np.unique(grid['radius']):
        mask=grid['radius']==r
        yield 'radius={}'.format(r),mask
        for dy,dx in sorted(set(zip(grid['dy'][mask],grid['dx'][mask]))):
            yield 'offset={},{}'.format(dy,dx),(grid['dy']==dy)&(grid['dx']==dx)
    if view!='fused':
        d=grid['row_distance'] if view.startswith('row') else grid['col_distance']
        for d2 in np.unique(grid['distance2']):
            mask=grid['distance2']==d2
            for sd in np.unique(d[mask]):
                yield 'distance2={};scan_distance={}'.format(d2,sd),mask&(d==sd)

def paired_ci(a,b,seed=42,repeats=1000):
    keys=sorted(set(a)&set(b));delta=np.asarray([a[k]-b[k] for k in keys],dtype=float)
    if not len(delta):return dict(n_images=0,mean_delta=None,ci_low=None,ci_high=None)
    # Resample images, NEVER treat within-image pairs as independent observations.
    if len(delta)<2:return dict(n_images=1,mean_delta=float(delta.mean()),ci_low=None,ci_high=None)
    rng=np.random.RandomState(seed)
    means=np.asarray([delta[rng.randint(0,len(delta),len(delta))].mean() for _ in range(repeats)])
    lo,hi=np.percentile(means,[2.5,97.5])
    return dict(n_images=len(delta),mean_delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi))

def analyze(cache_dir,out,grid,purity,seed,repeats):
    records=[];dice=[]
    for path in sorted(Path(cache_dir).glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            model=str(z['model']);sid=str(z['sample_id']);labels=classes(z['occupancy'],grid['p'],grid['q'],purity)
            for vi,view in enumerate(VIEWS):
                for group,mask in groups(grid,view):
                    records.append(dict(model=model,sample_id=sid,view=view,group=group,**measures(z['scores'][vi,mask],labels[mask])))
            if 'dice' in z:dice.append(dict(model=model,sample_id=sid,dice=float(z['dice'])))
    save_csv(out/'per_image_relations.csv',records);save_csv(out/'per_image_dice.csv',dice)
    models=sorted(set(r['model'] for r in records));keys=sorted(set((r['view'],r['group']) for r in records))
    summaries=[];paired=[]
    index={}
    for r in records:index.setdefault((r['model'],r['view'],r['group']),[]).append(r)
    for model in models:
        for view,group in keys:
            rows=index[(model,view,group)];valid=[r for r in rows if r['auc'] is not None]
            summaries.append(dict(model=model,view=view,group=group,n_images=len(rows),n_valid_images=len(valid),
                macro_auc=float(np.mean([r['auc'] for r in valid])) if valid else None,
                macro_gap=float(np.mean([r['gap'] for r in valid])) if valid else None,
                total_fg_fg=sum(r['n_fg_fg'] for r in rows),total_cross=sum(r['n_cross'] for r in rows),
                total_bg_bg=sum(r['n_bg_bg'] for r in rows),total_mixed=sum(r['n_mixed'] for r in rows)))
    for left,right in [('teacher','baseline'),('fused_relation','baseline'),('branch_relation','baseline'),('branch_relation','fused_relation')]:
        for view,group in keys:
            for metric in ['auc','gap']:
                a={r['sample_id']:r[metric] for r in index[(left,view,group)] if r[metric] is not None}
                b={r['sample_id']:r[metric] for r in index[(right,view,group)] if r[metric] is not None}
                paired.append(dict(comparison=left+' minus '+right,view=view,group=group,metric=metric,**paired_ci(a,b,seed,repeats)))
    save_csv(out/'relation_summary.csv',summaries);save_csv(out/'paired_changes.csv',paired)
    dicechanges=[]
    for model in ['fused_relation','branch_relation']:
        a={r['sample_id']:r['dice'] for r in dice if r['model']==model};b={r['sample_id']:r['dice'] for r in dice if r['model']=='baseline'}
        for sid in sorted(set(a)&set(b)):dicechanges.append(dict(model=model,sample_id=sid,baseline_dice=b[sid],student_dice=a[sid],delta=a[sid]-b[sid]))
    save_csv(out/'paired_dice.csv',dicechanges)
    save_json(out/'summary.json',dict(primary=[r for r in summaries if r['group']=='all'],
        primary_paired=[r for r in paired if r['group']=='all'],
        interpretation=['AUROC: foreground-foreground positive, foreground-background negative; no segmentation threshold tuning.',
            'Teacher views repeat the SAME teacher features solely to provide matched scan-distance strata.',
            'Macro AUROC averages eligible images; paired CI uses common eligible images, not pixel-pair pseudo-replication.',
            'Scan distance is confounded with orientation at a fixed grid; strata are descriptive, not proof of forgetting.',
            'Cosine is symmetric; forward/backward scan distances are identical, no causal influence claimed.',
            'Only eight-offset rings at fixed radii are sampled. Mixed cells excluded from AUROC and counted separately.',
            'Teacher backbone only: no teacher segmentation Dice is computed. Student Dice uses native masks and threshold 0.5.',
            'Bootstrap intervals cover image sampling, not training-seed uncertainty; many secondary strata are exploratory.']))
    return records

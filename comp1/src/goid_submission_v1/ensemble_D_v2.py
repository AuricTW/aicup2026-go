"""Fixed heterogeneous ensemble: 50% expanded U1 rank + 50% U0 fused rank.

Candidate/query alignment is explicit. Both components share the fulltrain C
encoder in the competition bundle. No weights are selected during inference.
"""
from pathlib import Path
import csv,time
import numpy as np
from goid import inference as base
from goid_v4.csv_u0_v1 import run_job as run_u0,write,load_bundle as load_u0
from .gaussian_D_csv_v1 import run_job as run_gaussian,load_bundle as load_gaussian
ROOT=Path(__file__).resolve().parents[2]


def fixed_rrf60(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.ndim!=2 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid aligned components')
    result=np.zeros(a.shape,np.float64)
    for score in (a,b):
        order=np.argsort(-score,axis=1,kind='stable')
        rank=np.empty_like(order);np.put_along_axis(rank,order,np.broadcast_to(np.arange(1,a.shape[1]+1),order.shape),axis=1)
        result+=.5/(60+rank)
    return result


def load_bundle(path):
    p=Path(path).resolve();m=base.read_json(p/'manifest.json')
    if m.get('schema')!='go-D-ensemble-v2' or base.sha(p/'manifest.json')!=base.read_json(p/'manifest.sha256.json')['manifest.json']:raise ValueError('Ensemble manifest invalid')
    if m.get('recipe')!={'gaussian_rank_weight':.5,'u0_fused_rank_weight':.5,'rrf_k':60}:raise ValueError('D formula changed')
    for name,h in m['files'].items():
        target=(p/name).resolve()
        if not target.is_relative_to(p) or base.sha(target)!=h:raise ValueError('D ensemble payload changed')
    for name,h in m['code_hashes'].items():
        if base.sha(ROOT/name)!=h:raise ValueError('D runtime changed')
    load_u0(p/'u0');load_gaussian(p/'gaussian')
    return p,m


def run_job(job,output):
    out=Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT):raise ValueError('New project output required')
    if job.get('method')!='D_ensemble' or any(k in job for k in ('alpha','weight','prior_scale','noise_scale')):raise ValueError('Frozen D ensemble only')
    bundle,manifest=load_bundle(job['bundle']);out.mkdir(parents=True);start=time.perf_counter();write(out/'job.json',job)
    try:
        for key,method,folder,runner in [('u0','u0','u0',run_u0),('gaussian','D_expanded_U1','gaussian',run_gaussian)]:
            component=dict(job,bundle=str(bundle/folder),method=method,include_comparisons=False)
            runner(component,out/key)
        rows=[]
        for pool in job['pools']:
            name=pool['name'];a=out/'gaussian'/name;b=out/'u0'/name
            mapping=base.read_json(a/'mapping.json')
            if mapping!=base.read_json(b/'mapping.json'):raise ValueError('Component candidate/query alignment differs')
            scores=fixed_rrf60(np.load(a/'D_scores.npy',allow_pickle=False),np.load(b/'u0_scores.npy',allow_pickle=False))
            pd=out/name;pd.mkdir();np.save(pd/'D_scores.npy',scores,allow_pickle=False);write(pd/'mapping.json',mapping)
            order=np.argsort(-scores,axis=1,kind='stable')[:,:5]
            for i,qid in enumerate(mapping['question_ids']):rows.append([qid,*[mapping['candidate_ids'][j] for j in order[i]]])
        with (out/'submission_selected.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(base.HEADER);writer.writerows(rows)
        load_bundle(bundle)
        result={'status':'complete','method':'D_ensemble','queries':len(rows),'elapsed_seconds':time.perf_counter()-start,
            'recipe':manifest['recipe'],'both_components_same_input':True,'training_or_tuning':False,'uploaded':False,
            'fulltrain_accuracy_claim':None,'component_summaries':['u0/summary.json','gaussian/summary.json']}
        write(out/'summary.json',result);write(out/'status.json',{'status':'complete'})
        write(out/'output_sha256.json',{p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file()})
        return result
    except BaseException as e:
        write(out/'status.json',{'status':'failed_preserved','error':str(e)});raise

"""Frozen development-best U1 inference; no fitting or hyperparameter changes."""
from pathlib import Path
import hashlib,importlib.metadata,json,sys,time,traceback
import numpy as np
from goid import inference as base,fixed_batch
from goid_v4.csv_u0_v1 import write,extract,canonical_pool
from goid_v4.p1 import compose_game_features

ROOT=Path(__file__).resolve().parents[2]

class FrozenGaussian:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            for name in ('center','basis','prior_mean','noise','K'):setattr(self,name,z[name].copy())
            self.prior_scale=float(z['prior_scale']);self.noise_scale=float(z['noise_scale'])
        self.projection=np.array(self.basis.T,np.float64,order='C',copy=True)
    def transform(self,features):
        out=np.empty((len(features),128),np.float64)
        for i,row in enumerate(features):out[i]=self.projection@np.array(np.asarray(row,np.float64)-self.center,order='C',copy=True)
        return out
    def posterior(self,summary):
        mean,n=summary;K=self.K*self.prior_scale;noise=self.noise*self.noise_scale
        determinant=K[0,0]*K[1,1]-K[0,1]**2
        if (determinant<=0).any():raise ValueError('Prior covariance not positive')
        k00=K[1,1]/determinant;k11=K[0,0]/determinant;k01=-K[0,1]/determinant
        a=k00+n[:,0,None]/noise[0];d=k11+n[:,1,None]/noise[1];b=k01;det=a*d-b*b
        C00=d/det;C11=a/det;C01=-b/det
        h0=k00*self.prior_mean[0]+k01*self.prior_mean[1]+n[:,0,None]*mean[:,0]/noise[0]
        h1=k01*self.prior_mean[0]+k11*self.prior_mean[1]+n[:,1,None]*mean[:,1]/noise[1]
        return (C00*h0+C01*h1,C01*h0+C11*h1),(C00,C01,C11)
    def score(self,query,support):
        qm,qn=query;(m0,m1),(C00,C01,C11)=self.posterior(support);noise=self.noise*self.noise_scale
        scores=np.empty((len(qm),len(m0)),np.float64)
        for i in range(len(qm)):
            if qn[i,0] and qn[i,1]:
                a=C00+noise[0]/qn[i,0];d=C11+noise[1]/qn[i,1];b=C01
                det=a*d-b*b;x=qm[i,0]-m0;y=qm[i,1]-m1
                scores[i]=-.5*np.sum(np.log(det)+(d*x*x+a*y*y-2*b*x*y)/det,axis=1)
            else:
                c=0 if qn[i,0] else 1;variance=(C00 if c==0 else C11)+noise[c]/qn[i,c]
                diff=qm[i,c]-(m0 if c==0 else m1);scores[i]=-.5*np.sum(np.log(variance)+diff*diff/variance,axis=1)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite Gaussian score')
        return scores

def summaries(vectors,colors,sets):
    means=np.zeros((len(sets),2,128),np.float64);counts=np.zeros((len(sets),2),np.int64)
    for i,ids in enumerate(sets):
        ids=np.asarray(sorted(ids),np.int64)
        if not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid Gaussian set')
        for c in (0,1):
            cc=ids[colors[ids]==c];counts[i,c]=len(cc)
            if len(cc):means[i,c]=np.asarray(vectors[cc],np.float64).mean(0)
    return means,counts

def load_bundle(path):
    p=Path(path).resolve();m=base.read_json(p/'manifest.json')
    if m.get('schema')!='go-frozen-U1-competition-v1' or base.sha(p/'manifest.json')!=base.read_json(p/'manifest.sha256.json')['manifest.json']:raise ValueError('U1 manifest invalid')
    for name,h in m['files'].items():
        target=(p/name).resolve()
        if not target.is_relative_to(p) or base.sha(target)!=h:raise ValueError('U1 payload changed')
    for name,h in m['code_hashes'].items():
        if base.sha(ROOT/name)!=h:raise ValueError('Pinned U1 runtime changed')
    for name,version in m['runtime'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Runtime changed')
    return {'path':p,'manifest':m,'banned_raw':set(),'banned_players':set(),'banned_canonical':set()}

def run_job(job,output):
    out=Path(output).resolve();cache=Path(job['cache']).resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not cache.is_relative_to(ROOT) or cache.is_relative_to(out):raise ValueError('Use new project output and separate cache')
    bundle=load_bundle(job['bundle']);before={name:base.sha(bundle['path']/name) for name in bundle['manifest']['files']}
    if job.get('method')!='u1_single' or any(k in job for k in ('prior_scale','noise_scale','dimension','alpha')):raise ValueError('Frozen U1 method/scales only')
    games={};pools=[]
    for spec in job['pools']:
        if spec.get('expected_candidates',400)!=400 or spec.get('reference_games',100)!=100:raise ValueError('400x100 required')
        pool=base.load_pool(spec,games,bundle);binding=base.read_json(spec['order_manifest']) if 'order_manifest' in spec else None
        pools.append(canonical_pool(pool,binding))
    qids=[q['question_id'] for p in pools for q in p.questions]
    if not pools or len(qids)!=len(set(qids)) or len({p.name for p in pools})!=len(pools):raise ValueError('Pool/qid mismatch')
    out.mkdir(parents=True);t=time.perf_counter();write(out/'job.json',job)
    try:
        raw,keys,metadata,cachestats=extract(games,cache,out,job.get('workers',4));positions={k:i for i,k in enumerate(keys)}
        encoder={'path':bundle['path']/'base','manifest':{'model_config':bundle['manifest']['model_config'],'synthetic_control':False}}
        (stat,mlp),evidence=fixed_batch.encode_features(raw,encoder,'cuda')
        count=np.asarray([metadata[k]['target_moves'] for k in keys],np.float64);stages=np.rint(raw[:,9:12].astype(np.float64)*count[:,None])
        colors=np.asarray([int(games[k].color=='W') for k in keys],np.int64)
        features=compose_game_features(stat,mlp,raw[:,:12],[games[k].color for k in keys],[metadata[k]['mainline_moves'] for k in keys],stages)
        gaussian=FrozenGaussian(bundle['path']/'u1/model.npz');latent=gaussian.transform(features);paths=[]
        for pool in pools:
            scores=gaussian.score(summaries(latent,colors,[[positions[k] for k in q['keys']] for q in pool.questions]),
                summaries(latent,colors,[[positions[k] for k in pool.supports[p]] for p in pool.candidate_ids]))
            pd=out/pool.name;pd.mkdir();np.save(pd/'u1_single_scores.npy',scores,allow_pickle=False)
            base.write_top5(pd/'top5_selected.csv',pool,scores);paths.append(pd/'top5_selected.csv')
            write(pd/'mapping.json',{'candidate_ids':pool.candidate_ids,'question_ids':[q['question_id'] for q in pool.questions]})
        base.combine_files(paths,out/'submission_selected.csv');check=base.check_output(out/'submission_selected.csv',pools)
        load_bundle(job['bundle'])
        if before!={name:base.sha(bundle['path']/name) for name in before}:raise ValueError('Frozen U1 changed')
        result={'status':'complete','queries':len(qids),'method':'u1_single','elapsed_seconds':time.perf_counter()-t,'cache':cachestats,
            'PCA_policy':'one_game_FP64_GEMV_128x1728','no_training_or_tuning':True,'no_upload':True,'original_cohort_not_read':True,'format_check':check,
            'encoder_evidence':evidence,'limitation':'development-selected best frozen point estimate; no new independent quality claim'}
        write(out/'summary.json',result);write(out/'status.json',{'status':'complete'});write(out/'output_sha256.json',{p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file()})
        return result
    except BaseException as e:write(out/'status.json',{'status':'failed_preserved','error':str(e),'traceback':traceback.format_exc()});raise

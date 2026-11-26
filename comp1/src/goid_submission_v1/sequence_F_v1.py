"""Frozen opening-sequence likelihood + immutable E; CSV inference only.

No fitting, training/audit/GT reads, query adaptation, global assignment or upload.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib,importlib.metadata,json,re,time,traceback
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from goid import inference as base
from goid_v4.csv_u0_v1 import canonical_pool,write
from goid_v4.score_push_features_v1 import design,BINS
from .plda_E_v1 import load_bundle as load_E,run_job as run_E
from .ensemble_D_v2 import fixed_rrf60
ROOT=Path(__file__).resolve().parents[2]

def fit_background(x):
    """Offline serialization helper only; never called by run_job."""
    bg=np.asarray(x.sum(axis=0)).ravel().astype(np.float64)+1e-3
    for start in range(0,len(bg),BINS):bg[start:start+BINS]/=bg[start:start+BINS].sum()
    return bg

class FrozenSequence:
    def __init__(self,background,variant,alpha=20.):
        self.background=np.asarray(background,np.float64);self.variant=variant;self.alpha=float(alpha)
        if variant not in ('bag_position','target_order','joint_order_absolute','joint_order_D4') or alpha!=20. or len(self.background)%BINS or not (self.background>0).all():raise ValueError('Sequence model contract')
        self.groups=len(self.background)//BINS
        for g in range(self.groups):
            if abs(self.background[g*BINS:(g+1)*BINS].sum()-1)>1e-12:raise ValueError('Unnormalized background')
    def score(self,x,supports,queries,game_keys):
        if x.shape[1]!=len(self.background):raise ValueError('Sequence feature dimensions')
        ss=[sorted(s,key=lambda i:game_keys[i]) for s in supports];qs=[sorted(q,key=lambda i:game_keys[i]) for q in queries]
        if any(len(s)!=100 or len(set(s))!=100 for s in ss):raise ValueError('Sequence100support contract')
        if any(not 1<=len(q)<=20 or len(set(q))!=len(q) for q in qs):raise ValueError('Sequence query count')
        bank=sparse.vstack([sparse.csr_matrix(x[s].sum(axis=0)) for s in ss],format='csr',dtype=np.float64)
        norm=np.stack([np.asarray(bank[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(self.groups)],axis=1)
        bank.data=np.log1p(bank.data/(self.alpha*self.background[bank.indices]))
        used=sorted({i for q in qs for i in q},key=lambda i:game_keys[i]);qx=x[used]
        valid=np.stack([np.asarray(qx[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(self.groups)],axis=1)
        # Per-game sparse row products have no cross-query dependence.
        gs=(qx@bank.T).toarray()-valid@np.log1p(norm/self.alpha).T
        pos={j:i for i,j in enumerate(used)}
        out=np.stack([gs[[pos[j] for j in q]].mean(0)/self.groups for q in qs])
        if not np.isfinite(out).all():raise ValueError('Nonfinite sequence score')
        return out

def parse_opening_batch(games):
    from sgfmill import sgf
    result=[]
    for key,text in games:
        game=sgf.Sgf_game.from_bytes(text.encode());assert game.get_size()==19
        row=np.full(40,65535,np.uint16);j=0
        for node in game.get_main_sequence():
            color,point=node.get_move()
            if color is None:continue
            row[j]=(361 if point is None else point[0]*19+point[1])+(362 if color=='w' else 0)
            j+=1
            if j==40:break
        result.append((key,row))
    return result

def openings(games,cache,workers):
    keys=sorted(games);binding=hashlib.sha256(('|'.join(keys)+'|sgfmill1.1.1|first40-v1').encode()).hexdigest()
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);path=cache/f'{binding}.npz';receipt=cache/f'{binding}.json'
    if path.exists():
        if not receipt.exists() or base.sha(path)!=base.read_json(receipt)['sha256']:raise ValueError('Opening cache checksum')
        with np.load(path,allow_pickle=False) as z:
            if z['keys'].astype(str).tolist()!=keys:raise ValueError('Opening cache key binding')
            arr=z['codes'].copy()
        return arr,keys,{'games':len(keys),'hits':len(keys),'misses':0}
    arr=np.full((len(keys),40),65535,np.uint16);pos={k:i for i,k in enumerate(keys)}
    batches=[[(k,games[k].sgf) for k in keys[i:i+128]] for i in range(0,len(keys),128)]
    if workers==1:
        iterator=map(parse_opening_batch,batches)
        for rows in iterator:
            for key,code in rows:arr[pos[key]]=code
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rows in pool.map(parse_opening_batch,batches,chunksize=1):
                for key,code in rows:arr[pos[key]]=code
    with path.open('xb') as f:np.savez_compressed(f,keys=np.asarray(keys,dtype='S64'),codes=arr)
    write(receipt,{'sha256':base.sha(path),'schema':'sgfmill1.1.1-first40-v1','games':len(keys)})
    return arr,keys,{'games':len(keys),'hits':0,'misses':len(keys)}

def load_bundle(path):
    p=Path(path).resolve();manifest=base.read_json(p/'manifest.json')
    if manifest['schema']!='go-F-E-sequence-v1' or base.sha(p/'manifest.json')!=base.read_json(p/'manifest.sha256.json')['manifest.json']:raise ValueError('F manifest differs')
    if manifest['recipe']!={'E_rank_weight':.5,'sequence_rank_weight':.5,'rrf_k':60,'alpha':20.,'first_total_plies':40,'bins_per_group':BINS}:raise ValueError('F frozen recipe differs')
    for name,h in manifest['files'].items():
        target=(p/name).resolve()
        if not target.is_relative_to(p) or base.sha(target)!=h:raise ValueError('F bundle payload differs')
    for name,h in manifest['code_hashes'].items():
        if base.sha(ROOT/name)!=h:raise ValueError('F runtime changed')
    for name,version in manifest['runtime'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('F runtime version differs')
    load_E(p/'E')
    return p,manifest

def run_job(job,output):
    out=Path(output).resolve();cache=Path(job['cache']).resolve();sequence_cache=Path(job['sequence_cache']).resolve()
    if out.exists() or not out.is_relative_to(ROOT) or any(not p.is_relative_to(ROOT) or p.is_relative_to(out) for p in (cache,sequence_cache)):raise ValueError('New project output and separate caches required')
    if job.get('method')!='F_E_sequence_RRF60_equal' or any(k in job for k in ('alpha','weights','weight','prior_scale','noise_scale')):raise ValueError('Frozen F recipe only')
    bundle,manifest=load_bundle(job['bundle']);games={};pools=[]
    _,em=load_E(bundle/'E');parser={'manifest':{'parser_sha256':em['parser_sha256'],'runtime':em['runtime']},'banned_raw':set(),'banned_players':set()}
    for spec in job['pools']:
        pool=base.load_pool(spec,games,parser)
        pools.append(canonical_pool(pool,base.read_json(spec['order_manifest']) if 'order_manifest' in spec else None))
    for game in games.values():
        if game.color!='B' or not re.match(r'^\(;SZ\[19\]HD\[[01]\];',game.sgf):raise ValueError('F requires B and explicitHD0/1')
    out.mkdir(parents=True);write(out/'job.json',job);started=time.perf_counter()
    try:
        ejob=dict(job,bundle=str(bundle/'E'),method='E_D_PLDA_raw_RRF60_equal');run_E(ejob,out/'E')
        # E first performs its complete strict SGF parse/cache verification on these same games.
        arr,keys,cache_report=openings(games,sequence_cache,job.get('workers',4));positions={k:i for i,k in enumerate(keys)}
        with np.load(bundle/'background.npz',allow_pickle=False) as z:model=FrozenSequence(z['background'],manifest['variant'])
        with threadpool_limits(limits=2):x=design(arr,manifest['variant'])
        files=[];diagnostics={}
        for pool in pools:
            mapping={'candidate_ids':pool.candidate_ids,'question_ids':[q['question_id'] for q in pool.questions]}
            if mapping!=base.read_json(out/f'E/{pool.name}/mapping.json'):raise ValueError('F/E axes differ')
            ss=[[positions[k] for k in pool.supports[p]] for p in pool.candidate_ids];qs=[[positions[k] for k in q['keys']] for q in pool.questions]
            with threadpool_limits(limits=2):sequence=model.score(x,ss,qs,keys)
            es=np.load(out/f'E/{pool.name}/E_scores.npy');score=fixed_rrf60(es,sequence)
            pd=out/pool.name;pd.mkdir();np.save(pd/'F_scores.npy',score,allow_pickle=False);np.save(pd/'sequence_scores.npy',sequence,allow_pickle=False)
            write(pd/'mapping.json',mapping);base.write_top5(pd/'top5_selected.csv',pool,score);files.append(pd/'top5_selected.csv')
            diagnostics[pool.name]={'questions':len(qs),'top1_different_vs_E':int((np.argmax(score,1)!=np.argmax(es,1)).sum()),'disagreement_not_accuracy':True}
        base.combine_files(files,out/'submission_selected.csv');check=base.check_output(out/'submission_selected.csv',pools)
        for pool in pools:
            for kind in ('candidates','queries'):
                if base.sha(pool.source[kind+'_csv'])!=pool.source[kind+'_sha256']:raise ValueError('F source CSV changed')
        load_bundle(bundle)
        result={'status':'complete','method':job['method'],'variant':manifest['variant'],'format_check':check,
            'queries':sum(len(p.questions) for p in pools),'seconds':time.perf_counter()-started,'opening_cache':cache_report,
            'diagnostics':diagnostics,'CSV_only_end_to_end_including_frozen_E':True,'training_or_adaptation':False,
            'GT_or_training_or_audit_loaded':False,'retired_details_read':False,'uploaded':False,'official_score':None}
        write(out/'SUMMARY.json',result);write(out/'STATUS.json',{'status':'complete'})
        return result
    except BaseException as error:
        write(out/'STATUS.json',{'status':'failed_preserved','error':str(error),'traceback':traceback.format_exc()});raise

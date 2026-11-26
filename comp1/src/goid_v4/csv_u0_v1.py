"""Frozen U0 CSV inference. No audit/GT loader, optimizer, fitting or submission.

The payload contains only two inference MLPs/statistics and the fixed U0 head.
Retired membership lists are deliberately absent; validation inputs use pinned
existing development CSV provenance, never historical outer or retired files.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import hashlib,importlib.metadata,json,multiprocessing,os,sqlite3,sys,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from goid import inference as base,fixed_batch,row_projection
from goid.sgf_features import analyze_sgf,FEATURE_DIM
from .p1 import compose_game_features
from .scoring import fixed_fusion

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='goid-u0-csv-v4.1'
METHODS=('u0','refit_color','refit_original','v3_color','v3_original')
SET_POLICY={'batch_sets':32,'padded_games':100,'dtype':'float32','tf32':False,'autocast':False,
    'game_order':'content-key lexical','new_rank_weight':.25,'rrf_k':60,
    'default_identity_order':'lexical candidate IDs and question IDs; explicit order binding overrides',
    'legacy_game_order':'bound original v3 key order when supplied; otherwise content-key lexical'}

def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def digest_object(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_bundle(path):
    p=Path(path).resolve();m=base.read_json(p/'manifest.json')
    if m.get('schema')!=SCHEMA or base.sha(p/'manifest.json')!=base.read_json(p/'manifest.sha256.json')['manifest.json']:raise ValueError('Bundle manifest invalid')
    if m['set_policy']!=SET_POLICY or m['game_policy']!=fixed_batch.POLICY or m['projection_policy']!=row_projection.POLICY:raise ValueError('Pinned numerical policy differs')
    if list(sys.version_info[:2])!=m['python_major_minor']:raise ValueError('Python version differs')
    for name,version in m['runtime'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Runtime version differs: '+name)
    for rel,h in m['files'].items():
        f=(p/rel).resolve()
        if not f.is_relative_to(p) or base.sha(f)!=h:raise ValueError('Bundle payload checksum: '+rel)
    for rel,h in m['code_hashes'].items():
        f=(ROOT/rel).resolve()
        if not f.is_relative_to(ROOT) or base.sha(f)!=h:raise ValueError('Pinned source checksum: '+rel)
    if m['parser_sha256']!=base.sha(ROOT/'src/goid/sgf_features.py'):raise ValueError('Parser differs')
    return {'path':p,'manifest':m,'manifest_sha256':base.sha(p/'manifest.json'),
        'banned_raw':set(),'banned_players':set(),'banned_canonical':set()}

def branch_bundle(bundle,branch):
    return {'path':bundle['path']/branch,'manifest':{'model_config':bundle['manifest']['model_config'],'synthetic_control':False}}

def canonical_pool(pool,binding=None,legacy=False):
    candidates=sorted(pool.candidate_ids);questions=sorted(pool.questions,key=lambda q:q['question_id'])
    if binding:
        if len(binding['candidate_ids'])!=len(set(binding['candidate_ids'])) or set(binding['candidate_ids'])!=set(candidates):raise ValueError('Bound candidate set mismatch')
        candidates=binding['candidate_ids']
        order=binding['question_ids']
        if len(order)!=len(set(order)) or not {q['question_id'] for q in questions}<=set(order):raise ValueError('Bound question set mismatch')
        position={q:i for i,q in enumerate(order)};questions=sorted(questions,key=lambda q:position[q['question_id']])
    supports={p:sorted(pool.supports[p]) for p in candidates};qs=[]
    for q in questions:qs.append(dict(q,keys=sorted(q['keys'])))
    if legacy and binding:
        for p in candidates:
            ids=binding['supports'][p]
            if len(ids)!=len(set(ids)) or set(ids)!=set(supports[p]):raise ValueError('Bound v3 support keys mismatch')
            supports[p]=ids
        for q in qs:
            ids=binding['queries'][q['question_id']]
            if len(ids)!=len(set(ids)) or set(ids)!=set(q['keys']):raise ValueError('Bound v3 query keys mismatch')
            q['keys']=ids
    return base.Pool(pool.name,candidates,supports,qs,pool.source)

def parse_batch(batch):
    out=[]
    for g in batch:
        try:
            r=analyze_sgf(g.sgf,g.color,include_features=True)
            x=np.asarray(r['features'],np.float32)
            if r['board_size']!=19 or x.shape!=(FEATURE_DIM,) or not np.isfinite(x).all():raise ValueError('Invalid 19x19 features')
            meta={k:r[k] for k in ('canonical_hash','move_hash','mainline_moves','target_moves')}
            out.append((g.key,x.tobytes(),meta,None))
        except Exception as e:out.append((g.key,None,None,f'{type(e).__name__}: {e}'))
    return out

class Cache:
    def __init__(self,path):
        if Path(path).exists():
            probe=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
            try:
                tables={r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if tables!={'u0_v1'}:raise ValueError('Refuse to modify an existing foreign/legacy cache')
            finally:probe.close()
        self.db=sqlite3.connect(path);self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS u0_v1 (key TEXT PRIMARY KEY,raw TEXT NOT NULL,color TEXT NOT NULL,meta TEXT NOT NULL,meta_sha TEXT NOT NULL,vector_sha TEXT NOT NULL,vector BLOB NOT NULL)');self.db.commit()
    @staticmethod
    def check(row,g):
        key,raw,color,text,mh,vh,blob=row
        meta=json.loads(text)
        if key!=g.key or raw!=g.raw_hash or color!=g.color or digest_object(meta)!=mh or hashlib.sha256(blob).hexdigest()!=vh or len(blob)!=FEATURE_DIM*4:raise ValueError('Feature/metadata cache checksum mismatch')
        if set(meta)!={'canonical_hash','move_hash','mainline_moves','target_moves'}:raise ValueError('Cache metadata schema mismatch')
        if any(not isinstance(meta[k],str) or len(meta[k])!=64 for k in ('canonical_hash','move_hash')):raise ValueError('Cache hash schema mismatch')
        if any(type(meta[k]) is not int or meta[k]<0 for k in ('mainline_moves','target_moves')):raise ValueError('Cache count schema mismatch')
        x=np.frombuffer(blob,np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite cache vector')
        return x,meta

def extract(games,cache_path,output,workers):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('workers must be1..8')
    keys=sorted(games);positions={k:i for i,k in enumerate(keys)};found=set();metadata={}
    matrix=np.empty((len(keys),FEATURE_DIM),np.float32);cache_path.parent.mkdir(parents=True,exist_ok=True)
    c=Cache(cache_path);start=time.perf_counter()
    try:
        for startrow in range(0,len(keys),800):
            batch=keys[startrow:startrow+800]
            for row in c.db.execute(f'SELECT * FROM u0_v1 WHERE key IN ({",".join("?" for _ in batch)})',batch):
                key=row[0];matrix[positions[key]],metadata[key]=c.check(row,games[key]);found.add(key)
        missing=[games[k] for k in keys if k not in found];parsed=0
        def accept(results):
            nonlocal parsed
            rows=[]
            for key,blob,meta,error in results:
                if error:
                    with (output/'parse_errors.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'key':key,'error':error})+'\n')
                    raise ValueError('SGF parse failed; no dropped games')
                row=(key,games[key].raw_hash,games[key].color,json.dumps(meta,sort_keys=True),digest_object(meta),hashlib.sha256(blob).hexdigest(),blob)
                matrix[positions[key]],metadata[key]=c.check(row,games[key]);rows.append(row)
            c.db.executemany('INSERT INTO u0_v1 VALUES (?,?,?,?,?,?,?)',rows);c.db.commit();parsed+=len(results)
            if parsed%4096==0 or parsed==len(missing):print(json.dumps({'stage':'parse','parsed':parsed,'misses':len(missing)}),flush=True)
        batches=iter([missing[i:i+128] for i in range(0,len(missing),128)])
        if workers==1:
            for batch in batches:accept(parse_batch(batch))
        elif missing:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
                pending=set()
                for _ in range(workers*2):
                    batch=next(batches,None)
                    if batch is not None:pending.add(executor.submit(parse_batch,batch))
                while pending:
                    done,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        accept(future.result());batch=next(batches,None)
                        if batch is not None:pending.add(executor.submit(parse_batch,batch))
        with (output/'feature_index.jsonl').open('x',encoding='utf-8') as f:
            for key,x in zip(keys,matrix):f.write(json.dumps({'key':key,**metadata[key],'vector_sha256':hashlib.sha256(x.tobytes()).hexdigest()})+'\n')
        report={'unique_games':len(keys),'hits':len(found),'misses':len(missing),'actually_parsed':parsed,'workers':workers,'elapsed_seconds':time.perf_counter()-start}
        write(output/'cache_report.json',report)
        return matrix,keys,metadata,report
    finally:c.db.close()

class FrozenU0(nn.Module):
    def __init__(self,floor):
        super().__init__();self.floor=floor
        self.register_buffer('prior_variance',torch.zeros(128))
        self.project=nn.Sequential(nn.Linear(1728,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1))
        self.mean_head=nn.Linear(133,128);self.var_head=nn.Linear(133,128)
    def forward(self,x,colors,mask):
        n=mask.sum(1);black=((colors==0)&mask).sum(1);white=n-black
        if not (n>0).all():raise ValueError('Empty set')
        counts=torch.stack((torch.log1p(n),torch.log1p(black),torch.log1p(white),(black==0).float(),(white==0).float()),1).to(x.dtype)
        h=self.project(x).masked_fill(~mask[:,:,None],0).sum(1)/n[:,None];h=torch.cat((h,counts),1)
        mean=F.normalize(self.mean_head(h),dim=-1,eps=1e-12);var=F.softplus(self.var_head(h))/n[:,None]+self.floor
        return mean,torch.where((n==1)[:,None],self.prior_variance[None],var)

def encode_sets(model,x,colors,sets):
    means=[];variances=[]
    for start in range(0,len(sets),32):
        real=sets[start:start+32];ids=np.zeros((32,100),np.int64);mask=np.zeros_like(ids,bool);mask[:,0]=True
        for i,keys in enumerate(real):
            if not 1<=len(keys)<=100 or len(set(keys))!=len(keys):raise ValueError('Invalid set cardinality')
            keys=sorted(keys);ids[i,:len(keys)]=keys;mask[i]=False;mask[i,:len(keys)]=True
        ti=torch.from_numpy(ids).to(x.device);tm=torch.from_numpy(mask).to(x.device)
        xx=x[ti].masked_fill(~tm[:,:,None],0);cc=colors[ti]
        if len(real)<32:xx[len(real):]=0;cc[len(real):]=0
        with torch.inference_mode():m,v=model(xx,cc,tm)
        means.append(m[:len(real)].cpu().numpy());variances.append(v[:len(real)].cpu().numpy())
    return np.concatenate(means),np.concatenate(variances)

def scores_from_embeddings(qm,qv,sm,sv,scale):
    sm=np.asarray(sm,np.float64);sv=np.asarray(sv,np.float64);scores=np.empty((len(qm),len(sm)),np.float64)
    for i in range(len(qm)):
        var=(np.asarray(qv[i],np.float64)+sv)*scale;diff=np.asarray(qm[i],np.float64)-sm
        if not np.isfinite(var).all() or (var<=0).any():raise ValueError('Invalid variance')
        scores[i]=-.5*np.sum(diff*diff/var+np.log(var),axis=1)
    return scores

def run_job(job,output):
    output=Path(output).resolve();cache=Path(job['cache']).resolve()
    if output.exists():raise FileExistsError('Preserve previous outputs; choose a new directory')
    if not output.is_relative_to(ROOT) or not cache.is_relative_to(ROOT):raise ValueError('Output/cache must stay inside project')
    if cache.is_relative_to(output) or cache.is_relative_to(Path(job['bundle']).resolve()):raise ValueError('Cache must be separate from output/model bundle')
    if any(k in job for k in ('alpha','seed','batch_size','batch_rows','floor','variance_scale','query_batch_rows')):raise ValueError('Frozen model/numerics are not job options')
    if job.get('device','cuda')!='cuda':raise ValueError('Real runtime requires local RTX3060')
    method=job.get('method','u0')
    if method not in METHODS:raise ValueError('Unknown method')
    bundle=load_bundle(job['bundle']);games={};pools=[];legacy=[];bindings={};input_hashes={}
    for spec in job['pools']:
        if spec.get('expected_candidates',400)!=400 or spec.get('reference_games',100)!=100:raise ValueError('This frozen real model requires400 candidates x100 supports')
        pool=base.load_pool(spec,games,bundle);binding=None
        if 'order_manifest' in spec:
            path=Path(spec['order_manifest']);binding=base.read_json(path);input_hashes[str(path.resolve())]=base.sha(path)
        pools.append(canonical_pool(pool,binding));legacy.append(canonical_pool(pool,binding,legacy=True));bindings[pool.name]=binding
    qids=[q['question_id'] for p in pools for q in p.questions]
    if not pools or len({p.name for p in pools})!=len(pools) or len(qids)!=len(set(qids)):raise ValueError('Repeated pool/question IDs')
    output.mkdir(parents=True);start=time.perf_counter()
    write(output/'job.json',job);write(output/'input_manifest.json',{'bundle_sha256':bundle['manifest_sha256'],'pools':[p.source for p in pools],'order_manifests':input_hashes})
    try:
        raw,keys,meta,cache_report=extract(games,cache,output,job.get('workers',4))
        positions={k:i for i,k in enumerate(keys)};evidence={};vectors={};available=[]
        branches=('refit','v3') if job.get('include_comparisons',True) else (('v3',) if method.startswith('v3_') else ('refit',))
        for branch in branches:
            vectors[branch],evidence[branch]=fixed_batch.encode_features(raw,branch_bundle(bundle,branch),'cuda')
        need_u0=method=='u0' or job.get('include_comparisons',True)
        head=None
        if need_u0:
            counts=np.asarray([meta[k]['target_moves'] for k in keys],np.float64)
            stages=np.rint(raw[:,9:12].astype(np.float64)*counts[:,None])
            features=compose_game_features(*vectors['refit'],raw[:,:12],[games[k].color for k in keys],[meta[k]['mainline_moves'] for k in keys],stages)
            x=torch.from_numpy(features).to('cuda');colors=torch.tensor([0 if games[k].color=='B' else 1 for k in keys],device='cuda')
            saved=torch.load(bundle['path']/'u0/model.pt',map_location='cpu',weights_only=True)
            frozen=bundle['manifest']['candidate']
            if saved['seed']!=31415 or saved['steps']!=5000 or saved['floor']!=frozen['floor'] or saved['variance_scale']!=frozen['variance_scale']:raise ValueError('Head metadata differs from freeze')
            torch.backends.mha.set_fastpath_enabled(False)
            head=FrozenU0(frozen['floor']).to('cuda');head.load_state_dict(saved['state_dict'],strict=True);head.eval().requires_grad_(False)
            before=fixed_batch.model_digest(head)
        selected_files=[]
        for pool,oldpool in zip(pools,legacy):
            pd=output/pool.name;pd.mkdir();matrices={}
            for branch in branches:
                usepool=oldpool if branch=='v3' else pool
                a,b,ainfo=row_projection.component_scores(vectors[branch][0],usepool,games,positions)
                c,d,dinfo=row_projection.component_scores(vectors[branch][1],usepool,games,positions)
                original=base.rrf(a,c);color,_,fallback=base.choose_scores(b,d,original,'color')
                matrices[branch+'_original']=original;matrices[branch+'_color']=color
                np.savez(pd/f'{branch}_color_counts.npz',**ainfo,numeric_row_fallback=fallback)
                for name,value in [('stat_original',a),('stat_color',b),('mlp_original',c),('mlp_color',d)]:np.save(pd/f'{branch}_{name}.npy',value,allow_pickle=False)
            if need_u0:
                sm,sv=encode_sets(head,x,colors,[[positions[k] for k in pool.supports[p]] for p in pool.candidate_ids])
                qm,qv=encode_sets(head,x,colors,[[positions[k] for k in q['keys']] for q in pool.questions])
                new=scores_from_embeddings(qm,qv,sm,sv,saved['variance_scale'])
                matrices['u0_single']=new;matrices['u0']=np.stack([fixed_fusion(b,s) for b,s in zip(matrices['refit_color'],new)])
            for name,scores in matrices.items():
                np.save(pd/f'{name}_scores.npy',scores,allow_pickle=False);base.write_top5(pd/f'top5_{name}.csv',pool,scores)
            base.write_top5(pd/'top5_selected.csv',pool,matrices[method]);selected_files.append(pd/'top5_selected.csv')
            write(pd/'mapping.json',{'candidate_ids':pool.candidate_ids,'question_ids':[q['question_id'] for q in pool.questions]})
            available=list(matrices)
        if head is not None:
            after=fixed_batch.model_digest(head)
            if before!=after or any(p.grad is not None for p in head.parameters()):raise ValueError('U0 state mutated/gradients')
            evidence['u0']={'before':before,'after':after,'unchanged':True,'requires_grad':False,'set_policy':SET_POLICY}
        base.combine_files(selected_files,output/'submission_selected.csv');checked=base.check_output(output/'submission_selected.csv',pools)
        for name in available:
            base.combine_files([output/p.name/f'top5_{name}.csv' for p in pools],output/f'submission_{name}.csv')
            base.check_output(output/f'submission_{name}.csv',pools)
        # Integrity and input sources checked again before complete status.
        again=load_bundle(job['bundle'])
        if again['manifest_sha256']!=bundle['manifest_sha256']:raise ValueError('Bundle changed during inference')
        for source in (p.source for p in pools):
            for kind in ('candidates','queries'):
                if base.sha(source[kind+'_csv'])!=source[kind+'_sha256']:raise ValueError('CSV changed during inference')
        for path,h in input_hashes.items():
            if base.sha(path)!=h:raise ValueError('Order binding changed during inference')
        write(output/'no_training_evidence.json',{'encoders':evidence,'optimizer_constructed':False,'backward_calls':0,'fit_calls':0,'GT_or_audit_loaded':False})
        result={'status':'complete','method':method,'queries':len(qids),'pools':len(pools),'elapsed_seconds':time.perf_counter()-start,
            'cache':cache_report,'submission_check':checked,'available_methods':available,'no_submission':True,
            'scope':'CSV inference/engineering only; no new quality or pristine claim','original_retired_files_loaded':False}
        write(output/'summary.json',result);write(output/'status.json',{'status':'complete','no_submission':True})
        write(output/'output_sha256.json',{p.relative_to(output).as_posix():base.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        return result
    except BaseException as e:
        write(output/'status.json',{'status':'failed_preserved','error':str(e),'type':type(e).__name__,'traceback':traceback.format_exc(),'no_submission':True});raise

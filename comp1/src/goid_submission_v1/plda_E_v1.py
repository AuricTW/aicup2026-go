"""CSV-only frozen E inference. No training data, evaluation labels or upload API."""
from pathlib import Path
import importlib.metadata,json,re,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits
from goid import inference as base
from goid_v4.csv_u0_v1 import extract,canonical_pool,write
from goid_v4.plda_metric_v1 import PLDA,histograms,summarize
from .ensemble_D_v2 import load_bundle as load_D,run_job as run_D,fixed_rrf60
ROOT=Path(__file__).resolve().parents[2]

def load_bundle(path):
    p=Path(path).resolve();m=base.read_json(p/'manifest.json')
    if m['schema']!='go-E-D-PLDA-v1' or base.sha(p/'manifest.json')!=base.read_json(p/'manifest.sha256.json')['manifest.json']:raise ValueError('E manifest invalid')
    if m['recipe']!={'D_rank_weight':.5,'PLDA_rank_weight':.5,'rrf_k':60}:raise ValueError('E frozen recipe differs')
    for rel,h in m['files'].items():
        f=(p/rel).resolve()
        if not f.is_relative_to(p) or base.sha(f)!=h:raise ValueError('E payload hash differs')
    for rel,h in m['code_hashes'].items():
        if base.sha(ROOT/rel)!=h:raise ValueError('E runtime code changed')
    for name,v in m['runtime'].items():
        if importlib.metadata.version(name)!=v:raise ValueError('E runtime version differs')
    load_D(p/'D')
    return p,m

def run_job(job,output):
    out=Path(output).resolve();cache=Path(job['cache']).resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not cache.is_relative_to(ROOT) or cache.is_relative_to(out):raise ValueError('Use new project output and separate cache')
    if job.get('method')!='E_D_PLDA_raw_RRF60_equal' or any(k in job for k in ('alpha','weight','prior_scale','noise_scale')):raise ValueError('Frozen E recipe only')
    bundle,manifest=load_bundle(job['bundle']);games={};pools=[]
    parser={'manifest':{'parser_sha256':manifest['parser_sha256'],'runtime':manifest['runtime']},'banned_raw':set(),'banned_players':set()}
    for spec in job['pools']:
        if spec.get('expected_candidates',400)!=400 or spec.get('reference_games',100)!=100:raise ValueError('E400x100 contract')
        pool=base.load_pool(spec,games,parser)
        pools.append(canonical_pool(pool,base.read_json(spec['order_manifest']) if 'order_manifest' in spec else None))
    if len({p.name for p in pools})!=len(pools) or len({q['question_id'] for p in pools for q in p.questions})!=sum(len(p.questions) for p in pools):raise ValueError('E repeated pool/query')
    for g in games.values():
        if g.color!='B' or not re.match(r'^\(;SZ\[19\]HD\[[01]\];',g.sgf):raise ValueError('E requires explicit B HD0/1, no guessed color/handicap')
    out.mkdir(parents=True);t=time.perf_counter();write(out/'job.json',job)
    try:
        djob=dict(job,bundle=str(bundle/'D'),method='D_ensemble');run_D(djob,out/'D')
        raw,keys,metadata,cache_report=extract(games,cache,out,job.get('workers',4));positions={k:i for i,k in enumerate(keys)}
        x=histograms(raw,base.read_json(bundle/'feature_schema.json'),sqrt=True)
        with threadpool_limits(limits=4):z=PLDA(bundle/'PLDA_raw.npz').transform(x)
        model=PLDA(bundle/'PLDA_raw.npz');files=[];diag={}
        for pool in pools:
            mapping={'candidate_ids':pool.candidate_ids,'question_ids':[q['question_id'] for q in pool.questions]}
            if mapping!=base.read_json(out/f'D/{pool.name}/mapping.json'):raise ValueError('E/D matrix alignment differs')
            si=[[positions[k] for k in pool.supports[p]] for p in pool.candidate_ids]
            qi=[[positions[k] for k in q['keys']] for q in pool.questions]
            qm,qn=summarize(z,qi);sm,sn=summarize(z,si);plda=model.score(qm,qn,sm,sn)
            ds=np.load(out/f'D/{pool.name}/D_scores.npy',allow_pickle=False);score=fixed_rrf60(ds,plda)
            pd=out/pool.name;pd.mkdir();np.save(pd/'E_scores.npy',score,allow_pickle=False);np.save(pd/'PLDA_scores.npy',plda,allow_pickle=False)
            write(pd/'mapping.json',mapping);base.write_top5(pd/'top5_selected.csv',pool,score);files.append(pd/'top5_selected.csv')
            diag[pool.name]={'questions':len(qi),'top1_different_vs_D':int((np.argmax(score,1)!=np.argmax(ds,1)).sum()),
                'disagreement_only_not_accuracy':True}
        base.combine_files(files,out/'submission_selected.csv');check=base.check_output(out/'submission_selected.csv',pools)
        for pool in pools:
            for k in ('candidates','queries'):
                if base.sha(pool.source[k+'_csv'])!=pool.source[k+'_sha256']:raise ValueError('E source CSV changed')
        load_bundle(bundle)
        result={'status':'complete','method':job['method'],'format_check':check,'queries':sum(len(p.questions) for p in pools),
            'seconds':time.perf_counter()-t,'cache':cache_report,'diagnostics':diag,
            'raw_CSV_end_to_end_including_frozen_D':True,'training_or_tuning':False,'GT_or_audit_loaded':False,
            'retired_details_read':False,'uploaded':False,'official_score':None}
        write(out/'SUMMARY.json',result);write(out/'STATUS.json',{'status':'complete'})
        write(out/'OUTPUT_SHA256.json',{p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file()})
        return result
    except BaseException as e:
        write(out/'STATUS.json',{'status':'failed_preserved','error':str(e),'traceback':traceback.format_exc()});raise

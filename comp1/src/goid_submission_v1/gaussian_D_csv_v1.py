"""Inference-only fourth plan. Reuse the fixed U1 math and numerical policy.

The new full-training weights have no independent self-test accuracy estimate.
"""
from pathlib import Path
import importlib.metadata, time, traceback
import numpy as np
from goid import inference as base, fixed_batch
from goid_v4.csv_u0_v1 import write, extract, canonical_pool
from goid_v4.p1 import compose_game_features
from .gaussian_csv import FrozenGaussian, summaries

ROOT=Path(__file__).resolve().parents[2]


def load_bundle(path):
    p=Path(path).resolve();m=base.read_json(p/'manifest.json')
    if m.get('schema')!='go-expanded-U1-D-v1' or base.sha(p/'manifest.json')!=base.read_json(p/'manifest.sha256.json')['manifest.json']:
        raise ValueError('D manifest invalid')
    for name,h in m['files'].items():
        target=(p/name).resolve()
        if not target.is_relative_to(p) or base.sha(target)!=h:raise ValueError('D payload changed')
    for name,h in m['code_hashes'].items():
        if base.sha(ROOT/name)!=h:raise ValueError('Pinned D runtime changed')
    for name,version in m['runtime'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Runtime changed')
    return {'path':p,'manifest':m,'banned_raw':set(),'banned_players':set(),'banned_canonical':set()}


def run_job(job,output):
    out=Path(output).resolve();cache=Path(job['cache']).resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not cache.is_relative_to(ROOT) or cache.is_relative_to(out):
        raise ValueError('Use new project output and separate cache')
    bundle=load_bundle(job['bundle'])
    if job.get('method')!='D_expanded_U1' or any(k in job for k in ('prior_scale','noise_scale','dimension','alpha')):
        raise ValueError('Fixed D method and scales only')
    games={};pools=[]
    for spec in job['pools']:
        if spec.get('expected_candidates',400)!=400 or spec.get('reference_games',100)!=100:raise ValueError('400x100 required')
        pool=base.load_pool(spec,games,bundle)
        binding=base.read_json(spec['order_manifest']) if 'order_manifest' in spec else None
        pools.append(canonical_pool(pool,binding))
    qids=[q['question_id'] for p in pools for q in p.questions]
    if not pools or len(qids)!=len(set(qids)) or len({p.name for p in pools})!=len(pools):raise ValueError('Pool/qid mismatch')
    out.mkdir(parents=True);t=time.perf_counter();write(out/'job.json',job)
    try:
        raw,keys,metadata,cachestats=extract(games,cache,out,job.get('workers',4));positions={k:i for i,k in enumerate(keys)}
        encoder={'path':bundle['path']/'base','manifest':{'model_config':bundle['manifest']['model_config'],'synthetic_control':False}}
        (stat,mlp),evidence=fixed_batch.encode_features(raw,encoder,'cuda')
        count=np.asarray([metadata[k]['target_moves'] for k in keys],np.float64)
        stages=np.rint(raw[:,9:12].astype(np.float64)*count[:,None])
        colors=np.asarray([int(games[k].color=='W') for k in keys],np.int64)
        features=compose_game_features(stat,mlp,raw[:,:12],[games[k].color for k in keys],
            [metadata[k]['mainline_moves'] for k in keys],stages)
        gaussian=FrozenGaussian(bundle['path']/'u1/model.npz');latent=gaussian.transform(features);paths=[]
        for pool in pools:
            scores=gaussian.score(summaries(latent,colors,[[positions[k] for k in q['keys']] for q in pool.questions]),
                summaries(latent,colors,[[positions[k] for k in pool.supports[p]] for p in pool.candidate_ids]))
            pd=out/pool.name;pd.mkdir();np.save(pd/'D_scores.npy',scores,allow_pickle=False)
            base.write_top5(pd/'top5_selected.csv',pool,scores);paths.append(pd/'top5_selected.csv')
            write(pd/'mapping.json',{'candidate_ids':pool.candidate_ids,'question_ids':[q['question_id'] for q in pool.questions]})
        base.combine_files(paths,out/'submission_selected.csv');check=base.check_output(out/'submission_selected.csv',pools)
        load_bundle(job['bundle'])
        result={'status':'complete','queries':len(qids),'method':'D_expanded_U1',
            'elapsed_seconds':time.perf_counter()-t,'cache':cachestats,
            'PCA_policy':'one_game_FP64_GEMV_128x1728','no_training_or_tuning':True,
            'no_upload':True,'original_cohort_not_read':True,'format_check':check,'encoder_evidence':evidence,
            'limitation':'All official training data refit; recycled self-tests cannot estimate generalization. Method comparison is exposed development only.'}
        write(out/'summary.json',result);write(out/'status.json',{'status':'complete'})
        write(out/'output_sha256.json',{p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file()})
        return result
    except BaseException as e:
        write(out/'status.json',{'status':'failed_preserved','error':str(e),'traceback':traceback.format_exc()});raise

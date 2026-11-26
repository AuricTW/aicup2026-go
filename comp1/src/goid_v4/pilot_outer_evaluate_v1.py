"""Single prospective INTERNAL reveal, after the three U0 checkpoints are fixed.

All normalization/MLP/head states exclude the outer identities. Their historical
exposure in older MLP training means this is not original pristine confirmation.
"""
from __future__ import annotations
import csv,json,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from .pilot_outer_v1 import ROOT,OUT
from .pilot_outer_heads_v2 import HEAD
from .pilot_scope_v1 import sha,write
from .pilot_evaluate_v1 import load_checkpoint,scaled_numeric_check,ranks_and_values,infer_sets
from .evaluation import PairedEvent,paired_cluster_intervals
from .scoring import rrf,fixed_fusion
from .gates import g1,g2,SEEDS
from .artifacts import ContractError
from goid.row_projection import component_scores


def frozen_checkpoints():
    models=[]
    for seed in SEEDS:
        p=HEAD/'models'/f'U0_mean_count_s{seed}'
        r=json.loads((p/'result.json').read_text(encoding='utf-8'))
        if r['status']!='complete_fit_only' or r['steps']!=5000 or sha(p/'model.pt')!=r['model_sha256']:
            raise ContractError('Unfrozen checkpoint')
        models.append((seed,p,r['model_sha256']))
    base=json.loads((OUT/'baseline_refit_result.json').read_text(encoding='utf-8'))
    if base['steps']!=5000 or base['outer_players_excluded']!=800 or base['old_weights_loaded']:
        raise ContractError('Baseline reset boundary failed')
    for name,field in [('model_state.pt','model_sha256'),('fit_transform.npz','fit_transform_sha256')]:
        if sha(OUT/'base_bundle'/name)!=base[field]:raise ContractError('Baseline state changed')
    return models


def check_plan(records,plans):
    allplayers=set()
    for p in plans:
        candidates=p['candidate_order']
        if len(candidates)!=400 or len(set(candidates))!=400 or set(candidates)&allplayers:
            raise ContractError('Two disjoint400-way banks required')
        allplayers.update(candidates);all_support=set()
        for player,ids in zip(candidates,p['supports']):
            if len(ids)!=100 or len(set(ids))!=100 or all_support&set(ids):raise ContractError('Support count/key overlap')
            if any(records[i]['player']!=player or not records[i]['part'].startswith('outer') for i in ids):raise ContractError('Support role mismatch')
            all_support.update(ids)
        qs={}
        for q in p['questions']:
            if len(q['indices'])!=q['q'] or len(set(q['indices']))!=q['q'] or all_support&set(q['indices']):raise ContractError('Query count or support overlap')
            if any(records[i]['player']!=q['player'] for i in q['indices']):raise ContractError('Query membership error')
            qs.setdefault(q['player'],{})[q['q']]=q['indices']
        if len(p['questions'])!=1200 or set(qs)!=set(candidates):raise ContractError('Question denominator error')
        for sets in qs.values():
            if set(sets)!={5,10,20} or sets[20][:10]!=sets[10] or sets[10][:5]!=sets[5]:raise ContractError('Nested query plan changed')
    return len(allplayers)


def main():
    start=time.perf_counter();models=frozen_checkpoints()
    cfgpath=ROOT/'configs/research_v4/p1_u0_outer_v1_20261026.json'
    config=json.loads(cfgpath.read_text(encoding='utf-8'))
    power=json.loads((OUT/'power_plan.json').read_text(encoding='utf-8'))
    if power['status']!='fixed_before_outer_scoring' or power['n_targets']!=800:raise ContractError('Power plan missing')
    scope=json.loads((OUT/'data/scope_receipt.json').read_text(encoding='utf-8'))
    if sha(OUT/'data/allowed_records.json')!=scope['allowed_records_sha256'] or scope['protocol_sha256']!=sha(cfgpath):
        raise ContractError('Frozen scope changed')
    registration={'status':'frozen_before_any_outer_quality_score','primary_method':'U0_mean_count',
        'primary_seed':31415,'replication_seeds':list(SEEDS),'fusion_new_rank_weight':.25,
        'family_size':3,'bootstrap_repetitions':10000,'bootstrap_seed':2026102602,
        'coverage_exact':'59/60','minimum_targets':800,'route':'prospective_blocked_internal',
        'protocol_sha256':sha(cfgpath),'power_plan_sha256':sha(OUT/'power_plan.json'),
        'plan_sha256':sha(OUT/'data/outer_plans.json'),'records_sha256':sha(OUT/'data/allowed_records.json'),
        'models':[{'seed':s,'sha256':h} for s,p,h in models],
        'source_sha256':sha(Path(__file__)),'head_adapter_sha256':sha(ROOT/'src/goid_v4/pilot_outer_heads_v2.py'),
        'baseline_numeric_policy':'v3 FP32 one-query GEMV; input games sorted by content-key/globalindex, FP64 RRF',
        'no_checkpoint_or_alpha_or_candidate_switch_after_reveal':True,
        'original_U2_primary_gate_remains_failed':True,'pristine_confirmation_claim':False}
    write(OUT/'analysis_registration.json',registration)
    reveal=OUT/'outer_reveal_v1';reveal.mkdir(exist_ok=False)
    from . import pilot_train_v1 as engine
    engine.RUN=HEAD;device=engine.runtime()
    fx,fr,_=engine.load_fit();fg=torch.from_numpy(fx).to(device)
    fc=torch.tensor([r['color'] for r in fr],dtype=torch.long,device=device)
    _,fs,fq,_,_=engine.fit_plan(fr);probes=[]
    with sdpa_kernel(SDPBackend.MATH):
        for seed,path,digest in models:
            model,scale=load_checkpoint(path,device)
            probes.append({'seed':seed,**scaled_numeric_check(model,scale,fg,fc,fs,fq)})
            del model
    write(reveal/'fit_only_actual_scale_numeric_probes.json',probes)
    del fg,fc,fx,fr;torch.cuda.empty_cache()
    d=OUT/'data';receipt=json.loads((d/'vector_receipt.json').read_text(encoding='utf-8'))
    for name in ('features.npy','stat.npy','mlp.npy'):
        if sha(d/name)!=receipt['hashes'][name]:raise ContractError('Refit vectors changed')
    records=json.loads((d/'allowed_records.json').read_text(encoding='utf-8'))
    plans=json.loads((d/'outer_plans.json').read_text(encoding='utf-8'))
    if check_plan(records,plans)!=800:raise ContractError('Insufficient targets')
    games={i:SimpleNamespace(color='B' if r['color']==0 else 'W') for i,r in enumerate(records)}
    positions={i:i for i in range(len(records))};baselines={};baseline_summary=[]
    baseline_start=time.perf_counter()
    for pool in plans:
        pd=reveal/pool['name'];pd.mkdir()
        # This inference object contains only observed support/query indices; no truth.
        inference=SimpleNamespace(candidate_ids=pool['candidate_order'],
            supports={p:sorted(ids) for p,ids in zip(pool['candidate_order'],pool['supports'])},
            questions=[{'keys':sorted(q['indices'])} for q in pool['questions']])
        components={}
        for branch in ('stat','mlp'):
            vectors=np.load(d/f'{branch}.npy',mmap_mode='r',allow_pickle=False)
            original,color,info=component_scores(vectors,inference,games,positions)
            components[branch]=(original,color)
            np.save(pd/f'{branch}_original_cosine.npy',original,allow_pickle=False)
            np.save(pd/f'{branch}_color_cosine.npy',color,allow_pickle=False)
            np.savez(pd/f'{branch}_counts.npz',**info)
        original=np.stack([rrf(a,b) for a,b in zip(components['stat'][0],components['mlp'][0])])
        color=np.stack([rrf(a,b) for a,b in zip(components['stat'][1],components['mlp'][1])])
        if not np.isfinite(original).all() or not np.isfinite(color).all():raise ContractError('Nonfinite baseline')
        np.save(pd/'original_rrf.npy',original,allow_pickle=False);np.save(pd/'color_rrf.npy',color,allow_pickle=False)
        targets=np.asarray([pool['candidate_order'].index(q['player']) for q in pool['questions']])
        qcounts=np.asarray([q['q'] for q in pool['questions']])
        ob,rb,vb=ranks_and_values(color,targets);_,rr,vr=ranks_and_values(original,targets)
        baselines[pool['name']]={'original':original,'color':color,'rank':rb,'value':vb,'target':targets,'q':qcounts}
        write(pd/'mapping.json',pool)
        for q in (5,10,20):
            mask=qcounts==q
            baseline_summary.append({'pool':pool['name'],'q':q,'color_score':float(vb[mask].mean()),
                'original_score':float(vr[mask].mean()),'color_top1':float((rb[mask]==1).mean()),'denominator':400})
    baseline_seconds=time.perf_counter()-baseline_start
    features=np.load(d/'features.npy',allow_pickle=False);x=torch.from_numpy(features).to(device)
    colors=torch.tensor([r['color'] for r in records],dtype=torch.long,device=device)
    results=[];perseed={};events=[];inference_times={}
    for seed,path,digest in models:
        model,scale=load_checkpoint(path,device);mdir=reveal/f'U0_s{seed}';mdir.mkdir();metrics=[]
        t=time.perf_counter()
        with (mdir/'predictions.csv').open('x',encoding='utf-8',newline='') as f:
            writer=None
            for pool in plans:
                b=baselines[pool['name']]
                new=infer_sets(model,x,colors,pool['supports'],[q['indices'] for q in pool['questions']],scale)
                fused=np.stack([fixed_fusion(a,c) for a,c in zip(b['color'],new)])
                fused_original=np.stack([fixed_fusion(a,c) for a,c in zip(b['original'],new)])
                order,rank,value=ranks_and_values(fused,b['target']);_,nr,nv=ranks_and_values(new,b['target'])
                pd=mdir/pool['name'];pd.mkdir()
                for name,array in [('new_scores',new),('fused_color_scores',fused),('fused_original_scores',fused_original)]:np.save(pd/f'{name}.npy',array,allow_pickle=False)
                for q in (5,10,20):
                    mask=b['q']==q;br=b['rank'][mask];cr=rank[mask]
                    row={'seed':seed,'pool':pool['name'],'q':q,'new_score':float(nv[mask].mean()),
                        'fused_color_score':float(value[mask].mean()),'fused_color_top1':float((cr==1).mean()),
                        'color_score_gain':float((value[mask]-b['value'][mask]).mean()),
                        'color_top1_gain':float(((cr==1).astype(float)-(br==1)).mean()),
                        'recovered':int(((br!=1)&(cr==1)).sum()),'regressed':int(((br==1)&(cr!=1)).sum())}
                    metrics.append(row);results.append(row)
                for i,q in enumerate(pool['questions']):
                    row={'pool':pool['name'],'question_id':q['question_id'],'player':q['player'],'q':q['q'],'seed':seed,
                        'color_rank':int(b['rank'][i]),'new_rank':int(nr[i]),'fused_color_rank':int(rank[i]),
                        'fused_color_score':float(value[i]),'color_score_gain':float(value[i]-b['value'][i]),
                        **{f'top{k+1}':pool['candidate_order'][j] for k,j in enumerate(order[i,:5])}}
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                    if seed==31415:events.append(PairedEvent(q['player'],pool['name'],seed,q['q'],q['question_id'],row['color_score_gain']))
        inference_times[str(seed)]=time.perf_counter()-t
        write(mdir/'metrics.json',metrics)
        perseed[seed]={q:{'score_gain':float(np.mean([r['color_score_gain'] for r in metrics if r['q']==q])),
            'top1_gain':float(np.mean([r['color_top1_gain'] for r in metrics if r['q']==q]))} for q in (5,10,20)}
        del model
        print(json.dumps({'stage':'outer_fixed_seed_complete','seed':seed,'seconds':inference_times[str(seed)]}),flush=True)
    ci=paired_cluster_intervals(events,family_size=3,repetitions=10000,seed=2026102602)
    point_gate=g1(perseed);outer_gate=g2(point_gate,ci,route='prospective_blocked_internal',minimum_targets=800,power_plan_locked=True,registered_family_size=3)
    aggregates=[]
    for q in (5,10,20):
        rows=[r for r in results if r['q']==q];base=[r for r in baseline_summary if r['q']==q]
        aggregates.append({'q':q,'color_baseline':float(np.mean([r['color_score'] for r in base])),
            'original_baseline':float(np.mean([r['original_score'] for r in base])),
            **{key:float(np.mean([r[key] for r in rows])) for key in ('fused_color_score','fused_color_top1','color_score_gain','color_top1_gain','new_score')},
            'preselected_seed31415_score':float(np.mean([r['fused_color_score'] for r in rows if r['seed']==31415])),
            'recovered_three_seed_events':sum(r['recovered'] for r in rows),'regressed_three_seed_events':sum(r['regressed'] for r in rows)})
    summary={'status':'complete_prospective_blocked_internal','aggregates':aggregates,'baseline_by_pool':baseline_summary,
        'G1_replication_numeric_gate':point_gate,'G2':outer_gate,'primary_seed31415_paired_CI':ci,
        'unique_outer_players':800,'pools':2,'candidate_count_per_pool':400,'denominator_per_q_per_seed':800,
        'baseline_projection_seconds':baseline_seconds,'per_seed_infer_score_write_seconds':inference_times,
        'analysis_wall_seconds':time.perf_counter()-start,'official_score':False,'pristine_confirmation':False,
        'original_retired400_read':False,'original_U2_primary_still_failed':True,
        'limitation':'conditional on two fixed candidate banks; historically MLP-fit identities, new states fully refitted; selected by earlier development; no original pristine claim'}
    write(reveal/'SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

"""One-batch development reveal after all six last checkpoints are frozen.

Inference helpers receive only observed set indices; truth is used only below
in the isolated ranking/metrics writer. Saved v3 baselines are SHA-checked.
"""
from __future__ import annotations
from collections import defaultdict
import json,time,csv
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from .pilot_scope_v1 import ROOT,RUN,sha,write,opaque
from .pilot_models_v1 import fixed_encode,gaussian_scores
from .pilot_train_v1 import runtime,make_model,load_fit,fit_plan
from .pilot_gaussian_v2 import JointRandomEffects
from .pilot_gaussian_v1 import summaries
from .evaluation import PairedEvent,paired_cluster_intervals
from .gates import g1,SEEDS
from .scoring import fixed_fusion
from .artifacts import ContractError


def all_results():
    models=[]
    for method in ('U2_ISAB_uncertainty','U0_mean_count'):
        for seed in SEEDS:
            directory=RUN/'models'/f'{method}_s{seed}'
            result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
            if result['status']!='complete_fit_only' or result['steps']!=5000 or sha(directory/'model.pt')!=result['model_sha256']:
                raise ContractError('Incomplete/unfrozen model, no development reveal')
            models.append((method,seed,directory,result['model_sha256']))
    directory=RUN/'models/U1_joint_random_effects_v2';result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if result['status']!='complete_fit_only' or sha(directory/'model.npz')!=result['model_sha256']:raise ContractError('Gaussian control not frozen')
    models.append(('U1_joint_random_effects',None,directory,result['model_sha256']))
    return models


def ranks_and_values(scores,targets):
    order=np.argsort(-scores,axis=1,kind='stable')
    rank=np.argmax(order==targets[:,None],axis=1)+1
    values=np.where(rank<=5,np.exp(-(rank-1)),0)
    return order,rank,values


def infer_sets(model,x,colors,supports,queries,variance_scale):
    sm,sv=fixed_encode(model,x,colors,supports);qm,qv=fixed_encode(model,x,colors,queries)
    return gaussian_scores(qm,qv,sm,sv,variance_scale=variance_scale)


def load_checkpoint(directory,device):
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    model=make_model(saved['method'],1/128,saved['floor'])
    model.load_state_dict(saved['state_dict'],strict=True)
    return model.to(device).eval().requires_grad_(False),saved['variance_scale']


def scaled_numeric_check(model,scale,x,colors,supports,queries):
    sm,sv=fixed_encode(model,x,colors,supports)
    a=fixed_encode(model,x,colors,queries[:3]);b=fixed_encode(model,x,colors,[queries[2]])
    sa=gaussian_scores(a[0][2:3],a[1][2:3],sm,sv,variance_scale=scale)
    sb=gaussian_scores(b[0],b[1],sm,sv,variance_scale=scale)
    equal=np.array_equal(np.argsort(-sa[0],kind='stable')[:5],np.argsort(-sb[0],kind='stable')[:5])
    delta=max(float(np.max(np.abs(a[i][2]-b[i][0]))) for i in (0,1))
    if delta>1e-5 or not equal:raise ContractError('Final calibrated model subset ranking mismatch')
    return {'actual_variance_scale':scale,'max_embedding_or_variance_delta':delta,'top5_equal':bool(equal)}


def main():
    registered=json.loads((RUN/'analysis_registration.json').read_text(encoding='utf-8'))
    models=all_results();out=RUN/'development_reveal_v1';out.mkdir(exist_ok=False)
    write(out/'frozen_model_receipt.json',{'models':[{'method':m,'seed':s,'sha256':h} for m,s,d,h in models],
          'analysis_code_sha256':sha(Path(__file__)),'registration_sha256':sha(RUN/'analysis_registration.json'),
          'development_scores_read_before_this_phase':False})
    device=runtime();fitx,fitrecords,_=load_fit();fitgpu=torch.from_numpy(fitx).to(device)
    fitcolors=torch.tensor([r['color'] for r in fitrecords],device=device,dtype=torch.long)
    _,supportfit,queryfit,_,_=fit_plan(fitrecords)
    probes=[]
    with sdpa_kernel(SDPBackend.MATH):
        for method,seed,directory,digest in models:
            if seed is None:continue
            model,scale=load_checkpoint(directory,device)
            probes.append({'method':method,'seed':seed,**scaled_numeric_check(model,scale,fitgpu,fitcolors,supportfit,queryfit)})
            del model
    write(out/'fit_only_final_numeric_checks.json',probes)
    del fitgpu,fitcolors,fitx,fitrecords;torch.cuda.empty_cache()
    d=RUN/'data';receipt=json.loads((d/'vector_receipt.json').read_text(encoding='utf-8'))
    for name in ('features.npy','stat.npy','mlp.npy'):
        if sha(d/name)!=receipt['hashes'][name]:raise ContractError('Allowed exported vectors changed')
    records=json.loads((d/'allowed_records.json').read_text(encoding='utf-8'))
    plans=json.loads((d/'development_plans.json').read_text(encoding='utf-8'))
    vectors=np.load(d/'features.npy',mmap_mode='r',allow_pickle=False)
    x=torch.from_numpy(np.array(vectors,copy=True)).to(device)
    colors_np=np.asarray([r['color'] for r in records],np.int64)
    colors=torch.from_numpy(colors_np).to(device)
    baseline_manifest=json.loads((ROOT/'runs/inference_fixed_development_v3/output_sha256.json').read_text(encoding='utf-8'))
    baselines={};baseline_metrics=[]
    for pool in plans:
        directory=ROOT/'runs/inference_fixed_development_v3'/pool['name']
        mapping=json.loads((directory/'mapping.json').read_text(encoding='utf-8'))
        if [opaque(p) for p in mapping['candidate_ids']]!=pool['candidate_order'] or mapping['question_ids']!=[q['question_id'] for q in pool['questions']]:
            raise ContractError('Saved baseline candidate/query alignment failed')
        targets=np.asarray([pool['candidate_order'].index(q['player']) for q in pool['questions']])
        qcounts=np.asarray([q['q'] for q in pool['questions']])
        values={}
        for kind in ('original','color'):
            name=f'{kind}_rrf_scores.npy';rel=f"{pool['name']}/{name}"
            if sha(directory/name)!=baseline_manifest[rel]:raise ContractError('Frozen baseline checksum failed')
            score=np.load(directory/name,allow_pickle=False)
            if score.shape!=(1200,400) or not np.isfinite(score).all():raise ContractError('Baseline shape/finite failure')
            order,rank,points=ranks_and_values(score,targets)
            values[kind]=(score,rank,points)
            for q in (5,10,20):
                mask=qcounts==q;baseline_metrics.append({'pool':pool['name'],'kind':kind,'q':q,'score':float(points[mask].mean()),'top1':float((rank[mask]==1).mean())})
        baselines[pool['name']]=values
    write(out/'baseline_summary.json',baseline_metrics)
    result_rows=[];events=[];perseed={};start=time.perf_counter()
    for method,seed,directory,digest in models:
        label=f'{method}_s{seed}' if seed is not None else method
        mdir=out/label;mdir.mkdir();rows_metrics=[]
        if seed is None:
            gaussian=JointRandomEffects.load(directory/'model.npz');latent=gaussian.transform(vectors)
        else:model,scale=load_checkpoint(directory,device)
        with (mdir/'predictions.csv').open('x',encoding='utf-8',newline='') as stream:
            fieldnames=['pool','question_id','player','q','seed','black_games','white_games','mean_game_plies','stage0_coverage','stage1_coverage','stage2_coverage',
                        'original_rank','color_rank','new_rank','fused_original_rank','fused_color_rank','new_score','fused_color_score','color_score_gain','top1_recovered','top1_regressed',
                        *[f'fused_color_top{i}' for i in range(1,6)]]
            writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader()
            for pool in plans:
                queries=[q['indices'] for q in pool['questions']];supports=pool['supports']
                if seed is None:new=gaussian.score(summaries(latent,colors_np,queries),summaries(latent,colors_np,supports))
                else:
                    with sdpa_kernel(SDPBackend.MATH):new=infer_sets(model,x,colors,supports,queries,scale)
                b=baselines[pool['name']]
                fo=np.stack([fixed_fusion(b['original'][0][i],s) for i,s in enumerate(new)])
                fc=np.stack([fixed_fusion(b['color'][0][i],s) for i,s in enumerate(new)])
                targets=np.asarray([pool['candidate_order'].index(q['player']) for q in pool['questions']]);qcounts=np.asarray([q['q'] for q in pool['questions']])
                orders,rn,vn=ranks_and_values(new,targets);oo,ro,vo=ranks_and_values(fo,targets);oc,rc,vc=ranks_and_values(fc,targets)
                pdir=mdir/pool['name'];pdir.mkdir()
                for name,array in [('new_scores',new),('fused_original_scores',fo),('fused_color_scores',fc)]:np.save(pdir/f'{name}.npy',array,allow_pickle=False)
                for qn in (5,10,20):
                    mask=qcounts==qn;bs=b['color'][2][mask];br=b['color'][1][mask]
                    row={'method':method,'seed':seed,'pool':pool['name'],'q':qn,'single_score':float(vn[mask].mean()),'single_top1':float((rn[mask]==1).mean()),
                         'fused_original_score':float(vo[mask].mean()),'fused_color_score':float(vc[mask].mean()),'color_score_gain':float((vc[mask]-bs).mean()),
                         'fused_color_top1':float((rc[mask]==1).mean()),'color_top1_gain':float(((rc[mask]==1).astype(float)-(br==1)).mean()),
                         'recovered':int(((br!=1)&(rc[mask]==1)).sum()),'regressed':int(((br==1)&(rc[mask]!=1)).sum())}
                    rows_metrics.append(row);result_rows.append(row)
                for i,q in enumerate(pool['questions']):
                    ids=q['indices'];black=int((colors_np[ids]==0).sum());stage=np.asarray(vectors[ids,1725:1728])
                    row={'pool':pool['name'],'question_id':q['question_id'],'player':q['player'],'q':q['q'],'seed':seed if seed is not None else 'deterministic',
                         'black_games':black,'white_games':q['q']-black,'mean_game_plies':float(np.mean([records[j]['mainline_moves'] for j in ids])),
                         **{f'stage{k}_coverage':float((stage[:,k]>0).mean()) for k in range(3)},
                         'original_rank':int(b['original'][1][i]),'color_rank':int(b['color'][1][i]),'new_rank':int(rn[i]),'fused_original_rank':int(ro[i]),'fused_color_rank':int(rc[i]),
                         'new_score':float(vn[i]),'fused_color_score':float(vc[i]),'color_score_gain':float(vc[i]-b['color'][2][i]),
                         'top1_recovered':int(b['color'][1][i]!=1 and rc[i]==1),'top1_regressed':int(b['color'][1][i]==1 and rc[i]!=1),
                         **{f'fused_color_top{k+1}':pool['candidate_order'][j] for k,j in enumerate(oc[i,:5])}}
                    writer.writerow(row)
                    if method=='U2_ISAB_uncertainty':events.append(PairedEvent(q['player'],pool['name'],seed,q['q'],q['question_id'],row['color_score_gain']))
        write(mdir/'metrics.json',rows_metrics)
        if seed is not None:
            perseed.setdefault(method,{})[seed]={q:{'score_gain':float(np.mean([r['color_score_gain'] for r in rows_metrics if r['q']==q])),
                 'top1_gain':float(np.mean([r['color_top1_gain'] for r in rows_metrics if r['q']==q]))} for q in (5,10,20)}
            del model
        else:del gaussian,latent
        print(json.dumps({'stage':'development_model_complete','method':method,'seed':seed,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    gates={method:g1(values) for method,values in perseed.items()}
    ci=paired_cluster_intervals(events,family_size=registered['family_size'],repetitions=registered['bootstrap_repetitions'],seed=registered['bootstrap_seed'])
    aggregates=[]
    for method in ('U2_ISAB_uncertainty','U0_mean_count','U1_joint_random_effects'):
        for q in (5,10,20):
            rr=[r for r in result_rows if r['method']==method and r['q']==q]
            aggregates.append({'method':method,'q':q,**{k:float(np.mean([r[k] for r in rr])) for k in ['single_score','single_top1','fused_original_score','fused_color_score','color_score_gain','fused_color_top1','color_top1_gain']},
                               'model_pool_observations':len(rr),'recovered_events':sum(r['recovered'] for r in rr),'regressed_events':sum(r['regressed'] for r in rr)})
    write(out/'SUMMARY.json',{'status':'complete_development_only','primary':'U2_ISAB_uncertainty','aggregates':aggregates,'gates':gates,
          'U1_gate':'deterministic_control_no_independent_training_seeds','primary_paired_player_cluster_CI':ci,
          'selected_unique_players':586,'G2':'not_available_not_passed','official_score':False,'retired_reevaluated':False,
          'development_adaptive_selection_bias_removed':False,'analysis_elapsed_seconds':time.perf_counter()-start})
    print(json.dumps({'stage':'development_reveal_complete','gates':gates,'aggregates':aggregates},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

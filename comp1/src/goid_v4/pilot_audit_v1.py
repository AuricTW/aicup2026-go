"""Closure audits: explicitly bounded new exports and saved outputs, no old holdout."""
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
import numpy as np
from .pilot_outer_v1 import ROOT,OUT,hist_block
from .pilot_scope_v1 import RUN,sha,write
from .artifacts import ContractError


def scope():
    old=json.loads((RUN/'protected_text_before.json').read_text(encoding='utf-8'))['files']
    changed=[r['path'] for r in old if sha(ROOT/r['path'])!=r['sha256']]
    pins=json.loads((ROOT/'configs/research_v4/safe_metadata_sources_v1.json').read_text(encoding='utf-8'))['assets']
    badpins=[k for k,r in pins.items() if sha(ROOT/r['path'])!=r['sha256']]
    bundle=ROOT/'artifacts/inference_bundle_v1'
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    safe=['model_state.pt','fit_transform.npz','feature_schema.json']
    badbundle=[n for n in safe if sha(bundle/n)!=manifest['files'][n]]
    if changed or badpins or badbundle:raise ContractError(str((changed,badpins,badbundle)))
    d=OUT/'data';rr=json.loads((d/'source_records.json').read_text(encoding='utf-8'))
    ids=np.load(d/'base_indices.npy',allow_pickle=False)
    base=[r for r in rr if r['outer_pool'] is None];outer=[r for r in rr if r['outer_pool'] is not None]
    expected=np.asarray([i for i,r in enumerate(rr) if r['outer_pool'] is None],np.int64)
    if not np.array_equal(ids,expected):raise ContractError('Background normalizer/MLP indices include outer')
    bp={r['player'] for r in base};op={r['player'] for r in outer}
    if len(bp)!=3974 or len(op)!=800 or bp&op:raise ContractError('Outer player exclusion error')
    overlaps={k:len({r[k] for r in base}&{r[k] for r in outer}) for k in ('raw_hash','move_hash','canonical_hash','group')}
    if any(overlaps.values()):raise ContractError('Known group leakage')
    schema=json.loads((OUT/'base_bundle/feature_schema.json').read_text(encoding='utf-8'))
    raw=np.load(d/'source_raw_features.npy',mmap_mode='r',allow_pickle=False)
    total=np.zeros(1582,np.float64);sq=np.zeros_like(total)
    for start in range(0,len(ids),4096):
        block=hist_block(raw[ids[start:start+4096]],schema).astype(np.float64)
        total+=block.sum(0);sq+=(block*block).sum(0)
    mean=total/len(ids);std=np.maximum(np.sqrt(np.maximum(sq/len(ids)-mean*mean,0)),.001)
    with np.load(OUT/'base_bundle/fit_transform.npz',allow_pickle=False) as fitted:
        mean_equal=np.array_equal(mean.astype(np.float32),fitted['mean'])
        std_equal=np.array_equal(std.astype(np.float32),fitted['std'])
    if not mean_equal or not std_equal:raise ContractError('Normalizer does not match nonouter fit rows')
    for part in ('train','cal'):
        adapted=json.loads((OUT/f'heads_v2/data/{part}_records.json').read_text(encoding='utf-8'))
        original=json.loads((d/f'{part}_records.json').read_text(encoding='utf-8'))
        for a,b in zip(adapted,original):
            if a.pop('effective_group_token')!=b['group'] or a!=b:raise ContractError('Adapter changed data')
        if len(adapted)!=len(original):raise ContractError('Adapter count changed')
        if sha(OUT/f'heads_v2/data/{part}_features.npy')!=sha(d/f'{part}_features.npy'):raise ContractError('Adapter changed features')
    result={'status':'passed','protected_existing_text_files':len(old),'protected_text_changed':changed,
        'pinned_development_metadata_verified':len(pins),'original_safe_bundle_files_verified':safe,
        'base_fit_rows':len(ids),'base_fit_players':len(bp),'outer_players_excluded':len(op),'known_group_intersections':overlaps,
        'normalizer_recomputed_from_only_nonouter_rows_exact':mean_equal and std_equal,
        'new_MLP_weight_sha_differs_original':sha(OUT/'base_bundle/model_state.pt')!=sha(bundle/'model_state.pt'),
        'head_adapter_only_alias_changed':True,'retired_files_or_lists_read':False,
        'mixed_source_whole_file_hash_or_scan':False,
        'limits':['logical row access, not physical OS page isolation','no fresh retired-membership reinspection',
                  'unknown edited/short-prefix/opponent/time leakage not excluded','historically MLP-fit outer players are not original pristine']}
    write(OUT/'scope_closure_audit.json',result);print(json.dumps(result),flush=True)


def pyorder(row):return sorted(range(len(row)),key=lambda j:-float(row[j]))

def ranks(row):
    result=[0]*len(row)
    for rank,index in enumerate(pyorder(row),1):result[index]=rank
    return result


def saved():
    reveal=OUT/'outer_reveal_v1';plans=json.loads((OUT/'data/outer_plans.json').read_text(encoding='utf-8'))
    checks=0;rows=0;gains=defaultdict(list)
    for pool in plans:
        pd=reveal/pool['name']
        cosine={n:np.load(pd/f'{n}.npy',allow_pickle=False) for n in ('stat_color_cosine','mlp_color_cosine','stat_original_cosine','mlp_original_cosine')}
        for flavor in ('color','original'):
            rr=np.load(pd/f'{flavor}_rrf.npy',allow_pickle=False)
            for i,row in enumerate(rr):
                a=ranks(cosine[f'stat_{flavor}_cosine'][i]);b=ranks(cosine[f'mlp_{flavor}_cosine'][i])
                expected=np.asarray([.5/(60+x)+.5/(60+y) for x,y in zip(a,b)])
                if not np.array_equal(row,expected):raise ContractError('Independent baseline RRF mismatch')
                checks+=1
        baseline=np.load(pd/'color_rrf.npy',allow_pickle=False)
        for seed in (31415,27182,16180):
            mdir=reveal/f'U0_s{seed}'
            predictions={r['question_id']:r for r in csv.DictReader((mdir/'predictions.csv').open(encoding='utf-8'))}
            scores=np.load(mdir/pool['name']/'new_scores.npy',allow_pickle=False)
            fused=np.load(mdir/pool['name']/'fused_color_scores.npy',allow_pickle=False)
            for i,q in enumerate(pool['questions']):
                nr=ranks(scores[i]);expected=np.asarray([.75*float(b)+.25/(60+r) for b,r in zip(baseline[i],nr)])
                if not np.array_equal(fused[i],expected):raise ContractError('Independent fixed fusion mismatch')
                order=pyorder(fused[i]);target=pool['candidate_order'].index(q['player']);rank=order.index(target)+1
                br=pyorder(baseline[i]).index(target)+1
                value=math.exp(1-rank) if rank<=5 else 0.;bv=math.exp(1-br) if br<=5 else 0.
                row=predictions[q['question_id']]
                if int(row['fused_color_rank'])!=rank or int(row['color_rank'])!=br:raise ContractError('CSV rank mismatch')
                if abs(float(row['fused_color_score'])-value)>1e-15 or abs(float(row['color_score_gain'])-(value-bv))>1e-15:raise ContractError('CSV metric mismatch')
                if [row[f'top{k+1}'] for k in range(5)]!=[pool['candidate_order'][j] for j in order[:5]]:raise ContractError('CSV top5 mismatch')
                gains[seed,q['q']].append(value-bv);checks+=1;rows+=1
    summary=json.loads((reveal/'SUMMARY.json').read_text(encoding='utf-8'))
    for q in (5,10,20):
        expected=np.mean([np.mean(gains[s,q]) for s in (31415,27182,16180)])
        actual=next(r['color_score_gain'] for r in summary['aggregates'] if r['q']==q)
        if abs(expected-actual)>1e-15:raise ContractError('Summary gain mismatch')
    write(OUT/'saved_output_review.json',{'status':'passed','independent_python_stable_sort_rows':checks,
        'CSV_rows':rows,'all400_candidate_scores_recomputed_per_row':True,'fixed_fusion_exact':True,
        'baseline_RRF_exact':True,'rank_score_and_summary_verified':True,'new_model_forward_calls':0})
    print(json.dumps({'stage':'saved_outer_outputs_reviewed','checks':checks,'rows':rows}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['scope','saved']);args=parser.parse_args()
    globals()[args.stage]()

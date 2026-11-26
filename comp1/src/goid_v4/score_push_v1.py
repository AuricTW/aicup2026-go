"""Prospective score-push experiments on original-fit allowlist only.

Old400 is opened development; remaining323 outcomes stay unopened unless a
single frozen selection winner passes. All NEW fits exclude their723 union.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/research_v4/official_aligned_v1_20261104'
RUN=ROOT/'runs/score_push_experiments_20261104_v2'
SEEDS=(31415,27182,16180)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def token(*x):return hashlib.sha256('|'.join(map(str,x)).encode()).hexdigest()
def log(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def metadata():
    with np.load(OLD/'scope_metadata.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def ranks(a):
    order=np.argsort(-a,axis=1,kind='stable');r=np.empty_like(order)
    np.put_along_axis(r,order,np.broadcast_to(np.arange(1,a.shape[1]+1),a.shape),axis=1);return r
def fusion(a,b):return .5/(60+ranks(a))+.5/(60+ranks(b))

def prepare():
    RUN.mkdir(exist_ok=False);m=metadata();old=read(OLD/'EPISODES.json')
    assert sha(OLD/'scope_metadata.npz')==read(OLD/'CAPACITY.json')['metadata_sha256']
    from .pilot_scope_v1 import RANKS
    good=(m['colors']==0)&(m['hd']<=1);members=defaultdict(list)
    for i in np.flatnonzero(good):members[m['owners'][i].decode()].append(int(i))
    eligible={p for p,ids in members.items() if len(ids)>=120}
    opened=set(old['candidates']);reserved=eligible-opened
    assert len(eligible)==723 and len(opened)==400 and len(reserved)==323
    excluded=eligible
    base=np.asarray([i for i,p in enumerate(m['owners']) if p.decode() not in excluded],np.int64)
    assert not (set(m['owners'][base].tolist())&{p.encode() for p in excluded})
    rp={p.decode():r for p,r in zip(m['owners'],m['ranks'])}
    distractors=sorted(opened,key=lambda p:token('scorepush-confirm-distractor-v1',p))[:77]
    future=sorted(reserved|set(distractors),key=lambda p:token('scorepush-confirm-order-v1',p))
    future_support=[];future_questions=[]
    for col,p in enumerate(future):
        ids=sorted(members[p],key=lambda i:token('scorepush-confirm-game-v1',m['group'][i].decode()))[:120]
        future_support.append(sorted(ids[:100],key=lambda i:m['raw'][i]))
        if p in reserved:
            for q in (5,10,15,20):future_questions.append({'id':f'push_c{col:03}_q{q}','q':q,'indices':sorted(ids[100:100+q],key=lambda i:m['raw'][i]),'target':col,'player':p,'rank':rp[p]})
    # Only metadata is written for the future cohort; no feature replay or scores.
    write(RUN/'reserved/EPISODES_METADATA_ONLY.json',{'candidates':future,'supports':future_support,'questions':future_questions,
        'targets':323,'support_only_distractors':77,'no_quality_computed':True})
    heads=[];allmembers=defaultdict(list)
    for i in base:allmembers[m['owners'][i].decode()].append(int(i))
    head_capacity={}
    head_players={r:sorted([p for p,ix in allmembers.items() if rp[p]==r and len(ix)>=120],key=lambda p:token('head',p)) for r in RANKS}
    head_capacity={r:len(ps) for r,ps in head_players.items()}
    assert min(head_capacity.values())>=40
    cal_quota={r:min(10,head_capacity[r]-40) for r in RANKS}
    while sum(cal_quota.values())<100:
        for r in RANKS:
            if 40+cal_quota[r]<head_capacity[r]:cal_quota[r]+=1
            if sum(cal_quota.values())==100:break
    for r in RANKS:
        for j,p in enumerate(head_players[r][:40+cal_quota[r]]):
            ix=sorted(allmembers[p],key=lambda i:token('headgame',m['group'][i].decode()))[:120]
            heads.extend({'i':i,'part':'train' if j<40 else 'cal','player':p,'rank':r} for i in ix)
    ep={'candidates':old['candidates'],'supports':old['supports'],'questions':old['questions'],'heads':heads,
        'role':'all400 already opened; selection only','no_independent_quality_claim':True}
    write(RUN/'EPISODES.json',ep)
    np.save(RUN/'base_indices.npy',base,allow_pickle=False)
    fitB=base[good[base]];np.save(RUN/'fitB_indices.npy',fitB,allow_pickle=False)
    shutil.copyfile(OLD/'scope_metadata.npz',RUN/'scope_metadata.npz')
    for key in ('raw','move','canonical','group'):
        trainkeys=set(m[key][base].tolist())
        val=[i for p in eligible for i in members[p]]
        assert not trainkeys&set(m[key][val].tolist()),key
    write(RUN/'EXPOSURE_LEDGER.json',{'source':str(OLD/'scope_metadata.npz'),'source_sha256':sha(OLD/'scope_metadata.npz'),
        'original_fit_rows':588294,'historical_fit_exposure':True,'pristine_players_proven':0,
        'opened_today400':400,'future_targets_outcomes_unopened':323,'new_fit_excluded_union':723,
        'rank_future_counts':dict(Counter(rp[p] for p in reserved)),
        'all_color_base_rows':len(base),'B_HD01_fit_rows':len(fitB),'head_eligible_per_rank':head_capacity,'head_cal_rank_quota':cal_quota,
        'retired400_membership_details_read':False,'new_fits_use_union_exclusion_more_conservative_than_research_proposal':True})
    protected=read(OLD/'PROTECTED_BEFORE.json')
    # Extend immutable coverage to current E and the preceding research delivery.
    for root in (ROOT/'artifacts/inference_bundle_E_D_PLDA_20261104_v1',ROOT/'runs/score_push_research_20261104_v1'):
        for p in root.rglob('*'):
            if p.is_file():protected[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    protected['submissions/20261104_E_PLDA_v1/05_E_D_PLDA.csv']=sha(ROOT/'submissions/20261104_E_PLDA_v1/05_E_D_PLDA.csv')
    write(RUN/'PROTECTED_BEFORE.json',protected)
    write(RUN/'PROTOCOL.json',{
        'created_at_Taipei':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'authorization':'user2026-11-04 autonomous experiments following score_push_research_v1',
        'scope':'original safe588294-fit rows only; all new fits exclude723 identities across both cohorts',
        'selection':'already-opened400 players/all1600 nested queries; historically exposed development',
        'confirmation':'at most one frozen winner;323 reserved targets with77 support-only old distractors;400 candidates; no outcome access until gate',
        'original_E_weights_not_used_as_unseen_comparator':True,
        'baseline':'same E method fully refit from scratch with union723 exclusion; source engine uses legacy400 metadata field; EXPOSURE_LEDGER gives actual723',
        'head_quota_addendum':'v1 metadata-only preparation failed at1D capacity48; v2 keeps40train/rank and redistributes2 calibration slots deterministically;400train+100cal total, no quality used',
        'F':{'range_total_plies':40,'ngrams':[2,3,4],'variants':['bag_position','target_order','joint_order_absolute','joint_order_D4'],
            'background':'train-only vocabulary/global distribution; fixed Dirichlet alpha20; candidate support100 counts; mean per-game log likelihood',
            'fusion':'one equal RRF60(E,F) per predeclared representation; no alpha search'},
        'G':{'seeds':list(SEEDS),'G0':[512,128,20000],'G1':[1024,256,20000],
            'baseline':'original architecture/SupCon5000 steps same fitB input, needed to isolate training budget',
            'sampling':'P16 K4 same rank-independent balanced sampler; no extra negative-mining confound',
            'seed_policy':'first seed compares G0/G1; only highest short-score improving branch gets other2seeds; aggregate game embeddings concatenated per seed after normalization',
            'G2':'angular-margin only if G1 is positive and stable; otherwise not admitted this round'},
        'H':{'from_scratch':True,'blocks':6,'channels':64,'positions':16,'embedding':256,'sequence_layers':2,
            'train_players_max':128,'games_per_train_player':24,'loss':'GE2E positive centroid leave-one-game-out',
            'batch':'P4 K2 physical8 complete games, no gradient accumulation claims','profile_steps':500,
            'controls':['state_action_order','state_only_order','action_only_order','state_action_mean'],
            'pilot_steps':2000,'seeds':list(SEEDS),'admission':'real replay and finite loss/representation/gradient pass before full selection inference',
            'additional_seeds':'only a selection-positive full pilot admitted; no automatic big-GPU retry'},
        'common_gate':{'short_delta_min':.01,'q5_delta_gt':0.,'q10_delta_gt':0.,'q15_q20_delta_min':-.005,
            'paired_player_cluster_95_CI_lower_gt':0.,'draws':10000,'bootstrap_seed':2026110402,
            'conditional_fixed_support_bank':True,'point_guard_not_statistical_noninferiority':True},
        'confirmation_policy':'selection gate required; freeze at most1 method and all hashes first; no repeat after reveal; if no winner preserve323 unopened',
        'resources':'local RTX3060 only;4CPU threads, ownGPU memory cap60%; timings may be contended; no peer processes terminated',
        'caps_seconds':{'F':10800,'G_per_run':1800,'H_profile':3600,'H_pilot':21600},
        'no_official_data_forward_in_this_round':True,'no_upload':True,'no_label_probing':True,'no_global_assignment':True,
        'no_retired400_read':True,'no_external_code_or_weights':True,
        'source_ledger':'runs/score_push_research_20261104_v1/SOURCES.json',
        'frozen_split_sha256':sha(RUN/'EPISODES.json'),'reserved_metadata_sha256':sha(RUN/'reserved/EPISODES_METADATA_ONLY.json'),
        'initial_code_sha256':sha(Path(__file__))})
    log(stage='protocol_prepared',base_rows=len(base),fitB_rows=len(fitB),reserved_targets=323,head_capacity=head_capacity)

def engine():
    from . import official_aligned_v1 as a
    a.OUT=RUN
    return a

def baseline():
    import torch
    torch.cuda.set_per_process_memory_fraction(.60)
    a=engine();a.base();a.heads();a.fit_metrics()
    ep=read(RUN/'EPISODES.json')
    write(RUN/'evaluation_inputs/selection.json',{'candidate_ids':ep['candidates'],'supports':ep['supports'],
        'questions':[{k:q[k] for k in ('id','q','indices')} for q in ep['questions']]})
    write(RUN/'evaluation_inputs/SELECTION_EVALUATOR_ONLY_GT.json',[{k:q[k] for k in ('id','q','target','player')} for q in ep['questions']])
    a.predict('selection')
    write(RUN/'BASELINE_SCOPE_ADDENDUM.json',{'all_fitted_states_exclude_union_players':723,
        'actual_base_rows':len(np.load(RUN/'base_indices.npy')),'legacy_receipt_field_400_is_minimum_not_actual_union_count':True,
        'confirmation_outcomes_not_evaluated':True,'E_method_prediction_file':'selection_prediction/RRF_D_PLDA_raw.npy'})
    log(stage='E_method_baseline_refit_complete')

def metrics(score,truth):
    rr=ranks(score)[np.arange(len(truth)),[q['target'] for q in truth]]
    values=np.where(rr<=5,np.exp(-(rr-1)),0.)
    qc=np.asarray([q['q'] for q in truth])
    out={str(q):{'score':float(values[qc==q].mean()),'top1':float((rr[qc==q]==1).mean()),'top5':float((rr[qc==q]<=5).mean()),'n':int((qc==q).sum())} for q in (5,10,15,20)}
    out['short']=float((out['5']['score']+out['10']['score'])/2)
    return out,values,rr

def evaluate_selection(name,score):
    truth=read(RUN/'evaluation_inputs/SELECTION_EVALUATOR_ONLY_GT.json')
    reference=np.load(RUN/'selection_prediction/RRF_D_PLDA_raw.npy',allow_pickle=False)
    assert score.shape==reference.shape==(1600,400) and np.isfinite(score).all()
    out=RUN/'selection_results'/name;out.mkdir(parents=True,exist_ok=False)
    np.save(out/'scores.npy',score,allow_pickle=False)
    mm,v,rr=metrics(score,truth);em,ev,er=metrics(reference,truth)
    ps=sorted({q['player'] for q in truth});idx={p:i for i,p in enumerate(ps)}
    diff=np.zeros((len(ps),4))
    for j,q in enumerate(truth):diff[idx[q['player']],(5,10,15,20).index(q['q'])]=v[j]-ev[j]
    rng=np.random.default_rng(2026110402);draws=rng.integers(len(ps),size=(10000,len(ps)))
    boot=diff[draws,:2].mean((1,2));delta=diff.mean(0)
    gate={'short_point':bool(delta[:2].mean()>=.01),'q5_q10_positive':bool((delta[:2]>0).all()),
          'long_point_guard':bool((delta[2:]>=-.005).all()),'short_cluster_lower':bool(np.quantile(boot,.025)>0)}
    qs=np.asarray([q['q'] for q in truth]);meta=metadata();ranks_by_owner={p:r for p,r in zip(meta['owners'].astype(str),meta['ranks'])}
    rank_slices={}
    for rank in sorted(set(ranks_by_owner[q['player']] for q in truth)):
        mask=np.asarray([ranks_by_owner[q['player']]==rank and q['q']<=10 for q in truth])
        rank_slices[rank]={'n':int(mask.sum()),'delta':float((v-ev)[mask].mean())}
    result={'method':name,'role':'already-opened selection only; not independent confirmation','metrics':mm,'E_reference':em,
        'delta_q5_q10_q15_q20':delta.tolist(),'short_delta':float(delta[:2].mean()),'short_cluster95CI':np.quantile(boot,[.025,.975]).tolist(),
        'gate':gate,'passed_selection_gate':all(gate.values()),'short_top1_recovered':int(((rr==1)&(er!=1)&(qs<=10)).sum()),
        'short_top1_regressed':int(((rr!=1)&(er==1)&(qs<=10)).sum()),'rank_slices':rank_slices,
        'reserved323_quality_read':False,'score_sha256':sha(out/'scores.npy')}
    write(out/'SUMMARY.json',result);log(stage='selection_evaluated',method=name,short=mm['short'],delta=result['short_delta'],gate=gate)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage');args=p.parse_args()
    globals()[args.stage]()

if __name__=='__main__':main()

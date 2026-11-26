"""Single frozen selection winner, reserved323 one-shot quality comparison."""
from pathlib import Path
import time
import numpy as np
from . import score_push_v1 as c

def E_predict(inputs,dest):
    """Truth-free reuse of fresh E method; every state excludes union723."""
    import torch
    from threadpoolctl import threadpool_limits
    from .pilot_train_v1 import runtime
    from .pilot_models_v1 import MeanCount,fixed_encode,gaussian_scores
    from .plda_metric_v1 import PLDA,histograms,summarize
    from goid_submission_v1.gaussian_csv import FrozenGaussian,summaries
    qs=[q['indices'] for q in inputs['questions']];ss=inputs['supports']
    used=sorted({i for s in ss+qs for i in s});pos={v:i for i,v in enumerate(used)}
    si=[[pos[i] for i in s] for s in ss];qi=[[pos[i] for i in s] for s in qs]
    m=c.metadata();x=np.array(np.load(c.RUN/'features.npy',mmap_mode='r')[used],copy=True)
    colors=m['colors'][used];assert (colors==0).all() and (m['hd'][used]<=1).all()
    def cosine(v):
        bank=np.stack([v[s].mean(0) for s in si]);bank/=np.maximum(np.linalg.norm(bank,axis=1,keepdims=True),1e-12);bank=np.array(bank,order='C')
        scores=np.empty((len(qi),400),np.float32)
        for j,q in enumerate(qi):
            vq=v[q].mean(0);vq/=max(float(np.linalg.norm(vq)),1e-12);scores[j]=bank@np.array(vq,order='C')
        return scores
    color=c.fusion(cosine(x[:,:1582]),cosine(x[:,1582:1710]))
    saved=torch.load(c.RUN/'models/U0_mean_count_s31415/model.pt',map_location='cpu',weights_only=True)
    device=runtime();torch.cuda.set_per_process_memory_fraction(.6)
    head=MeanCount(floor=saved['floor'],prior=.001).to(device);head.load_state_dict(saved['state_dict']);head.eval().requires_grad_(False)
    tx=torch.from_numpy(x).to(device);tc=torch.from_numpy(colors.astype(np.int64)).to(device)
    sm,sv=fixed_encode(head,tx,tc,si);qm,qv=fixed_encode(head,tx,tc,qi)
    u0=.75*color+.25/(60+c.ranks(gaussian_scores(qm,qv,sm,sv,variance_scale=saved['variance_scale'])))
    del tx,tc,head;torch.cuda.empty_cache()
    with threadpool_limits(limits=4):
        g=FrozenGaussian(c.RUN/'metrics/D_Gaussian.npz');z=g.transform(x)
        D=c.fusion(g.score(summaries(z,colors,qi),summaries(z,colors,si)),u0)
        raw=np.load(c.ROOT/'data/processed/features.npy',mmap_mode='r')
        xx=histograms(raw[m['rows'][used]],c.read(c.ROOT/'data/processed/feature_schema.json'),sqrt=True)
        p=PLDA(c.RUN/'metrics/PLDA_raw.npz');z=p.transform(xx)
        qm,qn=summarize(z,qi);sm,sn=summarize(z,si);P=p.score(qm,qn,sm,sn)
        E=c.fusion(D,P)
    for name,score in [('D',D),('PLDA',P),('E',E)]:np.save(Path(dest)/f'{name}.npy',score,allow_pickle=False)
    return E

def freeze():
    assert (c.RUN/'F/COMPLETE.json').exists() and (c.RUN/'G/DECISION.json').exists()
    h_primary=c.RUN/'H/state_action_order_s31415/RESULT.json'
    if not (c.RUN/'H/DECISION.json').exists():
        # The preregistered H full-model admission already failed; diagnostic
        # ablations may finish independently, but cannot reenter this selection.
        assert h_primary.exists() and not c.read(h_primary)['admitted_to_full_selection_inference']
        c.write(c.RUN/'H/PRIMARY_ELIGIBILITY_CLOSED.json',{'reason':'preregistered full-model admission failed after2000steps',
            'primary_result_sha256':c.sha(h_primary),'no_new_or_relaxed_gate':True,
            'diagnostic_controls_can_continue_but_no_H_candidate_can_reenter_this_round_selection':True})
    choices=[]
    for p in (c.RUN/'selection_results').glob('*/SUMMARY.json'):
        r=c.read(p);name=r['method']
        if not r['passed_selection_gate']:continue
        if name.startswith('E_F_') or (name.startswith('E_G') and 'three_seed' in name) or (name.startswith('E_H_') and 'three_seed' in name):
            choices.append((r['metrics']['short'],name,p))
    if not choices:
        c.write(c.RUN/'SELECTION_DECISION.json',{'status':'no_admissible_winner','reserved323_quality_remains_unopened':True});return
    _,name,path=max(choices)
    if not name.startswith('E_F_'):raise RuntimeError('Non-F winner requires its corresponding truth-free predictor before confirmation; do not substitute F')
    # The selected family and all computation used for E/F freeze before reserved scoring.
    bound=[c.RUN/'PROTOCOL.json',path,c.RUN/'metrics/PLDA_raw.npz',c.RUN/'metrics/D_Gaussian.npz',
        c.RUN/'base_bundle/model_state.pt',c.RUN/'base_bundle/fit_transform.npz',c.RUN/'models/U0_mean_count_s31415/model.pt',
        Path(__file__),c.ROOT/'src/goid_v4/score_push_features_v1.py',c.ROOT/'src/goid_v4/score_push_v1.py',
        c.RUN/'G/DECISION.json',h_primary]
    c.write(c.RUN/'SELECTED_FREEZE.json',{'selected':name,'variant':name.removeprefix('E_F_'),
        'eligible_selection_choices':[n for _,n,_ in choices],'rule':'largest selection short score among gate-passing admitted variants',
        'fusion':'fixed equal RRF60(E,F)','fitB_indices_sha256':c.sha(c.RUN/'fitB_indices.npy'),
        'bound_sha256':{str(p.relative_to(c.ROOT)).replace('\\','/'):c.sha(p) for p in bound},
        'reserved323_scores_or_outcomes_not_evaluated_yet':True,'no_further_selection_or_gate_adjustment':True})
    c.log(stage='single_winner_frozen',method=name)

def predict():
    from .score_push_features_v1 import sequence_export,design,likelihood
    from threadpoolctl import threadpool_limits
    frozen=c.read(c.RUN/'SELECTED_FREEZE.json')
    assert all(c.sha(c.ROOT/p)==h for p,h in frozen['bound_sha256'].items())
    dest=c.RUN/'reserved_prediction';dest.mkdir(exist_ok=False)
    ep=c.read(c.RUN/'reserved/EPISODES_METADATA_ONLY.json')
    inputs={'candidate_ids':ep['candidates'],'supports':ep['supports'],
        'questions':[{k:q[k] for k in ('id','q','indices')} for q in ep['questions']]}
    c.write(dest/'INPUTS_WITHOUT_GT.json',inputs)
    c.write(dest/'RESERVATION.json',{'one_quality_comparison_reserved':True,'selected_freeze_sha256':c.sha(c.RUN/'SELECTED_FREEZE.json'),'quality_not_loaded':True})
    E=E_predict(inputs,dest)
    m=c.metadata();fit=np.load(c.RUN/'fitB_indices.npy');sets=inputs['supports']+[q['indices'] for q in inputs['questions']]
    used=np.asarray(sorted(set(fit.tolist())|{i for s in sets for i in s}),np.int64);positions={i:j for j,i in enumerate(used)}
    arr=sequence_export(used,m)
    with threadpool_limits(limits=2):
        x=design(arr,frozen['variant'])
        F,receipt=likelihood(x,[positions[i] for i in fit],[[positions[i] for i in s] for s in inputs['supports']],
            [[positions[i] for i in q['indices']] for q in inputs['questions']])
    combined=c.fusion(E,F)
    np.save(dest/'F.npy',F,allow_pickle=False);np.save(dest/'SELECTED.npy',combined,allow_pickle=False)
    c.write(dest/'PREDICTION_RECEIPT.json',{'truth_free_score_function':True,'queries':len(inputs['questions']),'candidates':400,
        'background_and_E_states_exclude_all723':True,'F_model_receipt':receipt,
        'hashes':{p.name:c.sha(p) for p in dest.glob('*.npy')},'selected_freeze_sha256':c.sha(c.RUN/'SELECTED_FREEZE.json')})
    c.log(stage='reserved_prediction_saved_no_quality_yet',queries=len(inputs['questions']))

def evaluate():
    # Exclusive one-shot reservation before reading truth or computing quality.
    c.write(c.RUN/'RESERVED_QUALITY_OPENED.json',{'single_reveal':True,'frozen_sha256':c.sha(c.RUN/'SELECTED_FREEZE.json'),
        'cannot_reuse_this_cohort_as_unopened_after_this_point':True})
    ep=c.read(c.RUN/'reserved/EPISODES_METADATA_ONLY.json');truth=ep['questions'];dest=c.RUN/'reserved_prediction'
    reference=np.load(dest/'E.npy');score=np.load(dest/'SELECTED.npy')
    mm,v,rr=c.metrics(score,truth);em,ev,er=c.metrics(reference,truth)
    ps=sorted({q['player'] for q in truth});assert len(ps)==323
    idx={p:i for i,p in enumerate(ps)};diff=np.zeros((len(ps),4))
    for j,q in enumerate(truth):diff[idx[q['player']],(5,10,15,20).index(q['q'])]=v[j]-ev[j]
    rng=np.random.default_rng(2026110402);draws=rng.integers(len(ps),size=(10000,len(ps)))
    boot=diff[draws,:2].mean((1,2));delta=diff.mean(0)
    gate={'short_point':bool(delta[:2].mean()>=.01),'q5_q10_positive':bool((delta[:2]>0).all()),
        'long_point_guard':bool((delta[2:]>=-.005).all()),'short_cluster_lower':bool(np.quantile(boot,.025)>0)}
    qs=np.asarray([q['q'] for q in truth]);rank_slices={}
    for rank in sorted({q['rank'] for q in truth}):
        mask=np.asarray([q['rank']==rank and q['q']<=10 for q in truth]);rank_slices[rank]={'questions':int(mask.sum()),'players':int(mask.sum())//2,'short_delta':float((v-ev)[mask].mean())}
    result={'status':'one_shot_comparison_complete','selected':c.read(c.RUN/'SELECTED_FREEZE.json')['selected'],
        'targets':323,'queries':len(truth),'support_candidates':400,'support_only_distractors':77,
        'historical_training_exposure_not_pristine':True,'all_new_fitted_states_exclude_union723':True,
        'selected_metrics':mm,'E_reference_metrics':em,'delta_q5_q10_q15_q20':delta.tolist(),
        'short_delta':float(delta[:2].mean()),'short_cluster95CI':np.quantile(boot,[.025,.975]).tolist(),
        'gate':gate,'passed':all(gate.values()),'long_guard_is_point_only':True,'rank_slices':rank_slices,
        'short_top1_recovered':int(((rr==1)&(er!=1)&(qs<=10)).sum()),'short_top1_regressed':int(((rr!=1)&(er==1)&(qs<=10)).sum()),
        'conditional_fixed_support_bank':True,'not_official_score':True,'retired400_read':False,
        'this323_cohort_now_opened_and_retired_from_unopened_selection':True}
    c.write(c.RUN/'CONFIRMATION_SUMMARY.json',result)
    c.write(c.RUN/'confirmation_events.json',[dict(q,selected_rank=int(rr[i]),E_rank=int(er[i]),selected_value=float(v[i]),E_value=float(ev[i])) for i,q in enumerate(truth)])
    c.log(stage='one_shot_confirmation_complete',**result)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','predict','evaluate']);args=parser.parse_args();globals()[args.stage]()

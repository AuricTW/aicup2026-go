"""Real fit-only P1 training/calibration. Never loads development vectors/scores."""
from __future__ import annotations
from collections import defaultdict
import argparse,json,time,os,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from .pilot_scope_v1 import ROOT,RUN,sha,write
from .pilot_models_v1 import PackedISAB,MeanCount,TensorSampler,fixed_encode,gaussian_scores
from .pilot_gaussian_v1 import JointRandomEffects,summaries
from .p1 import uncertainty_logits,identity_loss
from .artifacts import ContractError


def runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.mha.set_fastpath_enabled(False)
    if not torch.cuda.is_available() or 'RTX 3060' not in torch.cuda.get_device_name():raise ContractError('Local RTX3060 required')
    return torch.device('cuda')


def load_fit():
    gate=json.loads((RUN/'G0_scope.json').read_text(encoding='utf-8'))
    if gate['status']!='passed_row_scope_contracts_requires_permodel_numeric_probe':raise ContractError('Real row scope G0 not ready')
    d=RUN/'data';receipt=json.loads((d/'vector_receipt.json').read_text(encoding='utf-8'))
    data=[];records=[];train_records=None
    for part in ('train','cal'):
        for suffix in ('features.npy','records.json'):
            p=f'{part}_{suffix}'
            if sha(d/p)!=receipt['hashes'][p]:raise ContractError('Fit export changed')
        rr=json.loads((d/f'{part}_records.json').read_text(encoding='utf-8'))
        if any(r['part']!=part for r in rr):raise ContractError('Role substitution')
        data.append(np.load(d/f'{part}_features.npy',allow_pickle=False));records.extend(rr)
        if part=='train':train_records=rr
    if {r['player'] for r in train_records}&{r['player'] for r in records[len(train_records):]}:raise ContractError('Train/cal identity overlap')
    return np.concatenate(data),records,train_records


def fit_plan(records):
    members=defaultdict(list)
    for i,r in enumerate(records):members[r['player']].append(i)
    order=sorted(members);supports=[];queries=[];targets=[];qcounts=[]
    for p in order:
        ids=sorted(members[p]);supports.append(ids[:100])
        if records[ids[0]]['part']=='cal':
            for n in (5,10,20):queries.append(ids[100:100+n]);targets.append(order.index(p));qcounts.append(n)
    if len(order)!=500 or len(queries)!=300:raise ContractError('Fit-only calibration shape changed')
    return order,supports,queries,np.asarray(targets),np.asarray(qcounts)


def metric(scores,targets,qcounts):
    order=np.argsort(-scores,axis=1,kind='stable')
    ranks=np.argmax(order==targets[:,None],axis=1)+1
    values=np.where(ranks<=5,np.exp(-(ranks-1)),0)
    return {str(q):float(values[qcounts==q].mean()) for q in (5,10,20)}


def make_model(method,prior,floor):
    if method=='U2_ISAB_uncertainty':return PackedISAB(1728,width=128,heads=4,ffn=256,inducing=16,embedding_dim=128,max_games=100,dropout=.1,variance_floor=float(floor),prior_variance=np.full(128,prior,np.float32))
    if method=='U0_mean_count':return MeanCount(floor=float(floor),prior=float(prior))
    raise ContractError('Unknown trainable method')


def initial_prior(x,records):
    members=defaultdict(list)
    for i,r in enumerate(records):members[r['player']].append(i)
    # Isotropic trace prior is invariant to arbitrary normalized128D coordinate rotations.
    prior=float(np.mean([np.var(x[ids,1582:1710].astype(np.float64),axis=0,ddof=1).mean() for ids in members.values()]))
    if not 1e-8<prior<1:raise ContractError('Invalid fit-only normalized-vector prior')
    return prior


def numeric_probe(model,x,colors,supports,queries):
    sm,sv=fixed_encode(model,x,colors,supports)
    a=fixed_encode(model,x,colors,queries[:3]);repeat=fixed_encode(model,x,colors,queries[:3]);b=fixed_encode(model,x,colors,[queries[2],queries[0]])
    delta=max(float(np.max(np.abs(a[j][2]-b[j][0]))) for j in (0,1))
    first=gaussian_scores(a[0][2:3],a[1][2:3],sm,sv)[0]
    second=gaussian_scores(b[0][:1],b[1][:1],sm,sv)[0]
    equal_top5=np.array_equal(np.argsort(-first,kind='stable')[:5],np.argsort(-second,kind='stable')[:5])
    exact=all(np.array_equal(v,w) for v,w in zip(a,repeat))
    if not exact or delta>1e-5 or not equal_top5:raise ContractError('Fixed numerical contract probe failed')
    return {'exact_same_job_repeat':exact,'subset_max_mean_or_variance_delta':delta,'subset_top5_equal':bool(equal_top5),'probe_scope':'fit-inner only, not development'}


def calibrate(model,x,records,colors,supports,queries,targets,qcounts,config):
    model.eval().requires_grad_(False)
    members=defaultdict(list)
    for i,r in enumerate(records):
        if r['part']=='cal':members[r['player']].append(i)
    singles=[ [i] for p in sorted(members) for i in sorted(members[p])[100:120] ]
    m,_=fixed_encode(model,x,colors,singles)
    m=m.reshape(100,20,128)
    prior=max(float(m.astype(np.float64).var(axis=1,ddof=1).mean()),1e-8)
    sm,sv=fixed_encode(model,x,colors,supports);qm,qv=fixed_encode(model,x,colors,queries)
    old_floor=model.floor
    base_s=np.maximum(sv.astype(np.float64)-old_floor,0);base_q=np.maximum(qv.astype(np.float64)-old_floor,0)
    candidates=[];best=None
    for mult in config['calibration']['floor_multiplier_grid']:
        floor=prior/100*mult
        for scale in config['calibration']['variance_scale_grid']:
            scores=gaussian_scores(qm,base_q+floor,sm,base_s+floor,variance_scale=scale)
            metrics=metric(scores,targets,qcounts);short=(metrics['5']+metrics['10'])/2
            row={'floor_multiplier':mult,'floor':floor,'variance_scale':scale,'fit_inner_scores':metrics,'short_score':short}
            candidates.append(row)
            if best is None or short>best['short_score']:best=row
    with torch.no_grad():model.prior_variance.fill_(prior)
    model.floor=best['floor']
    return {'prior_scalar':prior,'initial_floor':old_floor,'selected':best,'candidates':candidates,
            'selection_scope':'fit-inner100 targets,500 fit-only support candidates','no_development_scores_read':True}


def train(method,seed):
    config=json.loads((ROOT/'configs/research_v4/p1_pilot_v1_20261026.json').read_text(encoding='utf-8'))
    directory=RUN/'models'/f'{method}_s{seed}';directory.mkdir(parents=True,exist_ok=False)
    xnp,records,train_records=load_fit();ntrain=len(train_records)
    prior=initial_prior(xnp[:ntrain],train_records);floor=prior/100*.01
    device=runtime();torch.manual_seed(seed);np.random.seed(seed)
    x=torch.from_numpy(xnp).to(device);colors=torch.tensor([r['color'] for r in records],device=device,dtype=torch.long)
    order,supports,queries,targets,qcounts=fit_plan(records)
    model=make_model(method,prior,floor).to(device)
    sampler=TensorSampler(train_records,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    write(directory/'preregistration.json',{'method':method,'seed':seed,'max_steps':config['max_steps'],'wall_cap':config['wall_seconds_per_training_run'],
        'initial_prior':prior,'initial_floor':floor,'source_code':{p.name:sha(p) for p in sorted((ROOT/'src/goid_v4').glob('pilot_*_v1.py'))},
        'config_sha256':sha(ROOT/'configs/research_v4/p1_pilot_v1_20261026.json'),
        'scope_sha256':sha(RUN/'data/scope_receipt.json'),'vector_receipt_sha256':sha(RUN/'data/vector_receipt.json'),
        'sdpa':'MATH','mha_fastpath':False,'train_batch_sets':64,'inference_batch_sets':32,'checkpoint_selection':'last','development_quality_read':False,
        'optimizer':config['optimizer'],'parameter_count':sum(p.numel() for p in model.parameters())})
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();step=0;total_loss=0;windows=0
    try:
        with sdpa_kernel(SDPBackend.MATH):
            model.eval();probe=numeric_probe(model,x,colors,supports,queries)
            write(directory/'numeric_probe_before_training.json',probe)
            write(directory/'G0_permodel.json',{'status':'passed_for_fit_only_pilot','scope_gate_sha256':sha(RUN/'G0_scope.json'),
                  'numeric_probe':probe,'independent_confirmation_claim':False,'retired_scope_rechecked':False})
            model.train()
            with (directory/'training.jsonl').open('x',encoding='utf-8') as log:
                for step in range(1,config['max_steps']+1):
                    if time.perf_counter()-start>config['wall_seconds_per_training_run']:raise ContractError('Training wall cap exceeded')
                    idx,mask,negative=sampler.sample()
                    ti=torch.from_numpy(idx).to(device);tm=torch.from_numpy(mask).to(device)
                    xx=x[ti].masked_fill(~tm[:,:,None],0);cc=colors[ti]
                    optimizer.zero_grad(set_to_none=True)
                    mean,var=model.forward_packed(xx,cc,tm)
                    logits=uncertainty_logits(mean[32:],var[32:],mean[:32],var[:32])
                    loss=identity_loss(logits,negative_mask=torch.from_numpy(negative).to(device))
                    if not torch.isfinite(loss):raise ContractError('Nonfinite fit loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                    optimizer.step();total_loss+=float(loss.detach());windows+=1
                    if torch.cuda.max_memory_allocated()>config['peak_vram_GiB_limit']*2**30:raise ContractError('VRAM cap exceeded')
                    if step==1 or step%250==0:
                        row={'method':method,'seed':seed,'step':step,'loss_mean':total_loss/windows,'gradient_norm':float(norm.detach()),
                             'elapsed_seconds':time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
                        log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);total_loss=0;windows=0
            training_elapsed=time.perf_counter()-start
            calibration_start=time.perf_counter();calibration=calibrate(model,x,records,colors,supports,queries,targets,qcounts,config)
            if time.perf_counter()-calibration_start>config['calibration']['wall_seconds']:raise ContractError('Calibration cap exceeded')
            write(directory/'calibration.json',calibration)
            write(directory/'numeric_probe_final.json',numeric_probe(model,x,colors,supports,queries))
        with (directory/'model.pt').open('xb') as f:
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'method':method,'seed':seed,
                        'floor':model.floor,'variance_scale':calibration['selected']['variance_scale'],'steps':step},f)
        write(directory/'result.json',{'status':'complete_fit_only','method':method,'seed':seed,'steps':step,'training_elapsed_seconds':training_elapsed,
              'total_elapsed_seconds':time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
              'model_sha256':sha(directory/'model.pt'),'development_evaluated':False,'cuda_device':torch.cuda.get_device_name(),
              'torch':torch.__version__,'numpy':np.__version__,'cuda_compiled':torch.version.cuda})
        print(json.dumps({'stage':'trained_and_fit_calibrated','method':method,'seed':seed,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException as e:
        write(directory/'failure.json',{'status':'failed_preserved','step':step,'error':str(e),'type':type(e).__name__,'elapsed_seconds':time.perf_counter()-start,'development_evaluated':False})
        with (directory/'failed_model.pt').open('xb') as f:torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step},f)
        raise


def gaussian():
    config=json.loads((ROOT/'configs/research_v4/p1_pilot_v1_20261026.json').read_text(encoding='utf-8'))
    directory=RUN/'models/U1_joint_random_effects';directory.mkdir(parents=True,exist_ok=False)
    x,records,train_records=load_fit();n=len(train_records);colors=np.asarray([r['color'] for r in records])
    start=time.perf_counter();model=JointRandomEffects()
    model.fit(x[:n],colors[:n],[r['player'] for r in train_records],dimension=128)
    z=model.transform(x);order,supports,queries,targets,qcounts=fit_plan(records)
    ss=summaries(z,colors,supports);qs=summaries(z,colors,queries)
    grid=[];best=None
    for prior in config['calibration']['random_effects_prior_scale_grid']:
        for noise in config['calibration']['random_effects_noise_scale_grid']:
            model.prior_scale=prior;model.noise_scale=noise
            scores=model.score(qs,ss);metrics=metric(scores,targets,qcounts);short=(metrics['5']+metrics['10'])/2
            row={'prior_scale':prior,'noise_scale':noise,'fit_inner_scores':metrics,'short_score':short};grid.append(row)
            if best is None or short>best['short_score']:best=row
    model.prior_scale=best['prior_scale'];model.noise_scale=best['noise_scale'];model.save(directory/'model.npz')
    write(directory/'calibration.json',{'selected':best,'grid':grid,'selection_scope':'fit-inner100 targets only','development_scores_read':False})
    write(directory/'result.json',{'status':'complete_fit_only','deterministic_no_training_seed_variation':True,'source_code_sha256':sha(ROOT/'src/goid_v4/pilot_gaussian_v1.py'),
          'scope_receipt_sha256':sha(RUN/'data/scope_receipt.json'),'vector_receipt_sha256':sha(RUN/'data/vector_receipt.json'),
          'model_sha256':sha(directory/'model.npz'),'elapsed_seconds':time.perf_counter()-start,'development_evaluated':False})
    print(json.dumps({'stage':'Gaussian_fit_calibration_complete','elapsed_seconds':time.perf_counter()-start,'selected':best}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True,choices=['U0_mean_count','U2_ISAB_uncertainty','U1_joint_random_effects']);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.method=='U1_joint_random_effects':gaussian()
    else:
        if args.seed not in (31415,27182,16180):raise ContractError('Fixed seeds only')
        train(args.method,args.seed)


if __name__=='__main__':main()

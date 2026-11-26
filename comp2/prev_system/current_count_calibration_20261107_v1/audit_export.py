"""Independent metric/forward audit and reproducible unique official CSV export."""
from pathlib import Path
import os,sys,json,csv,hashlib,datetime,importlib.util
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import numpy as np
from scipy.special import softmax
B=Path(__file__).resolve().parents[2];R=Path(__file__).resolve().parent;C=B/'runs/current_recipe_calibration_20261107_v1';F=B/'runs/fullsystem_oof_20261106_v1';A=B/'runs/activity_replication_20261107_v1';E=B/'runs/extra_player_audit_20261107_v1';O=B/'submissions/two_candidates_20261106_v1';OUT=B/'submissions/count_calibration_20261107_v1';W=np.array([.5625,.1875,.25]);SEEDS=(42,123,2026);LABELS=np.array(['D','C','B','A','1D','2D','3D','4D','5D','6D'])
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def independent(streams,G,coeff):
 s=np.asarray(coeff);lam=np.log(np.array(G,dtype=np.float64)/5)/np.log(4);weights=(s[:3][None]*(1-lam[:,None])+s[3:][None]*lam[:,None])*W;z=np.zeros(streams.shape[:-2]+(10,),np.float64)
 for j in range(3):z+=np.log(streams[...,j,:].clip(1e-30))*weights[:,j,None]
 z-=z.max(-1,keepdims=True);p=np.exp(z);return p/p.sum(-1,keepdims=True)
def macro(q,x):
 a={}
 for row,v in zip(q,x):a.setdefault((row['fold'],row['rank_idx'],row['player_code']),[]).append(float(v))
 s={}
 for (f,r,c),v in a.items():s.setdefault((f,r),[]).append(float(np.mean(v)))
 return float(np.mean([np.mean(v) for v in s.values()]))
def reward(q,p):
 y=np.array([a['rank_idx'] for a in q]);d=abs(p.argmax(-1)-y);return np.where(d==0,1.,np.where(d==1,np.exp(-1),0.))
def audit():
 assert not (R/'INDEPENDENT_AUDIT.json').exists();result=load(R/'RESULTS.json');proto=load(R/'PROTOCOL.json');assert all(sha(B/p)==h for p,h in proto['sources'].items());paths={'old':F/'input/queries.json','extra':E/'QUERIES.json','activity':A/'QUERIES.json'};allq={};arrays={};reports=[];codes={}
 for c,path in paths.items():
  q=load(path);allq[c]=q;codes[c]={a['player_code'] for a in q};stored=np.load(R/(c.upper()+'_FUSED.npz'));assert np.array_equal(stored['question_ids'],[a['question_id'] for a in q]);maxerr=0.
  for f in range(3):
   ix=np.flatnonzero([a['fold']==f for a in q]);v=np.load(C/f'fold{f}/{c}.npz');G=[q[i]['num_games'] for i in ix];s=load(R/f'fold{f}/FITS.json')['count']['coefficients'];p=independent(v['streams'],G,s);maxerr=max(maxerr,float(abs(p[0]-stored['count'][ix]).max()),float(abs(p[1:]-stored['seed_count'][:,ix]).max()))
   assert np.array_equal(p[0].argmax(-1),stored['count'][ix].argmax(-1))
  assert maxerr<1e-12;d=macro(q,reward(q,stored['count'])-reward(q,stored['current']));assert abs(d-result['reports'][c]['comparison']['delta'])<1e-12;arrays[c]=stored;reports.append(dict(cohort=c,players=len(codes[c]),queries=len(q),posterior_replay_max_error=maxerr,delta=d))
 assert sum(map(len,codes.values()))==len(set.union(*codes.values()))==5529;q=allq['old']+allq['extra'];candidate=np.concatenate([arrays[c]['count'] for c in ('old','extra')]);reference=np.concatenate([arrays[c]['current'] for c in ('old','extra')]);d=macro(q,reward(q,candidate)-reward(q,reference));assert abs(d-result['pooled3022']['delta'])<1e-12
 aq=allq['activity'];G=np.array([a['num_games'] for a in aq]);risk={}
 for g in (5,10,15):
  mask=G==g;sub=[a for a,v in zip(aq,mask) if v];delta=macro(sub,reward(sub,arrays['activity']['count'][mask])-reward(sub,arrays['activity']['current'][mask]));assert abs(delta-result['activity_byG'][str(g)]['delta'])<1e-12;risk[str(g)]=dict(players=len({a['player_code'] for a in sub}),delta=delta,byRank=result['activity_byG'][str(g)]['byRank'],byFold=result['activity_byG'][str(g)]['byFold'])
 dump(R/'INDEPENDENT_AUDIT.json',dict(passed=True,source_hashes_valid=True,reports=reports,pooled3022_delta=d,activity_risk=risk,old_methods_not_reclassified=True,research_go=result['research_candidate_go'],strict_go=result['strict_upgrade_go'],results_sha256=sha(R/'RESULTS.json'),audit_code_sha256=sha(Path(__file__)),no_new_fitting_or_quality_selection=True));print('COUNT INDEPENDENT AUDIT PASSED',flush=True)
def export():
 assert load(R/'RESULTS.json')['research_candidate_go'] and load(R/'INDEPENDENT_AUDIT.json')['passed'];assert not OUT.exists();contract=load(B/'runs/day1_preparation_20261104_v1/OFFICIAL_RELEASE_CONTRACT_20261104.json');sources={};qpaths=[]
 for item in contract['question_files']+[contract['format_evidence']]:p=Path(item['path']);assert sha(p)==item['sha256'];sources[str(p.relative_to(B))]=sha(p)
 qpaths=[Path(x['path']) for x in contract['question_files']];rawids=[];Gs=[]
 for p in qpaths:
  with p.open(encoding='utf-8-sig',newline='') as f:
   for row in csv.DictReader(f):rawids.append(row['question_id']);Gs.append(int(row['num_games']))
 q=load(B/'data/dense_official_inputs_20261106_v1/queries.json');ids=[a['question_id'] for a in q];assert ids==rawids and [a['num_games'] for a in q]==Gs;assert len(ids)==contract['actual_query_count'] and len(set(ids))==len(ids)
 receipt=load(O/'go_rank_D64_HumanSL_profile5_parent5_shrink50_RESEARCH_20261106_v1_RECEIPT.json')
 for rel,h in receipt['source_sha256'].items():assert sha(B/rel)==h;sources[rel]=h
 sys.path[:0]=[str(B/'src'),str(B/'tools')];spec=importlib.util.spec_from_file_location('cnn_frozen_pool',B/'runs/overnight_rank_20261106_v1/cnn_comparison_helpers.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);cnn,_=m.read_cnn(q,[B/f'runs/official_cnn_20261106_v1/original_seed{s}_base.npz' for s in SEEDS],[B/f'runs/official_cnn_20261106_v1/original_seed{s}_Q.npz' for s in SEEDS]);sp=[];parent=[];profile=[]
 def posterior(p):
  a=np.load(p);assert np.array_equal(a['question_ids'],ids);v=a['probabilities'];assert v.shape==(len(ids),10) and np.isfinite(v).all() and np.max(abs(v.sum(-1)-1))<1e-5;sources[str(p.relative_to(B))]=sha(p);return v
 for s in SEEDS:sp.append(posterior(B/f'runs/official_cnn_20261106_v1/sp_official_seed{s}.npz'));parent.append(posterior(B/f'runs/official_model_export_20261106_v1/D64_seed{s}_official.npz'))
 for s in (17,31415):parent.append(posterior(O/f'parent5_CE_seed{s}_official.npz'))
 for s in (42,123,2026,17,31415):profile.append(posterior(O/f'profile5_CE_seed{s}_official.npz'))
 ce=.5*np.mean(parent,0)+.5*np.mean(profile,0);components=np.stack([cnn,np.mean(sp,0),ce],-2);baseline=softmax(np.sum(np.log(components.clip(1e-30))*W[:,None],axis=-2),-1);best=np.load(O/'go_rank_D64_HumanSL_profile5_parent5_shrink50_RESEARCH_20261106_v1_posteriors.npz');assert np.array_equal(best['question_ids'],ids);err=float(abs(baseline-best['probabilities']).max());assert err<2e-6 and np.array_equal(baseline.argmax(-1),best['probabilities'].argmax(-1))
 fit=load(R/'full/FITS.json')['count'];s=np.array(fit['coefficients']);p=independent(components,np.array(Gs),s);lam=np.log(np.array(Gs)/5)/np.log(4);coef=(1-lam[:,None])*s[None,:3]+lam[:,None]*s[None,3:];direct=softmax(np.einsum('nck,nc->nk',np.log(components.clip(1e-30)),coef*W),axis=-1);assert abs(p-direct).max()<1e-12 and np.array_equal(p.argmax(-1),direct.argmax(-1));pred=LABELS[p.argmax(-1)];OUT.mkdir();name='go_rank_D64_profile5_parent5_logG_cal6_RESEARCH_20261107_v1.csv';target=OUT/name
 with target.open('w',encoding='utf-8-sig',newline='') as f:w=csv.writer(f);w.writerow(['question_id','pred_rank']);w.writerows(zip(ids,pred))
 from validate_submission import validate_submission
 valid=validate_submission(target,qpaths);assert valid.valid,valid.errors
 with target.open(encoding='utf-8-sig',newline='') as f:back=list(csv.DictReader(f))
 assert [a['question_id'] for a in back]==ids and [a['pred_rank'] for a in back]==pred.tolist() and target.read_bytes().startswith(b'\xef\xbb\xbf');sample=Path(contract['format_evidence']['path'])
 with sample.open(encoding='utf-8-sig',newline='') as f:sample_ids=[a['question_id'] for a in csv.DictReader(f)]
 assert ids==sample_ids;np.savez_compressed(OUT/(target.stem+'_posteriors.npz'),probabilities=p,components=components,coefficients=s,question_ids=ids,num_games=Gs);result=load(R/'RESULTS.json');changed=np.flatnonzero(pred!=LABELS[baseline.argmax(-1)]);byG={str(g):int(np.sum((pred!=LABELS[baseline.argmax(-1)])&(np.array(Gs)==g))) for g in sorted(set(Gs))}
 record=dict(completed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CSV=str(target),CSV_sha256=sha(target),classification='RESEARCH_CANDIDATE_NOT_CONFIRMED_UPGRADE',research_gate=True,strict_upgrade_gate=result['strict_upgrade_go'],baseline_reproduction_max_error=err,baseline_hard_labels_exact=True,rows=len(ids),columns=['question_id','pred_rank'],UTF8_BOM=True,actual_sample_order_exact=True,format_valid=True,independent_forward_exact=True,changed_vs_best=len(changed),changed_byG=byG,recipe='Currentclean5529 parent5/profile5 CE halfmixture; logGinterpolatedsourcecal6 fittedcal400 only; fixedargmaxT1. No SE components, no refit/assessed500absorption.',full_coefficients=s.tolist(),local_dev_gain=result['pooled3022']['delta'],local_dev_CI=result['pooled3022']['CI'],cumulative52_CI=result['pooled3022']['cumulative52_CI'],unresolved_risks=['All dev reused, no fresh confirmation/Public forecast','Activity G15 1D delta -0.025641 (35players); strict finer-grained gate failed','Activity G5 fold0 delta negative; aggregateNI only, not allshortfoldpositive','SourceNNdirection/metadata/context limitations inherited; only fusion changed','Three matched sensitivity streams, not new trained backbone seeds'],sources_sha256=sources,run_protocol_sha256=sha(R/'PROTOCOL.json'),run_results_sha256=sha(R/'RESULTS.json'),audit_sha256=sha(R/'INDEPENDENT_AUDIT.json'),export_code_sha256=sha(Path(__file__)),no_upload=True,no_lockbox_confirm100_reserved80_Publiclabels=True,manual_label_edits=False,official_score='UNKNOWN')
 dump(OUT/(target.stem+'_RECEIPT.json'),record);dump(R/'EXPORT_COMPLETE.json',record);print('COUNT CSV EXPORTED',str(target),sha(target),'CHANGED',len(changed),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

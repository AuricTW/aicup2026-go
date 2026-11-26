"""One count-conditioned convex calibration hypothesis; no temperature/weight grid."""
from pathlib import Path
import os,sys,json,hashlib,time,datetime,importlib.util
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
B=Path(__file__).resolve().parents[2];R=Path(__file__).resolve().parent;C=B/'runs/current_recipe_calibration_20261107_v1';CC=B/'runs/component_calibration_20261107_v1';A=B/'runs/activity_replication_20261107_v1';F=B/'runs/fullsystem_oof_20261106_v1';E=B/'runs/extra_player_audit_20261107_v1';W=np.array([.5625,.1875,.25])
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def design(streams,G,count=True):
 x=np.log(streams.clip(1e-30))*W[:,None];x=x.transpose(*range(x.ndim-2),x.ndim-1,x.ndim-2)
 if not count:return x
 lam=np.log(np.asarray(G)/5)/np.log(4);return np.concatenate([x*(1-lam)[...,None,None],x*lam[...,None,None]],axis=-1)
def penalty(s,count):
 if not count:return .0025*np.mean((s-1)**2),2*.0025*(s-1)/3
 a,b=s[:3],s[3:];mid=(a+b)/2;half=(b-a)/2;v=.0025*np.mean((mid-1)**2)+.025*np.mean(half**2);ga=.0025*(mid-1)/3-.025*half/3;gb=.0025*(mid-1)/3+.025*half/3;return v,np.r_[ga,gb]
def objective(s,x,y,count):
 z=x@s;p=softmax(z,-1);v,g=penalty(s,count);loss=np.mean(logsumexp(z,-1)-z[np.arange(len(y)),y])+v;d=p.copy();d[np.arange(len(y)),y]-=1;grad=np.einsum('nk,nkd->d',d,x)/len(y)+g;return float(loss),grad
def apply(streams,G,s,count):return softmax(design(streams,G,count)@np.asarray(s),-1)
def freeze():
 assert not (R/'PROTOCOL.json').exists();files=[Path(__file__),C/'INDEPENDENT_INPUT_AUDIT.json',C/'INDEPENDENT_STATISTICS_AUDIT.json',CC/'CAL_QUERIES_RAW.json']
 dump(R/'PROTOCOL.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Before ownfits/quality; parallelCPUwhileunrelatedSEGPUextracts',hypothesis='CurrentCNN CountRoutechangesaggregationandquerymodelatG10; fixedsourcecoefficients may not reflect evidence growth. Test ONE smoothlogG-dependentpositive3source scaling, not new Go information. Previousconstant3NO_GO remains.',data='Sameclean5529currenthalfparent5/profile5 CE, CNN/SPfixed. Cal400/1600qfittingONLY, no otherlabelsfit. Entireproperheldactor/raw/D4exclusion inheritedandbound.',model='Sixendpointcoeffs(lo3,hi3),scale_j(G)=(1-lambda)*lo_j+lambda*hi_j,lambda=log(G/5)/log4. Bounds.25-4. ConvexNLLensembleonly +.0025mean(((lo+hi)/2-1)^2)+.025mean(((hi-lo)/2)^2). Sharedconstant3controlsame.0025prior. 3folds+fullallfitbeforequality, FDgradient+independentSLSQP. No Ggrid/classbias/utilitydecoder/trainingseedsearch.',research_gate=dict(pooled3022_gain_current=.003,local_Bonf2_CI_lower_current_positive=True,count_vs_constant_pooled_point_min=-.001,count_vs_constant_CI_lower_above=-.003,each_OLD_EXTRA_point_min=0,allfold_positive=True,two_of_three_paired_streams_positive_eachfold=True,each_G_floor=-.005,protected_ranks_eachpopulation_floor=-.01,any_rank_eachpopulation_floor=-.02,NLL_ceiling=.005,activity_eachG_point_min=-.005,activity_G5_CI95_lower_above=-.01),strict_diagnostic='Additionallycumulative52primaryCIpositive and allGxrankfloors andG5strongNI. Not requiredtooffer explicitlyRESEARCHcandidate, not overwriteconfirmedincumbent.',prospective_change_disclosure='Newcountconditionalmethodresearchgatefixedbeforeownoutcomes. Priorconstantcal/entropy/trimNO_GO unchanged, no oldfailureexport. Local2notglobalhistoricalFWER, reportcumulative52CIandallperGclassdeltas. Reuseddevnotconfirmation;Gcountnotobservablecareeractivitysoactivitydistributionconfoundingremains.',conditional='Onlynewcount6researchgatepasses->fullfrozenfit onallowedcal400 alreadydone; verify sources/contracts/code replay and uniqueCSV. Cal400fittingnotcompatibilityvalidation. No neuralupdates, sourcecandidatechanges, protectedscore/Publiclabels/upload.',sources={str(p.relative_to(B)):sha(p) for p in files}))
 print('COUNT CALIBRATION PROTOCOL FROZEN',flush=True)
def bind():assert all(sha(B/p)==h for p,h in load(R/'PROTOCOL.json')['sources'].items())
def fit():
 bind();assert not (R/'FITS_COMPLETE.json').exists();q=load(CC/'CAL_QUERIES_RAW.json');G=np.array([a['num_games'] for a in q]);y=np.array([a['rank_idx'] for a in q]);reports={};fd=[]
 for stage in ('fold0','fold1','fold2','full'):
  folder=R/stage;folder.mkdir();v=np.load(C/stage/'cal.npz');assert np.array_equal(v['question_ids'],[a['question_id'] for a in q]);out={}
  for name,count in [('constant',False),('count',True)]:
   x=design(v['streams'][0],G,count);n=6 if count else 3;s=np.ones(n);g=objective(s,x,y,count)[1];numeric=np.array([(objective(s+np.eye(n)[j]*1e-5,x,y,count)[0]-objective(s-np.eye(n)[j]*1e-5,x,y,count)[0])/2e-5 for j in range(n)]);err=float(abs(g-numeric).max());assert err<1e-7;opt=minimize(objective,s,args=(x,y,count),method='L-BFGS-B',jac=True,bounds=[(.25,4)]*n,options=dict(ftol=1e-14,gtol=1e-9,maxiter=1000));assert opt.success;alt=minimize(lambda s:objective(s,x,y,count)[0],s,method='SLSQP',bounds=[(.25,4)]*n,options=dict(ftol=1e-12,maxiter=1000));assert alt.success;diff=float(abs(opt.x-alt.x).max());assert diff<.001
   if not count:old=np.array(load(C/stage/'FITS.json')['joint']['scales']);assert abs(old-opt.x).max()<1e-4
   out[name]=dict(coefficients=opt.x.tolist(),objective=float(opt.fun),finite_difference_error=err,independent_optimizer_coefficient_error=diff,input_sha256=sha(C/stage/'cal.npz'),fitting_players=400,fitting_queries=1600)
  dump(folder/'FITS.json',out);reports[stage]=out
 dump(R/'FITS_COMPLETE.json',dict(completed=True,all4stages_before_quality=True,models=8,reports=reports));print('COUNT ALL FOUR FITS COMPLETE',flush=True)
def analyse():
 bind();assert (R/'FITS_COMPLETE.json').exists() and not (R/'RESULTS.json').exists();spec=importlib.util.spec_from_file_location('stats',A/'analyse.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats);stats.FAMILY=2;paths={'old':F/'input/queries.json','extra':E/'QUERIES.json','activity':A/'QUERIES.json'};qq={};pp={};ss={};reports={}
 for c,path in paths.items():
  q=load(path);qq[c]=q;p={k:np.empty((len(q),10)) for k in ('current','constant','count')};sp={k:np.empty((3,len(q),10)) for k in p}
  for f in range(3):
   ix=np.flatnonzero([a['fold']==f for a in q]);v=np.load(C/f'fold{f}/{c}.npz');assert np.array_equal(v['question_ids'],[q[i]['question_id'] for i in ix]);G=np.array([q[i]['num_games'] for i in ix]);fits=load(R/f'fold{f}/FITS.json')
   for k in p:s=np.ones(3) if k=='current' else fits[k]['coefficients'];out=apply(v['streams'],G,s,k=='count');p[k][ix]=out[0];sp[k][:,ix]=out[1:]
  pp[c]=p;ss[c]=sp;rep=stats.compare(q,p['count'],p['current'],sp['count'],sp['current'],-.005 if c=='activity' else 0,'secondary');reports[c]=dict(metrics={k:stats.metric(q,v) for k,v in p.items()},comparison=rep);np.savez_compressed(R/(c.upper()+'_FUSED.npz'),**p,**{'seed_'+k:v for k,v in sp.items()},question_ids=[a['question_id'] for a in q]);print('COUNT_RESULT',c,rep['delta'],rep['gates'],flush=True)
 q=qq['old']+qq['extra'];p={k:np.concatenate([pp[c][k] for c in ('old','extra')]) for k in pp['old']};sp={k:np.concatenate([ss[c][k] for c in ('old','extra')],axis=1) for k in ss['old']};pooled=stats.compare(q,p['count'],p['current'],sp['count'],sp['current'],.003,'primary');d=stats.reward(q,p['count'])-stats.reward(q,p['constant']);control=dict(delta=stats.point(q,d),CI=stats.boot(q,d));control['go']=control['delta']>=-.001 and control['CI'][0]>-.003;stats.FAMILY=52;pooled['cumulative52_CI']=stats.boot(q,stats.reward(q,p['count'])-stats.reward(q,p['current']));stats.FAMILY=2;activity={};aq=qq['activity'];G=np.array([a['num_games'] for a in aq]);strictG=True
 for g in (5,10,15):
  mask=G==g;sub=[a for a,v in zip(aq,mask) if v];z=stats.compare(sub,pp['activity']['count'][mask],pp['activity']['current'][mask],ss['activity']['count'][:,mask],ss['activity']['current'][:,mask],-.005,'secondary');strictG&=z['go'];z['legacy_perG_gates']=z['gates'];z['gates']={k:v for k,v in z['gates'].items() if k in ('minimum_delta','G_floor','NLL_floor')}
  if g==5:stats.FAMILY=1;z['CI95']=stats.boot(sub,stats.reward(sub,pp['activity']['count'][mask])-stats.reward(sub,pp['activity']['current'][mask]));stats.FAMILY=2;z['gates']['NI95_lower_above_minus001']=z['CI95'][0]>-.01
  z['go']=all(z['gates'].values());activity[str(g)]=z
 go=pooled['go'] and control['go'] and all(v['comparison']['go'] for v in reports.values()) and all(v['go'] for v in activity.values());strict=go and strictG and pooled['cumulative52_CI'][0]>0;dump(R/'RESULTS.json',dict(completed=True,research_candidate_go=go,strict_upgrade_go=strict,reports=reports,pooled3022=pooled,count_vs_constant=control,activity_byG=activity,limits='Conditional reuseddev only; oldconstantNO_GO remains. CountdependentfusionnotnewGo signal, no Publicforecast or protectedscore/upload.'));print('COUNT_RESEARCH_GATE',go,'STRICT',strict,'GAIN',pooled['delta'],'CI',pooled['CI'],flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

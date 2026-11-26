"""Independent new600 proxy; reject identity-exposed legacy parent qualification.

The proxy deliberately omits legacy learned MLP/U0/Gaussian features. It uses
only raw style histograms, frozen external HumanSL features and train-only
opening background; all fitted statistics exclude full600 owners/families.
It is not byte-identical to the deployed raw32 ensemble. Any final fusion
weight transfer remains a challenger hypothesis requiring official feedback.
"""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import q1_fresh600_dense64_20261111_v1 as q
import v2_color_plda_20261111_v1 as cp
from goid_v4.score_push_features_v1 import design,BINS
from scipy import sparse
import numpy as np
from threadpoolctl import threadpool_limits
p=q.p;RUN=q.RUN;DEST=RUN/'clean_proxy'

def prepare():
    DEST.mkdir(exist_ok=False);audit=p.b.read(RUN/'PREQUALITY_EXPOSURE_AUDIT.json');assert audit['600_candidate_owner_overlap_with_fitted_identities']['legacy_production_colorPLDA']==600
    p.b.write(RUN/'REJECTED_LEGACY_PARENT_QUALIFICATION.json',{'before_any_new_quality':True,'reason':'productioncolorPLDAfitincludesall600validationidentitiesand8000selectedgames; H8includes29identities','retained_path':str(RUN/'baseline_stat'),'do_not_use_for_gate_or_selection':True,'fit_or_eval_models_not_yet_scored':True})
    m=p.b.meta();new=set(p.b.read(RUN/'EPISODE_screen.json')['candidates']);old600=set(p.b.read(p.b.RUN/'calibration/ep0.json')['candidates']);old723=set(m['owners'][np.load(p.DENSE.parent/'validation_indices.npy')].astype(str));excluded=np.isin(m['owners'].astype(str),sorted(new|old600|old723));ok=(~excluded)&(m['hd']<=1)
    for f in p.b.FAMILIES:ok &= ~np.isin(m[f],np.unique(m[f][excluded]))
    fit=np.flatnonzero(ok);np.save(DEST/'fit_indices.npy',fit,allow_pickle=False);old8=np.load(p.n.RUN/'train_indices.npy');h8=old8[ok[old8]];np.save(DEST/'H8_fit_indices.npy',h8,allow_pickle=False);assert len(fit)>30000 and len(h8)>256 and len(set(m['owners'][h8]))>128
    p.b.write(RUN/'CLEAN_BASELINE_ADDENDUM.json',{'created_utc':p.b.base.utc(),'before_all_quality':True,'source_SHA256':p.b.sha(__file__),'rejected_old_parent_not_deleted':True,'original_gates_numeric_thresholds_unchanged':True,
      'new_fusion_gate_reference':'CLEAN_PROXY, not identity-exposed deployedraw32parent','reason':'stop identity contamination; componentmatched32vs64 remains primary independent evidence',
      'clean_proxy_recipe':'0.6*(0.75F_clean+0.25RRF60(H8_clean))+0.4RRF60(newmatchedraw32_128)',
      'F_clean':'RRF60(RRF60(globalrawhistPLDA256,colorrawhistPLDA256),opening40D4sequencewithtrainonlybackgroundalpha20)',
      'hist_fit_rows':len(fit),'hist_fit_players':len(set(m['owners'][fit])),'H8_refit_rows':len(h8),'H8_refit_players':len(set(m['owners'][h8])),
      'hist_feature':'sqrt1582 normalizedrawnonlearnedhistograms; no frozenstatMLP/U0/Gaussian features','H8':'refitpreprocess+PLDA128 ononlycleanhistoricalnative8trainingrows',
      'all_fitted_assets_exclude_full600identity_and4families_plushistorical723_andold600':True,
      'candidate_final_export_if_pass':'samefrozenw fusedwithactualraw32, onlynewQ1; evidence is proxy+pairedmetric, finalweighttransfer not independently validated onactualproductionparent',
      'must_call_submission':'challenger, not independentlyconfirmedproductiongain','no_new_quality_read_or_Publiclabels':True})
    p.b.log(stage='clean_proxy_prepared',hist_rows=len(fit),H8_rows=len(h8))

def build():
    m=p.b.meta();ids=np.load(DEST/'fit_indices.npy');schema=p.b.read(p.ROOT/'data/processed/feature_schema.json');raw=np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r');folder=DEST/'models';folder.mkdir(exist_ok=False)
    xx=np.lib.format.open_memmap(DEST/'hist_fit.npy',mode='w+',dtype=np.float32,shape=(len(ids),1582))
    for st in range(0,len(ids),2048):xx[st:st+2048]=cp.histograms(raw[m['rows'][ids[st:st+2048]]],schema,sqrt=True)
    xx.flush();report=p.fit_plda(xx,m['owners'][ids],folder/'global_PLDA.npz',dimension=256);p.b.write(folder/'global_FIT.json',report)
    for col,name in enumerate('BW'):
        at=np.flatnonzero(m['colors'][ids]==col);report=p.fit_plda(xx[at],m['owners'][ids[at]],folder/f'PLDA_{name}.npz',dimension=256);p.b.write(folder/f'{name}_FIT.json',report)
    bg=np.full(3*BINS,1e-3,np.float64)
    for st in range(0,len(ids),2048):
        ix=ids[st:st+2048];codes=p.b.sequence_export(ix,m)
        for row in np.flatnonzero(m['colors'][ix]==1):valid=codes[row]!=65535;codes[row,valid]=(codes[row,valid]+362)%724
        x=design(codes,'joint_order_D4');bg+=np.asarray(x.sum(0)).ravel()
    for g in range(3):bg[g*BINS:(g+1)*BINS]/=bg[g*BINS:(g+1)*BINS].sum()
    np.save(folder/'opening_background.npy',bg,allow_pickle=False)
    h8=np.load(DEST/'H8_fit_indices.npy');cache=p.n.RUN/'feature_cache';receipt=p.b.read(cache/'RECEIPT.json');assert p.b.sha(cache/'vectors.npy')==receipt['vectors_sha256'];cachedids=np.load(cache/'indices.npy');lookup={int(i):j for j,i in enumerate(cachedids)};vectors=np.load(cache/'vectors.npy',mmap_mode='r');v=np.stack([vectors[lookup[int(i)]] for i in h8]);hfolder=folder/'H8';hfolder.mkdir();mean=np.asarray(v,np.float64).mean(0);std=np.asarray(v,np.float64).std(0);active=std>1e-4;np.savez(hfolder/'preprocess.npz',mean=mean,std=np.where(active,std,1.),active=active);x=q.oldtrain.pre(v,hfolder);report=p.fit_plda(x,m['owners'][h8],hfolder/'PLDA.npz',dimension=128);p.b.write(hfolder/'FIT.json',report)
    p.b.write(folder/'COMPLETION.json',{'all_clean_fits_complete':True,'full600_identity_family_excluded':True,'no_quality':True});p.b.log(stage='clean_proxy_fits_complete')

def sequence_scores(codes,colors,ss,qs):
    codes=codes.copy()
    for row in np.flatnonzero(colors==1):valid=codes[row]!=65535;codes[row,valid]=(codes[row,valid]+362)%724
    x=design(codes,'joint_order_D4');bg=np.load(DEST/'models/opening_background.npy');bank=sparse.vstack([sparse.csr_matrix(x[sorted(s)].sum(0)) for s in ss],format='csr',dtype=np.float64);norm=np.stack([np.asarray(bank[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(3)],1);bank.data=np.log1p(bank.data/(20*bg[bank.indices]));used=sorted({i for z in qs for i in z});qx=x[used];mass=np.stack([np.asarray(qx[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(3)],1);values=(qx@bank.T).toarray()-mass@np.log1p(norm/20).T;lookup={i:j for j,i in enumerate(used)};return np.stack([values[[lookup[i] for i in sorted(z)]].mean(0)/3 for z in qs])

def stat_predictions():
    m=p.b.meta();ids=np.load(RUN/'cal_indices.npy');raw=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ids]],np.float32);hist=cp.histograms(raw,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True);model=p.PLDA(DEST/'models/global_PLDA.npz');z=model.transform(hist);pos={int(i):j for j,i in enumerate(ids)};codes=p.b.sequence_export(ids,m);dest=DEST/'predictions';dest.mkdir(exist_ok=False)
    for part in ('screen','confirmation'):
        ep=p.b.read(RUN/f'EPISODE_{part}.json');ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in v['indices']] for v in ep['questions']];sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);global_score=model.score(qm,qn,sm,sn);color=cp.color_score(raw,m['colors'][ids],ss,qs,DEST/'models');seq=sequence_scores(codes,m['colors'][ids],ss,qs);fc=cp.fixed_rrf60(cp.fixed_rrf60(global_score,color),seq);np.save(dest/f'F_C_{part}.npy',fc,allow_pickle=False)
    p.b.write(dest/'RECEIPT.json',{'no_quality':True,'all_fit_assets_clean':True,'deployedraw32recipe_not_equal':True});p.b.log(stage='clean_proxy_stat_ready')

def parent():
    ids=np.load(RUN/'cal_indices.npy');lookup={int(i):j for j,i in enumerate(ids)};model=p.PLDA(DEST/'models/H8/PLDA.npz');z8=model.transform(q.oldtrain.pre(np.load(RUN/'views/cal_original8.npy'),DEST/'models/H8'));m32=p.PLDA(RUN/'models/raw128_v32/PLDA.npz');z32=m32.transform(q.oldtrain.pre(np.load(RUN/'views/cal32.npy'),RUN/'models/raw128_v32'));dest=RUN/'parent';dest.mkdir(exist_ok=False)
    for part in ('screen','confirmation'):
        ep=p.b.read(RUN/f'EPISODE_{part}.json');ss=[[lookup[i] for i in s] for s in ep['supports']];qs=[[lookup[i] for i in v['indices']] for v in ep['questions']];sm,sn=p.summarize(z8,ss);qm,qn=p.summarize(z8,qs);k=model.score(qm,qn,sm,sn);sm,sn=p.summarize(z32,ss);qm,qn=p.summarize(z32,qs);r=m32.score(qm,qn,sm,sn);fc=np.load(DEST/f'predictions/F_C_{part}.npy');value=.6*(.75*fc+.25/(60+p.b.ranks(k)))+.4/(60+p.b.ranks(r));np.save(dest/f'{part}.npy',value,allow_pickle=False);np.save(dest/f'prior_{part}.npy',.75*fc+.25/(60+p.b.ranks(k)),allow_pickle=False);np.save(dest/f'raw32_{part}.npy',r,allow_pickle=False)
    p.b.write(dest/'RECEIPT.json',{'scope':'cleanmatchedproxy, NOT deployedraw32','identity_family_clean_for_all_fit_assets':True,'model_includes_no_oldMLP_U0_Gaussian':True,'quality_not_computed':True})

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','build','predict','parent']);v=a.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'build':build,'predict':stat_predictions,'parent':parent}[v.stage]()

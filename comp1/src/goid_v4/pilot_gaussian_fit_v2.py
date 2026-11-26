import json,time
from .pilot_train_v1 import load_fit,fit_plan,metric
from .pilot_scope_v1 import ROOT,RUN,sha,write
from .pilot_gaussian_v2 import JointRandomEffects
from .pilot_gaussian_v1 import summaries
import numpy as np


def main():
    config=json.loads((ROOT/'configs/research_v4/p1_pilot_v1_20261026.json').read_text(encoding='utf-8'))
    out=RUN/'models/U1_joint_random_effects_v2';out.mkdir(exist_ok=False)
    x,records,train_records=load_fit();n=len(train_records);colors=np.asarray([r['color'] for r in records])
    start=time.perf_counter();model=JointRandomEffects();model.fit(x[:n],colors[:n],[r['player'] for r in train_records],dimension=128)
    z=model.transform(x);order,supports,queries,targets,qcounts=fit_plan(records)
    ss=summaries(z,colors,supports);qs=summaries(z,colors,queries)
    grid=[];best=None
    for prior in config['calibration']['random_effects_prior_scale_grid']:
        for noise in config['calibration']['random_effects_noise_scale_grid']:
            model.prior_scale=prior;model.noise_scale=noise
            metrics=metric(model.score(qs,ss),targets,qcounts);short=(metrics['5']+metrics['10'])/2
            row={'prior_scale':prior,'noise_scale':noise,'fit_inner_scores':metrics,'short_score':short};grid.append(row)
            if best is None or short>best['short_score']:best=row
    model.prior_scale=best['prior_scale'];model.noise_scale=best['noise_scale'];model.save(out/'model.npz')
    write(out/'calibration.json',{'selected':best,'grid':grid,'selection_scope':'fit-inner100 only','fit_observation_counts':model.fit_observation_counts,'development_scores_read':False})
    write(out/'result.json',{'status':'complete_fit_only','deterministic_no_training_seed_variation':True,'model_sha256':sha(out/'model.npz'),
          'source_code_sha256':sha(ROOT/'src/goid_v4/pilot_gaussian_v2.py'),'elapsed_seconds':time.perf_counter()-start,'development_evaluated':False,
          'change_reason':'fix missing-stratum estimator; no data reselection or new hyperparameter grid'})
    print(json.dumps({'stage':'U1_v2_fit_complete','elapsed_seconds':time.perf_counter()-start,'fit_observation_counts':model.fit_observation_counts,'selected':best}),flush=True)


if __name__=='__main__':main()

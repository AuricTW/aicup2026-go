"""Missing-color-correct fit estimator; v1 failure preserved, selected data unchanged."""
import numpy as np
from scipy.linalg import eigh
from .pilot_gaussian_v1 import JointRandomEffects as Original,summaries
from .artifacts import ContractError


class JointRandomEffects(Original):
    def fit(self,features,colors,players,*,dimension=128):
        x=np.asarray(features,np.float64);self.center=x.mean(0);centered=x-self.center
        cov=centered.T@centered/(len(x)-1)
        values,vectors=eigh(cov,subset_by_index=[cov.shape[0]-dimension,cov.shape[0]-1],check_finite=True)
        if values.min()<=0:raise ContractError('PCA active spectrum not positive')
        self.basis=vectors/np.sqrt(values)[None];z=centered@self.basis
        unique=sorted(set(players));p=np.asarray(players);sets=[np.flatnonzero(p==who).tolist() for who in unique]
        mu,n=summaries(z,colors,sets)
        self.prior_mean=np.zeros((2,dimension));self.noise=np.zeros((2,dimension));diagonal=np.zeros((2,dimension))
        for c in (0,1):
            observed=n[:,c]>0
            if observed.sum()<2:raise ContractError('Population color moments unidentifiable')
            self.prior_mean[c]=mu[observed,c].mean(0)
            residual=np.zeros(dimension);df=0
            for i,ids in enumerate(sets):
                cc=np.asarray(ids)[colors[ids]==c]
                if len(cc)>1:
                    residual+=((z[cc]-mu[i,c])**2).sum(0);df+=len(cc)-1
            if df<1:raise ContractError('Population within-color noise unidentifiable')
            self.noise[c]=np.maximum(residual/df,1e-6)
            observed_variance=mu[observed,c].var(axis=0,ddof=1)
            diagonal[c]=np.maximum(observed_variance-(self.noise[c][None]/n[observed,c,None]).mean(0),1e-5)
        paired=(n>0).all(1)
        if paired.sum()>=2:
            a=mu[paired,0]-mu[paired,0].mean(0);b=mu[paired,1]-mu[paired,1].mean(0)
            shared=np.maximum((a*b).sum(0)/(paired.sum()-1),0)
        else:shared=np.zeros(dimension)
        self.K=np.zeros((2,2,dimension));self.K[0,0]=np.maximum(diagonal[0],shared+1e-5);self.K[1,1]=np.maximum(diagonal[1],shared+1e-5)
        self.K[0,1]=self.K[1,0]=shared;self.prior_scale=self.noise_scale=1.
        self.fit_observation_counts={'players':len(unique),'paired_players':int(paired.sum()),'missing_color_players':int((n==0).any(1).sum()),'no_player_dropped':True}
        return z

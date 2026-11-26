"""Fit-only whitened PCA + joint B/W Gaussian random-effects predictive model.

Each latent coordinate has a2x2 color covariance (shared identity + color
offset), fitted observation noise, and exact conjugate posterior covariance.
Absent colors are unobserved; a one-game stratum uses fitted noise.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from .artifacts import ContractError


def summaries(vectors,colors,sets):
    means=np.zeros((len(sets),2,vectors.shape[1]),np.float64)
    counts=np.zeros((len(sets),2),np.int64)
    for i,ids in enumerate(sets):
        ids=np.asarray(ids,np.int64)
        if not len(ids) or len(np.unique(ids))!=len(ids):raise ContractError('Invalid Gaussian set')
        for c in (0,1):
            cc=ids[colors[ids]==c];counts[i,c]=len(cc)
            if len(cc):means[i,c]=np.asarray(vectors[cc],np.float64).mean(0)
    return means,counts


class JointRandomEffects:
    def fit(self,features,colors,players,*,dimension=128):
        x=np.asarray(features,np.float64)
        self.center=x.mean(0);centered=x-self.center
        cov=centered.T@centered/(len(x)-1)
        values,vectors=eigh(cov,subset_by_index=[cov.shape[0]-dimension,cov.shape[0]-1],check_finite=True)
        if values.min()<=0:raise ContractError('PCA active spectrum not positive')
        self.basis=vectors/np.sqrt(values)[None]
        z=centered@self.basis
        unique=sorted(set(players));sets=[np.flatnonzero(np.asarray(players)==p).tolist() for p in unique]
        mu,n=summaries(z,colors,sets)
        if (n<2).any():raise ContractError('Fit random effects needs >=2 games of each color per player')
        self.prior_mean=mu.mean(0)
        residual=np.zeros((2,dimension));df=np.zeros(2)
        for i,ids in enumerate(sets):
            for c in (0,1):
                cc=np.asarray(ids)[colors[ids]==c]
                residual[c]+=((z[cc]-mu[i,c])**2).sum(0);df[c]+=len(cc)-1
        self.noise=np.maximum(residual/df[:,None],1e-6)
        centered_mu=mu-self.prior_mean[None]
        observed=(centered_mu**2).sum(0)/(len(mu)-1)
        diagonal=np.maximum(observed-(self.noise[None]/n[:,:,None]).mean(0),1e-5)
        shared=np.maximum((centered_mu[:,0]*centered_mu[:,1]).sum(0)/(len(mu)-1),0)
        # Nonnegative shared component plus positive color-specific components -> SPD.
        self.K=np.zeros((2,2,dimension))
        self.K[0,0]=np.maximum(diagonal[0],shared+1e-5)
        self.K[1,1]=np.maximum(diagonal[1],shared+1e-5)
        self.K[0,1]=self.K[1,0]=shared
        self.prior_scale=self.noise_scale=1.
        return z

    def transform(self,features):return (np.asarray(features,np.float64)-self.center)@self.basis

    def posterior(self,summary):
        mean,n=summary;K=self.K*self.prior_scale;noise=self.noise*self.noise_scale
        determinant=K[0,0]*K[1,1]-K[0,1]**2
        if (determinant<=0).any():raise ContractError('Prior covariance not positive')
        k00=K[1,1]/determinant;k11=K[0,0]/determinant;k01=-K[0,1]/determinant
        a=k00+n[:,0,None]/noise[0];d=k11+n[:,1,None]/noise[1];b=k01
        det=a*d-b*b
        C00=d/det;C11=a/det;C01=-b/det
        h0=k00*self.prior_mean[0]+k01*self.prior_mean[1]+n[:,0,None]*mean[:,0]/noise[0]
        h1=k01*self.prior_mean[0]+k11*self.prior_mean[1]+n[:,1,None]*mean[:,1]/noise[1]
        return (C00*h0+C01*h1,C01*h0+C11*h1),(C00,C01,C11)

    def score(self,query,support):
        qm,qn=query;(m0,m1),(C00,C01,C11)=self.posterior(support)
        noise=self.noise*self.noise_scale
        scores=np.empty((len(qm),len(m0)),np.float64)
        for i in range(len(qm)):
            if qn[i,0] and qn[i,1]:
                a=C00+noise[0]/qn[i,0];d=C11+noise[1]/qn[i,1];b=C01
                det=a*d-b*b;x=qm[i,0]-m0;y=qm[i,1]-m1
                scores[i]=-.5*np.sum(np.log(det)+(d*x*x+a*y*y-2*b*x*y)/det,axis=1)
            else:
                c=0 if qn[i,0] else 1
                variance=(C00 if c==0 else C11)+noise[c]/qn[i,c]
                diff=qm[i,c]-(m0 if c==0 else m1)
                scores[i]=-.5*np.sum(np.log(variance)+diff*diff/variance,axis=1)
        if not np.isfinite(scores).all():raise ContractError('Nonfinite random-effects score')
        return scores

    def save(self,path):
        np.savez(path,center=self.center,basis=self.basis,prior_mean=self.prior_mean,
                 noise=self.noise,K=self.K,prior_scale=self.prior_scale,noise_scale=self.noise_scale)

    @classmethod
    def load(cls,path):
        obj=cls()
        with np.load(path,allow_pickle=False) as z:
            for name in ('center','basis','prior_mean','noise','K'):setattr(obj,name,z[name].copy())
            obj.prior_scale=float(z['prior_scale']);obj.noise_scale=float(z['noise_scale'])
        return obj

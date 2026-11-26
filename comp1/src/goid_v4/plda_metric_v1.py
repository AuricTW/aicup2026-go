"""Local covariance PLDA for game-set identity, derived from Gaussian conditioning.

Concept references: Prince/Elder ICCV2007; Kaldi PLDA documentation.
No external implementation or pretrained weights. A query set is independent.
"""
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh

def histograms(raw,schema,*,sqrt=False):
    x=np.array(raw[:,12:1594],np.float32,order='C',copy=True)
    for g in schema['groups']:
        if g['kind']!='histogram':continue
        a,b=g['start']-12,g['stop']-12
        x[:,a:b]/=np.maximum(x[:,a:b].sum(1,keepdims=True),np.float32(1e-12))
    return np.sqrt(x) if sqrt else x

def fit(x,players,path,*,dimension=256,block=2048):
    path=Path(path)
    if path.exists():raise ValueError('Refuse PLDA overwrite')
    t=time.perf_counter();n,d=x.shape;players=np.asarray(players)
    if len(players)!=n or n<dimension+1 or dimension>d:raise ValueError('PLDA dimensions')
    names,codes=np.unique(players,return_inverse=True);counts=np.bincount(codes)
    if len(names)<2 or n<=len(names):raise ValueError('PLDA unidentifiable within-player variation')
    center=np.zeros(d,np.float64)
    for i in range(0,n,block):center+=np.asarray(x[i:i+block],np.float64).sum(0)
    center/=n;total=np.zeros((d,d),np.float64)
    for i in range(0,n,block):
        v=np.asarray(x[i:i+block],np.float64)-center
        if not np.isfinite(v).all():raise ValueError('Nonfinite PLDA features')
        total+=v.T@v
    ev,vec=eigh(total/(n-1),subset_by_index=[d-dimension,d-1]);assert ev.min()>0
    # PCA whiten first for conditioning; subsequent W/B transform carries identity information.
    vec*=np.where(vec[np.argmax(abs(vec),axis=0),np.arange(dimension)]>=0,1,-1)[None]
    pca=vec/np.sqrt(ev)[None];sums=np.zeros((len(names),dimension),np.float64)
    ztz=np.zeros((dimension,dimension),np.float64)
    for i in range(0,n,block):
        z=(np.asarray(x[i:i+block],np.float64)-center)@pca
        np.add.at(sums,codes[i:i+len(z)],z);ztz+=z.T@z
    means=sums/counts[:,None];prior_mean=means.mean(0)
    within=(ztz-(means*counts[:,None]).T@means)/(n-len(names))
    within=(within+within.T)/2
    W=.9*within+.1*np.diag(np.diag(within))+np.eye(dimension)*1e-6
    ew,vw=eigh(W);assert ew.min()>0
    white=(vw/np.sqrt(ew)[None])@vw.T
    centered=means-prior_mean
    between=centered.T@centered/(len(names)-1)-within*np.mean(1/counts)
    whitened=white.T@between@white
    psi,q=eigh((whitened+whitened.T)/2);psi=np.maximum(psi,1e-5)
    q*=np.where(q[np.argmax(abs(q),axis=0),np.arange(dimension)]>=0,1,-1)[None]
    projection=pca@white@q
    # Center by equal-player grand mean in original feature coordinates.
    offset=prior_mean@white@q
    with path.open('xb') as f:np.savez(f,center=center,projection=projection,offset=offset,psi=psi)
    return {'rows':n,'players':len(names),'dimension':dimension,'within_shrinkage':.1,
        'within_min_eigenvalue':float(ew.min()),'between_positive_dimensions':int((psi>1e-5).sum()),
        'between_eigenvalue_max':float(psi.max()),'seconds':time.perf_counter()-t}

class PLDA:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.center=z['center'].copy();self.projection=z['projection'].copy()
            self.offset=z['offset'].copy();self.psi=z['psi'].copy()
        self.matrix=np.array(self.projection.T,np.float64,order='C')
        if not (self.psi>0).all():raise ValueError('PLDA prior variance')
    def transform(self,x):
        z=np.empty((len(x),len(self.psi)),np.float64)
        for i,row in enumerate(x):
            z[i]=self.matrix@np.array(np.asarray(row,np.float64)-self.center,order='C')-self.offset
        if not np.isfinite(z).all():raise ValueError('Nonfinite PLDA projection')
        return z
    def score(self,qmean,qcount,smean,scount):
        qcount=np.asarray(qcount);scount=np.asarray(scount)
        if (qcount<=0).any() or (scount<=0).any():raise ValueError('Empty PLDA set')
        postvar=self.psi[None]/(1+scount[:,None]*self.psi[None])
        postmean=postvar*scount[:,None]*smean
        out=np.empty((len(qmean),len(smean)),np.float64)
        for i,(qm,n) in enumerate(zip(qmean,qcount)):
            var=postvar+1/float(n);delta=qm-postmean
            out[i]=-.5*np.sum(np.log(var)+delta*delta/var,axis=1)
        if not np.isfinite(out).all():raise ValueError('Nonfinite PLDA predictive score')
        return out

def summarize(z,sets):
    return (np.stack([np.asarray(z[sorted(s)],np.float64).mean(0) for s in sets]),
            np.asarray([len(s) for s in sets],np.int64))

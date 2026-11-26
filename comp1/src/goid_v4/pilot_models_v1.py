"""Packed canonical set models and deterministic, fit-only episodic batches."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .p1 import SetEncoder
from .artifacts import ContractError


def counts_features(colors,mask,dtype):
    n=mask.sum(1);black=((colors==0)&mask).sum(1);white=n-black
    if not (n>0).all():raise ContractError('Empty real set')
    counts=torch.stack((torch.log1p(n),torch.log1p(black),torch.log1p(white),
                        (black==0).float(),(white==0).float()),1).to(dtype)
    return n,counts


class PackedISAB(SetEncoder):
    """Identical U2 equations; canonical CPU indices are validated before CUDA gather."""
    def forward_packed(self,x,colors,mask):
        n,counts=counts_features(colors,mask,x.dtype)
        h=self.project(x)
        for block in self.blocks:h=block(h,mask)
        h=self.pool(self.pool_seed.expand(len(x),-1,-1),h,mask)[:,0]
        h=torch.cat((h,counts),1)
        mean=F.normalize(self.mean_head(h),dim=-1,eps=1e-12)
        var=F.softplus(self.var_head(h))/n[:,None]+self.floor
        var=torch.where((n==1)[:,None],self.prior_variance[None],var)
        return mean,var


class MeanCount(nn.Module):
    def __init__(self,*,floor,prior,input_dim=1728,width=128,embedding_dim=128,dropout=.1):
        super().__init__();self.floor=floor
        self.register_buffer('prior_variance',torch.full((embedding_dim,),float(prior)))
        self.project=nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout))
        self.mean_head=nn.Linear(width+5,embedding_dim);self.var_head=nn.Linear(width+5,embedding_dim)
    def forward_packed(self,x,colors,mask):
        n,counts=counts_features(colors,mask,x.dtype)
        h=self.project(x).masked_fill(~mask[:,:,None],0).sum(1)/n[:,None]
        h=torch.cat((h,counts),1)
        mean=F.normalize(self.mean_head(h),dim=-1,eps=1e-12)
        var=F.softplus(self.var_head(h))/n[:,None]+self.floor
        return mean,torch.where((n==1)[:,None],self.prior_variance[None],var)


class TensorSampler:
    def __init__(self,records,seed,*,per_rank=16,support=100):
        self.records=records;self.rng=np.random.default_rng(seed);self.per_rank=per_rank;self.support=support
        self.by_player=defaultdict(list);self.rank={};self.by_rank=defaultdict(list)
        for i,r in enumerate(records):
            if r['part']!='train':raise ContractError('Training sampler rejects calibration/dev rows')
            self.by_player[r['player']].append(i);self.rank[r['player']]=r['rank']
        for p,ids in self.by_player.items():
            if len(ids)<support+20 or len({records[i]['effective_group_token'] for i in ids})!=len(ids):raise ContractError('Insufficient unique groups')
            self.by_rank[self.rank[p]].append(p)
        self.ranks=sorted(self.by_rank)
        if len(self.ranks)<2 or any(len(ps)<per_rank for ps in self.by_rank.values()):raise ContractError('Rank batch capacity')
        for ps in self.by_rank.values():ps.sort()
    def sample(self):
        selected=self.rng.choice(self.ranks,2,replace=False)
        players=[p for rank in selected for p in self.rng.choice(self.by_rank[rank],self.per_rank,replace=False)]
        b=len(players);indices=np.zeros((2*b,100),np.int64);mask=np.zeros((2*b,100),bool)
        for j,p in enumerate(players):
            q=int(self.rng.choice([5,10,20],p=[.4,.4,.2]))
            chosen=self.rng.choice(self.by_player[p],self.support+q,replace=False)
            si=sorted(chosen[:self.support]);qi=sorted(chosen[self.support:])
            indices[j,:len(si)]=si;mask[j,:len(si)]=True
            indices[j+b,:q]=qi;mask[j+b,:q]=True
        negatives=np.ones((b,b),bool)
        for i,p in enumerate(players):
            cross=[j for j,other in enumerate(players) if self.rank[other]!=self.rank[p]]
            negatives[i,int(self.rng.choice(cross))]=False
        return indices,mask,negatives


def fixed_encode(model,features,colors,sets,*,batch_size=32):
    """Fixed32x100 forward including valid dummy sets, discarded after every batch."""
    model.eval();means=[];variances=[]
    for start in range(0,len(sets),batch_size):
        real=sets[start:start+batch_size]
        indices=np.zeros((batch_size,100),np.int64);mask=np.zeros_like(indices,dtype=bool)
        mask[:,0]=True  # Dummy set is one zero game, never enters a saved output.
        for i,ids in enumerate(real):
            if not 1<=len(ids)<=100 or len(set(ids))!=len(ids):raise ContractError('Invalid inference set')
            ids=sorted(ids);indices[i,:len(ids)]=ids;mask[i]=False;mask[i,:len(ids)]=True
        ti=torch.from_numpy(indices).to(features.device);tm=torch.from_numpy(mask).to(features.device)
        x=features[ti].masked_fill(~tm[:,:,None],0);c=colors[ti]
        if len(real)<batch_size:x[len(real):]=0;c[len(real):]=0
        with torch.inference_mode():m,v=model.forward_packed(x,c,tm)
        means.append(m[:len(real)].float().cpu().numpy());variances.append(v[:len(real)].float().cpu().numpy())
    return np.concatenate(means),np.concatenate(variances)


def gaussian_scores(qmean,qvar,smean,svar,*,variance_scale=1.):
    result=np.empty((len(qmean),len(smean)),np.float64)
    smean=np.asarray(smean,np.float64);svar=np.asarray(svar,np.float64)
    for i in range(len(qmean)):
        var=(np.asarray(qvar[i],np.float64)+svar)*variance_scale
        if not np.isfinite(var).all() or (var<=0).any():raise ContractError('Invalid predictive variance')
        diff=np.asarray(qmean[i],np.float64)-smean
        result[i]=-.5*np.sum(diff*diff/var+np.log(var),axis=1)
    return result

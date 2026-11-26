import numpy as np,torch
from torch import nn

class Strength(nn.Module):
    def __init__(self):
        super().__init__();self.decision=nn.Sequential(nn.Linear(772,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,1));self.base=nn.Parameter(torch.tensor(-4.));self.gaps=nn.Parameter(torch.full((8,),float(np.log(np.expm1(.95)))));self.player_noise=nn.Parameter(torch.tensor(float(np.log(np.expm1(.5)))));self.game_noise=nn.Parameter(torch.tensor(float(np.log(np.expm1(1.)))))
    def log_probabilities(self,beta,count):
        cuts=torch.cat([self.base[None],self.base+torch.cumsum(torch.nn.functional.softplus(self.gaps)+.05,0)]);noise=(torch.nn.functional.softplus(self.player_noise).square()+torch.nn.functional.softplus(self.game_noise).square()/count).sqrt();u=(cuts[None,:]-beta[:,None])/noise[:,None];gap=(cuts[1:]-cuts[:-1])[None,:]/noise[:,None];middle=torch.nn.functional.logsigmoid(u[:,1:])+torch.nn.functional.logsigmoid(-u[:,:-1])+torch.log(-torch.expm1(-gap));lp=torch.cat([torch.nn.functional.logsigmoid(u[:,:1]),middle,torch.nn.functional.logsigmoid(-u[:,-1:])],1);return lp-torch.logsumexp(lp,1,keepdim=True)
    def probabilities(self,beta,count):return self.log_probabilities(beta,count).exp()
    def beta(self,x,mask):
        b=self.decision(x).squeeze(-1).float().mean(-1);count=mask.sum(1);return (b*mask).sum(1)/count,count


def sampler(groups,ranks,seed,step,pools):
    rng=np.random.default_rng(seed+7919*step);ii=np.zeros((200,20),np.int64);mask=np.zeros((200,20),np.float32);labels=np.repeat(np.arange(10),20);sizes=np.tile(np.repeat([20,15,10,5],5),10)
    for rank in range(10):
        used=set()
        for offset,g in enumerate(np.repeat([20,15,10,5],5)):
            eligible=pools[rank,int(g)];assert len(eligible)>=20
            for attempt in range(1000):
                actor=int(eligible[int(rng.integers(len(eligible)))])
                if actor not in used:break
            else:raise AssertionError('Failed unique actor draw')
            used.add(actor);j=rank*20+offset;ii[j,:g]=rng.choice(groups[actor],int(g),replace=False);mask[j,:g]=1
        assert len(used)==20
    assert all(np.sum((labels==r)&(sizes==g))==5 for r in range(10) for g in (5,10,15,20))
    order=rng.permutation(200);return ii[order],mask[order],labels[order]


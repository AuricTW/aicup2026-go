"""Joint scalar evidence; original decision network plus a tiny game branch."""
import torch
from torch import nn
from base_method import Strength as BaseStrength,sampler
class Strength(BaseStrength):
    def __init__(self):
        super().__init__()
        self.profile=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1))
        nn.init.zeros_(self.profile[-1].weight);nn.init.zeros_(self.profile[-1].bias)
    def components(self,x,z):
        base=self.decision(x).squeeze(-1).float().mean(-1)
        correction=self.profile(z).squeeze(-1).float()
        return base+correction,base,correction
    def beta(self,x,z,mask):
        b,_,_=self.components(x,z);count=mask.sum(1)
        return (b*mask).sum(1)/count,count

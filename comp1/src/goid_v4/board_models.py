"""P2/P3 from-scratch model definitions. No checkpoint loader or execution driver."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .artifacts import ContractError


class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),
                               nn.GroupNorm(8,channels),nn.GELU(),
                               nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.GroupNorm(8,channels))
    def forward(self,x):return F.gelu(x+self.net(x))


class BoardEncoder(nn.Module):
    def __init__(self,input_planes,*,channels=128,blocks=8):
        super().__init__()
        self.input_planes=input_planes
        self.net=nn.Sequential(nn.Conv2d(input_planes,channels,3,padding=1,bias=False),
                               nn.GroupNorm(8,channels),nn.GELU(),*[ResidualBlock(channels) for _ in range(blocks)])
    def forward(self,x):
        if x.ndim!=4 or x.shape[1:]!=(self.input_planes,19,19) or not torch.isfinite(x).all():
            raise ContractError("Board tensor contract mismatch")
        return self.net(x)


class DecisionGameEncoder(nn.Module):
    """One batch item is one complete game; no cross-game attention.

    Complete-game microbatches can use CPU-reference gradient cache. This
    definition does not claim a CUDA memory footprint or cross-GPU bitwise parity.
    """
    def __init__(self,*,channels=128,blocks=8,width=256,layers=2,heads=8,ffn=2048,embedding_dim=256,dropout=.1):
        super().__init__()
        self.board=BoardEncoder(18,channels=channels,blocks=blocks)
        self.token=nn.Linear(channels*2+14,width)
        layer=nn.TransformerEncoderLayer(width,heads,ffn,dropout,activation='gelu',batch_first=True,norm_first=False)
        self.sequence=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.output=nn.Linear(width,embedding_dim)

    def forward(self,planes,scalars,mask):
        if planes.ndim!=5 or planes.shape[2:]!=(18,19,19) or scalars.shape!=planes.shape[:2]+(14,) or mask.shape!=planes.shape[:2] or mask.dtype!=torch.bool:
            raise ContractError("Game tensor/mask shape mismatch")
        if not mask.any(dim=1).all() or not torch.isfinite(scalars[mask]).all():
            raise ContractError("Empty game or nonfinite active scalar")
        # The caller supplies chronological selected positions. Padded tensors never enter CNN.
        active=planes[mask]
        h=self.board(active)
        action=active[:,-1:,:,:]
        pooled=h.mean(dim=(-2,-1))
        action_feature=(h*action).sum(dim=(-2,-1))
        token=self.token(torch.cat((pooled,action_feature,scalars[mask]),dim=-1))
        full=torch.zeros((*mask.shape,token.shape[-1]),dtype=token.dtype,device=token.device)
        full[mask]=token
        # Fixed sinusoidal position index supplies order; no future board/action fields are added.
        pos=torch.arange(mask.shape[1],device=token.device,dtype=token.dtype)[:,None]
        freq=torch.exp(torch.arange(0,token.shape[-1],2,device=token.device,dtype=token.dtype)*(-math.log(10000)/token.shape[-1]))
        pe=torch.zeros((mask.shape[1],token.shape[-1]),device=token.device,dtype=token.dtype)
        pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        seq=self.sequence(full+pe[None],src_key_padding_mask=~mask)
        mean=seq.masked_fill(~mask[:,:,None],0).sum(dim=1)/mask.sum(dim=1)[:,None]
        return F.normalize(self.output(mean),dim=-1,eps=1e-12)


class GE2ESoftmax(nn.Module):
    """GE2E softmax, normalized embeddings and positive leave-one-game-out centroid.

    https://google.github.io/speaker-id/publications/GE2E/ (Eq3/6 errata).
    Shared bias cancels in softmax; retaining it follows the proposed parameterization.
    """
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(math.log(math.expm1(10.)),dtype=torch.float32))
        self.bias=nn.Parameter(torch.tensor(-5.))

    def similarities(self,embeddings):
        if embeddings.ndim!=3 or min(embeddings.shape[:2])<2 or not torch.isfinite(embeddings).all():
            raise ContractError("GE2E needs >=2 players x >=2 distinct games")
        if (torch.linalg.vector_norm(embeddings,dim=-1)<=1e-12).any():
            raise ContractError("Degenerate game embedding")
        e=F.normalize(embeddings,dim=-1)
        centroid=e.mean(dim=1)
        loo=(e.sum(dim=1,keepdim=True)-e)/(e.shape[1]-1)
        if (torch.linalg.vector_norm(centroid,dim=-1)<=1e-12).any() or (torch.linalg.vector_norm(loo,dim=-1)<=1e-12).any():
            raise ContractError("Degenerate GE2E centroid")
        cosine=torch.einsum('nmd,kd->nmk',e,F.normalize(centroid,dim=-1))
        positive=(e*F.normalize(loo,dim=-1)).sum(dim=-1)
        own=torch.eye(len(e),dtype=torch.bool,device=e.device)[:,None,:]
        cosine=torch.where(own,positive[:,:,None],cosine)
        return F.softplus(self.raw_scale)*cosine+self.bias

    def forward(self,embeddings):
        scores=self.similarities(embeddings)
        targets=torch.arange(len(embeddings),device=scores.device)[:,None].expand(scores.shape[:2])
        return F.cross_entropy(scores.reshape(-1,len(embeddings)),targets.reshape(-1))


class PopulationPolicy(nn.Module):
    """17 pre-state planes +8 history-valid bits; never an actual-action input."""
    def __init__(self,*,channels=128,blocks=8):
        super().__init__()
        self.board=BoardEncoder(17,channels=channels,blocks=blocks)
        self.history=nn.Linear(8,channels)
        self.spatial=nn.Conv2d(channels,1,1)
        self.pass_head=nn.Linear(channels,1)
    def forward(self,state_planes,history_mask):
        if history_mask.shape!=(len(state_planes),8) or history_mask.dtype!=torch.bool or not history_mask[:,0].all() or (history_mask[:,1:]&~history_mask[:,:-1]).any():
            raise ContractError('Policy needs explicit prefix-valid prestate history masks')
        h=self.board(state_planes)
        h=F.gelu(h+self.history(history_mask.to(h.dtype))[:,:,None,None])
        return torch.cat((self.spatial(h).flatten(1),self.pass_head(h.mean(dim=(-2,-1)))),dim=1)

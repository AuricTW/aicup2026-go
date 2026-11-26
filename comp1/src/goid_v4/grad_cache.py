"""CPU reference for complete-game gradient caching, including dropout RNG replay.

No optimizer step, device probing, real-data driver or CUDA memory claim.
CUDA RNG/DDP/global-negative synchronization needs its own authorized profile.
"""
from __future__ import annotations
import torch
from torch import nn
from .artifacts import ContractError


def backward_cached(encoder,batches,loss_fn):
    if not batches or any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in encoder.modules()):
        raise ContractError("Cache needs independent complete examples and no BatchNorm state")
    if any(p.device.type!='cpu' for p in encoder.parameters()):
        raise ContractError("Only CPU gradient-cache reference is validated")
    for batch in batches:
        if not isinstance(batch,tuple) or any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in batch):
            raise ContractError("Explicit CPU tensor tuples required")
    rng,outputs=[],[]
    versions=[p._version for p in encoder.parameters()]
    with torch.no_grad():
        for batch in batches:
            rng.append(torch.get_rng_state().clone());outputs.append(encoder(*batch))
    if any(o.ndim!=2 or not torch.isfinite(o).all() for o in outputs):
        raise ContractError("Encoder cache expects finite game embeddings")
    cache=torch.cat(outputs).detach().requires_grad_(True)
    loss=loss_fn(cache)
    if loss.ndim!=0 or not torch.isfinite(loss):raise ContractError("Invalid cached loss")
    loss.backward()
    if cache.grad is None or not torch.isfinite(cache.grad).all():raise ContractError("Missing cache gradient")
    offset=0
    for batch,state,output in zip(batches,rng,outputs):
        if [p._version for p in encoder.parameters()]!=versions:
            raise ContractError("Encoder parameters changed between cache passes")
        # fork_rng restores the post-cache CPU RNG progression after each replay.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state)
            encoded=encoder(*batch)
            if not torch.equal(encoded.detach(),output):
                raise ContractError("Cache replay mismatch (stateful/stochastic encoder)")
            torch.autograd.backward(encoded,cache.grad[offset:offset+len(output)])
        offset+=len(output)
    return float(loss.detach())


class PresentationBudget:
    def __init__(self,*,max_updates,max_presentations):
        if type(max_updates) is not int or type(max_presentations) is not int or min(max_updates,max_presentations)<=0:
            raise ContractError("Explicit positive budget required")
        self.max_updates,self.max_presentations=max_updates,max_presentations
        self.updates=self.presentations=0
    def consume(self,valid_positions):
        if type(valid_positions) is not int or valid_positions<=0:
            raise ContractError("Padding cannot be charged as valid positions")
        if self.updates+1>self.max_updates or self.presentations+valid_positions>self.max_presentations:
            raise ContractError("Presentation/update budget exhausted before next update")
        self.updates+=1;self.presentations+=valid_positions

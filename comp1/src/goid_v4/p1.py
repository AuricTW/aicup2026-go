"""P1 engineering: coherent joint means/covariances and a two-ISAB set model.

No fitted prior, real weights, calibration or training loop is bundled.
Callers must supply the prior/floor/kappas explicitly; these are not tuned here.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .artifacts import ContractError
from .scoring import GameSet


def compose_game_features(stat, mlp, continuous, colors, length, stage_counts):
    stat, mlp, continuous = [np.asarray(x, dtype=np.float32) for x in (stat, mlp, continuous)]
    n = len(colors)
    if stat.shape != (n, 1582) or mlp.shape != (n, 128) or continuous.shape != (n, 12):
        raise ContractError("P1 requires the declared stat/MLP/continuous dimensions")
    length = np.asarray(length, dtype=np.float32)
    counts = np.asarray(stage_counts, dtype=np.float32)
    if length.shape != (n,) or counts.shape != (n, 3) or n == 0:
        raise ContractError("P1 length/stage shape mismatch")
    if any(c not in ("B", "W") for c in colors) or (length <= 0).any() or (counts < 0).any() or (counts.sum(axis=1) <= 0).any():
        raise ContractError("Invalid P1 metadata")
    onehot = np.asarray([[c == "B", c == "W"] for c in colors], dtype=np.float32)
    stage_fraction = counts / counts.sum(axis=1, keepdims=True)
    result = np.concatenate((stat, mlp, continuous, onehot,
                             np.minimum(length[:, None] / 720, 1), stage_fraction), axis=1)
    if not np.isfinite(result).all():
        raise ContractError("P1 features nonfinite")
    return result  # 1728D; existing stat/MLP preprocessing is an external pinned dependency.


@dataclass(frozen=True)
class JointSummary:
    means: np.ndarray       # global/B/W by dimension
    covariance: np.ndarray  # 3 x 3 x dimension, including shared-game covariance
    counts: tuple[int, int]


def joint_summary(games: GameSet, *, prior_variance, mean_kappa, variance_kappa, floor):
    g = games.canonical()
    x = g.vectors
    prior = np.asarray(prior_variance, dtype=np.float64)
    if prior.shape != (x.shape[1],) or not np.isfinite(prior).all() or (prior <= 0).any():
        raise ContractError("Explicit positive fit-only per-game prior required")
    if not all(math.isfinite(v) for v in (mean_kappa, variance_kappa, floor)) or mean_kappa < 0 or variance_kappa <= 0 or floor <= 0:
        raise ContractError("Invalid uncertainty hyperparameters")
    n = len(x)
    a = np.full((3, n), 1 / n, dtype=np.float64)
    per_game_var = np.zeros_like(x)
    counts = []
    for j, c in enumerate(("B", "W"), start=1):
        mask = np.asarray(g.colors) == c
        m = int(mask.sum()); counts.append(m)
        if m:
            lam = m / (m + mean_kappa)
            a[j] = (1 - lam) / n
            a[j, mask] += lam / m
            # n=1 uses the fit prior, never a zero sample variance.
            var = prior if m == 1 else ((m - 1) * x[mask].var(axis=0, ddof=1) + variance_kappa * prior) / (m - 1 + variance_kappa)
            per_game_var[mask] = np.maximum(var, floor)
        # Missing color keeps exactly the global linear estimator and covariance.
    means = a @ x
    cov = np.einsum("ai,bi,id->abd", a, a, per_game_var, optimize=False)
    return JointSummary(means, cov, tuple(counts))


def joint_uncertainty_score(query: JointSummary, support: JointSummary):
    for summary in (query,support):
        if summary.means.ndim!=2 or summary.means.shape[0]!=3 or summary.covariance.shape!=(3,3,summary.means.shape[1]) or not np.isfinite(summary.means).all() or not np.isfinite(summary.covariance).all() or len(summary.counts)!=2 or any(type(n) is not int or n<0 for n in summary.counts) or sum(summary.counts)==0:
            raise ContractError('Invalid joint mean/covariance summary')
    n = sum(query.counts)
    if n == 0 or query.means.shape != support.means.shape:
        raise ContractError("Joint summary mismatch")
    w = np.asarray((0, query.counts[0] / n, query.counts[1] / n), dtype=np.float64)
    delta = w @ query.means - w @ support.means
    variance = np.einsum("a,b,abd->d", w, w, query.covariance + support.covariance, optimize=False)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ContractError("Nonpositive total uncertainty")
    return float(-.5 * np.sum(delta * delta / variance + np.log(variance)))


def mean_count_features(games: GameSet):
    """U0 input contract; no secretly fitted head or choice of count feature weight."""
    g=games.canonical()
    counts=np.asarray([len(g.keys),g.colors.count('B'),g.colors.count('W')],dtype=np.float64)
    return np.concatenate((g.vectors.mean(axis=0),np.log1p(counts)))


def canonical_batch(x, colors, mask, keys, *, max_games):
    if x.ndim != 3 or colors.shape != x.shape[:2] or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
        raise ContractError("Set tensor/mask shape mismatch")
    if len(keys) != len(x) or x.shape[1] > max_games:
        raise ContractError("Set/key/padding capacity mismatch")
    xs, cs, ms = [], [], []
    for b in range(len(x)):
        if len(keys[b]) != x.shape[1]:
            raise ContractError("Set keys shape mismatch")
        ids = torch.nonzero(mask[b], as_tuple=False).flatten().tolist()
        if not ids or len({keys[b][i] for i in ids}) != len(ids):
            raise ContractError("Empty or repeated game set")
        if any(not isinstance(keys[b][i], str) or not keys[b][i] for i in ids):
            raise ContractError("Invalid canonical game key")
        ids.sort(key=lambda i: keys[b][i])
        indices = torch.tensor(ids, device=x.device)
        valid = x[b].index_select(0, indices)
        cc = colors[b].index_select(0, indices)
        if not torch.isfinite(valid).all() or not ((cc == 0) | (cc == 1)).all():
            raise ContractError("Active set values/color nonfinite or invalid")
        pad = max_games - len(ids)
        xs.append(F.pad(valid, (0, 0, 0, pad)))
        cs.append(F.pad(cc, (0, pad)))
        ms.append(torch.arange(max_games, device=x.device) < len(ids))
    return torch.stack(xs), torch.stack(cs), torch.stack(ms)


class MAB(nn.Module):
    def __init__(self, width, heads, ffn, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, width))

    def forward(self, q, kv, mask=None):
        z = self.norm1(q + self.attn(q, kv, kv, key_padding_mask=None if mask is None else ~mask, need_weights=False)[0])
        return self.norm2(z + self.ff(z))


class ISAB(nn.Module):
    def __init__(self, width, heads, ffn, inducing, dropout):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, inducing, width))
        nn.init.xavier_uniform_(self.inducing)
        self.to_inducing = MAB(width, heads, ffn, dropout)
        self.to_set = MAB(width, heads, ffn, dropout)

    def forward(self, x, mask):
        h = self.to_inducing(self.inducing.expand(len(x), -1, -1), x, mask)
        return self.to_set(x, h)


class SetEncoder(nn.Module):
    """U2 engineering choice: 2 ISAB + one seed PMA + mean/variance heads.

    Variance is in normalized identity-embedding coordinates. n=1 uses the
    explicit fit prior; n>1 uses softplus(head)/n + positive set-level floor.
    Exact hyperparameters remain engineering choices, not calibrated pilot settings.
    """
    def __init__(self, input_dim, *, width=128, heads=4, ffn=256, inducing=16,
                 embedding_dim=128, max_games=100, dropout=.1,
                 variance_floor, prior_variance):
        super().__init__()
        prior = torch.as_tensor(prior_variance, dtype=torch.float32).clone()
        if prior.shape != (embedding_dim,) or not torch.isfinite(prior).all() or not math.isfinite(variance_floor) or variance_floor <= 0 or not (prior >= variance_floor).all():
            raise ContractError("Explicit calibrated units/prior/floor required")
        self.max_games = max_games
        self.input_dim = input_dim
        self.register_buffer("prior_variance", prior)
        self.floor = variance_floor
        self.project = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ISAB(width, heads, ffn, inducing, dropout) for _ in range(2)])
        self.pool_seed = nn.Parameter(torch.empty(1, 1, width)); nn.init.xavier_uniform_(self.pool_seed)
        self.pool = MAB(width, heads, ffn, dropout)
        self.mean_head = nn.Linear(width + 5, embedding_dim)
        self.var_head = nn.Linear(width + 5, embedding_dim)

    def forward(self, x, colors, mask, keys):
        if x.shape[-1] != self.input_dim:
            raise ContractError("Set feature dimension mismatch")
        x, colors, mask = canonical_batch(x, colors, mask, keys, max_games=self.max_games)
        n = mask.sum(dim=1)
        black = ((colors == 0) & mask).sum(dim=1)
        white = n - black
        counts = torch.stack((torch.log1p(n), torch.log1p(black), torch.log1p(white),
                              (black == 0).float(), (white == 0).float()), dim=1).to(x.dtype)
        h = self.project(x)
        for block in self.blocks: h = block(h, mask)
        pooled = self.pool(self.pool_seed.expand(len(x), -1, -1), h, mask)[:, 0]
        pooled = torch.cat((pooled, counts), dim=1)
        mean = F.normalize(self.mean_head(pooled), dim=-1, eps=1e-12)
        learned = F.softplus(self.var_head(pooled)) / n[:, None] + self.floor
        var = torch.where((n == 1)[:, None], self.prior_variance[None, :], learned)
        return mean, var


def uncertainty_logits(query_mean, query_var, support_mean, support_var):
    if query_mean.shape != query_var.shape or support_mean.shape != support_var.shape or query_mean.ndim != 2 or support_mean.ndim != 2 or query_mean.shape[1] != support_mean.shape[1]:
        raise ContractError("Uncertainty score shape mismatch")
    if not all(torch.isfinite(x).all() for x in (query_mean, query_var, support_mean, support_var)) or not (query_var > 0).all() or not (support_var > 0).all():
        raise ContractError("Finite means and positive variances required")
    total_var = query_var[:, None, :] + support_var[None, :, :]
    diff = query_mean[:, None, :] - support_mean[None, :, :]
    return -.5 * (diff.square() / total_var + total_var.log()).sum(dim=-1)


def identity_loss(logits, *, negative_mask=None):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or not torch.isfinite(logits).all():
        raise ContractError("Identity episode needs square finite logits")
    if negative_mask is not None:
        if negative_mask.shape != logits.shape or negative_mask.dtype != torch.bool or not negative_mask.diagonal().all():
            raise ContractError("Invalid balanced-negative mask")
        logits = logits.masked_fill(~negative_mask, -torch.inf)
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


class FrozenSetScorer:
    """One set at a time, fixed padding, frozen tensor-only inference.

    A set is (features[N,D],colors[N],keys[N]); no IDs/GT/rank in feature tensors.
    The caller must already have a legal scope. This class cannot authorize it.
    """
    def __init__(self,encoder,candidate_order,supports):
        if any(m.training for m in encoder.modules()) or any(p.requires_grad for p in encoder.parameters()):
            raise ContractError('Inference encoder must explicitly be frozen and eval')
        self.encoder=encoder;self.order=tuple(candidate_order)
        if len(set(self.order))!=len(self.order) or not self.order or set(self.order)!=set(supports):
            raise ContractError('P1 candidate scope mismatch')
        self.versions=self._state()
        self.support_keys=set();means=[];vars_=[]
        for p in self.order:
            x,c,keys=supports[p]
            if self.support_keys&set(keys):raise ContractError('Repeated support game')
            self.support_keys.update(keys)
            with torch.no_grad():m,v=self._encode(x,c,keys)
            means.append(m[0]);vars_.append(v[0])
        self.mean=torch.stack(means);self.variance=torch.stack(vars_)

    def _encode(self,x,c,keys):
        mask=torch.ones((1,len(x)),device=x.device,dtype=torch.bool)
        return self.encoder(x[None],c[None],mask,[keys])

    def _state(self):
        return (tuple((id(p),p._version) for p in self.encoder.parameters()),
                tuple((id(b),b._version) for b in self.encoder.buffers()),
                self.encoder.floor,self.encoder.max_games,self.encoder.input_dim)

    def score_one(self,features,colors,keys):
        if any(m.training for m in self.encoder.modules()) or any(p.requires_grad for p in self.encoder.parameters()) or self.versions!=self._state():
            raise ContractError('Frozen encoder changed')
        if self.support_keys&set(keys):raise ContractError('Query shares support games')
        with torch.no_grad():
            m,v=self._encode(features,colors,keys)
            scores=uncertainty_logits(m,v,self.mean,self.variance)[0]
        if not torch.isfinite(scores).all():raise ContractError('P1 scores nonfinite')
        return scores

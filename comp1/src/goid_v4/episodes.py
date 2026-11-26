"""Fit-only sampling semantics, exact per-query same/cross-rank negative balance."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from .artifacts import ContractError
from .contracts import validate_scope, validate_episode


@dataclass(frozen=True)
class SampledEpisode:
    player_order: tuple
    supports: dict
    queries: dict
    negative_mask: np.ndarray


class BalancedSampler:
    """Two ranks x16 distinct players. Each query has 15 same +15 cross negatives.

    One of the 16 cross players is excluded per query by the seeded mask.
    All32 players still contribute100 supports and a q5/10/20 query set.
    This explicit engineering resolution is not an evaluated/frozen pilot.
    """
    def __init__(self, games, *, seed, players_per_rank=16, support_count=100):
        self.index = validate_scope(games)
        self.rng = np.random.default_rng(seed)
        self.k, self.support_count = players_per_rank, support_count
        if players_per_rank < 2 or support_count < 1:
            raise ContractError("Invalid episode cardinality")
        self.by_player = defaultdict(list)
        self.ranks = {}
        for g in games:
            if g.role != "fit" or g.rank_train_only is None:
                raise ContractError("Sampler accepts fit-only ranked metadata")
            self.by_player[g.opaque_player_key].append(g)
            if g.opaque_player_key in self.ranks and self.ranks[g.opaque_player_key] != g.rank_train_only:
                raise ContractError("Player rank ambiguous")
            self.ranks[g.opaque_player_key] = g.rank_train_only
        self.by_rank = defaultdict(list)
        for player, rows in self.by_player.items():
            # Deduplicate effective groups deterministically, never repeat rows to fill support.
            groups = {}
            for g in sorted(rows, key=lambda x: x.source_row_key): groups.setdefault(g.effective_group_token, g)
            self.by_player[player] = list(groups.values())
            if len(groups) >= support_count + 20:
                self.by_rank[self.ranks[player]].append(player)
        self.valid_ranks = sorted(r for r, ps in self.by_rank.items() if len(ps) >= self.k)
        if len(self.valid_ranks) < 2:
            raise ContractError("Need two ranks with enough distinct eligible fit players")
        for ps in self.by_rank.values(): ps.sort()

    def sample(self):
        ranks = self.rng.choice(self.valid_ranks, 2, replace=False)
        players = tuple(p for rank in ranks for p in self.rng.choice(self.by_rank[rank], self.k, replace=False).tolist())
        supports, queries = {}, {}
        for player in players:
            n = int(self.rng.choice([5, 10, 20], p=[.4, .4, .2]))
            rows = self.by_player[player]
            ids = self.rng.choice(len(rows), self.support_count + n, replace=False)
            supports[player] = tuple(rows[i].source_row_key for i in ids[:self.support_count])
            queries[player] = tuple(rows[i].source_row_key for i in ids[self.support_count:])
        # Validate each q cardinality together to allow the declared mixed q batch.
        for q in (5, 10, 20):
            active = tuple(p for p in players if len(queries[p]) == q)
            if active:
                validate_episode(self.index, active, {p:supports[p] for p in active},
                                 {p:queries[p] for p in active}, support_count=self.support_count,
                                 query_count=q, candidate_count=len(active))
        # Cross-q support/query collisions also must be rejected.
        from .contracts import HASH_FIELDS
        st = {(k,getattr(self.index[key],k)) for values in supports.values() for key in values for k in HASH_FIELDS}
        qt = {(k,getattr(self.index[key],k)) for values in queries.values() for key in values for k in HASH_FIELDS}
        if st & qt: raise ContractError("Cross-q game overlap")
        mask = np.ones((len(players),len(players)),dtype=bool)
        for i, p in enumerate(players):
            cross = [j for j, other in enumerate(players) if self.ranks[other] != self.ranks[p]]
            mask[i, int(self.rng.choice(cross))] = False
        return SampledEpisode(players, supports, queries, mask)

"""Read only pre-pinned, pre-existing lawful development metadata. No data rebuild."""
from __future__ import annotations
from collections import Counter
from .artifacts import ContractError, object_hash
from .contracts import EXPOSURES, identifier


def inventory(reader):
    allowed = reader.read("eligible_players")
    metadata = reader.read("player_metadata")
    capacity = reader.read("legacy_capacity")
    if not isinstance(allowed, list) or len(set(allowed)) != len(allowed):
        raise ContractError("Malformed pre-existing player allowlist")
    by_id = {}
    fields = {"player_id", "unique_games", "rank_counts", "modal_rank", "modal_rank_purity",
              "single_rank", "eligible", "eligible_within_rank"}
    for row in metadata:
        if set(row) != fields or row["player_id"] in by_id or type(row["unique_games"]) is not int or row["unique_games"] < 1:
            raise ContractError("Malformed player metadata")
        identifier(row["player_id"])
        by_id[row["player_id"]] = row
    eligible = {p for p, row in by_id.items() if row["unique_games"] >= 120}
    if set(allowed) != eligible or capacity["players_total"] != len(by_id) or capacity["eligible_players"] != len(eligible):
        raise ContractError("Pre-existing aggregate/allowlist disagreement")
    seen, occurrence = set(), Counter()
    for seed in range(20261027, 20261032):
        ep = reader.read(f"episode_{seed}")
        ids = ep["candidate_ids"]
        if len(ids) != 400 or len(set(ids)) != 400 or not set(ids) <= eligible:
            raise ContractError("Episode players outside lawful allowlist")
        seen.update(ids); occurrence.update(ids)
    ledger = []
    for player in sorted(by_id):
        exposure = {k: None for k in EXPOSURES}
        # Historical split provenance, not a new row-level or membership audit.
        exposure.update(representation_fit=False, normalizer_fit=False, policy_pretrain=False,
                        score_seen=True if player in seen else None,
                        protocol_selection=True if player in seen else None, unknown=True)
        ledger.append({"opaque_player_key": object_hash({"namespace": "goid-v4-player-v1", "player": player}),
                       "historical_group_count": by_id[player]["unique_games"],
                       "in_eligible_allowlist": player in eligible,
                       "development_pool_occurrences": occurrence[player],
                       "exposure": exposure, "first_exposure_version": None,
                       "last_inventory_version": "engineering_v1", "pristine_claimed": False})
    report = {"status": "metadata_inventory_complete_row_scope_unverified",
              "metadata_players": len(by_id), "eligible_allowlist_players": len(eligible),
              "development_selected_unique_players": len(seen),
              "eligible_not_selected": len(eligible - seen),
              "historical_metadata_threshold_counts": {str(q): sum(r["unique_games"] >= 100 + q for r in metadata) for q in (5, 10, 20)},
              "not_selected_threshold_counts": {str(q): sum(r["unique_games"] >= 100 + q and r["player_id"] not in seen for r in metadata) for q in (5, 10, 20)},
              "proven_pristine_players": 0, "all_players_have_unknown_exposure": True,
              "raw_games_or_feature_values_read": False, "retired_detail_read": False,
              "g0_lawful_row_scope_passed": False,
              "missing": ["independent allowed_games rows with all group/hash tokens and scope attestation",
                          "safe fit-only vector/SGF export and its provenance",
                          "complete exposure history and feasible G2 route"],
              "source_interpretation": "Existing exported metadata and historical exclusion provenance only; not fresh group counts or unseen-player confirmation capacity"}
    return report, ledger

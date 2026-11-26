"""Auditable main-line SGF replay and compact, target-player style features.

Coordinates use sgfmill/GTP convention: row 0 is the bottom, col 0 the left.
SGF ``aa`` is therefore (size - 1, 0), not (0, 0). Only the leftmost
variation is analysed. Later setup nodes are rejected rather than skipped.

Legality policy: occupied points, suicide and immediate simple-ko recapture
are rejected. Superko and turn alternation are NOT enforced: the SGFs do not
necessarily provide a ruleset. Consecutive same-colour moves are counted.

The small Board subclass instruments sgfmill 1.1.1's private
``_find_surrounded_groups`` hook. Captures come from the exact groups already
found by Board.play; no second per-move board scan is needed. The dependency
is pinned and the hook's capture/suicide/ko behaviour has regression tests.
Primary API/rules documentation:
https://mjw.woodcraft.me.uk/sgfmill/doc/1.1.1/boards.html
https://mjw.woodcraft.me.uk/sgfmill/doc/1.1.1/sgf_games.html

The SHA256 hashes identify UTF-8 CSV field content (raw_hash), or normalised
main-line moves plus effective root setup and size (move_hash). canonical_hash
also identifies all eight board symmetries; neither normalised hash swaps
black and white. Metadata, target_color and unplayed variations are excluded.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import struct
from typing import Any

import numpy as np
from sgfmill import boards, sgf, sgf_grammar


FEATURE_SCHEMA = "goid-style-v1"
SEQUENCE_SCHEMA = "goid-moves-u16le-v1"
PASS_POINT = 0x7FFF
WHITE_BIT = 0x8000
LEGALITY_POLICY = "no-occupied/no-suicide/simple-ko; no-superko; no-turn-enforcement"


def _make_feature_groups() -> tuple[dict[str, Any], ...]:
    specs = [("summary", 12, "continuous", "colour, length, pass and capture diagnostics")]
    phase_specs = (
        ("edge", 10, "minimum distance to an edge, clipped at 9"),
        ("corner", 10, "Chebyshev distance to closest corner, clipped at 9"),
        ("previous_distance", 12, "Manhattan distance to previous placement; 11 means unavailable"),
        ("own_distance", 12, "Manhattan distance to own previous placement; 11 means unavailable"),
        ("captures", 6, "captured stones: 0, 1, 2, 3-4, 5-8, 9+"),
        ("adjacency", 15, "orthogonal empty/own/opponent neighbour counts, five bins each"),
        ("local3x3", 256, "hashed relative-colour 3x3 before the action, excluding centre"),
        ("coarse5x5", 25, "absolute target placement on a normalised 5x5 grid"),
    )
    for phase, bounds in enumerate(("plies 1-60", "plies 61-180", "plies 181+")):
        for name, width, description in phase_specs:
            specs.append((f"phase{phase}_{name}", width, "histogram", f"{bounds}: {description}"))
    specs.extend((
        ("opening_folded", 55, "histogram", "first 20 target placements, D4-folded corner distances"),
        ("opening_absolute", 361, "histogram", "first 20 target placements on a normalised 19x19 board"),
        ("opening_transition", 128, "histogram", "hashed successive folded opening target placements"),
    ))
    groups = []
    start = 0
    for name, width, kind, description in specs:
        groups.append(dict(name=name, start=start, stop=start + width, kind=kind, description=description))
        start += width
    return tuple(groups)


FEATURE_GROUPS = _make_feature_groups()
FEATURE_DIM = FEATURE_GROUPS[-1]["stop"]
FEATURE_NAMES = tuple(
    f"{g['name']}:{i}" for g in FEATURE_GROUPS for i in range(g["stop"] - g["start"])
)
SUMMARY_NAMES = (
    "target_black", "target_white", "game_plies_over_720", "target_fraction",
    "pass_rate", "target_pass_rate", "setup_fraction", "first_target_ply_over_720",
    "target_capture_mean_over_20", "opening_target_fraction", "middle_target_fraction",
    "late_target_fraction",
)
_SLICES = {g["name"]: slice(g["start"], g["stop"]) for g in FEATURE_GROUPS}


class SGFAnalysisError(ValueError):
    """A visible, serialisable failure with a machine-readable code and ply."""

    def __init__(self, code: str, message: str, *, ply: int | None = None, node: int | None = None):
        self.code = code
        self.ply = ply
        self.node = node
        where = f" at ply {ply}" if ply is not None else (f" at node {node}" if node is not None else "")
        super().__init__(f"{code}{where}: {message}")


class _CaptureBoard(boards.Board):
    """Record captures from the work sgfmill Board.play already performs."""

    captured = 0
    self_captured = 0
    _playing_colour: str | None = None

    def play(self, row: int, col: int, colour: str):
        self.captured = 0
        self.self_captured = 0
        self._playing_colour = colour
        return super().play(row, col, colour)

    def _find_surrounded_groups(self, row: int, col: int):
        groups = super()._find_surrounded_groups(row, col)
        # This exactly mirrors which groups Board.play removes, including its
        # distinction between capturing the opponent and self-capture.
        removed = groups if len(groups) <= 1 else [g for g in groups if g.colour != self._playing_colour]
        self.captured = sum(len(g.points) for g in removed if g.colour != self._playing_colour)
        self.self_captured = sum(len(g.points) for g in removed if g.colour == self._playing_colour)
        return groups


@lru_cache(maxsize=26)
def _geometry(size: int) -> dict[str, Any]:
    points = size * size
    ids = np.arange(points, dtype=np.uint16)
    row, col = ids // size, ids % size
    n = size - 1
    # Reflection, then rotations; each row is one element of D4.
    transforms = []
    for reflect in (False, True):
        rr, cc = row, n - col if reflect else col
        for _ in range(4):
            transforms.append((rr * size + cc).astype(np.uint16))
            rr, cc = cc, n - rr
    # Include a pass sentinel so the same vectorised lookup handles passes.
    transform_table = np.full((8, PASS_POINT + 1), PASS_POINT, dtype=np.uint16)
    transform_table[:, :points] = np.asarray(transforms)
    neighbours = []
    local = []
    edge = []
    corner = []
    coarse = []
    absolute = []
    folded = []
    for r in range(size):
        for c in range(size):
            neighbours.append(tuple((a, b) for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= a < size and 0 <= b < size))
            local.append(tuple((r + dr, c + dc) if 0 <= r + dr < size and 0 <= c + dc < size else None
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc))
            a, b = sorted((min(r, n - r), min(c, n - c)))
            edge.append(min(a, 9))
            corner.append(min(b, 9))
            coarse.append(min(4, r * 5 // size) * 5 + min(4, c * 5 // size))
            absolute.append(round(r * 18 / n) * 19 + round(c * 18 / n))
            aa, bb = sorted((min(9, round(a * 18 / n)), min(9, round(b * 18 / n))))
            folded.append(bb * (bb + 1) // 2 + aa)
    return dict(transforms=transform_table, neighbours=neighbours, local=local,
                edge=edge, corner=corner, coarse=coarse, absolute=absolute, folded=folded)


def _encode_move(colour: str, point: tuple[int, int] | None, size: int) -> int:
    return (WHITE_BIT if colour == "w" else 0) | (PASS_POINT if point is None else point[0] * size + point[1])


def decode_sequence(sequence: bytes, board_size: int) -> list[tuple[str, tuple[int, int] | None]]:
    """Decode the cache BLOB. The separate board size is mandatory."""
    if not 2 <= board_size <= 26 or len(sequence) % 2:
        raise ValueError("invalid board size or truncated uint16 sequence")
    result = []
    for encoded in np.frombuffer(sequence, dtype="<u2"):
        value = int(encoded)
        point = value & PASS_POINT
        if point != PASS_POINT and point >= board_size * board_size:
            raise ValueError("sequence point outside board")
        result.append(("w" if value & WHITE_BIT else "b", None if point == PASS_POINT else divmod(point, board_size)))
    return result


def _normalised_hashes(size: int, setup: np.ndarray, moves: np.ndarray, geometry: dict[str, Any]) -> tuple[str, str]:
    prefix = b"goid-mainline-v1\0" + struct.pack("<BH", size, len(setup))
    original = prefix + setup.astype("<u2", copy=False).tobytes() + moves.astype("<u2", copy=False).tobytes()
    transforms = geometry["transforms"]
    moved = transforms[:, moves & PASS_POINT] | (moves & WHITE_BIT)[None, :]
    transformed_setup = transforms[:, setup & PASS_POINT] | (setup & WHITE_BIT)[None, :]
    transformed_setup.sort(axis=1)
    # Eight contiguous bytes conversions, not eight Python loops per move.
    canonical = min(prefix + transformed_setup[t].astype("<u2", copy=False).tobytes()
                    + moved[t].astype("<u2", copy=False).tobytes() for t in range(8))
    return sha256(original).hexdigest(), sha256(canonical).hexdigest()


def _distance_bin(point: tuple[int, int], previous: tuple[int, int] | None) -> int:
    if previous is None:
        return 11
    return min(10, abs(point[0] - previous[0]) + abs(point[1] - previous[1]))


def _capture_bin(count: int) -> int:
    return count if count <= 2 else 3 if count <= 4 else 4 if count <= 8 else 5


def feature_schema() -> dict[str, Any]:
    """JSON-serialisable description suitable for an audit cache manifest."""
    return dict(version=FEATURE_SCHEMA, dimension=FEATURE_DIM, dtype="float32",
                groups=[dict(g) for g in FEATURE_GROUPS], summary_names=list(SUMMARY_NAMES),
                histogram_normalisation="each group L1 per game; empty groups zero",
                sequence_schema=SEQUENCE_SCHEMA, coordinate_origin="bottom-left",
                legality_policy=LEGALITY_POLICY)


def analyze_sgf(sgf_content: str, target_color: str, *, include_features: bool = True) -> dict[str, Any]:
    """Parse and replay one game; never silently omit a malformed move.

    Returned ``move_sequence`` and ``setup_sequence`` are uint16 little-endian
    BLOBs, encoded ``(white << 15) | (row * size + col)``; point 32767 is pass.
    Both colours remain in the sequence, while features use only target moves.
    ``mainline_moves`` and ``target_moves`` include passes. ``setup_stones``
    counts effective occupied root points. Missing metadata remains None.
    """
    if not isinstance(sgf_content, str) or not sgf_content.strip():
        raise SGFAnalysisError("empty_sgf", "expected a nonempty SGF string")
    if not isinstance(target_color, str) or target_color.upper() not in ("B", "W"):
        raise SGFAnalysisError("invalid_target_color", repr(target_color))
    target = target_color.lower()
    raw = sgf_content.encode("utf-8")
    try:
        collection = sgf_grammar.parse_sgf_collection(raw)
        if len(collection) != 1:
            raise SGFAnalysisError("multiple_games", f"found {len(collection)} game trees in one field")
        game = sgf.Sgf_game.from_coarse_game_tree(collection[0], override_encoding="utf-8")
        size = game.get_size()
        if not 2 <= size <= 26:
            raise SGFAnalysisError("unsupported_size", f"board size {size}")
        root = game.get_root()
        if root.has_property("GM") and root.get("GM") != 1:
            raise SGFAnalysisError("not_go", "GM is not 1")
        ab, aw, ae = root.get_setup_stones()
        if ab & aw or ab & ae or aw & ae:
            raise SGFAnalysisError("overlapping_setup", "AB/AW/AE must be disjoint")
        board = _CaptureBoard(size)
        if not board.apply_setup(ab, aw, ae):
            raise SGFAnalysisError("illegal_setup", "root setup contains a group without liberties")
        initial_player = root.get("PL").upper() if root.has_property("PL") else None
        rules = root.get("RU") if root.has_property("RU") else None
        nodes = game.get_main_sequence()
    except SGFAnalysisError:
        raise
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise SGFAnalysisError("parse_error", str(exc) or type(exc).__name__) from exc

    geometry = _geometry(size)
    setup_codes = np.asarray(sorted([_encode_move("b", p, size) for p in ab]
                                   + [_encode_move("w", p, size) for p in aw]), dtype=np.uint16)
    codes: list[int] = []
    passes = target_passes = target_moves = target_placements = nonalternating = 0
    total_captures = target_captures = 0
    phase_counts = [0, 0, 0]
    first_target_ply = None
    last_placement = None
    own_placement = None
    previous_colour = None
    previous_folded = None
    ko_point = None
    ko_forbidden_colour = None
    features = np.zeros(FEATURE_DIM, dtype=np.float32) if include_features else None
    # Precompute offsets once per game; updates below touch only target moves.
    phase_starts = [{name: _SLICES[f"phase{i}_{name}"].start for name in
                     ("edge", "corner", "previous_distance", "own_distance", "captures", "adjacency", "local3x3", "coarse5x5")}
                    for i in range(3)] if include_features else None

    for node_index, node in enumerate(nodes):
        pending_ply = len(codes) + 1
        try:
            if node_index and node.has_setup_stones():
                raise SGFAnalysisError("unsupported_later_setup", "setup after the root is not a played game", node=node_index)
            if node.has_property("B") and node.has_property("W"):
                raise SGFAnalysisError("ambiguous_move", "node contains both B and W", node=node_index)
            raw_properties = node.get_raw_property_map()
            if any(len(raw_properties[colour]) != 1 for colour in ("B", "W") if colour in raw_properties):
                raise SGFAnalysisError("ambiguous_move", "move property must have exactly one value", node=node_index)
            colour, point = node.get_move()
            if colour is None:
                continue
            if node_index == 0 and (ab or aw or ae):
                raise SGFAnalysisError("mixed_root_setup_move", "root cannot contain both setup and a move", node=node_index)
            ply = len(codes) + 1
            if colour == previous_colour:
                nonalternating += 1
            previous_colour = colour
            codes.append(_encode_move(colour, point, size))
            is_target = colour == target
            phase = 0 if ply <= 60 else 1 if ply <= 180 else 2
            if is_target:
                target_moves += 1
                phase_counts[phase] += 1
                if first_target_ply is None:
                    first_target_ply = ply
            if point is None:
                passes += 1
                target_passes += int(is_target)
                ko_point = ko_forbidden_colour = None
                # A pass breaks direct move-to-move locality; it is not a stone.
                last_placement = None
                if is_target:
                    own_placement = None
                continue
            r, c = point
            if board.get(r, c) is not None:
                raise SGFAnalysisError("occupied_point", f"{colour.upper()} at {point}", ply=ply, node=node_index)
            if point == ko_point and colour == ko_forbidden_colour:
                raise SGFAnalysisError("simple_ko", f"immediate recapture at {point}", ply=ply, node=node_index)
            if include_features and is_target:
                position = r * size + c
                starts = phase_starts[phase]
                features[starts["edge"] + geometry["edge"][position]] += 1
                features[starts["corner"] + geometry["corner"][position]] += 1
                features[starts["previous_distance"] + _distance_bin(point, last_placement)] += 1
                features[starts["own_distance"] + _distance_bin(point, own_placement)] += 1
                features[starts["coarse5x5"] + geometry["coarse"][position]] += 1
                empty = friendly = enemy = 0
                for rr, cc in geometry["neighbours"][position]:
                    stone = board.get(rr, cc)
                    if stone is None:
                        empty += 1
                    elif stone == target:
                        friendly += 1
                    else:
                        enemy += 1
                features[starts["adjacency"] + empty] += 1
                features[starts["adjacency"] + 5 + friendly] += 1
                features[starts["adjacency"] + 10 + enemy] += 1
                pattern = 0
                for neighbour in geometry["local"][position]:
                    if neighbour is None:
                        state = 3
                    else:
                        stone = board.get(*neighbour)
                        state = 0 if stone is None else 1 if stone == target else 2
                    pattern = (pattern << 2) | state
                features[starts["local3x3"] + (((pattern * 0x9E3779B1) & 0xFFFFFFFF) >> 24)] += 1
                if target_placements < 20:
                    folded = geometry["folded"][position]
                    features[_SLICES["opening_folded"].start + folded] += 1
                    features[_SLICES["opening_absolute"].start + geometry["absolute"][position]] += 1
                    if previous_folded is not None:
                        features[_SLICES["opening_transition"].start + ((previous_folded * 55 + folded) * 2654435761 % 128)] += 1
                    previous_folded = folded
            new_ko = board.play(r, c, colour)
            if board.self_captured or board.get(r, c) != colour:
                raise SGFAnalysisError("suicide", f"{colour.upper()} at {point} under default no-suicide policy", ply=ply, node=node_index)
            total_captures += board.captured
            if is_target:
                target_captures += board.captured
                target_placements += 1
                own_placement = point
                if include_features:
                    features[phase_starts[phase]["captures"] + _capture_bin(board.captured)] += 1
            last_placement = point
            ko_point = new_ko
            ko_forbidden_colour = ("w" if colour == "b" else "b") if new_ko is not None else None
        except SGFAnalysisError:
            raise
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            raise SGFAnalysisError("invalid_move", str(exc) or type(exc).__name__, ply=pending_ply, node=node_index) from exc

    moves = np.asarray(codes, dtype=np.uint16)
    move_hash, canonical_hash = _normalised_hashes(size, setup_codes, moves, geometry)
    mainline_moves = len(codes)
    result = dict(board_size=size, board_size_explicit=root.has_property("SZ"),
                  mainline_moves=mainline_moves, target_moves=target_moves,
                  target_placements=target_placements, passes=passes, target_passes=target_passes,
                  setup_stones=len(setup_codes), setup_black=len(ab), setup_white=len(aw),
                  root_empty_points=len(ae), initial_player=initial_player, rules_property=rules,
                  first_move_color=("W" if codes[0] & WHITE_BIT else "B") if codes else None,
                  nonalternating_moves=nonalternating, has_variations=any(len(n) > 1 for n in nodes),
                  total_captures=total_captures, target_captures=target_captures,
                  raw_hash=sha256(raw).hexdigest(), move_hash=move_hash, canonical_hash=canonical_hash,
                  move_sequence=moves.astype("<u2", copy=False).tobytes(),
                  setup_sequence=setup_codes.astype("<u2", copy=False).tobytes(),
                  sequence_schema=SEQUENCE_SCHEMA, feature_schema=FEATURE_SCHEMA,
                  legality_policy=LEGALITY_POLICY)
    if include_features:
        denom = max(target_moves, 1)
        features[:12] = (float(target == "b"), float(target == "w"), min(mainline_moves / 720, 1),
                         target_moves / max(mainline_moves, 1), passes / max(mainline_moves, 1),
                         target_passes / denom, len(setup_codes) / (size * size),
                         min((first_target_ply or 0) / 720, 1), min(target_captures / max(target_placements, 1) / 20, 1),
                         *(count / denom for count in phase_counts))
        for group in FEATURE_GROUPS:
            if group["kind"] == "histogram":
                view = features[group["start"]:group["stop"]]
                total = float(view.sum())
                if total:
                    view /= total
        result["features"] = features
    return result

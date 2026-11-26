"""Target pre-state representation on the pinned legacy parser, without file I/O.

Engineering contract: current pre-state plus the previous7 ply pre-states,
newest first; zero padded with a separate history mask. No superko/turn
enforcement claim. P3 policy receives17 planes; only P2 adds the action plane.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import numpy as np
from sgfmill import boards
from goid.sgf_features import analyze_sgf, decode_sequence
from .artifacts import ContractError

REPLAY_CONTRACT = "goid-v4-prestate-current-plus7-v1"


def board_array(board):
    x = np.zeros((board.side, board.side), dtype=np.int8)
    for color, (r, c) in board.list_occupied_points(): x[r, c] = 1 if color == "b" else -1
    return x


def legal_actions(board, color, *, ko_point=None, ko_color=None):
    n = board.side
    mask = np.zeros(n*n+1, dtype=bool)
    mask[-1] = True
    for r in range(n):
        for c in range(n):
            if board.get(r,c) is not None or ((r,c)==ko_point and color==ko_color): continue
            copy = board.copy()
            try: copy.play(r,c,color)
            except ValueError: continue
            mask[r*n+c] = copy.get(r,c) == color
    return mask


@dataclass(frozen=True)
class Decision:
    state_planes: np.ndarray   # own/opponent history16 + target color1, never action
    history_mask: np.ndarray
    board: np.ndarray          # physical b=1/w=-1, only pre-action board
    target_color: str
    action: int
    previous_action: int | None
    ply: int
    target_move: int
    stage: int
    legal_mask: np.ndarray | None

    def p2_planes(self):
        n = self.board.shape[0]
        a = np.zeros((1,n,n), dtype=np.float32)
        if self.action < n*n: a[0].flat[self.action] = 1
        return np.concatenate((self.state_planes,a), axis=0)

    def scalars(self):
        stage = [float(self.stage==i) for i in range(3)]
        return np.asarray([min(self.ply/720,1), min(self.target_move/360,1),
                           *stage,float(self.action==self.board.size),
                           *self.history_mask.astype(float)],dtype=np.float32)


def select_plies(moves, target_color, canonical_hash, *, quotas=(6,6,4), seed=31415):
    if len(quotas)!=3 or any(type(q) is not int or q<0 for q in quotas) or sum(quotas)>16 or sum(quotas)==0:
        raise ContractError("Pilot position quotas must sum to 1..16")
    choices=[[] for _ in range(3)]
    for ply,(color,_) in enumerate(moves,1):
        if color==target_color:
            stage=0 if ply<=60 else 1 if ply<=180 else 2
            key=f"{canonical_hash}|{target_color}|{ply}|{seed}".encode()
            choices[stage].append((hashlib.sha256(key).digest(),ply))
    return {ply for stage,q in zip(choices,quotas) for _,ply in sorted(stage)[:q]}


def replay_positions(text, target_color, *, quotas=(6,6,4), seed=31415,
                     include_legal_mask=True, expected_size=19):
    parsed=analyze_sgf(text,target_color,include_features=False)
    if parsed['board_size']!=expected_size:
        raise ContractError("Unexpected board size")
    n=expected_size
    moves=decode_sequence(parsed['move_sequence'],n)
    setup=decode_sequence(parsed['setup_sequence'],n)
    chosen=select_plies(moves,target_color.lower(),parsed['canonical_hash'],quotas=quotas,seed=seed)
    board=boards.Board(n)
    if not board.apply_setup({p for c,p in setup if c=='b'},{p for c,p in setup if c=='w'},set()):
        raise ContractError("Invalid effective root setup")
    past=[]; target_move=0; ko_point=ko_color=previous=None; decisions=[]
    for ply,(color,point) in enumerate(moves,1):
        current=board_array(board)
        is_target=color==target_color.lower()
        if is_target: target_move+=1
        if ply in chosen:
            history=[current]+list(reversed(past[-7:]))
            hm=np.asarray([i<len(history) for i in range(8)],dtype=bool)
            history += [np.zeros_like(current)]*(8-len(history))
            own=1 if color=='b' else -1
            planes=np.stack([plane for h in history for plane in (h==own,h==-own)]
                            +[np.full_like(current,color=='b',dtype=bool)]).astype(np.float32)
            action=n*n if point is None else point[0]*n+point[1]
            mask=legal_actions(board,color,ko_point=ko_point,ko_color=ko_color) if include_legal_mask else None
            if mask is not None and not mask[action]:
                raise ContractError("Actual action masked illegal; abort rather than unmask")
            for x in (planes,current,hm,mask):
                if x is not None:x.setflags(write=False)
            decisions.append(Decision(planes,hm,current,color.upper(),action,previous,ply,target_move,
                                      0 if ply<=60 else 1 if ply<=180 else 2,mask))
        past.append(current)
        if len(past)>7:past.pop(0)
        if point is None:
            ko_point=ko_color=previous=None
        else:
            # Parser already rejected occupied, suicide, simple-ko and malformed SGF.
            ko_point=board.play(*point,color)
            ko_color=('w' if color=='b' else 'b') if ko_point is not None else None
            previous=point[0]*n+point[1]
    return decisions


def transform_spatial(x,symmetry):
    if type(symmetry) is not int or not 0<=symmetry<8:
        raise ContractError("D4 index must be0..7")
    reflected=x[...,::-1] if symmetry>=4 else x
    return np.ascontiguousarray(np.rot90(reflected,symmetry%4,axes=(-2,-1)))


def transform_action(action,n,symmetry):
    if type(action) is not int or not 0<=action<=n*n: raise ContractError("Invalid action")
    if action==n*n:return action
    marker=np.zeros((n,n),dtype=np.uint8);marker.flat[action]=1
    return int(np.argmax(transform_spatial(marker,symmetry)))


def transform_decision(d,symmetry):
    n=d.board.shape[0]
    legal=None
    if d.legal_mask is not None:
        legal=np.concatenate((transform_spatial(d.legal_mask[:-1].reshape(n,n),symmetry).reshape(-1),d.legal_mask[-1:]))
    return replace(d,state_planes=transform_spatial(d.state_planes,symmetry),
                   board=transform_spatial(d.board,symmetry),action=transform_action(d.action,n,symmetry),
                   previous_action=None if d.previous_action is None else transform_action(d.previous_action,n,symmetry),
                   history_mask=d.history_mask.copy(),legal_mask=legal)


def collate_games(games,*,max_positions=16):
    """Canonical chronology, fixed16 slots, true position/history masks; synthetic CPU API."""
    import torch
    if not games or not 1<=max_positions<=16:
        raise ContractError('Invalid game/position capacity')
    planes=np.zeros((len(games),max_positions,18,19,19),dtype=np.float32)
    scalars=np.zeros((len(games),max_positions,14),dtype=np.float32)
    mask=np.zeros((len(games),max_positions),dtype=bool)
    for i,game in enumerate(games):
        if not game or len(game)>max_positions or len({d.ply for d in game})!=len(game):
            raise ContractError('Empty, oversized or duplicate game positions')
        for j,d in enumerate(sorted(game,key=lambda d:d.ply)):
            if d.state_planes.shape!=(17,19,19) or d.history_mask.shape!=(8,) or not np.array_equal(d.history_mask,np.arange(8)<min(8,d.ply)):
                raise ContractError('History mask does not match explicit ply contract')
            planes[i,j]=d.p2_planes();scalars[i,j]=d.scalars();mask[i,j]=True
    return torch.from_numpy(planes),torch.from_numpy(scalars),torch.from_numpy(mask)

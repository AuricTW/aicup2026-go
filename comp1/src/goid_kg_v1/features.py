"""Incremental official-encoder replay and streamed frozen decision features."""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import hashlib,time,json
import numpy as np
from . import native as n
from goid_v4 import score_push_v1 as c

def encode_game(item):
    index,text,expected,hd,config=item
    from katago.game.board import Board
    from katago.game.features import Features
    from katago.game.gamestate import GameState
    from goid.sgf_features import analyze_sgf,decode_sequence
    from goid_v4.replay import select_plies
    assert hashlib.sha256(text.encode()).hexdigest()==expected and hd in (0,1)
    parsed=analyze_sgf(text,'B',include_features=False);moves=decode_sequence(parsed['move_sequence'],19)
    setup=decode_sequence(parsed['setup_sequence'],19);assert not setup
    chosen=select_plies(moves,'b',parsed['canonical_hash'],quotas=(4,4,0),seed=20261104)
    if not chosen:return (index,None,'no_target_decision')
    rules=GameState.RULES_JAPANESE.copy();rules['whiteKomi']=6.5 if hd==0 else 0.
    gs=GameState(19,rules);fe=Features(config,19);metadata=n.profile_metadata('rank_1d').get_metadata_row('B',361)
    packed=[];globals_=[];legal=[];actions=[];scalars=[];previous_pass=False;double_pass=False
    try:
        for ply,(color,point) in enumerate(moves,1):
            pla=Board.BLACK if color=='b' else Board.WHITE
            loc=Board.PASS_LOC if point is None else gs.board.loc(point[1],18-point[0])
            if ply in chosen:
                if double_pass:return (index,None,'encore_after_consecutive_passes_not_supported')
                assert color=='b' and gs.board.pla==Board.BLACK
                gs.rules['passWouldEndPhase']=previous_pass
                x,g=gs.get_input_features(fe)
                if hd==1:g[0,5]-=.5/20. # exact C++ drawEquivalentWinsForWhite=1 NN adjustment
                assert ((x==0)|(x==1)).all() and np.isfinite(g).all()
                mask=np.asarray([gs.board.would_be_legal(pla,fe.tensor_pos_to_loc(i,gs.board)) and not gs.board.would_be_suicide(pla,fe.tensor_pos_to_loc(i,gs.board)) for i in range(361)]+[True])
                action=361 if point is None else (18-point[0])*19+point[1]
                assert mask[action]
                packed.append(np.packbits(x[0].reshape(-1).astype(np.uint8)));globals_.append(g[0]);legal.append(np.packbits(mask));actions.append(action)
                scalars.append([float(ply<=60),float(ply>60),min(ply/180,1),float(hd),float(action==361)])
            if ply>=max(chosen):break
            gs.play(pla,loc)
            double_pass=double_pass or (previous_pass and point is None)
            previous_pass=point is None
    except Exception as error:return (index,None,f'{type(error).__name__}:{str(error)[:140]}')
    return (index,(np.stack(packed),np.stack(globals_),np.stack(legal),np.asarray(actions,np.int64),np.asarray(scalars,np.float32),metadata),None)

def aggregate(z,aux):
    # z = local384,global384,action minus policy expectation384
    residual=z[:,768:1152]
    parts=[z.mean(0),residual.std(0)]
    for stage in (0,1):
        mask=aux[:,stage]>0
        parts.append(residual[mask].mean(0) if mask.any() else np.zeros(384,np.float32))
    parts.extend([aux[:,2:].mean(0),aux[:,2:].std(0)])
    for stage in (0,1):
        mask=aux[:,stage]>0;parts.append(aux[mask,2:].mean(0) if mask.any() else np.zeros(aux.shape[1]-2,np.float32))
    parts.append(np.asarray([np.log1p(len(z)),float((aux[:,0]>0).any()),float((aux[:,1]>0).any())],np.float32))
    out=np.concatenate(parts).astype(np.float32)
    assert out.shape==(2339,) and np.isfinite(out).all()
    return out

def infer_games(model,results,*,batch_size=32):
    import torch
    records=[r for r in results if r[1] is not None]
    if not records:return {}
    sizes=[len(r[1][3]) for r in records];fields=[]
    for j in range(5):fields.append(np.concatenate([r[1][j] for r in records]))
    px,glob,maskpacked,actions,aux=fields
    meta=np.concatenate([np.repeat(r[1][5][None],len(r[1][3]),axis=0) for r in records])
    zs=[];extras=[]
    for start in range(0,len(px),batch_size):
        actual=min(batch_size,len(px)-start);ix=np.arange(start,start+actual)
        ix=np.pad(ix,(0,batch_size-actual),constant_values=ix[0])
        spatial=np.unpackbits(px[ix],axis=1)[:,:22*361].reshape(batch_size,22,19,19).astype(np.float32)
        mask=np.unpackbits(maskpacked[ix],axis=1)[:,:362].astype(np.bool_)
        device=next(model.parameters()).device
        pi,h=n.forward(model,torch.from_numpy(spatial).to(device),torch.from_numpy(glob[ix]).to(device),torch.from_numpy(meta[ix]).to(device),torch.from_numpy(mask).to(device),amp=True)
        h=h.float().flatten(2);ac=torch.from_numpy(actions[ix]).to(device)
        local=h.gather(2,ac.clamp_max(360)[:,None,None].expand(-1,384,1)).squeeze(2)
        local[ac==361]=0
        mean=h.mean(2);expected=(h*pi[:,:361,None].transpose(1,2)).sum(2)
        z=torch.cat((local,mean,local-expected),1).cpu().numpy()[:actual]
        chosen=pi.gather(1,ac[:,None]).squeeze(1)
        rank=((pi>chosen[:,None])&torch.from_numpy(mask).to(device)).sum(1).float()/torch.from_numpy(mask.sum(1)).to(device)
        entropy=-(pi*torch.log(pi.clamp_min(1e-12))).sum(1)
        top=pi.topk(2,dim=1).values
        numbers=torch.stack((torch.log(chosen.clamp_min(1e-12)),rank,entropy,torch.log(top[:,0].clamp_min(1e-12)),top[:,0]-top[:,1]),1).cpu().numpy()[:actual]
        # phase2, then eight aux channels:logp,rank,entropy,logtop,gap,ply,HD,pass
        a=np.concatenate((aux[ix[:actual],:2],numbers,aux[ix[:actual],2:]),axis=1)
        zs.append(z);extras.append(a)
    z=np.concatenate(zs);a=np.concatenate(extras);out={};offset=0
    for record,size in zip(records,sizes):
        out[record[0]]=aggregate(z[offset:offset+size],a[offset:offset+size]);offset+=size
    return out

def tasks(indices,m,config):
    from goid_v4.pilot_scope_v1 import PointReader
    reader=PointReader(n.ROOT/'data/processed/audit.sqlite')
    try:
        for start in range(0,len(indices),32):
            ix=indices[start:start+32];records=reader.sample_text(m['rows'][ix]);texts={r['row_id']:r['sgf_content'] for r in records}
            yield [(int(i),texts[int(m['rows'][i])],m['raw'][i].decode(),int(m['hd'][i]),config) for i in ix]
    finally:reader.close()

def encode_batch(items):return [encode_game(i) for i in items]

def extract(limit=None):
    from concurrent.futures import wait,FIRST_COMPLETED
    device=n.runtime();model=n.load_backbone(device,False);m=c.metadata();config=n.read(n.RUN/'model_config.json')
    indices=np.load(n.RUN/'feature_indices.npy',allow_pickle=False)
    if limit is not None:indices=indices[:limit]
    dest=n.RUN/('feature_probe32_v2' if limit else 'feature_cache');dest.mkdir(exist_ok=False)
    array=np.lib.format.open_memmap(dest/'vectors.npy',mode='w+',dtype=np.float32,shape=(len(indices),2339));array[:]=0
    positions={int(i):j for j,i in enumerate(indices)};valid=np.zeros(len(indices),np.bool_)
    start=time.perf_counter();done=0;decisions=0;errors=[];iterator=iter(tasks(indices,m,config))
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending=set()
        for _ in range(8):
            batch=next(iterator,None)
            if batch:pending.add(pool.submit(encode_batch,batch))
        while pending:
            finished,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                results=future.result();out=infer_games(model,results)
                for index,data,error in results:
                    j=positions[index];done+=1
                    if error:errors.append({'index':index,'error':error})
                    else:array[j]=out[index];valid[j]=True;decisions+=len(data[3])
                if done%512==0 or done==len(indices):
                    array.flush();n.log(stage='HumanSL_features',games=done,total=len(indices),decisions=decisions,seconds=time.perf_counter()-start,errors=len(errors))
                batch=next(iterator,None)
                if batch:pending.add(pool.submit(encode_batch,batch))
    array.flush();np.save(dest/'indices.npy',indices,allow_pickle=False);np.save(dest/'valid.npy',valid,allow_pickle=False)
    n.write(dest/'RECEIPT.json',{'games':len(indices),'valid_games':int(valid.sum()),'decisions':decisions,'errors':errors,
        'seconds':time.perf_counter()-start,'profile':'rank_1d','use_swa':False,'fixed_batch':32,'FP16':True,
        'no_quality_labels_in_extractor':True,'source':'allowed original-fit metadata indices only','vectors_sha256':n.sha(dest/'vectors.npy')})

def probe32():extract(32)
def extract_all():extract()

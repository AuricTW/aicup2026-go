"""Pinned official KataGo Python encoder and frozen backbone adapter."""
from pathlib import Path
from datetime import date
import sys,json,hashlib,time,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'third_party/katago_source_v1_18_2_20261104'
WEIGHTS=ROOT/'third_party/katago_humansl_20261104_v1'
RUN=ROOT/'runs/katago_identity_experiments_20261104_v1'
PIN='fd0723fdbc0e9d82cf269c9630af8c27c57c07c4'
sys.path.insert(0,str(SOURCE/'python'))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def log(**x):print(json.dumps(x),flush=True)

def runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.65)
    if 'RTX 3060' not in torch.cuda.get_device_name():raise RuntimeError('Local3060 only')
    return torch.device('cuda')

def setup():
    import torch,subprocess,zipfile
    from urllib.request import Request,urlopen
    RUN.mkdir(exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==PIN
    checkpoint=WEIGHTS/'b18c384nbt-humanv0.ckpt'
    assert sha(checkpoint)=='84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27'
    # Trusted official checkpoint only. Preserve original; remove optimizer and
    # defaultdict metrics into a NEW weights-only-loadable inference artifact.
    raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
    clean={k:raw[k] for k in ('model','config','swa_model')}
    torch.save(clean,RUN/'HumanSL_state_only.pt')
    torch.load(RUN/'HumanSL_state_only.pt',map_location='cpu',weights_only=True)
    write(RUN/'model_config.json',raw['config'])
    selection=read(ROOT/'runs/katago_backbone_research_20261104_v1/OFFICIAL_ASSET_SELECTION.json')['engine_reference_only'][0]
    folder=ROOT/'third_party/katago_engine_v1_18_2_20261104';folder.mkdir(exist_ok=False)
    archive=folder/'official_engine.zip'
    with urlopen(Request(selection['url'],headers={'User-Agent':'Go-research/1.0'}),timeout=90) as r,archive.open('xb') as f:
        while b:=r.read(2**20):f.write(b)
    assert archive.stat().st_size==selection['size'] and sha(archive)==selection['digest'].split(':')[1]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in z.infolist():
            assert (folder/item.filename).resolve().is_relative_to(folder.resolve())
        z.extractall(folder)
    write(RUN/'SETUP.json',{'source_commit':PIN,'vendor_source_not_patched':True,
        'checkpoint_source_SHA':sha(checkpoint),'new_state_only_SHA':sha(RUN/'HumanSL_state_only.pt'),
        'official_legacy_checkpoint_loaded_once_with_weights_only_false':True,'new_state_loads_weights_only_true':True,
        'engine_zip':selection,'engine_path':str(next(folder.rglob('katago.exe'))),
        'isolated_environment':str(ROOT/'.venv_katago_v1'),'base_venv_not_modified':True,
        'source_code_license':str(SOURCE/'LICENSE'),'weight_license':str(WEIGHTS/'NETWORK_LICENSE_source.html')})

def profile_metadata(profile):
    from katago.game.sgfmetadata import SGFMetadata
    # Exact C++ makeBasicRankProfile fields at pinned commit, not a guessed rank map.
    ir={'rank_10k':19,'rank_1d':9,'rank_5d':5}[profile]
    return SGFMetadata(inverseBRank=ir,inverseWRank=ir,bIsHuman=True,wIsHuman=True,
        gameRatednessIsUnknown=True,tcIsUnknown=False,tcIsByoYomi=True,mainTimeSeconds=1200,
        periodTimeSeconds=30,byoYomiPeriods=5,gameDate=date(2020,3,1),source=SGFMetadata.SOURCE_KGS)

def gtp_loc(text,board):
    from katago.game.board import Board
    if text.lower()=='pass':return Board.PASS_LOC
    col='ABCDEFGHJKLMNOPQRST'.index(text[0].upper());row=19-int(text[1:])
    return board.loc(col,row)

def encode_query(q,config):
    from katago.game.board import Board
    from katago.game.gamestate import GameState
    from katago.game.features import Features
    rules=GameState.RULES_JAPANESE.copy();rules['whiteKomi']=q['komi']
    gs=GameState(19,rules)
    for color,point in q.get('initialStones',[]):gs.board.set_stone(Board.BLACK if color=='B' else Board.WHITE,gtp_loc(point,gs.board))
    gs.board.set_pla(Board.BLACK if q.get('initialPlayer','B')=='B' else Board.WHITE);gs.boards=[gs.board.copy()]
    for color,point in q['moves']:gs.play(Board.BLACK if color=='B' else Board.WHITE,gtp_loc(point,gs.board))
    features=Features(config,19);x,g=gs.get_input_features(features)
    if float(q['komi']).is_integer():
        # Pinned BoardHistory::whiteKomiAdjustmentForDraws: the game's komi
        # stays0 for HD1; only the NN input reflects white winning a jigo.
        adjustment=q.get('overrideSettings',{}).get('drawEquivalentWinsForWhite',.5)-.5
        g[0,5]+=(adjustment if gs.board.pla==Board.WHITE else -adjustment)/20.
    assert np.isfinite(x).all() and np.isfinite(g).all()
    mask=np.asarray([gs.board.would_be_legal(gs.board.pla,features.tensor_pos_to_loc(i,gs.board)) and
        not gs.board.would_be_suicide(gs.board.pla,features.tensor_pos_to_loc(i,gs.board)) for i in range(361)]+[True])
    meta=profile_metadata(q['overrideSettings']['humanSLProfile']).get_metadata_row(gs.board.pla,361)
    return x[0],g[0],meta,mask,gs

def load_backbone(device,use_swa=False):
    from katago.train.load_model import load_model
    model,swa,_=load_model(RUN/'HumanSL_state_only.pt',use_swa,device=device,pos_len=19)
    chosen=swa if use_swa else model
    chosen.eval().requires_grad_(False)
    if use_swa:del model
    return chosen

def forward(model,x,g,meta,legal,*,amp=False,with_trunk=True):
    import torch
    from katago.train.model_pytorch import ExtraOutputs
    extra=ExtraOutputs(['trunkfinal'] if with_trunk else [])
    with torch.inference_mode(),torch.autocast('cuda',enabled=amp,dtype=torch.float16):
        out=model(x,g,input_meta=meta,extra_outputs=extra)
        logits=out[0][0][:,0,:].float()
        pi=torch.softmax(logits.masked_fill(~legal,-1e9),dim=-1)
    return pi,extra.returned.get('trunkfinal')

def engineering(label='v1',use_swa=True):
    import torch,subprocess
    device=runtime();config=read(RUN/'model_config.json')
    dest=RUN if label=='v1' else RUN/('engineering_'+label)
    if label!='v1':dest.mkdir(exist_ok=False)
    records=[json.loads(s) for s in (ROOT/'runs/katago_backbone_research_20261104_v1/pilot0/queries.jsonl').read_text().splitlines()]
    chosen=[records[i] for i in np.linspace(0,len(records)-1,24,dtype=int)]
    t=time.perf_counter();encoded=[encode_query(q,config) for q in chosen]
    encode_seconds=time.perf_counter()-t
    x=torch.from_numpy(np.stack([v[0] for v in encoded])).to(device)
    g=torch.from_numpy(np.stack([v[1] for v in encoded])).to(device)
    meta=torch.from_numpy(np.stack([v[2] for v in encoded])).to(device)
    legal=torch.from_numpy(np.stack([v[3] for v in encoded])).to(device)
    model=load_backbone(device,use_swa);torch.cuda.reset_peak_memory_stats()
    profiles=[];prob={}
    for amp in (False,True):
        for bs in (1,8,32):
            t=time.perf_counter();out=[]
            for _ in range(2):
                for start in range(0,len(x),bs):
                    ii=np.arange(start,min(start+bs,len(x)));ii=np.pad(ii,(0,bs-len(ii)),constant_values=ii[0])
                    p,h=forward(model,x[ii],g[ii],meta[ii],legal[ii],amp=amp);out.append(p[:min(bs,len(x)-start)].cpu().numpy())
            elapsed=time.perf_counter()-t
            arr=np.concatenate(out[:len(out)//2]);other=np.concatenate(out[len(out)//2:])
            assert np.array_equal(arr,other) and np.isfinite(arr).all()
            prob[f'{amp}_{bs}']=arr
            profiles.append({'amp':amp,'batch':bs,'effective_positions_per_second':2*len(x)/elapsed,
                'repeat_bitwise_equal':True,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
    np.save(dest/'engineering_Torch_policy.npy',prob['False_8'],allow_pickle=False)
    write(dest/'ENGINEERING_TORCH.json',{'input_positions':24,'source_queries':1476,'feature_encode_seconds':encode_seconds,
        'spatial_shape':list(x.shape),'global_shape':list(g.shape),'meta_shape':list(meta.shape),
        'params':sum(p.numel() for p in model.parameters()),'use_swa':use_swa,'profiles':profiles,
        'fp16_vs_fp32_maxabs':float(np.max(abs(prob['True_32']-prob['False_32']))),
        'batch1_vs8_fp32_maxabs':float(np.max(abs(prob['False_1']-prob['False_8']))),
        'feature_encoder_official':True,'no_identity_quality_evaluated':True})
    cfg='numAnalysisThreads=1\nnumSearchThreadsPerAnalysisThread=1\nnnMaxBatchSize=8\nnnCacheSizePowerOfTwo=10\nnnMutexPoolSizePowerOfTwo=8\nnnRandomize=false\ncudaUseFP16=false\nmaxVisits=1\nlogToStderr=true\n'
    (dest/'engine_probe.cfg').write_text(cfg,encoding='utf-8')
    env=os.environ.copy();env['PATH']=str(ROOT/'.venv/Lib/site-packages/torch/lib')+os.pathsep+env.get('PATH','')
    exe=read(RUN/'SETUP.json')['engine_path']
    command=[exe,'analysis','-model',str(WEIGHTS/'b18c384nbt-humanv0.bin.gz'),'-config',str(dest/'engine_probe.cfg')]
    result=subprocess.run(command,input='\n'.join(json.dumps(q) for q in chosen)+'\n',capture_output=True,text=True,env=env,timeout=600)
    (dest/'engine_stdout.jsonl').write_text(result.stdout,encoding='utf-8');(dest/'engine_stderr.log').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,(result.returncode,result.stderr[-1000:])
    replies={r['id']:r for line in result.stdout.splitlines() if (r:=json.loads(line)) and 'id' in r and 'policy' in r}
    assert all(q['id'] in replies for q in chosen)
    ref=np.stack([replies[q['id']]['policy'] for q in chosen]);mask=ref>=0
    difference=float(np.max(abs(ref[mask]-prob['False_8'][mask])))
    write(dest/'ENGINEERING_PARITY.json',{'requests':24,'engine_exit':result.returncode,'max_abs_policy_difference':difference,
        'tolerance':.002,'passed':difference<=.002,'engine_probability_and_coordinates':True,
        'native_legal_mask_equal':bool(np.array_equal(mask,legal.cpu().numpy()))})
    assert difference<=.002 and np.array_equal(mask,legal.cpu().numpy())
    log(stage='engineering_complete',max_abs_engine_difference=difference,profiles=profiles)

def engineering_v2():engineering('v2',False)

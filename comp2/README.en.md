<p align="center"><a href="README.md"><img src="../.github/assets/lang-zh-off.svg" alt="繁體中文"></a> <a href="README.en.md"><img src="../.github/assets/lang-en-on.svg" alt="English"></a></p>

# Competition 2: Predicting Go Player Rank from Game Records (TEAM_11306)

← Back to [repository home](../README.en.md)

Final submission: Private 0.789846 (6th place). This folder holds the code that produced that file: data processing, feature extraction, model training, evidence computation, fusion and ensembling, and reproduction checks.
The method is described in the report; this document explains how to install, where to put the data and weights, the order in which to run things, and how to reproduce the final submission.

| Folder | Contents |
|---|---|
| `src/` | Python code used by the final submission (all in one flat folder, importing each other) |
| `vm/` | Shell scripts run on the GPU host (KataGo features, identity networks, reuse of comp1 tokens, rank1) |
| `prev_system/` | Code of the previous system (Codex era); the second-stage fusion of the final file uses its fixed posteriors |
| `analysis/` | Analysis code needed only for the report's methods and figures (does not affect the submission) |
| `configs/` | Machine settings example `paths.example.yaml`; training arguments stored in the checkpoints, `training_args.json` |
| `SOURCES.tsv` | For every file in the repo: its original source file, that file's SHA-256, and whether paths were changed |
| `weights_manifest.tsv` | File names, sizes and SHA-256 of the Release weights and intermediate files |

## 1. Installation

| Item | Version |
|---|---|
| Python | 3.12 (3.12.14 used in development) |
| CPU stages (datasets, HT models, evidence, fusion, submission) | PyTorch 2.6.0+cu124 and other packages, see `requirements.txt` (versions pinned, taken from the environment that produced the final file) |
| GPU-host stages | Environment A: PyTorch 2.10.0+cu128 (KataGo features, identity networks); environment B: PyTorch 2.11.0+cu128 (comp1 token worker, heads, rank1), see `requirements-gpu.txt` |
| GPU | Reproducing R1 and R2 needs no GPU; HT models used an RTX 3060 12 GB; feature extraction and identity networks used an RTX 4090 24 GB |
| KataGo engine | v1.18.2, Linux CUDA 12.8 build: https://github.com/lightvector/KataGo/releases/tag/v1.18.2 (the executable is not in the repo) |
| Operating system | Local machine Windows 11; GPU host Ubuntu 24.04 |

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
cp configs/paths.example.yaml configs/paths.yaml      # then edit for your machine
```

- Machine-specific settings go in `configs/paths.yaml`, or can be overridden with environment variables `AICUP2_<KEY>`; they are read by `src/comp2_paths.py`.
  - Official data root `root`.
  - Intermediate-file folder `work`: the original `vm_out/`, holding datasets, model outputs and evidence files.
  - Member posterior folder `parts`.
  - Fixed outputs of the previous system `d64_*`.
  - Locations of the KataGo executable and weights.
- Settings for the GPU-host scripts go in `vm/env.sh`, copied from `vm/env.example.sh` and then edited.
  - Working folder `KQ`.
  - The two Python environments `PY`, `PY1`.
  - Location of the comp1 weights `C1RUNS`.
  - Scratch SSD `SCRATCH`.
- KataGo's Python modules and sgfmill are in `../third_party/` one level up (original files unmodified, with their original licences). `extract_trunk*.py` adds this path automatically.
- The comp1 token worker uses `../comp1/payload/`; see comp1's README.

## 2. Data placement

Only data provided by the organisers is used. This repo contains no raw data, features, evidence files or pseudo-labels.

| Data | Source | Location (relative to `root`) |
|---|---|---|
| Official training data v1.1 (10 CSVs, 1,000,000 games, 8,007 players) | https://github.com/AILAB-NDHU/aicup2026-go-dataset (release v1.1) | `data/raw/training_v1_1_20261104/training/train_*.csv` |
| Official training data v1.0 (used for the player split and for hashing the token payload) | Same as above (release v1.0) | `data/raw/training/` |
| Official test queries (400 each for Public / Private) and sample submission | Organisers (AIdea competition page) | `data/raw/official_rank_tests_20261104_v1/tests/` (`rank_prediction_test_public.csv`, `rank_prediction_test_private.csv`, `sample_submission_rank.csv`) |
| Player split `data/interim/index.csv` | Produced by `prev_system/tools/prepare_data.py` (train 6,402 players, dev 801, lockbox 804) | `data/interim/index.csv` |

The dataset-building scripts on the GPU host (`make_gamelist*.py`) read `raw/training/`, `raw/tests/` and `raw/split_index.csv` under `gpu_data` (copies of the same official files and split file).

## 3. Run order

The steps below are the actual chain of the final file.

- Steps 3.5 and 3.6 (R2, R1) have been run and verified in this repo; see section 6.
- The commands of steps 3.1 to 3.4 were reconstructed from each script's arguments, the `vm/` scripts and the training arguments stored in the checkpoints (`configs/training_args.json`). The dataset-building commands were not recorded one by one at the time, and these steps were not rerun in this repo.
- "GPU" means run under `$KQ` on the GPU host; everything else runs locally in the comp2 root, with intermediate files in `work/`.

### 3.1 Preprocessing (local, CPU)

```bash
python prev_system/tools/prepare_data.py --training-dir data/raw/training --output-dir data      # player split data/interim/index.csv
python src/build_moves_dataset.py work/moves_all.npz --L 200 --max-hd 1      # move sequences (identity networks)
python src/build_moves_extra.py work/moves_all.npz work/moves_all2.npz          # + remaining games of dev players (split code 3)
python src/build_moves_lockbox.py work/moves_all2.npz work/moves_all3.npz       # + games of lockbox accounts (split code 4, gallery classes only)
```

- `oof_games.csv` and `oof_dense_map.csv` in `fixed_inputs/work/` are the list of 949 OOF validation players (Release); put them in `work/`.
- The GPU host's `tm/` also needs the same `moves_all*.npz`.

### 3.2 Features (GPU)

```bash
python $SRC/make_gamelist.py gamelist.jsonl                     # game lists for dev / training / test queries
python $SRC/make_select2.py gamelist.jsonl select_h_stage2.json A:20,T:all,C:10   # selection spec reconstructed from the notes in chain_stage2.sh
bash vm/chain_stage2.sh                                          # Human SL per-move likelihoods (kata_human.py, KataGo engine, 1 visit)
bash vm/run_trunk_oof_vm.sh                                      # Human SL trunk PCA64 embeddings (extract_trunk.py, PCA from the Release's assets/pca64.npz)
python $SRC/make_gamelist_id.py gamelist_id.jsonl trunk/abt,trunk/oof 400 && bash vm/run_trunk_id_vm.sh   # embeddings of the first 80 moves of gallery games
```

### 3.3 Training

The HT sequence models were trained locally on an RTX 3060:

```bash
python src/build_h_dataset.py work/dsH_v1.npz <H_DIRS> --L 320
python src/build_ht_dataset.py work/dsH_v1.npz <TRUNK_DIRS> work/dsHT_v1.npz
python src/seq_model.py work/dsHT_v1.npz work/ht1_s1 --epochs 14 --d 192 --layers 4 --L 320 --bs 128 --lr 6e-4 --split --aux 0.5 --seed 1   # seed 2 → ht1_s2
```

The identity networks were trained on the GPU host: 15 networks in three tracks, run in order:

```bash
bash vm/run_idnet_vm.sh id_v2x --L 200 --epochs 30 --seed 0 --eval-every 4000 --holdout-players $KQ/oof_games.csv   # teacher; the rest are in the scripts below
bash vm/chain_board_vm.sh; bash vm/chain_board_ens_vm.sh; bash vm/chain_board2_vm.sh        # board-patch networks of the x / full tracks
bash vm/chain_trunkx2_vm.sh; bash vm/chain_trunkx3_vm.sh                                     # trunk-embedding networks of the x track → idnet/tx2_full/acc_6m.npz
bash vm/chain_lockbox_vm.sh; bash vm/chain_lockbox2_vm.sh; bash vm/chain_trunklb_vm.sh      # test track (with lockbox gallery) → idnet/ta2lb/acc_6m.npz
```

- The full arguments of every network (epochs, seed, L, board, trunk, pseudo-label file and threshold) are stored in the checkpoint's `args` and collected in `configs/training_args.json`.
- The arguments of the teacher networks `id_v2x`, `id_v2full`, `id_v4lb` and the first-round students `e3_ps05`, `f1_ps05`, `a2lb_ps05` are also defined by that file.
- Pseudo-labels are produced by `idnet_pseudo.py` (threshold 0.5).

rank1 was trained in environment B on the GPU host; its input is comp1's bigbank token cache (training-file games only):

```bash
bash vm/rank1_chain.sh      # tok_ranknet_vm.py rank1 tokres_rank_labels.npz <bigbank results> --epochs 8 --seed 0
```

### 3.4 Inference and evidence

```bash
# Identity-network ensemble evidence (GPU; the last step of each chain script above, idens.py --acc)
#   x track idnet/tx2_full/acc_6m.npz, full track idnet/bf1_p9c64/acc_3.npz, test track idnet/ta2lb/acc_6m.npz → copy to work/idnet_*/
# Reuse of comp1 tokens (GPU, environment B)
python $SRC/make_tok_payload.py tm/moves_all3.npz <root> <comp1 payload code dir> tokpay      # worker input for the opponent side of the query games
bash vm/run_tok.sh                    # early-game tokens (frozen comp1 worker)
bash vm/run_toklate_c2.sh             # late-game tokens (moves 181–300)
bash vm/ft_cache.sh testdev oof; bash vm/ft_run.sh t6 <trunk_ft.pt>; bash vm/ft_early_vm.sh t6     # once for each of the three fine-tuned trunks t6 / b3 / ha
python $SRC/tok_apply.py ... / tok_apply_late.py ...      # 10 heads → work/tokres/{td,oof}_<head>.npz, work/xf/{td,oof}_<head>_<trunk>.npz
# rank1 application and conversion
python src/tok_rank_apply.py <rank1/model.pt> work/moves_all3.npz work/tokrank/rank1_dir.npz "<own specs>" "<opp specs>"
python src/rank1_conv.py work/tokrank/rank1_dir.npz work/idnet_tx2_full/acc_6m.npz work/tokrank/r1
python src/rank1_conv.py work/tokrank/rank1_dir.npz - work/tokrank/r1full dev=work/idnet_bf1_p9c64/acc_3.npz test=work/idnet_ta2lb/acc_6m.npz
# Add opponent counts of the listed games to the account profiles (enr group)
python src/listed_counts.py <REF.npz> <PSEUDO.npz> work/tokeval/cntadd_<x|full|test>.npz
# HT model predictions for the dev / test / OOF query games
#   → work/ht1_s{1,2}/{dev,test}_pred.npz, work/oofpredHT_s{1,2}/all_pred.npz
```

`tok_classmap.npz` (comp1 classes → comp2 accounts) is produced by `tok_classmap.py`; the Release includes a copy in `assets/`.

### 3.5 Fusion: 91 members (local, CPU)

```bash
python src/run_parts.py --workers 4
```

- For each member, `tok_evidence.py` first produces its token evidence set (three tracks; skipped if it already exists), then `fuse_sources.py` is called.
- Each member writes `work/parts/go_rank_<tag>_*`.
- The settings of the 91 members are in `src/members.py`, a Python version of the original PowerShell drivers `make_variants2/6/7/8.ps1` and `cnt_round.ps1`:
  - free or ridge estimation in the first stage;
  - dev query sampling seed `DEV_SEED`;
  - `ZOPP_GATE`;
  - token network combination;
  - rank1 source.

### 3.6 Producing the submission (local, CPU)

```bash
python src/blend_final.py --out work/final/go_rank_final.csv
```

The posteriors of the 91 members are combined by geometric mean and then decoded by expected score. The script prints the SHA-256 at the end and compares it with the final file.

## 4. Inputs and outputs of the main modules

| Module | Input | Output |
|---|---|---|
| `build_moves_dataset.py` / `_extra.py` / `_lockbox.py` | Official training CSVs, test queries, `index.csv` | `moves_all{,2,3}.npz`: move coordinates, accounts, ranks, split code |
| `kata_human.py` (+ SGF parsing in `kata_eval.py`) | Game list jsonl, KataGo engine, Human SL network in engine format | Per-move log p, ranking and win rate under several rank settings, `hchunk_*.npz` |
| `extract_trunk.py` / `extract_trunk_id.py` | Game list, Human SL checkpoint, `pca64.npz` | Trunk PCA64 embedding of the position before each move, `tchunk_*.npz` |
| `build_h_dataset.py` → `build_ht_dataset.py` | Likelihood chunks, embedding chunks | HT dataset `dsHT_v1.npz` |
| `seq_model.py` | HT dataset | HT model `model.pt`; `dev_pred.npz`, `test_pred.npz` (rank logits of the target player and the opponent) |
| `idnet.py` / `idnet_pseudo.py` | `moves_all*.npz`, (optional) embeddings, pseudo-labels | Identity network `model.pt`, `idnet_pred.npz`; pseudo-labels `pseudo_*.npz` |
| `idens.py --acc` | Several identity networks | Ensemble evidence `acc_*.npz`: identification probabilities, account profiles, rank kernel |
| `tok_apply.py` / `tok_apply_late.py` | comp1 heads, token results, `tok_classmap.npz` | Account identification of the opponent side of each game, `{td,oof}_<head>.npz` |
| `tok_evidence.py` | Reference ensemble evidence, token head outputs | Second-identifier evidence `tokeval/{x,full,test}_<set>_A1.0.npz` |
| `tok_ranknet_vm.py` → `tok_rank_apply.py` → `rank1_conv.py` | Token cache, rank labels | rank1 `model.pt`; `tokrank/r1{,full}_{own,conv,raw}.npz` |
| `fuse_sources.py` (`stage1_lib.py`, `evaluate_queries.py`, `final_predict.py`, `predict_test.py`, `idnet_eval.py`) | The evidence above, previous-system posteriors, OOF player list | One member's `x_D64_posteriors.npz` (test queries), `OOF_posteriors.npz`, `VALIDATION.json` |
| `blend_final.py` (`blend_test.py`) | 91 `x_D64_posteriors.npz` | Submission CSV and posteriors |

## 5. Weights and intermediate files

Weights are not in git; download them from this repo's GitHub Release. The asset `comp2_weights.tar` of Release `v1.0` contains every file listed in this section plus `MANIFEST.csv` (the SHA-256 of the asset is in `SHA256SUMS.txt`):

```bash
mkdir -p weights/comp2
tar -xf comp2_weights.tar -C weights/comp2/     # the built-in tar also works in Windows cmd
```

- Extract the Release files, following the relative paths in `weights_manifest.tsv`, into any folder, for example `weights/comp2/` (the 3 rows whose kind is `comp1 release` are in comp1's Release, see below).
- Then point `configs/paths.yaml` and `vm/env.sh` to that location.
- Locations on the GPU host:
  - `ht/`, `idnet/<name>/`, `rank1/` correspond to `work/ht1_s{1,2}/`, `$KQ/idnet/<name>/`, `$KQ/rank1/` respectively.
  - Put `assets/pca64.npz` in `$KQ/trunk/`.
  - Put `assets/tok_classmap.npz` in `$KQ/ft/`.

Weights trained by our team in comp2 (18 files, 525.0 MB):

| File | bytes | SHA-256 |
|---|---|---|
| `ht/ht1_s1/model.pt` | 10,851,100 | `191dd74d32c9e71e4757415899a9f4b507480f1a7f1b6d2db5785429660f60d2` |
| `ht/ht1_s2/model.pt` | 10,851,100 | `aaabb4f40c64c5d01237912ac12c7b7e7bab98f7b12f48fbe51eb757fc633cc3` |
| `idnet/id_v2x/model.pt` | 30,244,206 | `28c099748ff3fc0fff786239ebb18539517c6d8ce1d768c091f444137430901c` |
| `idnet/e3_ps05/model.pt` | 30,121,454 | `5a82278adafaab2e72e2920121094dc522ccc2ca00f251bd23c2fa6d337c915c` |
| `idnet/bx1_p9c64/model.pt` | 30,972,786 | `ba01b920609d6741c6cf1c8c6ce221011dc0bc4484a87bfcc133b8397ca23c62` |
| `idnet/bx2_p9c64/model.pt` | 30,972,786 | `63e7aa37e7c3f849724e0b7aa86320e9eacb45179d3d7a010f518013273052b3` |
| `idnet/tx_full/model.pt` | 31,303,775 | `d5aab5c7e18f339d842fb48ac79f0858b19ffe1b7d5a437014dfec39dd422018` |
| `idnet/tx2_full/model.pt` | 31,303,839 | `4d9d3793569639ebd8d706c8db92ff450dd645314f49ab07d568ba755cfc4d57` |
| `idnet/id_v2full/model.pt` | 31,316,654 | `c12028ec5c8320c211aee9dd8d6530b9b2940622b98446c6528d9c54a3db43d6` |
| `idnet/f1_ps05/model.pt` | 31,193,902 | `c5686446741c93e1352eeb7f52b6b889d77d82a5dc40bc930b1f06c0655cab38` |
| `idnet/bf1_p9c64/model.pt` | 32,045,234 | `4cc077cbcbc8bb9ba17b95538ecc5d19cccd08b688f66ec4efdeb9289f094b7b` |
| `idnet/id_v4lb/model.pt` | 33,684,462 | `846691e089a98413b70ebb51cceb56406226da45ba0d5345afd92896457af390` |
| `idnet/a2lb_ps05/model.pt` | 33,684,526 | `38e27d40f53be401d15605f2f6bbb5fb5da1969e38c5a3c8b0e281225e81787e` |
| `idnet/ba2lb_p9c64/model.pt` | 34,535,602 | `685a8ff10ab4d9c011447a5eb7398b7c968881a190dbd632076f1fd61681e294` |
| `idnet/ba3lb_p9c64/model.pt` | 34,535,602 | `f89d256ca6554b0312f2a95d0c30075e12673efee55d9da6005e4f3e33c714a2` |
| `idnet/ta1lb/model.pt` | 34,866,591 | `2fb250dc5f9c70bd2407123a698785fced92d83cf5e92afc43493d034d22fb58` |
| `idnet/ta2lb/model.pt` | 34,866,655 | `9fe1d4abf6d59487dc760b68d981017663c3ba9f3a93ebdf94808110a90ae7c0` |
| `rank1/model.pt` | 17,664,452 | `c1692e222bd6b56263e5a1a53e20ae47bad9d79e0c82e78843996d304aceee0b` |

Other Release files (full SHA-256 in `weights_manifest.tsv`):

| Category | Files | Purpose |
|---|---|---|
| Small assets | `assets/pca64.npz` (PCA parameters), `assets/tok_classmap.npz` (class mapping), `rank1/LOG.json` | Trunk embeddings, applying the token heads |
| Fixed outputs of the previous system | `fixed_inputs/prev_system/`: test-query posteriors, OOF posteriors `OLD_FUSED.npz`, OOF queries `queries_dense.json` | Second-stage fusion |
| OOF validation player list | `fixed_inputs/work/oof_games.csv`, `oof_dense_map.csv` | Validation queries for fusion |
| Test-query posteriors of the 91 members | `parts/go_rank_<tag>_x_D64_posteriors.npz` (9.1 MB) | Reproducing R1 |

Weights provided by other sources:

- **The comp1 weights are provided by comp1's Release.** Their paths in comp1's `weights_manifest.tsv` are listed below. After downloading, keep the `runs/...` structure and point `C1RUNS` in `vm/env.sh` to that `runs/`.
  - 8 early-game heads `runs/tokhead_20261116_v1/{vm_t1,vm_t6,full_s1,full_big_s3,full6a_s12,vm_t4,vm_t10,vm_t2}/model.pt`.
  - Late-game heads `vm_t17late`, `full_late_s24`.
  - Fine-tuned trunks `wl_h8x48_b3_trunk_ft.pt` (b3), `e2e_warm_long_h5e5/trunk_ft_step7500.pt` (t6), `runs/tokfull_20261116/e2e_warm_h8x48a/trunk_ft.pt` (ha).
  - 3 of them are used only by comp2's second identifier; their SHA-256 is also listed in this repo's `weights_manifest.tsv` (kind `comp1 release`):

    | comp1 Release path | bytes | SHA-256 |
    |---|---:|---|
    | `runs/tokhead_20261116_v1/vm_t4/model.pt` | 54,487,821 | `51a42b2aed54aac715dd53740038ce0f81f57837bf85bcbda59c58db1d87eda6` |
    | `runs/tokhead_20261116_v1/vm_t10/model.pt` | 33,913,545 | `aa0a7d1677363bbc8cfd9bbefada25474c5c8c763ccde102ab2000bbe76ec356` |
    | `runs/tokfull_20261116/e2e_warm_h8x48a/trunk_ft.pt` | 35,157,460 | `fc33fa4407265f62b0f7b2c320a72ae7232c11262e32dec13b4a7091180c49be` |
  - `HumanSL_state_only.pt`.
  - These weights were trained only on the official training files, which are shared by both competitions and byte-identical.
- **External weights** are not redistributed; download them from KataGo's official site:
  - Human SL network `b18c384nbt-humanv0.ckpt` (training checkpoint, SHA-256 `84b23476…`) and the engine format `b18c384nbt-humanv0.bin.gz`; source https://katagotraining.org/extra_networks/ , licence https://katagotraining.org/network_license/ .
  - The engine-format file our team used was exported from the same checkpoint with KataGo's export tool (SHA-256 `8b03891e…`). The hash of the official `.bin.gz` may differ.

## 6. Reproducing the final submission

Final submission: `go_rank_IDv10x_union91_20261120.csv`, 12,043 bytes.

SHA-256 `1ADC681D208EF87977103CDA1604D22AAB3041A020F79471F13A35C59404CA4A`

| Level | Start → end | Command | Verification result in this repo |
|---|---|---|---|
| R1 | 91 member posteriors from the Release → CSV | `python src/blend_final.py --parts <weights>/comp2/parts --out out.csv` | **Byte-identical** (SHA-256 as above), a few seconds on CPU |
| R2 | Saved evidence files and model outputs → 91 members → CSV | `python src/run_parts.py`, then R1 | Run on the original project's evidence files: **all 91 members byte-identical, same CSV SHA-256** (CPU, 4 parallel workers, 22 minutes) |
| R3 | Raw game records + Release weights → inference → CSV | Inference parts of steps 3.2 and 3.4 + R2 | Not rerun, see section 8 |
| R4 | Raw training files → retrain all models → CSV | All of section 3 | Not rerun; only code and arguments are provided |

The evidence files used by R2 are features derived from the official data and are not in the Release; R2 can only be run after carrying out steps 3.1 to 3.4 yourself.

## 7. Hardware and time estimates

| Step | Hardware | Time (estimated, from the original records) |
|---|---|---|
| R1 | CPU | A few seconds |
| R2 (91 members) | CPU (i7-12700, 4 parallel workers) | 22 minutes (measured) |
| Preprocessing and datasets | CPU | About 10 minutes |
| Human SL per-move likelihoods | RTX 4090 | About 5 hours |
| Trunk embeddings (query games) | RTX 4090 | About 3 hours; the first 80 moves of the 792,000 gallery games take about 16 more hours |
| HT models (2 seeds × 14 epochs) | RTX 3060 | About 0.5 hours |
| Identity networks (15 networks + 6 pseudo-label rounds) | RTX 4090 | About 35–45 GPU hours (1.5–3.5 hours each) |
| comp1 token extraction and head application (43,389 games) | RTX 4090 | About 7 hours |
| rank1 | RTX 4090 | About 1–2 hours |

## 8. Known limitations

1. **Only R1 and R2 have been verified**.
   - Both were confirmed byte-identical.
   - R2 was run on the original project's intermediate files.
   - R3 and R4 were not rerun in this repo. The commands of steps 3.1 to 3.4 were reconstructed from script arguments and the training arguments in the checkpoints; the dataset-building commands were not recorded one by one at the time.
2. **GPU-host environment A no longer exists**.
   - The original environment was PyTorch 2.10.0+cu128, and the KataGo engine also took its cuBLAS / cuDNN from it; no full lock file was kept.
   - A rebuild uses the versions in `requirements-gpu.txt` instead, and results may differ slightly.
3. **There is no inference-only program for the identity networks**. `idnet.py` writes `idnet_pred.npz` at the end of training; running new games with only the Release weights requires writing a separate prediction program that loads `idnet.Net`.
4. **The previous-system code (`prev_system/`) is for reference only**.
   - It reads upstream runs through the original project's relative paths; the upstream features and models were not packaged into this repo.
   - The final file uses only the posteriors it saved; these are included in the Release under `fixed_inputs/prev_system/`.
5. **Byte-identical reproduction of retraining cannot be guaranteed**. The identity networks use non-deterministic GPU operations, so the same seed can only be expected to give similar validation scores. Byte-identical reproduction applies only to R1 and R2.
6. The `analysis/` code needs the members' OOF posteriors and the mirror-validation evidence. These files are not in the Release because they are derived from training players' data; the included code serves only to document the analysis methods in the report.
7. The identity-network checkpoints in the Release contain the anonymous account names of their classes (`owners`), which are the same as the player_ids in the official training files.

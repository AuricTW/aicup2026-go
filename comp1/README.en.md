<p align="center"><a href="README.md"><img src="../.github/assets/lang-zh-off.svg" alt="繁體中文"></a> <a href="README.en.md"><img src="../.github/assets/lang-en-on.svg" alt="English"></a></p>

# Competition 1: Identifying Players from Sets of Go Game Records (TEAM_11307)

← Back to [repository home](../README.en.md)

Final submission: Private 0.923441 (4th place). This folder contains the code that produced that file: data processing, model training, inference, and reproduction checks.
The method is described in the report; this document explains how to install the environment, place the data and weights, run the steps in order, and reproduce the final submission.

## 1. Environment setup

| Item | Version |
|---|---|
| Python | 3.12 (developed with 3.12.14) |
| PyTorch | 2.14.1+cu126 (CUDA 12.6); 2.11.0+cu128 on the GPU host |
| Other packages | See `requirements.txt` (versions pinned) |
| GPU | Inference and reproduction: NVIDIA GPU with 12 GB or more (developed on an RTX 3060 12 GB); per-move feature extraction and large training runs: RTX 4090 24 GB |
| Operating system | Tested on both Windows 11 and Ubuntu 24.04 (the code uses only the standard library and the packages above) |

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Machine-specific settings (location of the Human SL weights, working folder on the GPU host, the Python used to launch workers, a tar that can stream .7z) are in
`configs/paths.example.yaml`; copy it to `configs/paths.yaml` and edit it, or override values with the environment variables `AICUP_<KEY>`. All other code locates its input files with `comp1/` as the root directory (`data/`, `runs/`, `submissions/`; these three folders are not in git).

The per-move feature worker (`payload/`) needs the KataGo Python modules and sgfmill: before running it, copy `../third_party/katago` and `../third_party/sgfmill` to `payload/vendor/katago` and `payload/vendor/sgfmill` (the original files are unmodified; `payload/MANIFEST.json` checks their SHA-256).

## 2. Data download and placement

| Data | Source | Location |
|---|---|---|
| Official training data v1.0 (10 CSV files, 1,000,000 games) | Organizers: https://github.com/AILAB-NDHU/aicup2026-go-dataset (release v1.0) | `data/raw/training/` |
| Official training data v1.1 (same as above, with the handicap HD added at the start of each SGF) | Same as above (release v1.1) | `data/raw/training_v1_1_20261104/training/` |
| Official test data v2 (November 11 version) | Organizers (AIdea competition page) | Extract to `data/raw/official_tests_20261111_v2/extracted/tests/` (must contain `player_identification_candidates2_*.csv`, `player_identification_test2_*.csv`, `player_identification_candidates_*.csv`, `player_identification_test_*.csv`, `sample_submission_player_v2.csv`) |
| Fox Go game records (external data, GPL-3.0) | https://github.com/featurecat/go-dataset | Drive `D` or any location, passed as an argument when running `extfox_*.py` |

This repo does not include any raw data, external game records, or features derived from them.

### Obtaining and filtering the Fox game records
1. Download the 13 archives below (concatenate the split volumes `.7z.001`, `.002` in order into a single `.7z`). `Pro.7z` and `8d.7z` were used only to check the format and did not enter training.

| Archive | Bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `1d2.7z` | 168,279,345 | `2533a3582f868551ed246cbfeb9c056aca3d6e82b35d4b0b3070f6c795c0c7c5` | Account selection |
| `1k2.7z` | 119,031,002 | `d9d6d33ad9cec22037ef4136097b8bc32ca40a61c6758b3facbb78ecf5a4e641` | Account selection |
| `2d2.7z` | 215,462,071 | `2f696e4604153d1bec7626d595790359545fe741ef275ae0686a8dad962e8ca6` | Account selection |
| `2k2.7z` | 87,611,384 | `763c244fb700fee4715c3dc583a3a16883617581a6c29fdab33788fe15564748` | Account selection |
| `3d2.7z` | 255,343,139 | `ef73a31fd8912baaffb45b42120b6de96d337f297c2b82845ee0459352ba76da` | Account selection |
| `3k2.7z` | 57,840,412 | `09f232cdbac8746c7ec2cfbbbeab9e9bb9b35397275d11657b2f62521d11cde5` | Account selection |
| `4d2.7z` | 171,211,054 | `64d0e380034735dfbff1d9ecd08a393ad3545801b213ffd79fab24737c511fef` | Account selection |
| `5d2.7z` | 113,412,556 | `dd214c5700cfbe8f58c8338b032efdd0bfec313a7028ffda0db1b8049cad42d3` | Account selection |
| `5k2.7z` | 46,585,984 | `efa73a40aee2db76fdb7b1fe71e3ce6a67ea5ad3c815e0448e9bb5dbe698885c` | Account selection |
| `6d2.7z` | 48,503,393 | `c2b1ce24a9a6110017a49cf8a6cc0a17b3f59c1d81fe6498df41d86668feee21` | Account selection |
| `7d2.7z` | 19,521,177 | `96fc1fbdd0f46d33cf1923f5de2e2201697bd061c70ca4833a582180dff18ffa` | Account selection |
| `8d.7z` | 14,915,255 | `b212205339349e5c2ad7b6bbd2984cc18968322c8f6c690ffd20788782ac9f96` | Format check only |
| `Pro.7z` | 3,657,149 | `4b87b49adac5626d78ef48e05a122341b262d229e713d974338d573bff0ec735` | Format check only |

2. Scan (streams the 7z without extracting it into small files): `python scripts/extfox_20261118_v1.py scan <archive.7z> <meta.npz>`, once per volume.
   Filter: 19×19, no setup stones, handicap HA ≤ 1, at least 60 moves, no passes, both players have accounts. The raw bytes of each account name go straight into a salted SHA-256 and only the first 8 bytes are kept; names are never decoded or stored. External games are not matched against any competition game.
3. Select accounts: `python scripts/extfox_20261118_v1.py select <out dir> <meta.npz> ... --games-per 16 --accounts 24000` (first batch, 24,000 accounts × 16 games); the second batch of 48,000 distinct accounts × 8 games uses `scripts/extfox_20261119_v2.py`.
4. Export to the `games.jsonl` of the existing pipeline: `python scripts/extfox_20261118_v1.py export <out dir> <archive.7z> ...`, then extract per-move features with the same frozen Human SL trunk used for the competition games, and finally merge them into the per-move model labels with `scripts/extfox_merge_20261118_v1.py` (external accounts serve only as extra classification classes).

## 3. Execution order

All commands below are run in `comp1/`. Training arguments (the arguments each model actually used, number of classes, number of games, seconds) are listed in full in `configs/training_args.json`.

### 3.1 Data processing
| Step | Command | Output |
|---|---|---|
| Parse and audit the 1 million games | `python scripts/audit_training.py index`, `... parse` (first runs the default 10,000-game calibration), `... parse --selection all --workers 4 --batch-size 128`, `... summary` | `data/processed/audit.sqlite`, feature matrices, audit summary |
| Near-duplicate grouping | `python scripts/audit_near_duplicates.py --output data/processed`, then `python scripts/apply_family_groups.py` | `near_duplicate_groups.jsonl`; family keys in `audit.sqlite` (never split apart) |
| Rebuild intermediates from official data (not in the Release) | `python scripts/rebuild_official_derived_20261126_v1.py all`: runs `fitrows`, `splits`, `scope`, `keys`, `extmeta4`, `extmeta5`, `moves`, `moves_ext`, `bundle`, `labels`, `oppbank`, `oldpool` in order (the last two steps need a GPU), then checks every SHA-256 | The 11 files of section 5.3 (training metadata and player splits, test-game index, move sequences, account labels, game graph, old-pool game bundles) |
| Board replay (view90 features) | `bigbank90_20261114_v1.py`, `bigbank2_90_20261114_v1.py`, `bigbank3_90_20261114_v1.py`, `bigbank4_90_20261115_v1.py`, `bigbank5_90_20261115_v1.py`: `prepare` → `package` (packs the worker payload) → replay with the payload's `worker.py` on the GPU → `accept`; official games use `complete90_pilot_20261112_v1.py` | `runs/bigbank*/`, `runs/complete90_pilot_20261112_v1/official_results/` |
| Per-move features (Human SL tokens) | GPU host: `python scripts/tokfull_vm_20261116_v1.py <stream> <payload dir> ...`; local: `python scripts/tokpilot_20261116_v1.py <payload dir> <name>`; extra extraction `bigbank6_tok_20261117_v1.py prepare`; network input cache `tokcache_20261118_v1.py <payload> <games.jsonl> <out>` | `runs/tokfull_20261116/<bank>/results/` (tok_z etc.), `runs/tokpilot_20261116_v1/official_merged/` |
| Game bundles and capture events for the move-sequence networks | The game bundles come from the rebuild script above (`moves.npz`, `moves_ext.npz`, `bundle_g1.npz`); capture events: `python scripts/seqcap_20261115_v1.py build ...` (arguments described at the top of the script) | `runs/seqid_20261115_v1/` |

### 3.2 Training
| Model | Command (arguments in `configs/training_args.json`) | Output |
|---|---|---|
| 7 move-sequence networks | `python scripts/seqid_20261115_v1.py train <split> <fold> ...` (`full` for the all-player versions; the core is in `seqid_core_20261115_v2.py`/`v4.py`) | `runs/seqid_20261115_v1/<name>/.../model.pt` |
| Per-move models | `python scripts/tokhead_vm_20261116_v1.py <out> <labels.npz> <official dir> <token dirs...>`; late-game version `tokhead_vm_20261118_v5.py` | `model.pt`, `off_emb.npy` |
| Trunk fine-tuning (head frozen) | `python scripts/tokft_e2e_20261118_v2.py <out> <labels> <payload> <HumanSL.pt> <official cache> <order.npy> <caches...> --init-head ... --blocks 3 --lr-trunk 5e-5 --lr-head 0 --K 48 --Kn 48` | `trunk_ft.pt` |
| External-account heads and cross-readout | `tokhead_vm_20261118_v5.py` (after merging the external labels); `tokft_e2e_20261118_v3.py --init-trunk <trunk_ft.pt> --eval-only` | `xf*/model.pt`, `xread_*/` embeddings |
| BatchNorm re-estimation | `python scripts/tokbn_recal_20261117_v1.py <model dir> <out> off <official dir> <token dirs...>` | `*_bnoff/off_emb.npy` |
| Board-summary heads and prototype heads | `bigbank3_90_20261114_v1.py`, `mlphead_export_20261114_v2.py`, `protohead_20261115_v1.py run <variant>`; embedding cache `suppsplit_scoring_20261115_v1.py cache` | `runs/bigbank3_90_20261114_v1/export/models_{d,lin}/`, `runs/suppsplit_20261115_v1/heademb_b3.npz` |

### 3.3 Inference and submission generation
| Step | Command | Output |
|---|---|---|
| Pairwise fusion (new pool) | `python scripts/listfuse_20261115_v1.py official m39` | `runs/listfuse_20261115_v1/m39/official_{public2,private2}.npy` |
| Rematch evidence, MaxSim | `python scripts/tokopp_20261117_v1.py official runs/tokopp_20261117_v1/official_opp_tokens o5 runs/tokhead_20261116_v1/vm_t1 runs/tokhead_20261116_v1/full_s1 runs/tokhead_20261116_v1/full_big_s3 runs/tokhead_20261116_v1/vm_t6 runs/tokhead_20261116_v1/full6a_s12`; `python scripts/tokmaxsim_20261117_v1.py official runs/tokhead_20261116_v1/full_mv_s17 mv17` | `runs/tokopp_20261117_v1/o5/`, `runs/tokmaxsim_20261117_v1/mv17/` |
| First export (decides the lending) | `python scripts/recipe_export_20261121_v4.py export <name> "V('pair_m39')+.5*np.tanh(V('ro_o5')/5)+.3*V('ms_mv17')+.1*V('hgm_top30')" --temp-mult 0.7` | `runs/recipe_export_20261115_v1/out_<name>/` |
| Gallery lending | `python scripts/transgal_official_20261119_v1.py m39 m39htg <export folder from the previous step> 0.8` | `runs/listfuse_20261115_v1/m39htg/` |
| Final recipe export | `python scripts/recipe_export_20261122_v6.py export 20261122_v9qt0_listfuse_m39htg_og5_rematch_maxsim_sinkhorn_v1 "V('pair_m39htg')+.5*np.tanh(V('ro_o5')/5)+.3*V('ms_mv17')+.1*V('hgm_top30')+.1*V('og5q1_cos')" --temp-mult 0.7` (hubness and opponent community are computed at export time) | `<pool>_score.npy`, `RESULT.json` |
| New-pool ranking | `python scripts/nosink_20261119_v2.py <base submission dir> <export dir> <out name>`: single-game questions are ranked directly; multi-game questions get a capped Sinkhorn among themselves | New pool, 800 rows |
| Old pool | `python scripts/oldpool_upgrade_20261123_v1.py`: the 7 move-sequence networks embed the old pool, fused with the H scores, then Sinkhorn (T = 0.1) + Hungarian method | Old pool, 800 rows |

## 4. Inputs and outputs of the main modules

| Module | Input | Output |
|---|---|---|
| `audit_training.py` | Official training CSVs | `audit.sqlite` (per-game parse results, hashes, family keys) |
| `bigbank*_90_*.py`, `complete90_pilot_20261112_v1.py` | Training/test SGFs, Human SL weights | Board features at up to 90 decision points per game (view90) |
| `tokfull_vm_20261116_v1.py`, `tokpilot_20261116_v1.py` | Same as above | 1,162-dimensional Human SL features per decision point (memmaps such as tok_z, tok_s) |
| `seqid_20261115_v1.py` | Move sequences (first 160 moves of both players) + capture events | Move-sequence network weights; a 384-dimensional embedding per game (`off_own.npy`) |
| `tokhead_vm_20261116_v1.py` | Per-move features, player labels | Per-move model weights; per-game embeddings (`off_emb.npy`) |
| `listfuse_20261115_v1.py` | Per-game embeddings of each model (`*.npz`), official gallery | Pairwise-fusion scores of each question against the 600 candidates |
| `tokopp_20261117_v1.py`/`tokmaxsim_20261117_v1.py` | Opponent-side per-move features / decision-point vectors | Rematch evidence and MaxSim score matrices |
| `hubgal_20261121_v1.py`, `oppgraph_20261122_v2.py` | Gallery embeddings; game graph of the training bank | Hubness correction, opponent-community similarity (computed at export time) |
| `recipe_export_20261122_v6.py` | The score matrices above and the recipe | Final scores S (`<pool>_score.npy`) and temperature |
| `nosink_20261119_v2.py`, `assign_sinkhorn_20261113_v1.py` | Score matrices, number of games per question, temperature | Top-5 candidates for each question |
| `oldpool_upgrade_20261123_v1.py` | Old-pool SGFs, the 7 move-sequence networks, H scores | Old pool, 800 rows |
| `rebuild_official_derived_20261126_v1.py` | Official training/test data, `audit.sqlite`, the selection index pack, the `b1_full` weights | The 11 files of section 5.3 and their SHA-256 check |
| `report_repro_20261125_v1.py` | Weights and intermediates | The rebuilt submission and its SHA-256 check |

## 5. Weights and intermediates

The weights are in this repo's GitHub Releases (not in git). After downloading, extract them into `comp1/`; the paths are the relative paths in the tables below. The names, sizes, and SHA-256 of the 106 files in the Release (3.72 GiB) are listed in `weights_manifest.tsv` (`in_release` column `yes`); another 11 files that re-encode official data are not in the Release (`in_release` column `no`, see section 5.3).

The assets `comp1_weights.tar.part00`–`part02` of Release `v1.0` are three parts of a single tar (the SHA-256 of the assets themselves are in `SHA256SUMS.txt`). Download them into the repo root and run:

```bash
# Linux / macOS
cat comp1_weights.tar.part0* > comp1_weights.tar
sha256sum -c SHA256SUMS.txt --ignore-missing
tar -xf comp1_weights.tar -C comp1/
```

```bat
:: Windows (cmd)
copy /b comp1_weights.tar.part00+comp1_weights.tar.part01+comp1_weights.tar.part02 comp1_weights.tar
tar -xf comp1_weights.tar -C comp1
```

After extraction, `comp1/` gains folders such as `runs/` and the file `MANIFEST.csv`; the SHA-256 of each file can be checked against `weights_manifest.tsv`.

### 5.1 Model weights (60 files, 2.45 GiB)

| File (relative to comp1/) | MB | SHA-256 |
|---|---:|---|
| `runs/bigbank3_90_20261114_v1/export/models_d/net_s0.pt` | 30.1 | `edad19b5ecb3d07f7002576e81c2ad679ad773ec76e120e2a9aadda8fd2158b8` |
| `runs/bigbank3_90_20261114_v1/export/models_d/net_s1.pt` | 30.1 | `dce8b4b6cb560b9e6fd43fd6b43f1ef9a3af661575baabc9d8c536a341abe1e8` |
| `runs/bigbank3_90_20261114_v1/export/models_d/net_s2.pt` | 30.1 | `ca6ff3f452c8226bf164574d25103a12048c15d888c04632869c2bea7171525a` |
| `runs/bigbank3_90_20261114_v1/export/models_d/net_s3.pt` | 30.1 | `7589d9f2b776d58fc672786d9e62c83d207a9bc6804e36efe8d42310ad3e6402` |
| `runs/bigbank3_90_20261114_v1/export/models_d/net_s4.pt` | 30.1 | `1e566c7266d7994d022079570edc18e8ec52441ffc0c31f76527db1e45e1792a` |
| `runs/bigbank3_90_20261114_v1/export/models_d/preprocess.npz` | 0.0 | `9de192f0b5ca1afea55f4feb435bbf3087addbad44d537329a04219f0ca6d489` |
| `runs/bigbank3_90_20261114_v1/export/models_lin/net_s0.pt` | 12.4 | `4d71319b78649e0a93503927c3c15463ff22b9cc2950e4c49c9fc3188631a2e4` |
| `runs/bigbank3_90_20261114_v1/export/models_lin/net_s1.pt` | 12.4 | `2bd08b92f857efb837d87cb2d5ca70a1bd59ff256fa1809cd8dc5d6de8d4d762` |
| `runs/bigbank3_90_20261114_v1/export/models_lin/net_s2.pt` | 12.4 | `6db362ac76fee5886537232dc3cd1076e48ecedbf2491137a678deadc4441b5c` |
| `runs/bigbank3_90_20261114_v1/export/models_lin/net_s3.pt` | 12.4 | `b66c0142809d542325546b6eb6ec484e7409a4e73fdc3e857c24e4edb88af577` |
| `runs/bigbank3_90_20261114_v1/export/models_lin/net_s4.pt` | 12.4 | `270f50e40a3d356ceebabca9af1133cc2882b22dd9e45afa7afd72515aec013f` |
| `runs/bigbank3_90_20261114_v1/export/models_lin/preprocess.npz` | 0.0 | `9de192f0b5ca1afea55f4feb435bbf3087addbad44d537329a04219f0ca6d489` |
| `runs/fisher_policy_pilot_20261105_v2/assets/H_PLDA.npz` | 2.3 | `4fdea1c0688b7c0a2859e17fd19fd925e85c66e2c2f763b0a56dd84ca7e1eb2e` |
| `runs/fisher_policy_pilot_20261105_v2/assets/H_preprocess.npz` | 0.0 | `b511ddb083f87ec63678c5edc618675fc9ee45bc9ff3158c900d2aab49e82184` |
| `runs/fisher_policy_pilot_20261105_v2/assets/HumanSL_state_only.pt` | 205.8 | `37f2280111fe03da49715cf401d2fc70a8c22bbe7cb78305840daf71c1820dbb` |
| `runs/fisher_policy_pilot_20261105_v2/assets/KataGo_LICENSE` | 0.0 | `15a978650e81d1d32161bd39e453cfd2d423cee724cc7ab6aad0df6fec4722bf` |
| `runs/fisher_policy_pilot_20261105_v2/assets/MeanCount.pt` | 1.0 | `f2211ef8c4a414f9ba886810c3166dc0f78ed32d93cb81b440404991f6b5c390` |
| `runs/fisher_policy_pilot_20261105_v2/assets/NETWORK_LICENSE_source.html` | 0.0 | `590837548c66daca592955910aeeea36634daa49c722b1a80f6e24c0b714657e` |
| `runs/seqid_20261115_v1/a2/g1/f0/model.pt` | 28.9 | `d847e32669c0cf8b749c272c6de4699c130bc4336e4c4d5e8a66779be4a88ba7` |
| `runs/seqid_20261115_v1/a2/g1/f2/model.pt` | 28.9 | `31366ccd1157bb903b8bdc93f7363f86135a674417a7e73d4e49cb686853f43a` |
| `runs/seqid_20261115_v1/b1/g1/f0/model.pt` | 29.7 | `fdc2d1cbb47dac97890c44bb86fabfda389763a9056d0c4a054cc476ac70ff1f` |
| `runs/seqid_20261115_v1/b1_full/ffull/model.pt` | 30.7 | `f780f6317202dbef0f439a12efb669c15dab6d6568c407296c1f0d8264e0fa9e` |
| `runs/seqid_20261115_v1/b1aug_full/ffull/model.pt` | 30.7 | `e7ec1fb1e04e35c69abe9e0419559655af2876c307e2e01c569047e353b8fa8e` |
| `runs/seqid_20261115_v1/b1cap_full/ffull/model.pt` | 30.7 | `bc1a983e91b54735bcaac1b8ec8f5ae0adab52d6c069565988b64f91bb35f07a` |
| `runs/seqid_20261115_v1/b1s1_full/ffull/model.pt` | 30.7 | `50221f797e825092fd7686ad9b6202000edeb5e86b41f95c328e4d2e29cf9a38` |
| `runs/seqid_20261115_v1/b2_full/ffull/model.pt` | 69.1 | `05ec130e1ac1c0db3b3334bd6d88098bcae1420c3ed3ebac10598d47fa5d8d83` |
| `runs/tokfull_20261116/e2e_warm_h8x48a/trunk_ft.pt` (used by Competition 2) | 33.5 | `fc33fa4407265f62b0f7b2c320a72ae7232c11262e32dec13b4a7091180c49be` |
| `runs/tokfull_20261116/e2e_warm_long_h5e5/trunk_ft_step7500.pt` | 33.5 | `1b8db300ec4386f3c9f8a23b5ba7ed38b33b04596ca09bdbe188be5b1bc23d26` |
| `runs/tokfull_20261116/wl_h8x48_b3_trunk_ft.pt` | 16.9 | `981d4c25adb75282a1f763fbc59763240ca35b76ff6456b9cfbbcce8198dc88b` |
| `runs/tokfull_20261116/xf10_s23/model.pt` | 80.0 | `11ea610f88d78f28645f508e92e6fa4b081d968c1bc0c507ae5dad0c8b72f362` |
| `runs/tokfull_20261116/xf16_s24/model.pt` | 80.0 | `151c7ccee7b2ba73c8e095037c2a0f62ada0e5afa6db20c2a940dc9739039d00` |
| `runs/tokfull_20261116/xf16_s25/model.pt` | 80.0 | `ca91a0bbd4ea69f9461d88d629367117f3f0825ab58b5e9a3a15b3bf8e5e6407` |
| `runs/tokfull_20261116/xf16_s28/model.pt` | 80.0 | `2fcbb5b1627720a8bb4809f071afcfa2133527c08077f531b607bf06a43c0b4f` |
| `runs/tokfull_20261116/xf16_s29/model.pt` | 80.0 | `03f3815a14c879c090e7e7e37ce670587da1216d58cd73af0a3b263bc54c4c30` |
| `runs/tokfull_20261116/xf8a_s27/model.pt` | 174.1 | `5c70072b2142cf0e7ff77ec5be54dfa2c5a80fc4917c16014b666de6099c63be` |
| `runs/tokfull_20261116/xread_xf10_b3/model.pt` | 80.0 | `26a59ad7b8143b0ec0620c6590a5377ab566f49e3063cd021e88fcf34b5f066a` |
| `runs/tokfull_20261116/xread_xf16_b3/model.pt` | 80.0 | `48dc6ae59948faeace498caf5a3f5c32a97602115308bcf7526a514001e5a7c8` |
| `runs/tokfull_20261116/xread_xf16_s25_b3/model.pt` | 80.0 | `d16526403758596e4f86b4b0f9b8bcc086be26acbee9c52b3f68780433f9ccd2` |
| `runs/tokfull_20261116/xread_xf16_s28_b3/model.pt` | 80.0 | `cb5572218b9bd48582bc15fbab7daaa89f1d35b439120273f37474b90e994a26` |
| `runs/tokfull_20261116/xread_xf16_s29_b3/model.pt` | 80.0 | `4f1c2cd81302365cf43373dce3bb8dbd7f3f6b9b71d993e2277bd19e67ffe0b6` |
| `runs/tokfull_20261116/xread_xf8a_s27_b3/model.pt` | 174.1 | `128b78fe1162d6fe6125fa05b9aca85f015aab13268f567fa9ccca04f8579ec9` |
| `runs/tokhead_20261116_v1/full6a_s12/model.pt` | 32.3 | `d22e507340334a7912eec580cc6014171f5972bf8498f5a2517af8465d0104d6` |
| `runs/tokhead_20261116_v1/full_big_s3/model.pt` | 52.0 | `4fd20a0108f612de416f5d3e237a2db85f415c88debf3082fac39bef7b4e4f6c` |
| `runs/tokhead_20261116_v1/full_late_s24/model.pt` | 32.9 | `77a65ae99927b1aefa170d7b1b6d16fda130feb97109b53de860d6fdae1e9af4` |
| `runs/tokhead_20261116_v1/full_latedrop_s26/model.pt` | 32.9 | `4aa2ab519eebdda0aa0f1992d9655f3a32766a45d25054d5dec0a895b81c1560` |
| `runs/tokhead_20261116_v1/full_mv_s17/model.pt` | 32.9 | `6948d0a7acab1b20662b742d75cc74592b0392d9053626a2a0154838b750c420` |
| `runs/tokhead_20261116_v1/full_s1/model.pt` | 32.3 | `ab8b66f8fb2980acbca17b3ebffd52b83160ddb1fc344e502002304202ff37da` |
| `runs/tokhead_20261116_v1/p2/preprocess.npz` | 0.0 | `a9789ef3fab0c7a4596ad4a6c29294421f8c08886b41ea6da65047810b042685` |
| `runs/tokhead_20261116_v1/vm_t1/model.pt` | 32.3 | `ec706d2c65e6ee5c3cc1dbe7ad0baa56e451d10661b5206748bda5f103963899` |
| `runs/tokhead_20261116_v1/vm_t10/model.pt` (used by Competition 2) | 32.3 | `aa0a7d1677363bbc8cfd9bbefada25474c5c8c763ccde102ab2000bbe76ec356` |
| `runs/tokhead_20261116_v1/vm_t17late/model.pt` | 32.9 | `43de21551b9abe062c6e95d977bd1d865c61c9171c7aa2b4226e5c6aa506eefd` |
| `runs/tokhead_20261116_v1/vm_t2/model.pt` | 32.3 | `09b081fd6aa016506a53004c5d2d3471ced3c14187c2ca51362bda1a77cca930` |
| `runs/tokhead_20261116_v1/vm_t4/model.pt` (used by Competition 2) | 52.0 | `51a42b2aed54aac715dd53740038ce0f81f57837bf85bcbda59c58db1d87eda6` |
| `runs/tokhead_20261116_v1/vm_t6/model.pt` | 32.3 | `95d673ea99e7cd1647b707efe0201b1107466fccc8c9f7b89963614f6bb0b73b` |
| `runs/tokhead_20261116_v1/wl_h24_b3/model.pt` | 32.9 | `b178439369ceb2ac92cda338c7442853341fb4ee8ae16a806dd282fabf63e4cf` |
| `runs/tokhead_20261116_v1/wl_h24_b3/trunk_ft.pt` | 16.9 | `e7c246d3a9d5b6f0e987b4f54515010266c2c49999957999bbb0df52c5b19f73` |
| `runs/tokhead_20261116_v1/wl_h8x48_b3/model.pt` | 32.9 | `ab46cb373b832e93bedae252c1a953905d56b2e99553371d35ce1e28461b1c28` |
| `runs/tokhead_20261116_v1/wl_h8x48_b3/trunk_ft.pt` | 16.9 | `981d4c25adb75282a1f763fbc59763240ca35b76ff6456b9cfbbcce8198dc88b` |
| `runs/tokhead_20261116_v1/wl_hd26_b3/model.pt` | 32.9 | `d0d7aa3d3e1f5f83ee2bb60639fe04a3527c8d75e291da4cd8b9ae46f508b6d8` |
| `runs/tokhead_20261116_v1/wl_hd26_b3/trunk_ft.pt` | 16.9 | `dd43485a160f21e3ae0a4923810c742169cb3ab589a668714a4656c2886e52f9` |

The 3 weights marked "used by Competition 2" are models trained for Competition 1. The final Competition 1 submission does not use them, but the Competition 2 pipeline loads them (see `comp2/README.md`), so they are included in the Competition 1 Release.

### 5.2 Intermediates needed for reproduction (46 files, 1.26 GiB)

Summarized by folder; per-file SHA-256 are in `weights_manifest.tsv`.

| Folder | Files | MB | Purpose |
|---|---:|---:|---|
| `runs/complete90_pilot_20261112_v1/official_results/` | 1 | 0.1 | Official game index |
| `runs/fisher_policy_pilot_20261105_v2/public_inputs/` | 1 | 1.2 | Old-pool H scores (Public) |
| `runs/listfuse_20261115_v1/m39htg/` | 2 | 1.8 | Pairwise-fusion scores after lending |
| `runs/oldpool_upgrade_20261123_v1/` | 16 | 463.9 | Per-game embeddings of the 7 move-sequence networks on the old pool and the fused matrices (game bundles: see section 5.3) |
| `runs/recipe_export_20261115_v1/out_20261122_v9qt0_listfuse_m39htg_og5_rematch_maxsim_sinkhorn_v1/` | 3 | 3.7 | Final new-pool scores and temperature |
| `runs/selection_20261126_v1/` | 9 | 4.3 | Selection index pack (for the rebuild in section 5.3) |
| `runs/seqid_20261115_v1/b1_full/ffull/` | 1 | 20.8 | Move-sequence network embeddings of the official games |
| `runs/seqid_20261115_v1/b1cap_full/ffull/` | 1 | 20.8 | Move-sequence network embeddings of the official games |
| `runs/seqid_20261115_v1/b1s1_full/ffull/` | 1 | 20.8 | Move-sequence network embeddings of the official games |
| `runs/seqid_20261115_v1/b2_full/ffull/` | 1 | 20.8 | Move-sequence network embeddings of the official games |
| `runs/suppsplit_20261115_v1/` | 1 | 277.4 | Per-game embeddings of the board-summary heads |
| `runs/tokhead_20261116_v1/p2/` | 1 | 166.4 | Per-game embeddings of the per-move model groups (including p2) |
| `runs/tokhead_20261116_v1/r013/` | 1 | 83.2 | Per-game embeddings of the per-move model groups (including p2) |
| `runs/tokhead_20261116_v1/tuned7e/` | 1 | 194.1 | Per-game embeddings of the per-move model groups (including p2) |
| `runs/tokmaxsim_20261117_v1/mv17/` | 2 | 3.7 | MaxSim scores |
| `runs/tokopp_20261117_v1/o5/` | 3 | 10.6 | Rematch evidence and opponent posteriors |
| `runs/trainval_refinement_20261105_v1/release/` | 1 | 1.2 | Old-pool H scores (Private) |

- `HumanSL_state_only.pt` is the KataGo Human SL network b18c384nbt-humanv0 (KataGo v1.15.0 release) converted to a PyTorch state dict. It is licensed under the KataGo Neural Network License; the full license text is in the same folder.
- In 5 of the external-account head files `runs/tokfull_20261116/xf*/model.pt`, the host temporary path in the saved arguments (args) was replaced with `<ssd_stage>/` before release; the tensors are bit-identical to the originals, and the SHA-256 of both the original and the released files are listed in `weights_manifest.tsv`.
- The `RESULT.json` of the final new-pool export keeps only two items: the recipe and the temperature.
- The selection index pack `runs/selection_20261126_v1/` contains only integer indices and hashes, with no game content or account names:
  - the game indices that bigbank3/4/5 selected into the training banks;
  - the fold labels of bigbank90;
  - the question-game indices of the three folds of cross-validation g1;
  - the account hashes of the 400 sealed players.

  These selections depend on which games had been replayed earlier during the research and cannot be derived from the official data alone, so they are included as is.

### 5.3 Files not in the Release that must be rebuilt from official data (11 files, 0.85 GiB)

The files below are intermediates that re-encode official data. They contain:
- move sequences of training and test games;
- training metadata and player splits;
- account labels;
- the test-game index;
- the game graph of the training bank;
- game bundles of the old pool (the old Public/Private questions), with move sequences and capture events.

These files are not in the Release; generate them from the official data with the rebuild steps:
1. Place the official data as described in section 2.
2. Build `audit.sqlite` as described in section 3.1 (audit and near-duplicate grouping).
3. Run `python scripts/rebuild_official_derived_20261126_v1.py all`.

At the end, the script checks the SHA-256 in the table below one by one; in a clean environment all 11 files were bit-identical to the originals. If you only need the old-pool game bundles, run `python scripts/rebuild_official_derived_20261126_v1.py oldpool` on its own (the game content is read only from the official test CSVs).

| File | Bytes | Expected SHA-256 | Rebuild stage |
|---|---:|---|---|
| `runs/research_v4/official_aligned_v1_20261104/scope_metadata.npz` | 201,199,232 | `16e288236eb054214e65507753b9bb195b3165823db8a4422faa217144bb4a60` | `fitrows`, `splits`, `scope` |
| `runs/v2_baseline_20261111_v2/official_stat/keys.json` | 994,003 | `fb8a2eda6d2aca1ed32f3f13f41deb06223a01c59976ddd7056c83ca62d2caed` | `keys` |
| `runs/bigbank4_90_20261115_v1/EXTMETA.npz` | 85,376,780 | `3257dac140781b0ec50a76a3fab9015ec92d42ae8bb5980fdbeb79a10b06b84a` | `extmeta4` |
| `runs/bigbank5_90_20261115_v1/EXTMETA.npz` | 54,768,995 | `c04814f3fbbc22a40dd90d09783fc7ccc62c0ebb2c0840adfb5576029d9800fd` | `extmeta5` |
| `runs/seqid_20261115_v1/moves.npz` | 237,083,212 | `0f38299aaf3adf889771b0594164f35299e804348f253819d94ed57dc30f9229` | `moves` |
| `runs/seqid_20261115_v1/moves_ext.npz` | 101,194,433 | `6257aa33263ce6d62b042fdbad7e4c69c09da2153499d86d372c9086311f63cc` | `moves_ext` |
| `runs/seqid_20261115_v1/bundle_g1.npz` | 150,452,617 | `156a9b93bc6c5e58f479e74ec7fc29bbd1fc327cf4c5cf4a36e9e366bf468eba` | `bundle` |
| `runs/tokpilot_20261116_v1/labels_all.npz` | 5,487,913 | `acae73fada98989222bfda328038fb2059cca8643066417354b84d3807874d52` | `labels` |
| `runs/oppbank_20261120_v2/full.npz` | 10,296,112 | `e4da9e1650c188ea07b6d84e3c3f611df4e1609e1385b91d96c2d31914a984d3` | `oppbank` (GPU; identical on RTX 3060 in testing) |
| `runs/oldpool_upgrade_20261123_v1/public_moves_cap.npz` | 40,907,400 | `76bb52ed44c6ff3e1ea42cf92c7dfc2d630b5961de75effcf10fdafa63b730d9` | `oldpool` (GPU) |
| `runs/oldpool_upgrade_20261123_v1/private_moves_cap.npz` | 27,046,924 | `99dd07978ee8fe1f4a186bacae6246c8af2b467bd777f7d47ba19e18fcd6c113` | `oldpool` (GPU) |

Where each stage comes from:
- The training split uses the hash rule of `goid_v4.pilot_scope_v1.fit_player` (fit/validation/holdout = 60/25/15%), plus the 400 sealed players in the selection index pack.
- The other stages call the functions of the original pipeline directly:
  - `official_aligned_v1.capacity`
  - `extmeta_build` of `bigbank4/5`
  - `seqid_20261115_v1.moves_meta`/`moves_official`/`data2`
  - `seqid_20261115_v2.bundle`
  - `oldpool_upgrade_20261123_v1.pack_pool`
- The `labels` and `oppbank` stages were written as functions from the original generating command and the first part of `oppbank_tcond_20261120_v1.py`.

## 6. Reproducing the final submission

Expected SHA-256: `b3df6a05ddcc5614047691b44c0969fb60645b5aeab165f9c8d70de804fe82ad` (1,600 rows).

L1 needs only the Release files. L2 and L3 (old pool) also need the files of section 5.3: L2 uses the first 9, L3 uses the two old-pool game bundles. First place the official data as described in section 2, then run:

```bash
cd comp1
python scripts/audit_training.py index
python scripts/audit_training.py parse
python scripts/audit_training.py parse --selection all --workers 4 --batch-size 128
python scripts/audit_training.py summary
python scripts/audit_near_duplicates.py --output data/processed
python scripts/apply_family_groups.py
python scripts/rebuild_official_derived_20261126_v1.py all   # rebuilds the 11 files of section 5.3 and checks their SHA-256; the last two steps need a GPU
```

Reproduction steps:

```bash
cd comp1
python scripts/report_repro_20261125_v1.py check   # L1, L2: about 1 minute, CPU is enough
python scripts/report_repro_20261125_v1.py emb     # L3 (old pool): about 5 minutes, needs a GPU; run the rebuild above (oldpool stage) first
```

| Level | Rebuilt from | Result (tested in a clean environment) |
|---|---|---|
| L1 | Saved final score matrices → ranking and assignment → CSV | SHA-256 identical to the final file |
| L2 | Per-component score matrices and per-game embeddings, plus the 9 files rebuilt from official data → fusion (hubness and opponent community computed on the fly) → ranking and assignment → CSV | SHA-256 identical to the final file |
| L3 (old pool) | Old-pool game bundles rebuilt from official data → re-embed the 45,000×2 old-pool games with the 7 move-sequence network weights → fusion → assignment | The 800 old-pool rows are identical to the final file |

Outputs are in `runs/report_repro_20261125_v1/`: `L1_from_saved_final_scores.csv`, `L2_from_components.csv`, `CHECK.json`, `L3o.json`.

## 7. Hardware and time estimates
| Task | Hardware | Time |
|---|---|---|
| Audit the 1 million games (`audit_training.py`, 4 workers) | CPU | About 36 minutes |
| Near-duplicate grouping | CPU | About 12 minutes |
| Rebuild the 11 files of section 5.3 | CPU, RTX 3060 for the last two steps | About 10 minutes |
| Reproduce L1 + L2 | CPU | About 1 minute |
| Reproduce L3 (old pool) | RTX 3060 | About 5 minutes |
| Logged model training (33 models) | RTX 4090 and RTX 3060 | About 90 GPU hours in total |
| Redo everything from scratch (including feature extraction and caches) | Mainly RTX 4090 | About 190–240 GPU hours (estimate), about 450 GB of disk for caches |

## 8. Known limitations
- **L3 for the new pool (recomputing everything from the weights) is not implemented as a one-step pipeline and has not been verified.** It would require re-extracting per-move features for the official questions (about 3.1 GB each for the official games and the opponent side, not included), re-embedding with each model, and rerunning the pairwise-fusion fit and the lending. For now it is reproduced at L2 (per-component score matrices and per-game embeddings).
- **No weights were saved for p2 (the 40,000-game per-move model of the tkp group in the pairwise fusion)**; only its per-game embeddings `runs/tokhead_20261116_v1/p2/emb.npz` are included.
- **The H scores, weighted 25% in the old pool, come from an early pipeline** and are provided as intermediates (`runs/fisher_policy_pilot_20261105_v2/public_inputs/H.npy`, `runs/trainval_refinement_20261105_v1/release/Private_H_scores.npy`); the early code that produced H is not included.
- L3 for the Public part of the old pool: the original pipeline reused earlier embeddings for two of the models; after re-embedding, the fused matrix differs by at most 0.0013, and the rankings are identical to the final file.
- The 11 files of section 5.3 can be rebuilt bit-identically from the official data. However, "which games" each training bank selected is provided by the selection index pack rather than re-derived from the official data:
  - these selections accumulated round by round during the research, for example by preferring games that had already been replayed;
  - the 400 sealed players were frozen on October 25.
- The `oppbank` stage runs inference on the GPU in bfloat16; on an RTX 3060 (PyTorch 2.14.1+cu126) the result was identical to the original file. On other GPUs or versions, the values may differ in the last digits.
- `e2e_warm_long_h5e5` (the long-schedule trunk fine-tuning) left no LOG; its arguments were recovered from the launch command (see `configs/training_args.json`).
- Retraining from scratch is subject to GPU nondeterminism: the weights and the final file will not be bit-identical, and only similar scores can be expected.
- Code from the research that did not go into the final file is not included; some of the included modules still contain early experimental functions, because the final pipeline imports the same files.

<p align="center"><a href="README.md"><img src="https://img.shields.io/badge/%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87-2a78d6?style=for-the-badge" alt="繁體中文"></a> <a href="README.en.md"><img src="https://img.shields.io/badge/English-6e7781?style=for-the-badge" alt="English"></a></p>

# 競賽一：從圍棋棋譜集辨識玩家（TEAM_11307）

← 回到 [repo 首頁](../README.md)

最終提交檔：Private 0.923441（第 4 名）。本資料夾是產生該檔的程式碼：資料處理、模型訓練、推論，以及重現檢查。
方法說明見報告；本文件說明如何安裝、放置資料與權重、依序執行，以及如何重現最終提交檔。

## 1. 環境安裝

| 項目 | 版本 |
|---|---|
| Python | 3.12（開發用 3.12.14） |
| PyTorch | 2.14.1+cu126（CUDA 12.6）；GPU 主機用 2.11.0+cu128 |
| 其他套件 | 見 `requirements.txt`（版本已釘死） |
| GPU | 推論與重現：NVIDIA GPU 12 GB 以上（開發用 RTX 3060 12 GB）；逐手特徵抽取與大型訓練用 RTX 4090 24 GB |
| 作業系統 | Windows 11 與 Ubuntu 24.04 都跑過（程式只用標準函式庫與上表套件） |

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

機器相關的設定（Human SL 權重位置、GPU 主機工作資料夾、啟動 worker 用的 Python、能串流 .7z 的 tar）寫在
`configs/paths.example.yaml`；複製成 `configs/paths.yaml` 後修改，或用環境變數 `AICUP_<KEY>` 覆蓋。其餘程式一律以 `comp1/` 為根目錄找輸入檔（`data/`、`runs/`、`submissions/`，這三個資料夾不進 git）。

逐手特徵的 worker（`payload/`）需要 KataGo 的 Python 模組與 sgfmill：執行前把 `../third_party/katago` 與 `../third_party/sgfmill` 複製到 `payload/vendor/katago`、`payload/vendor/sgfmill`（原檔未修改，`payload/MANIFEST.json` 會核對 SHA-256）。

## 2. 資料下載與放置

| 資料 | 來源 | 放置位置 |
|---|---|---|
| 官方訓練資料 v1.0（10 個 CSV，1,000,000 局） | 主辦單位：https://github.com/AILAB-NDHU/aicup2026-go-dataset （release v1.0） | `data/raw/training/` |
| 官方訓練資料 v1.1（同上，SGF 開頭多了讓子數 HD） | 同上（release v1.1） | `data/raw/training_v1_1_20261104/training/` |
| 官方測試資料 v2（11 月 11 日版） | 主辦單位（AIdea 競賽頁） | 解壓到 `data/raw/official_tests_20261111_v2/extracted/tests/`（需含 `player_identification_candidates2_*.csv`、`player_identification_test2_*.csv`、`player_identification_candidates_*.csv`、`player_identification_test_*.csv`、`sample_submission_player_v2.csv`） |
| 野狐圍棋棋譜（外部資料，GPL-3.0） | https://github.com/featurecat/go-dataset | `D` 槽或任意位置，執行 `extfox_*.py` 時以參數指定 |

本 repo 不附任何原始資料、外部棋譜或其衍生的特徵。

### 野狐棋譜的取得與篩選
1. 下載下列 13 個壓縮檔（分卷 `.7z.001`、`.002` 依序串接成單一 `.7z`）。`Pro.7z`、`8d.7z` 只用來檢查格式，沒有進入訓練。

| 壓縮檔 | 位元組 | SHA-256 | 用途 |
|---|---:|---|---|
| `1d2.7z` | 168,279,345 | `2533a3582f868551ed246cbfeb9c056aca3d6e82b35d4b0b3070f6c795c0c7c5` | 帳號選取 |
| `1k2.7z` | 119,031,002 | `d9d6d33ad9cec22037ef4136097b8bc32ca40a61c6758b3facbb78ecf5a4e641` | 帳號選取 |
| `2d2.7z` | 215,462,071 | `2f696e4604153d1bec7626d595790359545fe741ef275ae0686a8dad962e8ca6` | 帳號選取 |
| `2k2.7z` | 87,611,384 | `763c244fb700fee4715c3dc583a3a16883617581a6c29fdab33788fe15564748` | 帳號選取 |
| `3d2.7z` | 255,343,139 | `ef73a31fd8912baaffb45b42120b6de96d337f297c2b82845ee0459352ba76da` | 帳號選取 |
| `3k2.7z` | 57,840,412 | `09f232cdbac8746c7ec2cfbbbeab9e9bb9b35397275d11657b2f62521d11cde5` | 帳號選取 |
| `4d2.7z` | 171,211,054 | `64d0e380034735dfbff1d9ecd08a393ad3545801b213ffd79fab24737c511fef` | 帳號選取 |
| `5d2.7z` | 113,412,556 | `dd214c5700cfbe8f58c8338b032efdd0bfec313a7028ffda0db1b8049cad42d3` | 帳號選取 |
| `5k2.7z` | 46,585,984 | `efa73a40aee2db76fdb7b1fe71e3ce6a67ea5ad3c815e0448e9bb5dbe698885c` | 帳號選取 |
| `6d2.7z` | 48,503,393 | `c2b1ce24a9a6110017a49cf8a6cc0a17b3f59c1d81fe6498df41d86668feee21` | 帳號選取 |
| `7d2.7z` | 19,521,177 | `96fc1fbdd0f46d33cf1923f5de2e2201697bd061c70ca4833a582180dff18ffa` | 帳號選取 |
| `8d.7z` | 14,915,255 | `b212205339349e5c2ad7b6bbd2984cc18968322c8f6c690ffd20788782ac9f96` | 只做格式檢查 |
| `Pro.7z` | 3,657,149 | `4b87b49adac5626d78ef48e05a122341b262d229e713d974338d573bff0ec735` | 只做格式檢查 |

2. 掃描（串流讀取 7z，不展開成小檔）：`python scripts/extfox_20261118_v1.py scan <archive.7z> <meta.npz>`，每個分卷一次。
   篩選條件：19 路、無擺子、讓子數 HA ≤ 1、至少 60 手、無虛手、雙方都有帳號。帳號名稱的原始位元組直接進加鹽 SHA-256，只保留前 8 位元組，從不解碼或儲存。外部棋局不與任何競賽棋局比對。
3. 選帳號：`python scripts/extfox_20261118_v1.py select <out dir> <meta.npz> ... --games-per 16 --accounts 24000`（第一批，24,000 個帳號 × 16 局）；第二批 48,000 個不重複帳號 × 8 局用 `scripts/extfox_20261119_v2.py`。
4. 匯出成既有管線的 `games.jsonl`：`python scripts/extfox_20261118_v1.py export <out dir> <archive.7z> ...`，再以與競賽棋局相同的凍結 Human SL 主幹抽取逐手特徵，最後用 `scripts/extfox_merge_20261118_v1.py` 併入逐手模型的標籤（外部帳號只當額外分類類別）。

## 3. 執行順序

以下指令都在 `comp1/` 執行。訓練參數（每個模型實際使用的參數、類別數、局數、秒數）完整列在 `configs/training_args.json`。

### 3.1 資料處理
| 步驟 | 指令 | 輸出 |
|---|---|---|
| 解析與稽核 100 萬局 | `python scripts/audit_training.py index`，`... parse`（先跑預設的 calibration 1 萬局），`... parse --selection all --workers 4 --batch-size 128`，`... summary` | `data/processed/audit.sqlite`、特徵矩陣、稽核摘要 |
| 近重複分組 | `python scripts/audit_near_duplicates.py --output data/processed`，再 `python scripts/apply_family_groups.py` | `near_duplicate_groups.jsonl`；`audit.sqlite` 的家族鍵（切分時不拆開） |
| 由官方資料重建中間產物（不隨 Release 提供） | `python scripts/rebuild_official_derived_20261126_v1.py all`：依序 `fitrows`、`splits`、`scope`、`keys`、`extmeta4`、`extmeta5`、`moves`、`moves_ext`、`bundle`、`labels`、`oppbank`、`oldpool`（最後兩步需要 GPU），結尾逐一核對 SHA-256 | 5.3 節的 11 個檔案（訓練中繼資料與玩家切分、測試棋局索引、落子序列、帳號標籤、對局圖、舊題池棋譜包） |
| 盤面重播（view90 特徵） | `bigbank90_20261114_v1.py`、`bigbank2_90_20261114_v1.py`、`bigbank3_90_20261114_v1.py`、`bigbank4_90_20261115_v1.py`、`bigbank5_90_20261115_v1.py`：`prepare` → `package`（打包 worker payload）→ 在 GPU 上執行 payload 的 `worker.py` 重播 → `accept`；官方棋局用 `complete90_pilot_20261112_v1.py` | `runs/bigbank*/`、`runs/complete90_pilot_20261112_v1/official_results/` |
| 逐手特徵（Human SL tokens） | GPU 主機：`python scripts/tokfull_vm_20261116_v1.py <stream> <payload dir> ...`；本機：`python scripts/tokpilot_20261116_v1.py <payload dir> <name>`；補抽 `bigbank6_tok_20261117_v1.py prepare`；網路輸入快取 `tokcache_20261118_v1.py <payload> <games.jsonl> <out>` | `runs/tokfull_20261116/<bank>/results/`（tok_z 等）、`runs/tokpilot_20261116_v1/official_merged/` |
| 手順網路的棋譜包與吃子事件 | 棋譜包由上面的重建腳本產生（`moves.npz`、`moves_ext.npz`、`bundle_g1.npz`）；吃子事件 `python scripts/seqcap_20261115_v1.py build ...`（參數見腳本開頭說明） | `runs/seqid_20261115_v1/` |

### 3.2 訓練
| 模型 | 指令（參數見 `configs/training_args.json`） | 輸出 |
|---|---|---|
| 手順網路 7 個 | `python scripts/seqid_20261115_v1.py train <split> <fold> ...`（全員版用 `full`；核心在 `seqid_core_20261115_v2.py`／`v4.py`） | `runs/seqid_20261115_v1/<name>/.../model.pt` |
| 逐手模型 | `python scripts/tokhead_vm_20261116_v1.py <out> <labels.npz> <official dir> <token dirs...>`；後段版 `tokhead_vm_20261118_v5.py` | `model.pt`、`off_emb.npy` |
| 主幹微調（頭凍結） | `python scripts/tokft_e2e_20261118_v2.py <out> <labels> <payload> <HumanSL.pt> <official cache> <order.npy> <caches...> --init-head ... --blocks 3 --lr-trunk 5e-5 --lr-head 0 --K 48 --Kn 48` | `trunk_ft.pt` |
| 外部帳號頭與交叉讀出 | `tokhead_vm_20261118_v5.py`（併入外部標籤後）；`tokft_e2e_20261118_v3.py --init-trunk <trunk_ft.pt> --eval-only` | `xf*/model.pt`、`xread_*/` 嵌入 |
| BatchNorm 重估 | `python scripts/tokbn_recal_20261117_v1.py <model dir> <out> off <official dir> <token dirs...>` | `*_bnoff/off_emb.npy` |
| 盤面彙總頭與原型頭 | `bigbank3_90_20261114_v1.py`、`mlphead_export_20261114_v2.py`、`protohead_20261115_v1.py run <variant>`；嵌入快取 `suppsplit_scoring_20261115_v1.py cache` | `runs/bigbank3_90_20261114_v1/export/models_{d,lin}/`、`runs/suppsplit_20261115_v1/heademb_b3.npz` |

### 3.3 推論與產生提交檔
| 步驟 | 指令 | 輸出 |
|---|---|---|
| 配對融合（新題池） | `python scripts/listfuse_20261115_v1.py official m39` | `runs/listfuse_20261115_v1/m39/official_{public2,private2}.npy` |
| 再戰證據、MaxSim | `python scripts/tokopp_20261117_v1.py official runs/tokopp_20261117_v1/official_opp_tokens o5 runs/tokhead_20261116_v1/vm_t1 runs/tokhead_20261116_v1/full_s1 runs/tokhead_20261116_v1/full_big_s3 runs/tokhead_20261116_v1/vm_t6 runs/tokhead_20261116_v1/full6a_s12`；`python scripts/tokmaxsim_20261117_v1.py official runs/tokhead_20261116_v1/full_mv_s17 mv17` | `runs/tokopp_20261117_v1/o5/`、`runs/tokmaxsim_20261117_v1/mv17/` |
| 第一次匯出（決定外借） | `python scripts/recipe_export_20261121_v4.py export <name> "V('pair_m39')+.5*np.tanh(V('ro_o5')/5)+.3*V('ms_mv17')+.1*V('hgm_top30')" --temp-mult 0.7` | `runs/recipe_export_20261115_v1/out_<name>/` |
| 畫廊外借 | `python scripts/transgal_official_20261119_v1.py m39 m39htg <上一步的匯出資料夾> 0.8` | `runs/listfuse_20261115_v1/m39htg/` |
| 最終配方匯出 | `python scripts/recipe_export_20261122_v6.py export 20261122_v9qt0_listfuse_m39htg_og5_rematch_maxsim_sinkhorn_v1 "V('pair_m39htg')+.5*np.tanh(V('ro_o5')/5)+.3*V('ms_mv17')+.1*V('hgm_top30')+.1*V('og5q1_cos')" --temp-mult 0.7`（樞紐度與對手社群在匯出時現算） | `<pool>_score.npy`、`RESULT.json` |
| 新題池排名 | `python scripts/nosink_20261119_v2.py <base submission dir> <export dir> <out name>`：單局題直接排序，多局題之間做有上限的 Sinkhorn | 新題池 800 列 |
| 舊題池 | `python scripts/oldpool_upgrade_20261123_v1.py`：7 個手順網路嵌入舊題池，與 H 分數融合後 Sinkhorn（T＝0.1）＋匈牙利法 | 舊題池 800 列 |

## 4. 重要模組的輸入與輸出

| 模組 | 輸入 | 輸出 |
|---|---|---|
| `audit_training.py` | 官方訓練 CSV | `audit.sqlite`（每局解析結果、雜湊、家族鍵） |
| `bigbank*_90_*.py`、`complete90_pilot_20261112_v1.py` | 訓練／測試 SGF，Human SL 權重 | 每局最多 90 個決策點的盤面特徵（view90） |
| `tokfull_vm_20261116_v1.py`、`tokpilot_20261116_v1.py` | 同上 | 每個決策點 1,162 維的 Human SL 特徵（tok_z、tok_s 等 memmap） |
| `seqid_20261115_v1.py` | 落子序列（雙方前 160 手）＋吃子事件 | 手順網路權重；每局 384 維嵌入（`off_own.npy`） |
| `tokhead_vm_20261116_v1.py` | 逐手特徵、玩家標籤 | 逐手模型權重；每局嵌入（`off_emb.npy`） |
| `listfuse_20261115_v1.py` | 各模型的每局嵌入（`*.npz`）、官方畫廊 | 每題對 600 位候選人的配對融合分數 |
| `tokopp_20261117_v1.py`／`tokmaxsim_20261117_v1.py` | 對手方逐手特徵／決策點向量 | 再戰證據、MaxSim 分數矩陣 |
| `hubgal_20261121_v1.py`、`oppgraph_20261122_v2.py` | 畫廊嵌入；訓練棋庫對局圖 | 樞紐度校正、對手社群相似度（匯出時現算） |
| `recipe_export_20261122_v6.py` | 上列分數矩陣與配方 | 最終分數 S（`<pool>_score.npy`）與溫度 |
| `nosink_20261119_v2.py`、`assign_sinkhorn_20261113_v1.py` | 分數矩陣、題目局數、溫度 | 每題前五名候選人 |
| `oldpool_upgrade_20261123_v1.py` | 舊題池 SGF、7 個手順網路、H 分數 | 舊題池 800 列 |
| `rebuild_official_derived_20261126_v1.py` | 官方訓練／測試資料、`audit.sqlite`、選取索引包、`b1_full` 權重 | 5.3 節的 11 個檔案與 SHA-256 核對 |
| `report_repro_20261125_v1.py` | 權重與中間產物 | 重建的提交檔與 SHA-256 核對 |

## 5. 權重與中間產物

權重放在本 repo 的 GitHub Releases（不進 git）。下載後解壓到 `comp1/`，路徑即下表的相對路徑。Release 內 106 個檔案（3.72 GiB）的名稱、大小與 SHA-256 列在 `weights_manifest.tsv`（`in_release` 欄為 `yes`）；另有 11 個由官方資料重新編碼的檔案不隨 Release 提供（`in_release` 欄為 `no`，見 5.3 節）。

Release `v1.0` 的附件 `comp1_weights.tar.part00`～`part02` 是同一個 tar 切成的三段（附件本身的 SHA-256 見 `SHA256SUMS.txt`）。在 repo 根目錄下載後執行：

```bash
# Linux / macOS
cat comp1_weights.tar.part0* > comp1_weights.tar
sha256sum -c SHA256SUMS.txt --ignore-missing
tar -xf comp1_weights.tar -C comp1/
```

```bat
:: Windows（cmd）
copy /b comp1_weights.tar.part00+comp1_weights.tar.part01+comp1_weights.tar.part02 comp1_weights.tar
tar -xf comp1_weights.tar -C comp1
```

解開後 `comp1/` 下會多出 `runs/` 等資料夾與 `MANIFEST.csv`，各檔的 SHA-256 可用 `weights_manifest.tsv` 核對。

### 5.1 模型權重（60 個，2.45 GiB）

| 檔案（相對於 comp1/） | MB | SHA-256 |
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
| `runs/tokfull_20261116/e2e_warm_h8x48a/trunk_ft.pt`（競賽二用） | 33.5 | `fc33fa4407265f62b0f7b2c320a72ae7232c11262e32dec13b4a7091180c49be` |
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
| `runs/tokhead_20261116_v1/vm_t10/model.pt`（競賽二用） | 32.3 | `aa0a7d1677363bbc8cfd9bbefada25474c5c8c763ccde102ab2000bbe76ec356` |
| `runs/tokhead_20261116_v1/vm_t17late/model.pt` | 32.9 | `43de21551b9abe062c6e95d977bd1d865c61c9171c7aa2b4226e5c6aa506eefd` |
| `runs/tokhead_20261116_v1/vm_t2/model.pt` | 32.3 | `09b081fd6aa016506a53004c5d2d3471ced3c14187c2ca51362bda1a77cca930` |
| `runs/tokhead_20261116_v1/vm_t4/model.pt`（競賽二用） | 52.0 | `51a42b2aed54aac715dd53740038ce0f81f57837bf85bcbda59c58db1d87eda6` |
| `runs/tokhead_20261116_v1/vm_t6/model.pt` | 32.3 | `95d673ea99e7cd1647b707efe0201b1107466fccc8c9f7b89963614f6bb0b73b` |
| `runs/tokhead_20261116_v1/wl_h24_b3/model.pt` | 32.9 | `b178439369ceb2ac92cda338c7442853341fb4ee8ae16a806dd282fabf63e4cf` |
| `runs/tokhead_20261116_v1/wl_h24_b3/trunk_ft.pt` | 16.9 | `e7c246d3a9d5b6f0e987b4f54515010266c2c49999957999bbb0df52c5b19f73` |
| `runs/tokhead_20261116_v1/wl_h8x48_b3/model.pt` | 32.9 | `ab46cb373b832e93bedae252c1a953905d56b2e99553371d35ce1e28461b1c28` |
| `runs/tokhead_20261116_v1/wl_h8x48_b3/trunk_ft.pt` | 16.9 | `981d4c25adb75282a1f763fbc59763240ca35b76ff6456b9cfbbcce8198dc88b` |
| `runs/tokhead_20261116_v1/wl_hd26_b3/model.pt` | 32.9 | `d0d7aa3d3e1f5f83ee2bb60639fe04a3527c8d75e291da4cd8b9ae46f508b6d8` |
| `runs/tokhead_20261116_v1/wl_hd26_b3/trunk_ft.pt` | 16.9 | `dd43485a160f21e3ae0a4923810c742169cb3ab589a668714a4656c2886e52f9` |

標「競賽二用」的 3 個權重是競賽一訓練的模型，競賽一的最終提交檔用不到，但競賽二的流程會載入（見 `comp2/README.md`），因此一併收在競賽一的 Release。

### 5.2 重現所需的中間產物（46 個，1.26 GiB）

依資料夾彙總，逐檔 SHA-256 見 `weights_manifest.tsv`。

| 資料夾 | 檔案數 | MB | 用途 |
|---|---:|---:|---|
| `runs/complete90_pilot_20261112_v1/official_results/` | 1 | 0.1 | 官方棋局索引 |
| `runs/fisher_policy_pilot_20261105_v2/public_inputs/` | 1 | 1.2 | 舊題池 H 分數（Public） |
| `runs/listfuse_20261115_v1/m39htg/` | 2 | 1.8 | 外借後的配對融合分數 |
| `runs/oldpool_upgrade_20261123_v1/` | 16 | 463.9 | 舊題池 7 個手順網路的每局嵌入、融合矩陣（棋譜包見 5.3 節） |
| `runs/recipe_export_20261115_v1/out_20261122_v9qt0_listfuse_m39htg_og5_rematch_maxsim_sinkhorn_v1/` | 3 | 3.7 | 新題池最終分數與溫度 |
| `runs/selection_20261126_v1/` | 9 | 4.3 | 選取索引包（供 5.3 節重建用） |
| `runs/seqid_20261115_v1/b1_full/ffull/` | 1 | 20.8 | 手順網路的官方棋局嵌入 |
| `runs/seqid_20261115_v1/b1cap_full/ffull/` | 1 | 20.8 | 手順網路的官方棋局嵌入 |
| `runs/seqid_20261115_v1/b1s1_full/ffull/` | 1 | 20.8 | 手順網路的官方棋局嵌入 |
| `runs/seqid_20261115_v1/b2_full/ffull/` | 1 | 20.8 | 手順網路的官方棋局嵌入 |
| `runs/suppsplit_20261115_v1/` | 1 | 277.4 | 盤面彙總頭的每局嵌入 |
| `runs/tokhead_20261116_v1/p2/` | 1 | 166.4 | 逐手模型群組的每局嵌入（含 p2） |
| `runs/tokhead_20261116_v1/r013/` | 1 | 83.2 | 逐手模型群組的每局嵌入（含 p2） |
| `runs/tokhead_20261116_v1/tuned7e/` | 1 | 194.1 | 逐手模型群組的每局嵌入（含 p2） |
| `runs/tokmaxsim_20261117_v1/mv17/` | 2 | 3.7 | MaxSim 分數 |
| `runs/tokopp_20261117_v1/o5/` | 3 | 10.6 | 再戰證據與對手後驗 |
| `runs/trainval_refinement_20261105_v1/release/` | 1 | 1.2 | 舊題池 H 分數（Private） |

- `HumanSL_state_only.pt` 是 KataGo Human SL 網路 b18c384nbt-humanv0（KataGo v1.15.0 release）轉成 PyTorch state dict，授權為 KataGo Neural Network License，授權全文附在同資料夾。
- 外部帳號頭 `runs/tokfull_20261116/xf*/model.pt` 中有 5 個檔案，在發布前把存檔參數（args）裡的主機暫存路徑換成 `<ssd_stage>/`；張量與原檔逐位元相同，原檔與發布檔的 SHA-256 都列在 `weights_manifest.tsv`。
- 新題池最終匯出的 `RESULT.json` 只保留配方與溫度兩項。
- 選取索引包 `runs/selection_20261126_v1/` 只有整數索引與雜湊，不含棋譜內容或帳號名稱：
  - bigbank3／4／5 選進訓練庫的棋局編號；
  - bigbank90 的摺標籤；
  - 交叉驗證 g1 三摺的題目棋局編號；
  - 400 位封存玩家的帳號雜湊。

  這些選擇取決於研究過程中先前重播過哪些棋局，無法只從官方資料推回，因此直接附上。

### 5.3 不隨 Release 提供、需由官方資料重建的檔案（11 個，0.85 GiB）

下列檔案是由官方資料重新編碼的中間產物，內容包括：
- 訓練與測試棋局的落子序列；
- 訓練中繼資料與玩家切分；
- 帳號標籤；
- 測試棋局索引；
- 訓練棋庫的對局圖；
- 舊題池（Public／Private 舊題目）的棋譜包，含落子序列與提子事件。

這些檔案不隨 Release 提供，請依重建步驟從官方資料產生。步驟如下：
1. 依第 2 節放好官方資料。
2. 依 3.1 節建立 `audit.sqlite`（稽核與近重複分組）。
3. 執行 `python scripts/rebuild_official_derived_20261126_v1.py all`。

腳本結尾會逐一核對下表的 SHA-256；乾淨環境實測 11 個檔案全部與原檔逐位元相同。只需要舊題池棋譜包時，可單獨執行 `python scripts/rebuild_official_derived_20261126_v1.py oldpool`（棋譜內容只讀官方測試 CSV）。

| 檔案 | 位元組 | 預期 SHA-256 | 重建階段 |
|---|---:|---|---|
| `runs/research_v4/official_aligned_v1_20261104/scope_metadata.npz` | 201,199,232 | `16e288236eb054214e65507753b9bb195b3165823db8a4422faa217144bb4a60` | `fitrows`、`splits`、`scope` |
| `runs/v2_baseline_20261111_v2/official_stat/keys.json` | 994,003 | `fb8a2eda6d2aca1ed32f3f13f41deb06223a01c59976ddd7056c83ca62d2caed` | `keys` |
| `runs/bigbank4_90_20261115_v1/EXTMETA.npz` | 85,376,780 | `3257dac140781b0ec50a76a3fab9015ec92d42ae8bb5980fdbeb79a10b06b84a` | `extmeta4` |
| `runs/bigbank5_90_20261115_v1/EXTMETA.npz` | 54,768,995 | `c04814f3fbbc22a40dd90d09783fc7ccc62c0ebb2c0840adfb5576029d9800fd` | `extmeta5` |
| `runs/seqid_20261115_v1/moves.npz` | 237,083,212 | `0f38299aaf3adf889771b0594164f35299e804348f253819d94ed57dc30f9229` | `moves` |
| `runs/seqid_20261115_v1/moves_ext.npz` | 101,194,433 | `6257aa33263ce6d62b042fdbad7e4c69c09da2153499d86d372c9086311f63cc` | `moves_ext` |
| `runs/seqid_20261115_v1/bundle_g1.npz` | 150,452,617 | `156a9b93bc6c5e58f479e74ec7fc29bbd1fc327cf4c5cf4a36e9e366bf468eba` | `bundle` |
| `runs/tokpilot_20261116_v1/labels_all.npz` | 5,487,913 | `acae73fada98989222bfda328038fb2059cca8643066417354b84d3807874d52` | `labels` |
| `runs/oppbank_20261120_v2/full.npz` | 10,296,112 | `e4da9e1650c188ea07b6d84e3c3f611df4e1609e1385b91d96c2d31914a984d3` | `oppbank`（GPU，RTX 3060 實測相同） |
| `runs/oldpool_upgrade_20261123_v1/public_moves_cap.npz` | 40,907,400 | `76bb52ed44c6ff3e1ea42cf92c7dfc2d630b5961de75effcf10fdafa63b730d9` | `oldpool`（GPU） |
| `runs/oldpool_upgrade_20261123_v1/private_moves_cap.npz` | 27,046,924 | `99dd07978ee8fe1f4a186bacae6246c8af2b467bd777f7d47ba19e18fcd6c113` | `oldpool`（GPU） |

各階段的來源：
- 訓練切分用 `goid_v4.pilot_scope_v1.fit_player` 的雜湊規則（fit／validation／holdout ＝ 60／25／15%），加上選取索引包裡的 400 位封存玩家。
- 其餘階段直接呼叫原流程的函式：
  - `official_aligned_v1.capacity`
  - `bigbank4/5` 的 `extmeta_build`
  - `seqid_20261115_v1.moves_meta`／`moves_official`／`data2`
  - `seqid_20261115_v2.bundle`
  - `oldpool_upgrade_20261123_v1.pack_pool`
- `labels` 與 `oppbank` 兩階段，依原本的產生指令與 `oppbank_tcond_20261120_v1.py` 的第一段寫成函式。

## 6. 重現最終提交檔

預期 SHA-256：`b3df6a05ddcc5614047691b44c0969fb60645b5aeab165f9c8d70de804fe82ad`（1,600 列）。

L1 只需要 Release 的檔案。L2 與 L3（舊題池）另外需要 5.3 節的檔案：L2 用前 9 個，L3 用兩個舊題池棋譜包。請先依第 2 節放好官方資料，再執行：

```bash
cd comp1
python scripts/audit_training.py index
python scripts/audit_training.py parse
python scripts/audit_training.py parse --selection all --workers 4 --batch-size 128
python scripts/audit_training.py summary
python scripts/audit_near_duplicates.py --output data/processed
python scripts/apply_family_groups.py
python scripts/rebuild_official_derived_20261126_v1.py all   # 重建 5.3 節的 11 個檔案並核對 SHA-256，最後兩步需要 GPU
```

重現步驟：

```bash
cd comp1
python scripts/report_repro_20261125_v1.py check   # L1、L2：約 1 分鐘，CPU 即可
python scripts/report_repro_20261125_v1.py emb     # L3（舊題池）：約 5 分鐘，需要 GPU；須先跑過上面的重建（oldpool 階段）
```

| 層級 | 從哪裡重建 | 結果（乾淨環境實測） |
|---|---|---|
| L1 | 存下的最終分數矩陣 → 排名與指派 → CSV | SHA-256 與最終檔相同 |
| L2 | 各成分分數矩陣與每局嵌入，以及由官方資料重建的 9 個檔案 → 融合（樞紐度、對手社群現算）→ 排名與指派 → CSV | SHA-256 與最終檔相同 |
| L3（舊題池） | 由官方資料重建的舊題池棋譜包 → 7 個手順網路權重重新嵌入 45,000×2 局 → 融合 → 指派 | 舊題池 800 列與最終檔完全相同 |

輸出在 `runs/report_repro_20261125_v1/`：`L1_from_saved_final_scores.csv`、`L2_from_components.csv`、`CHECK.json`、`L3o.json`。

## 7. 硬體與時間估計
| 工作 | 硬體 | 時間 |
|---|---|---|
| 稽核 100 萬局（`audit_training.py`，4 個 worker） | CPU | 約 36 分鐘 |
| 近重複分組 | CPU | 約 12 分鐘 |
| 重建 5.3 節的 11 個檔案 | CPU，最後兩步 RTX 3060 | 約 10 分鐘 |
| 重現 L1＋L2 | CPU | 約 1 分鐘 |
| 重現 L3（舊題池） | RTX 3060 | 約 5 分鐘 |
| 有紀錄的模型訓練（33 個模型） | RTX 4090 與 RTX 3060 | 合計約 90 GPU 小時 |
| 從零重做全部（含特徵抽取與快取） | RTX 4090 為主 | 約 190–240 GPU 小時（估計），快取約 450 GB 磁碟 |

## 8. 已知限制
- **新題池的 L3（全部從權重重算）未實作成一鍵流程，也未驗證。** 需要先對官方題目重抽逐手特徵（官方棋局與對手方各約 3.1 GB，未附），再以各模型重新嵌入、重跑配對融合擬合與外借。目前以 L2（各成分分數矩陣與每局嵌入）重現。
- **p2（配對融合中 tkp 群組的 4 萬局版逐手模型）沒有保存權重**，只附它的每局嵌入 `runs/tokhead_20261116_v1/p2/emb.npz`。
- 舊題池 25% 權重的 **H 分數由早期管線產生**，以中間產物提供（`runs/fisher_policy_pilot_20261105_v2/public_inputs/H.npy`、`runs/trainval_refinement_20261105_v1/release/Private_H_scores.npy`）；產生 H 的早期程式未收錄。
- 舊題池 Public 部分的 L3：原始流程對其中兩個模型沿用了先前的嵌入，重新嵌入後融合矩陣最大差 0.0013，名次與最終檔相同。
- 5.3 節的 11 個檔案可以由官方資料逐位元重建。但各訓練庫「選了哪些棋局」是以選取索引包提供，不是從官方資料重新推導：
  - 這些選擇是研究過程逐輪累積的結果，例如優先選已重播過的棋局；
  - 400 位封存玩家則是 10 月 25 日凍結的。
- `oppbank` 階段在 GPU 上以 bfloat16 推論，RTX 3060（PyTorch 2.14.1+cu126）實測與原檔相同；換成其他 GPU 或版本，數值可能有末位差異。
- `e2e_warm_long_h5e5`（主幹微調長程版）沒有留下 LOG，參數由啟動指令還原（見 `configs/training_args.json`）。
- 從零重訓受 GPU 非決定性影響，權重與最終檔不會逐位元相同，只能期望分數相近。
- 研究過程中未進入最終檔的程式沒有收錄；收錄的模組裡仍有部分早期實驗函式，因為最終流程會匯入同一檔案。

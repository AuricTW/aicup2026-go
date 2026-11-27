<p align="center"><a href="README.md"><img src="../.github/assets/lang-zh-on.svg" alt="繁體中文"></a> <a href="README.en.md"><img src="../.github/assets/lang-en-off.svg" alt="English"></a></p>

# 競賽二：從圍棋棋譜預測玩家棋力等級（TEAM_11306）

← 回到 [repo 首頁](../README.md)

最終提交檔：Private 0.789846（第 6 名）。本資料夾是產生該檔的程式碼：資料處理、特徵抽取、模型訓練、證據計算、融合與集成，以及重現檢查。
方法說明見報告；本文件說明如何安裝、放置資料與權重、依序執行，以及如何重現最終提交檔。

| 資料夾 | 內容 |
|---|---|
| `src/` | 最終提交檔用到的 Python 程式（全部平放在同一層，互相 import） |
| `vm/` | GPU 主機上執行的 shell 腳本（KataGo 特徵、身分辨識網路、競賽一 token 轉用、rank1） |
| `prev_system/` | 前一代系統（Codex 時期）的程式；最終檔的第二階段融合使用它的固定後驗 |
| `analysis/` | 只為報告方法與圖表需要的分析程式（不影響提交檔內容） |
| `configs/` | 機器設定範例 `paths.example.yaml`；checkpoint 內存的訓練參數 `training_args.json` |
| `SOURCES.tsv` | repo 內每個檔案對應的原始檔與其 SHA-256，以及是否改過路徑 |
| `weights_manifest.tsv` | Release 權重與中間產物的檔名、大小、SHA-256 |

## 1. 環境安裝

| 項目 | 版本 |
|---|---|
| Python | 3.12（開發用 3.12.14） |
| CPU 階段（資料集、HT 模型、證據、融合、提交檔） | PyTorch 2.6.0+cu124 與其他套件，見 `requirements.txt`（版本已釘死，取自產生最終檔的環境） |
| GPU 主機階段 | 環境 A：PyTorch 2.10.0+cu128（KataGo 特徵、身分辨識網路）；環境 B：PyTorch 2.11.0+cu128（競賽一 token worker、head、rank1），見 `requirements-gpu.txt` |
| GPU | 重現 R1、R2 不需要 GPU；HT 模型用 RTX 3060 12 GB；特徵抽取與身分辨識網路用 RTX 4090 24 GB |
| KataGo 引擎 | v1.18.2，Linux CUDA 12.8 版：https://github.com/lightvector/KataGo/releases/tag/v1.18.2 （執行檔不放在 repo） |
| 作業系統 | 本機 Windows 11；GPU 主機 Ubuntu 24.04 |

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
cp configs/paths.example.yaml configs/paths.yaml      # 再依機器修改
```

- 機器相關的設定寫在 `configs/paths.yaml`，或用環境變數 `AICUP2_<KEY>` 覆蓋，由 `src/comp2_paths.py` 讀取。
  - 官方資料根目錄 `root`。
  - 中間檔資料夾 `work`：原本的 `vm_out/`，放資料集、模型輸出與證據檔。
  - 成員後驗資料夾 `parts`。
  - 前一代系統的固定輸出 `d64_*`。
  - KataGo 執行檔與權重位置。
- GPU 主機腳本的設定寫在 `vm/env.sh`，由 `vm/env.example.sh` 複製後修改。
  - 工作資料夾 `KQ`。
  - 兩個 Python 環境 `PY`、`PY1`。
  - 競賽一權重位置 `C1RUNS`。
  - 暫存 SSD `SCRATCH`。
- KataGo 的 Python 模組與 sgfmill 在上層的 `../third_party/`（原檔未修改，附原授權）。`extract_trunk*.py` 會自動加入這個路徑。
- 競賽一 token worker 用 `../comp1/payload/`，見 comp1 的 README。

## 2. 資料放置

只使用主辦單位提供的資料，本 repo 不附任何原始資料、特徵、證據檔或偽標籤。

| 資料 | 來源 | 放置位置（相對於 `root`） |
|---|---|---|
| 官方訓練資料 v1.1（10 個 CSV，1,000,000 局，8,007 位玩家） | https://github.com/AILAB-NDHU/aicup2026-go-dataset （release v1.1） | `data/raw/training_v1_1_20261104/training/train_*.csv` |
| 官方訓練資料 v1.0（玩家切分與 token payload 的雜湊用） | 同上（release v1.0） | `data/raw/training/` |
| 官方測試題（Public / Private 各 400 題）與範例提交檔 | 主辦單位（AIdea 競賽頁） | `data/raw/official_rank_tests_20261104_v1/tests/`（`rank_prediction_test_public.csv`、`rank_prediction_test_private.csv`、`sample_submission_rank.csv`） |
| 玩家切分 `data/interim/index.csv` | 由 `prev_system/tools/prepare_data.py` 產生（train 6,402 人、dev 801 人、lockbox 804 人） | `data/interim/index.csv` |

GPU 主機上的建檔腳本（`make_gamelist*.py`）讀 `gpu_data` 下的 `raw/training/`、`raw/tests/` 與 `raw/split_index.csv`（同一份官方檔與切分檔的副本）。

## 3. 執行順序

下列步驟即最終檔的實際鏈路。

- 第 3.5、3.6 步（R2、R1）已在本 repo 實跑驗證，見第 6 節。
- 第 3.1 至 3.4 步的指令由各腳本的參數、`vm/` 腳本與 checkpoint 內存的訓練參數（`configs/training_args.json`）還原。當時的建檔指令沒有逐條記錄，這些步驟在本 repo 沒有重跑。
- 「GPU」表示在 GPU 主機的 `$KQ` 下執行；其餘在本機 comp2 根目錄執行，中間檔都在 `work/`。

### 3.1 前處理（本機，CPU）

```bash
python prev_system/tools/prepare_data.py --training-dir data/raw/training --output-dir data      # 玩家切分 data/interim/index.csv
python src/build_moves_dataset.py work/moves_all.npz --L 200 --max-hd 1      # 落子序列（身分辨識網路）
python src/build_moves_extra.py work/moves_all.npz work/moves_all2.npz          # + dev 玩家其餘棋局（split code 3）
python src/build_moves_lockbox.py work/moves_all2.npz work/moves_all3.npz       # + lockbox 帳號棋局（split code 4，只當 gallery 類別）
```

- `fixed_inputs/work/` 的 `oof_games.csv`、`oof_dense_map.csv` 是 949 位 OOF 驗證玩家清單（Release），放到 `work/`。
- GPU 主機的 `tm/` 也要放同一份 `moves_all*.npz`。

### 3.2 特徵（GPU）

```bash
python $SRC/make_gamelist.py gamelist.jsonl                     # dev / 訓練 / 測試題棋局清單
python $SRC/make_select2.py gamelist.jsonl select_h_stage2.json A:20,T:all,C:10   # 選取規格依 chain_stage2.sh 的說明還原
bash vm/chain_stage2.sh                                          # Human SL 逐手似然（kata_human.py，KataGo 引擎，1 visit）
bash vm/run_trunk_oof_vm.sh                                      # Human SL 主幹 PCA64 嵌入（extract_trunk.py，PCA 用 Release 的 assets/pca64.npz）
python $SRC/make_gamelist_id.py gamelist_id.jsonl trunk/abt,trunk/oof 400 && bash vm/run_trunk_id_vm.sh   # gallery 前 80 手嵌入
```

### 3.3 訓練

HT 序列模型在本機 RTX 3060 上訓練：

```bash
python src/build_h_dataset.py work/dsH_v1.npz <H_DIRS> --L 320
python src/build_ht_dataset.py work/dsH_v1.npz <TRUNK_DIRS> work/dsHT_v1.npz
python src/seq_model.py work/dsHT_v1.npz work/ht1_s1 --epochs 14 --d 192 --layers 4 --L 320 --bs 128 --lr 6e-4 --split --aux 0.5 --seed 1   # seed 2 → ht1_s2
```

身分辨識網路在 GPU 主機上訓練，共 15 個網路分三個軌道，依序執行：

```bash
bash vm/run_idnet_vm.sh id_v2x --L 200 --epochs 30 --seed 0 --eval-every 4000 --holdout-players $KQ/oof_games.csv   # 教師；其餘見下列腳本
bash vm/chain_board_vm.sh; bash vm/chain_board_ens_vm.sh; bash vm/chain_board2_vm.sh        # x / full 軌道的 board-patch 網路
bash vm/chain_trunkx2_vm.sh; bash vm/chain_trunkx3_vm.sh                                     # x 軌道的主幹嵌入網路 → idnet/tx2_full/acc_6m.npz
bash vm/chain_lockbox_vm.sh; bash vm/chain_lockbox2_vm.sh; bash vm/chain_trunklb_vm.sh      # test 軌道（含 lockbox gallery）→ idnet/ta2lb/acc_6m.npz
```

- 每個網路的完整參數（輪數、seed、L、board、trunk、偽標籤檔與門檻）存在 checkpoint 的 `args`，整理於 `configs/training_args.json`。
- 教師網路 `id_v2x`、`id_v2full`、`id_v4lb` 與第一輪學生 `e3_ps05`、`f1_ps05`、`a2lb_ps05` 的參數也以該檔為準。
- 偽標籤由 `idnet_pseudo.py` 產生（門檻 0.5）。

rank1 在 GPU 主機的環境 B 上訓練，輸入是競賽一的 bigbank token 快取（只含訓練檔棋局）：

```bash
bash vm/rank1_chain.sh      # tok_ranknet_vm.py rank1 tokres_rank_labels.npz <bigbank results> --epochs 8 --seed 0
```

### 3.4 推論與證據

```bash
# 身分辨識網路集成證據（GPU；即上面各 chain 腳本的最後一步 idens.py --acc）
#   x 軌道 idnet/tx2_full/acc_6m.npz、full 軌道 idnet/bf1_p9c64/acc_3.npz、test 軌道 idnet/ta2lb/acc_6m.npz → 複製到 work/idnet_*/
# 競賽一 token 轉用（GPU，環境 B）
python $SRC/make_tok_payload.py tm/moves_all3.npz <root> <comp1 payload code dir> tokpay      # 題目棋局對手側的 worker 輸入
bash vm/run_tok.sh                    # 早盤 token（競賽一凍結 worker）
bash vm/run_toklate_c2.sh             # 晚盤 token（手 181–300）
bash vm/ft_cache.sh testdev oof; bash vm/ft_run.sh t6 <trunk_ft.pt>; bash vm/ft_early_vm.sh t6     # 三個微調主幹 t6 / b3 / ha 各一次
python $SRC/tok_apply.py ... / tok_apply_late.py ...      # 10 個 head → work/tokres/{td,oof}_<head>.npz、work/xf/{td,oof}_<head>_<trunk>.npz
# rank1 套用與轉換
python src/tok_rank_apply.py <rank1/model.pt> work/moves_all3.npz work/tokrank/rank1_dir.npz "<own specs>" "<opp specs>"
python src/rank1_conv.py work/tokrank/rank1_dir.npz work/idnet_tx2_full/acc_6m.npz work/tokrank/r1
python src/rank1_conv.py work/tokrank/rank1_dir.npz - work/tokrank/r1full dev=work/idnet_bf1_p9c64/acc_3.npz test=work/idnet_ta2lb/acc_6m.npz
# 帳號檔加入列出棋局的對手計數（enr 組）
python src/listed_counts.py <REF.npz> <PSEUDO.npz> work/tokeval/cntadd_<x|full|test>.npz
# HT 模型對 dev / test / OOF 題目棋局的預測
#   → work/ht1_s{1,2}/{dev,test}_pred.npz、work/oofpredHT_s{1,2}/all_pred.npz
```

`tok_classmap.npz`（競賽一類別 → 競賽二帳號）由 `tok_classmap.py` 產生，Release 附一份在 `assets/`。

### 3.5 融合：91 個成員（本機，CPU）

```bash
python src/run_parts.py --workers 4
```

- 對每個成員，先以 `tok_evidence.py` 產生它的 token 證據組（三個軌道；已存在就跳過），再呼叫 `fuse_sources.py`。
- 每個成員輸出 `work/parts/go_rank_<tag>_*`。
- 91 個成員的設定寫在 `src/members.py`，內容是原 PowerShell 驅動檔 `make_variants2/6/7/8.ps1`、`cnt_round.ps1` 的 Python 版：
  - 第一階段用 free 或 ridge 估計；
  - dev 題抽樣種子 `DEV_SEED`；
  - `ZOPP_GATE`；
  - token 網路組合；
  - rank1 來源。

### 3.6 產生提交檔（本機，CPU）

```bash
python src/blend_final.py --out work/final/go_rank_final.csv
```

91 個成員後驗取幾何平均，再以期望得分解碼。最後印出 SHA-256，並與最終檔比對。

## 4. 重要模組的輸入與輸出

| 模組 | 輸入 | 輸出 |
|---|---|---|
| `build_moves_dataset.py` / `_extra.py` / `_lockbox.py` | 官方訓練 CSV、測試題、`index.csv` | `moves_all{,2,3}.npz`：落子座標、帳號、段位、split code |
| `kata_human.py`（+ `kata_eval.py` 的 SGF 解析） | 棋局清單 jsonl、KataGo 引擎、Human SL 引擎格式網路 | 每手在多個段位設定下的 log p、排名、勝率 `hchunk_*.npz` |
| `extract_trunk.py` / `extract_trunk_id.py` | 棋局清單、Human SL checkpoint、`pca64.npz` | 每手前局面的主幹 PCA64 嵌入 `tchunk_*.npz` |
| `build_h_dataset.py` → `build_ht_dataset.py` | 似然 chunk、嵌入 chunk | HT 資料集 `dsHT_v1.npz` |
| `seq_model.py` | HT 資料集 | HT 模型 `model.pt`；`dev_pred.npz`、`test_pred.npz`（目標玩家與對手的等級 logit） |
| `idnet.py` / `idnet_pseudo.py` | `moves_all*.npz`、（可選）嵌入、偽標籤 | 身分辨識網路 `model.pt`、`idnet_pred.npz`；偽標籤 `pseudo_*.npz` |
| `idens.py --acc` | 多個身分辨識網路 | 集成證據 `acc_*.npz`：辨識機率、帳號檔、等級核 |
| `tok_apply.py` / `tok_apply_late.py` | 競賽一 head、token 結果、`tok_classmap.npz` | 每局對手側的帳號辨識 `{td,oof}_<head>.npz` |
| `tok_evidence.py` | 參考集成證據、token head 輸出 | 第二辨識器證據 `tokeval/{x,full,test}_<set>_A1.0.npz` |
| `tok_ranknet_vm.py` → `tok_rank_apply.py` → `rank1_conv.py` | token 快取、段位標籤 | rank1 `model.pt`；`tokrank/r1{,full}_{own,conv,raw}.npz` |
| `fuse_sources.py`（`stage1_lib.py`、`evaluate_queries.py`、`final_predict.py`、`predict_test.py`、`idnet_eval.py`） | 上述證據、前一代系統後驗、OOF 玩家清單 | 一個成員的 `x_D64_posteriors.npz`（測試題）、`OOF_posteriors.npz`、`VALIDATION.json` |
| `blend_final.py`（`blend_test.py`） | 91 個 `x_D64_posteriors.npz` | 提交檔 CSV 與後驗 |

## 5. 權重與中間產物

權重不放在 git，請從本 repo 的 GitHub Release 下載。Release `v1.0` 的附件 `comp2_weights.tar` 收有本節列出的全部檔案與 `MANIFEST.csv`（附件的 SHA-256 見 `SHA256SUMS.txt`）：

```bash
mkdir -p weights/comp2
tar -xf comp2_weights.tar -C weights/comp2/     # Windows cmd 也可用內建的 tar
```

- Release 檔案依 `weights_manifest.tsv` 的相對路徑解開到任一資料夾，例如 `weights/comp2/`（kind 為 `comp1 release` 的 3 列在競賽一的 Release，見下文）。
- 再把 `configs/paths.yaml` 與 `vm/env.sh` 指到該處。
- GPU 主機上要放的位置：
  - `ht/`、`idnet/<name>/`、`rank1/` 分別對應 `work/ht1_s{1,2}/`、`$KQ/idnet/<name>/`、`$KQ/rank1/`。
  - `assets/pca64.npz` 放到 `$KQ/trunk/`。
  - `assets/tok_classmap.npz` 放到 `$KQ/ft/`。

本隊在競賽二訓練的權重（18 個，525.0 MB）：

| 檔案 | bytes | SHA-256 |
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

其他 Release 檔案（完整 SHA-256 見 `weights_manifest.tsv`）：

| 類別 | 檔案 | 用途 |
|---|---|---|
| 小型資產 | `assets/pca64.npz`（PCA 參數）、`assets/tok_classmap.npz`（類別對照）、`rank1/LOG.json` | 主幹嵌入、token head 套用 |
| 前一代系統的固定輸出 | `fixed_inputs/prev_system/`：測試題後驗、OOF 後驗 `OLD_FUSED.npz`、OOF 題目 `queries_dense.json` | 第二階段融合 |
| OOF 驗證玩家清單 | `fixed_inputs/work/oof_games.csv`、`oof_dense_map.csv` | 融合的驗證題 |
| 91 個成員的測試題後驗 | `parts/go_rank_<tag>_x_D64_posteriors.npz`（9.1 MB） | 重現 R1 |

由其他來源提供的權重：

- **競賽一的權重由 comp1 的 Release 提供**，以下列出各權重在 comp1 `weights_manifest.tsv` 中的路徑。下載後保持 `runs/...` 結構，並把 `vm/env.sh` 的 `C1RUNS` 指到該 `runs/`。
  - 8 個早盤 head `runs/tokhead_20261116_v1/{vm_t1,vm_t6,full_s1,full_big_s3,full6a_s12,vm_t4,vm_t10,vm_t2}/model.pt`。
  - 晚盤 head `vm_t17late`、`full_late_s24`。
  - 微調主幹 `wl_h8x48_b3_trunk_ft.pt`（b3）、`e2e_warm_long_h5e5/trunk_ft_step7500.pt`（t6）、`runs/tokfull_20261116/e2e_warm_h8x48a/trunk_ft.pt`（ha）。
  - 其中 3 個只用於競賽二的第二辨識器，SHA-256 也列在本 repo 的 `weights_manifest.tsv`（kind 為 `comp1 release`）：

    | comp1 Release 路徑 | bytes | SHA-256 |
    |---|---:|---|
    | `runs/tokhead_20261116_v1/vm_t4/model.pt` | 54,487,821 | `51a42b2aed54aac715dd53740038ce0f81f57837bf85bcbda59c58db1d87eda6` |
    | `runs/tokhead_20261116_v1/vm_t10/model.pt` | 33,913,545 | `aa0a7d1677363bbc8cfd9bbefada25474c5c8c763ccde102ab2000bbe76ec356` |
    | `runs/tokfull_20261116/e2e_warm_h8x48a/trunk_ft.pt` | 35,157,460 | `fc33fa4407265f62b0f7b2c320a72ae7232c11262e32dec13b4a7091180c49be` |
  - `HumanSL_state_only.pt`。
  - 這些權重只用兩競賽共用、逐位元組相同的官方訓練檔訓練。
- **外部權重**不重新散布，請從 KataGo 官方下載：
  - Human SL 網路 `b18c384nbt-humanv0.ckpt`（訓練 checkpoint，SHA-256 `84b23476…`）與引擎格式 `b18c384nbt-humanv0.bin.gz`，來源 https://katagotraining.org/extra_networks/ ，授權 https://katagotraining.org/network_license/ 。
  - 本隊當時用的引擎格式檔，是以 KataGo 的匯出工具從同一個 checkpoint 匯出的（SHA-256 `8b03891e…`）。官方 `.bin.gz` 的雜湊可能不同。

## 6. 重現最終提交檔

最終提交檔：`go_rank_IDv10x_union91_20261120.csv`，12,043 bytes。

SHA-256 `1ADC681D208EF87977103CDA1604D22AAB3041A020F79471F13A35C59404CA4A`

| 層級 | 起點 → 終點 | 指令 | 在本 repo 的驗證結果 |
|---|---|---|---|
| R1 | Release 的 91 個成員後驗 → CSV | `python src/blend_final.py --parts <weights>/comp2/parts --out out.csv` | **逐位元組相同**（SHA-256 如上），CPU 數秒 |
| R2 | 保存的證據檔與模型輸出 → 91 個成員 → CSV | `python src/run_parts.py`，再執行 R1 | 以原專案的證據檔實跑：**91 個成員全部逐位元組相同，CSV 的 SHA-256 相同**（CPU 4 個並行，22 分鐘） |
| R3 | 原始棋譜 + Release 權重 → 推論 → CSV | 第 3.2、3.4 步的推論部分 + R2 | 沒有重跑，見第 8 節 |
| R4 | 原始訓練檔 → 重訓全部模型 → CSV | 第 3 節全部 | 沒有重跑，只提供程式與參數 |

R2 用到的證據檔是官方資料的衍生特徵，不放進 Release；R2 只能在自行執行完第 3.1 至 3.4 步後進行。

## 7. 硬體與時間估計

| 步驟 | 硬體 | 時間（估計，依原始紀錄） |
|---|---|---|
| R1 | CPU | 數秒 |
| R2（91 個成員） | CPU（i7-12700，4 個並行） | 22 分鐘（實測） |
| 前處理與資料集 | CPU | 約 10 分鐘 |
| Human SL 逐手似然 | RTX 4090 | 約 10 小時（含 HT 訓練集；只推論開發、測試與 OOF 棋局約 5 小時） |
| 主幹嵌入（題目棋局） | RTX 4090 | 約 3 小時；gallery 79.2 萬局前 80 手另需約 16 小時 |
| HT 模型（2 個 seed × 14 輪） | RTX 3060 | 約 0.5 小時 |
| 身分辨識網路（15 個 + 6 次偽標籤） | RTX 4090 | 約 35–45 GPU 小時（每個 1.5–3.5 小時） |
| 競賽一 token 抽取與 head 套用（43,389 局） | RTX 4090 | 約 7 小時 |
| rank1 | RTX 4090 | 約 1–2 小時 |

## 8. 已知限制

1. **只驗證 R1 與 R2**。
   - 兩者都已確認逐位元組相同。
   - R2 是在原專案的中間檔上實跑的。
   - R3、R4 沒有在本 repo 重跑。第 3.1 至 3.4 步的指令由腳本參數與 checkpoint 內的訓練參數還原，當時的建檔指令沒有逐條記錄。
2. **GPU 主機環境 A 已不存在**。
   - 原本的環境是 PyTorch 2.10.0+cu128，KataGo 引擎的 cuBLAS / cuDNN 也取自該環境，沒有保留完整鎖檔。
   - 重建時改用 `requirements-gpu.txt` 的版本，數值可能有微小差異。
3. **身分辨識網路沒有只推論的程式**。`idnet.py` 在訓練結束時輸出 `idnet_pred.npz`；只用 Release 權重推論新棋局，需要另寫一個載入 `idnet.Net` 的預測程式。
4. **前一代系統（`prev_system/`）的程式只供參考**。
   - 這些程式以原專案的相對路徑讀取上游 run，上游的特徵與模型沒有整理進本 repo。
   - 最終檔只使用它保存的後驗，這些後驗附在 Release 的 `fixed_inputs/prev_system/`。
5. **重訓的逐位元組重現無法保證**。身分辨識網路使用 GPU 非決定性運算，相同 seed 只能期待驗證分數接近。逐位元組重現只適用於 R1、R2。
6. `analysis/` 的程式需要成員的 OOF 後驗與鏡像驗證證據。這些檔案沒有附在 Release，因為它們由訓練玩家的資料推得；附上的程式只用於說明報告中的分析方法。
7. Release 裡的身分辨識網路 checkpoint 內含類別對應的匿名帳號名稱（`owners`），它們與官方訓練檔中的 player_id 相同。

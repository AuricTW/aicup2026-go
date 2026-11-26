<p align="center"><a href="README.md"><img src="https://img.shields.io/badge/%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87-6e7781?style=for-the-badge" alt="繁體中文"></a> <a href="README.en.md"><img src="https://img.shields.io/badge/English-2a78d6?style=for-the-badge" alt="English"></a></p>

<h1 align="center">AI CUP 2026 Fall<br>Go Playing Strength and Style Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/KataGo-Human%20SL-444444" alt="KataGo Human SL">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2a8f3c" alt="MIT License"></a>
</p>

<p align="center">Complete code for both sub-competitions: data processing, model training, inference, and reproduction checks for the final submissions.</p>

## Competitions and results

| | Competition | Team | Private Leaderboard | Code |
|:---:|---|:---:|:---:|:---:|
| 1 | Identifying players from sets of Go game records | TEAM_11307 | **0.923441** (rank 4) | [`comp1/`](comp1/README.en.md) |
| 2 | Predicting player rank from Go game records | TEAM_11306 | **0.789846** (rank 6) | [`comp2/`](comp2/README.en.md) |

Each competition folder has its own README covering installation, data placement, run order, the inputs and outputs of the main modules, and how to reproduce the final submission.

## Repository layout

```text
aicup2026-go/
├── comp1/         Competition 1: player identification (code, configs, README)
├── comp2/         Competition 2: rank prediction (code, configs, README)
├── third_party/   Third-party code shared by both: KataGo Python modules, sgfmill (unmodified, original licenses)
└── LICENSE        MIT
```

## Quick start

**1. Get the code**

```bash
git clone https://github.com/AuricTW/aicup2026-go.git
cd aicup2026-go
```

**2. Download the weights** from [Release v1.0](https://github.com/AuricTW/aicup2026-go/releases/tag/v1.0) into the repository root.

| Asset | Size | Unpack to |
|---|---:|---|
| `comp1_weights.tar.part00`–`part02` | 3.99 GB (3 parts) | join into `comp1_weights.tar`, then unpack into `comp1/` |
| `comp2_weights.tar` | 553 MB | unpack into `weights/comp2/` |
| `SHA256SUMS.txt` | — | SHA-256 of each asset |

```bash
cat comp1_weights.tar.part0* > comp1_weights.tar
# Windows cmd: copy /b comp1_weights.tar.part00+comp1_weights.tar.part01+comp1_weights.tar.part02 comp1_weights.tar
sha256sum -c SHA256SUMS.txt --ignore-missing
tar -xf comp1_weights.tar -C comp1/
mkdir -p weights/comp2 && tar -xf comp2_weights.tar -C weights/comp2/
```

**3. Reproduce the final submissions** (full details in section 6 of each README)

| Competition | Fastest reproduction level | Command | Expected SHA-256 |
|---|---|---|---|
| 1 | L1: saved final score matrix → CSV (CPU) | `python scripts/report_repro_20261125_v1.py check` (run in `comp1/`; also runs L2, which needs the official-data rebuild first) | `b3df6a05…fe82ad` |
| 2 | R1: 91 member posteriors → CSV (CPU, seconds) | `python src/blend_final.py --parts ../weights/comp2/parts --out out.csv` (run in `comp2/`) | `1ADC681D…04CA4A` |

## Data and weights

- **Official data**: download it from the organizer's [aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset). This repository does not include raw data.
- **Files re-encoded from official data** (move sequences, account labels, test-game bundles and similar) are not in the Release. Each README gives the rebuild commands; the rebuilt files have been verified byte-identical to the originals.
- **Model weights** are on GitHub Releases, not in git. Sizes and SHA-256 of every file are listed in `comp1/weights_manifest.tsv` and `comp2/weights_manifest.tsv`.

## License

Our code is released under the [MIT License](LICENSE). KataGo and sgfmill in `third_party/` keep their original licenses. The KataGo Human SL network weights are used under the KataGo Neural Network License, whose full text is included in the Release.

"""Figures for 陸、分析與結論 (comp2), shared report style (report_fig_style.py, 11-26 revision: upright vertical CJK
y-labels, BLUE / GREY / ORANGE / AQUA palette, seq_cmap for the confusion matrix).  Reads analysis_numbers.json.
Run with the shared figure environment:
    python make_figures.py      (matplotlib; fonts Times New Roman + DFKai-SB)
Figures: 1 milestones, 2 by G, 3 confusion matrix (row-normalised, %), 4 by recognition quartile, 5 expected vs actual.
Milestone scores of fig1 are the OOF (7,592 questions, fused with the previous system, full coverage) values recorded in
runs/kata_quality_20261114_v1/FINDINGS_20261114.md and FINDINGS_20261117_se.md; the last bar is recomputed in analysis_numbers.json.
Public / Private expectation in fig5: FINAL_REVIEW_20261121.md §6 (posterior self-expectation) and the official leaderboard.
Colour meaning: BLUE = final submission (full coverage), GREY = comparison (earlier stages, mirrors, expectation).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_fig_style import (BLUE, CM, GREY, INK_2, WIDTH_CM, apply_style, cell_text_color, new_figure,  # noqa: E402
                              save_figure, seq_cmap, set_ylabel_cjk)

HERE = os.path.dirname(os.path.abspath(__file__))
O = json.load(open(os.path.join(HERE, "analysis_numbers.json"), encoding="utf8"))
apply_style()
W = 0.38
R = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def legend_below(ax, ncol=2, y=-0.16):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.4, columnspacing=1.6)


# fig1: milestones on the same validation questions
steps = [("前一代\n系統", 0.6936), ("加入\nHuman SL\n似然模型", 0.7215), ("HT 序列\n模型", 0.7371), ("加入對手\n身分證據", 0.7644),
         ("學生網路\n集成", 0.7709), ("主幹特徵\n網路", 0.7780), ("加入第二\n辨識器", 0.7899), ("最終\n提交檔", O["full"][0])]
fig, ax = new_figure(7.0)
x = np.arange(len(steps)); v = [s[1] for s in steps]; lo, hi = O["full"][1], O["full"][2]
ax.bar(x, v, color=[GREY] * (len(steps) - 1) + [BLUE], width=0.65)
ax.errorbar(x[-1], v[-1], yerr=[[v[-1] - lo], [hi - v[-1]]], color=INK_2, capsize=3, lw=1)
for i, val in enumerate(v):
    ax.text(i, (hi if i == len(v) - 1 else val) + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=8, color=INK_2)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=8)
ax.set_ylim(0.66, 0.835); set_ylabel_cjk(ax, "分數")
save_figure(fig, os.path.join(HERE, "fig1_milestones.png"))

# fig2: by G
fig, ax = new_figure(6.5)
Gs = sorted(O["by_G"], key=int); xg = np.arange(len(Gs))
fu = [O["by_G"][g]["full"][0] for g in Gs]; mi = [O["by_G"][g]["mirror"][0] for g in Gs]
ax.bar(xg - W / 2, fu, W, color=BLUE, label="完整驗證")
ax.bar(xg + W / 2, mi, W, color=GREY, label="鏡像平均")
for i in range(len(Gs)):
    ax.text(i - W / 2, fu[i] + 0.003, f"{fu[i]:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
ax.set_xticks(xg); ax.set_xticklabels(Gs); ax.set_xlabel("每題局數 G")
ax.set_ylim(0.6, 0.88); set_ylabel_cjk(ax, "分數")
legend_below(ax, y=-0.30)
save_figure(fig, os.path.join(HERE, "fig2_by_G.png"))

# fig3: confusion matrix, rows = true rank, row-normalised percentages
C = np.array(O["confusion"], float); P = 100 * C / C.sum(1, keepdims=True)
fig, ax = new_figure(11.5)
ax.grid(False)
im = ax.imshow(P, cmap=seq_cmap(), vmin=0, vmax=100, aspect="auto")
for i in range(10):
    for j in range(10):
        pct = int(round(P[i, j]))
        if pct > 0:
            ax.text(j, i, f"{pct}", ha="center", va="center", fontsize=9, color=cell_text_color(P[i, j], 100))
ax.set_xticks(range(10)); ax.set_xticklabels(R); ax.set_yticks(range(10)); ax.set_yticklabels(R)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("預測等級"); set_ylabel_cjk(ax, "真實等級")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(length=0)
cb.ax.set_title("比例（%）", fontsize=9, pad=6)
save_figure(fig, os.path.join(HERE, "fig3_confusion.png"))

# fig4: by recognition quartile
B = O["by_recognition"]; xb = np.arange(len(B))
fig, ax = new_figure(7.0)
ax.bar(xb - W / 2, [b["full"] for b in B], W, color=BLUE, label="完整驗證")
ax.bar(xb + W / 2, [b["mirror"] for b in B], W, color=GREY, label="鏡像平均")
for i, b in enumerate(B):
    ax.text(i - W / 2, b["full"] + 0.002, f"{b['full']:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
ax.set_xticks(xb); ax.set_xticklabels([f"第 {i + 1} 組\n{b['lo']:.2f}–{b['hi']:.2f}" for i, b in enumerate(B)], fontsize=8)
ax.set_ylim(0.7, 0.85); set_ylabel_cjk(ax, "分數"); ax.set_xlabel("對手平均辨識機率（四分位分組）")
legend_below(ax, y=-0.30)
save_figure(fig, os.path.join(HERE, "fig4_by_recognition.png"))

# fig5: expected vs realised on the official test
fig, ax = new_figure(6.5)
exp_ = [0.8170, 0.8149]; real = [0.8300, 0.7898]; xx = np.arange(2)
ax.bar(xx - W / 2, real, W, color=BLUE, label="實際分數")
ax.bar(xx + W / 2, exp_, W, color=GREY, label="期望分數")
for i in range(2):
    ax.text(i - W / 2, real[i] + 0.002, f"{real[i]:.4f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.text(i + W / 2, exp_[i] + 0.002, f"{exp_[i]:.4f}", ha="center", va="bottom", fontsize=8, color=INK_2)
ax.set_xticks(xx); ax.set_xticklabels(["Public", "Private"]); ax.set_ylim(0.76, 0.84); set_ylabel_cjk(ax, "分數")
legend_below(ax)
save_figure(fig, os.path.join(HERE, "fig5_expected_vs_actual.png"))
print("row 1D %:", [int(round(p)) for p in P[4]], " row 2D %:", [int(round(p)) for p in P[5]], " diag %:", [int(round(P[i, i])) for i in range(10)])
for f in sorted(os.listdir(HERE)):
    if f.endswith(".png"):
        print(f, os.path.getsize(os.path.join(HERE, f)) // 1024, "KB")

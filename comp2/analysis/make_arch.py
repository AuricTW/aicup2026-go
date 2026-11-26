"""圖 1 of 貳 (comp2): system architecture, top -> bottom, shared diagram style (report_fig_style.py, orthogonal
connectors: split / merge / down; group label on the frame line).  Based on the main session's diagtest2.py.
Run: python make_arch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_fig_style import _head, apply_style, box, down, group, merge, new_diagram, save_figure, split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
apply_style()
fig, ax = new_diagram(9.4)
CX, WM, DY = 7.25, 6.2, -0.6         # main column centre and width; vertical gap between stacked boxes = 0.5 cm

a = box(ax, CX, 9.45 + DY, 4.8, 0.75, "題目：同一玩家的 G 局棋", "data")
b1 = box(ax, 2.7, 7.35 + DY, 3.7, 1.0, "HT 序列模型\n（落子似然與主幹特徵）")
b2 = box(ax, CX, 7.35 + DY, 3.7, 1.0, "對手身分證據\n（手順網路與帳號檔）")
b3 = box(ax, 11.8, 7.35 + DY, 3.7, 1.0, "第二辨識器與等級網路\n（競賽一逐手模型）")
group(ax, 0.55, 6.55 + DY, 13.95, 8.15 + DY, "每局等級證據", label_x=3.85)
s1 = box(ax, CX, 5.2 + DY, WM, 1.0, "第一階段校準\n（每種 G 的尺度與等級偏差）")
s2 = box(ax, CX, 3.8 + DY, WM, 0.8, "第二階段：與前一代系統融合")
prev = box(ax, 12.5, 3.8 + DY, 2.9, 0.8, "前一代系統後驗", "data")
ens = box(ax, CX, 2.5 + DY, WM, 0.8, "91 個成員取幾何平均")
out = box(ax, CX, 1.2 + DY, WM, 0.8, "期望得分解碼 → 預測等級", "output")

split(ax, a, [b1, b2, b3], gap=8.55 + DY)
merge(ax, [b1, b2, b3], s1, gap=6.1 + DY)
down(ax, s1, s2); down(ax, s2, ens); down(ax, ens, out)
_head(ax, (prev["l"], prev["y"]), (s2["r"], s2["y"]))      # horizontal edge-to-edge arrow into the fusion box
save_figure(fig, os.path.join(HERE, "fig_arch.png"))
print("fig_arch.png", os.path.getsize(os.path.join(HERE, "fig_arch.png")) // 1024, "KB")

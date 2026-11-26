"""Shared figure style for both AI CUP 2026 Go reports (競賽一 / 競賽二).

Run figure scripts with the shared environment, not the competition venvs:
    python your_figure.py

Usage inside a figure script:
    import sys; sys.path.insert(0, "<folder of this file>")
    from report_fig_style import apply_style, new_figure, save_figure, PALETTE
    apply_style()
    fig, ax = new_figure()          # 14.5 cm x 7 cm by default
    ...
    save_figure(fig, r"...\\fig1_xxx.png")

Conventions (keep both reports identical):
- Fonts follow the official rule: Latin, digits and symbols in Times New Roman, Chinese in 標楷體 (DFKai-SB).
- Width 14.5 cm (text width is 14.83 cm), height 6-9 cm, 300 dpi PNG.
- Text in the figure is 10 pt at final size; no figure title inside the image (the caption goes in Word).
- Colours: BLUE, ORANGE, AQUA (PALETTE) in this fixed order; BLUE is "ours / final", GREY is
  "baseline / comparison / other". Never more than 3 hues in one figure; one colour = one meaning.
- Outcomes with a good -> bad order (第一名 / 第二至五名 / 未進前五; 答對 / 差一級 / 差兩級以上) use
  OUTCOME = [green, amber, red] (traffic light), not the categorical BLUE / ORANGE / AQUA order.
  Numbers inside green / red segments in white, inside amber in INK.
- Y-axis labels in Chinese are vertical with upright characters: use set_ylabel_cjk(ax, "分數"),
  never ax.set_ylabel(...) with the default 90° rotation. X-axis labels stay horizontal.
- Legends without frames; only a horizontal y-grid; numbers on bars only where they are discussed.
- Architecture diagrams: new_diagram(), box(kind="data"|"module"|"output"), group(), and the orthogonal
  connectors split() (one -> many), merge() (many -> one), down() (one -> one). Do not draw slanted
  arrows; every connector must start and end on a box edge. The group label sits on the dashed frame
  line; choose label_x so no connector crosses it. arrow() is only for rare special cases.
  Flow runs top -> bottom (or left -> right), exactly one "output" box (blue), data boxes grey,
  modules white; text 9 pt, 1-2 short lines per box; no colour beyond these.
- Insert in Word at width 14.5 cm, centred paragraph without first-line indent,
  caption paragraph directly below: 「圖 n　說明」 centred, 12 pt, keep-with-next on the picture paragraph.
- Refer to figures in the text as 「如圖 n」. Tables, if any, get 「表 n　說明」 above the table.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CM = 1 / 2.54
WIDTH_CM = 14.5
# Categorical slots 1-3 of the dataviz reference palette (light mode). These three are validated
# all-pairs for colour-vision deficiency, so any two of them can sit side by side. Use them in this
# fixed order; never invent a 4th hue: fold extra series into grey or split the figure.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#a8a7a1"          # baseline / comparison / "other" (non-categorical neutral)
# Outcome (state) colours for results with a good / medium / bad meaning, e.g.
# 第一名 / 第二至五名 / 未進前五, or 答對 / 差一級 / 差兩級以上. Traffic-light semantics (dataviz status
# palette; green toned down at the user's request) so good vs bad reads without the legend.
# Keep the order good -> medium -> bad in stacks so green and red are never adjacent (red-green CVD).
# Never use these for identities (models / configs), and never use BLUE for an outcome.
OUTCOME_GOOD, OUTCOME_OK, OUTCOME_BAD = "#2a8f3c", "#fab219", "#d03b3b"
OUTCOME = [OUTCOME_GOOD, OUTCOME_OK, OUTCOME_BAD]
PALETTE = [BLUE, ORANGE, AQUA]
# Sequential single-hue ramp (dataviz reference blue 100 -> 700) for magnitudes such as
# confusion matrices / heatmaps. Start from white so 0 % recedes into the page.
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2 = "#0b0b0b", "#52514e"          # primary text, secondary text (values, legends)
GRID, AXIS = "#e1e0d9", "#c3c2b7"

_FONT_FILES = [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf",
               r"C:\Windows\Fonts\timesi.ttf", r"C:\Windows\Fonts\kaiu.ttf"]


def apply_style() -> None:
    for f in _FONT_FILES:
        try:
            font_manager.fontManager.addfont(f)
        except (FileNotFoundError, RuntimeError):
            pass
    plt.rcParams.update({
        "font.family": ["Times New Roman", "DFKai-SB"],   # Latin first, Chinese falls back to 標楷體
        "mathtext.fontset": "stix",                        # Times-like maths
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "patch.force_edgecolor": True,      # thin white gap between adjacent bars / stacked segments
        "patch.edgecolor": "white",
        "patch.linewidth": 0.8,
        "axes.unicode_minus": False,
        "lines.linewidth": 1.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _vertical_text(text: str) -> str:
    """Stack CJK characters one per line, upright; keep Latin words / numbers as one horizontal token."""
    import re
    tokens = re.findall(r"[A-Za-z0-9.%+\-/=]+|\S", text)
    return "\n".join(tokens)


def set_ylabel_cjk(ax, text: str, pad: float = 10) -> None:
    """Y-axis label written vertically with upright characters (中文直排), the report convention.
    Keep these labels short and mostly Chinese; put units or long Latin text in the caption instead."""
    ax.set_ylabel(_vertical_text(text), rotation=0, va="center", ha="center",
                  labelpad=pad, linespacing=1.05)


def seq_cmap():
    """Single-hue blue colormap for heatmaps / confusion matrices (never a rainbow)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("report_seq_blue", SEQ_BLUE)


def cell_text_color(value: float, vmax: float = 1.0) -> str:
    """Ink for a number printed inside a heatmap cell: white on dark cells, primary ink on light ones."""
    return "white" if value / vmax >= 0.45 else INK


# ---------------------------------------------------------------- architecture / flow diagrams
# Three box kinds, used the same way in both reports:
#   "data"   : inputs and data (light grey fill)
#   "module" : models / processing steps (white fill, grey border)
#   "output" : the final prediction (BLUE fill, white text) - exactly one per diagram
#   "group"  : an unfilled dashed frame that groups modules (e.g. 新題池 / 舊題池)
_BOX = {
    "data":   dict(fc="#f0efec", ec=AXIS, lw=0.8, color=INK),
    "module": dict(fc="white", ec="#898781", lw=0.9, color=INK),
    "output": dict(fc=BLUE, ec=BLUE, lw=0.9, color="white"),
}


def new_diagram(height_cm: float = 9.0):
    """Blank canvas in centimetres: x from 0 to 14.5, y from 0 to height_cm (origin bottom-left)."""
    fig, ax = plt.subplots(figsize=(WIDTH_CM * CM, height_cm * CM))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, WIDTH_CM); ax.set_ylim(0, height_cm); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="module", fontsize=9):
    """Rounded box centred at (x, y) in cm; returns its edges for arrow placement."""
    from matplotlib.patches import FancyBboxPatch
    s = _BOX[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=s["fc"], ec=s["ec"], lw=s["lw"]))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=s["color"], linespacing=1.15)
    return dict(l=x - w / 2, r=x + w / 2, t=y + h / 2, b=y - h / 2, x=x, y=y)


def group(ax, x0, y0, x1, y1, label, fontsize=8.5, label_x=None, side="top"):
    """Dashed frame from (x0, y0) to (x1, y1). The label sits ON the frame line (like a fieldset legend)
    with a white background, so connectors never run through it. side="top" or "bottom"; label_x is
    the label's left x in cm (default x0 + 0.3) - put it where no connector crosses the frame."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.2",
                                fc="none", ec=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1))
    lx = x0 + 0.3 if label_x is None else label_x
    ly = y1 if side == "top" else y0
    ax.text(lx, ly, label, ha="left", va="center", fontsize=fontsize, color=INK_2, zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


def arrow(ax, p, q, text=None, fontsize=8):
    """Arrow from point p=(x, y) to q=(x, y) in cm, optional small label at the midpoint."""
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9,
                                                    shrinkA=0, shrinkB=0, mutation_scale=9))
    if text:
        ax.text((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2, text, fontsize=fontsize, color=INK_2,
                ha="left", va="center")


_LINE = dict(color="#52514e", lw=0.9, solid_capstyle="butt", zorder=2)


def _head(ax, p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=_LINE["color"], lw=_LINE["lw"],
                                                    shrinkA=0, shrinkB=0, mutation_scale=9), zorder=3)


def split(ax, src, dsts, gap=None):
    """Orthogonal fan-out: from the bottom of box `src` down to a horizontal bus, then an arrow down
    into the top of every box in `dsts` (boxes as returned by box()). Use this instead of slanted arrows."""
    bus_y = gap if gap is not None else (src["b"] + max(d["t"] for d in dsts)) / 2
    xs = [d["x"] for d in dsts]
    ax.plot([src["x"], src["x"]], [src["b"], bus_y], **_LINE)
    ax.plot([min(xs + [src["x"]]), max(xs + [src["x"]])], [bus_y, bus_y], **_LINE)
    for d in dsts:
        _head(ax, (d["x"], bus_y), (d["x"], d["t"]))


def merge(ax, srcs, dst, gap=None):
    """Orthogonal fan-in: a line down from the bottom of every box in `srcs` to a horizontal bus, then
    one arrow down into the top of `dst`."""
    bus_y = gap if gap is not None else (min(s["b"] for s in srcs) + dst["t"]) / 2
    xs = [s["x"] for s in srcs]
    for s in srcs:
        ax.plot([s["x"], s["x"]], [s["b"], bus_y], **_LINE)
    ax.plot([min(xs + [dst["x"]]), max(xs + [dst["x"]])], [bus_y, bus_y], **_LINE)
    _head(ax, (dst["x"], bus_y), (dst["x"], dst["t"]))


def down(ax, a, b):
    """Straight vertical arrow from the bottom of box a to the top of box b (boxes share the same x)."""
    _head(ax, (a["x"], a["b"]), (b["x"], b["t"]))


def new_figure(height_cm: float = 7.0, ncols: int = 1, nrows: int = 1, **kw):
    return plt.subplots(nrows, ncols, figsize=(WIDTH_CM * CM, height_cm * CM), constrained_layout=True, **kw)


def save_figure(fig, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)

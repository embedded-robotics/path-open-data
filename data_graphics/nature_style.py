"""Shared matplotlib style for the PathOPEN pillar figures.

Imported by every notebook in this folder so the ten figures read as one set. Defining
this once is the point: five notebooks each setting their own rcParams drift apart, and
a reviewer notices inconsistent type sizes before they notice the result.

## Nature figure requirements encoded here

    single column   89 mm      double column  183 mm
    max height      247 mm
    type            5 pt minimum; 8 pt base used here, floor of 6 (see MIN_FONT_SIZE)
    format          vector (SVG/EPS/PDF) for line art
    rule weights    0.5-1 pt

## One figure per pillar

Nature figures are multi-panel: a paper reports one figure per result, with sub-panels
a, b, c... inside it, and grows the figure in HEIGHT rather than splitting into separate
numbered figures. So each pillar produces exactly one figure here, however many panels
it contains - Pillar 1 is Fig. 1 with panels a-g, not Figs. 2 and 3.

Width is fixed at 183 mm (double column); height varies with panel count up to 247 mm.

Sources: Nature's "Final figure submission" guidelines. Sizes are in millimetres because
that is how journals specify them; `mm()` converts to the inches matplotlib wants.

## Colour

The categorical palette is colourblind-safe (Okabe-Ito, the standard accessible set) and
also survives greyscale printing, which still matters for print proofs. Model colours are
assigned ONCE here so a given model is the same colour in every figure - a reader tracking
Patho-R1 across Figures 6, 7, 8 and 9 should not have to re-read the legend each time.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature column widths, millimetres.
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183
# A full-page figure should not exceed this or it will be scaled down and the type with it.
MAX_HEIGHT_MM = 247


def mm(*values: float) -> tuple:
    """Millimetres -> inches, for figsize."""
    return tuple(v / 25.4 for v in values)


# Okabe-Ito: eight hues distinguishable under all common colour-vision deficiencies.
OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
}

# Fixed per-model colours. Ordered by the specialization tiers the model set is built
# around, so the legend also reads as the argument: general -> medical -> pathology.
MODEL_COLORS = {
    "qwen25vl": OKABE_ITO["sky"],          # general
    "internvl8b": OKABE_ITO["blue"],       # general
    "medgemma": OKABE_ITO["yellow"],       # broad medical
    "llavamed": OKABE_ITO["orange"],       # biomedical
    "quiltllava": OKABE_ITO["purple"],     # pathology
    "pathor1": OKABE_ITO["vermillion"],    # pathology SOTA
}
MODEL_LABELS = {
    "qwen25vl": "Qwen2.5-VL-7B", "internvl8b": "InternVL3.5-8B",
    "medgemma": "MedGemma-4B", "llavamed": "LLaVA-Med-7B",
    "quiltllava": "Quilt-LLaVA-7B", "pathor1": "Patho-R1-7B",
}
# Display order = specialization gradient, which is what Pillar 4 is arguing about.
MODEL_ORDER = ["qwen25vl", "internvl8b", "medgemma", "llavamed", "quiltllava", "pathor1"]

# PathOPEN is the subject of every comparison, so it gets the one saturated colour and
# the comparators get muted greys/blues. That does the visual work of the argument
# without a caption having to say which line to look at.
DATASET_COLORS = {
    "PathOPEN": OKABE_ITO["vermillion"], "PathOPEN_OE": OKABE_ITO["vermillion"],
    "PathOPEN_CE": OKABE_ITO["vermillion"],
    "PathMMU": OKABE_ITO["sky"], "PatchVQA": OKABE_ITO["green"],
    "PathVQA": "#7F7F7F", "PathVQA_OE": "#7F7F7F", "PathVQA_CE": "#7F7F7F",
    "QuiltVQA": "#BFBFBF", "QuiltVQA_OE": "#BFBFBF", "QuiltVQA_CE": "#BFBFBF",
}

# Ordinal rubric scale {-1,0,1,2}. Diverging, because -1 ("unable to evaluate") is a
# different KIND of judgement from 0-2 and should not read as "just a lower score".
SCORE_COLORS = {2: "#1A6E5A", 1: "#7BC0A8", 0: "#F2C57C", -1: "#B33A3A"}

FIGURE_DIR = "figures"

# Hard floor for any explicit fontsize= in a panel. Nature specifies 5 pt minimum in the
# printed figure; 6 keeps annotations readable on screen at 100% too, which is how they
# are actually reviewed. `fs()` clamps rather than trusting each call site to remember.
MIN_FONT_SIZE = 6


def fs(size: float) -> float:
    """Clamp a requested font size to the readable floor."""
    return max(float(size), MIN_FONT_SIZE)


def apply_style() -> None:
    """Set rcParams. Call once at the top of a notebook."""
    mpl.rcParams.update({
        # Type: Nature asks 5-7 pt in the final figure. Helvetica first, Arial as the
        # metric-compatible fallback, DejaVu as the always-present last resort.
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        # Nature's floor is 5 pt in the FINAL printed figure, but a figure that is only
        # legible at print size is hard to review on screen, and 5 pt annotations were
        # unreadable at 100% zoom. Base is 8 pt with nothing below 6 - see MIN_FONT_SIZE.
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,

        # Rules: thin, and only the two axes that carry information.
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "grid.linewidth": 0.4,
        "grid.color": "#D9D9D9",

        # Legend: no frame, tight.
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5,

        # Output: SVG with real text, not paths, so the text stays editable and
        # searchable in the typeset PDF.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,      # TrueType, embeddable - journals reject Type 3
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name: str, formats=("svg", "pdf")) -> list:
    """Save one figure under figures/ in every requested format.

    SVG is the submission format asked for; PDF is written alongside because it is what
    LaTeX includegraphics wants and it costs nothing to emit both from the same call."""
    import os
    os.makedirs(FIGURE_DIR, exist_ok=True)
    paths = []
    for extension in formats:
        path = os.path.join(FIGURE_DIR, f"{name}.{extension}")
        fig.savefig(path, format=extension)
        paths.append(path)
    return paths


def legend_outside(ax, where: str = "right", **kwargs):
    """Place a legend OUTSIDE the axes.

    A legend drawn inside (`loc="upper right"` and friends) sits on top of the data - on
    a dense panel it covers exactly the bars or points the reader is meant to compare.
    Nature figures put keys outside the plotting area, or in a shared figure-level legend.

        where="right"   to the right of the axes, vertically centred
        where="above"   above the axes, horizontal, for 2-6 short entries
        where="below"   under the axes, horizontal
    """
    anchors = {
        "right": dict(loc="center left", bbox_to_anchor=(1.02, 0.5)),
        "above": dict(loc="lower center", bbox_to_anchor=(0.5, 1.02)),
        "below": dict(loc="upper center", bbox_to_anchor=(0.5, -0.18)),
    }
    options = dict(anchors[where], frameon=False)
    options.update(kwargs)
    return ax.legend(**options)


def shared_legend(fig, handles, labels, ncol: int = 6, y: float = 1.02, **kwargs):
    """One figure-level legend for panels that share a colour scheme.

    Preferable to repeating the same key in every panel: it recovers the space six
    per-panel legends would occupy, and reinforces that a colour means the same thing
    throughout the figure."""
    options = dict(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)
    options.update(kwargs)
    return fig.legend(handles, labels, **options)


def lift_legend_above_titles(fig, legend, axes, gap_pt: float = 3.0) -> None:
    """Raise `legend` until it clears the titles of `axes`.

    Title padding is in points and legend anchors are in figure fractions, so tuning one
    against the other by hand does not survive a change in figure size. This measures both
    after a draw and moves the legend by the shortfall, which does.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    highest = max(ax.title.get_window_extent(renderer).y1 for ax in axes)
    shortfall = highest + gap_pt * fig.dpi / 72.0 - legend_box.y0
    if shortfall <= 0:
        return
    # Convert the pixel shortfall into the figure fraction the anchor is expressed in.
    offset = shortfall / fig.get_window_extent().height
    bbox = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    legend.set_bbox_to_anchor((bbox.x0, bbox.y0 + offset, bbox.width, bbox.height),
                              transform=fig.transFigure)


def panel_label_below(ax, letter: str, dx: float = 0.5, dy: float = None,
                      gap_pt: float = 6.0):
    """Panel letter centred beneath the axes, below everything else on the x side.

    With `dy=None` (the default) the position is MEASURED rather than guessed: the label is
    placed `gap_pt` points under whichever of the x tick labels or axis title hangs lowest.
    A fixed `dy` cannot do this - the depth of that material varies per panel with rotated
    ticks and multi-line labels, so one constant either collides in some panels or leaves
    a chasm in others. Pass an explicit `dy` only to override.

    Returns the Text so a caller can feed it to `avoid_overlap`."""
    if dy is None:
        figure = ax.get_figure()
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()]
        if ax.xaxis.label.get_text():
            boxes.append(ax.xaxis.label.get_window_extent(renderer))
        axes_box = ax.get_window_extent(renderer)
        lowest = min([b.y0 for b in boxes], default=axes_box.y0)
        # Convert the lowest point into this axes' fraction coordinates.
        dy = ((lowest - gap_pt * figure.dpi / 72.0) - axes_box.y0) / axes_box.height
    return ax.text(dx, dy, letter, transform=ax.transAxes,
                   fontsize=fs(10), fontweight="bold", va="top", ha="center")


def panel_label(ax, letter: str, dx: float = -0.18, dy: float = 1.06) -> None:
    """Bold lowercase panel letter, Nature's convention (a, b, c - not A, B, C)."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def significance_stars(p: float) -> str:
    """Conventional star notation. 'n.s.' when nothing is there - stating a null result
    explicitly is better than an empty space the reader has to interpret."""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def annotate_n(ax, x, y, n: int, threshold: int = 30) -> None:
    """Sample size above a bar, greyed when below the paper's own n>=30 flag threshold.

    Several strata in sub-pillars 4b/4c fall below it, and a figure that hides that would
    be presenting 13-item cells as if they were as solid as 44-item ones."""
    ax.text(x, y, f"n={n}", ha="center", va="bottom", fontsize=MIN_FONT_SIZE,
            color="#666666" if n >= threshold else "#B33A3A")

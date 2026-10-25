"""The figure style: one font, one line-weight scale, one palette for every figure.

    import figstyle as fs
    fs.use()                      # the chosen palette (palette.json "chosen"), or FIG_PALETTE=A..D
    fig = fs.figure(height_mm=60) # full text width of the journal
    ax.plot(t, y, color=fs.PILLOW["2,8,8"], lw=fs.LW["regular"])
    fs.letter(ax, "a")            # panel letter in the fixed corner
    fs.save(fig, "figures/out/F5.pdf")

Typography follows the papers: Computer Modern (the default of the LaTeX article classes),
through matplotlib's bundled cm fonts, so no TeX installation is needed.
Sizes and line weights are set below.
"""
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import colourtools as ct  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)    # silence font-table timestamp notes
HERE = Path(__file__).resolve().parent
PALETTE = json.loads((HERE / "palette.json").read_text())

# ---------------------------------------------------------------- sizes
MM = 1 / 25.4                      # inches per mm
WIDTH_MM = 119.0                   # text width of a small-sized Springer journal
MAX_HEIGHT_MM = 195.0
DPI_COMBINATION = 600              # raster panels combined with vector lettering
FONT_PT = {"tick": 9, "label": 10, "letter": 10}   # ticks 9 pt so that log exponents are >= 6.3 pt
LW = {"hair": 0.35, "thin": 0.5, "axis": 0.6, "regular": 0.9, "heavy": 1.4}   # pt; minimum 0.3 pt
MARKER_PT = {"small": 3.0, "regular": 4.2, "large": 5.5}


def table(name):
    return np.loadtxt(HERE / "third_party" / name)


def sample(name, at):
    t = table(name)
    return tuple(float(v) for v in t[int(round(at * (len(t) - 1)))])


def neutral(key):
    """sRGB grey with the CIE L* of palette.json neutral_Lstar[key]."""
    L = PALETTE["neutral_Lstar"][key]
    Y = ((L + 16) / 116) ** 3 if L > 8 else L / 903.3
    g = float(ct.to_srgb(Y))
    return (g, g, g)


def tint(c, w=None):
    """The light tone of a hue: mixed with white (in sRGB) by palette.json tint weight."""
    w = PALETTE["tint"]["light_tone_mix_with_white"] if w is None else w
    return tuple((1 - w) * x + w for x in c)


class _State:
    key = None


PILLOW = {}
SEQ = None
GREY = {}


def use(key=None):
    """Load a palette candidate ('A'..'D'); default: FIG_PALETTE, else palette.json 'chosen'."""
    global SEQ
    key = key or os.environ.get("FIG_PALETTE") or PALETTE["chosen"]
    assert key in PALETTE["candidates"], f"no palette chosen (palette.json 'chosen' or FIG_PALETTE): {key!r}"
    cand = PALETTE["candidates"][key]
    seq = table(cand["sequential"]["table"])
    if cand["sequential"]["reverse"]:
        seq = seq[::-1]
    SEQ = ListedColormap(seq, name=f"seq_{cand['name']}")
    PILLOW.clear()
    for sig, spec in cand["pillow"].items():
        PILLOW[sig] = sample(spec["table"], spec["at"])
    GREY.clear()
    GREY.update({k: neutral(k) for k in PALETTE["neutral_Lstar"]})
    _State.key = key
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "font.size": FONT_PT["label"],
        "axes.labelsize": FONT_PT["label"],
        "xtick.labelsize": FONT_PT["tick"],
        "ytick.labelsize": FONT_PT["tick"],
        "axes.linewidth": LW["axis"],
        "axes.edgecolor": GREY["ink"],
        "axes.labelcolor": GREY["ink"],
        "xtick.color": GREY["ink"],
        "ytick.color": GREY["ink"],
        "xtick.major.width": LW["axis"],
        "ytick.major.width": LW["axis"],
        "xtick.minor.width": LW["hair"],
        "ytick.minor.width": LW["hair"],
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "lines.linewidth": LW["regular"],
        "lines.markersize": MARKER_PT["regular"],
        "patch.linewidth": LW["thin"],
        "figure.dpi": 150,
        "savefig.dpi": DPI_COMBINATION,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelpad": 3.0,
    })
    return cand


def figure(height_mm, width_mm=WIDTH_MM):
    assert width_mm <= WIDTH_MM + 1e-9 and height_mm <= MAX_HEIGHT_MM
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def letter(ax, s, dx_pt=-26, dy_pt=4):
    """Panel letter '(s)' at the fixed corner: above the top-left corner of the axes."""
    ax.annotate(f"({s})", xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, dy_pt),
                textcoords="offset points", ha="left", va="bottom", fontsize=FONT_PT["letter"],
                color=GREY["ink"])


def letter_at(fig, x, y, s):
    """Panel letter '(s)' at figure coordinates (x, y) (top-left of the letter); used where an
    axes box is shrunk by an equal aspect ratio, so that letters stay on one line."""
    fig.text(x, y, f"({s})", ha="left", va="top", fontsize=FONT_PT["letter"], color=GREY["ink"])


def decimal_log_ticks(axis, ticks=None):
    """Plain decimal labels (0.001, 0.01, 1000) on a log axis instead of powers of ten, so that no
    tick text is a 0.7-size superscript; used wherever the range is short enough to read so."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    if ticks is not None:
        axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.10f}".rstrip("0").rstrip(".")))
    axis.set_minor_formatter(NullFormatter())


METADATA_PDF = {"Creator": None, "Producer": None, "CreationDate": None}
METADATA_PNG = {"Software": None}


def save(fig, path, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = METADATA_PDF if path.suffix == ".pdf" else METADATA_PNG if path.suffix == ".png" else None
    fig.savefig(path, metadata=md, **kw)
    return path


SHADE_WEIGHT = 0.30      # at most 30% darkening by the light, on the shading pass's range


def shade(colour_png, shade_png, background=None, weight=SHADE_WEIGHT):
    """Combine a flat colour render and a shading render: colour * (1 - w + w * s), s the
    shading normalised to [0, 1] over the object; the background keeps the colour pass."""
    c = plt.imread(colour_png)[..., :3].astype(float)
    s = plt.imread(shade_png)[..., :3].mean(-1).astype(float)
    bg = c[2, 2] if background is None else np.asarray(background)
    obj = np.abs(c - bg).sum(-1) > 1e-3
    lo, hi = np.percentile(s[obj], 1), np.percentile(s[obj], 99.5)
    sn = np.clip((s - lo) / (hi - lo), 0, 1)
    out = c.copy()
    out[obj] = c[obj] * (1 - weight + weight * sn[obj])[:, None]
    return out

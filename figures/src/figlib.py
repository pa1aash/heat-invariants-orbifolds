"""Small shared helpers for the figure scripts F1.py ... F9.py (figures/src/).

Every figure script has the same interface:
    python3 figures/src/Fn.py            assert the data, then draw figures/out/Fn.pdf (+ Fn.png preview)
    python3 figures/src/Fn.py --check    assert the data only (no drawing)
The assertions run in both modes, so a figure is never drawn from numbers that failed a check.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "out"
for p in ("figures/style", "figures/src"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

import figstyle as fs  # noqa: E402


def rows(path):
    with (ROOT / path).open() as fh:
        return list(csv.DictReader(fh))


def check_only():
    return "--check" in sys.argv


def finish(fig, name, preview_dpi=200):
    """Write figures/out/<name>.pdf (vector, fonts embedded) and a PNG preview."""
    pdf = fs.save(fig, OUT / f"{name}.pdf")
    fs.save(fig, OUT / f"{name}.png", dpi=preview_dpi)
    return pdf


def grey_ramp(n, lo="light", hi="ink"):
    """n neutral greys evenly spaced in L* from neutral `lo` to neutral `hi` (rule 1 of SPEC section 4)."""
    import colourtools as ct
    import numpy as np
    L0, L1 = fs.PALETTE["neutral_Lstar"][lo], fs.PALETTE["neutral_Lstar"][hi]
    out = []
    for L in np.linspace(L0, L1, n):
        Y = ((L + 16) / 116) ** 3 if L > 8 else L / 903.3
        g = float(ct.to_srgb(Y))
        out.append((g, g, g))
    return out

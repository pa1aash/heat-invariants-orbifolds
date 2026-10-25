"""Colour arithmetic for the figure style: sRGB <-> CIELAB (D65), colour-vision-deficiency
simulation and greyscale conversion. Pure numpy; used by the style checks and the contact sheet.

sRGB transfer function and D65 matrix: IEC 61966-2-1. CIELAB: CIE 15:2004, D65 white.
CVD: Machado, Oliveira and Fernandes (2009), matrices at severity 1.0 applied to linear RGB,
read from third_party/machado2009_cvd.json (fetched by fetch_third_party.py).
Greyscale: the CIE L* of each colour, i.e. what a luminance-preserving print conversion keeps.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = _M @ np.ones(3)
_CVD = json.loads((HERE / "third_party" / "machado2009_cvd.json").read_text())


def to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(l):
    l = np.clip(np.asarray(l, float), 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def lab(c):
    """CIELAB of sRGB colour(s) in [0,1], shape (..., 3)."""
    xyz = to_linear(c) @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lightness(c):
    return lab(c)[..., 0]


def delta_e(c1, c2):
    """CIE76 colour difference."""
    return np.linalg.norm(lab(c1) - lab(c2), axis=-1)


def simulate(c, kind):
    """Dichromat view of sRGB colour(s) or an image (..., 3); kind in protanopia, deuteranopia, tritanopia."""
    M = np.array(_CVD[kind])
    return to_srgb(to_linear(np.asarray(c, float)[..., :3]) @ M.T)


def grey(c):
    """Greyscale rendering that keeps CIE L*: the sRGB grey with the same lightness."""
    L = lightness(np.asarray(c, float)[..., :3])
    Y = np.where(L > 8, ((L + 16) / 116) ** 3, L / 903.3)
    g = to_srgb(Y)
    return np.stack([g, g, g], -1)

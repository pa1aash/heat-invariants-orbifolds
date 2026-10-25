"""F4 (Section 3): how many heat coefficients are needed, against the area.

Points: for each of the 525 values s = Area/2pi <= 7/5 attained by a signature of genus <= 1 with
at most four cone points of order <= 12, the largest K_mult(O; Sig) over the complete area class
Sig(s), enumerated with no bound on the orders (figures/data/f4_area_classes.csv; recomputed
independently by checks/pairs/a1_fig2_kmult.py, no value changed). Step curves: the proven upper
bound floor(2s) + 4 (Theorem S, Corollary S2) and, from s = 4 (A >= 8 pi, the hypothesis of
the growth theorem), the square-root lower bound floor(sqrt((s - 1)/3)) + 2 of
f(A) = max_{Area <= A} K_mult (the paper Theorem 4.1). Split disc: the class s = 1/4 of
(2,8,8) and (3,3,12). Open squares: the Prouhet pairs of Theorem N(a) for L = 2..6, which share
L coefficients, so K_mult >= L + 1 there (figures/data/f4_construction.csv). Filled diamonds: the
18 checked pairs of theory/pte/data/witnesses.json (genus pairs, equal-count genus-0 pairs and
genus-0 pairs with different cone counts, L = 2..7), each sharing exactly L coefficients.

Asserted: every class value lies strictly below the upper bound; the class data lie below
s = 4, outside the range of the square-root bound; the Prouhet points sit above the earlier
logarithmic lower bound floor(log_4(s+1)) + 2 at their own s (the CSV column) and below the upper
bound; the step functions are recomputed exactly (Fraction) and equal the CSV columns; the
square-root bound is at least the logarithmic one on its whole range s >= 4 (checked at every
breakpoint up to s = 2000); every witness shares exactly L coefficients, lies below the upper
bound, and the least areas per L satisfy s < 2L - 3 for 2 <= L <= 5 and s < 10, 18 for L = 6, 7
(Example ex:ptepairs of the paper).
"""
import json
from fractions import Fraction
from math import isqrt

import numpy as np

from figlib import ROOT, check_only, finish, fs, rows


def upper(s):
    return int(2 * s) + 4


def lower_log(s):
    """floor(log_4(s+1)) + 2, the earlier bound (still the CSV column)."""
    j = 0
    while 4 ** (j + 1) <= s + 1:
        j += 1
    return j + 2


def lower_sqrt(s):
    """floor(sqrt((s - 1)/3)) + 2 for s >= 4, exactly: the largest j with 3 j^2 + 1 <= s, plus 2."""
    s = Fraction(s)
    j = isqrt(int((s - 1) / 3))
    while 3 * (j + 1) ** 2 + 1 <= s:
        j += 1
    while 3 * j ** 2 + 1 > s:
        j -= 1
    return j + 2


S_LOWER = 4                                         # the square-root bound holds for A >= 8 pi, i.e. s >= 4


def witnesses():
    w = json.load(open(ROOT / "theory/pte/data/witnesses.json"))
    assert len(w) == 18
    out = []
    for r in w:
        L, s = int(r["L"]), Fraction(r["area_over_2pi"])
        assert int(r["shares_exactly"]) == L and abs(float(s) - r["area_over_2pi_float"]) < 1e-9
        assert 0 < s and L + 1 < upper(s)
        out.append((L, s, r["kind"]))
    least = {L: min(s for LL, s, _ in out if LL == L) for L in range(2, 8)}
    for L in range(2, 6):
        assert least[L] < 2 * L - 3
    assert least[6] < 10 and least[7] < 18
    genus_sizes = sorted(int(r["T"]) for r in w if r["kind"] == "genus")
    assert genus_sizes == [6, 10, 16, 20, 26, 40]
    return out


def data():
    cls = rows("figures/data/f4_area_classes.csv")
    con = rows("figures/data/f4_construction.csv")
    assert len(cls) == 525
    s = [Fraction(r["s"]) for r in cls]
    k = [int(r["max_K_mult"]) for r in cls]
    for si, ki, r in zip(s, k, cls):
        assert upper(si) == int(r["upper_bound_floor_2s_plus_4"]) and lower_log(si) == int(r["lower_bound_floor_log4_s_plus_1_plus_2"])
        assert 1 <= ki < upper(si)
    pil = next(r for r in cls if r["s"] == "1/4")    # the pillow class: (2,8,8) ~ (3,3,12) share c_1, c_2
    assert (pil["extremal_signature"], pil["partner_sharing_K_mult_minus_1"], pil["max_K_mult"]) == ("(0;2,8,8)", "(0;3,3,12)", "3")
    assert max(s) < S_LOWER                         # the class data lie below the range of the square-root bound
    cs = [float(r["s_float"]) for r in con]
    ck = [int(r["K_mult_at_least"]) for r in con]
    for r, c, kk in zip(con, cs, ck):
        L = int(r["L"])
        assert kk == L + 1 and int(r["shared_coefficients"]) == L
        assert c < 4 ** (L - 1) - 1 + 1e-9 and lower_log(Fraction(c).limit_denominator(10 ** 6)) <= kk < upper(c)
    # the square-root bound dominates the logarithmic one wherever it applies (both are step functions;
    # checking at every breakpoint of either in [4, 2000] covers the whole range)
    bps = {Fraction(4)} | {Fraction(4 ** j - 1) for j in range(1, 7)} | {Fraction(3 * j * j + 1) for j in range(1, 30)}
    for b in sorted(x for x in bps if 4 <= x <= 2000):
        assert lower_sqrt(b) >= lower_log(b), b
    wit = witnesses()
    return np.array([float(x) for x in s]), np.array(k), np.array(cs), np.array(ck), wit


def draw(s, k, cs, ck, wit):
    fs.use()
    fig = fs.figure(56)
    ax = fig.add_axes([0.09, 0.19, 0.88, 0.77])
    x = np.logspace(np.log10(0.02), np.log10(2000), 4000)
    ax.step(x, [upper(v) for v in x], where="post", color=fs.GREY["ink"], lw=fs.LW["regular"])
    xl = x[x >= S_LOWER]
    ax.step(xl, [lower_sqrt(Fraction(v).limit_denominator(10 ** 9)) for v in xl], where="post", color=fs.GREY["ink"],
            lw=fs.LW["regular"], ls=(0, (4, 2)))
    ax.plot(s, k, ls="none", marker="o", ms=fs.MARKER_PT["small"], mfc=fs.GREY["dark"], mec="none")
    ax.plot([0.25], [3], ls="none", marker="o", ms=fs.MARKER_PT["large"], fillstyle="left", mfc=fs.PILLOW["2,8,8"],
            mfcalt=fs.PILLOW["3,3,12"], mec=fs.GREY["ink"], mew=fs.LW["hair"], zorder=5)
    ax.plot(cs, ck, ls="none", marker="s", ms=fs.MARKER_PT["regular"], mfc="white", mec=fs.GREY["ink"], mew=fs.LW["regular"])
    ws = [float(sv) for L, sv, kind in wit]
    wk = [L + 1 for L, sv, kind in wit]
    ax.plot(ws, wk, ls="none", marker="D", ms=fs.MARKER_PT["regular"], mfc=fs.GREY["ink"], mec=fs.GREY["ink"],
            mew=fs.LW["hair"], zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 2000)
    fs.decimal_log_ticks(ax.xaxis, [0.1, 1, 10, 100, 1000])
    ax.set_ylim(0, 28.5)
    ax.set_yticks(range(0, 29, 4))
    ax.set_yticks(range(0, 29, 2), minor=True)
    ax.set_xlabel(r"$s=\mathrm{Area}/2\pi$")
    ax.set_ylabel(r"$K_{\mathrm{mult}}$")
    return fig


if __name__ == "__main__":
    d = data()
    if not check_only():
        finish(draw(*d), "F4")
    print("F4: assertions passed")

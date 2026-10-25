"""F3 (Section 3): the mirror argument of Theorem S on one real axis.

For two signatures with equal area, pad the order multisets with 1s to U, V (Lemma 4) and put
Z = U* (+) (-V*) after cancelling common elements. Equal c_1, ..., c_L means
R(U*) = R(V*) and P_j(U*) = P_j(V*) for odd j <= 2L-3, i.e. sum_{z in Z} z^j = 0 for those j and
sum 1/z = 0. If |Z| <= 2L these force Q(z) = prod (z - x) to be even, so Z is symmetric under
z -> -z; since U*, V* are disjoint, both are empty and the signatures coincide.

Filled discs: Z (positive side U*, negative side -V*; stacked for multiplicity). Open rings: the
mirror image -Z. Symmetry means every ring sits on a disc.
(a) top:    (2,8,8) against (3,3,12): Z = {2,8,8,-3,-3,-12}, |Z| = 6, they share exactly
            2 = |Z|/2 - 1 coefficients; no ring sits on a disc.
    bottom: the configuration the theorem forces once |Z| <= 2L: Z = -Z (shown for (2,8,8)
            against itself, before cancellation), every ring on a disc.
(b) (1;15) against (0;3,3,5,5): U = {1,15} (one padding 1), V = {3,3,5,5}; the sides differ by
    |V| - |U| = 2 = 2(g - g'), the genus difference; they share exactly 2 coefficients.

Asserted with theory/signatures/sig_common.py: the number of shared coefficients (from the
actual cone coefficients b_l), the padding, the vanishing sums of 1/z and z over Z, and the
first non-vanishing odd power sum.
"""
from collections import Counter
from fractions import Fraction as F

import numpy as np

from common_import import import_from
from figlib import check_only, finish, fs


def Z_of(sig1, sig2):
    (g1, m1), (g2, m2) = sig1, sig2
    d = sum(F(1, x) for x in m2) - sum(F(1, x) for x in m1)
    assert d.denominator == 1
    d = int(d)
    U = Counter(m1 + (1,) * max(d, 0))
    V = Counter(m2 + (1,) * max(-d, 0))
    Us, Vs = U - V, V - U
    return sorted(Us.elements()), sorted(Vs.elements()), sum(U.values()), sum(V.values())


def data():
    sc, _ = import_from("theory/signatures", "sig_common")
    cases = {}
    for key, a, b, shared in (("pillows", (0, (2, 8, 8)), (0, (3, 3, 12)), 2),
                              ("genus", (1, (15,)), (0, (3, 3, 5, 5)), 2)):
        assert sc.s_of(*a) == sc.s_of(*b)
        assert sc.shared(a, b, 6) == shared, key
        Us, Vs, nU, nV = Z_of(a, b)
        Z = [F(x) for x in Us] + [-F(x) for x in Vs]
        assert sum(1 / z for z in Z) == 0 and sum(Z) == 0            # c_1 and c_2 agree
        assert sum(z ** 3 for z in Z) != 0                            # c_3 does not
        assert len(Z) == 2 * shared + 2                               # Theorem S with equality
        assert nV - nU == 2 * (a[0] - b[0])
        cases[key] = (Us, Vs)
    assert cases["pillows"] == ([2, 8, 8], [3, 3, 12]) and cases["genus"] == ([1, 15], [3, 3, 5, 5])
    return cases


def row(ax, pos, neg, cpos, cneg, lim):
    """Discs at pos and -neg (stacked), rings at the mirror images."""
    ax.axhline(0, color=fs.GREY["ink"], lw=fs.LW["axis"], zorder=1)
    Z = [(x, cpos) for x in pos] + [(-x, cneg) for x in neg]
    seen = Counter()
    for z, c in Z:
        h = 0.55 + 0.75 * seen[z]
        seen[z] += 1
        ax.plot([z], [h], ls="none", marker="o", ms=fs.MARKER_PT["large"], mfc=c, mec=fs.GREY["ink"], mew=fs.LW["hair"], zorder=4)
    seen = Counter()
    for z, _ in Z:
        h = 0.55 + 0.75 * seen[-z]
        seen[-z] += 1
        ax.plot([-z], [h], ls="none", marker="o", ms=fs.MARKER_PT["large"] + 2.6, mfc="none", mec=fs.GREY["mid"],
                mew=fs.LW["regular"], zorder=5)
    for x in sorted({abs(z) for z, _ in Z} | {0}):
        for s in (x, -x):
            ax.plot([s, s], [-0.18, 0.18], color=fs.GREY["ink"], lw=fs.LW["axis"], zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.4, 2.3)
    ax.set_yticks([])
    for sp in ("left", "top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    marks = sorted({s for z, _ in Z for s in (z, -z)} | {0})
    ax.set_xticks(marks)
    crowded = any(abs(z) == 1 for z, _ in Z)          # keep the tick at 0 but not its label next to +-1
    labels = ax.set_xticklabels(["" if (m == 0 and crowded) else (f"${m}$" if m >= 0 else f"$-{-m}$") for m in marks])
    # labels of ticks one unit apart (-3, -2 and 2, 3 in (a)) would touch: align the left one to the
    # right of its tick and the right one to the left
    for i in range(len(marks) - 1):
        if marks[i + 1] - marks[i] <= 1 and labels[i].get_text() and labels[i + 1].get_text():
            labels[i].set_horizontalalignment("right")
            labels[i + 1].set_horizontalalignment("left")
    ax.tick_params(axis="x", length=0, pad=1)


def draw(cases):
    fs.use()
    fig = fs.figure(78)
    A, B = fs.PILLOW["2,8,8"], fs.PILLOW["3,3,12"]
    ax1 = fig.add_axes([0.04, 0.73, 0.94, 0.23])
    row(ax1, *cases["pillows"], A, B, 13)
    ax2 = fig.add_axes([0.04, 0.44, 0.94, 0.23])
    row(ax2, [2, 8, 8], [2, 8, 8], A, A, 13)
    ax3 = fig.add_axes([0.04, 0.07, 0.94, 0.23])
    row(ax3, *cases["genus"], fs.GREY["ink"], fs.GREY["mid"], 16)
    for ax in (ax1, ax2):
        ax.tick_params(labelbottom=True)
    fs.letter_at(fig, 0.005, 0.99, "a")
    fs.letter_at(fig, 0.005, 0.355, "b")
    return fig


if __name__ == "__main__":
    d = data()
    if not check_only():
        finish(draw(d), "F3")
    print("F3: assertions passed")

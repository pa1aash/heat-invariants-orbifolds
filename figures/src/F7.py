"""F7 (Section 5): the least-order strata in the (S, R) plane drift together.

For p = 2..8 the curves are the endpoints of the reciprocal sums R = 1/p + 1/q + 1/r over the
hyperbolic triads of sum S and least order p: R^-_{S,p} (balanced) and R^+_{S,p} (spread; for
p = 2 the largest hyperbolic value), as polylines over integer S <= 120 (log S axis),
with a light fill of the same grey between them
(figures/data/f7_strata.csv). Strata are an ordered category: neutral grey ramp, lighter for
smaller p (SPEC section 4, rule 1). Filled ink dots: the first overlap of strata p, p+1, at
(S*(p), R^+_{S*,p+1}), joined by a thin dashed line to the open circle of the first actual
collision of the same pair (figures/data/f7_overlap.csv). Log axes in S and R. The split disc at (18, 3/4) is (2,8,8) ~ (3,3,12).

Asserted: at S*(p) the lower end of stratum p is <= the upper end of stratum p+1 and at S*(p)-1
(same parity or not) it is not; every first collision lies inside both intervals and at or after
S*(p); the S = 18 collision is the pillow pair with R = 3/4.
"""
from fractions import Fraction as F

import numpy as np

from figlib import check_only, finish, fs, grey_ramp, rows

S_MAX = 120


def data():
    st = rows("figures/data/f7_strata.csv")
    ov = rows("figures/data/f7_overlap.csv")
    iv = {(int(r["S"]), int(r["p"])): (F(r["R_min"]), F(r["R_max"])) for r in st}
    over = []
    for o in ov:
        p, Ss, S1, R1 = int(o["p"]), int(o["S_star"]), int(o["first_collision_S"]), F(o["first_collision_R"])
        assert iv[(Ss, p)][0] <= iv[(Ss, p + 1)][1]
        if (Ss - 1, p + 1) in iv:
            assert iv[(Ss - 1, p)][0] > iv[(Ss - 1, p + 1)][1]
        assert S1 >= Ss
        for q in (p, p + 1):
            assert iv[(S1, q)][0] <= R1 <= iv[(S1, q)][1]
        over.append((p, Ss, float(iv[(Ss, p + 1)][1]), S1, float(R1)))
    assert over[0][3:] == (18, 0.75) and ov[0]["triads"] == "(2, 8, 8) ; (3, 3, 12)"
    strata = {}
    for (S, p), (lo, hi) in sorted(iv.items()):
        strata.setdefault(p, []).append((S, float(lo), float(hi)))
    return strata, over


def draw(strata, over):
    fs.use()
    fig = fs.figure(72)
    ax = fig.add_axes([0.12, 0.14, 0.85, 0.82])
    ps = sorted(strata)
    greys = grey_ramp(len(ps), lo="light", hi="ink")
    for p, g in zip(ps, greys):                      # light fill per stratum, smaller p on top
        S, lo, hi = (np.array(v) for v in zip(*strata[p]))
        ax.fill_between(S, lo, hi, color=fs.tint(g, 0.72), lw=0, zorder=1 + (ps[-1] - p) * 0.01)
    for p, g in zip(ps, greys):
        S, lo, hi = (np.array(v) for v in zip(*strata[p]))
        ax.plot(S, lo, color=g, lw=fs.LW["regular"], zorder=2)
        ax.plot(S, hi, color=g, lw=fs.LW["regular"], zorder=2)
    gp = dict(zip(ps, greys))
    for p, Ss, Rs, S1, R1 in over:             # overlap dot joined to the collision circle of the same pair
        ax.plot([Ss, S1], [Rs, R1], color=gp[p + 1] if p + 1 in gp else fs.GREY["ink"], lw=fs.LW["thin"],
                ls=(0, (2, 1.5)), zorder=3)
        ax.plot([Ss], [Rs], ls="none", marker="o", ms=fs.MARKER_PT["regular"], mfc=fs.GREY["ink"], mec="white",
                mew=fs.LW["hair"], zorder=4)
        if p > 2:
            ax.plot([S1], [R1], ls="none", marker="o", ms=fs.MARKER_PT["regular"], mfc="white", mec=fs.GREY["ink"],
                    mew=fs.LW["regular"], zorder=5)
    ax.plot([18], [0.75], ls="none", marker="o", ms=fs.MARKER_PT["large"], fillstyle="left", mfc=fs.PILLOW["2,8,8"],
            mfcalt=fs.PILLOW["3,3,12"], mec=fs.GREY["ink"], mew=fs.LW["hair"], zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(10, 126)
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.set_ylim(0.15, 1.0)
    ax.set_yticks([0.2, 0.3, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.2", "0.3", "0.5", "0.75", "1"])
    ax.minorticks_off()
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$R$")
    return fig


if __name__ == "__main__":
    d = data()
    if not check_only():
        finish(draw(*d), "F7")
    print("F7: assertions passed")

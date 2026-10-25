"""F5 (Sections 4 and 7): two timescales on one t axis.

(a) D(t) = Z_{(2,8,8)}(t) - Z_{(3,3,12)}(t) computed from the two spectra (ink, heavy), the
    cone-point (elliptic) contribution of the trace formula (light grey, dashed, drawn on top),
    and the partial sums of its small-t expansion d_3 t, d_3 t + d_4 t^2,
    d_3 t + d_4 t^2 + d_5 t^3 (grey ramp, light to dark, thin), from
    numerics/data/heat_trace_difference.csv. The computed D follows the cone terms until the
    closed geodesics take over and changes sign near t = 0.34; the cone-term sum stays positive.
(b) The moduli family (0;3,3,3,3), modulus theta: |Z_0 - Z_theta|(t) for theta = 0.4, ..., 2.8
    (grey ramp, light to dark: shorter systole is darker), the predicted geodesic difference
    |H_0 - H_theta| (thin ink, dashed), and the largest error budget of these pairs (ink,
    dotted), from numerics/moduli/data/trace_differences.csv. Open circles: the predicted
    emergence time t* of each pair (numerics/moduli/data/summary.json).

Asserted: d_3, d_4, d_5 = 25/12, -1775/24, 153025/48 from theory/stability/stab_common.heat_direct;
the CSV prediction columns equal d_3 t and d_3 t + d_4 t^2; |D - cone sum| < 1e-11 for t <= 0.03;
D changes sign exactly once, between 0.3 and 0.4, while the cone sum stays positive;
for every moduli pair |D - (H_i - H_j)| <= error budget everywhere; the emergence times in
summary.json are measured = predicted and decrease with the systole.
"""
import json
from fractions import Fraction as Fr

import numpy as np

from common_import import import_from
from figlib import ROOT, check_only, finish, fs, grey_ramp, rows

THETAS = ["0.4", "0.8", "1.2", "1.6", "2.0", "2.4", "2.8"]


def data():
    sc, _ = import_from("theory/stability", "stab_common")
    H = {m: sc.heat_direct(m, 5) for m in ((2, 8, 8), (3, 3, 12))}
    d = [a - b for a, b in zip(H[(2, 8, 8)], H[(3, 3, 12)])]
    assert d[:2] == [0, 0] and d[2:] == [Fr(25, 12), Fr(-1775, 24), Fr(153025, 48)]
    d3, d4, d5 = (float(x) for x in d[2:])
    r = rows("numerics/data/heat_trace_difference.csv")
    t = np.array([float(x["t"]) for x in r])
    D = np.array([float(x["D"]) for x in r])
    ell = np.array([float(x["pred_exact_elliptic_difference"]) for x in r])
    assert np.allclose([float(x["pred_c1_t"]) for x in r], d3 * t, rtol=1e-6)
    assert np.allclose([float(x["pred_c1_t_plus_c2_t2"]) for x in r], d3 * t + d4 * t ** 2, rtol=1e-6, atol=1e-12)
    assert np.all(np.abs(D - ell)[t <= 0.03] < 1e-11)
    sgn = np.flatnonzero(np.diff(np.sign(D)))
    assert len(sgn) == 1 and 0.3 < t[sgn[0]] < 0.4 and np.all(ell > 0)
    partial = [d3 * t, d3 * t + d4 * t ** 2, d3 * t + d4 * t ** 2 + d5 * t ** 3]

    m = rows("numerics/moduli/data/trace_differences.csv")
    summ = json.loads((ROOT / "numerics/moduli/data/summary.json").read_text())["pairs"]
    pairs = {}
    tstar_prev = 1.0
    for th in THETAS:
        sel = [x for x in m if x["tau_i"] == "0.0" and x["tau_j"] == th]
        tt = np.array([float(x["t"]) for x in sel])
        Dm = np.array([float(x["D_measured"]) for x in sel])
        Hp = np.array([float(x["D_predicted_Hi_minus_Hj"]) for x in sel])
        B = np.array([float(x["error_budget"]) for x in sel])
        assert np.all(np.abs(Dm - Hp) <= B)
        s = summ[f"0.0-{th}"]
        assert s["emergence_measured"] == s["emergence_predicted"] < tstar_prev
        tstar_prev = s["emergence_predicted"]
        pairs[th] = (tt, Dm, Hp, B, s["emergence_predicted"])
    return (t, D, ell, partial), pairs


def draw(a, pairs):
    fs.use()
    fig = fs.figure(108)
    t, D, ell, partial = a
    xl = (1.5e-3, 0.5)
    ax = fig.add_axes([0.13, 0.56, 0.84, 0.41])
    ax.axhline(0, color=fs.GREY["light"], lw=fs.LW["thin"], zorder=0)
    for y, g in zip(partial, grey_ramp(3, lo="light", hi="dark")):
        ax.plot(t, y, color=g, lw=fs.LW["regular"])
    ax.plot(t, D, color=fs.GREY["ink"], lw=fs.LW["heavy"])
    ax.plot(t, ell, color=fs.GREY["light"], lw=fs.LW["regular"], ls=(0, (3, 2)), zorder=4)   # on top of D
    ax.set_xscale("log")
    ax.set_xlim(*xl)
    fs.decimal_log_ticks(ax.xaxis, [0.01, 0.1])
    ax.set_ylim(-0.03, 0.09)
    ax.set_ylabel(r"$D(t)$")

    bx = fig.add_axes([0.13, 0.09, 0.84, 0.41])
    greys = grey_ramp(len(THETAS), lo="light", hi="ink")
    Bmax = None
    for th, g in zip(THETAS, greys):
        tt, Dm, Hp, B, ts = pairs[th]
        bx.plot(tt, np.abs(Hp), color=fs.GREY["ink"], lw=fs.LW["hair"], ls=(0, (3, 2)), zorder=2)
        Dz = np.where(Dm == 0, np.nan, np.abs(Dm))      # exact zeros (no noise at all) are left out of the log plot
        bx.plot(tt, Dz, color=g, lw=fs.LW["regular"], zorder=3)
        k = np.argmin(np.abs(tt - ts))
        bx.plot([tt[k]], [abs(Dm[k])], ls="none", marker="o", ms=fs.MARKER_PT["small"], mfc="white",
                mec=fs.GREY["ink"], mew=fs.LW["thin"], zorder=5)
        Bmax = B if Bmax is None else np.maximum(Bmax, B)
    bx.fill_between(pairs[THETAS[0]][0], 1e-16, Bmax, color=fs.GREY["faint"], lw=0, zorder=0)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(*xl)
    fs.decimal_log_ticks(bx.xaxis, [0.01, 0.1])
    bx.set_ylim(1e-15, 3)
    bx.set_yticks([1e-15, 1e-10, 1e-5, 1])
    bx.set_xlabel(r"$t$")
    bx.set_ylabel(r"$|Z_0(t)-Z_\vartheta(t)|$")
    fs.letter_at(fig, 0.005, 0.995, "a")
    fs.letter_at(fig, 0.005, 0.525, "b")
    return fig


if __name__ == "__main__":
    d = data()
    if not check_only():
        finish(draw(*d), "F5")
    print("F5: assertions passed")

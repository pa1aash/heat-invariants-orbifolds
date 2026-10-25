"""F8 (Section 6): recovery error of the cone orders against the error in the heat coefficients.

Data: figures/data/f8_recovery.csv (gen_f8_recovery.py). Curves, log-log:
  (2,3,7), general data (worst sign pattern of an error delta on every coefficient)  ink, solid
  (2,8,8), general data                                                             dark pillow hue, solid
  (4,4,4), general data                                                             ink, dashed
  (4,4,4), realisable data (heat data of the real multisets (4+s, 4, 4-s))          ink, dotted
The certified threshold delta_cert of each multiset (figures/data/f8_thresholds.csv, from
theory/stability/threshold_results.json) is an open diamond on that multiset's general-data curve.
The faint horizontal line is the rounding limit 1/2.

Asserted: the fitted slopes on delta <= 1e-8 are 1, 1/2, 1/3, 1/2 within 0.01; every point with
delta <= delta_cert has error < 1/2; the thresholds equal threshold_results.json.
"""
import json
import math

import numpy as np

from figlib import ROOT, check_only, finish, fs, rows

CURVES = [("simple", "general", 1.0), ("double", "general", 0.5), ("triple", "general", 1 / 3), ("triple", "realisable", 0.5)]


def data():
    r = rows("figures/data/f8_recovery.csv")
    thr = {t["case"]: (t["orders"], float(t["delta_cert"])) for t in rows("figures/data/f8_thresholds.csv")}
    res = json.loads((ROOT / "theory/stability/threshold_results.json").read_text())
    for case, (orders, dc) in thr.items():
        assert dc == res["(" + orders.strip("()").replace(",", ", ") + ")"]["delta_cert"]
    out = {}
    for case, kind, slope in CURVES:
        pts = sorted((float(x["delta"]), float(x["recovery_error"])) for x in r if x["case"] == case and x["kind"] == kind)
        d, e = np.array(pts).T
        sel = d <= 1e-8
        k = np.polyfit(np.log10(d[sel]), np.log10(e[sel]), 1)[0]
        assert abs(k - slope) < 0.01, (case, kind, k)
        if kind == "general":
            assert np.all(e[d <= thr[case][1]] < 0.5)
        out[(case, kind)] = (d, e)
    return out, thr


def draw(curves, thr):
    fs.use()
    fig = fs.figure(72)
    ax = fig.add_axes([0.13, 0.14, 0.84, 0.82])
    style = {("simple", "general"): dict(color=fs.GREY["ink"], ls="-"),
             ("double", "general"): dict(color=fs.PILLOW["2,8,8"], ls="-"),
             ("triple", "general"): dict(color=fs.GREY["ink"], ls=(0, (5, 2))),
             ("triple", "realisable"): dict(color=fs.GREY["ink"], ls=(0, (1, 1.6)))}
    ax.axhline(0.5, color=fs.GREY["light"], lw=fs.LW["thin"], zorder=0)
    for key, (d, e) in curves.items():
        ax.plot(d, e, lw=fs.LW["heavy"] if key[0] == "double" else fs.LW["regular"], **style[key])
        if key[1] == "general":
            dc = thr[key[0]][1]
            ec = 10 ** np.interp(math.log10(dc), np.log10(d), np.log10(e))
            ax.plot([dc], [ec], ls="none", marker="D", ms=fs.MARKER_PT["regular"], mfc="white",
                    mec=style[key]["color"], mew=fs.LW["regular"], zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-14, 1e-1)
    ax.set_ylim(1e-14, 10)
    ax.set_xticks([1e-14, 1e-11, 1e-8, 1e-5, 1e-2])
    ax.set_yticks([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\Vert \tilde m - m\Vert_\infty$")
    return fig


if __name__ == "__main__":
    d = data()
    if not check_only():
        finish(draw(*d), "F8")
    print("F8: assertions passed")

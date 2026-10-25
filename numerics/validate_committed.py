"""Validation of the committed numerics data.  Every check is an assert; nothing is solved
and nothing is written.

numerics/validate.py re-derives the data tables from the raw solver runs (numerics/runs/,
not committed) and the fetched Strohmaier-Uski list (numerics/refs_cache/, not committed),
and it rewrites data/*.csv.  This script instead reads only what is committed under
data/ and recomputes every derived quantity from it, so it needs numpy, scipy, mpmath and
sympy but neither NGSolve nor the network, and it cannot overwrite committed data.

  (0) geometry of the three triangles; the exact Selberg / heat-coefficient cross-checks of
      theory.py
  (a) eigenvalues_<p-q-r>_<bc>.csv: shape, ordering, lambda_0 = 0 (Neumann), the two error
      columns recomputed from the level columns, the 1e-8 target for the first 500, and
      agreement with convergence.csv
  (b) bolza_benchmark.csv: every row recomputed from its two eigenvalue columns, below 2e-10
  (c) weyl.csv recomputed from the production eigenvalues, with the validate.py asserts
  (d) heat_trace_checks.csv: the pillow traces, the Selberg identity + elliptic residuals
      and the mirror term, recomputed and compared with the committed rows
  (e) heat_trace_difference.csv, fits.csv and headline.json: D(t) = Z_(2,8,8) - Z_(3,3,12),
      the fit scan and the blind headline, recomputed and compared; c1, c2 within 2 sigma of
      the exact predictions

usage: python validate_committed.py [--quick]     (--quick skips (d) and (e): about 1 minute;
                                                   the whole script takes a few minutes)
"""

import csv
import json
import os
import sys

import mpmath as mp
import numpy as np

import theory
from eigdata import table_committed
from geometry import Triangle, PILLOWS, BENCH
from heat_trace import (pillow_trace, selberg_smooth_plus_elliptic, weyl_terms, fit_scan, headline,
                        elliptic_difference, T_GRID, WINDOWS, ORDERS)
from solve import PROBLEMS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def rows_of(name):
    return list(csv.DictReader(open(os.path.join(DATA, name))))


def check_geometry():
    for pqr in PILLOWS + [BENCH]:
        Triangle(*pqr).verify(tol=1e-12)
    assert theory.check_against_paper()
    theory.check_elliptic_expansion(numax=3)
    from fractions import Fraction as Fr
    c = theory.predicted_difference(3)
    assert c[-1] == 0 and c[0] == 0
    assert c[1] == Fr(25, 12) and c[2] == Fr(-1775, 24)
    print("(0) geometry and exact coefficient cross-checks: OK")


def check_convergence():
    conv = rows_of("convergence.csv")
    for pqr, bc in PROBLEMS:
        tag = "-".join(map(str, pqr))
        rows = rows_of(f"eigenvalues_{tag}_{bc}.csv")
        T = table_committed(pqr, bc)
        lam, err, errc = T["lam"], T["err"], T["err_cons"]
        l12, l10 = T["all"][1], T["all"][2]
        n = len(lam)
        assert n >= 1400, (pqr, bc, n)
        assert [int(r["index"]) for r in rows] == list(range(n))
        assert np.all(np.diff(lam) >= 0), (pqr, bc)
        if bc == "N":
            assert lam[0] == 0.0 and err[0] == 0.0
        else:
            assert lam[0] > 1.0
        # the error columns are the documented differences of the level columns (floor 1e-14 lam),
        # printed with four significant digits
        floor = 1e-14 * np.maximum(np.abs(lam), 1.0)
        e_ref = np.maximum(np.abs(lam - l12), floor)
        ec_ref = np.maximum(np.abs(lam - l10), floor)
        # (the level columns carry 15 significant digits, so the recomputed differences are good to
        # about 1e-14 lambda in absolute terms)
        slack = 1.2e-14 * np.maximum(np.abs(lam), 1.0)
        assert np.all(np.abs(err[1:] - e_ref[1:]) <= 1e-3 * e_ref[1:] + slack[1:]), (pqr, bc)
        assert np.all(np.abs(errc[1:] - ec_ref[1:]) <= 1e-3 * ec_ref[1:] + slack[1:]), (pqr, bc)
        rel = err / np.maximum(lam, 1)
        worst500 = float(np.max(rel[:500]))
        worst500_cons = float(np.max((errc / np.maximum(lam, 1))[:500]))
        assert worst500 < 1e-8, (pqr, bc, worst500)
        assert worst500_cons < 5e-8, (pqr, bc, worst500_cons)
        # convergence.csv carries the same production eigenvalues, error and (p10 - p12) difference
        mine = [r for r in conv if r["triangle"] == tag and r["bc"] == bc]
        assert len(mine) == n, (pqr, bc, len(mine), n)
        lam_c = np.array([float(r["lambda_prod_h0.05_p10"]) for r in mine])
        assert np.array_equal(lam_c, lam), (pqr, bc)
        err_c = np.array([float(r["err_estimate"]) for r in mine])
        assert np.all(np.abs(err_c - err) <= 1e-2 * err + 1e-18), (pqr, bc)   # printed with 3 digits
        d1012 = np.array([float(r["diff_h0.07_p10_vs_p12"]) for r in mine])
        ref = np.abs(l10 - l12)
        assert np.all(np.abs(d1012 - ref) <= 1e-2 * ref + slack), (pqr, bc)
        print(f"(a) {pqr} {bc}: n={n}, lambda_max={lam[-1]:.1f}, worst rel err (first 500) {worst500:.1e}")


def check_bolza():
    rows = rows_of("bolza_benchmark.csv")
    mp.mp.dps = 50
    key = [k for k in rows[0] if k.startswith("lambda_fem")][0]
    worst = 0.0
    counts = {}
    for r in rows:
        x = mp.mpf(r[key])
        y = mp.mpf(r["lambda_bolza_strohmaier_uski"])
        rel = abs(x - y) / y
        assert rel < 2e-10, (r["bc"], r[key], r["lambda_bolza_strohmaier_uski"])
        assert abs(float(rel) - float(r["rel_diff"])) <= 0.1 * float(r["rel_diff"]) + 1e-15, r
        assert int(r["bolza_multiplicity"]) == 1
        worst = max(worst, float(rel))
        counts[r["bc"]] = counts.get(r["bc"], 0) + 1
    assert set(counts) == {"N", "D", "M1", "M2"}, counts
    assert sum(counts.values()) == len(rows) == 42, counts
    vals = np.array([float(r[key]) for r in rows])
    assert len(np.unique(np.round(vals, 7))) == len(vals), "a Bolza eigenvalue is matched twice"
    print(f"(b) Bolza benchmark: {len(rows)} multiplicity-one eigenvalues, worst rel diff {worst:.1e}, {counts}")


def check_weyl():
    committed = rows_of("weyl.csv")
    for pqr, bc in PROBLEMS:
        tag = "-".join(map(str, pqr))
        lam = table_committed(pqr, bc)["lam"]
        a, b, c0 = weyl_terms(pqr, bc)
        n = len(lam)
        mid = 0.5 * (lam[:-1] + lam[1:])
        N = np.arange(1, n)
        W1 = a * mid
        W3 = W1 + b * np.sqrt(mid) + c0
        r1, r3 = N - W1, N - W3
        win = [(s, min(s + 100, n - 1)) for s in range(0, n - 1, 100)]
        m3 = [float(np.mean(r3[s:e])) for s, e in win]
        m1 = [float(np.mean(r1[s:e])) for s, e in win]
        assert abs(np.mean(r3)) < 0.5, (pqr, bc, np.mean(r3))
        assert max(abs(x) for x in m3) < 1.5, (pqr, bc, m3)
        assert abs(m1[-1]) > 5 * max(abs(x) for x in m3[-3:]), (pqr, bc)
        mine = [r for r in committed if r["triangle"] == tag and r["bc"] == bc]
        assert len(mine) == len(win), (pqr, bc, len(mine), len(win))
        for r, (s, e), x1, x3 in zip(mine, win, m1, m3):
            assert (int(r["i_from"]), int(r["i_to"])) == (s, e)
            assert abs(float(r["mean_N_minus_area_term"]) - x1) < 2e-4, (pqr, bc, s)
            assert abs(float(r["mean_N_minus_area_boundary_constant"]) - x3) < 2e-4, (pqr, bc, s)
        print(f"(c) {pqr} {bc}: N - Weyl(3 terms) mean {np.mean(r3):+.3f}; matches weyl.csv")


def check_heat_traces():
    committed = rows_of("heat_trace_checks.csv")
    t = np.geomspace(0.0015, 0.05, 60)
    for pqr in PILLOWS:
        tag = "-".join(map(str, pqr))
        P = pillow_trace(pqr, t)
        Z, bound = P["Z"], P["err_eig"] + P["tail"]
        pc = theory.pillow_coeffs(pqr, 8)
        area = float(pc["area_over_pi"]) * np.pi
        a0 = float(pc["coef"][0])
        a1 = float(pc["coef"][1])
        sel = t <= 0.008
        y = Z[sel] - area / (4 * np.pi * t[sel])
        V = np.vander(t[sel], 8, increasing=True)
        co, *_ = np.linalg.lstsq(V, y, rcond=None)
        V2 = np.column_stack([1 / t[sel], V])
        co2, *_ = np.linalg.lstsq(V2, Z[sel], rcond=None)
        assert abs(co[0] - a0) < 1e-6, (pqr, co[0], a0)
        assert abs(co2[0] * 4 * np.pi - area) < 1e-6, (pqr, co2[0] * 4 * np.pi, area)
        assert abs(co2[1] - a0) < 1e-4, (pqr, co2[1], a0)
        assert abs(co[1] - a1) / abs(a1) < 1e-3, (pqr, co[1], a1)
        S = selberg_smooth_plus_elliptic(pqr, t)
        resid = Z - S
        small = t <= 0.03
        assert np.all(np.abs(resid[small]) <= bound[small] + 1e-11), (pqr, np.max(np.abs(resid[small]) - bound[small]))
        L = float(Triangle(*pqr).perimeter)
        mirror = L * np.exp(-t / 4) / (4 * np.sqrt(np.pi * t))
        mres = (P["N"]["Z"] - P["D"]["Z"]) - mirror
        msmall = t <= 0.012
        assert np.all(np.abs(mres[msmall]) <= bound[msmall] + 1e-11), (pqr, np.max(np.abs(mres[msmall])))
        mine = [r for r in committed if r["pillow"] == tag]
        assert len(mine) == len(t)
        Zc = np.array([float(r["Z_fem"]) for r in mine])
        Sc = np.array([float(r["Z_selberg_identity_plus_elliptic"]) for r in mine])
        assert np.all(np.abs(Zc - Z) <= 1e-11 * np.abs(Z)), (pqr, np.max(np.abs(Zc - Z)))
        assert np.all(np.abs(Sc - S) <= 1e-11 * np.abs(S)), (pqr, np.max(np.abs(Sc - S)))
        print(f"(d) {pqr}: area fit {co2[0] * 4 * np.pi:.10f} (exact {area:.10f}); a0 fit {co[0]:.8f} "
              f"(exact {a0:.8f}); Selberg and mirror residuals within the error bounds; matches heat_trace_checks.csv")


def check_difference_and_fits():
    t = T_GRID
    Za = pillow_trace(PILLOWS[0], t)
    Zb = pillow_trace(PILLOWS[1], t)
    D = Za["Z"] - Zb["Z"]
    Derr = Za["err_eig"] + Zb["err_eig"] + Za["tail"] + Zb["tail"]
    committed = rows_of("heat_trace_difference.csv")
    assert len(committed) == len(t)
    assert np.all(np.abs(np.array([float(r["t"]) for r in committed]) - t) <= 1e-7 * t)
    Dc = np.array([float(r["D"]) for r in committed])
    # D is a difference of two traces of size ~85; both are rebuilt from the printed eigenvalues
    assert np.all(np.abs(Dc - D) <= 2e-12 + 1e-9 * np.abs(D)), np.max(np.abs(Dc - D))
    Eex = elliptic_difference(t)
    small = t <= 0.03
    assert float(np.max(np.abs(D[small] - Eex[small]))) < 1e-11
    c = theory.predicted_difference(8)
    c1, c2 = float(c[1]), float(c[2])
    rows = fit_scan(t, D, Derr, WINDOWS, ORDERS)
    fits = rows_of("fits.csv")
    assert len(fits) == len(rows), (len(fits), len(rows))
    for r, f in zip(rows, fits):
        assert (float(f["t_min"]), float(f["t_max"]), int(f["n_coeffs"])) == (r["ta"], r["tb"], r["order"])
        assert abs(float(f["c1_fit"]) - r["c1"]) <= 1e-6 * abs(r["c1"]) + 1e-9, (r["ta"], r["tb"], r["order"])
        assert abs(float(f["c2_fit"]) - r["c2"]) <= 1e-4 * abs(r["c2"]) + 1e-6, (r["ta"], r["tb"], r["order"])
    head = json.load(open(os.path.join(DATA, "headline.json")))
    for key, pred in (("c1", c1), ("c2", c2)):
        unc, win, n, val, derr, chg = headline(rows, key)[0]
        h = head[key]
        assert list(win) == h["window"] and n == h["n_coeffs"], (key, win, n)
        assert abs(val - h["value"]) <= 1e-6 * abs(val), (key, val, h["value"])
        assert abs(val - pred) <= 2 * unc, (key, val, pred, unc)
        assert h["within_2sigma"] is True
        assert abs(h["predicted"] - pred) < 1e-12
    assert head["verdict"] == "CONFIRMED"
    print(f"(e) D(t) and the fit scan match the committed files; c1 = {head['c1']['value']:.6f} (exact {c1:.6f}), "
          f"c2 = {head['c2']['value']:.4f} (exact {c2:.4f}); verdict {head['verdict']}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    check_geometry()
    check_convergence()
    check_bolza()
    check_weyl()
    if not quick:
        check_heat_traces()
        check_difference_and_fits()
    print("COMMITTED-DATA VALIDATION PASSED" + (" (quick: (d), (e) skipped)" if quick else ""))

"""Validation of the committed moduli data.  Every check is an assert; nothing is solved and
nothing is written.

analysis.py rebuilds data/*.csv and summary.json from the raw solver runs (runs/, not
committed, produced on a server) and rewrites those files.  This script instead takes the
committed data/convergence.csv (the production eigenvalue, error estimate and conservative
error of every eigenvalue of all 64 sector problems) and data/length_spectra.csv (the weighted
length spectrum of every member), recomputes the quantities of analysis.py (c)-(f) from them
with the same functions, and asserts both the analysis.py inequalities and agreement with the
committed derived files.  No NGSolve and no network are needed.

  (0) geometry of every member (quad.Lambert.verify), systole = 4b, w/l integral for the
      shortest lengths, systoles strictly decreasing, geometries.json consistent
  (a) per-sector error targets, and counts/values against summary.json
  (c) Weyl law with a0 = 7/9 (against weyl.csv); mirror term Z_N - Z_D
  (d) Z = identity + elliptic (+ geodesic term) to the error budget (against heat_traces.csv)
  (e) eigenvalue flow: non-isospectral by a margin of many orders (against eigenvalue_flow_*.csv)
  (f) pairwise differences = H_i - H_j with the bounds of Theorem 3.4 (b), (c)
      (against trace_differences.csv and summary.json)
  (g) the S3 cross-checks: s3_geodesic_check.csv, s3_repro.json, summary.json records of the
      symmetry reduction (the direct full-Q runs are not committed, so the record is asserted)

usage: python validate_committed.py [--quick]     (--quick: (0), (a), (c) only)
"""

import csv
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import analysis as A  # noqa: E402  (its main() is not run; only its functions and constants)
import geodesics as G  # noqa: E402
import trace_formula as TF  # noqa: E402
from quad import Lambert, TAUS  # noqa: E402
from solve_moduli import SECTORS  # noqa: E402

DATA = os.path.join(HERE, "data")
T = A.T
SUMMARY = json.load(open(os.path.join(DATA, "summary.json")))


def rows_of(name):
    return list(csv.DictReader(open(os.path.join(DATA, name))))


def tag(tau):
    return f"{tau:.1f}"


def load_tables():
    groups = defaultdict(list)
    for r in rows_of("convergence.csv"):
        groups[(float(r["tau"]), r["sector"])].append(r)
    tabs = {}
    for tau in TAUS:
        for sec in SECTORS:
            rs = groups[(tau, sec)]
            assert [int(r["index"]) for r in rs] == list(range(len(rs))), (tau, sec)
            lam = np.array([float(r["lambda_prod_h0.05_p10"]) for r in rs])
            err = np.array([float(r["err_estimate"]) for r in rs])
            errc = np.array([float(r["err_conservative"]) for r in rs])
            assert np.all(np.diff(lam) >= 0), (tau, sec)
            if sec == "NNN":
                assert lam[0] == 0.0 and err[0] == 0.0
            tabs[(tau, sec)] = dict(lam=lam, err=err, err_cons=errc, n=len(lam))
    assert len(groups) == 64, len(groups)
    return tabs


def load_spectra():
    spec = defaultdict(list)
    for r in rows_of("length_spectra.csv"):
        spec[float(r["tau"])].append((float(r["length"]), float(r["weight_w"])))
    for tau in TAUS:
        spec[tau].sort()
    return spec


def check_geometry(lam_objs, spectra):
    geo = json.load(open(os.path.join(DATA, "geometries.json")))
    assert geo["signature"] == "(0;3,3,3,3)"
    systoles = []
    for tau in TAUS:
        L = lam_objs[tau]
        L.verify()
        spec = spectra[tau]
        for l, w in spec[:3]:
            k = w / l
            assert abs(k - round(k)) < 1e-6 and round(k) >= 1, (tau, l, w)
        assert abs(spec[0][0] - 4 * float(L.b)) < 1e-9, (tau, spec[0][0], 4 * float(L.b))
        m = geo["members"][tag(tau)]
        assert abs(m["systole"] - spec[0][0]) < 1e-9
        assert abs(m["perimeter_Q"] - L.summary()["perimeter_Q"]) < 1e-9
        assert abs(m["area_L_quadrature_minus_exact"]) < 1e-12
        systoles.append(spec[0][0])
    assert all(systoles[i] > systoles[i + 1] for i in range(len(TAUS) - 1))
    print(f"(0) geometry of the {len(TAUS)} members verified; systoles {systoles[0]:.4f} ... {systoles[-1]:.4f}, "
          f"strictly decreasing; w/l integral")


def check_convergence(tabs):
    per = SUMMARY["convergence"]["per_sector"]
    worst = worstc = 0.0
    for tau in TAUS:
        for sec in SECTORS:
            tb = tabs[(tau, sec)]
            lam, err, errc = tb["lam"], tb["err"], tb["err_cons"]
            w300 = float(np.max((err / np.maximum(lam, 1))[:300]))
            w300c = float(np.max((errc / np.maximum(lam, 1))[:300]))
            assert w300 < 1e-8, (tau, sec, w300)
            assert w300c < 5e-8, (tau, sec, w300c)
            s = per[f"{tau:.1f}_{sec}"]
            assert s["n"] == tb["n"] and abs(s["lam_max"] - lam[-1]) <= 1e-12 * lam[-1], (tau, sec)
            assert abs(s["worst_rel_first300"] - w300) <= 1e-2 * w300 + 1e-17, (tau, sec)
            assert s["rate_h"] > 12 and s["ratio_p_8_to_10"] > 30 and s["monotone_frac"] > 0.99
            worst, worstc = max(worst, w300), max(worstc, w300c)
    print(f"(a) 64 sector problems: worst rel err (first 300) {worst:.1e} (conservative {worstc:.1e}); counts and "
          f"maxima agree with summary.json; recorded h-rates >= 12, p-ratios >= 30")
    return worstc


def check_weyl(tabs):
    committed = rows_of("weyl.csv")
    a0 = (2 - 4 * (1 - 1 / A.M)) / 6 + 4 * (A.M * A.M - 1) / (12 * A.M)
    assert abs(a0 - 7 / 9) < 1e-15
    for tau in TAUS:
        top = min(tabs[(tau, s)]["lam"][-1] for s in SECTORS)
        allv = np.sort(np.concatenate([tabs[(tau, s)]["lam"] for s in SECTORS]))
        allv = allv[allv < top]
        mid = 0.5 * (allv[:-1] + allv[1:])
        Nn = np.arange(1, len(allv))
        r = Nn - (A.AREA_O * mid / (4 * np.pi) + a0)
        win = [(s, min(s + 400, len(r))) for s in range(0, len(r), 400)]
        means = [float(np.mean(r[s:e])) for s, e in win]
        assert abs(np.mean(r)) < 0.5 and max(abs(x) for x in means) < 1.5, (tau, np.mean(r), means)
        mine = [x for x in committed if abs(float(x["tau"]) - tau) < 1e-9]
        assert len(mine) == len(win), (tau, len(mine), len(win))
        for x, (s, e), m_ in zip(mine, win, means):
            assert (int(x["i_from"]), int(x["i_to"])) == (s, e)
            assert abs(float(x["mean_N_minus_weyl_with_a0"]) - m_) < 2e-4, (tau, s)
        assert abs(SUMMARY["weyl"][tag(tau)]["mean"] - float(np.mean(r))) < 1e-9
    print("(c) orbifold Weyl law with a0 = 7/9: residual means below 0.5, window means below 1.5; matches weyl.csv")


def check_mirror(tabs, lam_objs, lengths, spectra):
    for tau in TAUS:
        geo = lam_objs[tau].summary()
        perim, sysl = geo["perimeter_Q"], spectra[tau][0][0]
        t_m = T[T <= min(0.008, sysl ** 2 / (4 * 32))]
        ZN = sum(np.exp(-np.outer(t_m, tabs[(tau, s)]["lam"])).sum(axis=1) for s in SECTORS if s[0] == "N")
        ZD = sum(np.exp(-np.outer(t_m, tabs[(tau, s)]["lam"])).sum(axis=1) for s in SECTORS if s[0] == "D")
        mterm = perim * np.exp(-t_m / 4) / (4 * np.sqrt(np.pi * t_m))
        bud = A.member_trace(tau, tabs, lengths, t_m)["budget"]
        dev = np.abs(ZN - ZD - mterm)
        assert np.all(dev <= bud + 1e-11), (tau, np.max(dev - bud))
    print("(c) Z_N(Q) - Z_D(Q) = perimeter e^{-t/4}/(4 sqrt(pi t)) to the budget for every member")


def check_traces(tabs, lengths, spectra):
    IE = TF.identity_plus_elliptic(A.AREA_O, A.ORDERS, T)
    committed = defaultdict(list)
    for r in rows_of("heat_traces.csv"):
        committed[float(r["tau"])].append(r)
    traces, Hs = {}, {}
    for tau in TAUS:
        tr = A.member_trace(tau, tabs, lengths)
        H = G.hyperbolic_term(spectra[tau], T)
        Htail, _ = G.tail_bound(spectra[tau], A.L_MAX, None, T)
        traces[tau], Hs[tau] = tr, (H, Htail)
        quiet = H + Htail < 1e-13
        assert quiet.sum() >= 10, (tau, quiet.sum())
        dev_ie = tr["Z"] - IE
        assert np.all(np.abs(dev_ie[quiet]) <= tr["budget"][quiet] + 1e-12), (tau, np.max(np.abs(dev_ie[quiet]) - tr["budget"][quiet]))
        complete = Htail < 1e-12
        resid = tr["Z"] - IE - H
        assert np.all(np.abs(resid[complete]) <= tr["budget"][complete] + Htail[complete] + 1e-12), tau
        mine = committed[tau]
        assert len(mine) == len(T), (tau, len(mine))
        Zc = np.array([float(r["Z_fem"]) for r in mine])
        assert np.all(np.abs(Zc - tr["Z"]) <= 1e-11 * np.abs(tr["Z"]) + 1e-12), (tau, np.max(np.abs(Zc - tr["Z"])))
        Hc = np.array([float(r["H_geodesic_prediction"]) for r in mine])
        assert np.all(np.abs(Hc - H) <= 1e-6 * np.abs(H) + 1e-290), tau      # lengths are printed to 12 decimals
        print(f"(d) tau={tau:.1f}: |Z - (identity + elliptic)| <= {np.max(np.abs(dev_ie[quiet])):.1e} for t <= "
              f"{T[quiet].max():.4f}; |Z - id - ell - H| within the budget up to t = {T[complete].max():.3f}")
    quiet_all = np.all([Hs[tau][0] + Hs[tau][1] < 1e-13 for tau in TAUS], axis=0)
    spread = np.max([traces[tau]["Z"] for tau in TAUS], axis=0) - np.min([traces[tau]["Z"] for tau in TAUS], axis=0)
    bud_all = np.max([traces[tau]["budget"] for tau in TAUS], axis=0)
    assert np.all(spread[quiet_all] <= 2 * bud_all[quiet_all] + 1e-12)
    print(f"(i) all {len(TAUS)} members: max_tau Z - min_tau Z <= {spread[quiet_all].max():.1e} for t <= {T[quiet_all].max():.4f}")
    return traces, Hs


def check_flow(tabs, worstc):
    committed = defaultdict(list)
    for r in rows_of("eigenvalue_flow_orbifold.csv"):
        committed[float(r["tau"])].append(r)
    flow = {}
    for tau in TAUS:
        allv = sorted((float(tabs[(tau, s)]["lam"][k]), s) for s in SECTORS for k in range(tabs[(tau, s)]["n"]))
        flow[tau] = np.array([x[0] for x in allv[:400]])
        mine = committed[tau]
        assert len(mine) == 400
        lam_c = np.array([float(r["lambda"]) for r in mine])
        assert np.all(np.abs(lam_c - flow[tau]) <= 1e-12 * np.maximum(flow[tau], 1)), tau
        # the sector label of an exactly degenerate pair can swap when eigenvalues are printed to 15 digits,
        # so compare the sector content of the prefix that ends at a clear gap
        gaps = np.diff(flow[tau]) > 1e-9 * flow[tau][1:]
        cut = 1 + max(i for i in range(len(gaps)) if gaps[i])
        assert cut > 300, (tau, cut)
        from collections import Counter
        assert Counter(r["sector"] for r in mine[:cut]) == Counter(x[1] for x in allv[:cut]), tau
    for ta, tb_ in combinations(TAUS, 2):
        relmove = float(np.max(np.abs(flow[ta][1:200] - flow[tb_][1:200]) / flow[ta][1:200]))
        assert relmove > 1e4 * worstc, (ta, tb_, relmove)
        assert abs(SUMMARY["eigenvalue_flow"]["max_rel_move_first200"][f"{ta:.1f}-{tb_:.1f}"] - relmove) <= 1e-9 * relmove
    print(f"(e) eigenvalue flow: every pair of the {len(TAUS)} members is non-isospectral by more than 1e4 x the error; "
          f"matches eigenvalue_flow_orbifold.csv and summary.json")


def check_pairs(traces, Hs, lam_objs, spectra):
    committed = defaultdict(list)
    for r in rows_of("trace_differences.csv"):
        committed[(float(r["tau_i"]), float(r["tau_j"]))].append(r)
    ems = []
    for ta, tb_ in combinations(TAUS, 2):
        Za, Zb = traces[ta], traces[tb_]
        D = Za["Z"] - Zb["Z"]
        Ha, Hta = Hs[ta]
        Hb, Htb = Hs[tb_]
        pred = Ha - Hb
        bud = Za["budget"] + Zb["budget"]
        complete = (Hta < 1e-12) & (Htb < 1e-12)
        dev = np.abs(D - pred)
        assert np.all(dev[complete] <= bud[complete] + Hta[complete] + Htb[complete] + 2e-12), (ta, tb_)
        ell = min(spectra[ta][0][0], spectra[tb_][0][0])
        diam = lambda tau: json.load(open(os.path.join(DATA, "geometries.json")))["members"][tag(tau)]["diam_O_upper_bound"]
        delta = max(diam(ta), diam(tb_))
        tb_win = T <= ell ** 2 / (2 * (1 + ell))
        bnd_b = (np.pi * np.exp(3 * delta) / (A.AREA_O * (1 - np.exp(-ell))) * ell * np.exp(ell / 2)
                 * (1 + 2 * T / (ell - T)) * np.exp(-ell ** 2 / (4 * T)) / np.sqrt(4 * np.pi * T))
        assert np.all(np.abs(D[tb_win]) <= bnd_b[tb_win] + bud[tb_win]), (ta, tb_)
        t1 = 0.4
        H1 = max(float(G.hyperbolic_term(spectra[x], t1)[0] + G.tail_bound(spectra[x], A.L_MAX, None, t1)[0][0])
                 for x in (ta, tb_))
        bnd_c = np.sqrt(t1 / T) * np.exp((t1 - T) / 4) * np.exp(ell ** 2 / (4 * t1)) * H1 * np.exp(-ell ** 2 / (4 * T))
        assert np.all(np.abs(D) <= bnd_c * (1 + 1e-9) + bud), (ta, tb_)
        mine = committed[(ta, tb_)]
        assert len(mine) == len(T)
        Dc = np.array([float(r["D_measured"]) for r in mine])
        assert np.all(np.abs(Dc - D) <= 2e-12 + 1e-6 * np.abs(D)), (ta, tb_, np.max(np.abs(Dc - D)))
        s = SUMMARY["pairs"][f"{ta:.1f}-{tb_:.1f}"]
        noise = bud + 1e-12
        above = np.abs(pred) > 10 * noise
        t_pred = float(T[np.argmax(above)]) if above.any() else np.nan
        assert abs(s["emergence_predicted"] - t_pred) <= 1e-9 or (np.isnan(t_pred) and s["emergence_predicted"] is None)
        ems.append((s["emergence_predicted"], s["emergence_measured"]))
    agree = float(max(abs(a - b) / a for a, b in ems))
    assert abs(SUMMARY["emergence_agreement_max_rel"] - agree) < 1e-12
    print(f"(f) {len(list(combinations(TAUS, 2)))} pairs: Z_i - Z_j = H_i - H_j within the budget, bounds of Theorem 3.4 (b), (c) "
          f"hold, matches trace_differences.csv; predicted vs measured emergence times agree to {agree:.2f} (max rel)")


def check_records():
    rows = rows_of("s3_geodesic_check.csv")
    assert len(rows) == 438
    for r in rows:
        Z, S, H = float(r["Z_fem_S3"]), float(r["identity_plus_elliptic"]), float(r["H_geodesic_prediction"])
        res = float(r["residual"])
        # the asserted budget of check_geodesics_s3.py, on the stored residual
        assert abs(res) <= 1e-10 + float(r["length_truncation_bound"]), (r["pillow"], r["t"], res)
        # and the stored residual is Z - identity_plus_elliptic - H (H is printed to 7 digits)
        assert abs(res - (Z - S - H)) <= 5e-13 + 2e-6 * abs(H) + 1e-3 * abs(res), (r["pillow"], r["t"])
    rep = json.load(open(os.path.join(DATA, "s3_repro.json")))
    assert rep["max_rel_diff_first500"] < 1e-8 and rep["max_rel_diff"] < 1e-8
    assert rep["max_diff_over_err_conservative"] < 2 and rep["n"] == 1434
    sr = SUMMARY["symmetry_reduction"]
    for bc in ("N", "D"):
        assert sr[bc]["n_compared"] > 1000 and sr[bc]["max_rel_diff"] < 1e-8, sr[bc]
    mis = SUMMARY["arpack_misses_repaired"]
    assert mis["total"] >= 0 and 0 <= mis["runs_with_misses"] <= 320
    print(f"(g) s3_geodesic_check.csv: {len(rows)} rows within the budget; s3_repro.json: max rel diff "
          f"{rep['max_rel_diff']:.1e}; symmetry reduction record: {sr['N']['n_compared']}/{sr['D']['n_compared']} "
          f"eigenvalues, max rel diff {max(sr['N']['max_rel_diff'], sr['D']['max_rel_diff']):.1e}; "
          f"ARPACK misses repaired by the double cover: {mis['total']} in {mis['runs_with_misses']} runs")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    lam_objs = {tau: Lambert(tau) for tau in TAUS}
    spectra = load_spectra()
    check_geometry(lam_objs, spectra)
    tabs = load_tables()
    worstc = check_convergence(tabs)
    check_weyl(tabs)
    if not quick:
        lengths = {tau: A.sector_lengths(lam_objs[tau]) for tau in TAUS}
        check_mirror(tabs, lam_objs, lengths, spectra)
        traces, Hs = check_traces(tabs, lengths, spectra)
        check_flow(tabs, worstc)
        check_pairs(traces, Hs, lam_objs, spectra)
        check_records()
    print("MODULI COMMITTED-DATA VALIDATION PASSED" + (" (quick: (c) mirror, (d)-(g) skipped)" if quick else ""))

"""T4 analysis: same heat expansion, different shape.  Every check is an assert.

Input: runs/ (solve_moduli.py suite and fullquad, copied back from the server).
Output (data/): see the supplement

  (0) geometry of every member (quad.Lambert.verify); weighted length spectra
      (geodesics.py) with integrality of w/l for the shortest lengths
  (a) convergence per sector: production (0.05,10) vs (0.07,12) [estimate] and
      (0.07,10) [conservative]; h- and p-rates; monotone convergence
  (b) symmetry reduction: union of the four Neumann (Dirichlet) sectors = spectrum of
      the whole quadrilateral Q(0.8) solved directly, with multiplicity
  (c) Weyl law of the orbifold: N_O(lambda) = Area lambda/(4 pi) + a0, a0 = 7/9
      (paper eq:a0conv for signature (0;3,3,3,3)); Neumann-minus-Dirichlet mirror term
  (d) (i)   Z_O = identity + elliptic to the error budget wherever the geodesic term
            is below 1e-13, for every member: the same function of t for all of them;
      full trace formula Z_O = identity + elliptic + H(tau) over the whole window
  (e) (ii)  eigenvalue flow
  (f) (iii) pairwise differences Z_i - Z_j = H_i - H_j: measured vs predicted, emergence
            times, leading-term asymptotics, and the bounds of Theorem 3.4 (b), (c)
"""

import csv
import json
import os
import sys
from itertools import combinations

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import geodesics as G  # noqa: E402
import trace_formula as TF  # noqa: E402
from quad import Lambert, TAUS  # noqa: E402
from solve_moduli import SECTORS, LEVELS, run_path, fullquad_path, FULLQ  # noqa: E402

DATA = os.path.join(HERE, "data")
M = 3
ORDERS = (M,) * 4
AREA_O = 8 * float(np.pi / 2 - np.pi / M)          # 4 pi/3
L_MAX = 6.5
T = np.unique(np.concatenate([np.geomspace(0.0025, 0.02, 90), np.geomspace(0.02, 0.4, 110)]))
REPORT = {}
MISSES = {}


def write_csv(name, header, rows):
    with open(os.path.join(DATA, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ data
def load(tau, sector, h, p):
    d = np.load(run_path(tau, sector, h, p), allow_pickle=True)
    return d["lam"], d


def table(tau, sector):
    lams = [load(tau, sector, h, p)[0] for h, p in LEVELS]
    # completeness across levels: below 0.8 of the smallest lambda_max all five levels are
    # accurate to far better than the spacing, so the counts must agree exactly (an ARPACK
    # miss in one run shows up here; this is how the original single-cover slicing failed)
    # The three accurate levels are compared at 0.8 lambda_max; the two coarse levels, whose
    # errors reach 1e-4..1e-3 near the top and can move an eigenvalue across a cut, at
    # 0.5 lambda_max, where their errors are far below the eigenvalue spacing.
    top = min(l[-1] for l in lams)
    for frac, lv_set in ((0.8, [(0.07, 12), (0.07, 10), (0.05, 10)]), (0.5, LEVELS)):
        counts = {lv: int(np.count_nonzero(lams[LEVELS.index(lv)] < frac * top)) for lv in lv_set}
        assert len(set(counts.values())) == 1, (tau, sector, frac, counts)
    MISSES[(tau, sector)] = sum(int(load(tau, sector, h, p)[1]["misses_repaired"]) for h, p in LEVELS)
    n = min(len(l) for l in lams)
    lams = [l[:n] for l in lams]
    L = dict(zip(LEVELS, lams))
    prod = L[(0.05, 10)].copy()
    floor = 1e-14 * np.maximum(np.abs(prod), 1.0)
    err = np.maximum(np.abs(prod - L[(0.07, 12)]), floor)
    errc = np.maximum(np.abs(prod - L[(0.07, 10)]), floor)
    if sector == "NNN":
        assert abs(prod[0]) < 1e-8, prod[0]
        prod[0], err[0], errc[0] = 0.0, 0.0, 0.0
    for h, p in LEVELS:
        _, d = load(tau, sector, h, p)
        assert abs(float(d["area_err"])) < 1e-12, (tau, sector, h, p, d["area_err"])
    return dict(lam=prod, err=err, err_cons=errc, L=L, n=n)


def sector_lengths(lam_obj):
    """(Neumann length, Dirichlet length) of the sides of L for each sector."""
    s = lam_obj.side_lengths()
    out = {}
    for sec in SECTORS:
        outer, x, y = sec
        ln = ld = 0.0
        for side, bc in (("e", outer), ("n", outer), ("x", x), ("y", y)):
            if bc == "N":
                ln += float(s[side])
            else:
                ld += float(s[side])
        out[sec] = (ln, ld)
    return out


# ------------------------------------------------------------------ traces
def tail_bound(lam, t, a, b, Cup):
    """S3 tail bound (numerics/heat_trace.py) for one sector:
    sum_{lam_j > Lam} e^{-lam_j t} <= -n e^{-Lam t} + t int_Lam^inf e^{-lam t} N_up(lam) dlam,
    N_up = a lam + b sqrt(lam) + Cup with b >= 0 (closed form via Gamma(3/2, Lam t))."""
    assert b >= 0
    Lam, n = lam[-1], len(lam)
    out = []
    for tt in np.atleast_1d(t):
        e = np.exp(-Lam * tt)
        g = float(mp.gammainc(1.5, Lam * tt)) / np.sqrt(tt)
        out.append(max(e * (a * (Lam + 1 / tt) + Cup) + b * g - n * e, 0.0))
    return np.array(out)


def member_trace(tau, tabs, lengths, t=T, conservative=False):
    A_L = AREA_O / 8
    Z = np.zeros_like(t)
    err = np.zeros_like(t)
    tail = np.zeros_like(t)
    for sec in SECTORS:
        tb = tabs[(tau, sec)]
        lam, e = tb["lam"], (tb["err_cons"] if conservative else tb["err"])
        Z += np.exp(-np.outer(t, lam)).sum(axis=1)
        err += (np.exp(-np.outer(t, np.maximum(lam - e, 0))) * e[None, :]).sum(axis=1) * t
        ln, ld = lengths[tau][sec]
        a, b = A_L / (4 * np.pi), (ln - ld) / (4 * np.pi)
        nn = np.arange(1, len(lam) + 1)
        Cup = 2 * max(float(np.max(nn - (a * lam + b * np.sqrt(np.maximum(lam, 0))))), 0) + 10
        # Cup is measured against the signed two-term law; N_up uses |b| >= b, which only
        # enlarges the bound (mixed sectors can have either sign of L_N - L_D)
        tail += tail_bound(lam, t, a, abs(b), Cup)
    return dict(Z=Z, err=err, tail=tail, budget=err + tail)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(DATA, exist_ok=True)
    lam_objs = {tau: Lambert(tau) for tau in TAUS}

    # (0) geometry and closed geodesics -------------------------------------------
    geo, spectra = {}, {}
    rows_ls = []
    for tau in TAUS:
        L = lam_objs[tau]
        v = L.verify()
        P = G.quad_polygon(L)
        spec, ntiles = G.length_spectrum(P, L_MAX)
        for l, w in spec[:3]:
            k = w / l
            assert abs(k - round(k)) < 1e-6 and round(k) >= 1, (tau, l, w)
        sysl = spec[0][0]
        assert abs(sysl - 4 * float(L.b)) < 1e-9, (tau, sysl, 4 * float(L.b))   # systole = 4b
        spectra[tau] = spec
        vs = L.quad_vertices()
        diamQ = max(float(2 * mp.atanh(abs((vs[i] - vs[j]) / (1 - mp.conj(vs[j]) * vs[i]))))
                    for i in range(4) for j in range(i + 1, 4))
        s = L.summary()
        s.update(angles_check={k: float(x) for k, x in v["angles"].items()},
                 area_L_quadrature_minus_exact=float(v["area_quadrature"] - L.area_exact),
                 quad_vertices=[(float(z.real), float(z.imag)) for z in vs],
                 quad_sides_circles=[dict(centre=(float(c.real), float(c.imag)), radius=float(r))
                                     for c, r in L.quad_sides()],
                 r_P=float(P.r_P), diam_Q=diamQ, diam_O_upper_bound=2 * diamQ,
                 systole=sysl, systole_equals_4b=True,
                 shortest_lengths=[dict(length=l, weight=w, weight_over_length=w / l) for l, w in spec[:8]],
                 tiles_enumerated=ntiles, l_max=L_MAX)
        geo[f"{tau:.1f}"] = s
        for l, w in spec:
            rows_ls.append([f"{tau:.1f}", f"{l:.12f}", f"{w:.12f}", f"{w / l:.6f}"])
        print(f"(0) tau={tau:.1f}: angles/area verified; systole 4b = {sysl:.9f}; {len(spec)} lengths <= {L_MAX} "
              f"({ntiles} tiles); diam Q = {diamQ:.4f}", flush=True)
    systoles = [geo[f"{t:.1f}"]["systole"] for t in TAUS]
    assert all(systoles[i] > systoles[i + 1] for i in range(len(TAUS) - 1))    # strictly decreasing
    with open(os.path.join(DATA, "geometries.json"), "w") as f:
        json.dump(dict(signature="(0;3,3,3,3)", modulus="tau = ln(sinh a / sinh b), sinh a sinh b = cos(pi/3)",
                       area_orbifold=AREA_O, members=geo), f, indent=1)
    write_csv("length_spectra.csv", ["tau", "length", "weight_w", "w_over_length"], rows_ls)

    # (a) convergence --------------------------------------------------------------
    tabs = {}
    conv_rows, summary = [], {}
    for tau in TAUS:
        for sec in SECTORS:
            tb = table(tau, sec)
            tabs[(tau, sec)] = tb
            lam, err, errc, Lv = tb["lam"], tb["err"], tb["err_cons"], tb["L"]
            rel = err / np.maximum(lam, 1)
            w300 = float(np.max(rel[:300]))
            w300c = float(np.max((errc / np.maximum(lam, 1))[:300]))
            assert w300 < 1e-8, (tau, sec, w300)
            assert w300c < 5e-8, (tau, sec, w300c)
            ref = Lv[(0.05, 10)]
            e_h = {h: np.abs(Lv[(h, 10)] - ref) for h in (0.1, 0.07)}
            e_p = {p: np.abs(Lv[(0.07, p)] - Lv[(0.07, 12)]) for p in (8, 10)}
            idx = np.arange(tb["n"])
            good = (e_h[0.07] > 1e-11 * lam) & (e_h[0.1] > 1e-11 * lam) & (idx < 500)
            rate_h = float(np.median(np.log(e_h[0.1][good] / e_h[0.07][good]) / np.log(0.1 / 0.07)))
            goodp = (e_p[10] > 1e-11 * lam) & (e_p[8] > 1e-11 * lam) & (idx < 500)
            ratio_p = float(np.median(e_p[8][goodp] / e_p[10][goodp]))
            assert rate_h > 12, (tau, sec, rate_h)
            assert ratio_p > 30, (tau, sec, ratio_p)
            mono = float(np.mean((Lv[(0.07, 10)] - ref)[1:500] >= -1e-9 * np.maximum(lam[1:500], 1)))
            assert mono > 0.99, (tau, sec, mono)
            summary[f"{tau:.1f}_{sec}"] = dict(n=tb["n"], lam_max=float(lam[-1]), worst_rel_first300=w300,
                                               worst_rel_first300_conservative=w300c, rate_h=rate_h,
                                               ratio_p_8_to_10=ratio_p, monotone_frac=mono)
            for i in range(tb["n"]):
                conv_rows.append([f"{tau:.1f}", sec, i, f"{lam[i]:.15g}", f"{err[i]:.3e}", f"{errc[i]:.3e}",
                                  f"{e_h[0.1][i]:.2e}", f"{e_p[8][i]:.2e}"])
    write_csv("convergence.csv", ["tau", "sector", "index", "lambda_prod_h0.05_p10", "err_estimate",
                                  "err_conservative", "diff_h0.1_p10", "diff_h0.07_p8_vs_p12"], conv_rows)
    worst = max(v["worst_rel_first300"] for v in summary.values())
    worstc = max(v["worst_rel_first300_conservative"] for v in summary.values())
    rates = [v["rate_h"] for v in summary.values()]
    ratios = [v["ratio_p_8_to_10"] for v in summary.values()]
    lmin = min(v["lam_max"] for v in summary.values())
    REPORT["arpack_misses_repaired"] = dict(total=int(sum(MISSES.values())),
                                            runs_with_misses=int(sum(1 for v in MISSES.values() if v)))
    REPORT["convergence"] = dict(worst_rel_first300=worst, worst_rel_first300_conservative=worstc,
                                 rate_h_range=[min(rates), max(rates)], ratio_p_range=[min(ratios), max(ratios)],
                                 min_lambda_max=lmin, n_per_sector_range=[min(v["n"] for v in summary.values()),
                                                                           max(v["n"] for v in summary.values())],
                                 per_sector=summary)
    print(f"(a) 64 sector problems: worst rel err (first 300) {worst:.1e} (conservative {worstc:.1e}); "
          f"h-rate {min(rates):.1f}-{max(rates):.1f}; p-ratio {min(ratios):.0f}-{max(ratios):.0f}; "
          f"lambda_max >= {lmin:.0f}; level counts agree; ARPACK misses repaired by double cover: "
          f"{REPORT['arpack_misses_repaired']['total']}", flush=True)

    # (b) symmetry reduction against the whole quadrilateral -----------------------
    tauq = FULLQ["tau"]
    out_b = {}
    for bc in ("N", "D"):
        full = np.load(fullquad_path(bc))["lam"]
        secs = [s for s in SECTORS if s[0] == bc]
        union = np.sort(np.concatenate([tabs[(tauq, s)]["lam"] for s in secs]))
        top = min(full[-1], min(tabs[(tauq, s)]["lam"][-1] for s in secs))
        fu, un = full[full < top - 1], union[union < top - 1]
        # same count below the cut: completeness with multiplicity
        assert len(fu) == len(un), (bc, len(fu), len(un))
        rel = np.abs(fu - un) / np.maximum(un, 1)
        if bc == "N":
            assert abs(fu[0]) < 1e-8
        # the full-Q run is at (0.07,10) on a 4x larger domain: compare with that level's error
        errc = np.sort(np.concatenate([tabs[(tauq, s)]["err_cons"] for s in secs]))[:len(un)]
        assert np.all(np.abs(fu - un) <= 50 * errc + 1e-9 * np.maximum(un, 1)), np.max(np.abs(fu - un) - 50 * errc)
        out_b[bc] = dict(n_compared=int(len(un)), lambda_cut=float(top), max_rel_diff=float(rel[1:].max()),
                         max_rel_diff_first300=float(rel[1:300].max()))
        print(f"(b) Q(tau=0.8) {bc}: union of the 4 sectors = direct full-Q spectrum, {len(un)} eigenvalues "
              f"below {top:.0f}, same count, max rel diff {rel[1:].max():.1e}", flush=True)
    REPORT["symmetry_reduction"] = out_b

    # (c) Weyl law of the orbifold and the mirror term ------------------------------
    # paper eq:a0conv: a0 = chi/6 + sum (m^2 - 1)/(12 m), chi = 2 - 4 (1 - 1/3) = -2/3
    a0 = (2 - 4 * (1 - 1 / M)) / 6 + 4 * (M * M - 1) / (12 * M)
    assert abs(a0 - 7 / 9) < 1e-15
    weyl = {}
    rows_w = []
    for tau in TAUS:
        top = min(tabs[(tau, s)]["lam"][-1] for s in SECTORS)
        allv = np.sort(np.concatenate([tabs[(tau, s)]["lam"] for s in SECTORS]))
        allv = allv[allv < top]
        mid = 0.5 * (allv[:-1] + allv[1:])
        Nn = np.arange(1, len(allv))
        r = Nn - (AREA_O * mid / (4 * np.pi) + a0)
        win = [(s, min(s + 400, len(r))) for s in range(0, len(r), 400)]
        means = [float(np.mean(r[s:e])) for s, e in win]
        assert abs(np.mean(r)) < 0.5 and max(abs(x) for x in means) < 1.5, (tau, np.mean(r), means)
        weyl[f"{tau:.1f}"] = dict(n=len(allv), lambda_max=float(top), mean=float(np.mean(r)),
                                  rms=float(np.sqrt(np.mean(r * r))), max_window_mean=max(abs(x) for x in means))
        for (s, e), m_ in zip(win, means):
            rows_w.append([f"{tau:.1f}", s, e, f"{mid[s]:.3f}", f"{mid[e - 1]:.3f}", f"{m_:.4f}"])
    write_csv("weyl.csv", ["tau", "i_from", "i_to", "lambda_from", "lambda_to", "mean_N_minus_weyl_with_a0"], rows_w)
    REPORT["weyl"] = weyl
    print("(c) orbifold Weyl law with a0 = 7/9: mean residuals "
          + ", ".join(f"{v['mean']:+.3f}" for v in weyl.values()), flush=True)

    lengths = {tau: sector_lengths(lam_objs[tau]) for tau in TAUS}
    mirror = {}
    for tau in TAUS:
        perim = geo[f"{tau:.1f}"]["perimeter_Q"]
        sysl = geo[f"{tau:.1f}"]["systole"]
        t_m = T[T <= min(0.008, sysl ** 2 / (4 * 32))]          # bounce orbits (length >= systole) < e^-32
        ZN = sum(np.exp(-np.outer(t_m, tabs[(tau, s)]["lam"])).sum(axis=1) for s in SECTORS if s[0] == "N")
        ZD = sum(np.exp(-np.outer(t_m, tabs[(tau, s)]["lam"])).sum(axis=1) for s in SECTORS if s[0] == "D")
        mterm = perim * np.exp(-t_m / 4) / (4 * np.sqrt(np.pi * t_m))
        bud = member_trace(tau, tabs, lengths, t_m)["budget"]
        dev = np.abs(ZN - ZD - mterm)
        assert np.all(dev <= bud + 1e-11), (tau, np.max(dev - bud))
        mirror[f"{tau:.1f}"] = dict(t_max=float(t_m[-1]), max_dev=float(dev.max()), n_t=int(len(t_m)))
    REPORT["mirror_term"] = mirror
    print("(c) Z_N(Q) - Z_D(Q) = perimeter e^{-t/4}/(4 sqrt(pi t)) to the budget for every member", flush=True)

    # (d) (i) heat traces against the trace formula --------------------------------
    IE = TF.identity_plus_elliptic(AREA_O, ORDERS, T)
    traces, Hs = {}, {}
    rows_h = []
    heat = {}
    for tau in TAUS:
        tr = member_trace(tau, tabs, lengths)
        H = G.hyperbolic_term(spectra[tau], T)
        Htail, _ = G.tail_bound(spectra[tau], L_MAX, None, T)
        traces[tau], Hs[tau] = tr, (H, Htail)
        dev_ie = tr["Z"] - IE
        quiet = H + Htail < 1e-13
        assert quiet.sum() >= 10, (tau, quiet.sum())   # tau = 2.8 (systole 0.69) has the shortest window
        assert np.all(np.abs(dev_ie[quiet]) <= tr["budget"][quiet] + 1e-12), (tau, np.max(np.abs(dev_ie[quiet]) - tr["budget"][quiet]))
        complete = Htail < 1e-12
        resid = tr["Z"] - IE - H
        assert np.all(np.abs(resid[complete]) <= tr["budget"][complete] + Htail[complete] + 1e-12), \
            (tau, np.max(np.abs(resid[complete]) - tr["budget"][complete]))
        heat[f"{tau:.1f}"] = dict(t_quiet_max=float(T[quiet].max()), max_abs_Z_minus_IE_quiet=float(np.max(np.abs(dev_ie[quiet]))),
                                  max_budget_quiet=float(tr["budget"][quiet].max()),
                                  t_complete_max=float(T[complete].max()),
                                  max_abs_Z_minus_IE_minus_H=float(np.max(np.abs(resid[complete]))),
                                  max_H=float(H[complete].max()))
        for i, tt in enumerate(T):
            rows_h.append([f"{tau:.1f}", f"{tt:.10g}", f"{tr['Z'][i]:.16e}", f"{IE[i]:.16e}", f"{H[i]:.6e}",
                           f"{resid[i]:.3e}", f"{tr['err'][i]:.3e}", f"{tr['tail'][i]:.3e}", f"{Htail[i]:.3e}"])
        print(f"(d) tau={tau:.1f}: |Z - (identity + elliptic)| <= {heat[f'{tau:.1f}']['max_abs_Z_minus_IE_quiet']:.1e} "
              f"for t <= {T[quiet].max():.4f} (budget {tr['budget'][quiet].max():.1e}); |Z - id - ell - H| <= "
              f"{np.max(np.abs(resid[complete])):.1e} up to t = {T[complete].max():.3f} (H up to {H[complete].max():.2e})",
              flush=True)
    write_csv("heat_traces.csv", ["tau", "t", "Z_fem", "identity_plus_elliptic", "H_geodesic_prediction",
                                  "Z_minus_id_ell_H", "err_eigenvalues", "err_tail", "H_length_truncation"], rows_h)
    REPORT["heat_traces"] = heat
    # the whole family is one function of t at small times
    quiet_all = np.all([Hs[tau][0] + Hs[tau][1] < 1e-13 for tau in TAUS], axis=0)
    spread = np.max([traces[tau]["Z"] for tau in TAUS], axis=0) - np.min([traces[tau]["Z"] for tau in TAUS], axis=0)
    bud_all = np.max([traces[tau]["budget"] for tau in TAUS], axis=0)
    assert np.all(spread[quiet_all] <= 2 * bud_all[quiet_all] + 1e-12)
    REPORT["family_small_t"] = dict(t_max=float(T[quiet_all].max()), max_spread=float(spread[quiet_all].max()),
                                    max_budget=float(bud_all[quiet_all].max()))
    print(f"(i) all 8 members: max_tau Z - min_tau Z <= {spread[quiet_all].max():.1e} for t <= {T[quiet_all].max():.4f}",
          flush=True)

    # (e) (ii) eigenvalue flow -----------------------------------------------------
    rows_f = []
    for tau in TAUS:
        for sec in SECTORS:
            tb = tabs[(tau, sec)]
            for k in range(min(120, tb["n"])):
                rows_f.append([f"{tau:.1f}", sec, k, f"{tb['lam'][k]:.15g}", f"{tb['err'][k]:.3e}", f"{tb['err_cons'][k]:.3e}"])
    write_csv("eigenvalue_flow_sectors.csv", ["tau", "sector", "k", "lambda", "err_estimate", "err_conservative"], rows_f)
    rows_o, flow = [], {}
    for tau in TAUS:
        allv = sorted((float(tabs[(tau, s)]["lam"][k]), s, float(tabs[(tau, s)]["err"][k]), float(tabs[(tau, s)]["err_cons"][k]))
                      for s in SECTORS for k in range(tabs[(tau, s)]["n"]))
        for j, (lv, s, e, ec) in enumerate(allv[:400]):
            rows_o.append([f"{tau:.1f}", j, f"{lv:.15g}", s, f"{e:.3e}", f"{ec:.3e}"])
        flow[tau] = np.array([x[0] for x in allv[:400]])
    write_csv("eigenvalue_flow_orbifold.csv", ["tau", "j", "lambda", "sector", "err_estimate", "err_conservative"], rows_o)
    l1 = {f"{tau:.1f}": float(flow[tau][1]) for tau in TAUS}
    moves = {}
    for ta, tb_ in combinations(TAUS, 2):
        relmove = float(np.max(np.abs(flow[ta][1:200] - flow[tb_][1:200]) / flow[ta][1:200]))
        moves[f"{ta:.1f}-{tb_:.1f}"] = relmove
        # every pair is non-isospectral by a margin of many orders over the eigenvalue errors
        assert relmove > 1e4 * REPORT["convergence"]["worst_rel_first300_conservative"], (ta, tb_, relmove)
    REPORT["eigenvalue_flow"] = dict(lambda_1=l1, max_rel_move_first200=moves)
    print("(ii) lambda_1(tau): " + ", ".join(f"{k}: {v:.6f}" for k, v in l1.items()), flush=True)

    # (f) (iii) pairwise differences -----------------------------------------------
    rows_d, pairs = [], {}
    for ta, tb_ in combinations(TAUS, 2):
        Za, Zb = traces[ta], traces[tb_]
        D = Za["Z"] - Zb["Z"]
        Ha, Hta = Hs[ta]
        Hb, Htb = Hs[tb_]
        pred = Ha - Hb
        bud = Za["budget"] + Zb["budget"]
        complete = (Hta < 1e-12) & (Htb < 1e-12)
        dev = np.abs(D - pred)
        assert np.all(dev[complete] <= bud[complete] + Hta[complete] + Htb[complete] + 2e-12), \
            (ta, tb_, np.max(dev[complete] - bud[complete]))
        # leading term: the shorter systole (always tb_, the larger tau)
        lb, wb = spectra[tb_][0]
        lead = -wb / (2 * np.sinh(lb / 2)) * np.exp(-T / 4 - lb * lb / (4 * T)) / np.sqrt(4 * np.pi * T)
        noise = bud + 1e-12
        above = np.abs(pred) > 10 * noise
        t_pred = float(T[np.argmax(above)]) if above.any() else np.nan
        above_m = np.abs(D) > 10 * noise
        # measured emergence: first t from which |D| stays above 10x noise
        idx = np.where(~above_m)[0]
        t_meas = float(T[idx[-1] + 1]) if len(idx) and idx[-1] + 1 < len(T) else float(T[0])
        # Theorem 3.4 (b) explicit bound and (c) controlled bound
        ell = min(spectra[ta][0][0], spectra[tb_][0][0])
        delta = max(geo[f"{ta:.1f}"]["diam_O_upper_bound"], geo[f"{tb_:.1f}"]["diam_O_upper_bound"])
        tb_win = T <= ell ** 2 / (2 * (1 + ell))
        bnd_b = (np.pi * np.exp(3 * delta) / (AREA_O * (1 - np.exp(-ell))) * ell * np.exp(ell / 2)
                 * (1 + 2 * T / (ell - T)) * np.exp(-ell ** 2 / (4 * T)) / np.sqrt(4 * np.pi * T))
        assert np.all(np.abs(D[tb_win]) <= bnd_b[tb_win] + bud[tb_win]), (ta, tb_)
        t1 = 0.4
        # H_i(t1) including the length-truncation bound, so that the constant is not underestimated
        H1 = max(float(G.hyperbolic_term(spectra[x], t1)[0] + G.tail_bound(spectra[x], L_MAX, None, t1)[0][0])
                 for x in (ta, tb_))
        bnd_c = np.sqrt(t1 / T) * np.exp((t1 - T) / 4) * np.exp(ell ** 2 / (4 * t1)) * H1 * np.exp(-ell ** 2 / (4 * T))
        assert np.all(np.abs(D) <= bnd_c * (1 + 1e-9) + bud), (ta, tb_)
        # leading-term ratio once the signal is 1e3 above the noise
        strong = np.abs(pred) > 1e3 * noise
        ratio_lead = float((D / lead)[np.argmax(strong)]) if strong.any() else np.nan
        pairs[f"{ta:.1f}-{tb_:.1f}"] = dict(systoles=[spectra[ta][0][0], spectra[tb_][0][0]],
                                           emergence_predicted=t_pred, emergence_measured=t_meas,
                                           max_abs_D_minus_pred=float(np.max(dev[complete])),
                                           max_budget=float(np.max(bud[complete])),
                                           D_over_leading_term_at_first_strong_t=ratio_lead,
                                           first_strong_t=float(T[np.argmax(strong)]) if strong.any() else None,
                                           bound_b_window_tmax=float(T[tb_win].max()) if tb_win.any() else None)
        for i, tt in enumerate(T):
            rows_d.append([f"{ta:.1f}", f"{tb_:.1f}", f"{tt:.10g}", f"{D[i]:.6e}", f"{pred[i]:.6e}", f"{lead[i]:.6e}",
                           f"{bud[i]:.3e}", f"{(Hta + Htb)[i]:.3e}", f"{bnd_c[i]:.3e}"])
    write_csv("trace_differences.csv", ["tau_i", "tau_j", "t", "D_measured", "D_predicted_Hi_minus_Hj",
                                        "leading_term_shorter_systole", "error_budget", "H_length_truncation",
                                        "bound_thm3.4c_t1_0.4"], rows_d)
    REPORT["pairs"] = pairs
    for k in ("0.0-0.4", "0.0-2.8", "2.4-2.8"):
        p = pairs[k]
        print(f"(iii) {k}: systoles {p['systoles'][0]:.4f}/{p['systoles'][1]:.4f}; emergence predicted t = "
              f"{p['emergence_predicted']:.4f}, measured {p['emergence_measured']:.4f}; max |D - pred| "
              f"{p['max_abs_D_minus_pred']:.1e} (budget {p['max_budget']:.1e}); D / leading = "
              f"{p['D_over_leading_term_at_first_strong_t']:.4f}", flush=True)
    em = [(p["emergence_predicted"], p["emergence_measured"]) for p in pairs.values()]
    REPORT["emergence_agreement_max_rel"] = float(max(abs(a - b) / a for a, b in em))
    with open(os.path.join(DATA, "summary.json"), "w") as f:
        json.dump(REPORT, f, indent=1, default=float)
    print("ALL MODULI CHECKS PASSED")


if __name__ == "__main__":
    main()

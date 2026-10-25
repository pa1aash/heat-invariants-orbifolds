"""Step 4: validation.  Every check is an assert; the script fails loudly.

  (0) geometry: angles, sides, area (geometry.Triangle.verify), FEM area of the
      curved mesh, and the exact Selberg/heat-coefficient cross-checks of theory.py
  (a) convergence in h and p, error estimate per eigenvalue, target 1e-8
      relative for the first 500, and empirical rates (no corner singularity)
  (b) benchmark against the published Bolza eigenvalues (Strohmaier-Uski)
  (c) Weyl law with boundary and constant corrections
  (d) heat traces of each pillow: Area/(4 pi t) and a0; the exact Selberg
      identity + elliptic terms; Z_N - Z_D against the mirror term

Writes data/convergence.csv, data/bolza_benchmark.csv, data/weyl.csv,
data/heat_trace_checks.csv, data/eigenvalues_<p-q-r>_<N|D>.csv.
"""

import csv
import os

import numpy as np

import theory
from eigdata import table, load_level
from geometry import Triangle, PILLOWS, BENCH
from heat_trace import (DATA, pillow_trace, trace, selberg_smooth_plus_elliptic, weyl_terms,
                        write_csv, fit)
from solve import LEVELS, PROBLEMS, BENCH_LEVELS

HERE = os.path.dirname(os.path.abspath(__file__))
BOLZA = os.path.join(HERE, "refs_cache", "SU_eig-bolza-refined0-1000.txt")
REPORT = {}


def check_geometry():
    for pqr in PILLOWS + [BENCH]:
        Triangle(*pqr).verify(tol=1e-12)
    for pqr, bc in PROBLEMS:
        for h, o in LEVELS:
            _, d = load_level(pqr, bc, h, o)
            assert abs(float(d["area_err"])) < 1e-12, (pqr, bc, h, o, d["area_err"])
    assert theory.check_against_paper()
    theory.check_elliptic_expansion(numax=3)
    c = theory.predicted_difference(3)
    from fractions import Fraction as Fr
    assert c[-1] == 0 and c[0] == 0
    assert c[1] == Fr(25, 12) and c[2] == Fr(-1775, 24)
    print("(0) geometry, FEM area and coefficient cross-checks: OK")


# ------------------------------------------------------------------ (a)
def check_convergence():
    rows = []
    summary = {}
    for pqr, bc in PROBLEMS:
        T = table(pqr, bc)
        lam, err, errc = T["lam"], T["err"], T["err_cons"]
        L = dict(zip([tuple(x) for x in T["levels"]], T["all"]))
        n = len(lam)
        rel = err / np.maximum(lam, 1)
        # (a) target: first 500 to 1e-8 relative (index 0 is the exact 0 for Neumann)
        worst500 = float(np.max(rel[:500]))
        worst500_cons = float(np.max((errc / np.maximum(lam, 1))[:500]))
        assert worst500 < 1e-8, (pqr, bc, worst500)
        # even the coarse-mesh worst case stays within a small factor of target
        assert worst500_cons < 5e-8, (pqr, bc, worst500_cons)
        # h-sweep at p = 10 and p-sweep at h = 0.07, measured against production
        ref = L[(0.05, 10)]
        e_h = {h: np.abs(L[(h, 10)] - ref) for h in (0.1, 0.07)}
        e_p = {p: np.abs(L[(0.07, p)] - L[(0.07, 12)]) for p in (8, 10)}
        # rates on the indices where both errors are well above round-off
        idx = np.arange(n)
        good = (e_h[0.07] > 1e-11 * lam) & (e_h[0.1] > 1e-11 * lam) & (idx < 1000)
        rate_h = np.median(np.log(e_h[0.1][good] / e_h[0.07][good]) / np.log(0.1 / 0.07))
        goodp = (e_p[10] > 1e-11 * lam) & (e_p[8] > 1e-11 * lam) & (idx < 1000)
        ratio_p = np.median(e_p[8][goodp] / e_p[10][goodp])
        # a corner singularity r^(pi/alpha) with pi/alpha = k would cap the
        # eigenvalue rate at h^(2k) only if k were not an integer; with reflection
        # smoothness we expect the full h^(2p) = h^20 at p = 10 (pre-asymptotic
        # values somewhat lower).  Require clearly super-algebraic behaviour:
        assert rate_h > 12, (pqr, bc, rate_h)
        assert ratio_p > 30, (pqr, bc, ratio_p)
        # monotone (conforming Galerkin: eigenvalues decrease under refinement)
        mono = float(np.mean((L[(0.07, 10)] - ref)[1:1000] >= -1e-9 * lam[1:1000]))
        summary[(pqr, bc)] = dict(n=n, lam_max=float(lam[-1]), worst500=worst500,
                                  worst500_cons=worst500_cons,
                                  rel_at={i: float(rel[i]) for i in (100, 500, 800, 1000, 1200) if i < n},
                                  rate_h=float(rate_h), ratio_p=float(ratio_p), monotone_frac=mono)
        for i in range(n):
            rows.append(["-".join(map(str, pqr)), bc, i, f"{lam[i]:.15g}", f"{err[i]:.2e}",
                         f"{e_h[0.1][i]:.2e}", f"{e_h[0.07][i]:.2e}", f"{e_p[8][i]:.2e}", f"{e_p[10][i]:.2e}"])
        tag = "-".join(map(str, pqr))
        write_csv(os.path.join(DATA, f"eigenvalues_{tag}_{bc}.csv"),
                  ["index", "lambda", "err_estimate", "err_conservative", "lambda_h0.07_p12", "lambda_h0.07_p10"],
                  [[i, f"{lam[i]:.15g}", f"{err[i]:.3e}", f"{errc[i]:.3e}", f"{L[(0.07, 12)][i]:.15g}",
                    f"{L[(0.07, 10)][i]:.15g}"] for i in range(n)])
        print(f"(a) {pqr} {bc}: n={n}, lambda_max={lam[-1]:.1f}, max rel err (first 500)={worst500:.1e}, "
              f"rel err @1000={rel[min(1000, n - 1)]:.1e}, h-rate={rate_h:.1f}, p-ratio(8->10)={ratio_p:.0f}, "
              f"monotone={mono:.3f}, conservative worst (first 500)={worst500_cons:.1e}")
    write_csv(os.path.join(DATA, "convergence.csv"),
              ["triangle", "bc", "index", "lambda_prod_h0.05_p10", "err_estimate",
               "diff_h0.1_p10", "diff_h0.07_p10", "diff_h0.07_p8_vs_p12", "diff_h0.07_p10_vs_p12"], rows)
    REPORT["convergence"] = summary


# ------------------------------------------------------------------ (b)
def check_bolza():
    """Every one-dimensional character of the extended (2,3,8) triangle group
    that factors through the Bolza surface gives a mixed boundary problem on
    the (2,3,8) triangle; the multiplicity-one Bolza eigenvalues must be exactly
    the union of their spectra.  N and D are the two that matter here."""
    raw = [l.strip() for l in open(BOLZA) if l.strip()]
    bol = np.array([float(x) for x in raw])
    top = bol.max()
    u, cnt = np.unique(np.round(bol, 7), return_counts=True)
    mult1 = u[cnt == 1]
    rows, ours = [], []
    worst = {}
    for bc in ("N", "D", "M1", "M2"):
        fine, _ = load_level(BENCH, bc, *BENCH_LEVELS[-1])
        coarse, _ = load_level(BENCH, bc, *BENCH_LEVELS[0])
        sel = (fine > 1e-6) & (fine < top - 1)
        for x, xc in zip(fine[sel], coarse[sel]):
            j = int(np.argmin(np.abs(bol - x)))
            m = int(np.sum(np.abs(bol - bol[j]) < 1e-7))
            rel = abs(x - bol[j]) / bol[j]
            # published digits: 12 or more significant digits (S-U section 7)
            assert m == 1, (bc, x, m)
            assert rel < 2e-10, (bc, x, raw[j], rel)
            worst[bc] = max(worst.get(bc, 0), rel)
            rows.append([bc, f"{x:.14f}", raw[j], f"{rel:.1e}", f"{abs(x - xc) / x:.1e}", m])
            ours.append(x)
    ours = np.array(ours)
    unexplained = [x for x in mult1 if np.min(np.abs(ours - x)) / x > 1e-9]
    assert not unexplained, unexplained
    assert len(ours) == len(mult1), (len(ours), len(mult1))
    write_csv(os.path.join(DATA, "bolza_benchmark.csv"),
              ["bc", "lambda_fem_(2,3,8)_triangle", "lambda_bolza_strohmaier_uski", "rel_diff",
               "fem_level_diff", "bolza_multiplicity"], rows)
    REPORT["bolza"] = dict(n_mult1=len(mult1), n_matched=len(ours), worst=worst,
                           counts={bc: sum(1 for r in rows if r[0] == bc) for bc in ("N", "D", "M1", "M2")})
    print(f"(b) Bolza: {len(mult1)} multiplicity-one eigenvalues below {top:.0f}, all matched "
          f"by the (2,3,8) triangle (N {REPORT['bolza']['counts']['N']}, D {REPORT['bolza']['counts']['D']}, "
          f"mixed {REPORT['bolza']['counts']['M1']}+{REPORT['bolza']['counts']['M2']}); worst rel diff "
          f"{max(worst.values()):.1e}")


# ------------------------------------------------------------------ (c)
def check_weyl():
    rows = []
    out = {}
    for pqr, bc in PROBLEMS:
        lam = table(pqr, bc)["lam"]
        a, b, c0 = weyl_terms(pqr, bc)
        n = len(lam)
        # counting function at the midpoints between consecutive eigenvalues
        mid = 0.5 * (lam[:-1] + lam[1:])
        N = np.arange(1, n)
        W1 = a * mid
        W2 = W1 + b * np.sqrt(mid)
        W3 = W2 + c0
        r1, r2, r3 = N - W1, N - W2, N - W3
        # averages over windows of 100 eigenvalues
        win = [(s, min(s + 100, n - 1)) for s in range(0, n - 1, 100)]
        m3 = [float(np.mean(r3[s:e])) for s, e in win]
        m1 = [float(np.mean(r1[s:e])) for s, e in win]
        # with the boundary term the residual has no sqrt(lambda) drift; with the
        # constant it averages to ~0
        assert abs(np.mean(r3)) < 0.5, (pqr, bc, np.mean(r3))
        assert max(abs(x) for x in m3) < 1.5, (pqr, bc, m3)
        # without the boundary term the residual grows like +-L sqrt(lambda)/(4 pi)
        assert abs(m1[-1]) > 5 * max(abs(x) for x in m3[-3:]), (pqr, bc)
        out[(pqr, bc)] = dict(mean_full=float(np.mean(r3)), rms_full=float(np.sqrt(np.mean(r3 ** 2))),
                              max_window_mean=max(abs(x) for x in m3), area_only_last_window=m1[-1])
        for (s, e), x1, x3 in zip(win, m1, m3):
            rows.append(["-".join(map(str, pqr)), bc, s, e, f"{lam[s]:.3f}", f"{lam[e]:.3f}",
                         f"{x1:.4f}", f"{x3:.4f}"])
        print(f"(c) {pqr} {bc}: N - Weyl(3 terms) mean {np.mean(r3):+.3f}, rms {np.sqrt(np.mean(r3**2)):.3f}; "
              f"area-only residual in last window {m1[-1]:+.1f}")
    write_csv(os.path.join(DATA, "weyl.csv"),
              ["triangle", "bc", "i_from", "i_to", "lambda_from", "lambda_to",
               "mean_N_minus_area_term", "mean_N_minus_area_boundary_constant"], rows)
    REPORT["weyl"] = out


# ------------------------------------------------------------------ (d)
def check_heat_traces():
    t = np.geomspace(0.0015, 0.05, 60)
    rows = []
    out = {}
    for pqr in PILLOWS:
        P = pillow_trace(pqr, t)
        Z, bound = P["Z"], P["err_eig"] + P["tail"]
        pc = theory.pillow_coeffs(pqr, 8)
        area = float(pc["area_over_pi"]) * np.pi
        a0 = float(pc["coef"][0])
        a1 = float(pc["coef"][1])
        # Area/(4 pi t) and a0: fit Z - Area/(4 pi t) = a0 + a1 t + ... at small t
        sel = t <= 0.008
        y = Z[sel] - area / (4 * np.pi * t[sel])
        V = np.vander(t[sel], 8, increasing=True)
        co, *_ = np.linalg.lstsq(V, y, rcond=None)
        # free fit including the 1/t coefficient as an unknown
        V2 = np.column_stack([1 / t[sel], V])
        co2, *_ = np.linalg.lstsq(V2, Z[sel], rcond=None)
        assert abs(co[0] - a0) < 1e-6, (pqr, co[0], a0)
        assert abs(co2[0] * 4 * np.pi - area) < 1e-6, (pqr, co2[0] * 4 * np.pi, area)
        assert abs(co2[1] - a0) < 1e-4, (pqr, co2[1], a0)
        assert abs(co[1] - a1) / abs(a1) < 1e-3, (pqr, co[1], a1)
        # exact: identity + elliptic terms of the trace formula; closed geodesics
        # are < 1e-12 for t <= 0.03 (shortest length >= 1.86, see REPORT)
        S = selberg_smooth_plus_elliptic(pqr, t)
        resid = Z - S
        small = t <= 0.03
        assert np.all(np.abs(resid[small]) <= bound[small] + 1e-11), (pqr, np.max(np.abs(resid[small]) - bound[small]))
        # mirror term: Z_N - Z_D = L e^{-t/4}/(4 sqrt(pi t)) + exponentially small
        L = float(Triangle(*pqr).perimeter)
        mirror = L * np.exp(-t / 4) / (4 * np.sqrt(np.pi * t))
        diffND = P["N"]["Z"] - P["D"]["Z"]
        mres = diffND - mirror
        # Z_N - Z_D also carries exponentially small terms from orientation-
        # reversing orbits (bouncing between sides), shorter than the closed
        # geodesics of the pillow; test the mirror term where they are < 1e-13
        msmall = t <= 0.012
        assert np.all(np.abs(mres[msmall]) <= bound[msmall] + 1e-11), (pqr, np.max(np.abs(mres[msmall])))
        # and measure the length of the orbit that produces the departure:
        # log|res| ~ -d^2/(4t) for larger t
        big = (t > 0.025) & (t < 0.05)
        d_bounce = float(np.sqrt(np.median(-4 * t[big] * np.log(np.abs(mres[big]))))) if big.any() else np.nan
        out[pqr] = dict(a0_fit=float(co[0]), a0=a0, a1_fit=float(co[1]), a1=a1,
                        area_fit=float(co2[0] * 4 * np.pi), area=area,
                        max_selberg_resid=float(np.max(np.abs(resid[small]))),
                        max_bound=float(np.max(bound[small])),
                        max_mirror_resid=float(np.max(np.abs(mres[msmall]))), d_bounce=d_bounce,
                        resid_at_005=float(resid[-1]), mirror_resid_at_005=float(mres[-1]))
        for i in range(len(t)):
            rows.append(["-".join(map(str, pqr)), f"{t[i]:.6g}", f"{Z[i]:.16e}", f"{S[i]:.16e}",
                         f"{resid[i]:.3e}", f"{bound[i]:.3e}", f"{diffND[i]:.16e}", f"{mirror[i]:.16e}",
                         f"{mres[i]:.3e}"])
        print(f"(d) {pqr}: area fit {co2[0]*4*np.pi:.10f} (exact {area:.10f}); a0 fit {co[0]:.10f} "
              f"(exact {a0:.10f}); a1 fit {co[1]:.6f} (exact {a1:.6f}); "
              f"|Z - Selberg(id+ell)| <= {np.max(np.abs(resid[small])):.1e} for t<=0.03 "
              f"(bound {np.max(bound[small]):.1e}); |Z_N - Z_D - mirror| <= {np.max(np.abs(mres[msmall])):.1e} "
              f"for t<=0.012, departure length ~{d_bounce:.3f}")
    write_csv(os.path.join(DATA, "heat_trace_checks.csv"),
              ["pillow", "t", "Z_fem", "Z_selberg_identity_plus_elliptic", "difference",
               "error_bound", "Z_N_minus_Z_D", "mirror_term", "mirror_difference"], rows)
    REPORT["heat"] = out


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    check_geometry()
    check_convergence()
    check_bolza()
    check_weyl()
    check_heat_traces()
    print("ALL VALIDATION CHECKS PASSED")

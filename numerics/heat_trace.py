"""Step 5: heat traces of the two pillows, their difference, and the fits.

Z_pillow(t) = Z_N(t) + Z_D(t) (doubling principle, the supplement, each
triangle trace summed over the computed eigenvalues, with two error terms:

  eigenvalue error   |e^{-lam t} - e^{-lam' t}| <= t |lam - lam'| e^{-(lam - err) t},
                     summed with the per-eigenvalue estimates of eigdata.py;
  truncation tail    sum_{lam_j > Lam} e^{-lam_j t}
                       = -n e^{-Lam t} + t int_Lam^inf e^{-lam t} N(lam) dlam
                      <= -n e^{-Lam t} + t int_Lam^inf e^{-lam t} N_up(lam) dlam,
                     N_up(lam) = A lam/(4 pi) + s L sqrt(lam)/(4 pi) + C_up
                     (s = +1 Neumann, -1 Dirichlet), C_up = twice the largest
                     excess of the computed counting function over the
                     two-term Weyl law, plus 10.  The integral is closed form:
                       e^{-Lam t}(a (Lam + 1/t) + C_up) + b Gamma(3/2, Lam t)/sqrt(t).

usage:  python heat_trace.py            (Step 5 fits + CSV exports)
        python heat_trace.py kernel     (Step 6 heat-kernel diagonal .npz)
"""

import csv
import os
import sys

import mpmath as mp
import numpy as np
from scipy import integrate

import theory
from eigdata import table
from geometry import Triangle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PAIR = theory.PAIR


# ------------------------------------------------------------ triangle traces
def weyl_terms(pqr, bc):
    T = Triangle(*pqr)
    A = float(T.area_exact)
    L = float(T.perimeter)
    s = 1.0 if bc == "N" else -1.0
    a = A / (4 * np.pi)
    b = s * L / (4 * np.pi)
    # constant term of each triangle trace: half the pillow's a0
    c0 = float(theory.pillow_coeffs(pqr, 0)["coef"][0]) / 2
    return a, b, c0


def counting_excess(lam, a, b):
    """max over the computed range of N(lambda) - (a lambda + b sqrt(lambda)),
    evaluated just above each eigenvalue (where N jumps up)."""
    n = np.arange(1, len(lam) + 1)
    return float(np.max(n - (a * lam + b * np.sqrt(np.maximum(lam, 0)))))


def tail_bound(lam, t, a, b, Cup):
    Lam = lam[-1]
    n = len(lam)
    out = []
    for tt in np.atleast_1d(t):
        e = np.exp(-Lam * tt)
        g = float(mp.gammainc(1.5, Lam * tt)) / np.sqrt(tt)
        val = e * (a * (Lam + 1 / tt) + Cup) + b * g - n * e
        out.append(max(val, 0.0))
    return np.array(out)


def trace(pqr, bc, t, nmax=None, conservative=False):
    T = table(pqr, bc, nmax=nmax)
    lam, err = T["lam"], (T["err_cons"] if conservative else T["err"])
    t = np.atleast_1d(t)
    E = np.exp(-np.outer(t, lam))
    Z = E.sum(axis=1)
    dZ = (np.exp(-np.outer(t, np.maximum(lam - err, 0))) * err[None, :]).sum(axis=1) * t
    a, b, c0 = weyl_terms(pqr, bc)
    Cup = 2 * max(counting_excess(lam, a, b), 0) + 10
    tail = tail_bound(lam, t, a, b, Cup)
    return dict(Z=Z, err_eig=dZ, tail=tail, lam_max=lam[-1], n=len(lam), Cup=Cup)


def pillow_trace(pqr, t, nmax=None, conservative=False):
    N = trace(pqr, "N", t, nmax, conservative)
    D = trace(pqr, "D", t, nmax, conservative)
    return dict(Z=N["Z"] + D["Z"], err_eig=N["err_eig"] + D["err_eig"],
                tail=N["tail"] + D["tail"], N=N, D=D)


# ------------------------------------------------------------ exact comparison
def identity_term(area, t):
    """Selberg identity term Area/(4 pi) int_R r tanh(pi r) e^{-t(1/4 + r^2)} dr."""
    f = lambda r: r * np.tanh(np.pi * r) * np.exp(-t * (0.25 + r * r))
    I, _ = integrate.quad(f, 0, np.inf, epsabs=1e-17, epsrel=2e-14, limit=400)
    return area / (4 * np.pi) * 2 * I


def elliptic(k, t):
    tot = 0.0
    for l in range(1, k):
        th = np.pi * l / k
        f = lambda r: np.exp(-2 * th * r - t * (0.25 + r * r) - np.logaddexp(0, -2 * np.pi * r))
        I1, _ = integrate.quad(f, -np.inf, 0, epsabs=1e-17, epsrel=2e-14, limit=400)
        I2, _ = integrate.quad(f, 0, np.inf, epsabs=1e-17, epsrel=2e-14, limit=400)
        tot += (I1 + I2) / (2 * k * np.sin(th))
    return tot


def selberg_smooth_plus_elliptic(pqr, t):
    area = 2 * np.pi * (1 - sum(1 / m for m in pqr))
    return np.array([identity_term(area, tt) + sum(elliptic(m, tt) for m in pqr)
                     for tt in np.atleast_1d(t)])


def elliptic_difference(t):
    return np.array([sum(elliptic(m, tt) for m in PAIR[0]) - sum(elliptic(m, tt) for m in PAIR[1])
                     for tt in np.atleast_1d(t)])


def geodesic_term(lengths_mult, t):
    """Hyperbolic part of the Selberg trace formula (Dryden-Strohmaier eq. (1))
    for h(r) = e^{-t(1/4 + r^2)}: g(u) = e^{-t/4} e^{-u^2/(4t)} / sqrt(4 pi t), and
    each primitive class of length l contributes l / (2 sinh(l/2)) g(l) (iterates
    are far smaller and are included up to k = 3)."""
    t = np.atleast_1d(t)
    out = np.zeros_like(t)
    for l, m in lengths_mult:
        for k in (1, 2, 3):
            out += m * l / (2 * np.sinh(k * l / 2)) * np.exp(-t / 4 - (k * l) ** 2 / (4 * t)) / np.sqrt(4 * np.pi * t)
    return out


def geodesic_check(pqr, tmin=0.06, tmax=0.5, nlen=4):
    """Z_fem - (identity + elliptic) should be the hyperbolic (closed geodesic)
    part of the trace formula.  Fit it, on [tmin, tmax], by the shortest
    enumerated lengths with free multiplicities (number of primitive oriented
    conjugacy classes of that length); integers are expected."""
    t = np.geomspace(tmin, tmax, 40)
    P = pillow_trace(pqr, t)
    resid = P["Z"] - selberg_smooth_plus_elliptic(pqr, t)
    lengths = theory.shortest_geodesics(pqr, maxlen=10, nmax=nlen)
    B = np.column_stack([geodesic_term([(l, 1)], t) for l in lengths])
    m, *_ = np.linalg.lstsq(B, resid, rcond=None)
    model = B @ m
    return dict(t=t, resid=resid, lengths=lengths, mult=m, misfit=float(np.max(np.abs(resid - model))),
                resid_max=float(np.max(np.abs(resid))))


# ------------------------------------------------------------ fitting
def fit(t, D, Derr, order):
    """Least squares for D(t)/t = c1 + c2 t + ... + c_order t^(order-1).
    Returns coefficients and a worst-case propagated data error
    |dc| <= |pinv| (Derr / t), i.e. the largest change any perturbation of the
    data within its error bound can make to each coefficient."""
    y = D / t
    yerr = Derr / t
    V = np.vander(t, order, increasing=True)
    # scale columns for conditioning
    s = np.max(np.abs(V), axis=0)
    P = np.linalg.pinv(V / s)
    c = (P @ y) / s
    dc = (np.abs(P) @ yerr) / s
    return c, dc


def fit_scan(t, D, Derr, windows, orders):
    rows = []
    for (ta, tb) in windows:
        sel = (t >= ta * (1 - 1e-12)) & (t <= tb * (1 + 1e-12))
        for n in orders:
            if np.count_nonzero(sel) < 3 * n:
                continue
            c, dc = fit(t[sel], D[sel], Derr[sel], n)
            rows.append(dict(ta=ta, tb=tb, order=n, c1=c[0], c1_err=dc[0], c2=c[1], c2_err=dc[1],
                             c3=c[2] if n > 2 else np.nan))
    return rows


def headline(rows, key):
    """Blind model-error estimate for one window: for each order n, the change
    from order n-1 measures the truncation error of the polynomial model; pick
    the order minimising max(data error, |change|) and report that as the
    total uncertainty."""
    best = None
    by_window = {}
    for r in rows:
        by_window.setdefault((r["ta"], r["tb"]), []).append(r)
    out = []
    for w, rs in by_window.items():
        rs = sorted(rs, key=lambda r: r["order"])
        for i in range(1, len(rs)):
            chg = abs(rs[i][key] - rs[i - 1][key])
            unc = max(rs[i][key + "_err"], chg)
            out.append((unc, w, rs[i]["order"], rs[i][key], rs[i][key + "_err"], chg))
    out.sort(key=lambda x: x[0])
    return out


# ------------------------------------------------------------ main (Step 5)
T_GRID = np.unique(np.concatenate([np.geomspace(0.0015, 0.02, 160), np.geomspace(0.02, 0.5, 60)]))
WINDOWS = [(0.0015, 0.004), (0.0015, 0.006), (0.0015, 0.008), (0.002, 0.006), (0.002, 0.008),
           (0.002, 0.01), (0.003, 0.01), (0.0015, 0.01), (0.0015, 0.012)]
ORDERS = list(range(2, 12))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(DATA, exist_ok=True)
    t = T_GRID
    Za = pillow_trace(PAIR[0], t)
    Zb = pillow_trace(PAIR[1], t)
    D = Za["Z"] - Zb["Z"]
    Derr = Za["err_eig"] + Zb["err_eig"] + Za["tail"] + Zb["tail"]
    c = theory.predicted_difference(8)
    c1, c2, c3 = (float(c[k]) for k in (1, 2, 3))
    Eex = elliptic_difference(t)
    # closed geodesics: shortest lengths of each pillow, multiplicity bound 8 (see REPORT)
    La = theory.shortest_geodesics(PAIR[0], maxlen=10, nmax=4)
    Lb = theory.shortest_geodesics(PAIR[1], maxlen=10, nmax=4)
    geo_bound = geodesic_term([(l, 8) for l in La], t) + geodesic_term([(l, 8) for l in Lb], t)
    rows = []
    for i, tt in enumerate(t):
        rows.append([f"{tt:.8g}", f"{Za['Z'][i]:.16e}", f"{Zb['Z'][i]:.16e}", f"{D[i]:.16e}",
                     f"{Za['err_eig'][i] + Zb['err_eig'][i]:.3e}", f"{Za['tail'][i] + Zb['tail'][i]:.3e}",
                     f"{c1 * tt:.16e}", f"{c1 * tt + c2 * tt ** 2:.16e}", f"{Eex[i]:.16e}",
                     f"{D[i] - Eex[i]:.3e}", f"{geo_bound[i]:.3e}"])
    write_csv(os.path.join(DATA, "heat_trace_difference.csv"),
              ["t", "Z_2_8_8", "Z_3_3_12", "D", "D_err_eigenvalues", "D_err_truncation_tail",
               "pred_c1_t", "pred_c1_t_plus_c2_t2", "pred_exact_elliptic_difference",
               "D_minus_exact_elliptic", "geodesic_term_bound"], rows)

    # --- fits
    rows_fit = fit_scan(t, D, Derr, WINDOWS, ORDERS)
    rows_syn = fit_scan(t, Eex, Derr, WINDOWS, ORDERS)   # same fits on the exact function
    syn = {(r["ta"], r["tb"], r["order"]): r for r in rows_syn}
    write_csv(os.path.join(DATA, "fits.csv"),
              ["t_min", "t_max", "n_coeffs", "c1_fit", "c1_data_err", "c2_fit", "c2_data_err",
               "c1_fit_on_exact_elliptic", "c2_fit_on_exact_elliptic"],
              [[r["ta"], r["tb"], r["order"], f"{r['c1']:.10f}", f"{r['c1_err']:.2e}", f"{r['c2']:.8f}",
                f"{r['c2_err']:.2e}", f"{syn[(r['ta'], r['tb'], r['order'])]['c1']:.10f}",
                f"{syn[(r['ta'], r['tb'], r['order'])]['c2']:.8f}"] for r in rows_fit])
    return dict(t=t, D=D, Derr=Derr, Eex=Eex, rows=rows_fit, rows_syn=rows_syn, c=(c1, c2, c3),
                geo=geo_bound, Za=Za, Zb=Zb, La=La, Lb=Lb)


HEADLINE_WINDOW = (0.0015, 0.012)


def summarize(res):
    """Headline numbers.  The fit window and order are chosen blind (smallest
    combined data + model uncertainty, headline()); the verdict compares the
    fitted values with the predictions using that uncertainty.  Asserts guard
    the preconditions of the fit, not its outcome."""
    import json
    t, D, Derr = res["t"], res["D"], res["Derr"]
    c1, c2, c3 = res["c"]
    w = (t >= HEADLINE_WINDOW[0] * (1 - 1e-12)) & (t <= HEADLINE_WINDOW[1] * (1 + 1e-12))
    tw = t[w]
    # preconditions over the fit window
    tail = res["Za"]["tail"] + res["Zb"]["tail"]
    assert np.all(tail[w] < 1e-9 * c1 * tw), "truncation tail not negligible against c1 t"
    assert np.all(res["geo"][w] < 1e-12 * c1 * tw), "geodesic terms not negligible"
    assert np.all(Derr[w] < 1e-6 * c1 * tw), "eigenvalue errors not negligible against c1 t"
    # computed D against the exact Selberg elliptic difference where geodesics are negligible
    small = t <= 0.03
    exact_resid = float(np.max(np.abs(D[small] - res["Eex"][small])))
    assert exact_resid < 1e-11, exact_resid
    h1 = headline(res["rows"], "c1")
    h2 = headline(res["rows"], "c2")
    # conservative (coarse-mesh) error propagation at the same configurations
    Zac = pillow_trace(PAIR[0], t, conservative=True)
    Zbc = pillow_trace(PAIR[1], t, conservative=True)
    Derrc = Zac["err_eig"] + Zbc["err_eig"] + Zac["tail"] + Zbc["tail"]

    def pick(h, key, pred):
        unc, win, n, val, derr, chg = h[0]
        sel = (t >= win[0] * (1 - 1e-12)) & (t <= win[1] * (1 + 1e-12))
        cc, dcc = fit(t[sel], D[sel], Derrc[sel], n)
        k = 0 if key == "c1" else 1
        unc_c = max(dcc[k], chg)
        return dict(value=float(val), uncertainty=float(unc), data_error=float(derr), order_change=float(chg),
                    window=list(win), n_coeffs=int(n), predicted=pred, deviation=float(val - pred),
                    sigma=float((val - pred) / unc), uncertainty_conservative=float(unc_c),
                    sigma_conservative=float((val - pred) / unc_c),
                    within_2sigma=bool(abs(val - pred) <= 2 * unc))
    r1 = pick(h1, "c1", c1)
    r2 = pick(h2, "c2", c2)
    three = {}
    for win in [(0.0015, 0.004), (0.0015, 0.006), (0.002, 0.01)]:
        sel = (t >= win[0] * (1 - 1e-12)) & (t <= win[1] * (1 + 1e-12))
        cc, dcc = fit(t[sel], D[sel], Derr[sel], 3)
        three[f"{win[0]}-{win[1]}"] = dict(c1=float(cc[0]), c2=float(cc[1]), c3=float(cc[2]))
    verdict = "CONFIRMED" if (r1["within_2sigma"] and r2["within_2sigma"]) else "DISAGREES"
    out = dict(verdict=verdict, c1=r1, c2=r2, c3_predicted=c3, three_term_fits=three,
               max_abs_D_minus_exact_elliptic_t_le_0_03=exact_resid,
               max_geodesic_term_in_fit_window=float(np.max(res["geo"][w])),
               max_truncation_tail_in_fit_window=float(np.max(tail[w])),
               max_D_error_in_fit_window=float(np.max(Derr[w])),
               max_D_error_conservative_in_fit_window=float(np.max(Derrc[w])),
               shortest_geodesics={"2-8-8": [float(x) for x in res["La"]],
                                   "3-3-12": [float(x) for x in res["Lb"]]})
    with open(os.path.join(DATA, "headline.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"c1: fitted {r1['value']:.8f} +- {r1['uncertainty']:.1e}  predicted {c1:.8f}  "
          f"({r1['sigma']:+.2f} sigma; conservative {r1['sigma_conservative']:+.2f})")
    print(f"c2: fitted {r2['value']:.5f} +- {r2['uncertainty']:.1e}  predicted {c2:.5f}  "
          f"({r2['sigma']:+.2f} sigma; conservative {r2['sigma_conservative']:+.2f})")
    print("verdict:", verdict)
    return out


# ------------------------------------------------------------ Step 6: heat kernel
def kernel(level=(0.07, 10), times=(0.005, 0.01, 0.02), sample_h=0.03, nev=None, outname="heat_kernel_diagonal.npz"):
    import ngsolve as ngs
    from geometry import make_mesh
    from solve import assemble, eigenvalues, NEV
    ngs.SetNumThreads(8)
    out = {}
    for pqr in PAIR:
        T = Triangle(*pqr)
        fine = make_mesh(T, sample_h, 1)
        pts = np.array([v.point for v in fine.vertices])
        # pull boundary points a hair inside so that point location in the
        # (differently curved) solver mesh cannot fail on round-off
        cen = pts.mean(axis=0)
        pts_eval = cen + (pts - cen) * (1 - 1e-9)
        tris = np.array([[v.nr for v in el.vertices] for el in fine.Elements(ngs.VOL)])
        Ksum = {bc: np.zeros((len(times), len(pts))) for bc in ("N", "D")}
        for bc in ("N", "D"):
            S = assemble(pqr, bc, *level)
            lam, vec = eigenvalues(S["A"], S["M"], nev or NEV, k=200, want_vectors=True, verbose=False)
            gf = ngs.GridFunction(S["fes"])
            fd = S["freedofs"]
            mpts = S["mesh"](pts_eval[:, 0], pts_eval[:, 1])
            for j in range(len(lam)):
                full = np.zeros(S["fes"].ndof)
                full[fd] = vec[:, j]
                gf.vec.FV().NumPy()[:] = full
                u = gf(mpts).ravel()
                for i, tt in enumerate(times):
                    Ksum[bc][i] += np.exp(-lam[j] * tt) * u * u
            print(pqr, bc, len(lam), "eigenfunctions, lambda_max", lam[-1], flush=True)
        # pillow heat kernel on one sheet: eigenfunctions of the pillow are the
        # even/odd extensions divided by sqrt(2)
        K = 0.5 * (Ksum["N"] + Ksum["D"])
        tag = "-".join(map(str, pqr))
        out[tag] = dict(points=pts, triangles=tris, t=np.array(times), K_pillow=K,
                        K_neumann=Ksum["N"], K_dirichlet=Ksum["D"])
    np.savez_compressed(os.path.join(DATA, outname),
                        **{f"{tag}_{k}": v for tag, d in out.items() for k, v in d.items()},
                        level=np.array(level), note=np.array(
                            "K(t,x,x) of the pillow at points x of one sheet (the triangle in the "
                            "Poincare disk, coordinates in the disk); hyperbolic normalisation, "
                            "int K dA_hyp = Z(t). K_neumann/K_dirichlet: triangle kernels."))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kernel":
        kernel()
    else:
        summarize(main())

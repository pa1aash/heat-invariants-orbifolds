"""Blind end-to-end pipeline, implementing PROTOCOL.md (committed before this file was run).

Input: the four eigenvalue CSVs only (paths below; names are never parsed).
Output: blind/RESULT.md and blind/result.json.  Sections 1-8 are written and flushed before
section 9 (final comparison) reads the label-to-truth map.
Exits nonzero on any failed assert (internal consistency), never on an unfavourable result.
"""
import csv
import json
import os
import sys
from fractions import Fraction as Fr

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from stab_common import front_end, heat_direct, mat_inv, mat_vec, theorem_B_system  # noqa: E402
from threshold import certify  # noqa: E402

mp.mp.dps = 50
N_CONE = 3                                   # structural: a pillow has three cone points
DATA = os.path.join(ROOT, "numerics", "data")
SPECIMENS = {
    "A": (os.path.join(DATA, "eigenvalues_2-8-8_N.csv"), os.path.join(DATA, "eigenvalues_2-8-8_D.csv")),
    "B": (os.path.join(DATA, "eigenvalues_3-3-12_N.csv"), os.path.join(DATA, "eigenvalues_3-3-12_D.csv")),
}
T_GRID = np.geomspace(0.0015, 0.02, 160)
WINDOWS = [(0.0015, 0.004), (0.0015, 0.006), (0.0015, 0.008), (0.002, 0.006), (0.002, 0.008),
           (0.002, 0.01), (0.003, 0.01), (0.0015, 0.01), (0.0015, 0.012)]
DEGREES = range(3, 13)


# ------------------------------------------------------------------ step 1-2
def load(path):
    lam, err, cons = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            lam.append(float(row["lambda"]))
            err.append(float(row["err_estimate"]))
            cons.append(float(row["err_conservative"]))
    lam = np.array(lam)
    assert np.all(np.diff(lam) >= 0), path
    return lam, np.array(err), np.array(cons)


def blind_weyl(lam):
    """least-squares fit of N(lambda) ~ a lambda + b sqrt(lambda) + c on the upper half of the range."""
    n = np.arange(1, len(lam) + 1) - 0.5
    sel = lam >= lam[-1] / 2
    V = np.column_stack([lam[sel], np.sqrt(lam[sel]), np.ones(sel.sum())])
    (a, b, c), *_ = np.linalg.lstsq(V, n[sel], rcond=None)
    excess = float(np.max(np.arange(1, len(lam) + 1) - (a * lam + b * np.sqrt(lam))))
    return a, b, c, excess


def tail_bound(lam, t, a, b, Cup):
    Lam, n = lam[-1], len(lam)
    out = []
    for tt in t:
        e = np.exp(-Lam * tt)
        g = float(mp.gammainc(1.5, Lam * tt)) / np.sqrt(tt)
        out.append(max(e * (a * (Lam + 1 / tt) + Cup) + b * g - n * e, 0.0))
    return np.array(out)


def trace(paths, errcol):
    Z = np.zeros_like(T_GRID)
    dZ = np.zeros_like(T_GRID)
    tail = np.zeros_like(T_GRID)
    weyl = []
    for p in paths:
        lam, err, cons = load(p)
        e = err if errcol == "estimate" else cons
        Z += np.exp(-np.outer(T_GRID, lam)).sum(axis=1)
        dZ += (np.exp(-np.outer(T_GRID, np.maximum(lam - e, 0))) * e[None, :]).sum(axis=1) * T_GRID
        a, b, c, excess = blind_weyl(lam)
        Cup = 2 * max(excess, 0) + 10
        tail += tail_bound(lam, T_GRID, 1.01 * abs(a), abs(b), Cup)
        weyl.append(dict(file=os.path.basename(p), n=len(lam), lam_max=float(lam[-1]), a=a, b=b, c=c,
                         area_from_weyl=4 * np.pi * a, Cup=Cup))
    return Z, dZ, tail, weyl


# ------------------------------------------------------------------ step 3
def fit(t, y, yerr, d):
    V = np.vander(t, d + 1, increasing=True)
    s = np.max(np.abs(V), axis=0)
    P = np.linalg.pinv(V / s)
    return (P @ y) / s, (np.abs(P) @ yerr) / s


def estimate(Z, dZ, tail):
    y = T_GRID * Z
    yerr = T_GRID * (dZ + tail)
    configs = {}
    for (ta, tb) in WINDOWS:
        sel = (T_GRID >= ta * (1 - 1e-12)) & (T_GRID <= tb * (1 + 1e-12))
        for d in DEGREES:
            if sel.sum() < 3 * (d + 1):
                continue
            configs[(ta, tb, d)] = fit(T_GRID[sel], y[sel], yerr[sel], d)
    est = {}
    for j, name in enumerate(["H-1", "H0", "H1"]):
        rows = []
        for (ta, tb, d), (c, dc) in configs.items():
            prev = configs.get((ta, tb, d - 1))
            if prev is None:
                continue
            model = abs(c[j] - prev[0][j])
            u = max(dc[j], model)
            rows.append(dict(window=(ta, tb), degree=d, value=float(c[j]), data=float(dc[j]), model=float(model), u=float(u)))
        rows.sort(key=lambda r: r["u"])
        top = rows[:8]
        half_range = (max(r["value"] for r in top) - min(r["value"] for r in top)) / 2
        best = rows[0]
        est[name] = dict(value=best["value"], u_best=best["u"], half_range_top8=half_range,
                         U=max(best["u"], half_range), config=dict(window=best["window"], degree=best["degree"],
                                                                     data=best["data"], model=best["model"]),
                         top8=top)
    tail_max = float(np.max(tail[T_GRID <= 0.012]))
    return est, tail_max


# ------------------------------------------------------------------ step 5-7
def fr_up(x):
    """a rational >= x (x >= 0 float)."""
    f = Fr(float(np.nextafter(x, np.inf)))
    assert f >= Fr(x)
    return f


def recover(est):
    H = [Fr(est[k]["value"]) for k in ("H-1", "H0", "H1")]
    L, h0 = front_end(N_CONE)
    I = mat_vec(mat_inv(L), [a - b for a, b in zip(H, h0)])
    M, b, _ = theorem_B_system(I)
    e = mat_vec(mat_inv(M), b)
    coeffs = [mp.mpf(1)] + [mp.mpf((-1) ** (j + 1) * e[j].numerator) / e[j].denominator for j in range(N_CONE)]
    roots = mp.polyroots(coeffs, maxsteps=400, extraprec=300)
    cand = tuple(sorted(int(mp.nint(mp.re(z))) for z in roots))
    return dict(I=[float(x) for x in I], e=[float(x) for x in e], roots=[complex(z) for z in roots], candidate=cand)


def admissible(m):
    return all(x >= 2 for x in m) and sum(Fr(1, x) for x in m) < N_CONE - 2


def run_certificate(est, cand):
    if not admissible(cand):
        return dict(admissible=False, certified=False)
    Hc = heat_direct(cand, N_CONE)
    rad = []
    for k, h in zip(("H-1", "H0", "H1"), Hc):
        rad.append(abs(Fr(est[k]["value"]) - h) + fr_up(est[k]["U"]))
    ok = certify(cand, rad)
    return dict(admissible=True, certified=bool(ok), radius=[float(x) for x in rad],
                H_candidate=[float(x) for x in Hc])


def enumerate_candidates(est, k):
    H1v, H1u = Fr(est["H-1"]["value"]), fr_up(k * est["H-1"]["U"])
    H0v, H0u = Fr(est["H0"]["value"]), fr_up(k * est["H0"]["U"])
    H2v, H2u = Fr(est["H1"]["value"]), fr_up(k * est["H1"]["U"])
    Pmax = int(12 * (H0v + H0u) + 2) + 1        # P_1 = 12 H_0 + 2 - R  (n = 3, front end)
    two, three = [], []
    for p in range(2, Pmax + 1):
        for q in range(p, Pmax + 1 - p):
            for r in range(q, Pmax + 1 - p - q):
                R = Fr(1, p) + Fr(1, q) + Fr(1, r)
                if R >= 1:
                    continue
                H = heat_direct((p, q, r), 3)
                if abs(H[0] - H1v) <= H1u and abs(H[1] - H0v) <= H0u:
                    two.append((p, q, r))
                    if abs(H[2] - H2v) <= H2u:
                        three.append((p, q, r))
    return dict(Pmax=Pmax, two=two, three=three)


# ------------------------------------------------------------------ main
def run(errcol):
    res = {}
    for lab, paths in SPECIMENS.items():
        Z, dZ, tail, weyl = trace(paths, errcol)
        est, tail_max = estimate(Z, dZ, tail)
        rec = recover(est)
        cert = run_certificate(est, rec["candidate"])
        enum = {k: enumerate_candidates(est, k) for k in (1, 3)}
        res[lab] = dict(estimates=est, tail_max_in_windows=tail_max, weyl=weyl, recovery=rec, certificate=cert,
                        enumeration=enum, area=4 * np.pi * est["H-1"]["value"], area_U=4 * np.pi * est["H-1"]["U"])
    return res


def fmt_est(e):
    return f"{e['value']:.10f} ± {e['U']:.2e}"


def main():
    out = ["# Blind end-to-end result (generated by blind_pipeline.py, following PROTOCOL.md)\n"]
    results = {}
    for errcol in ("estimate", "conservative"):
        res = run(errcol)
        results[errcol] = res
        out.append(f"## Eigenvalue errors: err_{errcol}\n")
        out.append("### Steps 1-4: heat coefficients and area\n")
        out.append("| specimen | H₋₁ = Area/4π | H₀ | H₁ | Area | max tail (t ≤ 0.012) |")
        out.append("|---|---|---|---|---|---|")
        for lab, r in res.items():
            e = r["estimates"]
            out.append(f"| {lab} | {fmt_est(e['H-1'])} | {fmt_est(e['H0'])} | {fmt_est(e['H1'])} | "
                       f"{r['area']:.10f} ± {r['area_U']:.2e} | {r['tail_max_in_windows']:.1e} |")
        out.append("\nSelected configurations (window, degree, data error, model error, half-range of top 8):\n")
        for lab, r in res.items():
            for k, e in r["estimates"].items():
                c = e["config"]
                out.append(f"- {lab} {k}: window {c['window']}, degree {c['degree']}, data {c['data']:.2e}, "
                           f"model {c['model']:.2e}, half-range {e['half_range_top8']:.2e}")
        out.append("\nBlind Weyl fits (area from the counting function, a cross-check only):\n")
        for lab, r in res.items():
            for w in r["weyl"]:
                out.append(f"- {lab} {w['file'][-5]}: n = {w['n']}, lambda_max = {w['lam_max']:.1f}, "
                           f"4 pi a = {w['area_from_weyl']:.6f}, b = {w['b']:.5f}")
        out.append("\n### Step 5-6: recovery and certificate\n")
        for lab, r in res.items():
            rec, cert = r["recovery"], r["certificate"]
            out.append(f"- **{lab}**: I~ = ({', '.join(f'{x:.6f}' for x in rec['I'])}); e~ = ({', '.join(f'{x:.6f}' for x in rec['e'])}); "
                       f"roots = {', '.join(f'{z.real:.6f}{z.imag:+.6f}i' for z in rec['roots'])}; "
                       f"candidate m^ = {rec['candidate']}; admissible: {cert['admissible']}; "
                       f"**{'CERTIFIED' if cert['certified'] else 'NOT CERTIFIED'}**"
                       + (f" (certified radius used: {', '.join(f'{x:.2e}' for x in cert['radius'])})" if cert.get('radius') else ""))
        out.append("\n### Step 7: exhaustive enumeration of hyperbolic integer triples\n")
        for lab, r in res.items():
            for k, en in r["enumeration"].items():
                out.append(f"- {lab}, k = {k} (P₁ ≤ {en['Pmax']}): two-coefficient candidates {en['two']}; "
                           f"three-coefficient candidates {en['three']}")
        A, B = res["A"]["estimates"], res["B"]["estimates"]
        out.append("\n### Step 8: can two coefficients separate the specimens?\n")
        out.append("(a) Structural (front end, T1): H₋₁ = (1 − R)/2 and H₀ = (P₁ + R − 2)/12 for n = 3 depend on (R, P₁) only.\n")
        out.append("(b) Differences A − B:\n")
        for k in ("H-1", "H0", "H1"):
            d = A[k]["value"] - B[k]["value"]
            u = A[k]["U"] + B[k]["U"]
            out.append(f"- {k}: {d:+.3e} ± {u:.2e}  ({abs(d) / u:.2f} σ)")
        ta = set(res["A"]["enumeration"][1]["two"])
        tb = set(res["B"]["enumeration"][1]["two"])
        out.append(f"\n(c) Two-coefficient candidate sets (k = 1) intersect in {sorted(ta & tb)}; "
                   f"A's set contains B's recovered candidate: {res['B']['recovery']['candidate'] in ta}; "
                   f"B's set contains A's recovered candidate: {res['A']['recovery']['candidate'] in tb}.\n")
    # ---- flush sections 1-8 before reading the truth
    path = os.path.join(HERE, "RESULT.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: str(o) if not isinstance(o, complex) else [o.real, o.imag])
    final_comparison(results, path)


def final_comparison(results, path):
    """Step 9: the only place the truth is used."""
    TRUTH = {"A": (2, 8, 8), "B": (3, 3, 12)}
    out = ["## Step 9: final comparison with the truth (read only now)\n"]
    summary = {}
    for errcol, res in results.items():
        out.append(f"### err_{errcol}\n")
        out.append("| specimen | truth | recovered | exact | certified | (est − true)/U for H₋₁, H₀, H₁ | 2-coeff set contains both truths |")
        out.append("|---|---|---|---|---|---|---|")
        for lab, r in res.items():
            true = TRUTH[lab]
            Ht = heat_direct(true, 3)
            dev = [(r["estimates"][k]["value"] - float(h)) / r["estimates"][k]["U"]
                   for k, h in zip(("H-1", "H0", "H1"), Ht)]
            exact = r["recovery"]["candidate"] == true
            two = set(r["enumeration"][1]["two"])
            both = all(t in two for t in TRUTH.values())
            summary[(errcol, lab)] = dict(exact=exact, certified=r["certificate"]["certified"], dev=dev, both=both)
            out.append(f"| {lab} | {true} | {r['recovery']['candidate']} | {exact} | {r['certificate']['certified']} | "
                       + ", ".join(f"{x:+.2f}" for x in dev) + f" | {both} |")
            if any(abs(x) > 1 for x in dev):
                out.append(f"\n**ERROR-BAR VIOLATION** for {lab} (err_{errcol}): some |estimate − truth| exceeds U.\n")
    with open(path, "a") as f:
        f.write("\n" + "\n".join(out) + "\n")
    print(open(path).read())


if __name__ == "__main__":
    main()

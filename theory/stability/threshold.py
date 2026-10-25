"""T4. Exact integer recovery: explicit thresholds delta*(m).

Recovery map (the algorithm whose stability is certified):
  heat data H~ (first n coefficients)  --L^{-1}-->  I~  --Theorem B solve-->  e~
  --roots of q~(z) = sum (-1)^j e~_j z^{n-j}-->  round the real part of every root.
Three numbers per multiset m (absolute error |H~_nu - H_nu(m)| <= delta for nu = -1..n-2):
  delta_thm  : Theorem S4 closed form (the paper section 5), exact rational;
  delta_cert : exact componentwise certification (Proposition S5): rigorous and sharper;
  delta_up   : an explicit data vector at sup-distance delta_up from H(m) whose recovery has a
               root with real part exactly at a half-integer (rounding fails beyond it):
               delta_true <= delta_up.  So delta_cert <= delta_true <= delta_up.
Also the relative version: |dH_nu| <= eps |H_nu| for all nu, eps_cert.
Every inequality used in a certificate is evaluated in exact rational arithmetic.
Writes threshold_output.md and threshold_results.json.  Exits nonzero on any failure.
"""
import json
import os
from fractions import Fraction as Fr
from math import comb

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

from stab_common import (U_series, elementary, front_end, heat_direct, inf_norm, invariants, mat_inv, mat_mul, mat_vec,
                         ser_mul, ser_tan, ser_tanh, theorem_B_system)
from lipschitz_e import N_matrix, sigma
from roots_holder import H_from_e, roots_from_e

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 50

CASES = [(2, 8, 8), (3, 3, 12), (3, 10, 15, 30), (4, 5, 21, 28),
         (2, 3, 7), (4, 4, 4), (7, 7, 7), (3, 3, 4, 4), (5, 5, 5, 5), (2, 2, 2, 3), (2, 2, 2, 2, 3)]


def clusters(m):
    out = {}
    for x in m:
        out[x] = out.get(x, 0) + 1
    return out


# ---------------------------------------------------------------- Theorem S4 (closed form)
def delta_thm(m):
    n = len(m)
    mu = Fr(max(m))
    L, _ = front_end(n)
    Li = mat_inv(L)
    ell = [sum(abs(x) for x in row) for row in Li]
    lam = max([mu * ell[0]] + [ell[r] / mu ** (2 * r - 1) for r in range(1, n)])
    mh = [Fr(x) / mu for x in m]
    e = elementary(mh)
    M, _, _ = theorem_B_system(invariants(mh))
    Mi = inf_norm(mat_inv(M))
    s = sigma(n, max(2 * n - 3, 1))
    rho = max([sum(s[2 * i + 1] * e[2 * j - 2 * i] for i in range(j + 1) if 2 * j - 2 * i <= n)
               for j in range(n - 1)] + [e[n]])
    zeta = max([sum(s[2 * i + 1] for i in range(j) if 2 <= 2 * j - 2 * i <= n) for j in range(n - 1)] + [Fr(1)])
    cand = [1 / lam, 1 / (2 * Mi * zeta * lam)]
    cl = clusters(m)
    for a, k in cl.items():
        Q = Fr(1)
        for b, kb in cl.items():
            if b != a:
                Q *= (abs(Fr(a - b)) / mu) ** kb
        # r_a^k = 2^{1-k} 3^n eps / Q <= (2 mu)^{-k},  eps = 2 Mi rho lam delta
        cand.append(Q * Fr(2) ** (k - 1) / (Fr(3) ** n * (2 * mu) ** k * 2 * Mi * rho * lam))
    return min(cand), dict(lam=lam, Minv=Mi, rho=rho, zeta=zeta)


# ---------------------------------------------------------------- Proposition S5 (certificate)
def _common(m, dH):
    """Shared part of both certificates: exact bounds on dI, dT (linear + remainder), r, dM, E."""
    n = len(m)
    L, _ = front_end(n)
    Li = mat_inv(L)
    dI = [sum(abs(Li[r][c]) * dH[c] for c in range(n)) for r in range(n)]
    I = invariants(m)
    e = elementary(m)
    M, b, T = theorem_B_system(I)
    Mi = mat_inv(M)
    N = max(2 * n - 3, 1)
    U = U_series({2 * k - 1: I[k] for k in range(1, n)}, N)
    dU = U_series({2 * k - 1: dI[k] for k in range(1, n)}, N)
    sech2 = [Fr(int(i == 0)) - x for i, x in enumerate(ser_mul(T, T, N))]
    lin = ser_mul([abs(x) for x in sech2], dU, N)
    tU = ser_tan(U, N)
    tUd = ser_tan([x + y for x, y in zip(U, dU)], N)
    sec2 = [Fr(int(i == 0)) + x for i, x in enumerate(ser_mul(tU, tU, N))]
    rem = [a - b - c for a, b, c in zip(tUd, tU, ser_mul(sec2, dU, N))]
    assert all(x >= 0 for x in rem)
    dT = [x + y for x, y in zip(lin, rem)]
    r = [sum(dT[2 * i + 1] * e[2 * j - 2 * i] for i in range(j + 1) if 2 * j - 2 * i <= n) for j in range(n - 1)]
    r.append(dI[0] * e[n])
    r_rem = [sum(rem[2 * i + 1] * e[2 * j - 2 * i] for i in range(j + 1) if 2 * j - 2 * i <= n) for j in range(n - 1)]
    r_rem.append(Fr(0))
    dM = [[Fr(0)] * n for _ in range(n)]
    for j in range(n - 1):
        for i in range(j):
            idx = 2 * j - 2 * i
            if 1 <= idx <= n:
                dM[j][idx - 1] += dT[2 * i + 1]
    dM[n - 1][n - 1] += dI[0]
    aMi = [[abs(x) for x in row] for row in Mi]
    A = [[sum(aMi[i][k] * dM[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    IA = [[Fr(int(i == j)) - A[i][j] for j in range(n)] for i in range(n)]
    try:
        IAi = mat_inv(IA)
    except StopIteration:
        return None
    v = mat_vec(IAi, [Fr(1)] * n)
    if not all(x > 0 for x in v):          # certifies spectral radius of A < 1
        return None
    E = mat_vec(IAi, mat_vec(aMi, r))
    if not all(x >= 0 for x in E):
        return None
    return dict(n=n, L=L, Li=Li, I=I, e=e, M=M, Mi=Mi, aMi=aMi, A=A, E=E, r_rem=r_rem)


RADII = [Fr(1, 2)] + [Fr(j, 40) for j in range(19, 0, -1)] + [Fr(1, 100), Fr(1, 1000)]


def certify_abs(m, dH):
    """Proposition S5(i): componentwise |de| <= E, then Rouche with |dq(z)| <= sum E_j |z|^{n-j}."""
    C = _common(m, dH)
    if C is None:
        return False
    n, E = C["n"], C["E"]
    cl = clusters(m)
    for a, k in cl.items():
        ok = False
        for rad in RADII:
            low = rad ** k
            for bb, kb in cl.items():
                if bb != a:
                    low *= (abs(Fr(a - bb)) - rad) ** kb
            if low > sum(E[j - 1] * (a + rad) ** (n - j) for j in range(1, n + 1)):
                ok = True
                break
        if not ok:
            return False
    return True


def taylor_at(coeffs_desc, a):
    """Taylor coefficients (ascending in w) of p(a + w) for p given by descending coefficients."""
    deg = len(coeffs_desc) - 1
    asc = list(reversed(coeffs_desc))
    out = []
    for l in range(deg + 1):
        out.append(sum(Fr(comb(i, l)) * asc[i] * Fr(a) ** (i - l) for i in range(l, deg + 1)))
    return out


def certify_coherent(m, dH):
    """Proposition S5(ii): e~ - e = G dH + rho with G = (D_I e) L^{-1} exact and
    |rho| <= |M^{-1}| |r_rem| + A E; on |z - a| = rad,
    |dq(z)| <= sum_c dH_c sum_l |p_c^{(l)}(a)/l!| rad^l + sum_j |rho_j| (a + rad)^{n-j}."""
    C = _common(m, dH)
    if C is None:
        return False
    n, e, I, Mi, Li, aMi, A, E = C["n"], C["e"], C["I"], C["Mi"], C["Li"], C["aMi"], C["A"], C["E"]
    J = [[-x for x in row] for row in mat_mul(Mi, N_matrix(I, e))]      # D_I e
    G = mat_mul(J, Li)
    rho = [x + y for x, y in zip(mat_vec(aMi, C["r_rem"]), mat_vec(A, E))]
    cl = clusters(m)
    for a, k in cl.items():
        taylors = []
        for c in range(n):
            # p_c(z) = sum_j (-1)^j G[j-1][c] z^{n-j}, j = 1..n  (descending coefficients, leading 0)
            desc = [Fr(0)] + [(-1) ** j * G[j - 1][c] for j in range(1, n + 1)]
            taylors.append([abs(x) for x in taylor_at(desc, a)])
        ok = False
        for rad in RADII:
            low = rad ** k
            for bb, kb in cl.items():
                if bb != a:
                    low *= (abs(Fr(a - bb)) - rad) ** kb
            s1 = sum(dH[c] * sum(t * rad ** l for l, t in enumerate(taylors[c])) for c in range(n))
            s2 = sum(rho[j - 1] * (a + rad) ** (n - j) for j in range(1, n + 1))
            if low > s1 + s2:
                ok = True
                break
        if not ok:
            return False
    return True


def certify(m, dH):
    """True if every data vector with |H~_nu - H_nu(m)| <= dH[nu] is provably recovered exactly
    (either certificate of Proposition S5 suffices)."""
    return certify_coherent(m, dH) or certify_abs(m, dH)


def bisect(pred, lo, hi, iters=48):
    """largest x in [lo, hi] (geometric bisection on dyadic rationals) with pred(x) true; pred(lo) must hold."""
    assert pred(lo) and not pred(hi)
    for _ in range(iters):
        mid = Fr(float(mp.sqrt(mp.mpf(lo.numerator) / lo.denominator * mp.mpf(hi.numerator) / hi.denominator)))
        mid = Fr(mid).limit_denominator(10 ** 30)
        if mid <= lo or mid >= hi:
            break
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


# ---------------------------------------------------------------- upper bound by construction
def H_float_factory(n):
    L, h0 = front_end(n)
    Lf = np.array([[float(x) for x in row] for row in L])
    hf = np.array([float(x) for x in h0])
    N = 2 * n - 3

    def H(q):
        """q: monic coefficients, descending (numpy); returns float heat data."""
        e = [(-1) ** j * q[j] for j in range(n + 1)]
        E = lambda i: e[i] if 0 <= i <= n else 0.0
        p = [0.0] * (N + 1)
        for k in range(1, N + 1):
            s = (-1) ** (k - 1) * k * E(k)
            for i in range(1, k):
                s += (-1) ** (k - 1 + i) * E(k - i) * p[i]
            p[k] = s
        I = np.array([e[n - 1] / e[n]] + [p[2 * l - 1] for l in range(1, n)])
        return Lf @ I + hf
    return H


def exact_poly(c, y2, g):
    """monic (z - c) g(z) if y2 is None, else ((z - c)^2 + y2) g(z); exact, descending."""
    lin = [Fr(1), -c] if y2 is None else [Fr(1), -2 * c, c * c + y2]
    out = [Fr(0)] * (len(lin) + len(g) - 1)
    for i, x in enumerate(lin):
        for j, z in enumerate(g):
            out[i + j] += x * z
    return out


def minimax(fun_vec, x0):
    """min_x max_nu |fun_vec(x)_nu| in the epigraph form (SLSQP), then a Nelder-Mead polish.
    Nelder-Mead alone on the max-norm stalls at a kink."""
    t0 = float(np.max(np.abs(fun_vec(x0))))
    best = (t0, x0)
    cons = [{"type": "ineq", "fun": lambda z: z[-1] - fun_vec(z[:-1])},
            {"type": "ineq", "fun": lambda z: z[-1] + fun_vec(z[:-1])}]
    try:
        r = minimize(lambda z: z[-1], np.concatenate([x0, [t0]]), method="SLSQP", constraints=cons,
                     options=dict(maxiter=500, ftol=1e-16))
        v = float(np.max(np.abs(fun_vec(r.x[:-1]))))
        if v < best[0]:
            best = (v, r.x[:-1])
    except (ValueError, ArithmeticError):
        pass
    r = minimize(lambda x: float(np.max(np.abs(fun_vec(x)))), best[1], method="Nelder-Mead",
                 options=dict(xatol=1e-13, fatol=1e-18, maxiter=6000, maxfev=6000))
    if r.fun < best[0]:
        best = (r.fun, r.x)
    return best


def counterexample(m):
    """Search real monic q~ = (z - c) g(z), c = a +- 1/2, and ((z-c)^2 + y^2) g(z), minimising
    max_nu |H(q~) - H(m)|; return the best one, rebuilt in exact rationals (root real part exactly c).
    Candidates: Nelder-Mead on the coefficients of g, and the epigraph solve on coefficients scaled
    by those of the start."""
    n = len(m)
    Hf = H_float_factory(n)
    H0 = np.array([float(x) for x in heat_direct(m, n)])
    best = None
    for a in sorted(set(m)):
        for sgn in (1, -1):
            c = Fr(2 * a + sgn, 2)
            rest = list(m)
            rest.remove(a)
            for shape in ("real", "pair"):
                if shape == "pair" and m.count(a) < 2:
                    continue
                if shape == "real":
                    x0 = np.poly(rest)[1:].astype(float)
                else:
                    rest2 = list(rest)
                    rest2.remove(a)
                    x0 = np.concatenate([[0.1], np.poly(rest2)[1:].astype(float)]) if rest2 else np.array([0.1])

                def qf(x):
                    if shape == "real":
                        return np.polymul([1.0, -float(c)], np.concatenate([[1.0], x]))
                    return np.polymul([1.0, -2 * float(c), float(c) ** 2 + x[0] ** 2], np.concatenate([[1.0], x[1:]]))

                def obj(x):
                    q = qf(x)
                    if abs(q[n]) < 1e-300:
                        return 1e30
                    return float(np.max(np.abs(Hf(q) - H0)))

                res = None
                for start in (x0, x0 * (1 + 1e-3)):
                    r1 = minimize(obj, start, method="Nelder-Mead",
                                  options=dict(xatol=1e-13, fatol=1e-17, maxiter=40000, maxfev=80000))
                    r1 = minimize(obj, r1.x, method="Nelder-Mead",
                                  options=dict(xatol=1e-15, fatol=1e-19, maxiter=40000, maxfev=80000))
                    if res is None or r1.fun < res.fun:
                        res = r1
                xbest = res.x
                sc = np.where(np.abs(x0) > 0, np.abs(x0), 1.0)
                v, y = minimax(lambda y: Hf(qf(x0 + sc * y)) - H0, np.zeros(len(x0)))
                if v < res.fun:
                    xbest = x0 + sc * y
                x = [Fr(float(v)).limit_denominator(10 ** 15) for v in xbest]
                if shape == "real":
                    q = exact_poly(c, None, [Fr(1)] + x)
                else:
                    q = exact_poly(c, x[0] ** 2, [Fr(1)] + x[1:])
                e = [(-1) ** j * q[j] for j in range(n + 1)]
                if e[n] == 0:
                    continue
                d = max(abs(u - v) for u, v in zip(H_from_e(e, n), heat_direct(m, n)))
                if best is None or d < best[0]:
                    best = (d, e, a, sgn, shape)
    d, e, a, sgn, shape = best
    # exact: recovery from H(q~) returns q~ itself (Theorem B), whose roots include real part a + sgn/2
    L, h0 = front_end(n)
    I = mat_vec(mat_inv(L), [u - v for u, v in zip(H_from_e(e, n), h0)])
    M, b, _ = theorem_B_system(I)
    assert mat_vec(mat_inv(M), b) == e[1:]
    rts = roots_from_e(e)
    assert min(abs(mp.re(z) - (a + mp.mpf(sgn) / 2)) for z in rts) < mp.mpf(10) ** -30
    return d, a, sgn, shape


def fmt_down(x, d=4):
    """decimal string <= x (certified lower bounds are printed rounded down)."""
    import math
    x = Fr(x)
    e = math.floor(math.log10(float(x)))
    s = Fr(10) ** (e - d + 1)
    v = (x / s).__floor__() * s
    assert v <= x
    return f"{float(v):.{d - 1}e}"


def fmt_up(x, d=4):
    """decimal string >= x (upper bounds are printed rounded up)."""
    import math
    x = Fr(x)
    e = math.floor(math.log10(float(x)))
    s = Fr(10) ** (e - d + 1)
    v = (x / s).__ceil__() * s
    assert v >= x
    return f"{float(v):.{d - 1}e}"


def main():
    out = ["# T4: exact integer recovery thresholds (generated by threshold.py)\n",
           "Absolute model: |H~_nu - H_nu(m)| <= delta for nu = -1..n-2. "
           "delta_thm <= delta_cert <= delta_true <= delta_up.\n",
           "| m | n | delta_thm | delta_cert | delta_up | worst root at | delta_cert/|H_nu| (nu = -1..n-2) | eps_cert (relative) |",
           "|---|---|---|---|---|---|---|---|"]
    results = {}
    for m in CASES:
        n = len(m)
        R = sum(Fr(1, x) for x in m)
        assert n - 2 - R > 0, m                                   # hyperbolic
        H = heat_direct(m, n)
        dthm, parts = delta_thm(m)
        assert certify(m, [dthm] * n), m                          # the closed form is certified too
        lo, hi = bisect(lambda d: certify(m, [d] * n), dthm, Fr(1))
        dcert = lo
        assert certify(m, [dcert] * n)
        assert dcert >= dthm
        # relative model
        rlo, _ = bisect(lambda eps: certify(m, [eps * abs(h) for h in H]), Fr(1, 10 ** 30), Fr(1))
        assert certify(m, [rlo * abs(h) for h in H])
        dup, a, sgn, shape = counterexample(m)
        assert dup >= dcert, (m, dup, dcert)
        rel = [float(dcert / abs(h)) for h in H]
        results[str(m)] = dict(delta_thm=float(dthm), delta_cert=float(dcert), delta_cert_exact=str(dcert),
                               delta_up=float(dup), worst=dict(order=a, side=sgn, shape=shape),
                               rel_precision=rel, eps_cert=float(rlo), H=[str(h) for h in H],
                               lam=float(parts["lam"]), Minv=float(parts["Minv"]), rho=float(parts["rho"]))
        out.append(f"| {m} | {n} | {fmt_down(dthm)} | {fmt_down(dcert)} | {fmt_up(dup)} | "
                   f"{a}{'+' if sgn > 0 else '-'}1/2 ({shape}) | " + ", ".join(f"{x:.2e}" for x in rel)
                   + f" | {float(rlo):.3e} |")
        print(out[-1], flush=True)
    out.append("\nH_nu (exact) for reference:\n")
    for m in CASES:
        out.append(f"- {m}: " + ", ".join(f"{float(h):.6g}" for h in heat_direct(m, len(m))))
    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "threshold_output.md"), "w") as f:
        f.write(text)
    with open(os.path.join(HERE, "threshold_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(text)
    print("threshold.py: all asserts passed")


if __name__ == "__main__":
    main()

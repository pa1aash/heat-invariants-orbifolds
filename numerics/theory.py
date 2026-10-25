"""Predicted heat-trace data for the pillows O(p,q,r), K = -1.

Conventions (Section 2 of the paper; DGGW 2008; Schueth
2019; Ucar 2017): for a pillow (closed orbifold, three cone points)

    Z(t) = sum_j exp(-lambda_j t)
         ~ (4 pi t)^{-1} sum_l a_l^sm t^l  +  sum_cones sum_l b_l(C) t^l .

Smooth part (Ucar thesis eq. (4.35), constant curvature kappa):
    a_nu^sm = vol / (nu! 4^nu) * sum_{l=0}^{nu} binom(nu,l) (-4)^l B_{2l}(1/2) * kappa^nu.
Cone point of order k (Ucar eqs. (4.25), (4.33), Theorem 4.20(ii)):
    c^S_l = 1/(4k) (-1)^l/(l+1)! 1/(2l+1) sum_{j=0}^{l+1} binom(2l+2,2j) (k^{2j}-1) B_{2j} B_{2l+2-2j}(1/2)
    b_nu  = sum_{l=0}^{nu} 2/(4^l l!) c^S_{nu-l} * kappa^nu.
The nu = 0, 1, 2 cases are cross-checked below against paper eqs. (2)/(4) (DGGW
5.6, Schueth Remark 4.2) and Schueth Theorem 4.1; the check is an assert.

Exact elliptic terms (Selberg trace formula for orbisurfaces, Dryden-Strohmaier
eq. (1)) with h(r) = exp(-t(1/4 + r^2)):
    E_k(t) = sum_{l=1}^{k-1} 1/(2k sin(pi l/k)) * int_R exp(-2 theta_l r)/(1+exp(-2 pi r)) h(r) dr,
    theta_l = pi l / k.
Its small-t Taylor coefficients must reproduce b_nu (checked numerically).

The difference D(t) = Z_(2,8,8)(t) - Z_(3,3,12)(t): areas, a_l^sm, b_0 sums all
coincide, so
    D(t) = sum_{nu>=1} c_nu t^nu  (asymptotic)  + hyperbolic (closed geodesic) terms,
    c_nu = sum_cones b_nu(2,8,8) - sum_cones b_nu(3,3,12).
Sign convention: D = Z_(2,8,8) - Z_(3,3,12); kappa = K = -1; eigenvalues of the
positive Laplacian.
"""

from fractions import Fraction as Fr
from math import comb, factorial

import mpmath as mp
import sympy as sp

KAPPA = -1


# ---------------------------------------------------------------- exact rationals
def bern(n):
    return Fr(sp.bernoulli(n).p, sp.bernoulli(n).q) if n != 1 else Fr(-1, 2)


def bern_half(n):
    v = sp.bernoulli(n, sp.Rational(1, 2))
    return Fr(int(v.p), int(v.q))


def cS(l, k):
    s = sum(comb(2 * l + 2, 2 * j) * (Fr(k) ** (2 * j) - 1) * bern(2 * j) * bern_half(2 * l + 2 - 2 * j)
            for j in range(l + 2))
    return Fr(1, 4 * k) * Fr((-1) ** l, factorial(l + 1)) * Fr(1, 2 * l + 1) * s


def b_cone(nu, k, kappa=KAPPA):
    """Ucar (4.33): coefficient of t^nu contributed by one cone point of order k."""
    return sum(Fr(2, 4 ** l * factorial(l)) * cS(nu - l, k) for l in range(nu + 1)) * Fr(kappa) ** nu


def a_smooth_over_vol(nu, kappa=KAPPA):
    """Ucar (4.35) divided by vol."""
    s = sum(comb(nu, l) * Fr(-4) ** l * bern_half(2 * l) for l in range(nu + 1))
    return s / (factorial(nu) * 4 ** nu) * Fr(kappa) ** nu


def check_against_paper():
    """nu = 0, 1, 2 from Ucar must equal paper eq. (2)/(4) (DGGW, Schueth Rem. 4.2)
    and Schueth Theorem 4.1 at constant curvature."""
    for k in range(2, 40):
        b0 = Fr(k * k - 1, 12 * k)
        b1 = (Fr(1, 360) * (Fr(k) ** 3 - Fr(1, k)) + Fr(1, 36) * (k - Fr(1, k))) * KAPPA
        b2 = (Fr(1, 2520) * (Fr(k) ** 5 - Fr(1, k)) + Fr(1, 720) * (Fr(k) ** 3 - Fr(1, k))
              + Fr(1, 180) * (k - Fr(1, k))) * KAPPA ** 2
        assert b_cone(0, k) == b0, k
        assert b_cone(1, k) == b1, k
        assert b_cone(2, k) == b2, k
    # smooth: a_0 = vol, a_1 = vol*kappa/3 (gives chi/6 via Gauss-Bonnet), a_2 = vol*kappa^2/15
    assert a_smooth_over_vol(0) == 1
    assert a_smooth_over_vol(1) == Fr(KAPPA, 3)
    assert a_smooth_over_vol(2) == Fr(1, 15)
    return True


def power_sums(pqr):
    R = sum(Fr(1, m) for m in pqr)
    return dict(R=R, S1=sum(pqr), P3=sum(m ** 3 for m in pqr), P5=sum(m ** 5 for m in pqr),
                P7=sum(m ** 7 for m in pqr))


def pillow_coeffs(pqr, numax=8):
    """Exact data for O(p,q,r): area/pi, and the full coefficient of t^nu,
    nu = -1, 0, 1, ...  (smooth + cones)."""
    R = sum(Fr(1, m) for m in pqr)
    area_over_pi = 2 * (1 - R)         # Area = 2 pi (1 - R), eq. (1)
    coef = {}
    # t^{-1}: Area/(4 pi);  t^{nu}: a^sm_{nu+1}/(4 pi) + sum b_nu
    coef[-1] = area_over_pi / 4         # Area/(4 pi) = (1-R)/2
    for nu in range(0, numax + 1):
        sm = area_over_pi * a_smooth_over_vol(nu + 1) / 4
        coef[nu] = sm + sum(b_cone(nu, m) for m in pqr)
    return dict(R=R, area_over_pi=area_over_pi, coef=coef)


PAIR = ((2, 8, 8), (3, 3, 12))


def predicted_difference(numax=8):
    c = {}
    A, B = (pillow_coeffs(x, numax) for x in PAIR)
    for nu in range(-1, numax + 1):
        c[nu] = A["coef"][nu] - B["coef"][nu]
    return c


# ---------------------------------------------------------------- exact elliptic term
def elliptic_term(k, t, dps=30):
    """Dryden-Strohmaier eq. (1), elliptic part for one cone point of order k,
    with h(r) = exp(-t (1/4 + r^2))."""
    with mp.workdps(dps):
        t = mp.mpf(t)
        tot = mp.mpf(0)
        for l in range(1, k):
            th = mp.pi * l / k
            f = lambda r: mp.exp(-2 * th * r - t * (mp.mpf(1) / 4 + r * r)) / (1 + mp.exp(-2 * mp.pi * r))
            # integrand ~ exp(-(2 pi - 2 th) |r|) for r -> -inf, exp(-2 th r) for r -> +inf
            I = mp.quad(f, [-mp.inf, -20, -5, 0, 5, 20, mp.inf])
            tot += I / (2 * k * mp.sin(th))
        return tot


def elliptic_difference(t, dps=30):
    return (sum(elliptic_term(k, t, dps) for k in PAIR[0])
            - sum(elliptic_term(k, t, dps) for k in PAIR[1]))


def identity_term(area, t, dps=30):
    """Selberg identity term Area/(4 pi) int r h(r) tanh(pi r) dr."""
    with mp.workdps(dps):
        t = mp.mpf(t)
        f = lambda r: r * mp.exp(-t * (mp.mpf(1) / 4 + r * r)) * mp.tanh(mp.pi * r)
        return area / (4 * mp.pi) * 2 * mp.quad(f, [0, 1, 5, mp.inf])


def elliptic_taylor(k, nu, dps=40):
    """t^nu Taylor coefficient of E_k(t) at t = 0: differentiate h under the
    integral, (1/nu!) int (-(1/4 + r^2))^nu exp(-2 theta r)/(1 + exp(-2 pi r)) dr
    (absolutely convergent: the kernel decays exponentially in both directions)."""
    with mp.workdps(dps):
        tot = mp.mpf(0)
        for l in range(1, k):
            th = mp.pi * l / k
            f = lambda r: (-(mp.mpf(1) / 4 + r * r)) ** nu * mp.exp(-2 * th * r) / (1 + mp.exp(-2 * mp.pi * r))
            I = mp.quad(f, [-mp.inf, -40, -10, 0, 10, 40, mp.inf])
            tot += I / (2 * k * mp.sin(th))
        return tot / mp.factorial(nu)


def check_elliptic_expansion(numax=4):
    """The Taylor coefficients of the exact elliptic term E_k(t) at t = 0 must
    equal Ucar's b_nu(k) at kappa = -1.  This ties the trace-formula
    normalisation to the heat-coefficient one and fixes the sign convention
    (b_nu carries kappa^nu = (-1)^nu)."""
    out = {}
    for k in (2, 3, 8, 12):
        for nu in range(numax + 1):
            co = elliptic_taylor(k, nu)
            ex = b_cone(nu, k)
            exm = mp.mpf(ex.numerator) / ex.denominator
            assert abs(co - exm) < mp.mpf(10) ** -15 * max(1, abs(exm)), (k, nu, co, ex)
            out[(k, nu)] = (mp.nstr(co, 20), str(ex))
    return out


# ---------------------------------------------------------------- closed geodesics
def shortest_geodesics(pqr, maxlen=12, nmax=6):
    """Translation lengths of hyperbolic elements of the triangle group
    Delta(p,q,r) (orientation-preserving), found by enumerating words in the
    rotation generators.  l = 2 arccosh(|tr|/2) for an SU(1,1) matrix.
    Returns the smallest distinct lengths (used only to estimate the size of
    the hyperbolic terms in the Selberg trace formula)."""
    import numpy as np
    from geometry import Triangle
    T = Triangle(*pqr)

    def rot(center, angle):
        a = complex(center)
        s = 1 / np.sqrt(1 - abs(a) ** 2)
        Tm = s * np.array([[1, a], [a.conjugate(), 1]])
        Ti = s * np.array([[1, -a], [-a.conjugate(), 1]])
        Rm = np.array([[np.exp(0.5j * angle), 0], [0, np.exp(-0.5j * angle)]])
        return Tm @ Rm @ Ti

    p, q, r = pqr
    gens = []
    for c, m in ((T.A, p), (T.B, q), (T.C, r)):
        g = rot(complex(c), 2 * np.pi / m)
        gens += [g, np.linalg.inv(g)]
    seen = {}
    frontier = [np.eye(2, dtype=complex)]
    lengths = set()
    for depth in range(maxlen):
        new = []
        for g in frontier:
            for h in gens:
                x = g @ h
                key = (round(x[0, 0].real, 8), round(x[0, 0].imag, 8), round(x[0, 1].real, 8),
                       round(x[0, 1].imag, 8))
                key2 = tuple(-v for v in key)
                if key in seen or key2 in seen:
                    continue
                seen[key] = True
                new.append(x)
                tr = abs(np.trace(x).real)
                if tr > 2 + 1e-9:
                    lengths.add(round(2 * np.arccosh(tr / 2), 9))
        frontier = new
        if len(seen) > 400000:
            break
    return sorted(lengths)[:nmax]


if __name__ == "__main__":
    check_against_paper()
    for pqr in PAIR:
        ps = power_sums(pqr)
        pc = pillow_coeffs(pqr, 4)
        print(pqr, {k: str(v) for k, v in ps.items()})
        print("   Area/pi =", pc["area_over_pi"], " coef:", {k: str(v) for k, v in pc["coef"].items()})
    c = predicted_difference(6)
    print("D(t) coefficients:", {k: f"{v} = {float(v):.10g}" for k, v in c.items()})

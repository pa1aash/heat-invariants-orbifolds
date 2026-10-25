"""Exact checks for the paper.  Every check is an assert.

(A) T1, identity term.  The asymptotic series of
        I(t) = Area/(4 pi) [ e^{-t/4}/t - 4 int_0^inf r e^{-t(1/4+r^2)}/(e^{2 pi r}+1) dr ]
    has t^{nu-1} coefficient (Area/4pi) alpha_nu with alpha_nu from Ucar (4.35) at kappa = -1
    (numerics/theory.py, a_smooth_over_vol).  Uses the Fermi-Dirac moments
        J_k = int_0^inf r^{2k+1}/(e^{2 pi r}+1) dr = (1 - 2^{-2k-1}) (-1)^k B_{2k+2} / (4(k+1)),
    themselves checked against mpmath quadrature at 50 digits.  Exact rationals, nu <= 15.
(B) T1, elliptic term.  The t^nu Taylor coefficient of E_m(t) is
        (-1)^nu/(nu! 4^nu) sum_{l=1}^{m-1} 1/(2m sin th_l) sum_k binom(nu,k) D^{2k}[1/(2 sin u)](th_l),
    th_l = pi l/m (from int_R e^{-a r}/(1+e^{-2 pi r}) dr = 1/(2 sin(a/2)), 0 < a < 2pi).
    Evaluated exactly in the cyclotomic field Q(w), w = e^{i pi/(2m)} (so e^{i th_l} = w^{2l},
    i = w^m), as polynomials modulo Phi_{4m}; asserted to equal Ucar's beta_nu(m) =
    b_cone(nu, m) of numerics/theory.py.  nu <= 5, m in {2,3,4,5,6,8,12}.
(C) T2.  dim T = -3 chi(X_O) + 2k (Thurston 13.3.7, no corner reflectors) = 6g - 6 + 2n; zero set
    among hyperbolic signatures is exactly (0; 3 cone points), and the value is even.
(D) T3.  Fourier normalisation of the heat test function (and the Marklof (192) misprint);
    completing the square; the closed form of int_{l-t}^inf (u+t) e^{-u^2/4t} du; the
    monotonicity conditions; the bound of step 3 against direct quadrature at random points;
    the full bound of Theorem 3.4(b) against a synthetic worst case.

usage: python check_locality.py         (about one minute)
"""

import os
import random
import sys
from fractions import Fraction as Fr
from math import comb, factorial

import mpmath as mp
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "numerics"))
import theory as th  # noqa: E402  (S3 module, unmodified: Ucar (4.25), (4.33), (4.35))

mp.mp.dps = 50


def bern(n):
    b = sp.bernoulli(n)
    return Fr(int(b.p), int(b.q)) if n != 1 else Fr(-1, 2)


# ---------------------------------------------------------------- (A)
def J(k):
    return (1 - Fr(1, 2 ** (2 * k + 1))) * (-1) ** k * bern(2 * k + 2) / (4 * (k + 1))


def check_identity_term(numax=15):
    for k in range(4):
        q = mp.quad(lambda r: r ** (2 * k + 1) / (mp.exp(2 * mp.pi * r) + 1), [0, 1, 5, mp.inf])
        e = J(k)
        assert abs(q - mp.mpf(e.numerator) / e.denominator) < mp.mpf(10) ** -45, (k, q, e)
    for nu in range(numax + 1):
        # t^{nu-1} coefficient of the bracket: from e^{-t/4}/t and from the integral (index nu-1)
        c = Fr(-1, 4) ** nu / factorial(nu)
        if nu >= 1:
            n1 = nu - 1
            mom = sum(comb(n1, k) * Fr(1, 4) ** (n1 - k) * J(k) for k in range(n1 + 1))
            c += -4 * Fr((-1) ** n1, factorial(n1)) * mom
        assert c == th.a_smooth_over_vol(nu), (nu, c, th.a_smooth_over_vol(nu))
    print(f"(A) identity term: series coefficients = Ucar alpha_nu exactly, nu = 0..{numax}")


# ---------------------------------------------------------------- (B)
x = sp.symbols("x")
u = sp.symbols("u", real=True)
z = sp.symbols("z")


def _csc_derivs(kmax):
    """D^{2k}[1/(2 sin u)] as rational functions of z = e^{iu}, with I kept symbolic."""
    f = 1 / (2 * sp.sin(u))
    out = []
    for k in range(kmax + 1):
        d = sp.diff(f, u, 2 * k)
        d = d.rewrite(sp.exp).subs(sp.exp(sp.I * u), z)
        out.append(sp.together(sp.simplify(d.subs(u, -sp.I * sp.log(z)))))
    return out


def _in_field(expr, zval_pow, m):
    """Evaluate a rational function of z and I at z = w^zval_pow, I = w^m, in Q[x]/Phi_{4m}."""
    phi = sp.Poly(sp.cyclotomic_poly(4 * m, x), x, domain="QQ")
    e = sp.together(expr)
    num, den = sp.fraction(e)
    sub = {z: x ** zval_pow, sp.I: x ** m}
    N = sp.Poly(sp.expand(num.subs(sub)), x, domain="QQ").rem(phi)
    D = sp.Poly(sp.expand(den.subs(sub)), x, domain="QQ").rem(phi)
    Dinv = sp.Poly(sp.invert(D.as_expr(), phi.as_expr(), x), x, domain="QQ")
    return (N * Dinv).rem(phi), phi


def check_elliptic_term(numax=5, orders=(2, 3, 4, 5, 6, 8, 12)):
    derivs = _csc_derivs(numax)
    sin_z = (z - 1 / z) / (2 * sp.I)
    for m in orders:
        for nu in range(numax + 1):
            inner = sum(comb(nu, k) * derivs[k] for k in range(nu + 1))
            term = inner / (2 * m * sin_z)
            tot = None
            for l in range(1, m):
                val, phi = _in_field(term, 2 * l, m)
                tot = val if tot is None else (tot + val).rem(phi)
            assert tot.degree() <= 0, (m, nu, tot)
            c = sp.Rational(tot.as_expr()) * sp.Rational((-1) ** nu, factorial(nu) * 4 ** nu)
            b = th.b_cone(nu, m)
            assert c == sp.Rational(b.numerator, b.denominator), (m, nu, c, b)
    print(f"(B) elliptic term: Taylor coefficients = Ucar beta_nu(m) exactly, nu <= {numax}, m in {orders}")


# ---------------------------------------------------------------- (C)
def check_dimension():
    zero = set()
    for g in range(0, 7):
        for n in range(0, 12):
            for ms in ([2] * n, [3] * n, [7] * n, [2, 3, 7, 5, 11, 4, 9, 13, 6, 8, 10][:n]):
                chi = 2 - 2 * g - sum(1 - Fr(1, m) for m in ms)
                if chi >= 0:
                    continue                      # not hyperbolic
                d = -3 * (2 - 2 * g) + 2 * n      # Thurston 13.3.7 with k = n, l = 0
                assert d == 6 * g - 6 + 2 * n and d % 2 == 0 and d >= 0
                if d == 0:
                    zero.add((g, n))
    assert zero == {(0, 3)}, zero
    print("(C) dim T = 6g-6+2n >= 0, even; zero exactly for (g,n) = (0,3) among hyperbolic signatures")


# ---------------------------------------------------------------- (D)
def check_t3():
    t, r, uu, l, b = sp.symbols("t r u ell beta", positive=True)
    rr = sp.symbols("rho", real=True)
    g = sp.exp(-t / 4) * sp.exp(-uu ** 2 / (4 * t)) / sp.sqrt(4 * sp.pi * t)
    h = sp.integrate(g * sp.exp(sp.I * rr * uu), (uu, -sp.oo, sp.oo))
    assert sp.simplify(h - sp.exp(-t * (sp.Rational(1, 4) + rr ** 2))) == 0
    # g(u) = (1/2pi) int h e^{-iru} dr  (the DS normalisation)
    g2 = sp.integrate(sp.exp(-t * (sp.Rational(1, 4) + rr ** 2)) * sp.exp(-sp.I * rr * uu), (rr, -sp.oo, sp.oo)) / (2 * sp.pi)
    assert sp.simplify(g2 - g) == 0
    # Marklof (192): h(rho) = e^{-beta rho^2} has g(t) = e^{-t^2/(4 beta)}/sqrt(4 pi beta); the
    # printed exponent -t^2/(2 beta) is inconsistent with the printed prefactor
    gM = sp.integrate(sp.exp(-b * rr ** 2) * sp.exp(-sp.I * rr * uu), (rr, -sp.oo, sp.oo)) / (2 * sp.pi)
    assert sp.simplify(gM - sp.exp(-uu ** 2 / (4 * b)) / sp.sqrt(4 * sp.pi * b)) == 0
    assert sp.simplify(gM - sp.exp(-uu ** 2 / (2 * b)) / sp.sqrt(4 * sp.pi * b)) != 0
    # completing the square
    L = sp.symbols("L", positive=True)
    assert sp.simplify((L / 2 - L ** 2 / (4 * t)) - (-(L - t) ** 2 / (4 * t) + t / 4)) == 0
    # int_{l-t}^inf (u+t) e^{-u^2/4t} du = 2t e^{-(l-t)^2/4t} + t sqrt(pi t) erfc((l-t)/(2 sqrt t))
    a = sp.symbols("a", positive=True)          # a = l - t > 0
    rhs = 2 * t * sp.exp(-a ** 2 / (4 * t)) + t * sp.sqrt(sp.pi * t) * sp.erfc(a / (2 * sp.sqrt(t)))
    # d/da rhs = -(integrand at a), and rhs -> 0 as a -> oo: rhs is the tail integral
    assert sp.simplify(sp.diff(rhs, a) + (a + t) * sp.exp(-a ** 2 / (4 * t))) == 0
    assert sp.limit(rhs, a, sp.oo) == 0
    for av, tv in ((sp.Rational(7, 10), sp.Rational(1, 20)), (sp.Integer(2), sp.Rational(3, 10))):
        am, tm = mp.mpf(av.p) / av.q, mp.mpf(tv.p) / tv.q
        q = mp.quad(lambda w: (w + tm) * mp.exp(-w * w / (4 * tm)), [am, am + 1, mp.inf])
        assert abs(q - mp.mpf(str(rhs.subs({a: av, t: tv}).evalf(60)))) < mp.mpf(10) ** -40
    # e^{t/4} e^{-(l-t)^2/4t} = e^{l/2} e^{-l^2/4t}
    assert sp.simplify(sp.exp(t / 4 - (l - t) ** 2 / (4 * t)) - sp.exp(l / 2 - l ** 2 / (4 * t))) == 0
    # monotonicity: for t <= l^2/(2(1+l)):  l/(2t) >= 1/l + 1  (>= 1/x + 1/2 for x >= l)
    tmax = l ** 2 / (2 * (1 + l))
    assert sp.simplify(l / (2 * tmax) - (1 / l + 1)) == 0
    assert sp.simplify(tmax - l ** 2 / 2) != 0 and sp.simplify((l ** 2 / 2 - tmax) - l ** 3 / (2 * (1 + l))) == 0
    # step 3 against quadrature, and the full single-orbifold bound against a synthetic worst
    # case N(L) = (pi/A) e^{L+3 delta} (continuous density, the extreme the proof allows)
    rnd = random.Random(20261001)
    for _ in range(40):
        ell = mp.mpf(rnd.uniform(0.3, 4.0))
        tt = mp.mpf(rnd.uniform(0.02, 1.0)) * ell ** 2 / (2 * (1 + ell))
        phi = lambda X: X * mp.exp(-X / 2 - X * X / (4 * tt)) / mp.sqrt(4 * mp.pi * tt)
        lhs = mp.quad(lambda X: mp.exp(X) * phi(X), [ell, ell + 1, mp.inf])
        bnd = 2 * tt * ell / (ell - tt) * mp.exp(ell / 2 - ell ** 2 / (4 * tt)) / mp.sqrt(4 * mp.pi * tt)
        assert lhs <= bnd * (1 + mp.mpf(10) ** -30), (ell, tt, lhs, bnd)
        A = mp.mpf(rnd.uniform(0.5, 20))
        dl = mp.mpf(rnd.uniform(0.2, 3))
        dens = lambda X: mp.pi / A * mp.exp(X + 3 * dl)            # dN/dL of the synthetic count
        Hs = mp.quad(lambda X: X / (2 * mp.sinh(X / 2)) * mp.exp(-tt / 4 - X * X / (4 * tt))
                     / mp.sqrt(4 * mp.pi * tt) * dens(X), [ell, ell + 1, mp.inf])
        # the synthetic count jumps by N(ell) at ell: include that atom
        Hs += ell / (2 * mp.sinh(ell / 2)) * mp.exp(-tt / 4 - ell * ell / (4 * tt)) / mp.sqrt(4 * mp.pi * tt) \
            * mp.pi / A * mp.exp(ell + 3 * dl)
        full = (mp.pi * mp.exp(3 * dl) / (A * (1 - mp.exp(-ell))) * ell * mp.exp(ell / 2)
                * (1 + 2 * tt / (ell - tt)) * mp.exp(-ell ** 2 / (4 * tt)) / mp.sqrt(4 * mp.pi * tt))
        assert Hs <= full * (1 + mp.mpf(10) ** -30), (ell, tt, Hs, full)
    print("(D) heat test function normalisation (DS), Marklof (192) exponent misprint, completing the "
          "square, erfc closed form, monotonicity window, step-3 and Theorem 3.4(b) bounds: OK")


if __name__ == "__main__":
    assert th.check_against_paper()
    check_identity_term()
    check_dimension()
    check_t3()
    check_elliptic_term()
    print("ALL LOCALITY CHECKS PASSED")

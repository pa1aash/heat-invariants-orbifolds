"""Exact heat coefficients of closed orientable hyperbolic 2-orbifolds (curvature -1).

Convention (DF.1 + Proposition heatinput, as given in the brief):
  tr e^{-t Delta} ~ sum_{j>=1} c_j t^{j-2},
  c_1 = Area/(4 pi),  c_j = alpha_{j-1} Area/(4 pi) + sum_i b_{j-2}(m_i)  (j >= 2),
  alpha_k = (-1)^k/(k! 4^k) sum_l C(k,l) (-4)^l B_{2l}(1/2),
  b_l(m) = (-1)^l p_l(m)/m,  p_l(m) = 4^{-l} sum_{k=0}^{l} (2k)!/(k!(l-k)!) m phi_k(m),
  Phi_m(u) = (cot u - m cot(m u))/(4 m sin u) = sum_k phi_k(m) u^{2k}.

phi_k(m) is computed here as an exact polynomial in m from the Laurent series
  cot u   = sum_{n>=0} c_n u^{2n-1},  c_n = (-1)^n 2^{2n} B_{2n}/(2n)!
  1/sin u = sum_{n>=0} d_n u^{2n-1},  d_n = (-1)^{n+1} 2 (2^{2n-1}-1) B_{2n}/(2n)!
so that cot u - m cot(mu) = sum_{n>=1} c_n (1-m^{2n}) u^{2n-1} and
  phi_k(m) = (1/(4m)) sum_{n=1}^{k+1} c_n d_{k+1-n} (1 - m^{2n}).
Independent cross-check: Ucar (4.25),(4.33)-(4.35) (function ucar_b, ucar_alpha).
All arithmetic is exact (Fraction / int).
"""
from fractions import Fraction as F
from math import comb, factorial
import sympy

def bern(n):
    # sympy >= 1.12 uses B_1 = +1/2; only even indices are used here.
    return F(sympy.Rational(sympy.bernoulli(n)).p, sympy.Rational(sympy.bernoulli(n)).q)

def bernpoly_half(n):
    v = sympy.Rational(sympy.bernoulli(n, sympy.Rational(1, 2)))
    return F(v.p, v.q)

def c_cot(n):
    return F((-1) ** n * 2 ** (2 * n), factorial(2 * n)) * bern(2 * n)

def d_csc(n):
    if n == 0:
        return F(1)
    return F((-1) ** (n + 1) * 2 * (2 ** (2 * n - 1) - 1), factorial(2 * n)) * bern(2 * n)

# polynomials are dicts {power: Fraction}
def m_phi(k):
    """m*phi_k(m) as polynomial in m."""
    poly = {}
    for n in range(1, k + 2):
        coef = c_cot(n) * d_csc(k + 1 - n) / 4
        poly[0] = poly.get(0, F(0)) + coef
        poly[2 * n] = poly.get(2 * n, F(0)) - coef
    return poly

_P = {}
def p_poly(l):
    if l in _P:
        return _P[l]
    poly = {}
    for k in range(l + 1):
        w = F(factorial(2 * k), factorial(k) * factorial(l - k) * 4 ** l)
        for e, c in m_phi(k).items():
            poly[e] = poly.get(e, F(0)) + w * c
    poly = {e: c for e, c in poly.items() if c != 0}
    _P[l] = poly
    return poly

def peval(poly, x):
    return sum(c * x ** e for e, c in poly.items())

def b(l, m):
    return F((-1) ** l) * peval(p_poly(l), m) / m

def alpha(k):
    s = sum(comb(k, l) * F((-4) ** l) * bernpoly_half(2 * l) for l in range(k + 1))
    return F((-1) ** k, factorial(k) * 4 ** k) * s

# ---- Ucar cross-check (fetched text, arXiv:1711.03405 p.137, eqs (4.25),(4.33),(4.35))
def ucar_cS(l, k):
    s = sum(comb(2 * l + 2, 2 * j) * (k ** (2 * j) - 1) * bern(2 * j) * bernpoly_half(2 * l + 2 - 2 * j)
            for j in range(l + 2))
    return F(1, 4 * k) * F((-1) ** l, factorial(l + 1)) * F(1, 2 * l + 1) * s

def ucar_b(nu, k, kappa=-1):
    return sum(F(2, 4 ** l * factorial(l)) * ucar_cS(nu - l, k) for l in range(nu + 1)) * F(kappa) ** nu

def ucar_alpha(nu, kappa=-1):
    # a_nu = vol/(nu! 4^nu) sum C(nu,l)(-4)^l B_{2l}(1/2) kappa^nu ; heat term a_nu/(4 pi t) t^nu
    return F(1, factorial(nu) * 4 ** nu) * sum(comb(nu, l) * F((-4) ** l) * bernpoly_half(2 * l)
                                               for l in range(nu + 1)) * F(kappa) ** nu

# ---- orbifolds
def s_area(g, ms):
    """Area/(2 pi) = 2g-2+sum(1-1/m)."""
    return 2 * g - 2 + sum(1 - F(1, m) for m in ms)

def is_hyperbolic(g, ms):
    """Closed orientable 2-orbifold with cone orders ms (each >= 2), genus g >= 0:
    hyperbolic iff chi < 0 iff s_area > 0."""
    assert g >= 0 and all(isinstance(m, int) and m >= 2 for m in ms)
    return s_area(g, ms) > 0

def heat(g, ms, J):
    """[c_1, ..., c_J] exactly."""
    s = s_area(g, ms)
    A4pi = s / 2  # Area/(4 pi)
    out = [A4pi]
    for j in range(2, J + 1):
        out.append(alpha(j - 1) * A4pi + sum(b(j - 2, m) for m in ms))
    return out

def shared(g1, m1, g2, m2, Jmax):
    c1 = heat(g1, m1, Jmax)
    c2 = heat(g2, m2, Jmax)
    k = 0
    while k < Jmax and c1[k] == c2[k]:
        k += 1
    return k, c1, c2

def R(ms):
    return sum(F(1, m) for m in ms)

def P(ms, j):
    return sum(m ** j for m in ms)

def criterion_L(g1, m1, g2, m2, Kmax):
    """Number of shared coefficients predicted by [Sig] Lemma 4 / Lemma 2:
    equal area, and Psi_k = P_{2k-1} - R equal for k = 1..L-1; returns L (<= Kmax)."""
    if s_area(g1, m1) != s_area(g2, m2):
        return 0
    L = 1
    while L < Kmax and P(m1, 2 * L - 1) - R(m1) == P(m2, 2 * L - 1) - R(m2):
        L += 1
    return L

def config(g1, m1, g2, m2):
    """Lemma 4 padding + cancellation; returns (Ustar, Vstar)."""
    n1, n2 = len(m1), len(m2)
    d = R(m2) - R(m1)
    assert d.denominator == 1
    d = int(d)
    U = list(m1) + [1] * max(d, 0)
    V = list(m2) + [1] * max(-d, 0)
    from collections import Counter
    cu, cv = Counter(U), Counter(V)
    common = cu & cv
    return sorted((cu - common).elements()), sorted((cv - common).elements())

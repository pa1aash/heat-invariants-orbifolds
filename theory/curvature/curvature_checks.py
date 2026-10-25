#!/usr/bin/env python3
"""Exact checks for the curvature comparison (the paper in this directory).

FLAT (K = 0)
  F1  The genus-0 cone data with sum(1 - 1/m_i) = 2 are exactly
      (2,2,2,2), (3,3,3), (2,4,4), (2,3,6); with the torus these are the
      closed orientable flat 2-orbifolds (Thurston, Thm 13.3.6).
  F2  Their degree-0 heat coefficients sum_i (m_i^2-1)/(12 m_i) are
      0, 1/2, 2/3, 3/4, 5/6: pairwise distinct (DGGW Table 1 values).
  F3  At K = 0 every cone coefficient beyond l = 0 vanishes:
      b_l = K^l (1/m) p_l(m) with p_l from Ucar (4.25)+(4.33).
  F4  A flat cone sphere that is not an orbifold has the same heat
      coefficients as the flat orbifold S^2(2,4,4): doubled triangles with
      angles pi(1/5,2/5,2/5) and pi(1/4,1/4,1/2), scaled to equal area
      (Kokotov, Thm 1: Area/(4 pi t) + (1/12) sum(2pi/beta - beta/2pi)).

SPHERICAL (K = +1)
  S1  For a finite rotation group G with cone orders (m_1,...,m_k) of S^2/G,
      the G-invariant dimension in the degree-l harmonics is
          N_l = (2l+1)/|G| + (1/2) sum_i [2 floor(l/m_i) + 1 - (2l+1)/m_i],
      checked against explicit character sums over generated matrix groups.
  S2  Hence Z_G(t) = (chi/2) Z_{S^2}(t) + sum_i B_{m_i}(t) with
      B_m(t) = (1/2) sum_l [2 floor(l/m) + 1 - (2l+1)/m] e^{-l(l+1)t};
      the exact small-t expansion of B_m (Hurwitz zeta at negative
      integers) has t^l coefficient equal to Ucar's b_l(m) at K = +1, for
      l <= L_MAX and 2 <= m <= M_MAX.
  S3  The degree-0 coefficient c/12 separates all good closed orientable
      spherical 2-orbifolds (S^2, (n,n), (2,2,n), (2,3,3), (2,3,4), (2,3,5));
      the t^{-1} coefficient alone does not (chi(2,2,n) = chi(2n,2n) at K=1).
"""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction as F
from math import comb, factorial, floor

import mpmath as mp
import sympy as sp

L_MAX = 12
M_MAX = 30


# ------------------------------------------------------------ Bernoulli tools
_B: dict[int, F] = {0: F(1)}


def bern(n: int) -> F:
    if n not in _B:
        _B[n] = -sum(comb(n + 1, j) * bern(j) for j in range(n)) / (n + 1)
    return _B[n]


def bern_poly(n: int, x: F) -> F:
    return sum(comb(n, k) * bern(k) * x ** (n - k) for k in range(n + 1))


def hurwitz_neg(N: int, a: F) -> F:
    """zeta(-N, a) = -B_{N+1}(a)/(N+1), N >= 0."""
    return -bern_poly(N + 1, a) / (N + 1)


# ------------------------------------------------- Ucar's cone coefficients
def bern_half(n: int) -> F:
    return (F(1, 2 ** (n - 1)) - 1) * bern(n) if n >= 1 else F(1)


def c_ucar(l: int, k: int) -> F:
    s = sum(comb(2 * l + 2, 2 * j) * (F(k) ** (2 * j) - 1) * bern(2 * j) * bern_half(2 * l + 2 - 2 * j)
            for j in range(l + 2))
    return F((-1) ** l, 4 * factorial(l + 1) * (2 * l + 1)) * s / k


def b_ucar(l: int, m: int) -> F:
    """b_l(m) / K^l (Ucar (4.33)/(4.34))."""
    return sum(F(2, 4 ** i * factorial(i)) * c_ucar(l - i, m) for i in range(l + 1))


# -------------------------------------------------- exact series arithmetic
def exp_series(c: F, n: int) -> list[F]:
    """coefficients of e^{c t} up to t^n."""
    return [c ** k / factorial(k) for k in range(n + 1)]


def mul(a: list[F], b: list[F], n: int) -> list[F]:
    return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(n + 1)]


def B_cone_series(m: int, n: int) -> list[F]:
    """Small-t expansion of B_m(t) up to t^n (all coefficients of t^k, k >= 0).

    B_m(t) = (1/2) e^{t/4} sum_{r=0}^{m-1} f_r sum_{j>=0} exp(-t m^2 (j + a_r)^2),
    f_r = (m-1-2r)/m, a_r = (r + 1/2)/m, and
    sum_j exp(-s (j+a)^2) ~ sqrt(pi)/(2 sqrt s) + sum_k (-s)^k/k! zeta(-2k, a).
    The s^{-1/2} terms cancel because sum_r f_r = 0.
    """
    f = [F(m - 1 - 2 * r, m) for r in range(m)]
    a = [F(2 * r + 1, 2 * m) for r in range(m)]
    assert sum(f) == 0
    inner = [F(1, 2) * sum(f[r] * hurwitz_neg(2 * k, a[r]) for r in range(m)) * F((-m * m) ** k, factorial(k))
             for k in range(n + 1)]
    return mul(exp_series(F(1, 4), n), inner, n)


def sphere_series(n: int) -> list[F]:
    """t * Z_{S^2}(t) up to t^{n+1}: Z = e^{t/4} sum_j 2(j+1/2) e^{-t (j+1/2)^2}
    ~ e^{t/4} [1/t + 2 sum_k (-t)^k/k! zeta(-1-2k, 1/2)]."""
    inner = [F(1)] + [2 * F((-1) ** k, factorial(k)) * hurwitz_neg(1 + 2 * k, F(1, 2)) for k in range(n + 1)]
    return mul(exp_series(F(1, 4), n + 1), inner, n + 1)


# ------------------------------------------------------------- FLAT checks
def cone0(m: int) -> F:
    return F(m * m - 1, 12 * m)


def flat_checks() -> None:
    # F1: genus-0 cone data with sum (1 - 1/m) = 2. Each term >= 1/2, so n <= 4,
    # and the least order is bounded: searching m <= 60 is exhaustive because
    # for n = 3 the condition 1/a + 1/b + 1/c = 1 with a <= b <= c forces a <= 3,
    # b <= 6 and c <= 6 (the paper); n = 4 forces all orders 2; n <= 2 impossible.
    found = []
    assert 5 * F(1, 2) > 2          # n >= 5 cone points: sum(1 - 1/m) >= 5/2 > 2
    for n in range(1, 5):
        for ms in itertools.combinations_with_replacement(range(2, 61 if n < 4 else 13), n):
            if sum(1 - F(1, m) for m in ms) == 2:
                found.append(ms)
    assert sorted(found) == sorted([(2, 2, 2, 2), (3, 3, 3), (2, 4, 4), (2, 3, 6)]), found
    # F2: degree-0 terms (chi = 0, so only cone terms); DGGW Table 1 values
    vals = {(): F(0)}
    for ms in found:
        vals[ms] = sum(cone0(m) for m in ms)
    dggw = {(): F(0), (2, 2, 2, 2): F(1, 2), (2, 4, 4): F(3, 4), (3, 3, 3): F(2, 3), (2, 3, 6): F(5, 6)}
    assert vals == dggw, vals
    assert len(set(vals.values())) == len(vals)
    print("F1/F2 flat orientable: degree-0 terms", {k: str(v) for k, v in vals.items()})
    # F3: K^l factor kills every cone coefficient beyond l = 0
    K = sp.symbols("K")
    for l in range(1, 7):
        for m in range(2, 13):
            expr = K ** l * sp.Rational(b_ucar(l, m).numerator, b_ucar(l, m).denominator)
            assert expr.subs(K, 0) == 0 and b_ucar(l, m) != 0
    print("F3 at K = 0 the cone coefficients b_l, 1 <= l <= 6, vanish; at K != 0 none vanish (m <= 12)")
    # F4: doubled triangles. Cone angles beta_i = 2 alpha_i, alpha = pi x_i.
    # (1/12) sum (2pi/beta - beta/2pi) = (1/12) sum (1/x_i - x_i)
    def h(xs):
        assert sum(xs) == 1 and all(x > 0 for x in xs)
        return F(1, 12) * sum(1 / x - x for x in xs)
    orb = (F(1, 4), F(1, 4), F(1, 2))       # cone angles pi/2, pi/2, pi : S^2(4,4,2)
    non = (F(1, 5), F(2, 5), F(2, 5))       # cone angles 2pi/5, 4pi/5, 4pi/5
    assert h(orb) == h(non) == F(3, 4) == vals[(2, 4, 4)]
    assert not all((1 / (2 * x)).denominator == 1 for x in non)   # 4pi/5 is not 2pi/m
    assert sorted(orb) != sorted(non)
    # a whole curve: x + y + z = 1, 1/x + 1/y + 1/z = 10 is a conic in the simplex
    # eliminate: y + z = 1 - x, yz = (1 - x)/(10 - 1/x); two distinct positive
    # real y, z exist iff 10 - 1/x > 0 and the discriminant is positive
    def disc(x):
        s_, p_ = 1 - x, (1 - x) / (10 - 1 / x)
        return s_ * s_ - 4 * p_, p_
    xs_ok = []
    for k in range(1, 1000):
        xx = F(k, 1000)
        if xx * 10 > 1:
            d, p_ = disc(xx)
            if d > 0 and p_ > 0:
                xs_ok.append(xx)
    assert len(xs_ok) > 50, len(xs_ok)
    print("F4 doubled triangles pi(1/4,1/4,1/2) [orbifold S^2(2,4,4)] and pi(1/5,2/5,2/5) [not an",
          "orbifold] share every heat coefficient; the level set 1/x+1/y+1/z = 10 has interior",
          "points for", len(xs_ok), "of the sampled x = k/1000 (a curve, not isolated points)")


# --------------------------------------------------------- SPHERICAL checks
def rot(axis, ang):
    ax = mp.matrix(axis)
    ax = ax / mp.norm(ax)
    x, y, z = ax
    c, s = mp.cos(ang), mp.sin(ang)
    C = 1 - c
    return mp.matrix([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                      [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                      [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def close_group(gens):
    def keyf(M):
        return tuple(int(mp.nint(M[i, j] * 10 ** 20)) for i in range(3) for j in range(3))
    I = mp.eye(3)
    elems = {keyf(I): I}
    frontier = [I]
    while frontier:
        new = []
        for A in frontier:
            for g in gens:
                B = A * g
                k = keyf(B)
                if k not in elems:
                    elems[k] = B
                    new.append(B)
        frontier = new
        assert len(elems) <= 200
    return list(elems.values())


def invariant_dims_numeric(G, lmax):
    """N_l = (1/|G|) sum_g chi_l(g), chi_l(theta) = sin((2l+1)theta/2)/sin(theta/2)."""
    out = []
    angs = []
    for g in G:
        tr = g[0, 0] + g[1, 1] + g[2, 2]
        angs.append(mp.acos(max(-1, min(1, (tr - 1) / 2))))
    for l in range(lmax + 1):
        tot = mp.mpf(0)
        for th in angs:
            if abs(th) < mp.mpf(10) ** -30:
                tot += 2 * l + 1
            else:
                tot += mp.sin((2 * l + 1) * th / 2) / mp.sin(th / 2)
        out.append(tot / len(G))
    return out


def N_formula(l, order, cones):
    return F(2 * l + 1, order) + F(1, 2) * sum(2 * (l // m) + 1 - F(2 * l + 1, m) for m in cones)


def spherical_checks() -> None:
    mp.mp.dps = 40
    phi = (1 + mp.sqrt(5)) / 2
    groups = {
        "C5": (close_group([rot([0, 0, 1], 2 * mp.pi / 5)]), (5, 5)),
        "D6": (close_group([rot([0, 0, 1], 2 * mp.pi / 6), rot([1, 0, 0], mp.pi)]), (2, 2, 6)),
        "D2": (close_group([rot([0, 0, 1], mp.pi), rot([1, 0, 0], mp.pi)]), (2, 2, 2)),
        "T": (close_group([rot([0, 0, 1], mp.pi), rot([1, 1, 1], 2 * mp.pi / 3)]), (2, 3, 3)),
        "O": (close_group([rot([0, 0, 1], mp.pi / 2), rot([1, 1, 1], 2 * mp.pi / 3)]), (2, 3, 4)),
        "I": (close_group([rot([0, 1, phi], 2 * mp.pi / 5), rot([1, 1, 1], 2 * mp.pi / 3)]), (2, 3, 5)),
    }
    for name, (G, cones) in groups.items():
        chi = 2 - sum(1 - F(1, m) for m in cones)
        assert F(2, len(G)) == chi, (name, len(G), chi)     # |G| = 2/chi
        num = invariant_dims_numeric(G, 61)
        for l, v in enumerate(num):
            Nf = N_formula(l, len(G), cones)
            assert Nf.denominator == 1 and Nf >= 0
            assert abs(v - int(Nf)) < mp.mpf(10) ** -25, (name, l, v, Nf)
    print("S1 N_l formula = explicit character sums, l <= 61, groups", list(groups),
          "(|G| = 2/chi asserted)")

    # S2: cone coefficients at K = +1 from the exact spectrum
    for m in range(2, M_MAX + 1):
        ser = B_cone_series(m, L_MAX)
        for l in range(L_MAX + 1):
            assert ser[l] == b_ucar(l, m), (m, l, ser[l], b_ucar(l, m))
    print(f"S2 [t^l] B_m(t) == Ucar b_l(m) at K=+1, exactly, for l <= {L_MAX}, 2 <= m <= {M_MAX}")
    # the S^2 smooth series begins 1/t + 1/3 + t/15 + 4t^2/315 + t^3/315
    sph = sphere_series(6)
    assert sph[:5] == [1, F(1, 3), F(1, 15), F(4, 315), F(1, 315)], sph[:5]
    # numerical sanity of the Hurwitz-zeta expansion itself at small t
    mp.mp.dps = 30
    for m in (2, 3, 7):
        t = mp.mpf("0.004")
        direct = mp.mpf(0)
        l = 0
        while True:
            fr = F(1, 2) * (2 * (l // m) + 1 - F(2 * l + 1, m))
            term = mp.mpf(fr.numerator) / fr.denominator * mp.e ** (-l * (l + 1) * t)
            direct += term
            if l > 50 and l * (l + 1) * t > 120:
                break
            l += 1
        ser = B_cone_series(m, 6)
        approx = sum(mp.mpf(ser[k].numerator) / ser[k].denominator * t ** k for k in range(4))
        assert abs(direct - approx) < 1e-6, (m, direct, approx)
    print("S2 numeric sanity: B_m(0.004) summed directly agrees with the 4-term expansion to 1e-6")

    # full orbifold check: Z_G expansion equals smooth + cone terms (degree 0 and 1)
    def c_over_12(cones):
        chi = 2 - sum(1 - F(1, m) for m in cones)
        return chi / 6 + sum(cone0(m) for m in cones)
    table = {}
    for name, cones in [("S2", ()), ("(2,3,3)", (2, 3, 3)), ("(2,3,4)", (2, 3, 4)), ("(2,3,5)", (2, 3, 5))] + \
            [(f"({n},{n})", (n, n)) for n in range(2, 8)] + [(f"(2,2,{n})", (2, 2, n)) for n in range(2, 8)]:
        chi = 2 - sum(1 - F(1, m) for m in cones)
        sm = sphere_series(3)
        a0 = chi / 2 * sm[1] + sum(B_cone_series(m, 2)[0] for m in cones)
        assert a0 == c_over_12(cones), name
        table[name] = a0
    dggw = {"(2,3,3)": F(43, 72), "(2,3,4)": F(97, 144), "(2,3,5)": F(271, 360)}
    for k, v in dggw.items():
        assert table[k] == v, (k, table[k])
    print("S2 degree-0 terms from the spectrum match DGGW Table 1 for (2,3,3), (2,3,4), (2,3,5)")

    # S3: injectivity of the degree-0 term on the good orientable spherical class
    vals = {}
    def add(key, cones):
        v = c_over_12(cones)
        assert v not in vals, ("collision", key, vals.get(v))
        vals[v] = key
    add("S2", ())
    for c in [(2, 3, 3), (2, 3, 4), (2, 3, 5)]:
        add(str(c), c)
    NBIG = 20000
    for n in range(2, NBIG + 1):
        add(f"({n},{n})", (n, n))
        add(f"(2,2,{n})", (2, 2, n))
    # closed forms used in the proof: 12 a0 = 2n + 2/n ; 3 + n + 1/n
    for n in range(2, 50):
        assert 12 * c_over_12((n, n)) == 2 * n + F(2, n)
        assert 12 * c_over_12((2, 2, n)) == 3 + n + F(1, n)
    print(f"S3 degree-0 term injective on the good spherical class with n <= {NBIG} (and proved for all n)")
    # the t^{-1} coefficient (area at K = 1, i.e. 2 pi chi) is not injective
    for n in range(2, 200):
        chi_a = 2 - sum(1 - F(1, m) for m in (2, 2, n))
        chi_b = 2 - sum(1 - F(1, m) for m in (2 * n, 2 * n))
        assert chi_a == chi_b == F(1, n)
        assert c_over_12((2, 2, n)) != c_over_12((2 * n, 2 * n))
    print("S3 chi(2,2,n) = chi(2n,2n) = 1/n: the area coefficient alone never suffices")


def main() -> int:
    flat_checks()
    spherical_checks()
    print("ALL ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

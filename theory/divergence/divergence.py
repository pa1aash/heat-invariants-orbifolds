#!/usr/bin/env python3
"""Large-l asymptotics of the cone coefficients b_l(m) and the divergence-rate
statement for closed constant-curvature orbifolds (the paper in this directory).

Inputs (transcribed in Ucar's thesis):
  Ucar (4.25): c_l(pi/k) = (-1)^l / (4k (l+1)! (2l+1))
               * sum_j binom(2l+2,2j) (k^{2j}-1) B_{2j} B_{2l+2-2j}(1/2)
  Ucar (4.33)/(4.34): b_l(k)/K^l = sum_i 2/(4^i i!) c_{l-i}(pi/k).
  Dryden-Strohmaier, arXiv:math/0504571, eq. (1): elliptic term of a cone
  point of order m, h(r) = exp(-t(1/4 + r^2)),
      E_m(t) = sum_{l=1}^{m-1} 1/(2 m sin th_l) int_R e^{-2 th_l r}/(1+e^{-2 pi r}) h(r) dr,
      th_l = pi l / m.

Checks (exact unless marked numeric):
  D1  sum_j binom(N,2j)(k^{2j}-1) B_{2j} B_{N-2j}(1/2) = N! [t^N] G_k(t),
      G_k(t) = (t/2)/sinh(t/2) * ((kt/2) coth(kt/2) - (t/2) coth(t/2)).
  D2  c_l(pi/k) > 0 and b_l(k)/K^l > 0 for all l <= L_POS, 2 <= k <= 40.
  D3  numeric (60 digits): b_l(m)/K^l = A_l(m) (1 + pi^2/(2 m^2 (2l-1)) + r_l),
      A_l(m) = (2l)! / (l! m sin(pi/m)) * (m/(2 pi))^{2l+1}, with l^2 |r_l|
      bounded, l <= L_ASY.
  D4  numeric (50 digits): Taylor coefficients of the Dryden-Strohmaier
      elliptic term, computed from the exact moment series
      int_0^inf r^n e^{-a r}/(1+e^{-2 pi r}) dr = n! sum_j (-1)^j (a + 2 pi j)^{-n-1},
      equal Ucar's b_n(m) at K = -1, n <= 30.
  D5  exact: the t^1 coefficients of O(2,8,8) and O(3,3,12) at K = -1 are
      -1601/480 and -867/160.
  D6  numeric from exact coefficients: for several hyperbolic orbifolds the
      estimator M_l^2 = 2 pi^2 |a_l| / ((2l-1) |a_{l-1}|) converges to the
      largest cone order squared, and peeling recovers the whole multiset
      from the coefficients a_l, l in [L_PEEL-1, L_PEEL] only.
"""
from __future__ import annotations

import sys
from fractions import Fraction as F
from functools import lru_cache
from math import comb, factorial

import mpmath as mp
import sympy as sp

L_POS = 60
L_ASY = 120
L_PEEL = 220


# ------------------------------------------------------------------ Bernoulli
_B: dict[int, F] = {0: F(1)}


def bern(n: int) -> F:
    if n not in _B:
        _B[n] = -sum(comb(n + 1, j) * bern(j) for j in range(n)) / (n + 1)
    return _B[n]


def bern_half(n: int) -> F:
    return (F(1, 2 ** (n - 1)) - 1) * bern(n) if n >= 1 else F(1)


@lru_cache(maxsize=None)
def c_ucar(l: int, k: int) -> F:
    s = sum(comb(2 * l + 2, 2 * j) * (F(k) ** (2 * j) - 1) * bern(2 * j) * bern_half(2 * l + 2 - 2 * j)
            for j in range(l + 2))
    return F((-1) ** l, 4 * factorial(l + 1) * (2 * l + 1)) * s / k


@lru_cache(maxsize=None)
def b_ucar(l: int, m: int) -> F:
    return sum(F(2, 4 ** i * factorial(i)) * c_ucar(l - i, m) for i in range(l + 1))


def mpf(x: F):
    return mp.mpf(x.numerator) / x.denominator


# ----------------------------------------------------------------- D1, D2
def check_generating_function() -> None:
    t = sp.symbols("t")
    for k in (2, 3, 5, 12):
        G = (t / 2) / sp.sinh(t / 2) * ((k * t / 2) * sp.coth(k * t / 2) - (t / 2) * sp.coth(t / 2))
        ser = sp.series(G, t, 0, 23).removeO()
        for N in range(0, 23):
            lhs = sum(comb(N, 2 * j) * (F(k) ** (2 * j) - 1) * bern(2 * j) * bern_half(N - 2 * j)
                      for j in range(N // 2 + 1)) if N % 2 == 0 else F(0)
            rhs = sp.Rational(factorial(N)) * ser.coeff(t, N)
            assert sp.Rational(lhs.numerator, lhs.denominator) == rhs, (k, N)
    # x cot x = 1 - sum_n 2 x^2/(n^2 pi^2 - x^2): every coefficient beyond x^0 is negative
    x = sp.symbols("x")
    s = sp.series(x * sp.cot(x), x, 0, 30).removeO()
    for n in range(2, 30, 2):
        assert s.coeff(x, n) < 0
    # (y/2)/sin(y/2): every even coefficient positive
    s2 = sp.series((x / 2) / sp.sin(x / 2), x, 0, 30).removeO()
    for n in range(0, 30, 2):
        assert s2.coeff(x, n) > 0
    print("D1 generating-function identity for N <= 22, k in {2,3,5,12}; sign pattern of x cot x"
          " and (x/2)/sin(x/2) through x^28")


def check_positivity() -> None:
    for k in range(2, 41):
        for l in range(L_POS + 1):
            assert c_ucar(l, k) > 0, (l, k)
            assert b_ucar(l, k) > 0, (l, k)
    for l in range(L_POS + 1):
        assert c_ucar(l, 1) == 0 and b_ucar(l, 1) == 0
    print(f"D2 c_l(pi/k) > 0 and b_l(k)/K^l > 0 for l <= {L_POS}, 2 <= k <= 40; zero at k = 1")


# --------------------------------------------------------------------- D3
def A_lead(l: int, m: int):
    return mp.factorial(2 * l) / (mp.factorial(l) * m * mp.sin(mp.pi / m)) * (m / (2 * mp.pi)) ** (2 * l + 1)


def check_asymptotics() -> None:
    mp.mp.dps = 60
    print("D3  l^2 * r_l  (r_l = b_l/A_l - 1 - pi^2/(2 m^2 (2l-1)))")
    for m in (2, 3, 4, 5, 8, 12, 30):
        worst = mp.mpf(0)
        samples = []
        for l in range(1, L_ASY + 1):
            rho = mpf(b_ucar(l, m)) / A_lead(l, m)
            r = rho - 1 - mp.pi ** 2 / (2 * m * m * (2 * l - 1))
            if l >= 10:
                worst = max(worst, abs(r) * l * l)
            if l in (5, 10, 20, 40, 80, L_ASY):
                samples.append((l, mp.nstr(rho, 12), mp.nstr(r * l * l, 6)))
        # second-order term of the i-sum is (pi^2/(2m^2))^2/2 / ((2l-1)(2l-3)) ~ pi^4/(32 m^4 l^2)
        assert worst < 1, (m, worst)
        print(f"   m={m:>2}: sup_{{l>=10}} l^2|r_l| = {mp.nstr(worst, 6)};  (l, b_l/A_l, l^2 r_l):", samples)
    # rho_l -> 1 at rate 1/l exactly as predicted: l*(rho-1) -> pi^2/(4 m^2)
    for m in (2, 3, 7):
        l = L_ASY
        val = l * (mpf(b_ucar(l, m)) / A_lead(l, m) - 1)
        assert abs(val - mp.pi ** 2 / (4 * m * m)) < 0.01, (m, val)
    # the leading coefficient of p_l, times m^{2l+1}, misses exactly (pi/m)/sin(pi/m)
    for m in (2, 3, 6):
        l = L_ASY
        lead = abs(bern(2 * l + 2)) / (2 * factorial(l + 1) * (2 * l + 1))
        ratio = mpf(b_ucar(l, m)) / (mpf(lead) * m ** (2 * l + 1))
        target = (mp.pi / m) / mp.sin(mp.pi / m) * (1 + mp.pi ** 2 / (2 * m * m * (2 * l - 1)))
        assert abs(ratio / target - 1) < 1e-4, (m, ratio, target)
    print("D3 l(b_l/A_l - 1) -> pi^2/(4m^2); b_l / (lead_l m^{2l+1}) -> (pi/m)/sin(pi/m)")


# --------------------------------------------------------------------- D4
def moment(n: int, a):
    """int_0^inf r^n e^{-a r} / (1 + e^{-2 pi r}) dr."""
    return mp.factorial(n) * mp.nsum(lambda j: (-1) ** int(j) / (a + 2 * mp.pi * j) ** (n + 1), [0, mp.inf])


def ds_coeff(n: int, m: int):
    """t^n Taylor coefficient of the Dryden-Strohmaier elliptic term."""
    tot = mp.mpf(0)
    for l in range(1, m):
        th = mp.pi * l / m
        # int_R r^{2b} e^{-2 th r}/(1+e^{-2 pi r}) dr: r > 0 part, and r < 0 part
        # after r -> -s: e^{2 th s}/(1+e^{2 pi s}) = e^{-(2pi-2th)s}/(1+e^{-2 pi s})
        acc = mp.mpf(0)
        for b in range(n + 1):
            a_ = n - b
            mom = moment(2 * b, 2 * th) + moment(2 * b, 2 * mp.pi - 2 * th)
            acc += (mp.mpf(-1) / 4) ** a_ / mp.factorial(a_) * (-1) ** b / mp.factorial(b) * mom
        tot += acc / (2 * m * mp.sin(th))
    return tot


def check_dryden_strohmaier() -> None:
    mp.mp.dps = 50
    worst = mp.mpf(0)
    for m in (2, 3, 5, 8, 12):
        for n in range(0, 31):
            exact = mpf(b_ucar(n, m)) * (-1) ** n
            ds = ds_coeff(n, m)
            rel = abs(ds / exact - 1)
            worst = max(worst, rel)
            assert rel < mp.mpf(10) ** -35, (m, n, ds, exact)
    print(f"D4 Dryden-Strohmaier Taylor coefficients == Ucar b_n(m) at K=-1, n <= 30,"
          f" m in {{2,3,5,8,12}}: max rel. diff {mp.nstr(worst, 3)}")


# ------------------------------------------------------------- D5, D6
@lru_cache(maxsize=None)
def sphere_s(k: int) -> F:
    """t Z_{S^2}(t) = sum_k s_k t^k (K = +1 per area 4 pi); s_0 = 1."""
    def hz(N, a):  # zeta(-N, a)
        return -sum(comb(N + 1, i) * bern(i) * a ** (N + 1 - i) for i in range(N + 2)) / (N + 1)
    inner = [F(1)] + [2 * F((-1) ** j, factorial(j)) * hz(1 + 2 * j, F(1, 2)) for j in range(k)]
    return sum(F(1, 4) ** (k - i) / factorial(k - i) * inner[i] for i in range(k + 1))


def coeff_hyperbolic(l: int, cones: tuple[int, ...], genus: int = 0) -> F:
    """t^l coefficient (l >= 0) of the heat trace of a closed orbifold, K = -1."""
    chi = F(2 - 2 * genus) - sum(1 - F(1, m) for m in cones)
    assert chi < 0
    smooth = (-chi) / 2 * sphere_s(l + 1) * (-1) ** (l + 1)     # Area/(4 pi) = |chi|/2
    return smooth + (-1) ** l * sum(b_ucar(l, m) for m in cones)


def check_orbifold_level() -> None:
    assert all(sphere_s(k) > 0 for k in range(0, 80))
    # D5 against the supplement 4(d)
    assert coeff_hyperbolic(1, (2, 8, 8)) == F(-1601, 480)
    assert coeff_hyperbolic(1, (3, 3, 12)) == F(-867, 160)
    assert coeff_hyperbolic(0, (2, 8, 8)) == F(67, 48)          # REPORT a0
    # a0 = (S1 + R - 2)/12 (paper eq. s1inv) for triads
    for c in [(2, 3, 7), (2, 8, 8), (3, 3, 12), (4, 5, 6)]:
        assert coeff_hyperbolic(0, c) == F(sum(c), 12) + sum(F(1, m) for m in c) / 12 - F(1, 6)
    print("D5 a_1(2,8,8) = -1601/480, a_1(3,3,12) = -867/160, a_0(2,8,8) = 67/48 (REPORT), a_0 = (S1+R-2)/12")

    mp.mp.dps = 80
    print("D6 largest-order estimator M_l = pi sqrt(2|a_l| / ((2l-1)|a_{l-1}|)):")
    for cones, g in [((2, 8, 8), 0), ((3, 3, 12), 0), ((7, 8, 9), 0), ((11, 12, 12, 12), 0), ((2, 3), 1), ((), 2)]:
        row = []
        for l in (10, 40, 120):
            a, b = coeff_hyperbolic(l, cones, g), coeff_hyperbolic(l - 1, cones, g)
            M = mp.pi * mp.sqrt(2 * abs(mpf(a)) / ((2 * l - 1) * abs(mpf(b))))
            row.append(mp.nstr(M, 8))
        target = max(cones) if cones else 1
        assert abs(mp.mpf(row[-1]) - target) < (0.05 if cones else 0.1), (cones, row)
        print(f"   cones {cones}, genus {g}: M_10, M_40, M_120 = {row}  (max order {target if cones else 'none: smooth rate 1'})")

    # peeling: recover the multiset from two consecutive coefficients at l = L_PEEL
    for cones, g in [((3, 5, 5, 9), 0), ((2, 2, 3, 3, 3, 7), 0), ((4, 4), 1), ((6, 6, 7), 0)]:
        l = L_PEEL
        a = [coeff_hyperbolic(l - 1, cones, g), coeff_hyperbolic(l, cones, g)]
        found = []
        for _ in range(20):
            M = int(mp.nint(mp.pi * mp.sqrt(2 * abs(mpf(a[1])) / ((2 * l - 1) * abs(mpf(a[0]))))))
            if M < 2:
                break
            mu = int(mp.nint(mpf(a[1]) * (-1) ** l / mpf(b_ucar(l, M))))
            assert mu >= 1
            found += [M] * mu
            a = [a[0] - (-1) ** (l - 1) * mu * b_ucar(l - 1, M), a[1] - (-1) ** l * mu * b_ucar(l, M)]
        assert sorted(found) == sorted(cones), (cones, found)
        # what remains is the smooth part, which fixes chi hence the area
        chi = F(2 - 2 * g) - sum(1 - F(1, m) for m in cones)
        assert a[1] == (-chi) / 2 * sphere_s(l + 1) * (-1) ** (l + 1)
        print(f"   peeling at l = {l}: recovered {sorted(found)} (genus {g}); remainder = smooth term exactly")


def main() -> int:
    check_generating_function()
    check_positivity()
    check_asymptotics()
    check_dryden_strohmaier()
    check_orbifold_level()
    print("ALL ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

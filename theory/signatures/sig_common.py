"""Shared exact-arithmetic helpers for theory/signatures/.

Conventions.

* A signature is (g, m) with g >= 0 the genus and m a sorted tuple of cone orders >= 2.
  Order-1 entries are "padding": they are allowed in intermediate multisets and are
  stripped by `strip`.
* s(g, m) = 2g - 2 + sum(1 - 1/m_i) = -chi = Area / (2 pi). The orbifold is hyperbolic
  iff s > 0. The first heat coefficient is c_1 = Area / (4 pi) = s / 2.
* Gauss curvature K = -1. A cone point of order k contributes b_l(k) at order t^l,
  b_l(k) = K^l * sum_{i=0}^{l} 2/(4^i i!) c_{l-i}(pi/k)        (Ucar (4.33)/(4.34)),
  c_l(pi/k) = 1/(4k) (-1)^l/((l+1)!) 1/(2l+1)
              * sum_{j=0}^{l+1} binom(2l+2,2j) (k^{2j}-1) B_{2j} B_{2l+2-2j}(1/2)   (Ucar (4.25)),
  transcribed in Ucar's thesis.
* The heat coefficient c_{l+2} (order t^l, l >= 0) is (smooth part) + C_l with
  C_l = sum_i b_l(m_i). At constant curvature the smooth part is a fixed multiple of the
  area (see the paper), so two orbifolds of equal area have equal c_{l+2} iff
  they have equal C_l. Hence "the first k heat coefficients agree" is tested by
  heat_key(g, m, k) = (s, C_0, ..., C_{k-2}).
* psi_j(x) = x^(2j-1) - 1/x. Psi_j(m) = sum_i psi_j(m_i) = P_{2j-1} - R.
  int_key(g, m, k) = (s, P_1 - nu, P_3 - nu, ..., P_{2k-3} - nu), nu = n + 2g.
  heat_structure.py proves (symbolically, l <= 15, and the paper for all l) that
  heat_key and int_key define the same equivalence; int_key is integer apart from s.

Everything is exact (Fraction / sympy); nothing here uses floating point.
"""
from fractions import Fraction
from math import comb, factorial

_B = {0: Fraction(1)}


def bern(n):
    """Bernoulli number B_n (B_1 = -1/2), own recursion."""
    if n in _B:
        return _B[n]
    s = sum(comb(n + 1, j) * bern(j) for j in range(n))
    _B[n] = -s / (n + 1)
    return _B[n]


def bern_half(n):
    """B_n(1/2) = (2^{1-n} - 1) B_n (Ucar Lemma 4.14)."""
    return (Fraction(1, 2 ** (n - 1)) - 1) * bern(n) if n >= 1 else Fraction(1)


def c_ucar(l, k):
    k = Fraction(k)
    s = sum(comb(2 * l + 2, 2 * j) * (k ** (2 * j) - 1) * bern(2 * j)
            * bern_half(2 * l + 2 - 2 * j) for j in range(l + 2))
    return Fraction((-1) ** l, 4 * factorial(l + 1) * (2 * l + 1)) * s / k


_bcache = {}


def b_cone(l, k):
    """b_l(k) at K = -1: the t^l heat contribution of one cone point of order k."""
    key = (l, k)
    if key not in _bcache:
        _bcache[key] = (-1) ** l * sum(Fraction(2, 4 ** i * factorial(i)) * c_ucar(l - i, k)
                                       for i in range(l + 1))
    return _bcache[key]


def s_of(g, m):
    return 2 * g - 2 + sum(1 - Fraction(1, x) for x in m)


def strip(m):
    return tuple(sorted(x for x in m if x != 1))


def heat_key(g, m, k):
    """(s, C_0, ..., C_{k-2}): determines the first k heat coefficients given equal area."""
    assert k >= 1
    return (s_of(g, m),) + tuple(sum(b_cone(l, x) for x in m) for l in range(k - 1))


def int_key(g, m, k):
    nu = len(m) + 2 * g
    return (s_of(g, m),) + tuple(sum(x ** (2 * j - 1) for x in m) - nu for j in range(1, k))


def shared(sig1, sig2, kmax):
    """Number of leading heat coefficients (c_1, c_2, ...) on which sig1, sig2 agree, capped
    at kmax. Computed from the actual cone coefficients b_l (heat_key), not from int_key."""
    (g1, m1), (g2, m2) = sig1, sig2
    if s_of(g1, m1) != s_of(g2, m2):
        return 0
    k = 1
    while k < kmax:
        l = k - 1
        if sum(b_cone(l, x) for x in m1) != sum(b_cone(l, x) for x in m2):
            break
        k += 1
    return k


def shared_int(sig1, sig2, kmax):
    """Same as `shared`, computed through int_key (fast, integer arithmetic)."""
    (g1, m1), (g2, m2) = sig1, sig2
    if s_of(g1, m1) != s_of(g2, m2):
        return 0
    nu1, nu2 = len(m1) + 2 * g1, len(m2) + 2 * g2
    k = 1
    while k < kmax:
        j = 2 * k - 1
        if sum(x ** j for x in m1) - nu1 != sum(x ** j for x in m2) - nu2:
            break
        k += 1
    return k


def odd_balanced(U, V, L):
    """True iff R(U) = R(V) and P_j(U) = P_j(V) for all odd j <= 2L-3."""
    if sum(Fraction(1, x) for x in U) != sum(Fraction(1, x) for x in V):
        return False
    return all(sum(x ** j for x in U) == sum(x ** j for x in V) for j in range(1, 2 * L - 2, 2))


def realise(U, V, gmin=0):
    """Given positive-integer multisets U, V with |V| - |U| even, R(U) = R(V), return the
    pair of signatures (g, strip U), (g', strip V) with g - g' = (|V| - |U|)/2 and the
    smallest g' >= gmin making both genera >= 0 and both hyperbolic (s > 0)."""
    d = len(V) - len(U)
    assert d % 2 == 0
    gp = max(gmin, -d // 2)
    while True:
        g = gp + d // 2
        a, b = (g, strip(U)), (gp, strip(V))
        if s_of(*a) > 0 and s_of(*b) > 0:
            assert s_of(*a) == s_of(*b)
            return a, b
        gp += 1


def thue_morse(i):
    return bin(i).count("1") & 1

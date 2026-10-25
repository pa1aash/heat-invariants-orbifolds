"""Shared exact-arithmetic helpers for the audibility verification scripts.

Conventions (fixed throughout theory/audibility/):

* m_1, ..., m_n are the cone orders.
* e_k = e_k(m) is the k-th elementary symmetric function, e_0 = 1.
* f(z) = prod_i (1 + m_i z) = sum_k e_k z^k        (the "heat side" polynomial)
* p(z) = prod_i (z + m_i)   = sum_k e_k z^(n-k)    (characteristic polynomial of
  the stable linear system x' = -diag(m) x; p is the reversal of f)
* P_k = sum_i m_i^k, R = sum_i 1/m_i = e_{n-1}/e_n.
* Hurwitz matrix and determinants follow Holtz-Tyaglov, SIAM Rev. 54 (2012),
  eq. (1.37): for p(z) = a_0 z^n + a_1 z^(n-1) + ... + a_n, the (i, j) entry
  (1-indexed) is a_(2j - i), with a_k = 0 for k < 0 or k > n, and Delta_j is the
  leading principal j x j minor.

Everything is sympy exact arithmetic; nothing here uses floating point.
"""
import itertools

import sympy as sp


def elementary(ms):
    """[e_0, e_1, ..., e_n] of the list ms, as expanded sympy expressions."""
    n = len(ms)
    return [sp.Integer(1)] + [
        sp.expand(sum(sp.Mul(*c) for c in itertools.combinations(ms, k)))
        for k in range(1, n + 1)
    ]


def power_sums_from_e(e, kmax):
    """Newton's identities: P_k as polynomials in e = [e_0=1, e_1, ..., e_n].

    P_k = sum_{i=1}^{k-1} (-1)^(i-1) e_i P_{k-i} + (-1)^(k-1) k e_k,
    with e_i = 0 for i > n. Returns {k: P_k} for 1 <= k <= kmax.
    """
    n = len(e) - 1
    E = lambda i: e[i] if i <= n else 0
    P = {}
    for k in range(1, kmax + 1):
        P[k] = sp.expand(
            sum((-1) ** (i - 1) * E(i) * P[k - i] for i in range(1, k))
            + (-1) ** (k - 1) * k * E(k)
        )
    return P


def hurwitz_matrix(a, j):
    """Leading j x j block of the Hurwitz matrix of a_0 z^n + ... + a_n.

    a = [a_0, ..., a_n]; entry (r, c), 1-indexed, is a_(2c - r) (Holtz-Tyaglov
    (1.37); Barkovsky arXiv:0802.1805, eq. (34)).
    """
    n = len(a) - 1
    A = lambda k: a[k] if 0 <= k <= n else 0
    return sp.Matrix(j, j, lambda r, c: A(2 * (c + 1) - (r + 1)))


def hurwitz_det(a, j):
    return sp.expand(hurwitz_matrix(a, j).det(method="berkowitz"))


def generic_hurwitz_det(n, j):
    """Delta_j of a_0 z^n + ... + a_n with symbolic a_k, as a sympy Poly in a_0..a_n."""
    a = sp.symbols(f"a0:{n + 1}")
    return sp.Poly(hurwitz_matrix(list(a), j).det(method="berkowitz"), *a)


def evaluate_poly(P, values, zero):
    """Evaluate the sympy Poly P at ring elements `values` (sparse, exact).

    Used to substitute a_k -> (polynomial in the roots) without sympy's slow
    expression expansion. `zero` is the zero of the target ring.
    """
    out = zero
    for mon, c in P.terms():
        term = zero + int(c)
        for v, pw in zip(values, mon):
            if pw:
                term = term * v**pw
        out += term
    return out


def ring_elementary(gens, one):
    """[e_0, ..., e_n] of the ring generators `gens`, computed in the ring."""
    n = len(gens)
    e = [one] + [one * 0] * n
    for x in gens:
        for k in range(n, 0, -1):
            e[k] = e[k] + e[k - 1] * x
    return e


def odd_tanh_coefficients(Podd, order):
    """Taylor coefficients T_1, T_3, ... of tanh(sum_{k odd} P_k z^k / k).

    Podd maps odd k -> value (symbol or number). Returns {k: T_k} for odd
    k < order. Only P_k with k < order can contribute.
    """
    z = sp.Symbol("z")
    S = sum(Podd[k] * z**k / sp.Integer(k) for k in Podd if k < order)
    T = sp.expand(sp.series(sp.tanh(S), z, 0, order).removeO())
    poly = sp.Poly(T, z)
    return {k: poly.coeff_monomial(z**k) for k in range(1, order, 2)}


def pade_system(n, R, Podd):
    """The linear system for (e_1, ..., e_n) given R and P_1, P_3, ..., P_{2n-3}.

    With f = E(z^2) + z O(z^2) (even/odd parts) the odd power sums fix
    T(z) = tanh(sum_{k odd} P_k z^k / k) = z O / E  modulo z^(2n-1).
    Rows j = 0..n-2 are the z^(2j+1) coefficients of  z O - T E = 0:
        e_{2j+1} - sum_{i=0}^{j} T_{2i+1} e_{2j-2i} = 0      (e_0 = 1)
    and the last row is  e_{n-1} - R e_n = 0.
    Returns (M, b, e_symbols) with M e = b.
    """
    T = odd_tanh_coefficients(Podd, 2 * n - 1)
    e = sp.symbols(f"e1:{n + 1}")
    E = lambda k: sp.Integer(1) if k == 0 else (e[k - 1] if k <= n else 0)
    rows = [
        E(2 * j + 1) - sum(T[2 * i + 1] * E(2 * j - 2 * i) for i in range(j + 1))
        for j in range(n - 1)
    ]
    rows.append(E(n - 1) - R * E(n))
    M, b = sp.linear_eq_to_matrix(rows, e)
    return M, b, e


def invariants(ms):
    """Exact (R, S1, P3, ..., P_{2n-3}) of a list of positive rationals."""
    n = len(ms)
    ms = [sp.Rational(x) for x in ms]
    return (sum(1 / x for x in ms),) + tuple(
        sum(x**k for x in ms) for k in range(1, 2 * n - 2, 2)
    )

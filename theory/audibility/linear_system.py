"""T3: the linear system behind audibility, and its determinant, n = 3..8.

Setting (the paper, section 2). With f(z) = prod (1 + m_i z) = E(z^2) + z O(z^2),
    log f(z) - log f(-z) = 2 sum_{k odd} P_k z^k / k,
so  T(z) := tanh( sum_{k odd} P_k z^k / k ) = z O(z^2) / E(z^2).
The odd power sums P_1, ..., P_{2n-3} fix T modulo z^(2n-1); the n - 1 odd
coefficients z^1, z^3, ..., z^(2n-3) of  z O - T E = 0  are LINEAR in
e_1, ..., e_n, and R = e_{n-1}/e_n adds the linear row e_{n-1} - R e_n = 0.
This is a square n x n system M e = b whose entries depend only on the heat
invariants (R, S1 = P_1, P_3, ..., P_{2n-3}).

Checked here, exactly, for n = 3..8:
  (a) the true e solves the system identically (P_k replaced by their Newton
      expressions in free variables e_1..e_n, R by e_{n-1}/e_n);
  (b) det M = sigma_n * Delta_{n-1}(p) / e_n  as an identity in Q(e_1..e_n),
      p(z) = z^n + e_1 z^(n-1) + ... + e_n, sigma_n = +/-1; by Orlando
      Delta_{n-1}(p) = prod_{i<j} (m_i + m_j), so det M != 0 for positive m;
  (c) leading principal minors (rows in the order 0..n-3, R-row, n-2) are
      +/- Hurwitz determinants: D_k = +/-Delta_{k-1} (k <= n-2),
      D_{n-1} = +/-Delta_{n-3}, D_n = +/-Delta_{n-1}/e_n; so Gaussian elimination
      in that order has pivots that are ratios of Hurwitz determinants (the
      Routh scheme), none of which vanishes for a stable p;
  (d) end to end: for fixed random positive-rational multisets the system,
      built from the invariants alone, returns exactly the true e.
Exits nonzero on any failure.
"""
import random
import sys
import time
from fractions import Fraction

import sympy as sp
from sympy import QQ
from sympy.polys.rings import ring

from audibility_common import (
    elementary,
    generic_hurwitz_det,
    invariants,
    odd_tanh_coefficients,
    pade_system,
    power_sums_from_e,
)

NS = range(3, 9)


def symbolic_system(n):
    R = sp.Symbol("R")
    Podd = {k: sp.Symbol(f"P{k}") for k in range(1, 2 * n - 2, 2)}
    M, b, e = pade_system(n, R, Podd)
    return M, b, R, Podd


def e_ring(n):
    Rg, *es = ring(",".join(f"e{i}" for i in range(1, n + 1)), QQ)
    e = [Rg(1)] + es
    return Rg, e


def to_ring(expr, R, Podd, Rg, e, Pval, n):
    """Evaluate a polynomial in (R, P's) at P = Newton(e), R = e_{n-1}/e_n.

    Returns (value * e_n^d, d) with d = deg_R(expr), so the result is a ring
    element (no division).
    """
    gens = [R] + list(Podd.values())
    poly = sp.Poly(sp.expand(expr), *gens)
    d = poly.degree(R)
    out = Rg(0)
    for mon, c in poly.terms():
        r = mon[0]
        term = Rg(QQ(int(sp.numer(c)), int(sp.denom(c))))
        term *= e[n - 1] ** r * e[n] ** (d - r)
        for (k, _), pw in zip(Podd.items(), mon[1:]):
            if pw:
                term *= Pval[k] ** pw
        out += term
    return out, d


def ring_newton(e, n, kmax):
    E = lambda i: e[i] if i <= n else e[0] * 0
    P = {}
    for k in range(1, kmax + 1):
        acc = e[0] * 0
        for i in range(1, k):
            acc += (-1) ** (i - 1) * E(i) * P[k - i]
        P[k] = acc + (-1) ** (k - 1) * k * E(k)
    return P


def hurwitz_in_ring(n, j, e):
    if j == 0:
        return e[0]
    from audibility_common import evaluate_poly

    return evaluate_poly(generic_hurwitz_det(n, j), e, e[0] * 0)


def check_n(n):
    M, b, R, Podd = symbolic_system(n)
    Rg, e = e_ring(n)
    Pnewton = ring_newton(e, n, 2 * n - 3)
    Pval = {k: Pnewton[k] for k in Podd}

    # (a) the true e solves M e = b identically
    esyms = sp.symbols(f"e1:{n + 1}")
    for i in range(n):
        row = sum(M[i, j] * esyms[j] for j in range(n)) - b[i]
        # row is polynomial in (R, P, e); evaluate P, R in the ring, e_j -> e[j]
        gens = [R] + list(Podd.values()) + list(esyms)
        poly = sp.Poly(sp.expand(row), *gens)
        d = poly.degree(R)
        val = Rg(0)
        for mon, c in poly.terms():
            term = Rg(QQ(int(sp.numer(c)), int(sp.denom(c))))
            term *= e[n - 1] ** mon[0] * e[n] ** (d - mon[0])
            for (k, _), pw in zip(Podd.items(), mon[1 : 1 + len(Podd)]):
                if pw:
                    term *= Pval[k] ** pw
            for j, pw in enumerate(mon[1 + len(Podd) :]):
                if pw:
                    term *= e[j + 1] ** pw
            val += term
        assert val == 0, f"n={n}: true e violates row {i}"

    # reorder rows: 0..n-3, R-row, n-2
    order = list(range(n - 2)) + [n - 1, n - 2]
    Mr = M.extract(order, list(range(n)))

    # (b) determinant and (c) leading principal minors
    H = {j: hurwitz_in_ring(n, j, e) for j in range(0, n)}
    minors = []
    for k in range(1, n + 1):
        Dk = Mr[:k, :k].det(method="berkowitz")
        val, d = to_ring(Dk, R, Podd, Rg, e, Pval, n)
        # val = D_k * e_n^d
        if k <= n - 2:
            target = H[k - 1] * e[n] ** d
        elif k == n - 1:
            target = H[n - 3] * e[n] ** d
        else:
            target = H[n - 1] * e[n] ** (d - 1)
        if val == target:
            s = 1
        elif val == -target:
            s = -1
        else:
            raise AssertionError(f"n={n}: leading minor D_{k} is not +/- the predicted Hurwitz determinant")
        minors.append(s)
        if k == n:
            nterms = len(sp.Add.make_args(sp.expand(Dk)))
    sigma = minors[-1] * (1 if order == list(range(n)) else -1)  # row swap flips sign
    return sigma, minors, nterms


def end_to_end(n, rng, trials=3):
    for _ in range(trials):
        ms = [Fraction(rng.randint(2, 40), rng.randint(1, 5)) for _ in range(n)]
        inv = invariants(ms)
        R = inv[0]
        Podd = {2 * i + 1: inv[1 + i] for i in range(n - 1)}
        T = odd_tanh_coefficients(Podd, 2 * n - 1)
        M, b, e = pade_system(n, R, Podd)
        sol = M.LUsolve(b)
        truth = elementary([sp.Rational(x) for x in ms])[1:]
        assert list(sol) == truth, f"n={n}: linear system did not return the true e for {ms}"
        assert M.det() != 0


def main():
    rng = random.Random(20261001)
    for n in NS:
        t = time.time()
        sigma, minors, nterms = check_n(n)
        end_to_end(n, rng)
        print(
            f"n={n}: true e solves M e = b; det M = {'+' if sigma > 0 else '-'}"
            f"Delta_{n-1}(p)/e_n  (det has {nterms} terms in the invariants); "
            f"leading-minor signs {minors}; end-to-end recovery OK  [{time.time() - t:.1f}s]",
            flush=True,
        )
    print("linear_system: all assertions passed for n = 3..8")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

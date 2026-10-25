"""T1/T2: exact verification of the planning hypotheses H1-H4 and the n = 6..8
elimination, independent of any earlier code.

Notation: e_k elementary symmetric in the cone orders m_i, p(z) = prod (z + m_i),
Delta_k = k-th Hurwitz determinant of p (Holtz-Tyaglov (1.37)), invariants
R = e_{n-1}/e_n, S1 = e_1, P_k = sum m_i^k (Newton's identities in the e_k).

H1 (n=3)  3 - 3 S1 R = -3 Delta_2 / e_3 and Delta_2 = (p+q)(q+r)(r+p).
H2 (n=4)  e1 = S1, e3 = R e4, e2 from P3: the P5 equation is linear in e4 with
          coefficient a nonzero multiple of Delta_3; Delta_3 has 38 terms, all
          positive.
H3 (n=5)  e1 = S1, e4 = R e5, e2 from P3; Res_{e5}(F5, F7) is linear in e3 with
          leading coefficient const * Delta_4 / e5; the e5-step has coefficient
          -5 (R S1 - 1). Every denominator and leading coefficient is checked to be
          nonzero, and the unique solution is checked to be the true e, so the
          resultant introduces no spurious solution.
H4        Delta_{n-1}(p) = prod_{i<j} (m_i + m_j)  (Orlando; sign +), n = 3..8.
n=6..8    the naive Newton substitution is NOT linear (degrees are recorded); the
          tanh/Pade combination is linear at every step and its determinant is
          +/- Delta_{n-1}/e_n (imported from linear_system.check_n).
Also:     R S1 - n^2 = sum_{i<j} (m_i - m_j)^2 / (m_i m_j) >= 0, n = 3..8;
          Lemma 1 of the paper (parity of Newton's identities), odd j <= 11;
          Jacobian of (R, P1, ..., P_{2n-3}) = c_n V(m) prod(m_i+m_j)/prod m_i^2,
          n = 2..7.

Exits nonzero on any failure.
"""
import itertools
import sys
import time

import sympy as sp
import sympy.combinatorics
from sympy import ZZ
from sympy.polys.rings import ring

from audibility_common import (
    elementary,
    evaluate_poly,
    generic_hurwitz_det,
    hurwitz_det,
    power_sums_from_e,
    ring_elementary,
)
from linear_system import check_n

S1, R, P3, P5, P7 = sp.symbols("S1 R P3 P5 P7")


def true_values(n):
    ms = sp.symbols(f"m1:{n + 1}", positive=True)
    e = elementary(list(ms))
    vals = {S1: e[1], R: sp.cancel(e[n - 1] / e[n])}
    for k, s in ((3, P3), (5, P5), (7, P7)):
        vals[s] = sum(x**k for x in ms)
    return ms, e, vals


def is_zero(expr):
    return sp.cancel(sp.together(expr)) == 0


def h1():
    ms, e, vals = true_values(3)
    es = sp.symbols("e1:4")
    P = power_sums_from_e([1, *es], 3)
    # e2 = R e3; solve the P3 equation for e3
    eq = sp.expand(P[3].subs({es[0]: S1, es[1]: R * es[2]}) - P3)
    assert sp.degree(eq, es[2]) == 1
    coeff = sp.expand(eq.coeff(es[2], 1))
    assert sp.expand(coeff - (3 - 3 * S1 * R)) == 0, f"H1: e3-coefficient is {coeff}"
    D2 = hurwitz_det(e, 2)
    assert is_zero(coeff.subs(vals) - (-3 * D2 / e[3])), "H1: 3-3 S1 R != -3 Delta_2/e3"
    p, q, r = ms
    assert sp.expand(D2 - (p + q) * (q + r) * (r + p)) == 0, "H1: Delta_2 != (p+q)(q+r)(r+p)"
    print("H1 CONFIRMED: denominator 3 - 3 S1 R = -3 Delta_2/e3, Delta_2 = (p+q)(q+r)(r+p)")


def h2():
    ms, e, vals = true_values(4)
    es = sp.symbols("e1:5")
    P = power_sums_from_e([1, *es], 5)
    base = {es[0]: S1, es[2]: R * es[3]}
    e2 = sp.solve(sp.Eq(P[3].subs(base), P3), es[1])
    assert len(e2) == 1, "H2: P3 equation not linear in e2"
    # the e2-step divides by the coefficient of e2, -3 S1, nonzero
    assert sp.expand(P[3].subs(base)).coeff(es[1], 1) == -3 * S1
    num, den = sp.fraction(sp.together(P[5].subs(base).subs(es[1], e2[0]) - P5))
    num = sp.expand(num)
    assert sp.factor(den) == 9 * S1, f"H2: unexpected denominator {den}"
    assert sp.degree(num, es[3]) == 1, "H2: P5 equation is not linear in e4"
    c = sp.factor(num.coeff(es[3], 1))
    assert sp.expand(c + 15 * (P3 * R - R * S1**3 + 3 * S1**2)) == 0
    D3 = hurwitz_det(e, 3)
    ratio = sp.factor(sp.cancel(sp.together(c.subs(vals)) / D3))
    assert sp.expand(ratio * e[4] - 45) == 0, f"H2: coefficient/Delta_3 = {ratio}"
    prod = sp.expand(sp.Mul(*[ms[i] + ms[j] for i in range(4) for j in range(i + 1, 4)]))
    assert sp.expand(D3 - prod) == 0
    coeffs = sp.Poly(prod, *ms).coeffs()
    assert len(coeffs) == 38 and all(x > 0 for x in coeffs), "H2: Delta_3 term count/sign"
    # the unique root is the true e4
    sol = sp.solve(num, es[3])[0]
    assert is_zero(sol.subs(vals) - e[4]), "H2: solution is not the true e4"
    print(
        "H2 CONFIRMED: P5 equation linear in e4; numerator coefficient "
        "-15(P3 R - R S1^3 + 3 S1^2) = 45 Delta_3/e4 at the true data (over 9 S1); "
        "Delta_3 = prod(m_i+m_j), 38 terms, all positive; root = true e4"
    )


def h3():
    ms, e, vals = true_values(5)
    es = sp.symbols("e1:6")
    P = power_sums_from_e([1, *es], 7)
    base = {es[0]: S1, es[3]: R * es[4]}
    assert sp.expand(P[3].subs(base)).coeff(es[1], 1) == -3 * S1
    e2 = sp.solve(sp.Eq(P[3].subs(base), P3), es[1])[0]
    F = {}
    for k, s in ((5, P5), (7, P7)):
        num, den = sp.fraction(sp.together(P[k].subs(base).subs(es[1], e2) - s))
        f, _ = sp.factor_list(den)[0], None
        # denominator must be a constant times a power of S1 (nonzero)
        den_free = sp.factor(den)
        assert den_free.free_symbols <= {S1}, f"H3: P{k} denominator {den} involves unknowns"
        F[k] = sp.expand(num)
    e3, e5 = es[2], es[4]
    degs = {k: (sp.degree(F[k], e3), sp.degree(F[k], e5)) for k in F}
    assert degs[5] == (1, 1) and degs[7] == (2, 1), f"H3: degrees {degs}"
    a5 = sp.factor(F[5].coeff(e5, 1))
    assert sp.expand(a5 + 45 * S1 * (R * S1 - 1)) == 0, f"H3: e5-coefficient {a5}"
    # with the 9 S1 denominator restored the e5-coefficient is -5 (R S1 - 1)
    den5 = sp.fraction(sp.together(P[5].subs(base).subs(es[1], e2)))[1]
    assert sp.cancel(a5 / den5 + 5 * (R * S1 - 1)) == 0
    res = sp.resultant(F[5], F[7], e5)
    const, factors = sp.factor_list(res)
    lin = [f for f, k in factors if sp.degree(f, e3) == 1]
    extra = [(f, k) for f, k in factors if sp.degree(f, e3) == 0]
    assert len(lin) == 1 and all(sp.degree(f, e3) <= 1 for f, _ in factors), "H3: not linear in e3"
    assert extra == [(S1, 2)] and const == -9, f"H3: unexpected extra factors {const}, {extra}"
    L = lin[0]
    lead = sp.Poly(L, e3).LC()
    D4 = hurwitz_det(e, 4)
    ratio = sp.factor(sp.cancel(sp.together(lead.subs(vals)) / sp.cancel(D4 / e[5])))
    assert ratio == -945, f"H3: lead/(Delta_4/e5) = {ratio}"
    # spurious-solution analysis: F5 has degree 1 in e5 with leading coefficient
    # -45 S1 (R S1 - 1) != 0, so Res_{e5}(F5, F7) = 0 iff F7(e3, e5(e3)) = 0,
    # where e5(e3) is the unique root of F5; the extra factor S1^2 != 0. Hence
    # the system has exactly one solution; check it is the true e.
    # L, F5, F7 are polynomials; evaluate them at the true (data, e3, e5).
    # L linear with nonzero lead => its unique root is the true e3; F5 linear in
    # e5 with nonzero lead => its unique root is the true e5.
    at_truth = dict(vals)
    at_truth.update({e3: e[3], e5: e[5]})
    for name, poly in (("L", L), ("F5", F[5]), ("F7", F[7])):
        num = sp.numer(sp.together(poly.subs(at_truth)))
        assert sp.expand(num) == 0, f"H3: {name} does not vanish at the true e"
    assert is_zero(e2.subs(e3, e[3]).subs(vals) - e[2]), "H3: e2 is not the true e2"
    print(
        "H3 CONFIRMED: F5 has degree 1 in each of e3, e5; F7 degrees (2, 1); "
        "Res_e5(F5,F7) = -9 S1^2 * L(e3), L linear with lead = -945 Delta_4/e5; "
        "e5-step coefficient -5(R S1 - 1); unique solution = true (e2,e3,e4,e5); "
        "nonvanishing used: S1, R S1 - 1, Delta_4/e5"
    )


def h4_and_cs():
    for n in range(3, 9):
        Rg, *ms = ring(",".join(f"m{i}" for i in range(1, n + 1)), ZZ)
        e = ring_elementary(ms, Rg(1))
        D = evaluate_poly(generic_hurwitz_det(n, n - 1), e, Rg(0))
        prod = Rg(1)
        for i in range(n):
            for j in range(i + 1, n):
                prod *= ms[i] + ms[j]
        assert D == prod, f"H4 fails at n={n}"
        # R S1 - n^2 = sum_{i<j} (m_i - m_j)^2/(m_i m_j): multiply by e_n
        lhs = e[n - 1] * e[1] - n * n * e[n]
        rhs = Rg(0)
        for i in range(n):
            for j in range(i + 1, n):
                other = Rg(1)
                for k in range(n):
                    if k not in (i, j):
                        other *= ms[k]
                rhs += (ms[i] - ms[j]) ** 2 * other
        assert lhs == rhs, f"R S1 identity fails at n={n}"
    print(
        "H4 CONFIRMED (n=3..8): Delta_{n-1}(prod(z+m_i)) = +prod_{i<j}(m_i+m_j); "
        "and R S1 - n^2 = sum_{i<j}(m_i-m_j)^2/(m_i m_j) >= 0"
    )


def parity_lemma(jmax=11):
    """Lemma 1 of the paper: for odd j, every monomial of e_j written in the power
    sums p_1..p_j contains a p_i with i odd (so e_j lies in the ideal of the odd
    power sums), and symmetrically for p_j written in e_1..e_j."""
    p = sp.symbols(f"p1:{jmax + 1}")
    e = [sp.Integer(1)]
    for j in range(1, jmax + 1):  # j e_j = sum_{i=1}^j (-1)^(i-1) e_{j-i} p_i
        e.append(sp.expand(sum((-1) ** (i - 1) * e[j - i] * p[i - 1] for i in range(1, j + 1)) / j))
    es = sp.symbols(f"e1:{jmax + 1}")
    P = power_sums_from_e([1, *es], jmax)
    for j in range(1, jmax + 1, 2):
        for mon in sp.Poly(e[j], *p).monoms():
            assert any(mon[i] for i in range(0, jmax, 2)), f"e_{j} has a monomial with no odd p_i"
        for mon in sp.Poly(P[j], *es).monoms():
            assert any(mon[i] for i in range(0, jmax, 2)), f"p_{j} has a monomial with no odd e_i"
    print(f"Lemma 1 (parity) CONFIRMED for odd j <= {jmax}, in both directions")


def jacobian_factor():
    """det d(R, P1, P3, ..., P_{2n-3})/d(m) = c_n V(m) prod(m_i+m_j) / prod m_i^2.

    Scaling column i by m_i^2 turns the Jacobian into c_r x_i^r (r = 0..n-1,
    x_i = m_i^2, c_0 = -1, c_r = 2r - 1), a scaled Vandermonde in the x_i, and
    prod_{i<j}(x_j - x_i) = prod_{i<j}(m_j - m_i)(m_j + m_i). Checked as a
    polynomial identity.
    """
    for n in range(2, 8):
        Rg, *ms = ring(",".join(f"m{i}" for i in range(1, n + 1)), ZZ)
        # column i scaled by m_i^2: row 0 is d(1/m)/dm * m^2 = -1, row r >= 1 is
        # d(m^(2r-1))/dm * m^2 = (2r-1) m^(2r); each entry is a monomial, so the
        # Leibniz expansion is exact and cheap in the sparse ring.
        entry = lambda r, i: Rg(-1) if r == 0 else (2 * r - 1) * ms[i] ** (2 * r)
        lhs = Rg(0)
        for perm in itertools.permutations(range(n)):
            sign = sp.combinatorics.Permutation(list(perm)).signature()
            term = Rg(sign)
            for r in range(n):
                term *= entry(r, perm[r])
            lhs += term
        c = -1
        for r in range(1, n):
            c *= 2 * r - 1
        rhs = Rg(c)
        for i in range(n):
            for j in range(i + 1, n):
                rhs *= (ms[j] - ms[i]) * (ms[j] + ms[i])
        assert lhs == rhs, f"Jacobian factorisation fails at n={n}"
    print("Jacobian of (R,P1,...,P_{2n-3}) = c_n V(m) prod(m_i+m_j)/prod m_i^2 with "
          "c_n = -prod_{r<n}(2r-1) (n=2..7)")


def naive_degrees(n):
    """Degrees of the Newton equations after the forced linear substitutions."""
    es = sp.symbols(f"e1:{n + 1}")
    P = power_sums_from_e([1, *es], 2 * n - 3)
    Ps = {k: sp.Symbol(f"P{k}") for k in range(3, 2 * n - 2, 2)}
    base = {es[0]: S1, es[n - 2]: R * es[n - 1]}
    e2 = sp.solve(sp.Eq(P[3].subs(base), Ps[3]), es[1])[0]
    unknowns = [es[k] for k in range(2, n) if k != n - 2]
    out = {}
    for k in range(5, 2 * n - 2, 2):
        num = sp.expand(sp.numer(sp.together(P[k].subs(base).subs(es[1], e2) - Ps[k])))
        out[k] = sp.Poly(num, *unknowns).total_degree()
    return unknowns, out


def main():
    t = time.time()
    h1()
    h2()
    h3()
    h4_and_cs()
    parity_lemma()
    jacobian_factor()
    for n in (6, 7, 8):
        unknowns, degs = naive_degrees(n)
        assert max(degs.values()) >= 2, f"n={n}: naive system unexpectedly linear"
        sigma, minors, nterms = check_n(n)
        print(
            f"n={n}: naive Newton route (after e1=S1, e_(n-1)=R e_n, e2 from P3) in "
            f"{[str(u) for u in unknowns]} has total degrees "
            f"{ {f'P{k}': d for k, d in degs.items()} } -> not linear; "
            f"tanh/Pade route: {n} linear steps, determinant "
            f"{'+' if sigma > 0 else '-'}Delta_{n-1}/e_n = "
            f"{'+' if sigma > 0 else '-'}prod(m_i+m_j)/prod(m_i), never zero"
        )
    print(f"verify_elimination: all assertions passed [{time.time() - t:.1f}s]")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

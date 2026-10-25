"""Structure of the heat data of a closed orientable hyperbolic 2-orbifold (exact).

(1) From Ucar (4.25)+(4.33): p_l(m) = m * b_l(m)/K^l is an even polynomial of degree 2l+2,
    p_l(1) = 0, leading coefficient |B_{2l+2}| / (2 (l+1)! (2l+1)). Checked l <= LMAX;
    the paper Lemma 1 proves it for all l.
(2) phi_l(x) = p_l(x)/x = sum_{k=1}^{l+1} a_{l,k} psi_k(x), psi_k(x) = x^(2k-1) - 1/x,
    with a_{l,l+1} != 0. (Triangularity: C_0..C_l agree iff Psi_1..Psi_{l+1} agree.)
(3) Padding: adding an order-1 point changes neither the area nor any C_l (l <= LMAX),
    checked directly on the Fraction implementation for many signatures.
(4) T2(b) genus reduction. Pad two signatures (g; m), (g'; m') to a common length N.
    Equal area  <=>  R - R' = 2(g - g') (=: 2 Delta).  Equal C_0..C_{L-2} then forces
    P_{2k-1} - P'_{2k-1} = 2 Delta for k = 1..L-1, and conversely. Solved as a linear
    system in the unknown differences with sympy, for L <= LMAX + 1.
(5) heat_key and int_key (sig_common) induce the same agreement count on a sample.
Exits nonzero on any failure.
"""
import itertools
import random
import sys
from fractions import Fraction
from math import factorial

import sympy as sp

from sig_common import (b_cone, bern, heat_key, int_key, s_of, shared, shared_int)

LMAX = 15
x = sp.symbols("x")


def Q(fr):
    return sp.Rational(fr.numerator, fr.denominator)


def p_poly(l):
    """p_l(x) = x * b_l(x) / K^l from (4.25)+(4.33), as a sympy polynomial in x."""
    from sig_common import bern_half
    tot = 0
    for i in range(l + 1):
        ll = l - i
        s = sum(sp.binomial(2 * ll + 2, 2 * j) * (x ** (2 * j) - 1) * Q(bern(2 * j))
                * Q(bern_half(2 * ll + 2 - 2 * j)) for j in range(ll + 2))
        c = sp.Rational((-1) ** ll, 4 * factorial(ll + 1) * (2 * ll + 1)) * s   # = x * c_ll
        tot += sp.Rational(2, 4 ** i * factorial(i)) * c
    return sp.Poly(sp.expand(tot), x)


failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
        print("FAIL:", msg)
    assert cond, msg


print("(1)-(2) structure of p_l and triangular expansion in psi_k")
A = {}
for l in range(LMAX + 1):
    P = p_poly(l)
    check(P.degree() == 2 * l + 2, f"deg p_{l}")
    check(all(e[0] % 2 == 0 for e in P.monoms()), f"p_{l} even")
    check(P.eval(1) == 0, f"p_{l}(1) = 0")
    want = abs(Q(bern(2 * l + 2))) / (2 * factorial(l + 1) * (2 * l + 1))
    check(P.LC() == want, f"leading coefficient of p_{l}")
    # agreement with the Fraction implementation used by all searches (K = -1)
    for k in (1, 2, 3, 7, 12):
        check(Q(b_cone(l, k)) == (-1) ** l * P.eval(k) / k, f"b_cone({l},{k})")
    # phi_l = sum_k a_{l,k} psi_k : coefficients of x^(2k-1), k >= 1, in p_l(x)/x
    coeffs = {k: P.coeff_monomial(x ** (2 * k)) for k in range(1, l + 2)}
    phi = sp.expand(P.as_expr() / x)
    recon = sp.expand(sum(coeffs[k] * (x ** (2 * k - 1) - 1 / x) for k in coeffs))
    check(sp.simplify(phi - recon) == 0, f"phi_{l} = sum a psi")
    check(coeffs[l + 1] != 0, f"a_{{{l},{l+1}}} != 0")
    A[l] = coeffs
    if l <= 3:
        print(f"  l={l}: phi_l = " + " + ".join(f"({coeffs[k]}) psi_{k}" for k in coeffs))
print(f"  checked l = 0..{LMAX}: even, degree 2l+2, p_l(1)=0, Bernoulli leading term, "
      f"triangular in psi with nonzero diagonal")

print("(3) padding by order-1 points")
random.seed(1)
for trial in range(300):
    g = random.randint(0, 3)
    n = random.randint(0, 6)
    m = tuple(sorted(random.randint(2, 40) for _ in range(n)))
    r = random.randint(1, 4)
    mp = m + (1,) * r
    check(s_of(g, m) == s_of(g, mp), "area unchanged by padding")
    for l in range(LMAX + 1):
        check(sum(b_cone(l, y) for y in m) == sum(b_cone(l, y) for y in mp),
              f"C_{l} unchanged by padding")
check(all(b_cone(l, 1) == 0 for l in range(LMAX + 1)), "b_l(1) = 0")
print(f"  300 random signatures, l <= {LMAX}: area and every C_l unchanged; b_l(1) = 0")

print("(4) genus reduction: differences forced by equal area and equal C_0..C_{L-2}")
Delta = sp.symbols("Delta")
for L in range(2, LMAX + 2):
    dR = sp.symbols("dR")
    d = sp.symbols(f"d1:{L}")          # d[k-1] = P_{2k-1} - P'_{2k-1}, k = 1..L-1
    # padded to common length N: area difference 2pi(2(g-g') - (R - R'))
    eqs = [sp.Eq(2 * Delta - dR, 0)]
    for l in range(L - 1):
        # C_l - C'_l = sum_k a_{l,k} (d_k - dR)
        eqs.append(sp.Eq(sum(A[l][k] * (d[k - 1] - dR) for k in range(1, l + 2)), 0))
    sol = sp.solve(eqs, [dR, *d], dict=True)
    check(len(sol) == 1, f"unique solution L={L}")
    sol = sol[0]
    check(sol[dR] == 2 * Delta, f"dR = 2 Delta, L={L}")
    check(all(sp.simplify(sol[dk] - 2 * Delta) == 0 for dk in d), f"d_k = 2 Delta, L={L}")
print(f"  L = 2..{LMAX + 1}: unique solution R-R' = P_(2k-1)-P'_(2k-1) = 2(g-g') for all k <= L-1")

print("(5) heat_key and int_key give the same agreement count")
pool = []
for g in range(0, 3):
    for n in range(0, 5):
        for m in itertools.combinations_with_replacement(range(2, 13), n):
            if s_of(g, m) > 0:
                pool.append((g, m))
by_s = {}
for sig in pool:
    by_s.setdefault(s_of(*sig), []).append(sig)
pairs = 0
maxshared = 0
for s, cls in by_s.items():
    for a, b in itertools.combinations(cls, 2):
        k1 = shared(a, b, 8)
        k2 = shared_int(a, b, 8)
        check(k1 == k2, f"shared vs shared_int {a} {b}")
        check(heat_key(*a, k1) == heat_key(*b, k1), "heat_key agreement")
        check(int_key(*a, k2) == int_key(*b, k2), "int_key agreement")
        pairs += 1
        maxshared = max(maxshared, k1)
print(f"  {len(pool)} signatures (g<=2, n<=4, orders<=12), {pairs} equal-area pairs: "
      f"counts identical; max shared = {maxshared}")

if failures:
    print(f"{failures} FAILURES")
    sys.exit(1)
print("heat_structure.py: all assertions passed")

"""Exact checks for the extension and obstruction statements (Theorems VC4-VC6).

(A) VC6 (all orders, flat cone points): pairs of cone-order multisets m, m' with
        sum(1 - 1/m_i) = sum(1 - 1/m'_j) > 1   and   sum(m_i - 1/m_i) = sum(m'_j - 1/m'_j).
    Checks: the flat sphere |prod (z - z_i)^{alpha_i - 1} dz| has Gauss-Bonnet sum(alpha_i - 1) = -2
    with alpha_i = 1/m_i and alpha_0 = sum(1-1/m_i) - 1 > 0; the t^0 coefficients agree; the pair is
    separated at constant curvature -1 by c_3 (so variable curvature loses information Paper A has).
    Enumerates all such genus-0 pairs with orders <= 24 and at most 4 cone points.
(B) VC5 (finite order): the t^0 coefficient chi(|O|)/6 + sum (m-1)^2/(12m) is the only constraint;
    example S^2(2,2,2,2,2,2,2,2) vs S^2(3,3,3); the cone differences D_l at kappa = -1 are nonzero for
    l = 1..6 (the smooth part has to absorb them); the generalised Vandermonde det[eps_k^{e_i}],
    e = (1, -3, -5, ..., -(2L+1)), is nonzero at distinct positive rational nodes (L <= 6).
(C) VC4 (extension): the cone sums sum_i Pi_d(m_i), d = 1..n, and Paper A's Psi_1..Psi_n determine
    each other (triangular, exact); in the common-jet version the diagonal is beta_{l,l+1}(J), whose
    values for l <= 3 are taken from twisted_mp.py (b_0..b_3).
"""
from fractions import Fraction as Fr
from itertools import combinations_with_replacement
import random
import sympy as sp

k = sp.Symbol('k')


def a0(chi, ms):
    return Fr(chi, 6) + sum(Fr((m - 1)**2, 12 * m) for m in ms)


def chiorb(g, ms):
    return 2 - 2 * g - sum(1 - Fr(1, m) for m in ms)


def psi1(ms):
    return sum(m - Fr(1, m) for m in ms)


# Paper A cone polynomials p_l (closed form (phik),(bl)); b_l = kappa^l p_l(m)/m
def sigma_coeffs(n):
    u = sp.Symbol('u')
    ser = sp.series(u / sp.sin(u), u, 0, 2 * n + 2).removeO()
    return [ser.coeff(u, 2 * i) for i in range(n + 1)]


def p_l(l):
    sig = sigma_coeffs(l + 1)
    tot = 0
    for kk in range(l + 1):
        mphi = sp.Rational(1, 4) * sum(sig[kk + 1 - n] * 4**n * abs(sp.bernoulli(2 * n))
                                       / sp.factorial(2 * n) * (k**(2 * n) - 1) for n in range(1, kk + 2))
        tot += sp.factorial(2 * kk) / (sp.factorial(kk) * sp.factorial(l - kk)) * mphi
    return sp.expand(tot / 4**l)


PL = {l: p_l(l) for l in range(0, 7)}


def cone_sum(l, ms, kappa=-1):
    vals = [PL[l].subs(k, m) for m in ms]
    assert all(v.is_Rational for v in vals)
    return sum(sp.Rational(kappa)**l * v / m for v, m in zip(vals, ms))


# ---------------- (A) ----------------
A_pairs = []
pool = range(2, 25)
by_key = {}
for n in range(1, 5):
    for ms in combinations_with_replacement(pool, n):
        key = (sum(1 - Fr(1, m) for m in ms), psi1(ms))
        if key[0] > 1:
            by_key.setdefault(key, []).append(ms)
for key, lst in by_key.items():
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            A_pairs.append((lst[i], lst[j]))
A_pairs.sort(key=lambda pq: (sum(pq[0]) + sum(pq[1]), pq))
print("VC6 genus-0 pairs (orders <= 24, <= 4 cone points):", len(A_pairs))
for pq in A_pairs[:12]:
    print("   ", pq)
assert ((2, 8, 8), (3, 3, 12)) in A_pairs or ((3, 3, 12), (2, 8, 8)) in A_pairs
for kk in range(1, 9):
    m, mp = (2 * kk, 8 * kk, 8 * kk), (3 * kk, 3 * kk, 12 * kk)
    assert sum(1 - Fr(1, x) for x in m) == sum(1 - Fr(1, x) for x in mp) > 1
    assert psi1(m) == psi1(mp)
for (m, mp) in A_pairs:
    alpha = [Fr(1, x) for x in m]
    alpha0 = sum(1 - x for x in alpha) - 1
    assert alpha0 > 0
    assert sum(x - 1 for x in alpha) + (alpha0 - 1) == -2          # Gauss-Bonnet, flat sphere
    for g in range(0, 3):
        assert a0(2 - 2 * g, m) == a0(2 - 2 * g, mp)                # t^0 coefficient
        assert chiorb(g, m) == chiorb(g, mp)
    # at constant curvature -1 Paper A separates them at c_3 (t^1) unless P_3 also agrees
sep = [pq for pq in A_pairs if cone_sum(1, pq[0]) != cone_sum(1, pq[1])]
print("   of which separated at t^1 at curvature -1:", len(sep), "of", len(A_pairs))
assert cone_sum(1, (2, 8, 8)) != cone_sum(1, (3, 3, 12))

# ---------------- (B) ----------------
m, mp = (2,) * 8, (3, 3, 3)
assert a0(2, m) == a0(2, mp)
print("VC5 example S^2(2^8) vs S^2(3,3,3): a_0 =", a0(2, m), "; chi_orb =", chiorb(0, m), chiorb(0, mp))
for l in range(1, 7):
    D = cone_sum(l, mp) - cone_sum(l, m)
    assert D != 0
    print(f"   D_{l} (kappa=-1) = {D}")

random.seed(20261008)
for L in range(1, 7):
    e = [1] + [-(2 * i + 1) for i in range(1, L + 1)]
    for trial in range(5):
        nodes = sorted(random.sample(range(1, 400), L + 1))
        nodes = [sp.Rational(x, 97) for x in nodes]
        M = sp.Matrix([[nd**ex for nd in nodes] for ex in e])
        assert M.det() != 0
print("generalised Vandermonde: nonzero for L <= 6 (exact)")

# ---------------- (C) ----------------
# k*Pi_d(k) from twisted_mp.py output (exact, interpolated and verified there)
kPi = {1: (k - 1) * (k + 1) / 12,
       2: (k - 1) * (k + 1) * (k**2 + 11) / 720,
       3: (k - 1) * (k + 1) * (2 * k**4 + 23 * k**2 + 191) / 60480,
       4: (k - 1) * (k + 1) * (k**2 + 11) * (3 * k**4 + 10 * k**2 + 227) / 3628800}
# cross-check kPi_d numerically against the defining sum
for d, pol in kPi.items():
    for kv in range(1, 12):
        num = sum((4 * sp.sin(sp.pi * j / kv)**2)**(-d) for j in range(1, kv))
        assert abs(sp.N(num - pol.subs(k, kv), 40)) < sp.Float(10)**-30
# triangularity: k*Pi_d = sum_{n<=d} t_{d,n} (k^{2n} - 1) with t_{d,d} != 0
for d, pol in kPi.items():
    P = sp.Poly(sp.expand(pol), k)
    assert P.degree() == 2 * d and pol.subs(k, 1) == 0
    assert all(e % 2 == 0 for (e,) in P.as_dict())
# common-jet diagonal (from twisted_mp.py, b_l(C^2) top coefficients)
K, LK, L2K = sp.symbols('K LapK Lap2K')
diag = {0: sp.Integer(1), 1: 2 * K, 2: 12 * K**2 - 2 * LK, 3: 120 * K**3 - 42 * K * LK + L2K}
kap = sp.Symbol('kappa')
for l, dg in diag.items():
    const = dg.subs({LK: 0, L2K: 0, K: kap})
    assert sp.expand(const - sp.factorial(2 * l) / sp.factorial(l) * kap**l) == 0
print("diagonal beta_{l,l+1}(J), l<=3:", diag)
print("ALL ASSERTS PASSED")

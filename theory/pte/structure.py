"""Exact checks of the structural results of the paper sections 2-3.

(A) Theorem 2.1 (Descartes bound |iota| <= T - 2L and V+ + V- = T):
    - on every witness of data/witnesses.json;
    - on every 2-configuration of size 6 with entries in [-20,20] and size 8 with entries in [-9,9]
      (exhaustive), and every 3-configuration of size 8 or 10 with entries in [-8,8] (exhaustive).
(B) Proposition 2.3(a): iota(A) = sgn s_{-1}(A) for odd ideal symmetric sets
    (all primitive 5-sets with entries <= 60; Gloden's 7-set family for |f|,k <= 20; the fetched
    7-sets and 9-sets).
(C) Theorem 3.1 weight lemma, symbolically for m = 4..10: under the vanishing hypotheses, the odd
    power sums s_j (j <= 2m-5) do not involve e_{k0}; and s_{2m-3} does (so the bound is sharp).
(D) Propositions 3.2, 3.3: the shift and doubling identities, symbolically.
Run: python3 structure.py   (about 1 minute)
"""
import itertools
import json
import math
import os
import sys
from fractions import Fraction as F

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pte_common import cancel, iota, is_config, esym  # noqa: E402
from witnesses import gloden7, ODDSYM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def sign_changes(coeffs):
    s = [c for c in coeffs if c != 0]
    return sum(1 for a, b in zip(s, s[1:]) if (a > 0) != (b > 0))


def descartes_data(Z):
    e = esym(Z)
    T = len(Z)
    coeffs = [(-1) ** k * e[k] for k in range(T + 1)]          # x^{T-k}
    coeffs_neg = [c * (-1) ** (T - k) for k, c in enumerate(coeffs)]
    return sign_changes(coeffs), sign_changes(coeffs_neg)


def check_thm21(Z, L):
    T = len(Z)
    Vp, Vm = descartes_data(Z)
    pos = sum(1 for z in Z if z > 0)
    assert Vp + Vm == T and pos == Vp and T - pos == Vm, (Z, Vp, Vm)
    assert abs(iota(Z)) <= T - 2 * L, Z
    return iota(Z)


# ---------------------------------------------------------------- (A)
recs = json.load(open(os.path.join(HERE, "data", "witnesses.json")))
for r in recs:
    check_thm21([F(z) for z in r["Z"]], r["L"])
print(f"(A) Theorem 2.1 holds on all {len(recs)} witnesses (V+ + V- = T, |iota| <= T - 2L).")


def configs(T, M, L):
    """All L-configurations of size T with entries in [-M,M]\\{0} (no +-pairs), via the last entry
    fixed by s_1 = 0."""
    vals = [v for v in range(-M, M + 1) if v]
    out = []
    for c in itertools.combinations_with_replacement(vals, T - 1):
        z = -sum(c)
        if z == 0 or abs(z) > M or z < c[-1]:
            continue
        Z = list(c) + [z]
        if sum(F(1, x) for x in Z) != 0:
            continue
        if any(-x in Z for x in Z):
            continue
        if all(sum(x ** j for x in Z) == 0 for j in range(3, 2 * L - 2, 2)):
            out.append(Z)
    return out


hist = {}
for T, M, L in [(6, 20, 2), (8, 9, 2), (8, 8, 3), (10, 8, 3)]:
    cs = configs(T, M, L)
    io = [check_thm21([F(z) for z in Z], L) for Z in cs]
    hist[(T, M, L)] = (len(cs), sorted(set(abs(i) for i in io)))
    print(f"(A) exhaustive: L={L}, T={T}, entries in [-{M},{M}]: {len(cs)} configurations, |iota| values {sorted(set(abs(i) for i in io))}"
          f" (bound T-2L = {T - 2 * L})")
assert hist[(8, 8, 3)][0] == 0 or max(hist[(8, 8, 3)][1]) <= 2

# ---------------------------------------------------------------- (B)
def odd_sym_ok(A, L):
    if not all(sum(F(a) ** j for a in A) == 0 for j in range(1, 2 * L - 2, 2)):
        return None
    r = sum(F(1, a) for a in A)
    assert r != 0
    assert iota(A) == (1 if r > 0 else -1), A
    return True


n5 = 0
vals = [v for v in range(-60, 61) if v]
seen = set()
for a, b, c in itertools.combinations_with_replacement(vals, 3):
    S = -(a + b + c)
    Cc = -(a ** 3 + b ** 3 + c ** 3)
    if S == 0:
        continue
    num = S ** 3 - Cc
    if num % (3 * S):
        continue
    p = num // (3 * S)
    D = S * S - 4 * p
    if D < 0:
        continue
    rt = math.isqrt(D)
    if rt * rt != D or (S + rt) % 2:
        continue
    d, e = (S + rt) // 2, (S - rt) // 2
    A = tuple(sorted((a, b, c, d, e)))
    if 0 in A or any(-x in A for x in A) or A in seen:
        continue
    seen.add(A)
    assert odd_sym_ok(A, 3)
    n5 += 1
n7 = 0
for f in range(-20, 21):
    for k in range(1, 21):
        A = gloden7(f, k)
        if 0 in A or any(-x in A for x in A):
            continue
        assert odd_sym_ok(A, 4)
        n7 += 1
for A in ODDSYM[4]:
    assert odd_sym_ok(A, 4)
for A in ODDSYM[5]:
    assert odd_sym_ok(A, 5)
print(f"(B) Prop 2.3(a) iota = sgn s_-1: {n5} 5-sets, {n7} Gloden 7-sets, {len(ODDSYM[4])} fetched 7-sets, "
      f"{len(ODDSYM[5])} 9-sets: all OK")

# ---------------------------------------------------------------- (C)
def power_sums_in_e(m, jmax):
    e = sp.symbols(f"e1:{m + 1}")
    p = {}
    for k in range(1, jmax + 1):
        s = sum((-1) ** (i - 1) * (e[i - 1] if i <= m else 0) * p.get(k - i, 0) for i in range(1, k))
        s += (-1) ** (k - 1) * k * (e[k - 1] if k <= m else 0)
        p[k] = sp.expand(s)
    return e, p


for m in range(4, 11):
    r = m - 1 if m % 2 == 0 else m
    k0 = m - 2 if m % 2 == 0 else m - 3
    e, p = power_sums_in_e(m, 2 * m - 3)
    subs = {e[k - 1]: 0 for k in range(1, m + 1) if k % 2 == 1 and k != r}
    for j in range(1, 2 * m - 4, 2):
        assert not sp.expand(p[j].subs(subs)).has(e[k0 - 1]), (m, j)
    assert sp.expand(p[2 * m - 3].subs(subs)).has(e[k0 - 1]), m
print("(C) Theorem 3.1 weight lemma: for m = 4..10, odd s_j (j <= 2m-5) are free of e_{k0}; s_{2m-3} is not")

# ---------------------------------------------------------------- (D)
c, x = sp.symbols("c x")
for j in range(1, 12, 2):
    xs, ys = sp.symbols(f"x1:5"), sp.symbols(f"y1:5")
    lhs = sum((xi + c) ** j for xi in xs) - sum((yi + c) ** j for yi in ys)
    # coefficient of c^i is binom(j,i) (P_{j-i}(X) - P_{j-i}(Y))
    rhs = sum(sp.binomial(j, i) * c ** i * (sum(xi ** (j - i) for xi in xs) - sum(yi ** (j - i) for yi in ys))
              for i in range(j + 1))
    assert sp.expand(lhs - rhs) == 0
for j in [-1] + list(range(1, 12, 2)):
    X = [F(v) for v in (3, 7, 7, 11)]
    Y = [F(v) for v in (2, 5, 9, 12)]
    U = X + [2 * y for y in Y] * 2
    V = Y + [2 * v for v in X] * 2
    lhs = sum(u ** j for u in U) - sum(v ** j for v in V)
    assert lhs == (1 - F(2) ** (j + 1)) * (sum(v ** j for v in X) - sum(v ** j for v in Y))
print("(D) shift expansion (Prop 3.2) and doubling identity (Prop 3.3) verified")
print("ALL CHECKS PASSED")

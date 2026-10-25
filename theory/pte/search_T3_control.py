"""Positive control for search_T3.c (its decide() function, via `search_T3 test`).

Plants genuine rational 3-element sides U (so that S = sum u and C = sum u^3 are integers, as they
are when they equal the sums of an integral 5-element side V) and feeds (S, C, Rn, Rd) to the exact
decision routine the search uses. A genuine U must never be rejected: every planted case must be
accepted (1) or routed to the exact check (2, 3).  Families:
  A  integer U (K' = 1);
  B  U = {p1/q, p2/q, q t} with p1 + p2 = 0 mod q^3, q = 2..40 (non-integral, K' > 1);
  C  near-double roots U = {(q^3-d)/(2q), (q^3+d)/(2q), t}, q = 8..40, d = 1..400 (clustered);
  D  exactly repeated U = {u, u, w};
  E  generic small denominators by rejection, q = 2..6.
Also checks that search_T3 reproduces the exact counts of an independent exact search for v5 <= 30.
Run: python3 search_T3_control.py   (about 1 minute)
"""
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction as F
from math import gcd

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
binp = os.path.join(tempfile.gettempdir(), f"search_T3_ctl_{os.getpid()}")
subprocess.run(["clang", "-O2", "-o", binp, os.path.join(HERE, "search_T3.c"), "-lm"], check=True)

random.seed(20261006)
planted = {k: [] for k in "ABCDE"}


def add(fam, U):
    U = [F(u) for u in U]
    if min(U) <= 0:
        return
    S, C, R = sum(U), sum(u ** 3 for u in U), sum(1 / u for u in U)
    if S.denominator != 1 or C.denominator != 1 or max(U) > 1100:
        return
    planted[fam].append((int(S), int(C), R.numerator, R.denominator, U))


for _ in range(20000):
    add("A", [random.randint(1, 1000) for _ in range(3)])
for q in range(2, 41):
    for _ in range(200):
        p1 = random.randint(1, q ** 3 - 1)
        if p1 % q == 0:
            continue
        add("B", [F(p1, q), F(q ** 3 * random.randint(1, 2) - p1, q), q * random.randint(1, 30)])
for q in range(8, 41):
    for d in range(1, 401):
        if (q ** 3 - d) % 2 or d >= q ** 3:
            continue
        add("C", [F(q ** 3 - d, 2 * q), F(q ** 3 + d, 2 * q), random.randint(1, 1000)])
for _ in range(5000):
    u = random.randint(1, 800)
    add("D", [u, u, random.randint(1, 800)])
for q in range(2, 7):
    for _ in range(40000):
        add("E", [F(random.randint(1, 300 * q), q) for _ in range(3)])

bad = []
for fam, lst in planted.items():
    inp = "\n".join(f"{S} {C} {Rn} {Rd}" for S, C, Rn, Rd, _ in lst)
    out = subprocess.run([binp, "test"], input=inp, capture_output=True, text=True, check=True).stdout.split()
    res = list(map(int, out))
    assert len(res) == len(lst)
    counts = {k: res.count(k) for k in sorted(set(res))}
    nonint = sum(1 for *_, U in lst if any(u.denominator > 1 for u in U))
    print(f"family {fam}: {len(lst)} planted ({nonint} non-integral); decisions {counts}")
    for (S, C, Rn, Rd, U), r in zip(lst, res):
        if r <= 0:
            # r < 0 is legitimate only if the planted U is not three distinct-or-repeated positive reals,
            # which cannot happen for positive U; so any r <= 0 is a false negative.
            bad.append((fam, [str(u) for u in U], r))
print(f"false negatives: {len(bad)}", bad[:5])
assert not bad

# exact cross-check of the search counters for v5 <= 30
t = sp.symbols("t")
exact_real3 = exact_split = nV = 0
for v5 in range(1, 31):
    for v4 in range(1, v5 + 1):
        for v3 in range(1, v4 + 1):
            for v2 in range(1, v3 + 1):
                for v1 in range(1, v2 + 1):
                    if gcd(gcd(gcd(v1, v2), gcd(v3, v4)), v5) != 1:
                        continue
                    nV += 1
                    V = (v1, v2, v3, v4, v5)
                    S, C, R = sum(V), sum(v ** 3 for v in V), sum(F(1, v) for v in V)
                    e3 = F(C - S ** 3) / (3 * (1 - S * R))
                    e2 = R * e3
                    if not (e3 > 0 and e2 > 0):
                        continue
                    P = sp.Poly(t ** 3 - S * t ** 2 + sp.Rational(e2.numerator, e2.denominator) * t
                                - sp.Rational(e3.numerator, e3.denominator), t)
                    if sp.discriminant(P) >= 0 and all(r > 0 for r in sp.real_roots(P)) and len(sp.real_roots(P)) == 3:
                        exact_real3 += 1
                    if sum(m for f, m in P.factor_list()[1] if f.degree() == 1) == 3:
                        exact_split += 1
r = subprocess.run([binp, "30", "1", "30"], capture_output=True, text=True, check=True)
print("search_T3 30 1 30:", r.stderr.strip())
print(f"independent exact count, v5 <= 30: V-sets {nV}, cubics with 3 positive real roots {exact_real3}, splitting over Q {exact_split}")
assert f"V-sets {nV}," in r.stderr and f"3 positive real roots {exact_real3}," in r.stderr and exact_split == 0
os.remove(binp)
print("ALL CHECKS PASSED")

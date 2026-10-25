"""The sharpness statements (Proposition S3.3 and Remark S3.4).

  (a) general data at a triple order (n = k = 3, a = 8): q_s(z) = (z-a)^3 - s^3 has roots at distance
      exactly s from a, and its data (R, P_1, P_3) differ from those of (a,a,a) by O(s^3):
      exponent 1/3 for arbitrary (complex-root) data vectors                          [sympy, exact]
  (b) realisable data at a triple order: (a+s, a-s, a) has data change exactly of order s^2,
      so the exponent 1/2 of Remark 6.8 is attained                                   [sympy, exact]
  (c) Remark 6.8's inequality: for real |d_i| <= a,
      (P_3 - 3a^2 P_1)(a+d) - (P_3 - 3a^2 P_1)(a) = 3a sum d_i^2 + sum d_i^3 >= 2a sum d_i^2
      [sympy identity + exact rational grid]
  (d) realisable data at a double order (Prop. 6.7(i)), (2,8,8) with a = 8:
      dR = 2 s^2/(8(64 - s^2)), dP_1 = 0, dP_3 = 48 s^2                                 [sympy, exact]

Run from the repository root:
    python3 checks/proofs/check_sharpness.py
Writes checks/proofs/check_sharpness.txt.  Exits nonzero on any failure.
"""
import itertools
import os
import sys
from fractions import Fraction as Fr

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_sharpness.txt")
log = []
fails = 0


def check(cond, msg):
    global fails
    log.append(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


s = sp.symbols("s", positive=True)
z = sp.symbols("z")
a = sp.Integer(8)


def data_from_e(e1, e2, e3):
    P1 = e1
    P3 = e1 ** 3 - 3 * e1 * e2 + 3 * e3
    R = e2 / e3
    return [R, P1, P3]


base = data_from_e(3 * a, 3 * a ** 2, a ** 3)
# (a)
q = sp.expand((z - a) ** 3 - s ** 3)
e1, e2, e3 = -q.coeff(z, 2), q.coeff(z, 1), -q.coeff(z, 0)
dd = [sp.simplify(x - y) for x, y in zip(data_from_e(e1, e2, e3), base)]
orders = [sp.Poly(sp.series(d, s, 0, 7).removeO(), s).monoms()[-1][0] if d != 0 else None for d in dd]
check(dd[1] == 0 and orders[0] == 3 and orders[2] == 3, f"(a) general data: data change orders in s = {orders} (P_1 unchanged)")
roots = sp.solve(q, z)
check(len(roots) == 3 and all(sp.simplify(sp.expand((r - a) * sp.conjugate(r - a)) - s ** 2) == 0 for r in roots),
      "(a) every root of q_s is at distance exactly s from a")

# (b)
m = [a + s, a - s, a]
R = sum(1 / x for x in m)
P1 = sum(m)
P3 = sum(x ** 3 for x in m)
d_b = [sp.simplify(R - Fr(3, 8)), sp.simplify(P1 - 24), sp.simplify(P3 - 3 * 512)]
check(d_b[1] == 0 and sp.simplify(d_b[2] - 6 * a * s ** 2) == 0 and sp.simplify(d_b[0] - 2 * s ** 2 / (a * (a ** 2 - s ** 2))) == 0,
      f"(b) realisable triple: dR = {d_b[0]}, dP_1 = 0, dP_3 = {d_b[2]} (order s^2, roots move by s)")

# (c)
d1, d2, d3 = sp.symbols("d1 d2 d3")
dvec = [d1, d2, d3]
lhs = sum((a + d) ** 3 for d in dvec) - 3 * a ** 2 * sum(a + d for d in dvec) - (3 * a ** 3 - 9 * a ** 3)
check(sp.expand(lhs - (3 * a * sum(d ** 2 for d in dvec) + sum(d ** 3 for d in dvec))) == 0, "(c) identity 3a sum d^2 + sum d^3")
grid = [Fr(j, 4) for j in range(-32, 33)]          # |d| <= a = 8
ok = all(3 * 8 * sum(x * x for x in t) + sum(x ** 3 for x in t) >= 2 * 8 * sum(x * x for x in t)
         for t in itertools.product(grid[::4], repeat=3))
check(ok, "(c) >= 2a sum d^2 on an exact grid with |d_i| <= a")

# (d)
m = [sp.Integer(2), a + s, a - s]
dR = sp.simplify(sum(1 / x for x in m) - sp.Rational(3, 4))
dP1 = sp.simplify(sum(m) - 18)
dP3 = sp.simplify(sum(x ** 3 for x in m) - (8 + 2 * 512))
check(sp.simplify(dR - 2 * s ** 2 / (8 * (64 - s ** 2))) == 0 and dP1 == 0 and sp.simplify(dP3 - 48 * s ** 2) == 0,
      f"(d) (2,8,8) double order: dR = {dR}, dP_1 = {dP1}, dP_3 = {dP3}")

# (e) all orders equal, n = 4: Remark 6.8's inequality and the family (a+s, a-s, a, a)
d4 = sp.symbols("d4")
dv = [d1, d2, d3, d4]
lhs4 = sum((a + d) ** 3 for d in dv) - 3 * a ** 2 * sum(a + d for d in dv) - (4 * a ** 3 - 12 * a ** 3)
check(sp.expand(lhs4 - (3 * a * sum(d ** 2 for d in dv) + sum(d ** 3 for d in dv))) == 0, "(e) n=4: identity 3a sum d^2 + sum d^3")
m4 = [a + s, a - s, a, a]
d_e = [sp.simplify(sum(1 / x for x in m4) - sp.Rational(4, 8)), sp.simplify(sum(m4) - 32),
       sp.simplify(sum(x ** 3 for x in m4) - 4 * 512), sp.simplify(sum(x ** 5 for x in m4) - 4 * 8 ** 5)]
check(d_e[1] == 0 and all(sp.limit(d / s ** 2, s, 0) not in (0, sp.oo, -sp.oo) for d in (d_e[0], d_e[2], d_e[3])),
      f"(e) n=4, (a+s,a-s,a,a): dP_1 = 0 and dR, dP_3, dP_5 are exactly of order s^2")

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{len(log)} checks, {fails} failures\n")
print(f"{len(log)} checks, {fails} failures")
sys.exit(1 if fails else 0)

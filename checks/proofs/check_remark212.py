"""The divergence of the heat series (Section 4).

  (a) Phi_m has simple poles at u = +-pi/m with principal parts -+1/(4 m sin(pi/m) (u -+ pi/m)),
      and no other singularity in |u| < 2 pi/m (m >= 3), |u| < 3 pi/2 (m = 2)            [50 digits]
  (b) phi_k(m) = (m/pi)^{2k} (1 + eps_k)/(2 pi sin(pi/m)) with |eps_k| <= C 4^{-k} (m >= 3), 9^{-k} (m = 2)
      [exact phi_k, 60 digits, k <= 120]
  (c) p_l(m)/m = A_l(m) (1 + pi^2/(2 m^2 (2l-1)) + O(l^-2)), A_l(m) = (2l)!/(l! m sin(pi/m)) (m/2pi)^{2l+1}:
      l^2 * remainder bounded and tending to pi^4/(32 m^4)                                [60 digits, l <= 120]
  (d) |alpha_{l+1}| <= 4^{-l-1}/(l+1)! + 4 e^{pi^2} (2l+1)!/(l! (2pi)^{2l+2})                                [exact vs float bound, l <= 60]
  (e) |c_{l+2}|/(l |c_{l+1}|) -> M^2/pi^2 for (0;2,8,8), (0;3,3,12), (1;2,3), (10^4; 2)       [60 digits, l = 150]
  (f) genus 10^4 with one cone point of order 2: the smooth part dominates exactly for l <= 8,
      with opposite sign  [exact]

Run from the repository root:
    python3 checks/proofs/check_remark212.py
Writes checks/proofs/check_remark212.txt.  Exits nonzero on any failure.
"""
import os
import sys
from fractions import Fraction as Fr
from math import comb, factorial

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_remark212.txt")
log = []
fails = 0


def check(cond, msg):
    global fails
    log.append(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


_B = {0: Fr(1)}


def B(n):
    if n not in _B:
        _B[n] = -sum(comb(n + 1, j) * B(j) for j in range(n)) / (n + 1)
    return _B[n]


def B_half(n):
    return Fr(1) if n == 0 else (Fr(2) ** (1 - n) - 1) * B(n)


def alpha(k):
    return Fr((-1) ** k, factorial(k) * 4 ** k) * sum(comb(k, l) * Fr(-4) ** l * B_half(2 * l) for l in range(k + 1))


def sig(i):
    return Fr((-1) ** (i + 1) * (2 ** (2 * i) - 2)) * B(2 * i) / factorial(2 * i)


_phi = {}


def m_phi(k, m):
    if (k, m) not in _phi:
        _phi[(k, m)] = sum(Fr(1, 4) * sig(k + 1 - n) * Fr(4 ** n) * abs(B(2 * n)) / factorial(2 * n) * (Fr(m) ** (2 * n) - 1)
                           for n in range(1, k + 2))
    return _phi[(k, m)]


def p(l, m):
    return sum(Fr(factorial(2 * k), 4 ** l * factorial(k) * factorial(l - k)) * m_phi(k, m) for k in range(l + 1))


def mpq(q):
    return mp.mpf(q.numerator) / q.denominator


mp.mp.dps = 60
# (a)
for m in [2, 3, 5, 8]:
    u0 = mp.pi / m
    Phi = lambda u: (mp.cot(u) - m * mp.cot(m * u)) / (4 * m * mp.sin(u))
    for sgn in (1, -1):
        h = mp.mpf(10) ** -20
        res = Phi(sgn * u0 + h) * h
        check(abs(res + sgn / (4 * m * mp.sin(mp.pi / m))) < mp.mpf(10) ** -15, f"(a) m={m}: residue at {'+' if sgn > 0 else '-'}pi/m = {mp.nstr(res, 12)}")
    # remainder after removing the two principal parts is bounded on |u| = 0.95 * (next radius)
    rnext = 2 * mp.pi / m if m >= 3 else 3 * mp.pi / 2
    pp = lambda u: (-1 / (u - u0) + 1 / (u + u0)) / (4 * m * mp.sin(mp.pi / m))
    vals = [abs(Phi(z) - pp(z)) for z in [0.95 * rnext * mp.expjpi(mp.mpf(j) / 50 + mp.mpf(1) / 100) for j in range(100)]]
    check(max(vals) < 10 ** 6, f"(a) m={m}: Phi - principal parts bounded ({mp.nstr(max(vals), 5)}) on |u| = 0.95 * {mp.nstr(rnext, 5)}")

# (b)
mp.mp.dps = 150
for m in [2, 3, 5, 12]:
    q = mp.mpf(1) / 9 if m == 2 else mp.mpf(1) / 4
    eps = []
    for k in range(5, 121, 5):
        main = (mp.mpf(m) / mp.pi) ** (2 * k) / (2 * mp.pi * mp.sin(mp.pi / m))
        eps.append(abs(mpq(m_phi(k, m) / m) / main - 1) / q ** k)
    check(max(eps) < 50 and eps[-1] < 50, f"(b) m={m}: |eps_k| / q^k <= {mp.nstr(max(eps), 5)} for 5 <= k <= 120, q = {mp.nstr(q, 3)}")

# (c)
mp.mp.dps = 60
for m in [2, 3, 4, 8, 12]:
    lims = []
    for l in [20, 60, 120]:
        A = mp.factorial(2 * l) / (mp.factorial(l) * m * mp.sin(mp.pi / m)) * (mp.mpf(m) / (2 * mp.pi)) ** (2 * l + 1)
        rem = mpq(p(l, m)) / m / A - 1 - mp.pi ** 2 / (2 * m * m * (2 * l - 1))
        lims.append(l * l * rem)
    target = mp.pi ** 4 / (32 * m ** 4)
    check(abs(lims[-1] - target) < 0.1 * target + mp.mpf(10) ** -6 and all(abs(x) < 2 for x in lims),
          f"(c) m={m}: l^2 * remainder at l = 20, 60, 120: {[mp.nstr(x, 6) for x in lims]}; pi^4/(32 m^4) = {mp.nstr(target, 6)}")

# (d)
for l in range(0, 61):
    bound = (mp.mpf(4) ** (-l - 1) / mp.factorial(l + 1)
             + 4 * mp.e ** (mp.pi ** 2) * mp.factorial(2 * l + 1) / (mp.factorial(l) * (2 * mp.pi) ** (2 * l + 2)))
    check(abs(mpq(alpha(l + 1))) <= bound, f"(d) l={l}: |alpha_(l+1)| <= 4^(-l-1)/(l+1)! + 4 e^(pi^2) (2l+1)!/(l! (2pi)^(2l+2))")
log[:] = [x for x in log if not x.startswith("PASS (d)")] + ["PASS (d) bound holds for 0 <= l <= 60"] if all(
    not x.startswith("FAIL (d)") for x in log) else log


# (e)
def c_coef(j, g, ms):
    area4pi = Fr(2 * g - 2) / 2 + sum(Fr(1) - Fr(1, x) for x in ms) / 2
    l = j - 2
    return alpha(l + 1) * area4pi + sum(Fr((-1) ** l) * p(l, x) / x for x in ms)


for g, ms in [(0, (2, 8, 8)), (0, (3, 3, 12)), (1, (2, 3)), (10 ** 4, (2,))]:
    l = 150
    ratio = abs(mpq(c_coef(l + 2, g, ms))) / (l * abs(mpq(c_coef(l + 1, g, ms))))
    M = max(ms)
    check(abs(ratio * mp.pi ** 2 / M ** 2 - 1) < 0.02, f"(e) (g={g}; {ms}): pi^2 |c_(l+2)|/(l |c_(l+1)| M^2) at l=150 = {mp.nstr(ratio * mp.pi ** 2 / M ** 2, 8)}")

# (f)
g, ms = 10 ** 4, (2,)
area4pi = Fr(2 * g - 2) / 2 + Fr(1, 2) / 2
dom = [abs(alpha(l + 1) * area4pi) > abs(Fr((-1) ** l) * p(l, 2) / 2) for l in range(20)]
opp = all(alpha(l + 1) * Fr((-1) ** l) < 0 for l in range(20))
check(dom == [True] * 9 + [False] * 11 and opp, "(f) genus 10^4, one cone of order 2: smooth part dominates exactly for l <= 8, with opposite sign")

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{len(log)} checks, {fails} failures\n")
print("\n".join(log))
print(f"{fails} failures")
sys.exit(1 if fails else 0)

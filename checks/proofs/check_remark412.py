"""The moments alpha_k and the remainders (Propositions 2.8 and B.2).

  (a) the moments  int_0^inf r^{2k+1}/(e^{2 pi r}+1) dr = (1 - 2^{-2k-1}) (-1)^k B_{2k+2}/(4(k+1))  [quadrature]
  (b) the identity term I(t) = (A/4pi) int r tanh(pi r) e^{-t(1/4+r^2)} dr has the expansion
      (A/4pi t) sum_k alpha_k t^k with alpha_k exactly the paper's (eq:pl), k <= 14        [exact]
  (c) I(t) against its truncated series at small t                                       [quadrature]
  (d) the a-priori Weyl bound used by Lemma 4.7: #{lambda_j <= x} <= e Z(1/x), on the
      round-sphere spectrum, where every quantity is explicit                             [exact/float]

Run from the repository root:
    python3 checks/proofs/check_remark412.py
Writes checks/proofs/check_remark412.txt.  Exits nonzero on any failure.
"""
import os
import sys
from fractions import Fraction as Fr
from math import comb, factorial

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_remark412.txt")
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


def mpq(q):
    return mp.mpf(q.numerator) / q.denominator


def alpha_paper(k):
    return Fr((-1) ** k, factorial(k) * 4 ** k) * sum(comb(k, l) * Fr(-4) ** l * B_half(2 * l) for l in range(k + 1))


def nu(k):
    """int_0^inf r^{2k+1}/(e^{2 pi r}+1) dr."""
    return (1 - Fr(1, 2 ** (2 * k + 1))) * Fr((-1) ** k) * B(2 * k + 2) / (4 * (k + 1))


# (a)
mp.mp.dps = 30
for k in range(0, 9):
    num = mp.quad(lambda r: r ** (2 * k + 1) / (mp.e ** (2 * mp.pi * r) + 1), [0, 5, 20, mp.inf])
    check(abs(num - mpq(nu(k))) < mp.mpf(10) ** -25 * abs(num), f"(a) moment k={k}: {mp.nstr(num, 15)}")

# (b)  I(t) * 4pi/A = e^{-t/4} ( 1/t - 4 sum_k (-t)^k/k! nu(k) );  coefficient of t^{K-1} is alpha_K
KMAX = 14
inner = [Fr(1)] + [Fr(-4) * Fr((-1) ** k, factorial(k)) * nu(k) for k in range(KMAX)]   # t * (...) by powers t^0, t^1, ...
exp_q = [Fr((-1) ** i, 4 ** i * factorial(i)) for i in range(KMAX + 1)]
for K in range(KMAX + 1):
    a = sum(exp_q[i] * inner[K - i] for i in range(K + 1))
    check(a == alpha_paper(K), f"(b) alpha_{K} from the identity term = {a}")
check([alpha_paper(k) for k in range(5)] == [1, Fr(-1, 3), Fr(1, 15), Fr(-4, 315), Fr(1, 315)],
      "(b) alpha_0..alpha_4 = 1, -1/3, 1/15, -4/315, 1/315 (manuscript)")

# (c)
mp.mp.dps = 25
for t in [mp.mpf("0.01"), mp.mpf("0.005"), mp.mpf("0.0025")]:
    It = mp.quad(lambda r: r * mp.tanh(mp.pi * r) * mp.e ** (-t * (mp.mpf(1) / 4 + r * r)), [-mp.inf, -30, 0, 30, mp.inf])
    ser = sum(mpq(alpha_paper(k)) * t ** (k - 1) for k in range(4))
    rel = (It - ser) / (mpq(alpha_paper(4)) * t ** 3)
    check(abs(rel - 1) < mp.mpf("0.05"), f"(c) t={t}: (I - series_(k<4))/(alpha_4 t^3) = {mp.nstr(rel, 8)}")

# (d)  unit sphere: eigenvalues l(l+1), multiplicity 2l+1
for x in [10, 100, 1000, 10 ** 4]:
    N = sum(2 * l + 1 for l in range(0, 200) if l * (l + 1) <= x)
    Z = mp.nsum(lambda l: (2 * l + 1) * mp.e ** (-l * (l + 1) / mp.mpf(x)), [0, mp.inf])
    check(N <= mp.e * Z, f"(d) x={x}: N(x) = {N} <= e Z(1/x) = {mp.nstr(mp.e * Z, 8)}")

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{len(log)} checks, {fails} failures\n")
print(f"{len(log)} checks, {fails} failures")
sys.exit(1 if fails else 0)

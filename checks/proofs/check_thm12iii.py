"""The explicit constant C_hyp of Lemma 2.5.

Theorem 4.9(b) gives, for 0 < t <= l^2/(2(1+l)),
  |Z1 - Z2| <= pi e^{3D}/(A(1-e^{-l})) * l e^{l/2} (1 + 2t/(l-t)) * e^{-l^2/4t}/sqrt(4 pi t).
The statement in the paper replaces (1 + 2t/(l-t)) by its maximum (2+3l)/(2+l) on that range:
  C(A, l, D) = sqrt(pi) e^{3D} l e^{l/2} (2+3l) / (2 A (1-e^{-l}) (2+l)).

  (a) on 0 < t <= l^2/(2(1+l)):  t < l  and  1 + 2t/(l-t) <= (2+3l)/(2+l), with equality at the endpoint
      [sympy, exact; plus a rational grid]
  (b) pi/sqrt(4 pi) = sqrt(pi)/2, so the two displays agree                        [sympy]
  (c) the two monotonicity facts of the proof of Theorem 4.9(b) on that range        [sympy + grid]
  (d) the erfc step: int_l^inf e^x phi_t(x) dx <= 2 t l/(l-t) e^{l/2} e^{-l^2/4t}/sqrt(4 pi t),
      phi_t(x) = x e^{-x/2} e^{-x^2/4t}/sqrt(4 pi t)                                 [quadrature grid]
  (e) the counting bound of Lemma 4.8 enters only through n(x) <= (pi/A) e^{x+3D}: the final bound
      dominates (1/(1-e^{-l})) int phi_t dn for the extremal counting function
      n(x) = floor((pi/A) e^{x+3D}) restricted to x >= l                             [quadrature grid]

Run from the repository root:
    python3 checks/proofs/check_thm12iii.py
Writes checks/proofs/check_thm12iii.txt.  Exits nonzero on any failure.
"""
import os
import sys
from fractions import Fraction as Fr

import mpmath as mp
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_thm12iii.txt")
log = []
fails = 0


def check(cond, msg):
    global fails
    log.append(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


l, t, x = sp.symbols("ell t x", positive=True)
tmax = l ** 2 / (2 * (1 + l))

# (a)
check(sp.simplify(l - tmax - l * (l + 2) / (2 * (l + 1))) == 0 and True, "(a) l - tmax = l(l+2)/(2(l+1)) > 0, so t < l")
fac = 1 + 2 * t / (l - t)
check(sp.simplify(sp.diff(fac, t) - 2 * l / (l - t) ** 2) == 0, "(a) d/dt (1 + 2t/(l-t)) = 2l/(l-t)^2 > 0: increasing in t")
check(sp.simplify(fac.subs(t, tmax) - (2 + 3 * l) / (2 + l)) == 0, "(a) value at t = l^2/(2(1+l)) is (2+3l)/(2+l)")
for lv in [Fr(1, 100), Fr(1, 3), Fr(1), Fr(5, 2), Fr(10), Fr(100)]:
    tm = lv ** 2 / (2 * (1 + lv))
    ok = all(1 + 2 * tv / (lv - tv) <= (2 + 3 * lv) / (2 + lv) for tv in [tm * Fr(j, 50) for j in range(1, 51)])
    check(ok and (2 + 3 * lv) / (2 + lv) < 3, f"(a) l={lv}: grid of 50 rational t in (0, tmax]: factor <= (2+3l)/(2+l) < 3")

# (b)
check(sp.simplify(sp.pi / sp.sqrt(4 * sp.pi) - sp.sqrt(sp.pi) / 2) == 0, "(b) pi/sqrt(4 pi) = sqrt(pi)/2")

# (c) for x >= l and t <= tmax:  x/(2t) >= 1/l + 1,  so 1/x + 1/2 - x/(2t) <= 1/l + 1/2 - 1/l - 1 < 0
check(sp.simplify(l / (2 * tmax) - (1 / l + 1)) == 0, "(c) l/(2 tmax) = 1/l + 1")
check(sp.simplify(tmax - l ** 2 / 2) != 0 and sp.simplify(l ** 2 / 2 - tmax - l ** 3 / (2 * (1 + l))) == 0,
      "(c) tmax <= l^2/2 (phi_t decreasing on [l, inf) needs t <= l^2/2)")

# (d), (e)
mp.mp.dps = 30


def phi(tv, xv):
    return xv * mp.e ** (-xv / 2) * mp.e ** (-xv ** 2 / (4 * tv)) / mp.sqrt(4 * mp.pi * tv)


for lv in [mp.mpf("0.2"), mp.mpf(1), mp.mpf(3), mp.mpf(8)]:
    tm = lv ** 2 / (2 * (1 + lv))
    for frac in [mp.mpf(1), mp.mpf("0.5"), mp.mpf("0.1")]:
        tv = tm * frac
        lhs = mp.quad(lambda xv: mp.e ** xv * phi(tv, xv), [lv, lv + 1, lv + 10, mp.inf])
        rhs = 2 * tv * lv / (lv - tv) * mp.e ** (lv / 2) * mp.e ** (-lv ** 2 / (4 * tv)) / mp.sqrt(4 * mp.pi * tv)
        check(lhs <= rhs, f"(d) l={mp.nstr(lv, 3)}, t={mp.nstr(tv, 4)}: erfc step {mp.nstr(lhs, 6)} <= {mp.nstr(rhs, 6)}")
        # (e) for every counting function n <= nmaj with n = 0 below l, int_[l,inf) phi dn = -int_l^inf n phi'
        #     <= -int_l^inf nmaj phi' (phi' <= 0 there); evaluate that majorant directly.
        A, D = mp.mpf(3), mp.mpf("0.7")
        nmaj = lambda xv: mp.pi / A * mp.e ** (xv + 3 * D)
        dphi = lambda xv: mp.diff(lambda y: phi(tv, y), xv)
        check(all(dphi(lv + mp.mpf(k) / 4) < 0 for k in range(0, 40)), f"(c) l={mp.nstr(lv, 3)}, t={mp.nstr(tv, 4)}: phi_t decreasing on [l, l+10]")
        stieltjes = mp.quad(lambda xv: -nmaj(xv) * dphi(xv), [lv, lv + 1, lv + 10, lv + 60])
        C = mp.sqrt(mp.pi) * mp.e ** (3 * D) * lv * mp.e ** (lv / 2) * (2 + 3 * lv) / (2 * A * (1 - mp.e ** (-lv)) * (2 + lv))
        bound = C * mp.e ** (-lv ** 2 / (4 * tv)) / mp.sqrt(tv)
        check(stieltjes / (1 - mp.e ** (-lv)) <= bound,
              f"(e) l={mp.nstr(lv, 3)}, t={mp.nstr(tv, 4)}: Hyp majorant {mp.nstr(stieltjes / (1 - mp.e ** (-lv)), 6)} <= C t^-1/2 e^-l^2/4t = {mp.nstr(bound, 6)}")

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{len(log)} checks, {fails} failures\n")
print(f"{len(log)} checks, {fails} failures")
sys.exit(1 if fails else 0)

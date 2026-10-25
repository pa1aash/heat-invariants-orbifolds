"""Growth of f(A) (the paper section 4) and the general constructions of Theorem 3.4.

(1) Theorem 4.1 versus [Sig] Cor. N1, exactly: with x = A/2pi,
      new(x) = floor(sqrt((x-1)/3)) + 2   (x >= 4),     old(x) = floor(log_4(x+1)) + 2   (x >= 3).
    Both are step functions; we compare them on every interval between consecutive breakpoints
    (new: x = 3(L-1)^2 + 1; old: x = 4^t - 1) for x in [4, 10^6], and assert new >= old everywhere,
    new > old exactly on [13,15) and [28, 10^6).
(2) Theorem 3.4, instantiated (unoptimised) for L = 2..7 from fetched ideal PTE solutions and
    odd-power equalities:  doubling (tau_L <= 6 N_odd(L)), two shifts (T_L <= 4 N(2L-3)),
    translation + doubling (T_cone_L <= 6 N(2L-3)).  Every object is checked exactly and its pair is
    checked with the actual cone coefficients.
(3) The thresholds A_L of the paper section 5 from data/witnesses.json.
Run: python3 growth.py   (about 1 minute)
"""
import json
import math
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pte_common import cancel, iota, is_config, normalize, realise, shared_exact, area_over_2pi, level, pte_degree, pm  # noqa: E402
from witnesses import PTE, ODDEQ, shift_piece, doubling, scale_combine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- (1)
def isqrt_floor_frac(q):
    n = math.isqrt(q.numerator // q.denominator)
    while F((n + 1) ** 2) <= q:
        n += 1
    while F(n * n) > q:
        n -= 1
    return n


def new(x):
    return isqrt_floor_frac((x - 1) / 3) + 2


def old(x):
    t = 0
    while 4 ** (t + 1) <= x + 1:
        t += 1
    return t + 2


XMAX = 10 ** 6
bps = {F(4), F(XMAX)}
L = 2
while 3 * (L - 1) ** 2 + 1 <= XMAX:
    bps.add(F(3 * (L - 1) ** 2 + 1))
    L += 1
t = 1
while 4 ** t - 1 <= XMAX:
    bps.add(F(4 ** t - 1))
    t += 1
bps = sorted(b for b in bps if b >= 4)
strict = []   # maximal intervals [a, b) on which new > old
for a, b in zip(bps, bps[1:]):
    pts = (a, (a + b) / 2, b - F(1, 10 ** 9))
    for x in pts:
        assert new(x) >= old(x), x
    gt = [new(x) > old(x) for x in pts]
    assert len(set(gt)) == 1, (a, b)           # both sides are constant on [a, b)
    if gt[0]:
        if strict and strict[-1][1] == a:
            strict[-1] = (strict[-1][0], b)
        else:
            strict.append((a, b))
print(f"(1) Theorem 4.1 bound >= Cor. N1 bound on [4, 10^6] ({len(bps)} breakpoints); strictly larger exactly on "
      + ", ".join(f"[{a}, {b})" for a, b in strict))
assert strict == [(F(13), F(15)), (F(28), F(XMAX))]
for x in [F(28), F(10 ** 3), F(10 ** 6)]:
    print(f"    A/2pi = {x}: new {new(x)}, old {old(x)}")

# ---------------------------------------------------------------- (2)
print("(2) Theorem 3.4 constructions (unoptimised):")
ideal = {2: (pm([1]), [F(0), F(0)]), }
ideal_by_degree = {3: PTE[4], 5: PTE[6], 7: PTE[8], 9: PTE[10][1], 11: PTE[12]}
for L in range(2, 8):
    k = 2 * L - 3
    if L == 2:
        X, Y = [0, 3], [1, 2]          # [0,3] =_1 [1,2]
    else:
        X, Y = ideal_by_degree[k]
    assert pte_degree(X, Y) >= k
    n = len(X)
    # genus: Z(c) u lam Z(c'), c in an unbalanced window (largest gap), c' > -min
    pts = sorted(set(F(v) for v in X + Y))
    windows = [(a, b) for a, b in zip(pts, pts[1:])
               if sum(1 for x in X if x > (a + b) / 2) != sum(1 for y in Y if y > (a + b) / 2)]
    a, b = max(windows, key=lambda w: w[1] - w[0])
    c = -((a + b) / 2 + F(1, 7))
    c2 = -min(pts) + F(1, 3)
    P1, P2 = shift_piece(X, Y, c), shift_piece(X, Y, c2)
    assert iota(cancel(P1)) != 0 and iota(cancel(P2)) == 0
    Zg = scale_combine(P1, P2)
    assert Zg and is_config(Zg, L) and iota(Zg) != 0 and len(Zg) <= 4 * n
    ga, gb = realise(normalize(Zg))
    assert shared_exact(ga, gb, level(normalize(Zg)) + 1) >= L
    # balanced: doubling of an odd-power equality (or of the PTE translated to positive)
    if L in ODDEQ:
        Xo, Yo = ODDEQ[L][0]
    else:
        m = min(X + Y)
        Xo, Yo = [x - m + 2 for x in X], [y - m + 2 for y in Y]
    no = len(Xo)
    Zb = cancel(doubling(Xo, Yo))
    assert is_config(Zb, L) and iota(Zb) == 0 and len(Zb) <= 6 * no
    # cone: translate so that the least element is 1, then double
    m = min(X + Y)
    Xc, Yc = [x - m + 1 for x in X], [y - m + 1 for y in Y]
    if 1 in Yc:
        Xc, Yc = Yc, Xc
    Zc = cancel(doubling(Xc, Yc))
    Zcn = normalize(Zc)
    assert is_config(Zc, L) and iota(Zc) == 0 and len(Zc) <= 6 * n and (1 in Zcn or -1 in Zcn)
    ca, cb = realise(Zcn, genus0_only=True)
    assert len(ca[1]) != len(cb[1]) and shared_exact(ca, cb, L + 3) >= L
    print(f"    L={L}: N(2L-3)<={n}: genus T={len(Zg)} <= {4 * n}; balanced T={len(Zb)} <= {6 * no}; "
          f"cone T={len(Zc)} <= {6 * n} (cone counts {len(ca[1])} vs {len(cb[1])})")

# ---------------------------------------------------------------- (3)
recs = json.load(open(os.path.join(HERE, "data", "witnesses.json")))
print("(3) thresholds: f(A) >= L+1 for A/2pi >= A_L (least area among the section-5 witnesses)")
for L in range(2, 8):
    best = min((r for r in recs if r["L"] == L), key=lambda r: F(r["area_over_2pi"]))
    AL = F(best["area_over_2pi"])
    ceil3 = math.ceil(AL * 1000) / 1000
    print(f"    L={L}: A_L/2pi = {float(AL):.6f} (<= {ceil3:.3f}; {best['kind']} pair, T={best['T']}); "
          f"Cor. N1 needs A/2pi >= {4 ** (L - 1) - 1}; Theorem 4.1 needs A/2pi >= {3 * (L - 1) ** 2 + 1}")
    assert AL < 4 ** (L - 1) - 1 and AL <= 3 * (L - 1) ** 2 + 1
print("ALL CHECKS PASSED")

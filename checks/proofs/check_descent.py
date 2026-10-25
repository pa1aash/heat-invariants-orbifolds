"""Theorem 5.11 and Appendix E: rank 0 and torsion Z/2 x Z/6 for E: y^2 = x(x+9)(x+384), the model of C_{27/2}.
Checks every step of the descent in exact arithmetic.

  (a) E: y^2 = x^3 + a x^2 + b x, a = 393, b = 3456 = 2^7 3^3; E': y^2 = x^3 - 2a x^2 + (a^2-4b) x,
      a^2 - 4b = 140625 = 3^2 5^6; both nonsingular; bad primes {2, 3, 5}
  (b) the birational map C_{27/2} -> E,
        x = -16 e2/Z^2,  y = 8 (-4 e2/Z^2 - 54)(X - Y)/Z,   e2 = XY + YZ + ZX,
      and its inverse, are inverse rational maps between C_{27/2} and E           [sympy, exact]
  (c) descent.  For b1 | b squarefree-class representative, C_{b1}: N^2 = b1 M^4 + a M^2 e^2 + (b/b1) e^4;
      for b1' | b', C'_{b1'}: N^2 = b1' M^4 - 2a M^2 e^2 + (b'/b1') e^4.
      Every class not in the claimed image has an explicit local obstruction (no primitive solution
      mod p^k, found by exhaustive search; or no real solution); every class in the image is the
      image of a torsion point.  So alpha(E) = {1, -1, 6, -6}, alpha'(E') = {1}, and
      2^r = 4 * 1 / 4 = 1.
  (d) torsion: #E(F_7) = #E(F_11) = 12 (brute force); the 12 explicit rational points form a group,
      orders {1:1, 2:3, 3:2, 6:6}, hence Z/2 x Z/6.
  (e) C_{27/2}: nonsingular; the 12 listed points lie on it and are distinct; the positive ones are
      the permutations of (1:4:4) and (1:1:4); (1:4:4) -> (-24, 360) of order 6.

Run from the repository root:
    python3 checks/proofs/check_descent.py
Writes checks/proofs/check_descent.txt.  Exits nonzero on any failure.
"""
import os
import sys
from fractions import Fraction as Fr
from itertools import permutations, product
from math import gcd, isqrt

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_descent.txt")
log = []
fails = 0


def check(cond, msg):
    global fails
    log.append(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


a, b = 393, 3456
ap, bp = -2 * a, a * a - 4 * b
# ---------------------------------------------------------------- (a)
check(b == 2 ** 7 * 3 ** 3 and bp == 140625 == 3 ** 2 * 5 ** 6, "(a) b = 2^7 3^3, a^2 - 4b = 140625 = 3^2 5^6")
check(sp.expand(sp.Symbol("x") * (sp.Symbol("x") + 9) * (sp.Symbol("x") + 384)) == sp.Symbol("x") ** 3 + a * sp.Symbol("x") ** 2 + b * sp.Symbol("x"),
      "(a) x(x+9)(x+384) = x^3 + 393 x^2 + 3456 x")
disc = 16 * b * b * (a * a - 4 * b)
discp = 16 * bp * bp * (ap * ap - 4 * bp)
check(disc != 0 and discp != 0 and set(sp.factorint(disc)) == {2, 3, 5} and set(sp.factorint(discp)) == {2, 3, 5},
      f"(a) Disc(E) = {sp.factorint(disc)}, Disc(E') = {sp.factorint(discp)}: bad primes {{2,3,5}}")

# ---------------------------------------------------------------- (b) birational map
X, Y, x, y = sp.symbols("X Y x y")
Lam = sp.Rational(27, 2)
Ceq = (X + Y + 1) * (X * Y + X + Y) - Lam * X * Y          # chart Z = 1
e2 = X * Y + X + Y
xf = -16 * e2
yf = 8 * (-4 * e2 - 54) * (X - Y)
Eeq = lambda u, v: v ** 2 - (u ** 3 + a * u ** 2 + b * u)
rem = sp.rem(sp.expand(sp.numer(sp.together(Eeq(xf, yf)))), sp.expand(Ceq), X)
check(sp.simplify(rem) == 0, "(b) (x, y)(X:Y:1) satisfies E on C_{27/2} (remainder mod the equation of C is 0)")
s_, eta = x / 4, y / 8
u = s_ * (Lam - 1) / (s_ - 4 * Lam)
dXY = eta / (s_ - 4 * Lam)
Xi, Yi = (u + dXY) / 2, (u - dXY) / 2
back = sp.together(Ceq.subs({X: Xi, Y: Yi}, simultaneous=True))
num = sp.expand(sp.numer(back))
check(sp.rem(num, sp.expand(Eeq(x, y)), y) == 0, "(b) the inverse map sends E into C_{27/2}")
comp_x = sp.simplify(xf.subs({X: Xi, Y: Yi}, simultaneous=True))
check(sp.simplify(sp.rem(sp.expand(sp.numer(sp.together(comp_x - x))), sp.expand(Eeq(x, y)), y)) == 0,
      "(b) x o inverse = x on E")
comp_y = sp.together(yf.subs({X: Xi, Y: Yi}, simultaneous=True) - y)
check(sp.simplify(sp.rem(sp.expand(sp.numer(comp_y)), sp.expand(Eeq(x, y)), y)) == 0, "(b) y o inverse = y on E")
pt = {X: sp.Rational(1, 4), Y: 1}
check((xf.subs(pt), yf.subs(pt)) == (-24, 360), "(b) (1:4:4) = (1/4 : 1 : 1) -> (-24, 360)")


# ---------------------------------------------------------------- (c) descent
def sqfree_class(n):
    s = -1 if n < 0 else 1
    out = 1
    for p, e in sp.factorint(abs(n)).items():
        if e % 2:
            out *= p
    return s * out


def classes(bb):
    ps = [p for p in sp.factorint(abs(bb))]
    out = set()
    for sgn in (1, -1):
        for mask in product((0, 1), repeat=len(ps)):
            v = sgn
            for p, m in zip(ps, mask):
                if m:
                    v *= p
            out.add(v)
    return sorted(out)


def real_solvable(c4, c2, c0):
    """N^2 = c4 M^4 + c2 M^2 e^2 + c0 e^4 has a real point with (M, e) != 0 iff the quadratic form
    c4 X^2 + c2 X W + c0 W^2 is >= 0 somewhere on X, W >= 0 not both 0."""
    if c4 > 0 or c0 > 0:
        return True
    # both <= 0: need c2 X W >= -c4 X^2 - c0 W^2 for some X, W > 0: iff c2 > 0 and c2^2 >= 4 c4 c0
    return c2 > 0 and c2 * c2 >= 4 * c4 * c0


def local_obstruction(c4, c2, c0, p, kmax):
    """Smallest k <= kmax such that N^2 = c4 M^4 + c2 M^2 e^2 + c0 e^4 has no solution mod p^k with
    (M, e) not both divisible by p; None if solutions exist mod p^kmax."""
    for k in range(1, kmax + 1):
        q = p ** k
        sq = {(n * n) % q for n in range(q)}
        found = False
        for M in range(q):
            for e in range(q):
                if M % p == 0 and e % p == 0:
                    continue
                if (c4 * M ** 4 + c2 * M * M * e * e + c0 * e ** 4) % q in sq:
                    found = True
                    break
            if found:
                break
        if not found:
            return k
    return None


KMAX = {2: 7, 3: 5, 5: 4}


def descent(aa, bb, name):
    image, obstructed = [], {}
    for d in classes(bb):
        c4, c2, c0 = d, aa, bb // d
        if not real_solvable(c4, c2, c0):
            obstructed[d] = "R"
            continue
        obs = None
        for p in (2, 3, 5):
            k = local_obstruction(c4, c2, c0, p, KMAX[p])
            if k is not None:
                obs = f"no primitive solution mod {p}^{k}"
                break
        if obs:
            obstructed[d] = obs
        else:
            image.append(d)
    log.append(f"     {name}: classes {classes(bb)}")
    for d, why in obstructed.items():
        log.append(f"     {name}: d = {d}: {why if why != 'R' else 'no real point (all coefficients of the quartic <= 0 and no positive value)'}")
    return image, obstructed


img, obs = descent(a, b, "alpha (E, b = 3456)")
check(sorted(img) == [-6, -1, 1, 6], f"(c) Selmer bound for alpha: {sorted(img)} = {{1, -1, 6, -6}}")
imgp, obsp = descent(ap, bp, "alpha' (E', b' = 140625)")
check(sorted(imgp) == [1], f"(c) Selmer bound for alpha': {sorted(imgp)} = {{1}}")
# the image of alpha contains the classes of the 2-torsion: alpha(0,0) = b, alpha(-9,0) = -9, alpha(-384,0) = -384
tors_img = {sqfree_class(b), sqfree_class(-9), sqfree_class(-384), 1}
check(tors_img == {1, -1, 6, -6}, f"(c) alpha of O, (0,0), (-9,0), (-384,0) = {sorted(tors_img)}: the bound is attained")
check(sqfree_class(bp) == 1, "(c) alpha'(0,0) = b' = 375^2 is a square: alpha'(E') = {1}")
check(len(img) * len(imgp) // 4 == 1, "(c) 2^r = #alpha(E) #alpha'(E') / 4 = 4*1/4 = 1, so r = 0")


# the obstructions printed in the paper, checked one by one
for d in (2, -2, 3, -3):
    vals = {(d * M ** 4 + a * M * M * e * e + (b // d) * e ** 4) % 5 for M in range(5) for e in range(5) if (M, e) != (0, 0)}
    check(vals <= {2, 3}, f"(c) d = {d}: d M^4 + 393 M^2 e^2 + (3456/d) e^4 mod 5 takes only the non-residues {sorted(vals)}")
for d in (-1, -3, -5, -15):
    check(d < 0 and ap < 0 and bp // d < 0, f"(c) d' = {d}: all three coefficients of N^2 = d M^4 - 786 M^2 e^2 + (140625/d) e^4 are negative")


def v3(n):
    k = 0
    while n % 3 == 0:
        n //= 3
        k += 1
    return k, n


for d in (3, 5, 15):
    ok = True
    for M in range(81):
        for e in range(81):
            if M % 3 == 0 and e % 3 == 0:
                continue
            f = d * M ** 4 + ap * M * M * e * e + (bp // d) * e ** 4
            k, u = v3(f)
            if not (k % 2 == 1 or u % 3 == 2):
                ok = False
    check(ok, f"(c) d' = {d}: for every 3-adically primitive (M, e) mod 81, v_3(f) is odd or f/3^v = 2 mod 3: no Q_3-point")
    check(local_obstruction(d, ap, bp // d, 3, 4) is not None, f"(c) d' = {d}: no primitive solution mod 3^{local_obstruction(d, ap, bp // d, 3, 4)}")

# ---------------------------------------------------------------- (d) torsion
def count_Fp(p):
    n = 1
    for xx in range(p):
        r = (xx ** 3 + a * xx ** 2 + b * xx) % p
        if r == 0:
            n += 1
        elif pow(r, (p - 1) // 2, p) == 1:
            n += 2
    return n


check(count_Fp(7) == 12 and count_Fp(11) == 12, f"(d) #E(F_7) = {count_Fp(7)}, #E(F_11) = {count_Fp(11)}")
check(disc % 7 != 0 and disc % 11 != 0, "(d) good reduction at 7 and 11")

INF = None


def eadd(P, Q):
    if P is INF:
        return Q
    if Q is INF:
        return P
    (x1, y1), (x2, y2) = P, Q
    if x1 == x2 and y1 == -y2:
        return INF
    if P == Q:
        lam = (3 * x1 * x1 + 2 * a * x1 + b) / (2 * y1)
    else:
        lam = (y2 - y1) / (x2 - x1)
    x3 = lam * lam - a - x1 - x2
    return (x3, -(y1 + lam * (x3 - x1)))


def eorder(P):
    Q, k = P, 1
    while Q is not INF:
        Q, k = eadd(Q, P), k + 1
        if k > 20:
            return None
    return k


pts = [INF, (Fr(0), Fr(0)), (Fr(-9), Fr(0)), (Fr(-384), Fr(0))]
for xx in range(-400, 400):
    r = xx ** 3 + a * xx * xx + b * xx
    if r > 0 and isqrt(r) ** 2 == r:
        pts += [(Fr(xx), Fr(isqrt(r))), (Fr(xx), Fr(-isqrt(r)))]
check(all(P is INF or P[1] ** 2 == P[0] ** 3 + a * P[0] ** 2 + b * P[0] for P in pts), "(d) all listed points lie on E")
G = set(pts)
check(len(G) == 12 and all(eadd(P, Q) in G for P in G for Q in G), f"(d) 12 integral points found with |x| < 400; closed under addition")
hist = {}
for P in G:
    o = 1 if P is INF else eorder(P)
    hist[o] = hist.get(o, 0) + 1
check(hist == {1: 1, 2: 3, 3: 2, 6: 6}, f"(d) order histogram {dict(sorted(hist.items()))} = that of Z/2 x Z/6")
expect = {(0, 0): 2, (-9, 0): 2, (-384, 0): 2, (16, 400): 3, (16, -400): 3, (-24, 360): 6, (-24, -360): 6,
          (-144, 2160): 6, (-144, -2160): 6, (216, 5400): 6, (216, -5400): 6}
for (xx, yy), o in expect.items():
    check(eorder((Fr(xx), Fr(yy))) == o, f"(d) ({xx}, {yy}) has order {o} (as printed in the paper)")
check(set(expect) | {INF} == {P if P is INF else (int(P[0]), int(P[1])) for P in G}, "(d) the printed list is all of the 12 points")
# psi sends the point at infinity of E to O = (1:-1:0): along y^2 ~ x^3, (25x+y : 25x-y : 4(x-216)) -> (1 : -1 : 0)
t = sp.symbols("t", positive=True)
xs, ys = 1 / t ** 2, sp.sqrt(1 / t ** 6 + a / t ** 4 + b / t ** 2)
lim = [sp.limit(c_ * t ** 3, t, 0) for c_ in (25 * xs + ys, 25 * xs - ys, 4 * (xs - 216))]
check(lim == [1, -1, 0], f"(b) psi(O_E) = (1:-1:0): limits {lim}")
log.append("     E(Q) = " + ", ".join("O" if P is INF else f"({P[0]}, {P[1]})" for P in sorted(G, key=lambda P: (-1e9, 0) if P is INF else (P[0], P[1]))))

# ---------------------------------------------------------------- (e) C_{27/2}
def Cf(P):
    p, q, r = P
    return 2 * (p + q + r) * (p * q + q * r + r * p) - 27 * p * q * r


Xs, Ys, Zs = sp.symbols("Xs Ys Zs")
Ch = 2 * (Xs + Ys + Zs) * (Xs * Ys + Ys * Zs + Zs * Xs) - 27 * Xs * Ys * Zs
sing = sp.solve([sp.diff(Ch, v) for v in (Xs, Ys, Zs)] + [Zs - 1], [Xs, Ys, Zs], dict=True)
sing += sp.solve([sp.diff(Ch, v) for v in (Xs, Ys, Zs)] + [Zs, Ys - 1], [Xs, Ys, Zs], dict=True)
sing += sp.solve([sp.diff(Ch, v) for v in (Xs, Ys, Zs)] + [Zs, Ys, Xs - 1], [Xs, Ys, Zs], dict=True)
check(len(sing) == 0, "(e) C_{27/2} is nonsingular (gradient vanishes nowhere on P^2(C))")
listed = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (0, 1, -1), (1, 0, -1)]
listed += [tuple(P) for P in {tuple(p) for p in permutations((1, 4, 4))}] + [tuple(P) for P in {tuple(p) for p in permutations((1, 1, 4))}]


def pnorm(P):
    g = 0
    for c in P:
        g = gcd(g, c)
    P = tuple(c // g for c in P)
    for c in P:
        if c:
            return P if c > 0 else tuple(-v for v in P)


check(len({pnorm(P) for P in listed}) == 12 and all(Cf(P) == 0 for P in listed),
      "(e) the 12 points (6 base points, permutations of (1:4:4) and (1:1:4)) lie on C_{27/2} and are distinct")
pos = sorted(pnorm(P) for P in listed if all(c > 0 for c in pnorm(P)))
check(pos == sorted({pnorm(p) for p in permutations((1, 4, 4))} | {pnorm(p) for p in permutations((1, 1, 4))}),
      f"(e) positive points: {pos}")
check(sum(1 for t in [(2, 8, 8), (3, 3, 12)] if Fr(1, t[0]) + Fr(1, t[1]) + Fr(1, t[2]) == Fr(3, 4) and sum(t) == 18) == 2,
      "(e) (2,8,8) and (3,3,12): sum 18, R = 3/4, Lambda = 27/2")

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{sum(1 for l in log if l[:4] in ('PASS', 'FAIL'))} checks, {fails} failures\n")
print("\n".join(log))
print(f"{fails} failures")
sys.exit(1 if fails else 0)

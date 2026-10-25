"""
Exact chord-and-tangent arithmetic on the cubics

    C_lambda :  b (x + y + z)(xy + yz + zx) - a xyz = 0,      lambda = a/b,

on which every degeneracy lives: a triple (p, q, r) lies on C_lambda exactly
when e1 e2 / e3 = lambda, and lambda = S R is the same for every triple of a
degeneracy class (equal S, equal R). Points are primitive integer vectors in
the projective plane; every operation is integer arithmetic.

The group law uses the base point O = (1 : -1 : 0), which lies on every
C_lambda. Torsion is decided with Mazur's theorem: a point of finite order in
E(Q) has order at most 12, so a point P with nP != O for 1 <= n <= 12 has
infinite order.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Iterable

Point = tuple[int, int, int]

TRIVIAL: tuple[Point, ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                              (1, -1, 0), (0, 1, -1), (1, 0, -1))
O: Point = (1, -1, 0)


def normalize(v: Iterable[int]) -> Point:
    x, y, z = v
    g = gcd(gcd(x, y), z)
    if g == 0:
        raise ZeroDivisionError("zero vector is not a projective point")
    x, y, z = x // g, y // g, z // g
    first = next(c for c in (x, y, z) if c != 0)
    if first < 0:
        x, y, z = -x, -y, -z
    return (x, y, z)


def lam(t: Iterable[int]) -> Fraction:
    """lambda = e1 e2 / e3 = S * R of a triple."""
    x, y, z = t
    return Fraction((x + y + z) * (x * y + y * z + z * x), x * y * z)


class Cubic:
    def __init__(self, lam_value: Fraction):
        lam_value = Fraction(lam_value)
        self.lam = lam_value
        self.a, self.b = lam_value.numerator, lam_value.denominator

    # F and its gradient --------------------------------------------------
    def F(self, P: Point) -> int:
        x, y, z = P
        return self.b * (x + y + z) * (x * y + y * z + z * x) - self.a * x * y * z

    def grad(self, P: Point) -> Point:
        x, y, z = P
        e1, e2 = x + y + z, x * y + y * z + z * x
        a, b = self.a, self.b
        return (b * (e2 + e1 * (y + z)) - a * y * z,
                b * (e2 + e1 * (x + z)) - a * x * z,
                b * (e2 + e1 * (x + y)) - a * x * y)

    def on(self, P: Point) -> bool:
        return self.F(P) == 0

    # third intersection --------------------------------------------------
    def chord(self, P: Point, Q: Point) -> Point:
        assert self.on(P) and self.on(Q), (P, Q, self.lam)
        P, Q = normalize(P), normalize(Q)
        if P == Q:
            g = self.grad(P)
            if g == (0, 0, 0):
                raise ValueError(f"singular point {P} on C_{self.lam}")
            # a second point d on the tangent line g . X = 0
            for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                d = (g[1] * e[2] - g[2] * e[1],
                     g[2] * e[0] - g[0] * e[2],
                     g[0] * e[1] - g[1] * e[0])
                if d != (0, 0, 0) and normalize(d) != P:
                    break
            gd = self.grad(d)
            C = sum(gd[i] * P[i] for i in range(3))
            D = self.F(d)
            R = tuple(D * P[i] - C * d[i] for i in range(3))
        else:
            gp, gq = self.grad(P), self.grad(Q)
            B = sum(gp[i] * Q[i] for i in range(3))
            C = sum(gq[i] * P[i] for i in range(3))
            R = tuple(C * P[i] - B * Q[i] for i in range(3))
        if R == (0, 0, 0):
            raise ValueError("line contained in the cubic")
        R = normalize(R)
        assert self.on(R)
        return R

    # group law with base point O ------------------------------------------
    def add(self, P: Point, Q: Point) -> Point:
        return self.chord(O, self.chord(P, Q))

    def neg(self, P: Point) -> Point:
        return self.chord(P, self.chord(O, O))

    def mul(self, n: int, P: Point) -> Point:
        if n < 0:
            return self.mul(-n, self.neg(P))
        result, base = O, normalize(P)
        while n:
            if n & 1:
                result = self.add(result, base)
            base = self.add(base, base)
            n >>= 1
        return result

    def order(self, P: Point, bound: int = 12) -> int | None:
        """Exact order if <= bound, else None (infinite order by Mazur when bound = 12)."""
        Q = normalize(P)
        for n in range(1, bound + 1):
            if Q == O:
                return n
            Q = self.add(Q, P)
        return None


def positive_triple(P: Point) -> tuple[int, int, int] | None:
    """The sorted positive triple of a projective point, or None if not all of one sign."""
    x, y, z = P
    if x > 0 and y > 0 and z > 0:
        return tuple(sorted((x, y, z)))
    if x < 0 and y < 0 and z < 0:
        return tuple(sorted((-x, -y, -z)))
    return None


def dual(t: Iterable[int]) -> tuple[int, int, int]:
    """The reciprocal triple (1/p, 1/q, 1/r), made primitive and sorted."""
    p, q, r = t
    return tuple(sorted(normalize((q * r, p * r, p * q))))


def common_sum_fibre(points: Iterable[tuple[int, int, int]]) -> tuple[int, list[tuple[int, int, int]]]:
    """Scale primitive positive triples on one C_lambda to their least common sum."""
    pts = [tuple(sorted(normalize(t))) for t in points]
    L = 1
    for t in pts:
        s = sum(t)
        L = L * s // gcd(L, s)
    return L, sorted(tuple(m * (L // sum(t)) for m in t) for t in pts)

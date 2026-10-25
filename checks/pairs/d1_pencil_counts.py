"""Recount the pencil witnesses of Remark S8.4.

    python3 checks/pairs/d1_pencil_counts.py [N ...]      defaults 130 220 (about 3 minutes)

Definition used (and printed in the paper). A *pencil splitting* of bound N is a pair
(A, B) of 4-multisets of nonzero integers, all entries of absolute value at most N, with
  e_1(A) = e_1(B) = 0,  e_3(A) = e_3(B),  e_4(A) = e_4(B),  A != B,
such that Z = A + (-B) (multiset union) has four positive and four negative entries and no pair
{z, -z}. Its *witness* is the pair {U, V}, U = Z_{>0}, V = -Z_{<0}: two disjoint 4-multisets of
positive integers with equal R, P_1, P_3 (so equal first three heat invariants as spheres with four
cone points). We count distinct witnesses {U, V} (unordered), all of them and the primitive ones
(gcd of the eight orders equal to 1), and check each witness exactly: equal R, P_1, P_3, different
P_5, every order >= 2.

Independent of theory/pte: the enumeration of A is written anew (numpy, exact int64 keys, then
exact Python integers for every candidate pair). Writes checks/pairs/output/d1_pencil_N<N>.csv
and prints the counts.
"""
import csv
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "output"


def sets_sum_zero(N):
    """All sorted (a <= b <= c <= d), nonzero, |.| <= N, a + b + c + d = 0, as an int64 array."""
    chunks = []
    for a in range(-N, 0):                     # a < 0 since the sum is 0 and the entries are nonzero
        for b in range(a, N + 1):
            if b == 0:
                continue
            c = np.arange(b, N + 1, dtype=np.int64)
            c = c[c != 0]
            d = -(a + b + c)
            ok = (d >= c) & (np.abs(d) <= N) & (d != 0)
            if ok.any():
                k = int(ok.sum())
                chunks.append(np.column_stack([np.full(k, a), np.full(k, b), c[ok], d[ok]]))
    return np.concatenate(chunks)


def esym(x):
    a, b, c, d = map(int, x)
    return (a * b * c + a * b * d + a * c * d + b * c * d, a * b * c * d)


def witnesses(N):
    S = sets_sum_zero(N)
    a, b, c, d = (S[:, i] for i in range(4))
    e3 = a * b * c + a * b * d + a * c * d + b * c * d
    e4 = a * b * c * d
    order = np.lexsort((e4, e3))
    e3, e4, S = e3[order], e4[order], S[order]
    same = (e3[1:] == e3[:-1]) & (e4[1:] == e4[:-1])
    starts = np.flatnonzero(np.r_[True, ~same])
    ends = np.r_[starts[1:], len(S)]
    found, nsplit = set(), 0
    for s0, s1 in zip(starts, ends):
        if s1 - s0 < 2:
            continue
        grp = [tuple(map(int, S[i])) for i in range(s0, s1)]
        for i in range(len(grp)):
            for j in range(len(grp)):
                if i == j:
                    continue
                A, B = grp[i], grp[j]
                assert esym(A) == esym(B)
                Z = list(A) + [-x for x in B]
                if any(-z in Z for z in Z):
                    continue
                U = tuple(sorted(z for z in Z if z > 0))
                V = tuple(sorted(-z for z in Z if z < 0))
                if len(U) != 4 or len(V) != 4:
                    continue
                nsplit += 1
                found.add(tuple(sorted((U, V))))
    return len(S), nsplit, sorted(found, key=lambda w: (max(w[0] + w[1]), w))


def check(w):
    U, V = w
    P = lambda X, k: sum(Fraction(x) ** k for x in X)
    assert min(U + V) >= 2 and not set(U) & set(V)
    assert P(U, -1) == P(V, -1) and P(U, 1) == P(V, 1) and P(U, 3) == P(V, 3), w
    assert P(U, 5) != P(V, 5), w


def main():
    bounds = [int(x) for x in sys.argv[1:]] or [130, 220]
    OUT.mkdir(exist_ok=True)
    for N in bounds:
        nsets, nsplit, ws = witnesses(N)
        for w in ws:
            check(w)
        prim = [w for w in ws if math.gcd(*(w[0] + w[1])) == 1]
        with open(OUT / f"d1_pencil_N{N}.csv", "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["U", "V", "primitive", "R", "P1", "P3"])
            for U, V in ws:
                wr.writerow([" ".join(map(str, U)), " ".join(map(str, V)), int(math.gcd(*(U + V)) == 1),
                             str(sum(Fraction(1, x) for x in U)), sum(U), sum(x ** 3 for x in U)])
        print(f"N={N}: {nsets} sets A with e_1 = 0 and all |entries| <= N; {nsplit} ordered pencil splittings; "
              f"{len(ws)} distinct witnesses, {len(prim)} primitive; smallest primitive: "
              + "; ".join(f"{p[0]}/{p[1]}" for p in prim[:3]))
    print("d1_pencil_counts.py: every witness checked exactly (equal R, P1, P3; different P5)")


if __name__ == "__main__":
    main()

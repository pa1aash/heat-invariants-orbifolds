"""T4: sharpness witnesses. K_mult >= n needs two distinct n-cone hyperbolic
multisets sharing the first n-1 invariants (R, S1, P3, ..., P_{2n-5}).

Part A re-verifies the known witnesses exactly:
    n = 3: {2,8,8} vs {3,3,12}          share (R, S1)
    n = 4: {3,10,15,30} vs {4,5,21,28}   share (R, S1, P3)
and checks the pair criterion of the paper (section 4): with p(z) = prod(z+m_i),
    Phi(z) := p(z) p'(-z) - p'(z) p(-z) = kappa z^3,  kappa != 0.

Part B is an exhaustive n = 5 search: every multiset 2 <= m_1 <= ... <= m_5 <= N
(all are hyperbolic, since sum(1 - 1/m_i) >= 5/2 > 2), grouped by S1 = sum m_i
(a shared invariant), keyed on the exact integer tuple (P3, P5, L*R) with
L = lcm(2..N). The key is exact, not a hash digest. As a control the same pass
records (R, S1, P3) collisions, i.e. the n = 5 failures of three coefficients.

Part C (cross-n): 3-cone vs 4-cone orbifolds with orders <= min(N, 60) sharing
the first three heat coefficients (a 3-cone orbifold is a 4-multiset padded
with the invisible order 1).

usage: sharpness_search.py [N]   (default N = 60; writes sharpness_n5_N<N>.json)
Exits nonzero on any failed assertion.
"""
import itertools
import json
import math
import sys
import time
from fractions import Fraction

import sympy as sp


def invariants(ms, count):
    """First `count` invariants (R, S1, P3, P5, ...) as exact Fractions."""
    out = [sum(Fraction(1, m) for m in ms), Fraction(sum(ms))]
    k = 3
    while len(out) < count:
        out.append(Fraction(sum(m**k for m in ms)))
        k += 2
    return tuple(out[:count])


def hyperbolic(ms):
    return sum(1 - Fraction(1, m) for m in ms) > 2


def phi(ms, ms2):
    z = sp.Symbol("z")
    p = sp.Mul(*[z + m for m in ms])
    q = sp.Mul(*[z + m for m in ms2])
    return sp.Poly(sp.expand(p * q.subs(z, -z) - q * p.subs(z, -z)), z)


def verify_known():
    z = sp.Symbol("z")
    for ms, ms2 in [((2, 8, 8), (3, 3, 12)), ((3, 10, 15, 30), (4, 5, 21, 28))]:
        n = len(ms)
        assert sorted(ms) != sorted(ms2)
        assert hyperbolic(ms) and hyperbolic(ms2)
        a, b = invariants(ms, n), invariants(ms2, n)
        assert a[: n - 1] == b[: n - 1], f"{ms},{ms2} do not share the first n-1 invariants"
        assert a[n - 1] != b[n - 1], f"{ms},{ms2} are not separated by invariant n"
        F = phi(ms, ms2)
        kappa = F.coeff_monomial(z**3)
        assert F == sp.Poly(kappa * z**3, z) and kappa != 0, f"Phi != kappa z^3 for {ms},{ms2}"
        print(
            f"n={n}: {ms} vs {ms2}: share {['R', 'S1', 'P3'][: n - 1]} = "
            f"{[str(x) for x in a[: n - 1]]}; invariant {n}: {a[n - 1]} != {b[n - 1]}; "
            f"Phi = {kappa} z^3"
        )


def search_n5(N):
    L = math.lcm(*range(2, N + 1))
    inv = [0, 0] + [L // m for m in range(2, N + 1)]
    c3 = [m**3 for m in range(N + 1)]
    c5 = [m**5 for m in range(N + 1)]
    scanned = 0
    full = []  # (R, S1, P3, P5) collisions
    three = []  # (R, S1, P3) collisions
    for s in range(10, 5 * N + 1):
        seen4, seen3 = {}, {}
        for a in range(2, s // 5 + 1):
            ra = s - a
            for b in range(a, ra // 4 + 1):
                rb = ra - b
                for c in range(b, rb // 3 + 1):
                    rc = rb - c
                    for d in range(max(c, rc - N), rc // 2 + 1):
                        e = rc - d
                        scanned += 1
                        k3 = (c3[a] + c3[b] + c3[c] + c3[d] + c3[e],
                              inv[a] + inv[b] + inv[c] + inv[d] + inv[e])
                        k4 = k3 + (c5[a] + c5[b] + c5[c] + c5[d] + c5[e],)
                        ms = (a, b, c, d, e)
                        if k3 in seen3:
                            three.append((seen3[k3], ms))
                        else:
                            seen3[k3] = ms
                        if k4 in seen4:
                            full.append((seen4[k4], ms))
                        else:
                            seen4[k4] = ms
    return scanned, full, three


def cross_n_search(N):
    """3-cone vs 4-cone orbifolds sharing the first three heat coefficients.

    A cone of order 1 contributes nothing to any heat coefficient (b_l(1) = 0,
    and the area 2 pi (n - 2 - R) is unchanged), so a 3-cone orbifold is the
    4-multiset {1} + (its orders). Agreement of the first three coefficients
    is then agreement of (R, S1, P3) of the padded 4-multisets.
    """
    seen, hits, scanned = {}, [], 0
    for ms in itertools.combinations_with_replacement(range(1, N + 1), 4):
        if ms[1] == 1:
            continue  # at most one trivial cone
        R = sum(Fraction(1, m) for m in ms)
        if ms[0] == 1 and not R - 1 < 1:
            continue  # the 3-cone orbifold must be hyperbolic
        if ms[0] > 1 and not R < 2:
            continue  # the 4-cone orbifold must be hyperbolic
        scanned += 1
        key = (sum(ms), R, sum(m**3 for m in ms))
        if key in seen:
            other = seen[key]
            if (other[0] == 1) != (ms[0] == 1):
                hits.append((other, ms))
        else:
            seen[key] = ms
    return scanned, hits


def classify(pairs):
    """Counts of all / primitive (gcd 1) / primitive-and-disjoint pairs.

    Padding a collision with a common order, or scaling it, yields another
    collision; primitive disjoint pairs are the genuinely new ones.
    """
    prim = [pr for pr in pairs if math.gcd(*pr[0], *pr[1]) == 1]
    disj = [pr for pr in prim if not set(pr[0]) & set(pr[1])]
    return {"all": len(pairs), "primitive": len(prim), "primitive_disjoint": len(disj)}, disj


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    verify_known()

    t = time.time()
    scanned, full, three = search_n5(N)
    wall = time.time() - t
    expected = math.comb(N - 2 + 5, 5)  # multisets of size 5 from {2..N}
    assert scanned == expected, f"scanned {scanned} != C({N + 3},5) = {expected}"
    # every reported collision is re-verified with Fractions
    for x, y in full:
        assert x != y and invariants(x, 4) == invariants(y, 4)
    for x, y in three:
        assert x != y and invariants(x, 3) == invariants(y, 3)
    cls4, _ = classify(full)
    cls3, disj3 = classify(three)
    out = {
        "n": 5,
        "max_order": N,
        "multisets_scanned": scanned,
        "wall_seconds": round(wall, 1),
        "counts_R_S1_P3_P5": cls4,
        "counts_R_S1_P3": cls3,
        "collisions_R_S1_P3_P5": [list(map(list, pr)) for pr in full],
        "collisions_R_S1_P3": [list(map(list, pr)) for pr in three],
    }
    Nc = min(N, 60)
    cross_scanned, cross_hits = cross_n_search(Nc)
    out["cross_n_3_vs_4"] = {"max_order": Nc, "scanned": cross_scanned,
                             "collisions": [list(map(list, pr)) for pr in cross_hits]}
    print(f"3-cone vs 4-cone, orders <= {Nc}: {cross_scanned} padded multisets, "
          f"{len(cross_hits)} sharing the first three coefficients")
    path = f"sharpness_n5_N{N}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(
        f"n=5, orders 2..{N}: scanned {scanned} multisets in {wall:.1f}s; "
        f"(R,S1,P3,P5) collisions: {cls4}; (R,S1,P3) collisions: {cls3}; "
        f"data -> {path}"
    )
    if full:
        print("first (R,S1,P3,P5) collision:", full[0])
    if disj3:
        print("smallest-S1 primitive disjoint (R,S1,P3) collision:",
              min(disj3, key=lambda pr: sum(pr[0])))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

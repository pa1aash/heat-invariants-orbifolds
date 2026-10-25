"""T2: can orbifolds of different genus share their first L heat coefficients? (exact)

Reduction (the paper section 3): (g; m) and (g'; m') share the first L coefficients iff,
with U = m + 1^a, V = m' + 1^b (a, b >= 0, a - b = R(m') - R(m)),
    R(U) = R(V),  P_j(U) = P_j(V) for odd j <= 2L-3,  and |V| - |U| = 2(g - g').
Theorem S: after cancelling common elements, |U*| + |V*| <= 2L forces U* = V* = {} and
hence equal signatures. So a collision sharing L coefficients needs |U*| + |V*| >= 2L+2.

Parts:
 (A) Thue-Morse construction: for every L >= 2 an explicit pair of hyperbolic orbifolds of
     genera 1 and 0 sharing the first L coefficients (L = 2..LMAX built and verified with the
     actual cone coefficients b_l). The construction is proved for all L in the paper.
 (B) Minimal genus-changing collisions: exhaustive search over disjoint U, V with
     |U| + |V| = 2L + 2 (the least size Theorem S allows), entries <= M.
 (C) Direct search in signature space: all hyperbolic (g; m) with g <= 2, n <= NMAX,
     orders <= MMAX, grouped by the first k coefficients; genus-changing collisions counted.
 (D) Brute-force test of Theorem S on every equal-area pair of a signature pool.
Exits nonzero on any failure.
"""
import itertools
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from math import lcm

from sig_common import (odd_balanced, realise, s_of, shared, shared_int, strip, thue_morse)

failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
        print("FAIL:", msg)
    assert cond, msg


def R(ms):
    return sum(Fraction(1, x) for x in ms)


def ms_minus(a, b):
    """multiset difference a - b and b - a (cancel common elements)."""
    ca, cb = Counter(a), Counter(b)
    return tuple(sorted((ca - cb).elements())), tuple(sorted((cb - ca).elements()))


def UV_of(sig1, sig2):
    """Minimal padded pair (U, V) of a pair of signatures with equal area."""
    (g1, m1), (g2, m2) = sig1, sig2
    d = R(m2) - R(m1)          # = a - b, an integer when areas agree
    check(d.denominator == 1, "a - b integral")
    d = int(d)
    U = tuple(sorted(m1 + (1,) * max(d, 0)))
    V = tuple(sorted(m2 + (1,) * max(-d, 0)))
    return U, V


# ---------------------------------------------------------------- (A) construction
def genus_pair(L):
    """Thue-Morse pair (U, V), |V| = |U| + 2, odd-balanced to order 2L-3, no entry 1."""
    K = 2 * L - 2
    T0 = [i for i in range(2 ** K) if thue_morse(i) == 0]
    T1 = [i for i in range(2 ** K) if thue_morse(i) == 1]
    U0 = [2 * i - 1 for i in T0 if i != 0]                # A minus the point -1
    V0 = [2 * i - 1 for i in T1] + [1]                    # B plus the point +1
    Ap = [2 * i + 1 for i in T0]
    Bp = [2 * i + 1 for i in T1]
    r0 = R(U0) - R(V0)
    rho = R(Ap) - R(Bp)
    check(rho > 0, "rho = int_0^1 x^0 prod(1 - x^(2^k)) ... > 0 (shift-1/2 form)")
    if r0 == 0:
        U, V = [2 * u for u in U0], [2 * v for v in V0]
    else:
        if r0 < 0:
            Ap, Bp, rho = Bp, Ap, -rho
        ratio = rho / r0            # = p/q > 0
        p, q = ratio.numerator, ratio.denominator
        if min(p, q) == 1:
            p, q = 2 * p, 2 * q
        U = [q * u for u in U0] + [p * b for b in Bp]
        V = [q * v for v in V0] + [p * a for a in Ap]
    return tuple(sorted(U)), tuple(sorted(V)), r0, rho


LMAX_A = 6
print("(A) Thue-Morse genus-changing pairs (genus 1 vs genus 0)")
for L in range(2, LMAX_A + 1):
    U, V, r0, rho = genus_pair(L)
    check(odd_balanced(U, V, L), f"odd-balanced L={L}")
    check(min(U + V) >= 2, "no padding entries")
    check(len(V) - len(U) == 2, "size difference 2")
    a, b = realise(U, V)
    check(a[0] == 1 and b[0] == 0, f"genera (1, 0) at L={L}")
    k = shared(a, b, L + 2)     # actual cone coefficients b_l, not the reduction
    check(k >= L, f"shares first {L} coefficients (got {k})")
    s = s_of(*a)
    check(s < len(V) - 2, "area < 2 pi (|V| - 2)")
    check(len(U) + len(V) == 2 ** (2 * L - 1), "total size 2^(2L-1)")
    print(f"  L={L}: (1; {len(U)} cones) vs (0; {len(V)} cones), max order {max(U + V)}, "
          f"Area/2pi < {len(V) - 2} = 4^{L - 1} - 1; "
          f"shares exactly {k} coefficients")
    if L == 2:
        print(f"        U = {U}\n        V = {V}")

# ---------------------------------------------------------------- (B) minimal collisions
def minimal_search(L, M, shapes):
    D = lcm(*range(1, M + 1))       # R scaled by D is an integer: exact integer keys
    js = range(1, 2 * L - 2, 2)

    def keyed(ms):
        return (sum(D // x for x in ms),) + tuple(sum(x ** j for x in ms) for j in js)

    found = []
    for (u, v) in shapes:
        table = defaultdict(list)
        for Ut in itertools.combinations_with_replacement(range(1, M + 1), u):
            table[keyed(Ut)].append(Ut)
        for Vt in itertools.combinations_with_replacement(range(1, M + 1), v):
            kv = keyed(Vt)
            if kv in table:
                for Ut in table[kv]:
                    if not set(Ut) & set(Vt):
                        found.append((Ut, Vt))
    return found


print("(B) genus-changing collisions of size |U*|+|V*| = 2L+2 (the least Theorem S allows) "
      "and 2L+4")
SEARCH_B = [(2, 40, [(2, 4), (1, 5)]),
            (3, 30, [(3, 5), (2, 6), (1, 7)]),
            (4, 22, [(4, 6), (3, 7), (2, 8)]),
            (3, 30, [(4, 6)]),
            (3, 24, [(3, 7)]),
            (4, 16, [(5, 7)])]
minimal_record = {}
for L, M, shapes in SEARCH_B:
    found = minimal_search(L, M, shapes)
    for Ut, Vt in found:
        check(odd_balanced(Ut, Vt, L), "found pair balanced")
        a, b = realise(Ut, Vt)
        check(shared(a, b, L + 2) >= L, "found pair shares L")
        check(a[0] != b[0], "genera differ")
    found.sort(key=lambda p: (max(p[0] + p[1]), p))
    minimal_record[(L, tuple(shapes))] = found
    shapes_s = ", ".join(f"({u},{v})" for u, v in shapes)
    print(f"  L={L}: shapes (|U|,|V|) in {{{shapes_s}}}, entries <= {M}: {len(found)} pairs")
    for Ut, Vt in found[:3]:
        a, b = realise(Ut, Vt)
        print(f"        U={Ut} V={Vt}  ->  {a} vs {b}, Area/2pi = {s_of(*a)}, "
              f"shares {shared(a, b, L + 3)}")
check(len(minimal_record[(2, ((2, 4), (1, 5)))]) > 0, "L=2 collisions of least size exist")
check(len(minimal_record[(3, ((4, 6),))]) > 0, "L=3 collisions of size 10 exist")

# ---------------------------------------------------------------- (C) signature space
NMAX, MMAX = 5, 16
print(f"(C) signature space: g <= 2, n <= {NMAX}, orders <= {MMAX}, hyperbolic")
pool = []
for g in range(0, 3):
    for n in range(0, NMAX + 1):
        for m in itertools.combinations_with_replacement(range(2, MMAX + 1), n):
            if s_of(g, m) > 0:
                pool.append((g, m))
groups = defaultdict(list)
for sig in pool:
    groups[s_of(*sig)].append(sig)
best = Counter()
examples = {}
for s, cls in groups.items():
    for a, b in itertools.combinations(cls, 2):
        if a[0] == b[0]:
            continue
        k = shared_int(a, b, 12)
        best[k] += 1
        if k >= 2 and (k not in examples or max(a[1] + b[1]) < max(examples[k][0][1] + examples[k][1][1])):
            examples[k] = (a, b)
print(f"  {len(pool)} signatures; genus-changing equal-area pairs by number of shared "
      f"coefficients: {dict(sorted(best.items()))}")
for k in sorted(examples):
    a, b = examples[k]
    check(shared(a, b, 12) == k, "example recomputed with b_l")
    print(f"    shares {k}: {a} vs {b} (Area/2pi = {s_of(*a)})")

# ---------------------------------------------------------------- (D) Theorem S, brute force
print("(D) Theorem S on every equal-area pair of the pool of (C)")
tight = Counter()
npairs = 0
for s, cls in groups.items():
    for a, b in itertools.combinations(cls, 2):
        k = shared_int(a, b, 40)
        U, V = UV_of(a, b)
        check(len(V) - len(U) == 2 * (a[0] - b[0]), "|V|-|U| = 2(g-g')")
        Us, Vs = ms_minus(U, V)
        T = len(Us) + len(Vs)
        check(T % 2 == 0 and T > 0, "T* even and positive for distinct signatures")
        check(k <= T // 2 - 1, f"Theorem S: shared {k} <= T*/2 - 1 for {a} {b}")
        (g1, m1), (g2, m2) = a, b
        check(k < max(len(m1) + g1 - g2, len(m2) + g2 - g1), "pairwise corollary")
        check(k < 2 * s + 4, "area corollary")
        tight[(k == T // 2 - 1)] += 1
        npairs += 1
print(f"  {npairs} pairs: shared <= T*/2 - 1 always; attained with equality in "
      f"{tight[True]} pairs")

if failures:
    print(f"{failures} FAILURES")
    sys.exit(1)
print("genus.py: all assertions passed")

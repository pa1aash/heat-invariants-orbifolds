"""T1: the cone count of a genus-0 hyperbolic orbifold from its heat coefficients (exact).

(a) n <= Area/pi + 4 for every hyperbolic genus-0 orbifold, with equality iff all orders are 2
    (checked on a pool; the one-line proof is in the paper).
(b) Theorem T1 by brute force on the pool: two genus-0 orbifolds sharing their first
    floor(Area/pi) + 4 coefficients have equal cone-order multisets; sharper, sharing
    max(n, n') coefficients already forces it.
(c) Exhaustive searches for genus-0 pairs with DIFFERENT cone counts sharing the first k
    coefficients, k = 1..4. In the padded form: N-multisets U (containing padding 1s) and V
    (no 1s), disjoint, with R(U) = R(V) and P_j(U) = P_j(V) for odd j <= 2k-3; then
    m = U minus its 1s, m' = V. Theorem A forces N >= k + 1.
(d) Construction for every k (the paper, Proposition 4): genus-0 pairs with n' = n + 1 sharing
    the first k coefficients, built and verified for k = 2..KMAX_D.
All pairs reported are re-verified with the actual cone coefficients b_l (sig_common.shared).
Exits nonzero on any failure.
"""
import itertools
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from math import lcm

from sig_common import odd_balanced, s_of, shared, shared_int, strip, thue_morse

failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
        print("FAIL:", msg)
    assert cond, msg


def R(ms):
    return sum(Fraction(1, x) for x in ms)


# ------------------------------------------------------------------- (a), (b)
MMAX, NMAX = 14, 6
pool = [(0, m) for n in range(3, NMAX + 1)
        for m in itertools.combinations_with_replacement(range(2, MMAX + 1), n)
        if s_of(0, m) > 0]
print(f"(a) n <= Area/pi + 4 on {len(pool)} hyperbolic genus-0 signatures "
      f"(n <= {NMAX}, orders <= {MMAX})")
for g, m in pool:
    s = s_of(g, m)                      # Area/pi = 2s
    check(len(m) <= 2 * s + 4, "n <= Area/pi + 4")
    check((len(m) == 2 * s + 4) == all(x == 2 for x in m), "equality iff all orders 2")
print("    holds; equality exactly for (0; 2,...,2)")

print("(b) Theorem T1 by brute force on the same pool")
by_s = defaultdict(list)
for sig in pool:
    by_s[s_of(*sig)].append(sig)
hist = Counter()
sharp = Counter()
for s, cls in by_s.items():
    bound = int(2 * s) + 4                          # floor(Area/pi) + 4
    for a, b in itertools.combinations(cls, 2):
        k = shared_int(a, b, bound + 2)
        check(k < bound, f"T1: {a} {b} share {k} >= floor(A/pi)+4 = {bound}")
        check(k < max(len(a[1]), len(b[1])), f"pairwise form: {a} {b}")
        hist[(len(a[1]) != len(b[1]), k)] += 1
        sharp[k == max(len(a[1]), len(b[1])) - 1] += 1
print(f"    {sum(hist.values())} equal-area pairs: no pair shares floor(A/pi)+4, nor max(n,n'), "
      f"coefficients")
print(f"    (different n?, shared) histogram: {dict(sorted(hist.items()))}")
print(f"    pairs with shared = max(n,n') - 1 (pairwise bound attained): {sharp[True]}")


# ------------------------------------------------------------------- (c)
def search_diff_n(k, N, M, ones_range):
    """Disjoint N-multisets U (with u ones, u in ones_range) and V (entries >= 2), entries
    <= M, odd-balanced to order 2k-3, realised as genus-0 hyperbolic orbifolds."""
    D = lcm(*range(1, M + 1))
    js = list(range(1, 2 * k - 2, 2))

    def key(ms):
        return (sum(D // x for x in ms),) + tuple(sum(x ** j for x in ms) for j in js)

    table = defaultdict(list)
    for u in ones_range:
        for rest in itertools.combinations_with_replacement(range(2, M + 1), N - u):
            Ut = (1,) * u + rest
            table[key(Ut)].append(Ut)
    found = []
    for Vt in itertools.combinations_with_replacement(range(2, M + 1), N):
        kv = key(Vt)
        if kv in table:
            for Ut in table[kv]:
                if set(Ut) & set(Vt):
                    continue
                a, b = (0, strip(Ut)), (0, Vt)
                if s_of(*a) <= 0:
                    continue
                found.append((a, b))
    return found


print("(c) genus-0 pairs with different cone counts sharing the first k coefficients")
SEARCH_C = [  # (k, N, M, numbers of padding ones in U)
    (1, 4, 12, (1,)),
    (2, 4, 30, (1,)),
    (2, 5, 16, (1, 2)),
    (3, 4, 80, (1,)),
    (3, 5, 40, (1, 2)),
    (3, 6, 20, (1, 2, 3)),
    (4, 5, 40, (1, 2)),
    (4, 6, 22, (1, 2, 3)),
]
record_c = {}
for k, N, M, ones in SEARCH_C:
    found = search_diff_n(k, N, M, ones)
    for a, b in found:
        check(len(a[1]) != len(b[1]), "cone counts differ")
        check(shared(a, b, k + 3) >= k, f"re-verified with b_l: {a} {b}")
    found.sort(key=lambda p: (max(p[0][1] + p[1][1]), p))
    record_c[(k, N, M)] = found
    exact = Counter(shared(a, b, k + 3) for a, b in found)
    print(f"  k={k}, padded length N={N}, orders <= {M}, padding ones in {ones}: "
          f"{len(found)} pairs; exact shared counts {dict(sorted(exact.items()))}")
    for a, b in found[:2]:
        print(f"      {a} vs {b}: Area/2pi = {s_of(*a)}, shares {shared(a, b, k + 3)}")
check(len(record_c[(2, 4, 30)]) > 0, "k=2 different-n pairs exist")


# ------------------------------------------------------------------- (d)
def diff_n_pair(k):
    """Genus-0 pair with n' = n + 1 sharing the first k coefficients (the paper Prop. 4)."""
    K = 2 * k - 2
    A = [i + 1 for i in range(2 ** K) if thue_morse(i) == 0]       # contains 1
    B = [i + 1 for i in range(2 ** K) if thue_morse(i) == 1]       # no 1
    Ap = [2 * i + 1 for i in range(2 ** K) if thue_morse(i) == 0]
    Bp = [2 * i + 1 for i in range(2 ** K) if thue_morse(i) == 1]
    r0 = R(A) - R(B)
    rho = R(Ap) - R(Bp)
    check(r0 > 0 and rho > 0, "r0, rho > 0 (integral representation)")
    # U = A + sum_i p_i B', V = B + sum_i p_i A' with r0/rho = sum_i 1/p_i (distinct p_i >= 2,
    # greedy Egyptian fraction): R(U) - R(V) = r0 - rho * sum_i 1/p_i = 0.
    x, p, ps = r0 / rho, 2, []
    while x > 0:
        p = max(p, -(-x.denominator // x.numerator))      # ceil(1/x)
        ps.append(p)
        x -= Fraction(1, p)
        p += 1
    check(sum(Fraction(1, q) for q in ps) == r0 / rho, "Egyptian fraction")
    U = A + [q * y for q in ps for y in Bp]
    V = B + [q * y for q in ps for y in Ap]
    p, t = ps, len(ps)
    return tuple(sorted(U)), tuple(sorted(V)), p, t


KMAX_D = 3   # k >= 4: the paper Theorem N(b); greedy denominators grow doubly exponentially
print("(d) construction: genus-0 pairs with n' = n + 1 sharing the first k coefficients")
for k in range(2, KMAX_D + 1):
    U, V, p, t = diff_n_pair(k)
    check(odd_balanced(U, V, k), f"balanced k={k}")
    check(U.count(1) == 1 and 1 not in V and len(U) == len(V), "one padding 1, sizes equal")
    a, b = (0, strip(U)), (0, V)
    check(s_of(*a) > 0, "hyperbolic")
    kk = shared(a, b, k + 2)
    check(kk >= k, f"shares first {k}")
    check(len(b[1]) == len(a[1]) + 1, "n' = n + 1")
    print(f"  k={k}: (0; {len(a[1])} cones) vs (0; {len(b[1])} cones), {t} scaled pieces, "
          f"shares exactly {kk}")
    if k == 2:
        print(f"      U = {U}\n      V = {V}")

if failures:
    print(f"{failures} FAILURES")
    sys.exit(1)
print("cone_count.py: all assertions passed")

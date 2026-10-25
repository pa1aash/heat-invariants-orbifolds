"""Exact K values on complete area classes (T1 tightness, T2(c) per-area).

For a rational s > 0 the class Sig(s) of signatures (g; m) of closed orientable hyperbolic
2-orbifolds with -chi = s (Area = 2 pi s) is finite: 2g - 2 + sum(1 - 1/m_i) = s with every term
1 - 1/m_i in [1/2, 1). This script enumerates Sig(s) completely and computes, exactly,
  K_mult(O) = least k such that the first k coefficients separate O from every O' in Sig(s)
             with a different signature (= K_mult(O; all closed orientable hyperbolic
             2-orbifolds), since the first coefficient is the area),
  K_n(O)    = least k separating O from every genus-0 O' with a different cone count
             (for genus-0 O),
  K_g(O)    = least k separating O from every O' of a different genus,
and compares max_O K_mult with the bound floor(Area/pi) + 4 = floor(2s) + 4 (Theorem S).
Every pair is also checked against Theorem S in its sharp form (shared <= T*/2 - 1).

The s examined: every value of -chi in (0, SMAX] attained by a signature with
g <= 2, n <= 4, orders <= ORD. Exits nonzero on any failure.
"""
import itertools
import sys
from collections import Counter, defaultdict
from fractions import Fraction

from sig_common import int_key, s_of, shared, shared_int

SMAX = Fraction(7, 5)
ORD = 12

failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
        print("FAIL:", msg)
    assert cond, msg


def multisets_with_sum(sigma, lo=2):
    """All sorted tuples (m_1 <= ... <= m_n), m_i >= lo, with sum(1 - 1/m_i) = sigma."""
    out = []
    if sigma == 0:
        return [()]
    nmin = int(sigma) + 1                      # each term < 1
    nmax = int(2 * sigma)                      # each term >= 1/2
    for n in range(nmin, nmax + 1):
        _rec(sigma, n, lo, (), out)
    return out


def _rec(sig, r, lo, pre, out):
    if r == 1:
        rest = 1 - sig                         # = 1/m
        if rest > 0 and rest.numerator == 1 and rest.denominator >= lo:
            out.append(pre + (rest.denominator,))
        return
    # smallest remaining entry a >= lo; all r remaining terms are >= 1 - 1/a, so
    # r (1 - 1/a) <= sig, i.e. a <= r / (r - sig); and sig < r.
    if not (Fraction(r, 2) <= sig < r):
        return
    amax = Fraction(r) / (r - sig)
    a = lo
    while a <= amax:
        _rec(sig - (1 - Fraction(1, a)), r - 1, a, pre + (a,), out)
        a += 1


def area_class(s):
    cls = []
    g = 0
    while 2 * g - 2 <= s:
        sigma = s + 2 - 2 * g
        for m in multisets_with_sum(sigma):
            cls.append((g, m))
        g += 1
    for sig in cls:
        check(s_of(*sig) == s, "class member has area s")
    return cls


def T_star(a, b):
    (g1, m1), (g2, m2) = a, b
    d = sum(Fraction(1, x) for x in m2) - sum(Fraction(1, x) for x in m1)
    d = int(d)
    U = Counter(m1 + (1,) * max(d, 0))
    V = Counter(m2 + (1,) * max(-d, 0))
    return sum((U - V).values()) + sum((V - U).values())


svals = set()
for g in range(0, 3):
    for n in range(0, 5):
        for m in itertools.combinations_with_replacement(range(2, ORD + 1), n):
            s = s_of(g, m)
            if 0 < s <= SMAX:
                svals.add(s)
svals = sorted(svals)
print(f"{len(svals)} area values s = -chi in (0, {SMAX}] from g<=2, n<=4, orders<={ORD}")

rows = []
total_members = 0
total_pairs = 0
gap_hist = Counter()
kmult_hist = Counter()
for s in svals:
    cls = area_class(s)
    total_members += len(cls)
    bound = int(2 * s) + 4
    best = {sig: (1 if len(cls) > 1 else 0) for sig in cls}   # equal area: c_1 always shared
    best_n = {sig: 0 for sig in cls}
    best_g = {sig: 0 for sig in cls}
    for sig in cls:
        if sig[0] == 0 and any(o[0] == 0 and len(o[1]) != len(sig[1]) for o in cls):
            best_n[sig] = 1
        if any(o[0] != sig[0] for o in cls):
            best_g[sig] = 1
    # Pairs sharing >= 2 coefficients have equal int_key(., 2); only those need comparing.
    # (Pairs sharing exactly 1 satisfy Theorem S trivially: T* >= 4 for distinct signatures
    # of equal area, since T* = 2 would force U* = {a}, V* = {b}, 1/a = 1/b.)
    groups = defaultdict(list)
    for sig in cls:
        groups[int_key(*sig, 2)].append(sig)
    for grp in groups.values():
        for a, b in itertools.combinations(grp, 2):
            k = shared_int(a, b, bound + 2)
            total_pairs += 1
            check(k >= 2, "equal int_key(2) => share c_1, c_2")
            check(k <= T_star(a, b) // 2 - 1, f"Theorem S sharp form {a} {b}")
            check(k < bound, f"Theorem S area form {a} {b}")
            for x, y in ((a, b), (b, a)):
                best[x] = max(best[x], k)
                if x[0] == 0 and y[0] == 0 and len(x[1]) != len(y[1]):
                    best_n[x] = max(best_n[x], k)
                if x[0] != y[0]:
                    best_g[x] = max(best_g[x], k)
    for a, b in itertools.islice(itertools.combinations(cls, 2), 2000):
        check(T_star(a, b) >= 4, "T* >= 4 (sample)")
    Kmult = {sig: best[sig] + 1 for sig in cls}
    kmax = max(Kmult.values())
    kmult_hist[kmax] += 1
    gap_hist[bound - kmax] += 1
    argmax = [sig for sig in cls if Kmult[sig] == kmax]
    if len(rows) % 100 == 0:
        print(f"  ... {len(rows)} classes done (s = {s}, |class| = {len(cls)})", flush=True)
    rows.append((s, len(cls), bound, kmax, max(best_n.values()) + 1,
                 max(best_g.values()) + 1, argmax))

# spot re-verification of the extremal pairs with the actual cone coefficients b_l
for s, size, bound, kmax, kn, kg, argmax in rows:
    if kmax >= 3:
        cls = area_class(s)
        sig = argmax[0]
        partners = [o for o in cls if o != sig and shared(sig, o, bound + 2) == kmax - 1]
        check(len(partners) > 0, f"extremal partner re-verified with b_l for {sig}")

print(f"{total_members} signatures in {len(svals)} complete area classes; {total_pairs} pairs "
      f"sharing >= 2 coefficients, all checked: Theorem S (sharp and area forms) holds")
print(f"max_O K_mult over the class, histogram: {dict(sorted(kmult_hist.items()))}")
print(f"bound floor(2s)+4 minus max K_mult, histogram: {dict(sorted(gap_hist.items()))}")
print("classes with the largest max K_mult:")
rows.sort(key=lambda r: (-r[3], r[0]))
for s, size, bound, kmax, kn, kg, argmax in rows[:12]:
    cls = area_class(s)
    sig = argmax[0]
    partner = next(o for o in cls if o != sig and shared_int(sig, o, bound + 2) == kmax - 1)
    print(f"  s={s}: |class|={size}, bound={bound}, max K_mult={kmax}, max K_n={kn}, "
          f"max K_g={kg}; e.g. {sig} ~ {partner}")
gmax = max(r[5] for r in rows)
nmax = max(r[4] for r in rows)
print(f"largest K_g (genus) over all classes examined: {gmax}; largest K_n (cone count, genus 0): {nmax}")
for s, size, bound, kmax, kn, kg, argmax in sorted(rows, key=lambda r: (-r[5], r[0]))[:5]:
    print(f"  K_g={kg} at s={s} (class size {size})")

if failures:
    print(f"{failures} FAILURES")
    sys.exit(1)
print("area_classes.py: all assertions passed")

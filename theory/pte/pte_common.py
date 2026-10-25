"""Shared exact helpers for theory/pte/.

Vocabulary (the paper section 1).

* An *L-configuration* is a nonempty finite multiset Z of nonzero rationals with
      s_j(Z) = sum z^j = 0   for odd 1 <= j <= 2L-3,      s_{-1}(Z) = sum 1/z = 0,
  and no pair {z, -z} inside Z.  Its size is T = |Z| and its imbalance is
  iota(Z) = #{z > 0} - #{z < 0}.
* By Lemma 4 of the paper, two closed orientable hyperbolic 2-orbifolds with
  different signatures share their first L heat coefficients iff the cancelled mirror multiset
  U* u (-V*) is an L-configuration; the genus difference is g - g' = -iota/2.
* Everything is exact (Fraction / int).  Heat coefficients are compared with the actual cone
  coefficients b_l of Ucar (4.25)+(4.33) through theory/signatures/sig_common.shared (read-only
  import; that module is not modified here).
"""
import math
import os
import sys
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "signatures"))
import sig_common as SC  # noqa: E402  (b_cone, s_of, shared, strip)


def pm(a):
    """{±a_i}: the symmetric form used by Borwein-Ingalls for even ideal symmetric solutions."""
    return sorted(x for v in a for x in (v, -v))


def cancel(Z):
    """Remove pairs {z, -z} (they contribute 0 to every odd power sum and to s_{-1})."""
    Z = sorted(F(z) for z in Z)
    out = []
    pool = {}
    for z in Z:
        pool[z] = pool.get(z, 0) + 1
    for z in list(pool):
        if z > 0 and -z in pool:
            k = min(pool[z], pool[-z])
            pool[z] -= k
            pool[-z] -= k
    for z, k in pool.items():
        out += [z] * k
    return sorted(out)


def iota(Z):
    return sum(1 if z > 0 else -1 for z in Z)


def is_config(Z, L):
    """Exact test of the L-configuration conditions (Z must already be cancelled)."""
    if not Z or any(z == 0 for z in Z):
        return False
    S = set(Z)
    if any(-z in S for z in S):
        return False
    if sum(1 / F(z) for z in Z) != 0:
        return False
    return all(sum(F(z) ** j for z in Z) == 0 for j in range(1, 2 * L - 2, 2))


def first_odd_failure(Z):
    """Least odd j with s_j(Z) != 0 (Z not symmetric, so it exists)."""
    j = 1
    while sum(F(z) ** j for z in Z) == 0:
        j += 2
    return j


def level(Z):
    """The largest L for which Z (with s_{-1}=0) is an L-configuration: s_j = 0 for odd j <= 2L-3.
    If the first nonzero odd power sum is s_{2L-1}, the pair shares exactly L coefficients."""
    assert sum(1 / F(z) for z in Z) == 0
    return (first_odd_failure(Z) + 1) // 2


def normalize(Z):
    """Primitive integer representative with iota >= 0 (Z -> -Z if needed)."""
    Z = [F(z) for z in Z]
    den = 1
    for z in Z:
        den = den * z.denominator // math.gcd(den, z.denominator)
    Zi = [int(z * den) for z in Z]
    g = 0
    for z in Zi:
        g = math.gcd(g, z)
    Zi = sorted(z // g for z in Zi)
    if iota(Zi) < 0 or (iota(Zi) == 0 and -Zi[0] > Zi[-1]):
        Zi = sorted(-z for z in Zi)
    return Zi


def sides(Zi):
    U = sorted(z for z in Zi if z > 0)
    V = sorted(-z for z in Zi if z < 0)
    return U, V


def realise(Zi, genus0_only=False):
    """Orbifold pair from an integral configuration Zi (U = positive part, V = negated negative part).

    Returns (sig, sig') with sig = (g, cone orders of U without 1s), sig' = (g', orders of V without 1s),
    g - g' = (|V| - |U|)/2, the smaller genus as small as possible subject to both being hyperbolic.
    With genus0_only, require g = g' = 0 (balanced configurations) and return None if not hyperbolic.
    """
    U, V = sides(Zi)
    d = len(V) - len(U)
    assert d % 2 == 0
    if genus0_only:
        assert d == 0
        a, b = (0, SC.strip(U)), (0, SC.strip(V))
        if SC.s_of(*a) > 0 and SC.s_of(*b) > 0:
            return a, b
        return None
    gp = max(0, -d // 2)
    while True:
        g = gp + d // 2
        a, b = (g, SC.strip(U)), (gp, SC.strip(V))
        if SC.s_of(*a) > 0 and SC.s_of(*b) > 0:
            assert SC.s_of(*a) == SC.s_of(*b)
            return a, b
        gp += 1


def shared_exact(a, b, cap):
    """Number of leading heat coefficients shared, from the actual cone coefficients b_l."""
    return SC.shared(a, b, cap)


def area_over_2pi(sig):
    return SC.s_of(*sig)


def odd_sums_equal(X, Y, L):
    return all(sum(F(x) ** j for x in X) == sum(F(y) ** j for y in Y) for j in range(1, 2 * L - 2, 2))


def pte_degree(X, Y):
    k = 0
    while sum(F(x) ** (k + 1) for x in X) == sum(F(y) ** (k + 1) for y in Y):
        k += 1
        if k > 64:
            break
    return k


def esym(A):
    e = [F(1)] + [F(0)] * len(A)
    for x in A:
        for k in range(len(A), 0, -1):
            e[k] += e[k - 1] * x
    return e

"""Recompute every dot of Figure 3 (F4) from scratch.

    python3 checks/pairs/a1_fig2_kmult.py        (about one minute)

For each area value s = Area/2pi of figures/data/f4_area_classes.csv this script
  1. re-derives the selection rule: the 525 values are exactly the values s in (0, 7/5]
     attained by a signature with genus g <= 2, at most 4 cone points and all orders <= 12
     (g = 2 contributes nothing, since then s >= 2);
  2. enumerates the COMPLETE area class Sig(s) = {(g; m) : 2g - 2 + sum(1 - 1/m_i) = s}, with no
     bound on the cone orders, by an exact Egyptian-fraction enumeration: for each g, the orders
     solve sum 1/m_i = n - (s + 2 - 2g) with s + 2 - 2g < n <= 2(s + 2 - 2g), finitely many;
  3. computes K_mult(O; Sig) for every member exactly, from the cone coefficients
     b_l(m) = (-1)^l p_l(m)/m of Proposition 2.8 (p_l built from Bernoulli numbers, eqs. (4)-(5)),
     and independently from the power sums Psi_k = P_{2k-1} - R (Lemma 2.11); the two must agree;
  4. compares the class size and the largest K_mult (and K_n, K_g) with the committed CSV.

This file imports nothing from theory/ or figures/: the enumeration and the coefficients are
written anew. Writes checks/pairs/output/a1_fig2_kmult.csv (one row per class, recomputed
and committed values side by side) and prints a summary; exits nonzero on any disagreement.
"""
import csv
import itertools
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
SMAX, NMAX, ORD = Fraction(7, 5), 4, 12
LMAX = 9                                     # compare c_1 .. c_{LMAX+2}; Theorem 3.4 gives far fewer

# ------------------------------------------------------------------ cone coefficients


def bernoulli(n):
    """B_0..B_n (B_1 = -1/2), by the standard recurrence."""
    B = [Fraction(0)] * (n + 1)
    B[0] = Fraction(1)
    for k in range(1, n + 1):
        B[k] = -sum(comb(k + 1, j) * B[j] for j in range(k)) / (k + 1)
    return B


BERN = bernoulli(2 * LMAX + 6)


def u_over_sin_u(N):
    """sigma_0..sigma_N with u/sin u = sum sigma_i u^{2i}: invert sin u / u = sum (-1)^i u^{2i}/(2i+1)!."""
    a = [Fraction((-1) ** i, factorial(2 * i + 1)) for i in range(N + 1)]
    s = [Fraction(0)] * (N + 1)
    s[0] = Fraction(1)
    for i in range(1, N + 1):
        s[i] = -sum(a[j] * s[i - j] for j in range(1, i + 1))
    return s


SIGMA = u_over_sin_u(LMAX + 2)


def poly_m_phi(k):
    """Coefficients (in m^{2n}, n = 0..k+1) of m phi_k(m), eq. (4)."""
    c = [Fraction(0)] * (k + 2)
    for n in range(1, k + 2):
        w = Fraction(1, 4) * SIGMA[k + 1 - n] * 4 ** n * abs(BERN[2 * n]) / factorial(2 * n)
        c[n] += w
        c[0] -= w
    return c


def poly_p(l):
    """Coefficients (in m^{2n}) of p_l(m), eq. (5)."""
    c = [Fraction(0)] * (l + 2)
    for k in range(l + 1):
        w = Fraction(factorial(2 * k), factorial(k) * factorial(l - k) * 4 ** l)
        for n, x in enumerate(poly_m_phi(k)):
            c[n] += w * x
    return c


P = [poly_p(l) for l in range(LMAX + 1)]
# sanity: eq. (7) of the paper
assert P[0] == [Fraction(-1, 12), Fraction(1, 12)]
assert P[1] == [Fraction(-11, 360), Fraction(1, 36), Fraction(1, 360)]
assert P[2] == [Fraction(-37, 5040), Fraction(1, 180), Fraction(1, 720), Fraction(1, 2520)]
for l in range(LMAX + 1):
    assert sum(P[l]) == 0                    # p_l(1) = 0
    assert P[l][-1] == abs(BERN[2 * l + 2]) / (2 * factorial(l + 1) * (2 * l + 1))

_bcache = {}


def b(l, m):
    if (l, m) not in _bcache:
        _bcache[(l, m)] = (-1) ** l * sum(c * Fraction(m) ** (2 * n) for n, c in enumerate(P[l])) / m
    return _bcache[(l, m)]


def cone_vector(m):
    """(C_0, ..., C_LMAX), C_l = sum_i b_l(m_i): with the area, these give c_2, ..., c_{LMAX+2}."""
    return tuple(sum((b(l, x) for x in m), Fraction(0)) for l in range(LMAX + 1))


def psi_vector(m):
    """(Psi_1, ..., Psi_{LMAX+1}), Psi_k = P_{2k-1} - R."""
    R = sum(Fraction(1, x) for x in m)
    return tuple(sum(Fraction(x) ** (2 * k - 1) for x in m) - R for k in range(1, LMAX + 2))


def prefix(u, v):
    n = 0
    while n < len(u) and u[n] == v[n]:
        n += 1
    return n


# ------------------------------------------------------------------ complete area classes


def s_of(g, m):
    return 2 * g - 2 + sum(1 - Fraction(1, x) for x in m)


def unit_fraction_sums(rho, n, lo):
    """All nondecreasing (m_1..m_n), m_1 >= lo >= 2, with sum 1/m_i = rho (exact)."""
    if n == 0:
        return [()] if rho == 0 else []
    if rho <= 0:
        return []
    if n == 1:
        return [(rho.denominator,)] if rho.numerator == 1 and rho.denominator >= lo else []
    out = []
    a = max(lo, -(-rho.denominator // rho.numerator))     # 1/a <= rho
    while Fraction(n, a) >= rho:                           # the largest term 1/a is >= rho/n
        for rest in unit_fraction_sums(rho - Fraction(1, a), n - 1, a):
            out.append((a,) + rest)
        a += 1
    return out


def area_class(s):
    cls = []
    g = 0
    while 2 * g - 2 <= s:
        sigma = s + 2 - 2 * g                  # = sum(1 - 1/m_i), each term in [1/2, 1)
        n = 0
        while Fraction(n, 2) <= sigma:
            if sigma < n or (n == 0 and sigma == 0):
                for m in unit_fraction_sums(n - sigma, n, 2):
                    cls.append((g, m))
            n += 1
        g += 1
    assert len(set(cls)) == len(cls) and all(s_of(*o) == s for o in cls)
    return cls


# ------------------------------------------------------------------ K values on one class


def k_values(cls):
    """Largest K_mult, K_n (genus 0 vs genus 0 with another cone count), K_g over the class."""
    vec = {o: cone_vector(o[1]) for o in cls}
    psi = {o: psi_vector(o[1]) for o in cls}
    best = {o: (1 if len(cls) > 1 else 0) for o in cls}    # equal area: c_1 shared by all members
    best_n = {o: (1 if o[0] == 0 and any(x[0] == 0 and len(x[1]) != len(o[1]) for x in cls) else 0) for o in cls}
    best_g = {o: (1 if any(x[0] != o[0] for x in cls) else 0) for o in cls}
    groups = defaultdict(list)
    for o in cls:
        groups[vec[o][0]].append(o)
    for grp in groups.values():
        for x, y in itertools.combinations(grp, 2):
            k = 1 + prefix(vec[x], vec[y])                 # shared coefficients c_1, ..., c_k
            assert k == 1 + prefix(psi[x], psi[y]), (x, y)  # Lemma 2.11: same count from power sums
            assert k <= LMAX, (x, y)                       # never saturates the comparison
            for o, p in ((x, y), (y, x)):
                best[o] = max(best[o], k)
                if o[0] == 0 and p[0] == 0 and len(o[1]) != len(p[1]):
                    best_n[o] = max(best_n[o], k)
                if o[0] != p[0]:
                    best_g[o] = max(best_g[o], k)
    # pairs in different C_0 groups share exactly c_1; check that on a sample against psi
    for x, y in itertools.islice(itertools.combinations(cls, 2), 3000):
        if vec[x][0] != vec[y][0]:
            assert psi[x][0] != psi[y][0]
    return (max(best.values()) + 1, max(best_n.values()) + 1, max(best_g.values()) + 1,
            [o for o in cls if best[o] + 1 == max(best.values()) + 1])


def fmt(o):
    return f"({o[0]};{','.join(map(str, o[1]))})"


def main():
    committed = list(csv.DictReader(open(ROOT / "figures/data/f4_area_classes.csv")))
    svals = sorted({s_of(g, m) for g in range(3) for n in range(NMAX + 1)
                    for m in itertools.combinations_with_replacement(range(2, ORD + 1), n)
                    if 0 < s_of(g, m) <= SMAX})
    assert not any(s_of(2, m) <= SMAX for n in range(NMAX + 1)
                   for m in itertools.combinations_with_replacement(range(2, ORD + 1), n))
    assert len(svals) == 525 == len(committed)
    assert [Fraction(r["s"]) for r in committed] == svals, "selection rule does not reproduce the CSV's s values"
    # the area values are not all area values below 7/5: e.g. (0;2,3,13) has s = 1/6 - 1/13
    assert s_of(0, (2, 3, 13)) not in svals

    rows, changed, hist, sizes, maxorder = [], [], Counter(), 0, 0
    for r in committed:
        s = Fraction(r["s"])
        cls = area_class(s)
        km, kn, kg, argmax = k_values(cls)
        sizes += len(cls)
        hist[km] += 1
        big = max((max(o[1]) for o in cls if o[1]), default=0)
        maxorder = max(maxorder, big)
        old = (int(r["class_size"]), int(r["max_K_mult"]), int(r["max_K_n"]), int(r["max_K_g"]))
        new = (len(cls), km, kn, kg)
        if old != new:
            changed.append((s, old, new))
        assert r["extremal_signature"] in {fmt(o) for o in argmax} or old[1] != km, (s, r["extremal_signature"])
        rows.append([str(s), len(cls), big, km, kn, kg, *old])

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_fig2_kmult.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["s", "class_size", "largest_order_in_class", "max_K_mult", "max_K_n", "max_K_g",
                    "committed_class_size", "committed_max_K_mult", "committed_max_K_n", "committed_max_K_g"])
        w.writerows(rows)

    print("A1: Fig. 2 (F4) dots recomputed from complete area classes")
    print(f"selection rule: the {len(svals)} values s in (0, 7/5] attained with genus <= 2 (in effect <= 1), "
          f"at most {NMAX} cone points, orders <= {ORD}; equals the committed s column")
    print(f"complete classes: {sizes} signatures in total, no order bound; largest order met {maxorder}")
    print(f"max K_mult histogram: {dict(sorted(hist.items()))}")
    print(f"coefficient criterion (b_l) and power-sum criterion (Psi_k) agree on every pair")
    if changed:
        print(f"{len(changed)} CLASSES DIFFER from figures/data/f4_area_classes.csv:")
        for s, old, new in changed:
            print(f"  s={s}: committed (size, K_mult, K_n, K_g) = {old}, recomputed {new}")
        sys.exit(1)
    print("no value changes: every class size and every max K_mult, K_n, K_g equals the committed CSV")


if __name__ == "__main__":
    main()

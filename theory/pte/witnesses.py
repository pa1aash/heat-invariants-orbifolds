"""Explicit L-configurations and the orbifold pairs they give (the paper sections 3 and 5).

Three kinds of witness, for L = 2..7:
  genus    : iota != 0  -> orbifolds of different genus sharing exactly L heat coefficients;
  balanced : iota == 0, no padding point -> genus-0 orbifolds with the same cone count;
  cone     : iota == 0 with a padding point -> genus-0 orbifolds with different cone counts.

Every witness is rebuilt here from a short recipe (fetched PTE data plus a shift / doubling /
scaling), checked exactly as an L-configuration, realised as an orbifold pair, and the number of
shared heat coefficients is recomputed from the actual cone coefficients b_l (sig_common.shared).
Run: python3 witnesses.py   (about 2 minutes)
Writes data/witnesses.json and prints a transcript (saved as output/witnesses.txt).
"""
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pte_common import (pm, cancel, iota, is_config, level, normalize, sides, realise,  # noqa: E402
                        shared_exact, area_over_2pi, odd_sums_equal, pte_degree)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- fetched input data
# Even ideal symmetric PTE solutions, {±a} =_{2m-1} {±b} (Borwein-Ingalls 1994 p. 9;
# BLP 2003 p. 2069; CMSV 2023 eqs. (4)-(5)).
PTE = {
    4: (pm([3, 11]), pm([7, 9])),                                       # B-I p.9, size 4
    6: (pm([4, 9, 13]), pm([1, 11, 12])),                               # B-I p.9, size 6
    8: (pm([2, 16, 21, 25]), pm([5, 14, 23, 24])),                      # B-I p.9, size 8
    10: [(pm([436, 11857, 20449, 20667, 23750]), pm([12, 11881, 20231, 20885, 23738])),  # B-I p.9
         (pm([71, 131, 180, 307, 308]), pm([99, 100, 188, 301, 313])),                 # BLP p.2069
         (pm([18, 245, 331, 471, 508]), pm([103, 189, 366, 452, 515]))],               # BLP p.2069
    12: (pm([22, 61, 86, 127, 140, 151]), pm([35, 47, 94, 121, 146, 148])),            # CMSV (5)
}
# Equal sums of odd powers, [a] = [b] for k = 1,3,...,2L-3 (Chen survey A.1.6, A.1.17, A.1.26, A.1.33).
ODDEQ = {
    3: [([1, 5, 5], [2, 3, 6])],
    4: [([1, 13, 17, 23], [3, 9, 21, 21]), ([6, 16, 18, 24], [7, 13, 21, 23])],
    5: [([3, 19, 37, 51, 53], [9, 11, 43, 45, 55])],
    6: [([7, 91, 173, 269, 289, 323], [29, 59, 193, 247, 311, 313]),
        ([23, 163, 181, 341, 347, 407], [37, 119, 221, 311, 371, 403]),
        ([43, 161, 217, 335, 391, 463], [85, 91, 283, 287, 403, 461]),
        ([57, 399, 679, 995, 1167, 1293], [115, 299, 767, 925, 1205, 1279])],
}
# Odd ideal symmetric solutions (B = -A): s_j(A) = 0 for odd j <= |A|-2 (B-I p.9 and p.25; CMSV (3)).
ODDSYM = {
    4: [[-51, -33, -24, 7, 13, 38, 50], [-90, -86, -39, -5, 48, 77, 95], [-116, -104, -36, -19, 75, 77, 123],
        [-120, -110, -23, -13, 38, 105, 123], [-134, -75, -66, 8, 47, 87, 133]],
    5: [[-98, -82, -58, -34, 13, 16, 69, 75, 99], [-169, -161, -119, -63, 8, 50, 132, 148, 174]],
}


def gloden7(f, k):
    """Gloden's two-parameter size-7 family, as printed in BLP 2003 p. 2064."""
    return [-(f * f - k * f + k * k) * (-3 * k * f * f + k ** 3 + f ** 3),
            -(k - f) * (f + k) * (f * f - 3 * k * f + k * k) * f,
            (-f + 2 * k) * (-f * f - k * f + k * k) * k * f,
            (k - f) * (k - 2 * f) * (-f * f + k * f + k * k) * k,
            (k - f) * (f ** 4 - 2 * k * f ** 3 - k * k * f * f + k ** 4),
            -(k ** 4 - 2 * f * k ** 3 - k * k * f * f + 4 * k * f ** 3 - f ** 4) * k,
            -(k ** 4 - 5 * k * k * f * f + 4 * k * f ** 3 - f ** 4) * f]


# ---------------------------------------------------------------- constructions
def shift_piece(X, Y, c):
    """(X+c) u -(Y+c): odd power sums vanish to degree k if X =_k Y (the paper Prop. 3.2)."""
    return [F(x) + c for x in X] + [-(F(y) + c) for y in Y]


def doubling(X, Y):
    """U = X u 2Y u 2Y, V = Y u 2X u 2X (the paper Prop. 3.3): configuration with s_{-1} = 0."""
    U = list(X) + [2 * y for y in Y] * 2
    V = list(Y) + [2 * x for x in X] * 2
    return [F(u) for u in U] + [-F(v) for v in V]


def scale_combine(P, B):
    """P u t*B with t chosen so that s_{-1} vanishes; returns cancelled multiset or None."""
    if any(p == 0 for p in P) or any(b == 0 for b in B):
        return None
    rP = sum(1 / F(p) for p in P)
    rB = sum(1 / F(b) for b in B)
    if rP == 0:
        return cancel(P)
    if rB == 0:
        return None
    t = -rB / rP  # s_{-1}(tB) = rB/t, so rP + rB/t = 0
    return cancel(list(P) + [t * F(b) for b in B])


def halfsums(X, Y):
    s = set()
    for S in (X, Y):
        for a, b in itertools.combinations_with_replacement(S, 2):
            s.add(-F(a + b, 2))
    return sorted(s)


# ---------------------------------------------------------------- bookkeeping
records = []


def record(kind, L, Z, recipe):
    Z = cancel(Z)
    assert is_config(Z, L), (kind, L, recipe)
    Zi = normalize(Z)
    assert is_config([F(z) for z in Zi], L)
    Lmax = level(Zi)
    io = iota(Zi)
    if kind == "genus":
        assert io != 0
        a, b = realise(Zi)
    else:
        assert io == 0
        r = realise(Zi, genus0_only=True)
        if r is None:
            return None
        a, b = r
        pad = (1 in Zi) or (-1 in Zi)
        assert pad == (kind == "cone"), (kind, recipe)
        if kind == "cone":
            assert len(a[1]) != len(b[1])
        else:
            assert len(a[1]) == len(b[1])
    sc = shared_exact(a, b, Lmax + 2)
    assert sc == Lmax >= L, (kind, L, sc, Lmax)
    rec = dict(kind=kind, L=L, shares_exactly=sc, T=len(Zi), iota=io, Z=Zi,
               sig1=dict(g=a[0], orders=list(a[1])), sig2=dict(g=b[0], orders=list(b[1])),
               area_over_2pi=str(area_over_2pi(a)), area_over_2pi_float=float(area_over_2pi(a)),
               cone_counts=[len(a[1]), len(b[1])], max_order=max(abs(z) for z in Zi), recipe=recipe)
    records.append(rec)
    return rec


def best_of(cands):
    cands = [c for c in cands if c is not None]
    return min(cands, key=lambda r: (r["T"], Fraction_key(r["area_over_2pi"]), r["max_order"])) if cands else None


def Fraction_key(s):
    return F(s)


# ---------------------------------------------------------------- genus witnesses
def genus_witnesses():
    out = {}
    # L = 2, 3: the smallest pairs of the paper section 5 (exact search there)
    out[2] = record("genus", 2, [F(15), F(1), F(-3), F(-3), F(-5), F(-5)], "(1;15) vs (0;3,3,5,5) [paper]")
    out[3] = record("genus", 3, [F(x) for x in [15, 15, 15, 1, -3, -3, -5, -7, -7, -21]],
                    "(1;15,15,15) vs (0;3,3,5,7,7,21) [paper]")
    # L = 4 also uses the size-6 symmetric solution {±7,±11,±18} =_5 {±3,±14,±17}, found by our own
    # enumeration of size-6 symmetric solutions (entries <= 400; degree asserted below). Shifted by
    # c = -11/2 it loses two cancelling pairs and becomes an imbalanced 8-element set with s_1=s_3=s_5=0.
    plan = {4: [PTE[6], (pm([7, 11, 18]), pm([3, 14, 17]))], 5: [PTE[8]], 6: PTE[10], 7: [PTE[12]]}
    for L, ptes in plan.items():
        cands = []
        for X, Y in ptes:
            assert pte_degree(X, Y) >= 2 * L - 3
            cs = halfsums(X, Y)
            balanced_pieces = [(f"odd-power equality {a}={b}", [F(x) for x in a] + [-F(y) for y in b]) for a, b in ODDEQ.get(L, [])
                               if odd_sums_equal(a, b, L)]
            for c2 in cs + [F(0)]:
                P2 = shift_piece(X, Y, c2)
                if 0 in P2:
                    continue
                P2 = cancel(P2)
                if P2 and iota(P2) == 0:
                    balanced_pieces.append((f"shift c'={c2}", P2))
            for c in cs:
                P = shift_piece(X, Y, c)
                if 0 in P:
                    continue
                P = cancel(P)
                if not P or iota(P) == 0:
                    continue
                for tag, B in balanced_pieces:
                    Z = scale_combine(P, B)
                    if Z and iota(Z) != 0:
                        cands.append((len(Z), max(abs(z) for z in normalize(Z)), Z,
                                      f"shift of {X},{Y} by c={c} plus scaled balanced piece ({tag})"))
        best = min(cands, key=lambda t: (t[0], t[1]))
        out[L] = record("genus", L, best[2], best[3])
    return out


# ---------------------------------------------------------------- balanced (same genus, same n)
def balanced_witnesses():
    out = {}
    out[2] = record("balanced", 2, [F(x) for x in [2, 8, 8, -3, -3, -12]], "{2,8,8} vs {3,3,12} [paper]")
    out[3] = record("balanced", 3, [F(x) for x in [3, 10, 15, 30, -4, -5, -21, -28]],
                    "{3,10,15,30} vs {4,5,21,28}: a pencil point (the paper Thm 3.1); also audibility Thm C, Chen A.685")
    # L = 4: pairs of odd ideal symmetric 7-sets (Prop. 2.3(b)); fetched sets plus Gloden's family
    sets7 = [tuple(A) for A in ODDSYM[4]]
    for f in range(-12, 13):
        for k in range(1, 13):
            if math.gcd(f, k) != 1:
                continue
            A = gloden7(f, k)
            if 0 in A or any(-a in A for a in A):
                continue
            g = 0
            for a in A:
                g = math.gcd(g, a)
            A = tuple(sorted(a // g for a in A))
            if A not in sets7 and tuple(sorted(-a for a in A)) not in sets7:
                sets7.append(A)
    for A in sets7:
        assert all(sum(F(a) ** j for a in A) == 0 for j in (1, 3, 5))
    cands = []
    for A, B in itertools.combinations(sets7, 2):
        Z = scale_combine([F(a) for a in A], [F(b) for b in B])
        if Z and iota(Z) == 0 and 1 not in normalize(Z) and -1 not in normalize(Z):
            cands.append((len(Z), max(abs(z) for z in normalize(Z)), Z, f"7-sets {A} and {B}"))
    best = min(cands, key=lambda t: (t[0], t[1]))
    out[4] = record("balanced", 4, best[2], best[3])
    out["L4_pairs_tested"] = len(cands)
    A, B = ODDSYM[5]
    out[5] = record("balanced", 5, scale_combine([F(a) for a in A], [F(b) for b in B]), "Letac's two 9-sets (B-I p.9)")
    # L = 6: two odd-power equalities of Chen A.1.33, scaled against each other
    cands = []
    for (a, b), (a2, b2) in itertools.combinations(ODDEQ[6], 2):
        Z = scale_combine([F(x) for x in a] + [-F(y) for y in b], [F(x) for x in a2] + [-F(y) for y in b2])
        if Z and iota(Z) == 0:
            cands.append((len(Z), max(abs(z) for z in normalize(Z)), Z, f"odd equalities {a}={b} and {a2}={b2}"))
    best = min(cands, key=lambda t: (t[0], t[1]))
    out[6] = record("balanced", 6, best[2], best[3])
    # L = 7: two balanced shifts of the size-12 solution
    X, Y = PTE[12]
    cands = []
    cs = halfsums(X, Y)
    pieces = []
    for c in cs + [F(0)]:
        P = shift_piece(X, Y, c)
        if 0 in P:
            continue
        P = cancel(P)
        if P and iota(P) == 0:
            pieces.append((c, P))
    for (c1, P1), (c2, P2) in itertools.combinations(pieces, 2):
        Z = scale_combine(P1, P2)
        if Z and iota(Z) == 0:
            Zn = normalize(Z)
            if 1 not in Zn and -1 not in Zn:
                cands.append((len(Z), max(abs(z) for z in Zn), Z, f"size-12 solution shifted by {c1} and {c2}"))
    best = min(cands, key=lambda t: (t[0], t[1]))
    out[7] = record("balanced", 7, best[2], best[3])
    return out


# ---------------------------------------------------------------- cone-count (genus 0, n vs n')
def cone_witnesses():
    out = {}
    out[2] = record("cone", 2, [F(x) for x in [5, 5, 5, 1, -2, -2, -2, -10]], "(0;5,5,5) vs (0;2,2,2,10) [paper]")
    for L in (3, 4):
        cands = []
        for a, b in ODDEQ[L]:
            if 1 in a:
                cands.append(doubling(a, b))
        best = min((cancel(Z) for Z in cands), key=len)
        out[L] = record("cone", L, best, f"doubling (Prop. 3.3) of the odd-power equality {ODDEQ[L][0]}")
    # L >= 5: shift an even ideal symmetric solution so that its least element is 1, then double
    for L, X, Y in [(5,) + PTE[8], (6,) + PTE[10][1], (7,) + PTE[12]]:
        m = min(X + Y)
        Xs, Ys = [x - m + 1 for x in X], [y - m + 1 for y in Y]
        if 1 in Ys:
            Xs, Ys = Ys, Xs
        assert 1 in Xs and 1 not in Ys and pte_degree(Xs, Ys) >= 2 * L - 3
        out[L] = record("cone", L, doubling(Xs, Ys), f"shift of the size-{len(X)} ideal solution by {1 - m}, then doubling")
    return out


def main():
    t0 = time.time()
    G = genus_witnesses()
    B = balanced_witnesses()
    C = cone_witnesses()
    print("kind      L  T   iota  cone counts   genera   Area/2pi        max order (digits)")
    for r in records:
        print(f"{r['kind']:8s} {r['L']:2d} {r['T']:3d} {r['iota']:4d}   {str(r['cone_counts']):12s} "
              f"({r['sig1']['g']},{r['sig2']['g']})    {r['area_over_2pi']:>16s} ~ {r['area_over_2pi_float']:9.4f}"
              f"   {len(str(r['max_order']))}")
    for r in records:
        print(f"\n[{r['kind']} L={r['L']}] shares exactly {r['shares_exactly']} (actual b_l); recipe: {r['recipe']}")
        print(f"  O  = ({r['sig1']['g']}; {r['sig1']['orders']})")
        print(f"  O' = ({r['sig2']['g']}; {r['sig2']['orders']})")
    json.dump(records, open(os.path.join(HERE, "data", "witnesses.json"), "w"), indent=1, default=str)
    with open(os.path.join(HERE, "data", "orbifold_pairs.md"), "w") as fh:
        fh.write("# Orbifold pairs sharing exactly L heat coefficients (generated by witnesses.py)\n\n"
                 "Signature (g; cone orders). Area = 2*pi*s. Shared count recomputed from the cone "
                 "coefficients b_l of Ucar (4.25)+(4.33). Configuration Z = U* u (-V*) (primitive integers).\n")
        for r in records:
            fh.write(f"\n## {r['kind']}, L = {r['L']}  (T = {r['T']}, iota = {r['iota']}, cone counts "
                     f"{r['cone_counts'][0]} vs {r['cone_counts'][1]})\n\n")
            fh.write(f"- O  = ({r['sig1']['g']}; {', '.join(map(str, r['sig1']['orders']))})\n")
            fh.write(f"- O' = ({r['sig2']['g']}; {', '.join(map(str, r['sig2']['orders']))})\n")
            fh.write(f"- Area/2pi = {r['area_over_2pi']} (~ {r['area_over_2pi_float']:.9f})\n")
            fh.write(f"- recipe: {r['recipe']}\n")
    print(f"\nL=4 balanced: {B['L4_pairs_tested']} balanced pairs of 7-sets tested")
    print(f"ALL CHECKS PASSED ({len(records)} witnesses, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""Pencil points (the paper Theorem 3.1).

m = 4 (L = 3): A = {a, b, c, -(a+b+c)} with |entries| <= N, primitive; two sets are a pencil pair iff
              e_3^4/e_4^3 agree (then a scaling of B has the same e_3, e_4 as A).
m = 5 (L = 4): odd symmetric 5-sets (e_1 = e_3 = 0) with |entries| <= N; pencil pair iff e_4^5/e_5^4 agree.
Every hit is verified to be an (m-1)-configuration of size 2m with iota = 0, and checked for a
padding point +-1 (which would give a genus-0 pair with cone counts m-1 vs m).
Usage: python3 pencil_search.py [N4] [N5]     defaults 130 200 (about 2 minutes);
       the log in data/pencil_log.txt also records the run with N4 = 220 (about 6 minutes).
"""
import json
import math
import os
import sys
from collections import defaultdict
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pte_common import cancel, iota, is_config, normalize, esym, realise, shared_exact  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def pencil_pairs(sets, key, m):
    groups = defaultdict(list)
    for A in sets:
        groups[key(esym(A))].append(A)
    hits = []
    for lst in groups.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                A, B = lst[i], lst[j]
                eA, eB = esym(A), esym(B)
                if m == 4:
                    lam = (eA[4] / eB[4]) / (eA[3] / eB[3])
                else:
                    lam = (eA[5] / eB[5]) / (eA[4] / eB[4])
                Bs = [lam * b for b in B]
                if sorted(Bs) == sorted(F(a) for a in A):
                    continue
                Z = cancel([F(a) for a in A] + [-b for b in Bs])
                if not Z:
                    continue
                assert len(Z) == 2 * m and is_config(Z, m - 1) and iota(Z) == 0, (A, B)
                hits.append(normalize(Z))
    uniq = sorted({tuple(z) for z in hits}, key=lambda z: (max(map(abs, z)), z))
    return uniq


def sets_m4(N):
    out = set()
    for a in range(-N, N + 1):
        for b in range(a, N + 1):
            for c in range(b, N + 1):
                d = -(a + b + c)
                A = tuple(sorted((a, b, c, d)))
                if 0 in A or any(-x in A for x in A):
                    continue
                if math.gcd(math.gcd(a, b), math.gcd(c, d)) != 1:
                    continue
                if esym(A)[3] == 0:
                    continue
                out.add(A)
    return out


def sets_m5(N):
    out = set()
    vals = [v for v in range(-N, N + 1) if v]
    for i, a in enumerate(vals):
        for j in range(i, len(vals)):
            b = vals[j]
            for k in range(j, len(vals)):
                c = vals[k]
                S, C = -(a + b + c), -(a ** 3 + b ** 3 + c ** 3)
                if S == 0:
                    continue
                num = S ** 3 - C
                if num % (3 * S):
                    continue
                p = num // (3 * S)
                D = S * S - 4 * p
                if D < 0:
                    continue
                r = math.isqrt(D)
                if r * r != D or (S + r) % 2:
                    continue
                d, e = (S + r) // 2, (S - r) // 2
                A = sorted((a, b, c, d, e))
                if 0 in A or any(-x in A for x in A):
                    continue
                g = 0
                for x in A:
                    g = math.gcd(g, x)
                out.add(tuple(x // g for x in A))
    return out


def main():
    N4 = int(sys.argv[1]) if len(sys.argv) > 1 else 130
    N5 = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    s4 = sets_m4(N4)
    h4 = pencil_pairs(s4, lambda e: e[3] ** 4 / e[4] ** 3, 4)
    pad4 = [z for z in h4 if 1 in z or -1 in z]
    print(f"m=4 (L=3): {len(s4)} sets with e1=0, |entries|<={N4}; {len(h4)} distinct pencil configurations "
          f"(size 8, iota 0); with a padding point +-1: {len(pad4)}")
    print("  smallest:", h4[:3])
    s5 = sets_m5(N5)
    h5 = pencil_pairs(s5, lambda e: e[4] ** 5 / e[5] ** 4, 5)
    print(f"m=5 (L=4): {len(s5)} odd symmetric 5-sets, |entries|<={N5}; {len(h5)} pencil configurations")
    json.dump(dict(N4=N4, m4=h4, N5=N5, m5=h5), open(os.path.join(HERE, "data", f"pencil_N{N4}_{N5}.json"), "w"))
    # every m = 4 pencil configuration is an integer sharpness witness for Theorem A at n = 4
    stored = os.path.join(HERE, "data", "pencil_m4_N220.json")
    extra = json.load(open(stored))["m4"] if os.path.exists(stored) else []
    allz = {tuple(z) for z in h4} | {tuple(z) for z in extra}
    for z in allz:
        Z = [F(v) for v in z]
        assert len(Z) == 8 and is_config(Z, 3) and iota(Z) == 0
        a, b = realise(list(z), genus0_only=True)
        assert len(a[1]) == len(b[1]) == 4 and shared_exact(a, b, 5) == 3, z
    print(f"sharpness of Theorem A at n=4: {len(allz)} distinct genus-0 pairs (4 vs 4 cone points) sharing exactly 3 "
          f"coefficients (actual b_l), incl. the {len(extra)} stored from the N4=220 run")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

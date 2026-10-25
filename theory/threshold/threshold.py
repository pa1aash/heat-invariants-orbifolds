#!/usr/bin/env python3
"""Closed-form first-overlap sum S*(p) of adjacent least-order strata, and its
comparison with the first actual two-coefficient collision.

Notation (Section 5 of the paper): a hyperbolic triad is
2 <= p <= q <= r with 1/p + 1/q + 1/r < 1; the (S, p)-stratum is the set of
hyperbolic triads with sum S and least order p. Lemma lem:chamber gives the
endpoints

    R^+_{S,p} = 2/p + 1/(S-2p)                        (spread triad (p,p,S-2p))
    R^-_{S,p} = 1/p + 1/floor(D/2) + 1/ceil(D/2),  D = S-p   (balanced triad)

and the proof of Theorem thm:separation writes, for S-p even,
R^-_{S,p} - R^+_{S,p+1} = phi_p(S) - tau_p with

    phi_p(S) = 4/(S-p) - 1/(S-2p-2),   tau_p = (p-1)/(p(p+1)).

Claims verified here (the paper gives the proofs; every check is exact):

  (C1) phi_p(S) - tau_p = -(p-1)(S-3p-2)(S - x*(p)) / (p(p+1)(S-p)(S-2p-2))
       with x*(p) = 3p(p+1)/(p-1); discriminant of the quadratic 4(2p+1)^2.
  (C2) For S-p odd, R^- - R^+_{p+1} = phi_p(S) - tau_p + 4/(D(D^2-1)), and at
       S = 3p+7 it equals -2(p^2-5p-30) / (p(p+1)(p+3)(p+4)(p+5)), so the
       strata overlap at 3p+7 iff p >= 9.
  (C3) S*(p) = 18, 19 for p = 2, 3;  3p+8 for 4 <= p <= 8;  3p+7 for p >= 9;
       and the strata p, p+1 overlap at sum S iff S >= S*(p).
  (C4) min_p x*(p) = 18, attained at p = 2, 3 only, so no two strata of any
       sum S <= 17 overlap, hence no two-coefficient collision below S = 18.
  (C5) (C3) agrees with the endpoints computed by brute force from the actual
       hyperbolic triads, for every p and every S <= 600.
  (C6) First actual collision between strata p and p+1 (and between any two
       strata), from exact enumeration S <= 600, compared with S*(p).
"""
from __future__ import annotations

import csv
import sys
from fractions import Fraction as F
from math import gcd
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
S_MAX = 600


# ---------------------------------------------------------------- symbolic part
def symbolic_checks() -> None:
    p, S, D = sp.symbols("p S D", positive=True)
    phi = 4 / (S - p) - 1 / (S - 2 * p - 2)
    tau = (p - 1) / (p * (p + 1))
    xstar = 3 * p * (p + 1) / (p - 1)
    claimed = -(S - 3 * p - 2) * (S - xstar) * (p - 1) / (p * (p + 1) * (S - p) * (S - 2 * p - 2))
    assert sp.simplify(phi - tau - claimed) == 0, "C1 factorisation"
    quad = (p - 1) * S**2 - (6 * p**2 + 2 * p - 2) * S + 3 * p * (p + 1) * (3 * p + 2)
    assert sp.expand(sp.discriminant(quad, S) - 4 * (2 * p + 1) ** 2) == 0, "C1 discriminant"
    assert set(sp.solve(quad, S)) == {3 * p + 2, xstar}, "C1 roots"
    # x*(p) - 18 = 3(p-2)(p-3)/(p-1)  (C4)
    assert sp.simplify(xstar - 18 - 3 * (p - 2) * (p - 3) / (p - 1)) == 0, "C4"
    # x*(p) = 3p + 6 + 6/(p-1)
    assert sp.simplify(xstar - (3 * p + 6 + 6 / (p - 1))) == 0
    # odd D: balanced pair (D-1)/2, (D+1)/2
    odd_bal = 2 / (D - 1) + 2 / (D + 1)
    assert sp.simplify(odd_bal - 4 * D / (D**2 - 1)) == 0
    assert sp.simplify(odd_bal - 4 / D - 4 / (D * (D**2 - 1))) == 0, "C2 odd correction"
    # gap at S = 3p+7 (D = 2p+7 odd): R^- - R^+_{p+1}
    gap = (1 / p + odd_bal) - (2 / (p + 1) + 1 / (D - p - 2))
    g37 = sp.factor(sp.together(gap.subs(D, 2 * p + 7)))
    want = -8 * (p**2 - 5 * p - 30) / (4 * p * (p + 1) * (p + 3) * (p + 4) * (p + 5))
    assert sp.simplify(g37 - want) == 0, ("C2", g37)
    # p^2 - 5p - 30 >= 0 for all p >= 9: substitute p = 9 + u, all coefficients >= 0
    u = sp.symbols("u")
    coeffs = sp.Poly(sp.expand((p**2 - 5 * p - 30).subs(p, 9 + u)), u).all_coeffs()
    assert all(c > 0 for c in coeffs), coeffs
    assert (p**2 - 5 * p - 30).subs(p, 8) < 0
    # phi_p decreasing for S > 3p+4, and 4/(D(D^2-1)) decreasing in D > 1
    dphi = sp.diff(phi, S)
    assert sp.simplify(dphi - (-4 / (S - p) ** 2 + 1 / (S - 2 * p - 2) ** 2)) == 0
    # sign of dphi: numerator (S-p)^2 - 4(S-2p-2)^2 = (3p+4-S)(S+... ) factor
    numer = sp.factor((S - p) ** 2 - 4 * (S - 2 * p - 2) ** 2)
    assert sp.expand(numer - (-(S - 3 * p - 4) * (3 * S - 5 * p - 4))) == 0, numer
    print("symbolic: C1 factorisation, discriminant 4(2p+1)^2, roots {3p+2, 3p(p+1)/(p-1)}")
    print("symbolic: gap at S=3p+7 =", want, " (<= 0 iff p >= 9)")


# ------------------------------------------------------------ endpoint formulas
def R_plus(S: int, p: int) -> F:
    return F(2, p) + F(1, S - 2 * p)


def R_minus(S: int, p: int) -> F:
    d = S - p
    return F(1, p) + F(1, d // 2) + F(1, d - d // 2)


def x_star(p: int) -> F:
    return F(3 * p * (p + 1), p - 1)


def overlap_formula(S: int, p: int) -> bool:
    """Adjacent strata p, p+1 at sum S overlap: R^-_{S,p} <= R^+_{S,p+1}."""
    assert S >= 3 * p + 3
    return R_minus(S, p) <= R_plus(S, p + 1)


def S_star_closed(p: int) -> int:
    if p == 2:
        return 18
    if p == 3:
        return 19
    if p <= 8:
        return 3 * p + 8
    return 3 * p + 7


def S_star_from_cases(p: int) -> int:
    """Independent route: least S >= x*(p) with S = p mod 2, versus the odd test."""
    xs = x_star(p)
    s_even = -(-xs.numerator // xs.denominator)
    if (s_even - p) % 2:
        s_even += 1
    s = 3 * p + 3
    while not ((s - p) % 2 == 1 and overlap_formula(s, p)):
        s += 1
    return min(s_even, s)


def window_counts(S: int, p: int) -> tuple[int, int]:
    """Triads of stratum p (resp. p+1) lying in the overlap window
    [R^-_{S,p}, R^+_{S,p+1}]: the only candidates for an adjacent collision."""
    lo, hi = R_minus(S, p), R_plus(S, p + 1)
    n_p = sum(1 for q in range(p, (S - p) // 2 + 1)
              if hyperbolic(p, q, S - p - q) and F(1, p) + F(1, q) + F(1, S - p - q) <= hi)
    n_p1 = sum(1 for q in range(p + 1, (S - p - 1) // 2 + 1)
               if F(1, p + 1) + F(1, q) + F(1, S - p - 1 - q) >= lo)
    return n_p, n_p1


# ------------------------------------------------------------------ enumeration
def hyperbolic(p: int, q: int, r: int) -> bool:
    return q * r + p * r + p * q < p * q * r


def key(p: int, q: int, r: int) -> tuple[int, int]:
    e2, e3 = q * r + p * r + p * q, p * q * r
    g = gcd(e2, e3)
    return e2 // g, e3 // g


def enumerate_all(s_max: int):
    """Per S: brute-force stratum endpoints and the R-fibres."""
    endpoints = {}    # (S, p) -> (min R, max R) over hyperbolic triads
    fibres = {}       # S -> {key: [triads]}
    for S in range(10, s_max + 1):
        fib: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for p in range(2, S // 3 + 1):
            lo = hi = None
            for q in range(p, (S - p) // 2 + 1):
                r = S - p - q
                if not hyperbolic(p, q, r):
                    continue
                k = key(p, q, r)
                fib.setdefault(k, []).append((p, q, r))
                val = F(k[0], k[1])
                lo = val if lo is None or val < lo else lo
                hi = val if hi is None or val > hi else hi
            if lo is not None:
                endpoints[(S, p)] = (lo, hi)
        fibres[S] = {k: v for k, v in fib.items() if len(v) > 1}
    return endpoints, fibres


def main() -> int:
    symbolic_checks()

    # C3 for every p whose adjacent pair can occur below S_MAX, plus a long range
    for p in range(2, 5000):
        a, b = S_star_closed(p), S_star_from_cases(p)
        assert a == b, (p, a, b)
        # overlap iff S >= S*(p), checked on a window that covers both parities
        for S in range(3 * p + 3, S_star_closed(p) + 12):
            assert overlap_formula(S, p) == (S >= S_star_closed(p)), (p, S)
    print("C3: closed form S*(p) equals the case analysis for 2 <= p < 5000;"
          " overlap iff S >= S*(p) on [3p+3, S*(p)+11]")

    # C4: no overlap of any adjacent pair below 18 (and hence of any pair)
    for p in range(2, 5000):
        assert x_star(p) >= 18 and (x_star(p) == 18) == (p in (2, 3))
        assert S_star_closed(p) >= 18
    assert [p for p in range(2, 5000) if S_star_closed(p) == 18] == [2]
    print("C4: min_p S*(p) = 18, attained only at p = 2")

    endpoints, fibres = enumerate_all(S_MAX)

    # C5: brute-force endpoints equal the formulas; brute-force overlap equals C3
    n_checked = 0
    for (S, p), (lo, hi) in endpoints.items():
        assert lo == R_minus(S, p), (S, p)
        if p >= 3:
            assert hi == R_plus(S, p), (S, p)
        else:
            assert hi < 1 <= R_plus(S, p)   # stratum 2 is truncated by R < 1
        if (S, p + 1) in endpoints:
            lo1, hi1 = endpoints[(S, p + 1)]
            brute = lo <= hi1 and lo1 <= hi
            assert brute == (S >= S_star_closed(p)), (S, p, brute)
            n_checked += 1
    print(f"C5: {n_checked} adjacent (S,p) pairs with S <= {S_MAX}: brute-force"
          " interval overlap == [S >= S*(p)]")

    # pairwise disjointness of all strata below 18 directly from the data
    for S in range(10, 18):
        assert not fibres[S], S
        ivs = sorted(v for (s, _), v in endpoints.items() if s == S)
        for (l1, h1), (l2, h2) in zip(ivs, ivs[1:]):
            assert h1 < l2
    assert list(fibres[18].values()) == [[(2, 8, 8), (3, 3, 12)]]

    # C6: first collisions
    first_adj: dict[int, tuple[int, list]] = {}
    first_any: dict[tuple[int, int], int] = {}
    for S in range(10, S_MAX + 1):
        for k, trs in fibres[S].items():
            ps = sorted({t[0] for t in trs})
            for i, a in enumerate(ps):
                for b in ps[i + 1:]:
                    first_any.setdefault((a, b), S)
                if a + 1 in ps and a not in first_adj:
                    first_adj[a] = (S, [t for t in trs if t[0] in (a, a + 1)], k)
    # sanity: every collision joins different strata (Lemma lem:chamber)
    for S in fibres:
        for trs in fibres[S].values():
            assert len({t[0] for t in trs}) == len(trs)

    # cross-check against the committed degeneracy data (read only)
    gpath = HERE.parent / "diophantine" / "data" / "groups.csv"
    if gpath.exists():
        mine = {(S, k[0], k[1]): sorted(v) for S in fibres for k, v in fibres[S].items()}
        theirs = {}
        with gpath.open() as fh:
            for row in csv.DictReader(fh):
                S = int(row["S"])
                if S > S_MAX:
                    continue
                trs = sorted(tuple(int(x) for x in t.strip().strip("()").split(","))
                             for t in row["triples"].split(";"))
                theirs[(S, int(row["R_num"]), int(row["R_den"]))] = trs
        assert mine == theirs, "fibres differ from theory/diophantine/data/groups.csv"
        print(f"cross-check: all {len(mine)} collision fibres for S <= {S_MAX} equal"
              " theory/diophantine/data/groups.csv")
    else:
        print("cross-check skipped: groups.csv not present")

    rows = []
    pmax = max(p for (S, p) in endpoints if (S, p + 1) in endpoints)
    for p in range(2, pmax + 1):
        ss = S_star_closed(p)
        fa = first_adj.get(p)
        w0 = window_counts(ss, p)
        assert w0[0] >= 1 and w0[1] >= 1, (p, w0)   # both extremes lie in the window
        rows.append({
            "p": p,
            "x_star": str(x_star(p)),
            "S_star": ss,
            "first_collision_S": fa[0] if fa else "",
            "gap": fa[0] - ss if fa else "",
            "window_at_S_star": "%d+%d" % w0,
            "window_at_collision": "%d+%d" % window_counts(fa[0], p) if fa else "",
            "R": f"{fa[2][0]}/{fa[2][1]}" if fa else "",
            "triads": " ; ".join(str(t) for t in fa[1]) if fa else "",
        })
        if fa:
            assert fa[0] >= ss, (p, fa[0], ss)   # overlap is necessary
    with (HERE / "first_overlap_vs_collision.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # tangency collisions: balanced end of p = spread end of p+1 exactly
    tang = [p for p in range(2, 5000)
            if x_star(p).denominator == 1 and (int(x_star(p)) - p) % 2 == 0]
    assert tang == [2, 4], tang
    for p in tang:
        S = int(x_star(p))
        d = (S - p) // 2
        assert R_minus(S, p) == R_plus(S, p + 1)
        assert first_adj[p][0] == S
        assert sorted(first_adj[p][1]) == sorted([(p, d, d), (p + 1, p + 1, S - 2 * p - 2)])
    print("tangency (endpoint = endpoint) collisions occur exactly for p in", tang)

    # at S*(p) a collision is possible only as a tangency
    for r in rows:
        if r["first_collision_S"] == r["S_star"]:
            assert r["p"] in tang
    big = [(r["p"], r["window_at_S_star"]) for r in rows if r["window_at_S_star"] != "1+1"]
    print("window at S = S*(p) is one triad per stratum for every p except", big)
    print("first adjacent collision equals S*(p) exactly for p in",
          [r["p"] for r in rows if r["first_collision_S"] == r["S_star"]])

    print("\n p | x*(p)    | S*(p) | first adj. collision | gap | window at S*, at collision | R | triads")
    for r in rows[:30]:
        print(f"{r['p']:>2} | {r['x_star']:>8} | {r['S_star']:>5} | {str(r['first_collision_S']):>20} |"
              f" {str(r['gap']):>3} | {r['window_at_S_star']}, {r['window_at_collision']:>6} | {r['R']} | {r['triads']}")
    found = [r for r in rows if r["first_collision_S"] != ""]
    print(f"\nadjacent pairs with a collision at S <= {S_MAX}: p = 2..{max(r['p'] for r in found)}"
          f" ({len(found)} of {len(rows)} pairs whose strata coexist below {S_MAX})")
    # the paper, Theorem 1(d) and Proposition 3: the explicit odd-parity gaps
    gap = lambda S, p: R_minus(S, p) - R_plus(S, p + 1)
    assert gap(18, 3) == F(1, 840) and gap(28, 7) == F(1, 2310) and gap(31, 8) == F(1, 10296)
    assert [gap(S_star_closed(p) + 1, p) for p in range(2, 9)] == [
        F(-7, 936), F(-1, 72), F(-19, 3960), F(-1, 180), F(-76, 15015), F(-1, 231), F(-17, 4680)]
    gaps = {r["p"]: r["gap"] for r in rows if r["gap"] != ""}
    assert len(gaps) == 50 and sorted(gaps.values())[:3] == [0, 0, 8] and max(gaps.values()) == 455
    assert gaps[33] == 455 and S_star_closed(33) == 106
    print("gap range over the 50 colliding adjacent pairs, tangencies p = 2, 4 excepted: 8 ..", max(gaps.values()), "(p = 33)")
    # Prop 3(1), odd D, p <= 8: no odd sum gives an endpoint equality
    for p in range(2, 9):
        for S in range(3 * p + 3, 400):
            if (S - p) % 2 == 1:
                assert gap(S, p) != 0, (p, S)
    # Prop 3(2): window is one triad per stratum at S* = 3p+7 for p >= 9
    for p in range(9, 400):
        assert window_counts(3 * p + 7, p) == (1, 1), p
    # Prop 3(2): no collision at S*(p) for p not in {2,4}, wherever data exist
    for r in rows:
        if r["p"] not in (2, 4) and r["S_star"] <= S_MAX:
            k_at = [trs for trs in fibres[r["S_star"]].values()
                    if {r["p"], r["p"] + 1} <= {t[0] for t in trs}]
            assert not k_at, r["p"]
    print("Prop 3: odd gaps 1/840, 1/2310, 1/10296; no odd endpoint equality for p <= 8;"
          " window (1,1) at 3p+7 for 9 <= p < 400; no collision at S*(p) for p not in {2,4}")

    first_overall = min(first_any.values())
    assert first_overall == 18 and first_any[(2, 3)] == 18
    print("first collision between ANY two strata:", first_overall,
          "; first non-adjacent:", min((v, k) for k, v in first_any.items() if k[1] > k[0] + 1))
    print("ALL ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

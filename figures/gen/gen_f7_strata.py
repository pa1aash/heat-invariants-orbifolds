"""F7 data: the reciprocal-sum interval of every least-order stratum, p = 2..8, and the first
overlap and first collision of each adjacent pair.

    python3 figures/gen/gen_f7_strata.py        (a few seconds)

Writes
  figures/data/f7_strata.csv      one row per nonempty (S, p) stratum with p <= 8 and S <= S_MAX:
                                  the attained minimum and maximum of R = 1/p + 1/q + 1/r over its
                                  hyperbolic triads (= R^-_{S,p} and, for p >= 3, R^+_{S,p}; for p = 2
                                  the largest hyperbolic value, since the spread triad is not), the
                                  closed forms R^-_{S,p}, R^+_{S,p}, and the number of triads;
  figures/data/f7_overlap.csv     per p = 2..7 (adjacent pair p, p+1): x*(p), S*(p), the first
                                  collision S, its common R and its two triads.

S_MAX = 120, past the prompt's S <= 80, because the first collision of the pair (5, 6) is at S = 117.
Everything is computed by theory/threshold/threshold.py (imported, unchanged): the brute-force
enumeration enumerate_all() and the closed forms R_plus, R_minus, x_star, S_star_closed.

Asserted: brute-force endpoints equal the closed forms (p >= 3 both ends; p = 2 the lower end,
with the upper end below 1 <= R^+_{S,2}); S*(p), x*(p), the first collision, its R and triads
equal theory/threshold/first_overlap_vs_collision.csv; overlap of the brute-force intervals
happens exactly from S*(p) on; the only collision at S = 18 is {(2,8,8), (3,3,12)} at R = 3/4.
"""
from fractions import Fraction as F

from common import import_from, read_csv, write_csv

S_MAX = 120
P_MAX = 8


def main():
    th, _ = import_from("theory/threshold", "threshold")
    endpoints, fibres = th.enumerate_all(S_MAX)

    rows = []
    for (S, p), (lo, hi) in sorted(endpoints.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if p > P_MAX:
            continue
        n = sum(1 for q in range(p, (S - p) // 2 + 1) if th.hyperbolic(p, q, S - p - q))
        assert lo == th.R_minus(S, p)
        if p >= 3:
            assert hi == th.R_plus(S, p)
        else:
            assert hi < 1 <= th.R_plus(S, p)
        rows.append([S, p, str(lo), str(hi), f"{float(lo):.15g}", f"{float(hi):.15g}",
                     str(th.R_minus(S, p)), str(th.R_plus(S, p)), n])
    write_csv("f7_strata.csv", ["S", "p", "R_min", "R_max", "R_min_float", "R_max_float",
                                "R_minus_closed_form", "R_plus_closed_form", "triads"], rows)

    # first adjacent collision, from the same fibres
    first = {}
    for S in sorted(fibres):
        for k, trs in fibres[S].items():
            ps = {t[0] for t in trs}
            for p in range(2, P_MAX):
                if p not in first and {p, p + 1} <= ps:
                    first[p] = (S, F(k[0], k[1]), sorted(t for t in trs if t[0] in (p, p + 1)))
    committed = {int(r["p"]): r for r in read_csv("theory/threshold/first_overlap_vs_collision.csv")}
    ov = []
    for p in range(2, P_MAX):
        c = committed[p]
        S_star = th.S_star_closed(p)
        assert S_star == int(c["S_star"]) and str(th.x_star(p)) == c["x_star"]
        S1, R1, trs = first[p]
        assert S1 == int(c["first_collision_S"]) and str(R1) == c["R"], p
        assert " ; ".join(str(t) for t in trs) == c["triads"], p
        # the brute-force intervals of p and p+1 overlap exactly from S*(p) on
        for S in range(3 * p + 3, S_MAX + 1):
            if (S, p) in endpoints and (S, p + 1) in endpoints:
                (l0, h0), (l1, h1) = endpoints[(S, p)], endpoints[(S, p + 1)]
                assert (l0 <= h1 and l1 <= h0) == (S >= S_star), (p, S)
        ov.append([p, str(th.x_star(p)), f"{float(th.x_star(p)):.15g}", S_star, S1, str(R1),
                   f"{float(R1):.15g}", " ; ".join(str(t) for t in trs)])
    write_csv("f7_overlap.csv", ["p", "x_star", "x_star_float", "S_star", "first_collision_S",
                                 "first_collision_R", "first_collision_R_float", "triads"], ov)
    assert list(fibres[18].values()) == [[(2, 8, 8), (3, 3, 12)]] and ov[0][4] == 18 and ov[0][5] == "3/4"
    print(f"f7: {len(rows)} strata (p <= {P_MAX}, S <= {S_MAX}); first collisions "
          + ", ".join(f"p={r[0]}: S={r[4]}" for r in ov) + "; all assertions passed")


if __name__ == "__main__":
    main()

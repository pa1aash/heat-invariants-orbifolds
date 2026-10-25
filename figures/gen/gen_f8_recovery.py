"""F8 data: recovery error of the cone orders against the error in the heat coefficients.

    python3 figures/gen/gen_f8_recovery.py        (under a minute)

The recovery map is the one whose stability theory/stability certifies (threshold.py, T4):
    heat data H~ = (H~_{-1}, ..., H~_{n-2})  --L^{-1}-->  I~  --Theorem B solve-->  e~
    --roots of q~(z) = sum_j (-1)^j e~_j z^{n-j}-->  recovered orders z_1, ..., z_n  (complex).
It is assembled from stab_common.front_end, stab_common.theorem_B_system and
roots_holder.roots_from_e (imported, unchanged), in exact rational arithmetic up to the root
finder (mpmath, 60 digits). The recovery error is min over matchings of max_i |z_i - m_i|.

Three multisets: (2,3,7) (simple orders), (2,8,8) (a double order), (4,4,4) (a triple order).
Two kinds of data error:
  general     H~ = H(m) + delta * sigma, sigma in {-1,+1}^n: the same absolute error delta on
              every coefficient (the setting of threshold.py); the error plotted is the worst of
              the 2^n sign patterns. delta is log-spaced over 1e-14 .. 1e-1.
  realisable  H~ = H(m(s)) for a real multiset m(s) near m: (2,3,7+s), (2,8+s,8-s), (4+s,4,4-s);
              delta = max_nu |H_nu(m(s)) - H_nu(m)| exactly, error = s (asserted), s log-spaced.

Writes figures/data/f8_recovery.csv (one row per case, kind and step) and
figures/data/f8_thresholds.csv (delta_cert, delta_up of the three multisets, copied from
theory/stability/threshold_results.json and asserted equal to it).

Asserted: exact data recover e(m) exactly; fitted log-log slopes on the asymptotic range are
1 (simple), 1/2 (double), 1/3 (triple, general) and 1/2 (triple, realisable; double, realisable),
each within 0.01; for every delta <= delta_cert the worst error is below 1/2 (rounding recovers m),
consistent with the certificate.
"""
import itertools
import json
import math
from fractions import Fraction as Fr

from common import ROOT, import_from, write_csv

CASES = {"simple": (2, 3, 7), "double": (2, 8, 8), "triple": (4, 4, 4)}
FAMILY = {"simple": lambda s: (2, 3, 7 + s), "double": lambda s: (2, 8 + s, 8 - s), "triple": lambda s: (4 + s, 4, 4 - s)}
SLOPE = {("simple", "general"): Fr(1), ("double", "general"): Fr(1, 2), ("triple", "general"): Fr(1, 3),
         ("simple", "realisable"): Fr(1), ("double", "realisable"): Fr(1, 2), ("triple", "realisable"): Fr(1, 2)}
GENERAL_EXP = [Fr(-56 + i, 4) for i in range(53)]          # log10 delta = -14, -13.75, ..., -1
REAL_EXP = [Fr(-48 + i, 4) for i in range(45)]             # log10 s = -12, ..., -1
FIT_DELTA_MAX = 1e-8                                       # asymptotic range for the slope fit


def rational_pow10(x):
    """10^x as a Fraction with 30 significant digits (x rational)."""
    v = 10 ** float(x)
    e = math.floor(math.log10(v)) - 29
    return Fr(round(v / 10 ** e)) * Fr(10) ** e


def main():
    sc, _ = import_from("theory/stability", "stab_common")
    rh, _ = import_from("theory/stability", "roots_holder")
    mp = rh.mp

    def recover_e(H):
        n = len(H)
        L, h0 = sc.front_end(n)
        I = sc.mat_vec(sc.mat_inv(L), [a - b for a, b in zip(H, h0)])
        M, b, _ = sc.theorem_B_system(I)
        return [Fr(1)] + sc.mat_vec(sc.mat_inv(M), b)

    def recover(H):
        return rh.roots_from_e(recover_e(H))

    def error(z, m):
        return min(max(abs(z[j] - m[i]) for i, j in enumerate(perm)) for perm in itertools.permutations(range(len(m))))

    res = json.loads((ROOT / "theory/stability/threshold_results.json").read_text())
    rows, thr = [], []
    fits = {}
    for case, m in CASES.items():
        n = len(m)
        H = sc.heat_direct(m, n)
        assert recover_e(H) == sc.elementary(m), case      # exact data: exact e (roots of q are m)
        r = res[str(m)]
        thr.append([case, str(m).replace(" ", ""), r["delta_cert"], r["delta_up"]])
        # general: worst sign pattern
        pts = []
        for x in GENERAL_EXP:
            d = rational_pow10(x)
            worst = max(error(recover([h + d * sg for h, sg in zip(H, sig)]), m)
                        for sig in itertools.product((-1, 1), repeat=n))
            pts.append((float(d), float(worst)))
            rows.append([case, "general", str(m).replace(" ", ""), float(x), f"{float(d):.6e}", f"{float(worst):.10e}"])
            if d <= Fr(r["delta_cert"]):
                assert worst < 0.5, (case, float(d), float(worst))
        fits[(case, "general")] = pts
        # realisable: a real one-parameter family through m
        pts = []
        for x in REAL_EXP:
            s = rational_pow10(x)
            ms = FAMILY[case](s)
            Hs = sc.heat_direct(ms, n)
            d = max(abs(a - b) for a, b in zip(Hs, H))
            err = error(recover(Hs), m)
            assert abs(err - s) < mp.mpf(10) ** -30 * (1 + s), (case, float(s), float(err))
            pts.append((float(d), float(err)))
            rows.append([case, "realisable", str(m).replace(" ", ""), float(x), f"{float(d):.6e}", f"{float(err):.10e}"])
        fits[(case, "realisable")] = pts

    slopes = {}
    for key, pts in fits.items():
        sel = [(math.log10(d), math.log10(e)) for d, e in pts if d <= FIT_DELTA_MAX]
        assert len(sel) >= 6, key
        mx = sum(a for a, _ in sel) / len(sel)
        my = sum(b for _, b in sel) / len(sel)
        k = sum((a - mx) * (b - my) for a, b in sel) / sum((a - mx) ** 2 for a, _ in sel)
        slopes[key] = k
        assert abs(k - float(SLOPE[key])) < 0.01, (key, k)
    write_csv("f8_recovery.csv", ["case", "kind", "orders", "log10_step", "delta", "recovery_error"], rows)
    write_csv("f8_thresholds.csv", ["case", "orders", "delta_cert", "delta_up"], thr)
    print("f8: fitted slopes " + ", ".join(f"{c}/{k}: {v:.4f}" for (c, k), v in slopes.items())
          + "; all assertions passed")


if __name__ == "__main__":
    main()

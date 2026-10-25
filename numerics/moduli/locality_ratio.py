"""Two numbers quoted in the paper (Section 4 and supplement S4, S6), from committed data only.

(1) How loose the explicit bound of thm:quantlocality(a) (the bound B(l, diam, t) of lem:hypbound) with the diameter bound
    2 diam Q) is on the (0;3,3,3,3) family: for every pair of the 28, at the first time at which the
    difference of the computed traces is 10^3 times its error budget (summary.json, first_strong_t),
    the ratio B / |Z_i - Z_j| (data/trace_differences.csv). Pairs whose time lies beyond the window
    t <= l^2/(2(1+l)) of the bound are skipped (none is).
(2) The contribution of the shortest closed geodesics to the heat traces of O(2,8,8) and O(3,3,12) at
    t = 0.025 and 0.03, with weight w = 2 l (one geodesic, two orientations; supplement S4) for each of
    the three shortest lengths listed there, so that the window 0.0015 <= t <= 0.025 of Section 4 is
    one on which the geodesic term is negligible against the stated agreements (7e-13, 4e-13).

    python3 numerics/moduli/locality_ratio.py        (a second; reads data/, writes nothing)
Asserted: (1) every ratio lies in [4e6, 1.1e10] and the extremes are those quoted (4.4e6, 1.0e10); (1') with the
computed diameter 4.0e3 to 8.8e4, with the diameter-free count C_* of the companion paper 4.1e5 to 1.95e6;
(2) at t = 0.025 both contributions are below 3e-15, and at t = 0.03 that of O(3,3,12) is 7.5e-13 to
8e-13.
"""
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
AREA = 4 * math.pi / 3


def B(ell, diam, A, t):
    """Lemma 2.5 of the paper: the bound on the hyperbolic term for t <= l^2/(2(1+l))."""
    return (math.pi * math.exp(3 * diam) * ell * math.exp(ell / 2) / (A * (1 - math.exp(-ell)))
            * (1 + 2 * t / (ell - t)) * math.exp(-ell ** 2 / (4 * t)) / math.sqrt(4 * math.pi * t))


summary = json.load(open(os.path.join(DATA, "summary.json")))
geo = json.load(open(os.path.join(DATA, "geometries.json")))["members"]
rows = list(csv.DictReader(open(os.path.join(DATA, "trace_differences.csv"))))
ratios = {}
for key, pair in summary["pairs"].items():
    ta, tb = key.split("-")
    ell = min(pair["systoles"])
    diam = max(geo[ta]["diam_O_upper_bound"], geo[tb]["diam_O_upper_bound"])
    r = [x for x in rows if x["tau_i"] == ta and x["tau_j"] == tb]
    x = min(r, key=lambda x: abs(float(x["t"]) - pair["first_strong_t"]))
    t = float(x["t"])
    assert t <= ell ** 2 / (2 * (1 + ell)), key
    ratios[key] = B(ell, diam, AREA, t) / abs(float(x["D_measured"]))
lo, hi = min(ratios, key=ratios.get), max(ratios, key=ratios.get)
print(f"(1) B / |Z_i - Z_j| at the first strong time, 28 pairs: from {ratios[lo]:.2e} ({lo}) to {ratios[hi]:.2e} ({hi})")
assert len(ratios) == 28
assert all(4e6 <= v <= 1.1e10 for v in ratios.values())
assert f"{ratios[lo]:.1e}" == "4.4e+06" and f"{ratios[hi]:.1e}" == "1.0e+10"

# (1') the same ratios with two other count constants C in n_O(x) <= C e^x: the computed diameter of the
# companion paper's instances (theory/eigen/data/instances.csv, diam_true_numerical; a computed value, not a bound),
# C = pi e^{3 diam}/A; and the diameter-free count C_*(A, l, M) of the companion's lemma (eigen_common.count_constant),
# with A = 4 pi/3, l = the smaller systole lower bound, M = 3.
import sys  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "..", "theory", "eigen"))
import mpmath as mp  # noqa: E402
from eigen_common import count_constant  # noqa: E402
inst = {r["orbifold"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "..", "theory", "eigen", "data", "instances.csv")))}


def B_C(ell, logC, t):
    return (math.exp(logC) * ell * math.exp(ell / 2) / (1 - math.exp(-ell))
            * (1 + 2 * t / (ell - t)) * math.exp(-ell ** 2 / (4 * t)) / math.sqrt(4 * math.pi * t))


alt = {"computed diameter": [], "C_*": []}
for key, pair in summary["pairs"].items():
    ta, tb = key.split("-")
    ell = min(pair["systoles"])
    r = [x for x in rows if x["tau_i"] == ta and x["tau_j"] == tb]
    x = min(r, key=lambda x: abs(float(x["t"]) - pair["first_strong_t"]))
    t, d = float(x["t"]), abs(float(x["D_measured"]))
    Ia, Ib = inst[f"O_tau={float(ta):.1f}"], inst[f"O_tau={float(tb):.1f}"]
    dt = max(float(Ia["diam_true_numerical"]), float(Ib["diam_true_numerical"]))
    lb = min(float(Ia["systole_lower_bound"]), float(Ib["systole_lower_bound"]))
    alt["computed diameter"].append(B_C(ell, math.log(math.pi / AREA) + 3 * dt, t) / d)
    alt["C_*"].append(B_C(ell, float(mp.log(count_constant(mp.mpf(AREA), mp.mpf(lb), 3))), t) / d)
for k, v in alt.items():
    print(f"(1') with {k}: from {min(v):.4e} to {max(v):.4e}")
assert f"{min(alt['computed diameter']):.1e}" == "4.0e+03" and f"{max(alt['computed diameter']):.1e}" == "8.8e+04"
assert 4.05e5 <= min(alt['C_*']) < 4.15e5 and 1.9e6 <= max(alt['C_*']) < 2.0e6


def hyp(lengths, t):
    """Sum over the listed primitive lengths of w g_t(l) / (2 sinh(l/2)), w = 2 l (two orientations)."""
    return sum(2 * ell * math.exp(-t / 4) * math.exp(-ell ** 2 / (4 * t)) / math.sqrt(4 * math.pi * t)
               / (2 * math.sinh(ell / 2)) for ell in lengths)


SHORTEST = {"O(2,8,8)": (2.256768, 2.8816, 3.0571), "O(3,3,12)": (1.862604, 2.9807, 3.4027)}
for t in (0.025, 0.03):
    vals = {k: hyp(v, t) for k, v in SHORTEST.items()}
    print(f"(2) t = {t}: geodesic term " + ", ".join(f"{k} {v:.2e}" for k, v in vals.items()))
    if t == 0.025:
        assert all(v < 3e-15 for v in vals.values())
    else:
        assert 7.5e-13 <= vals["O(3,3,12)"] <= 8e-13
print("all checks passed")

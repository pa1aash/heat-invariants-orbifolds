"""Validate the closed-geodesic machinery of geodesics.py on the S3 pillows.

numerics/data/heat_trace_difference.csv holds the FEM heat traces Z(t) of the
pillows O(2,8,8) and O(3,3,12) for t in [0.0015, 0.5].  The trace formula says

    Z(t) = identity(t) + elliptic(t) + H(t),

with H the hyperbolic term (*) of geodesics.py.  The identity and elliptic terms
are evaluated by trace_formula.py, H with the enumerated weighted length spectrum.  Nothing is fitted.

Asserts:
  - the shortest lengths have integer w/l (numbers of oriented closed geodesics);
  - |Z - identity - elliptic - H| <= budget for every t in the file, where
    budget = 1e-10 (the S3 eigenvalue + tail budget is <= 3.9e-10 conservative,
    REPORT section 5, and far smaller for t >= 0.03) + the length-truncation bound.

Writes data/s3_geodesic_check.csv and prints the worst residual.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from heat_trace import T_GRID  # noqa: E402  (S3 code, unmodified)

import geodesics as G  # noqa: E402
import trace_formula as TF  # noqa: E402

L_MAX = 6.5
DATA = os.path.join(HERE, "data")


def main():
    os.makedirs(DATA, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, "..", "data", "heat_trace_difference.csv"))))
    # the CSV prints t with 8 significant digits, which alone moves Z ~ 1/(8t) by
    # ~1e-6; the exact grid is heat_trace.T_GRID, matched here row by row
    t = np.array(T_GRID, dtype=float)
    t_csv = np.array([float(r["t"]) for r in rows])
    assert len(t) == len(t_csv) and np.all(np.abs(t - t_csv) <= 1e-7 * t), "t grid mismatch"
    out_rows = []
    worst = {}
    for pqr, col in (((2, 8, 8), "Z_2_8_8"), ((3, 3, 12), "Z_3_3_12")):
        Z = np.array([float(r[col]) for r in rows])
        P = G.triangle_polygon(pqr)
        spec, ntiles = G.length_spectrum(P, L_MAX)
        for l, w in spec[:3]:
            k = w / l
            assert abs(k - round(k)) < 1e-6 and round(k) >= 1, (pqr, l, w)
        H = G.hyperbolic_term(spec, t)
        tail, Cg = G.tail_bound(spec, L_MAX, None, t)
        area = 2 * np.pi * (1 - sum(1.0 / m for m in pqr))
        S = TF.identity_plus_elliptic(area, pqr, t)
        resid = Z - S - H
        budget = 1e-10 + tail
        ok = np.abs(resid) <= budget
        assert ok.all(), (pqr, t[~ok], resid[~ok], budget[~ok])
        worst[pqr] = (float(np.max(np.abs(resid))), float(np.max(H)), len(spec), ntiles)
        print(f"{pqr}: {ntiles} tiles, {len(spec)} lengths <= {L_MAX}; shortest "
              + ", ".join(f"{l:.6f} (x{w / l:.3f})" for l, w in spec[:4])
              + f"; max H = {np.max(H):.3e}; max |Z - id - ell - H| = {np.max(np.abs(resid)):.2e}"
              + f" (max |Z - id - ell| = {np.max(np.abs(Z - S)):.2e})")
        for i in range(len(t)):
            out_rows.append(["-".join(map(str, pqr)), f"{t[i]:.8g}", f"{Z[i]:.16e}", f"{S[i]:.16e}",
                             f"{H[i]:.6e}", f"{resid[i]:.3e}", f"{tail[i]:.3e}"])
    with open(os.path.join(DATA, "s3_geodesic_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pillow", "t", "Z_fem_S3", "identity_plus_elliptic", "H_geodesic_prediction",
                    "residual", "length_truncation_bound"])
        w.writerows(out_rows)
    print("S3 GEODESIC CHECK PASSED")
    return worst


if __name__ == "__main__":
    main()

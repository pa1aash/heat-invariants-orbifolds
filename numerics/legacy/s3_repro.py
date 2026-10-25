"""Cross-machine reproducibility check: rerun one S3 problem on this machine and
compare with the committed eigenvalues.

Problem: the (3,3,12) triangle, Neumann, at the S3 production level
(h, p) = (0.05, 10), with the S3 slicing parameters (NEV = 1300, k = 200),
solved by the OLD single-window routine (numerics/legacy/single_window.py;
LEGACY, no longer used).  Reference:
numerics/data/eigenvalues_3-3-12_N.csv (15 significant digits).

Asserts (the precision S3 states, the supplement:
  - same number of eigenvalues up to the S3 count, lambda_0 = 0;
  - |lambda_here - lambda_S3| <= err_conservative + 1e-13 lambda for every eigenvalue
    (the per-eigenvalue worst-case error S3 publishes);
  - relative difference < 1e-8 for the first 500 (the S3 target).
Reports max |difference| / err_estimate as well (not asserted: the estimate is a
two-discretisation agreement, not a bound, and a different mesher round-off can
legitimately move eigenvalues at that level).

usage: python s3_repro.py [threads]      writes data/s3_repro.json
"""

import csv
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = os.path.join(HERE, "..")
MODULI = os.path.join(NUM, "moduli")
sys.path.insert(0, NUM)
sys.path.insert(0, HERE)


def main(threads=16):
    import ngsolve as ngs
    from single_window import run_single_window as run      # the old S3 single-window solver
    ngs.SetNumThreads(threads)
    ref = list(csv.DictReader(open(os.path.join(NUM, "data", "eigenvalues_3-3-12_N.csv"))))
    lam_ref = np.array([float(r["lambda"]) for r in ref])
    err = np.array([float(r["err_estimate"]) for r in ref])
    err_c = np.array([float(r["err_conservative"]) for r in ref])
    t0 = time.time()
    lam, S = run((3, 3, 12), "N", 0.05, 10, 1300, k=200, verbose=True)
    secs = time.time() - t0
    n = len(lam_ref)
    assert len(lam) >= n, (len(lam), n)
    lam = lam[:n].copy()
    assert abs(lam[0]) < 1e-8
    lam[0] = 0.0
    d = np.abs(lam - lam_ref)
    rel = d / np.maximum(lam_ref, 1)
    assert np.all(d <= err_c + 1e-13 * np.maximum(lam_ref, 1)), np.max(d - err_c)
    assert np.max(rel[:500]) < 1e-8, np.max(rel[:500])
    ratio = d[1:] / np.maximum(err[1:], 1e-300)
    out = dict(problem="(3,3,12) Neumann, h=0.05, p=10, NEV=1300, k=200", n=int(n),
               ndof=int(S["A"].shape[0]), max_abs_diff=float(d.max()), max_rel_diff=float(rel.max()),
               max_rel_diff_first500=float(rel[:500].max()),
               max_diff_over_err_estimate=float(ratio.max()), median_diff_over_err_estimate=float(np.median(ratio)),
               max_diff_over_err_conservative=float((d[1:] / np.maximum(err_c[1:], 1e-300)).max()),
               seconds=secs, threads=threads, machine=platform.platform(), python=platform.python_version())
    os.makedirs(os.path.join(MODULI, "data"), exist_ok=True)
    with open(os.path.join(MODULI, "data", "s3_repro.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("S3 REPRODUCIBILITY CHECK PASSED")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)

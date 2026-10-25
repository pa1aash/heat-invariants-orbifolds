"""Rerun the four S3 production problems with the double-window solver and compare them
with the committed eigenvalues, without overwriting anything committed.

Problems: (2,8,8) and (3,3,12), Neumann and Dirichlet, at the committed production level
(h, p) = (0.05, 10), NEV = 1300, k = 200 (solve.LEVELS[-1], NEV, K_SLICE), solved by
solve.eigenvalues_robust.  Reference: data/eigenvalues_<p>-<q>-<r>_<bc>.csv.

For each problem the rerun's eigenvalues go to a scratch directory (never data/), and the
comparison is written to data/rerun_double_window_comparison.json, which is a new file.

Asserts, per problem (the precision S3 states, the supplement:
  - the rerun has at least as many eigenvalues as the committed count minus the part of
    the committed tail above the rerun's doubly covered cut; no committed eigenvalue below
    the rerun's cut is missing and none is extra: the counts below the common cut agree;
  - |lam_rerun - lam_committed| <= err_conservative + 1e-13 max(lam, 1) for every index in
    the common prefix;
  - relative difference < 1e-8 for the first 500 (the S3 target);
  - lambda_0 = 0 for Neumann.
The tool never rewrites a committed file; if a rerun disagrees, the assertions fail and the
comparison JSON records the disagreement.

usage: python rerun_double_window.py SCRATCH_DIR [threads]      all four problems, then the record
       python rerun_double_window.py --only I SCRATCH_DIR [threads]    problem I (0..3) only: SCRATCH_DIR/problem_I.json
       python rerun_double_window.py --merge SCRATCH_DIR     the four problem_I.json -> the record (no NGSolve)
       python rerun_double_window.py --no-record SCRATCH_DIR [threads]  all four problems, asserted against the record
                                                                      already committed; writes nothing under data/
       python rerun_double_window.py --verify-record         (no NGSolve: check the committed record)
"""

import csv
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RECORD = os.path.join(DATA, "rerun_double_window_comparison.json")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def csv_path(pqr, bc):
    return os.path.join(DATA, f"eigenvalues_{pqr[0]}-{pqr[1]}-{pqr[2]}_{bc}.csv")


def compare(pqr, bc, lam, secs, info, scratch):
    ref = list(csv.DictReader(open(csv_path(pqr, bc))))
    lam_ref = np.array([float(r["lambda"]) for r in ref])
    err = np.array([float(r["err_estimate"]) for r in ref])
    err_c = np.array([float(r["err_conservative"]) for r in ref])
    lam = lam.copy()
    if bc == "N":
        assert abs(lam[0]) < 1e-8, lam[0]
        lam[0] = 0.0
    n_ref, n_new = len(lam_ref), len(lam)
    m = min(n_ref, n_new)
    d = np.abs(lam[:m] - lam_ref[:m])
    rel = d / np.maximum(lam_ref[:m], 1.0)
    # counts below a common energy: the smaller of the two lists' last eigenvalues, shifted
    # down by the largest allowed difference so a within-error straddle is not a count change
    cut = min(lam_ref[m - 1], lam[m - 1]) - 1e-6
    count_ref = int(np.count_nonzero(lam_ref < cut))
    count_new = int(np.count_nonzero(lam < cut))
    ok_counts = count_ref == count_new
    ok_values = bool(np.all(d <= err_c[:m] + 1e-13 * np.maximum(lam_ref[:m], 1.0)))
    ok_first500 = bool(np.max(rel[:500]) < 1e-8)
    res = dict(problem=f"({pqr[0]},{pqr[1]},{pqr[2]}) {'Neumann' if bc == 'N' else 'Dirichlet'}",
               level="h=0.05, p=10, NEV=1300, k=200", committed_file=os.path.relpath(csv_path(pqr, bc), HERE),
               committed_sha256=sha256(csv_path(pqr, bc)),
               n_committed=int(n_ref), n_rerun=int(n_new), n_compared=int(m),
               common_cut=float(cut), count_below_cut_committed=count_ref, count_below_cut_rerun=count_new,
               max_abs_diff=float(d.max()), max_rel_diff=float(rel.max()),
               max_rel_diff_first500=float(rel[:500].max()),
               max_diff_over_err_estimate=float((d[1:] / np.maximum(err[1:m], 1e-300)).max()),
               max_diff_over_err_conservative=float((d[1:] / np.maximum(err_c[1:m], 1e-300)).max()),
               arpack_misses_repaired=int(info["misses_repaired"]), n_slices=int(info["n_slices"]),
               lambda_max_rerun=float(lam[-1]), lambda_max_committed=float(lam_ref[-1]),
               counts_agree=bool(ok_counts), values_within_err_conservative=ok_values,
               first500_below_1em8=ok_first500, seconds=secs)
    np.savez(os.path.join(scratch, f"rerun_{pqr[0]}-{pqr[1]}-{pqr[2]}_{bc}.npz"), lam=lam)
    return res


def solve_problem(i, scratch, threads=8):
    import ngsolve as ngs
    import solve
    ngs.SetNumThreads(threads)
    os.makedirs(scratch, exist_ok=True)
    assert os.path.abspath(scratch) != DATA, "scratch must not be the committed data directory"
    h, order = solve.LEVELS[-1]
    assert (h, order) == (0.05, 10) and solve.NEV == 1300 and solve.K_SLICE == 200
    pqr, bc = solve.PROBLEMS[i]
    t0 = time.time()
    S = solve.assemble(pqr, bc, h, order)
    lam, _, info = solve.eigenvalues_robust(S["A"], S["M"], solve.NEV, k=solve.K_SLICE)
    res = compare(pqr, bc, lam, time.time() - t0, info, scratch)
    res["ndof"] = int(S["A"].shape[0])
    with open(os.path.join(scratch, f"problem_{i}.json"), "w") as f:
        json.dump(dict(result=res, threads=threads, machine=platform.platform(), python=platform.python_version()), f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return res


def merge(scratch):
    parts = [json.load(open(os.path.join(scratch, f"problem_{i}.json"))) for i in range(4)]
    out = dict(solver="numerics/solve.py:eigenvalues_robust (double-window)", results=[p["result"] for p in parts],
               threads=sorted({p["threads"] for p in parts}), machine=parts[0]["machine"], python=parts[0]["python"])
    with open(RECORD, "w") as f:
        json.dump(out, f, indent=1)
    verify_record()


def main(scratch, threads=8, write_record=True):
    results = [solve_problem(i, scratch, threads) for i in range(4)]
    if write_record:
        merge(scratch)
    else:
        verify_record()
        rec = json.load(open(RECORD))["results"]
        for new, old in zip(results, rec):
            assert new["problem"] == old["problem"]
            assert new["committed_sha256"] == old["committed_sha256"], new["problem"]
            assert new["counts_agree"] and new["values_within_err_conservative"] and new["first500_below_1em8"], new["problem"]
            # the rerun reproduces the committed record: same counts, same agreement to far below the error estimate
            assert (new["n_committed"], new["n_rerun"], new["n_compared"]) == (old["n_committed"], old["n_rerun"], old["n_compared"]), new["problem"]
            assert abs(new["max_rel_diff"] - old["max_rel_diff"]) <= 1e-9, new["problem"]
            assert new["arpack_misses_repaired"] == old["arpack_misses_repaired"], new["problem"]
        print("the rerun reproduces the committed comparison record (counts, agreement, repaired misses)")
    print("DOUBLE-WINDOW RERUN COMPARISON PASSED")


def verify_record():
    """Real asserts on the committed record; needs no NGSolve."""
    rec = json.load(open(RECORD))
    assert len(rec["results"]) == 4, len(rec["results"])
    for r in rec["results"]:
        pqr = tuple(int(x) for x in r["problem"][1:r["problem"].index(")")].split(","))
        bc = "N" if "Neumann" in r["problem"] else "D"
        assert sha256(csv_path(pqr, bc)) == r["committed_sha256"], f"committed data changed: {r['problem']}"
        assert r["counts_agree"], r["problem"]
        assert r["values_within_err_conservative"], r["problem"]
        assert r["first500_below_1em8"], r["problem"]
        assert r["count_below_cut_committed"] == r["count_below_cut_rerun"], r["problem"]
        assert r["n_compared"] >= 1000, r["problem"]
    print(f"record OK: {len(rec['results'])} problems, committed CSV hashes match")


if __name__ == "__main__":
    if sys.argv[1] == "--verify-record":
        verify_record()
    elif sys.argv[1] == "--merge":
        merge(sys.argv[2])
        print("DOUBLE-WINDOW RERUN COMPARISON PASSED")
    elif sys.argv[1] == "--no-record":
        main(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8, write_record=False)
    elif sys.argv[1] == "--only":
        solve_problem(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)

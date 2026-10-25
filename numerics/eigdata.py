"""Load eigenvalue runs and attach per-eigenvalue error estimates.

Production level = last entry of solve.LEVELS, (h, p) = (0.05, 10).

Two error measures per production eigenvalue, both floored at 1e-14 * lambda:

  err       |lambda(0.05,10) - lambda(0.07,12)|: disagreement of two independent
            discretisations, each far more accurate than either coarser level
            (they agree to ~1e-11 relative where the coarser levels differ by
            ~1e-8).  This is the headline estimate.
  err_cons  |lambda(0.05,10) - lambda(0.07,10)|: the difference to the same
            order on the coarser mesh.  Convergence is monotone and exponential,
            so this bounds the production error with a large margin (it is
            essentially the error of the coarser level).  Propagated separately
            as a worst case.
"""

import os

import numpy as np

from solve import LEVELS, run_path


def load_level(pqr, bc, h, order):
    d = np.load(run_path(pqr, bc, h, order), allow_pickle=True)
    return d["lam"], d


def table_committed(pqr, bc, nmax=None):
    """The same dict from the committed data/eigenvalues_<p-q-r>_<bc>.csv (production
    eigenvalue and both error measures; no raw runs needed).  'all' holds the three levels
    the CSV keeps: production, (0.07, 12), (0.07, 10)."""
    import csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        f"eigenvalues_{pqr[0]}-{pqr[1]}-{pqr[2]}_{bc}.csv")
    rows = list(csv.DictReader(open(path)))
    if nmax is not None:
        rows = rows[:nmax]
    col = lambda k: np.array([float(r[k]) for r in rows])
    return dict(lam=col("lambda"), err=col("err_estimate"), err_cons=col("err_conservative"),
                levels=[(0.05, 10), (0.07, 12), (0.07, 10)],
                all=[col("lambda"), col("lambda_h0.07_p12"), col("lambda_h0.07_p10")])


def table(pqr, bc, levels=LEVELS, nmax=None):
    """Production eigenvalues with error estimates: from the raw runs when they are present,
    otherwise from the committed CSV (so that checks on committed data need no solver run)."""
    if not available(pqr, bc, levels):
        return table_committed(pqr, bc, nmax)
    lams = []
    for h, order in levels:
        lam, _ = load_level(pqr, bc, h, order)
        lams.append(lam)
    n = min(len(l) for l in lams)
    if nmax is not None:
        n = min(n, nmax)
    lams = [l[:n] for l in lams]
    lv = [tuple(x) for x in levels]
    prod = lams[-1]
    floor = 1e-14 * np.maximum(np.abs(prod), 1.0)
    err = np.maximum(np.abs(prod - lams[lv.index((0.07, 12))]), floor)
    err_cons = np.maximum(np.abs(prod - lams[-2]), floor)
    if bc == "N":
        # the constant eigenfunction: lambda_0 = 0 exactly
        assert abs(prod[0]) < 1e-8, prod[0]
        prod = prod.copy()
        prod[0] = 0.0
        err[0] = 0.0
        err_cons = err_cons.copy()
        err_cons[0] = 0.0
    return dict(lam=prod, err=err, err_cons=err_cons, levels=list(levels), all=lams)


def available(pqr, bc, levels=LEVELS):
    return all(os.path.exists(run_path(pqr, bc, h, o)) for h, o in levels)

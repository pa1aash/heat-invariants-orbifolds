"""LEGACY, NOT USED BY ANYTHING UNDER numerics/ (kept for the record).

The single-window slicing routine of the S3 solver.  Each shift-invert Lanczos slice
certifies the window (sigma - rho, sigma + rho) from the k values it returns, so an
eigenvalue that ARPACK silently drops inside a window is lost with no warning.  It was
replaced by the doubly covered solver numerics/solve.py:eigenvalues_robust, which covers
every eigenvalue by two independent windows.  The committed S3 data
(numerics/data/eigenvalues_*.csv) were produced with this routine; the rerun with the
replacement is recorded in numerics/data/rerun_double_window_comparison.json.
"""

import time

import numpy as np

from solve import slice_eigs


def eigenvalues_single_window(A, M, nev, k=160, want_vectors=False, verbose=True):
    """Complete sorted list of the eigenvalues below the end of the slice that
    first certifies at least nev of them.

    Slice i (shift sigma_i, radius rho_i) certifies every eigenvalue in
    (sigma_i - rho_i, sigma_i + rho_i).  Consecutive windows must overlap; the
    cut between slices i and i+1 is the midpoint of their overlap and slice i
    contributes exactly the eigenvalues in [cut_{i-1}, cut_i)."""
    slices = []
    sigma = -1.0
    while True:
        t0 = time.time()
        vals, vecs = slice_eigs(A, M, sigma, k, want_vectors)
        rho = float(np.max(np.abs(vals - sigma)))
        tries = 0
        while slices and not sigma - rho < slices[-1][0] + slices[-1][1]:
            # window too narrow for this shift: move the shift back so that the
            # new window starts inside the previous one, and redo the slice
            tries += 1
            if tries > 8:
                raise RuntimeError(f"slice windows do not overlap at sigma={sigma}")
            sigma = slices[-1][0] + slices[-1][1] + 0.7 * rho
            vals, vecs = slice_eigs(A, M, sigma, k, want_vectors)
            rho = float(np.max(np.abs(vals - sigma)))
        slices.append((sigma, rho, vals, vecs))
        if verbose:
            print(f"  slice sigma={sigma:10.2f} window=({sigma-rho:9.2f},{sigma+rho:9.2f})"
                  f"  {time.time()-t0:6.1f}s", flush=True)
        n_cert = np.count_nonzero(np.concatenate([s[2] for s in slices]) < sigma + rho)
        # (an over-count of the certified total because of overlaps; exact count below)
        cuts = [-np.inf] + [0.5 * ((slices[i + 1][0] - slices[i + 1][1]) + (slices[i][0] + slices[i][1]))
                            for i in range(len(slices) - 1)] + [sigma + rho]
        n_cert = sum(np.count_nonzero((s[2] >= cuts[i]) & (s[2] < cuts[i + 1]))
                     for i, s in enumerate(slices))
        if n_cert >= nev:
            break
        sigma = sigma + 1.5 * rho
    lam, vec = [], []
    for i, s in enumerate(slices):
        sel = (s[2] >= cuts[i]) & (s[2] < cuts[i + 1])
        lam.append(s[2][sel])
        if want_vectors:
            vec.append(s[3][:, sel])
    lam = np.concatenate(lam)
    order = np.argsort(lam)
    if want_vectors:
        return lam[order], np.concatenate(vec, axis=1)[:, order]
    return lam[order], None



def run_single_window(pqr, bc, h, order, nev, k=160, verbose=True):
    """The old numerics/solve.run: assemble, then the single-window eigenvalues."""
    from solve import assemble
    S = assemble(pqr, bc, h, order)
    lam, _ = eigenvalues_single_window(S["A"], S["M"], nev, k=k, verbose=verbose)
    if bc == "N":
        lam[0] = max(lam[0], 0.0) if abs(lam[0]) < 1e-8 else lam[0]
    return lam, S

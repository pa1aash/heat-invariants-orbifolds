"""High-order FEM eigenvalues of a hyperbolic triangle, Neumann or Dirichlet.

    -Delta_E u = lambda w(z) u  on the triangle T(p,q,r) in the Poincare disk,
    w(z) = 4 / (1 - |z|^2)^2,
discretised with NGSolve H1 elements of order `order` on a mesh of uniform
hyperbolic size `h` (Euclidean size h (1-|z|^2)/2), with the geodesic side
represented exactly (rational quadratic spline) and curved to the element order.

Many eigenvalues are obtained by doubly covered spectral slicing: shift-invert
Lanczos (ARPACK) at a sequence of shifts sigma_i; each slice returns the k
eigenvalues nearest sigma_i.  Every eigenvalue below the final cut lies in the
inner part of at least two independent windows, and values seen by fewer than all
of their covering windows are ARPACK misses, repaired by the union and counted
(eigenvalues_robust).  This is the only eigenvalue routine used by numerics/; the
earlier single-window routine is kept, unused, in legacy/single_window.py.

usage:  python solve.py P Q R {N|D} --h H --order P --nev N [--out file.npz]
"""

import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from geometry import Triangle, make_mesh, weight_cf


# Dirichlet sides per boundary condition.  "N" and "D" are the pillow's even and
# odd parts; the two mixed conditions are used only in the (2,3,8) benchmark,
# where they are the remaining one-dimensional characters of the reflection
# group (sides ab and bc meet at the pi/3 corner, so their signs must agree).
DIRICHLET_SIDES = {"N": "", "D": "ab|bc|ac", "M1": "ab|bc", "M2": "ac"}


def assemble(pqr, bc, h, order, geom_order=None):
    import ngsolve as ngs
    T = Triangle(*pqr)
    mesh = make_mesh(T, h, geom_order or order)
    fes = ngs.H1(mesh, order=order, dirichlet=DIRICHLET_SIDES[bc])
    u, v = fes.TnT()
    w = weight_cf()
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()
    # w is a rational function: give the quadrature extra order
    m = ngs.BilinearForm(w * u * v * ngs.dx(bonus_intorder=2 * order)).Assemble()
    fd = np.array(list(fes.FreeDofs()), dtype=bool)
    A = sp.csr_matrix(a.mat.CSR())[fd][:, fd]
    M = sp.csr_matrix(m.mat.CSR())[fd][:, fd]
    A = (A + A.T) * 0.5
    M = (M + M.T) * 0.5
    area = ngs.Integrate(w * ngs.dx(bonus_intorder=2 * order), mesh)
    return dict(A=A.tocsc(), M=M.tocsc(), mesh=mesh, fes=fes, freedofs=fd,
                area_fem=area, area_exact=float(T.area_exact), ne=mesh.ne)


def slice_eigs(A, M, sigma, k, want_vectors=False):
    """k eigenvalues of A x = lam M x nearest sigma, via one LU of A - sigma M."""
    n = A.shape[0]
    lu = sla.splu((A - sigma * M).tocsc(), permc_spec="COLAMD")
    op = sla.LinearOperator((n, n), matvec=lu.solve, dtype=float)
    out = sla.eigsh(A, k=k, M=M, sigma=sigma, which="LM", OPinv=op,
                    return_eigenvectors=want_vectors, tol=0)
    if want_vectors:
        vals, vecs = out
        idx = np.argsort(vals)
        return vals[idx], vecs[:, idx]
    return np.sort(out), None


K_SLICE = 200


def eigenvalues_robust(A, M, nev, k=K_SLICE, want_vectors=False, inner=0.8, rtol=1e-9):
    """Doubly covered spectral slicing.

    This is the only eigenvalue routine used by numerics/.  The single-window routine it
    replaces (numerics/legacy/single_window.py) certifies the window (sigma - rho, sigma + rho)
    of one shift-invert Lanczos run from the k values it returns; ARPACK was found to drop an
    eigenvalue inside such a window (detected by comparing mesh levels; the supplement
    section 4a).  Here every eigenvalue below the final cut lies in the inner part
    (radius inner * rho) of at least TWO independent windows, the next shift being placed at
    the centre-plus-half-radius of the previous one.  Values are clustered (relative
    tolerance rtol); a cluster's multiplicity is the largest count any single covering
    window reports.  Clusters seen by fewer than all of their covering windows are ARPACK
    misses repaired by the union; their number is returned.

    Returns (lam, vecs or None, info)."""
    slices = []
    sigma = -1.0
    while True:
        vals, vecs = slice_eigs(A, M, sigma, k, want_vectors)
        rho = float(np.max(np.abs(vals - sigma)))
        lo, hi = sigma - inner * rho, sigma + inner * rho
        tries = 0
        while slices and lo > slices[-1][0]:
            # the new inner window must start below the centre of the previous one, so that
            # every point is covered twice: pull the shift back and redo
            tries += 1
            if tries > 10:
                raise RuntimeError(f"cannot keep double coverage at sigma={sigma}")
            sigma = slices[-1][0] + 0.5 * (sigma - slices[-1][0])
            vals, vecs = slice_eigs(A, M, sigma, k, want_vectors)
            rho = float(np.max(np.abs(vals - sigma)))
            lo, hi = sigma - inner * rho, sigma + inner * rho
        slices.append((sigma, rho, lo, hi, vals, vecs))
        cut = hi if len(slices) >= 2 else -np.inf
        # every value below `cut` and above the first window's lower edge is covered twice
        if len(slices) >= 2:
            # distinct values below the doubly covered top, counted from the windows that own them
            top_ = slices[-2][3]
            nd = sum(np.count_nonzero((x[4] >= max(x[2], -np.inf)) & (x[4] < min(x[3], top_)) &
                                      (x[4] >= (slices[i - 1][3] if i > 0 else -np.inf)))
                     for i, x in enumerate(slices))
            if nd >= nev:
                break
        sigma = sigma + 0.5 * inner * rho          # next centre inside this inner window
        if len(slices) > 400:
            raise RuntimeError("too many slices")
    top = slices[-2][3]                            # below this every point has two windows
    # cluster all inner values below top
    allv = []
    for i, (sg, rh, lo, hi, vals, vecs) in enumerate(slices):
        for j, v in enumerate(vals):
            if lo <= v < hi and v < top:
                allv.append((v, i, j))
    allv.sort()
    clusters = []
    for v, i, j in allv:
        if clusters and abs(v - clusters[-1][-1][0]) <= rtol * max(abs(v), 1.0):
            clusters[-1].append((v, i, j))
        else:
            clusters.append([(v, i, j)])
    lam, vec_idx, misses = [], [], 0
    for cl in clusters:
        v0 = cl[0][0]
        cover = [i for i, s in enumerate(slices) if s[2] <= v0 < s[3]]
        counts = {i: sum(1 for (_, ii, _) in cl if ii == i) for i in cover}
        mult = max(counts.values())
        if min(counts.values()) < mult:
            misses += 1
        best = max(cover, key=lambda i: (counts[i], -abs(v0 - slices[i][0])))   # most central full copy
        mine = sorted([(v, j) for (v, ii, j) in cl if ii == best])
        for v, j in mine[:mult]:
            lam.append(v)
            vec_idx.append((best, j))
    lam = np.array(lam)
    order = np.argsort(lam)
    lam = lam[order]
    info = dict(n_slices=len(slices), top=top, misses_repaired=misses)
    if want_vectors:
        V = np.column_stack([slices[i][5][:, j] for (i, j) in vec_idx])[:, order]
        return lam, V, info
    return lam, None, info


def eigenvalues(A, M, nev, k=K_SLICE, want_vectors=False, verbose=False):
    """(lam, vecs or None) of eigenvalues_robust; the signature the callers use."""
    lam, vecs, info = eigenvalues_robust(A, M, nev, k=k, want_vectors=want_vectors)
    if verbose:
        print(f"  {len(lam)} eigenvalues in {info['n_slices']} slices, "
              f"{info['misses_repaired']} ARPACK misses repaired", flush=True)
    return lam, vecs


def run(pqr, bc, h, order, nev, k=K_SLICE, verbose=True):
    t0 = time.time()
    S = assemble(pqr, bc, h, order)
    if verbose:
        print(f"{pqr} {bc} h={h} p={order}: ne={S['ne']} ndof={S['A'].shape[0]} "
              f"area_err={S['area_fem']-S['area_exact']:.2e} "
              f"assembly {time.time()-t0:.1f}s", flush=True)
    lam, _ = eigenvalues(S["A"], S["M"], nev, k=k, verbose=verbose)
    if bc == "N":
        lam[0] = max(lam[0], 0.0) if abs(lam[0]) < 1e-8 else lam[0]
    if verbose:
        print(f"  {len(lam)} eigenvalues, max {lam[-1]:.3f}, total {time.time()-t0:.1f}s", flush=True)
    return lam, S


# Convergence levels (hyperbolic mesh size h, polynomial order p).  The last
# level is the production level; the one before it is the same order on a
# coarser mesh, and their difference is the (conservative) error estimate.
# The others give an h-sweep at p = 10 (h = 0.1, 0.07, 0.05) and a p-sweep at
# h = 0.07 (p = 8, 10, 12) for the convergence rates.
LEVELS = [(0.1, 10), (0.07, 8), (0.07, 12), (0.07, 10), (0.05, 10)]
PROBLEMS = [((2, 8, 8), "N"), ((2, 8, 8), "D"), ((3, 3, 12), "N"), ((3, 3, 12), "D")]
NEV = 1300           # per triangle and boundary condition
BENCH_LEVELS = [(0.1, 8), (0.07, 10)]
BENCH_NEV = 60       # (2,3,8): covers the Bolza list (lambda < 998) with margin
RUNDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def run_path(pqr, bc, h, order):
    return os.path.join(RUNDIR, f"eig_{pqr[0]}-{pqr[1]}-{pqr[2]}_{bc}_h{h}_p{order}.npz")


def suite(problems, levels, nev, k):
    os.makedirs(RUNDIR, exist_ok=True)
    for pqr, bc in problems:
        for h, order in levels:
            out = run_path(pqr, bc, h, order)
            if os.path.exists(out):
                print("exists:", out, flush=True)
                continue
            t0 = time.time()
            lam, S = run(pqr, bc, h, order, nev, k=k)
            np.savez(out, lam=lam, pqr=pqr, bc=bc, h=h, order=order,
                     ndof=S["A"].shape[0], ne=S["ne"],
                     area_err=S["area_fem"] - S["area_exact"], seconds=time.time() - t0)


if __name__ == "__main__":
    import sys
    import ngsolve as ngs
    if len(sys.argv) > 1 and sys.argv[1] in ("suite", "bench"):
        ngs.SetNumThreads(8)
        if sys.argv[1] == "suite":
            suite(PROBLEMS, LEVELS, NEV, k=200)
        else:
            suite([((2, 3, 8), bc) for bc in ("N", "D", "M1", "M2")], BENCH_LEVELS, BENCH_NEV, k=40)
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("p", type=int)
    ap.add_argument("q", type=int)
    ap.add_argument("r", type=int)
    ap.add_argument("bc", choices=list(DIRICHLET_SIDES))
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--order", type=int, required=True)
    ap.add_argument("--nev", type=int, default=1000)
    ap.add_argument("--k", type=int, default=K_SLICE)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    ngs.SetNumThreads(a.threads)
    lam, S = run((a.p, a.q, a.r), a.bc, a.h, a.order, a.nev, k=a.k)
    if a.out:
        np.savez(a.out, lam=lam, pqr=(a.p, a.q, a.r), bc=a.bc, h=a.h, order=a.order,
                 ndof=S["A"].shape[0], ne=S["ne"], area_err=S["area_fem"] - S["area_exact"])

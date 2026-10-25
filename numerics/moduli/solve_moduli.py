"""High-order FEM spectra of the doubled quadrilaterals O(tau), sector by sector.

Symmetry reduction.  O(tau) is the double of Q(tau), so
spec(O) = spec_N(Q) u spec_D(Q) (S3 doubling principle, extended to polygons).
Q has the mirror symmetries z -> conj z and z -> -conj z; each of spec_N(Q),
spec_D(Q) is the union of four mixed problems on the Lambert quarter L:

    sector  "<outer><x><y>",  outer = BC on the sides e, n of L (the sides of Q),
                              x, y  = BC on the mirror segments O-P (real axis)
                                      and Y-O (imaginary axis);
            N = Neumann (even), D = Dirichlet (odd).

So spec(O) is the union over the eight sectors NNN, NND, NDN, NDD, DNN, DND, DDN, DDD.
Every corner of L is regular for every sector (right angles with any pair of
conditions: exponents (2k+1) or 2k; the pi/3 corner carries the same condition on
both sides: exponents 3k), so the eigenfunctions extend smoothly across corners,
as for the S3 triangles.

Discretisation: identical to S3 (numerics/solve.py): -Delta_E u = lambda w u,
w = 4/(1-|z|^2)^2, NGSolve H1 elements of order p, mesh of uniform hyperbolic
size h, both arcs exact rational quadratic splines curved to order p; the
spectrum by doubly covered shift-invert slicing (solve.eigenvalues_robust, the one solver
of numerics/; moved there from this file without change).

usage:
  python solve_moduli.py suite [workers]          all members x sectors x levels
  python solve_moduli.py fullquad [workers]       direct full-Q check (tau = 0.8)
  python solve_moduli.py one TAU SECTOR H P NEV   single run, printed
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from quad import Lambert, TAUS  # noqa: E402
from solve import eigenvalues_robust, K_SLICE  # noqa: E402  (the one solver, numerics/solve.py)

SECTORS = ["NNN", "NND", "NDN", "NDD", "DNN", "DND", "DDN", "DDD"]
LEVELS = [(0.1, 10), (0.07, 8), (0.07, 12), (0.07, 10), (0.05, 10)]   # last = production
NEV = 700            # per sector: lambda_max ~ 1.6e4
RUNDIR = os.path.join(HERE, "runs", "robust")


def dirichlet_sides(sector):
    outer, x, y = sector
    s = []
    if outer == "D":
        s += ["e", "n"]
    if x == "D":
        s.append("x")
    if y == "D":
        s.append("y")
    return "|".join(s)


def make_geometry(L):
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    f = lambda z: (float(z.real), float(z.imag))
    O, P, V, Y = (0.0, 0.0), f(L.P), f(L.V), f(L.Y)
    cE = L.arc_control(L.cE, L.P, L.V)
    cN = L.arc_control(L.cN, L.V, L.Y)
    pO, pP, pE, pV, pN, pY = (geo.AppendPoint(*q) for q in (O, P, cE, V, cN, Y))
    geo.Append(["line", pO, pP], bc="x", leftdomain=1, rightdomain=0)
    geo.Append(["spline3", pP, pE, pV], bc="e", leftdomain=1, rightdomain=0)
    geo.Append(["spline3", pV, pN, pY], bc="n", leftdomain=1, rightdomain=0)
    geo.Append(["line", pY, pO], bc="y", leftdomain=1, rightdomain=0)
    return geo


def make_mesh(L, h_hyp, order_geom):
    """Uniform hyperbolic mesh size: local Euclidean size h (1 - |z|^2)/2."""
    import ngsolve as ngs
    from netgen.meshing import MeshingParameters
    geo = make_geometry(L)
    mp_ = MeshingParameters(maxh=h_hyp / 2, grading=0.2)
    xmax = max(float(L.P.real), float(L.V.real))
    ymax = max(float(L.Y.imag), float(L.V.imag))
    n = 80
    for i in range(n + 1):
        for j in range(n + 1):
            x, y = xmax * i / n, ymax * j / n
            mp_.RestrictH(x=x, y=y, z=0, h=h_hyp * (1 - x * x - y * y) / 2)
    for c, r, z1, z2 in ((L.cE, L.rE, L.P, L.V), (L.cN, L.rN, L.V, L.Y)):
        c, r, z1, z2 = complex(c), float(r), complex(z1), complex(z2)
        t1, t2 = np.angle(z1 - c), np.angle(z2 - c)
        if abs(t2 - t1) > np.pi:
            t2 += 2 * np.pi * (1 if t2 < t1 else -1)
        for s in np.linspace(t1, t2, 300):
            z = c + r * np.exp(1j * s)
            mp_.RestrictH(x=z.real, y=z.imag, z=0, h=h_hyp * (1 - abs(z) ** 2) / 2)
    mesh = ngs.Mesh(geo.GenerateMesh(mp=mp_))
    mesh.Curve(order_geom)
    return mesh


def assemble_mesh(mesh, dirichlet, order):
    import ngsolve as ngs
    fes = ngs.H1(mesh, order=order, dirichlet=dirichlet)
    u, v = fes.TnT()
    w = 4 / (1 - ngs.x * ngs.x - ngs.y * ngs.y) ** 2
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()
    m = ngs.BilinearForm(w * u * v * ngs.dx(bonus_intorder=2 * order)).Assemble()
    fd = np.array(list(fes.FreeDofs()), dtype=bool)
    A = sp.csr_matrix(a.mat.CSR())[fd][:, fd]
    M = sp.csr_matrix(m.mat.CSR())[fd][:, fd]
    A = ((A + A.T) * 0.5).tocsc()
    M = ((M + M.T) * 0.5).tocsc()
    area = ngs.Integrate(w * ngs.dx(bonus_intorder=2 * order), mesh)
    return dict(A=A, M=M, fes=fes, freedofs=fd, mesh=mesh, area_fem=area, ne=mesh.ne)


def assemble(tau, sector, h, order):
    L = Lambert(tau)
    mesh = make_mesh(L, h, order)
    S = assemble_mesh(mesh, dirichlet_sides(sector), order)
    S["area_exact"] = float(L.area_exact)
    return S


def run_path(tau, sector, h, order):
    return os.path.join(RUNDIR, f"tau{tau:.1f}_{sector}_h{h}_p{order}.npz")


def run(tau, sector, h, order, nev=NEV, k=K_SLICE, verbose=False):
    t0 = time.time()
    S = assemble(tau, sector, h, order)
    lam, _, info = eigenvalues_robust(S["A"], S["M"], nev, k=k)
    S["slice_info"] = info
    if sector == "NNN":
        assert abs(lam[0]) < 1e-8, lam[0]
    return lam, S, time.time() - t0


def _job(args):
    tau, sector, h, order = args
    import ngsolve as ngs
    ngs.SetNumThreads(1)
    out = run_path(tau, sector, h, order)
    if os.path.exists(out):
        return out, 0.0
    lam, S, secs = run(tau, sector, h, order)
    tmp = out + ".tmp.npz"
    np.savez(tmp, lam=lam, tau=tau, sector=sector, h=h, order=order, ndof=S["A"].shape[0],
             ne=S["ne"], area_err=S["area_fem"] - S["area_exact"], seconds=secs,
             misses_repaired=S["slice_info"]["misses_repaired"], n_slices=S["slice_info"]["n_slices"])
    os.replace(tmp, out)
    print(f"done tau={tau:.1f} {sector} h={h} p={order}: n={len(lam)} lam_max={lam[-1]:.1f} "
          f"ndof={S['A'].shape[0]} area_err={S['area_fem'] - S['area_exact']:.1e} "
          f"misses_repaired={S['slice_info']['misses_repaired']} {secs:.0f}s", flush=True)
    return out, secs


def suite(workers=16, taus=TAUS, sectors=SECTORS, levels=LEVELS):
    from multiprocessing import get_context
    os.makedirs(RUNDIR, exist_ok=True)
    # most expensive first, so the pool drains evenly
    cost = {(0.05, 10): 5, (0.07, 12): 4, (0.07, 10): 3, (0.07, 8): 2, (0.1, 10): 1}
    jobs = [(t, s, h, p) for (h, p) in levels for t in taus for s in sectors]
    jobs.sort(key=lambda j: -cost[(j[2], j[3])])
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        tot = 0.0
        for i, (out, secs) in enumerate(pool.imap_unordered(_job, jobs)):
            tot += secs
            if (i + 1) % 16 == 0:
                print(f"[{i + 1}/{len(jobs)}] wall {time.time() - t0:.0f}s, cpu-sum {tot:.0f}s", flush=True)
    print(f"SUITE DONE: {len(jobs)} runs, wall {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- full-Q check
def make_quad_mesh(L, h_hyp, order_geom):
    """The whole quadrilateral Q (four arcs), for the symmetry-reduction check."""
    import ngsolve as ngs
    from netgen.geom2d import SplineGeometry
    from netgen.meshing import MeshingParameters
    geo = SplineGeometry()
    vs = L.quad_vertices()                                 # V, -conj V, -V, conj V
    sides = L.quad_sides()                                 # east, north, west, south
    side_of = {0: 1, 1: 2, 2: 3, 3: 0}                     # vertex i -> i+1 lies on side_of[i]
    pts = [geo.AppendPoint(float(v.real), float(v.imag)) for v in vs]
    for i in range(4):
        c = sides[side_of[i]][0]
        z1, z2 = vs[i], vs[(i + 1) % 4]
        ctrl = geo.AppendPoint(*L.arc_control(c, z1, z2))
        geo.Append(["spline3", pts[i], ctrl, pts[(i + 1) % 4]], bc="s", leftdomain=1, rightdomain=0)
    mp_ = MeshingParameters(maxh=h_hyp / 2, grading=0.2)
    R = float(abs(L.V)) * 1.05
    n = 120
    for i in range(n + 1):
        for j in range(n + 1):
            x, y = -R + 2 * R * i / n, -R + 2 * R * j / n
            if x * x + y * y < 0.999:
                mp_.RestrictH(x=x, y=y, z=0, h=h_hyp * (1 - x * x - y * y) / 2)
    mesh = ngs.Mesh(geo.GenerateMesh(mp=mp_))
    mesh.Curve(order_geom)
    return mesh


FULLQ = dict(tau=0.8, h=0.07, order=10, nev=1200)


def fullquad_path(bc):
    return os.path.join(RUNDIR, f"fullQ_tau{FULLQ['tau']:.1f}_{bc}_h{FULLQ['h']}_p{FULLQ['order']}.npz")


def _fullq_job(bc):
    import ngsolve as ngs
    ngs.SetNumThreads(1)
    t0 = time.time()
    L = Lambert(FULLQ["tau"])
    mesh = make_quad_mesh(L, FULLQ["h"], FULLQ["order"])
    S = assemble_mesh(mesh, "s" if bc == "D" else "", FULLQ["order"])
    lam, _, info = eigenvalues_robust(S["A"], S["M"], FULLQ["nev"], k=K_SLICE)
    np.savez(fullquad_path(bc), lam=lam, ndof=S["A"].shape[0], area_err=S["area_fem"] - 4 * float(L.area_exact),
             seconds=time.time() - t0, misses_repaired=info["misses_repaired"], n_slices=info["n_slices"])
    print(f"fullQ {bc}: n={len(lam)} lam_max={lam[-1]:.1f} ndof={S['A'].shape[0]} "
          f"area_err={S['area_fem'] - 4 * float(L.area_exact):.1e} {time.time() - t0:.0f}s", flush=True)


def fullquad(workers=2):
    from multiprocessing import get_context
    os.makedirs(RUNDIR, exist_ok=True)
    with get_context("spawn").Pool(workers) as pool:
        list(pool.imap_unordered(_fullq_job, ["N", "D"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "suite":
        suite(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    elif cmd == "fullquad":
        fullquad()
    elif cmd == "one":
        tau, sector, h, p, nev = float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        import ngsolve as ngs
        ngs.SetNumThreads(4)
        lam, S, secs = run(tau, sector, h, p, nev=nev, verbose=True)
        print(f"ndof={S['A'].shape[0]} area_err={S['area_fem'] - S['area_exact']:.2e} {secs:.1f}s")
        print("first eigenvalues:", np.array2string(lam[:12], precision=10))

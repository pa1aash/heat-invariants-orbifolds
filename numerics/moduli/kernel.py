"""Heat-kernel diagonal K(t, x, x) of two contrasting members of the family.

Members: tau = 0 (the square) and tau = 2.8 (the most elongated, systole 0.694).
For x in the Lambert quarter L (one quarter of one sheet of the double),

    K_O(t, x, x) = (1/8) sum_{sectors s} sum_j e^{-lambda_j^s t} u_j^s(x)^2,

where u_j^s are the L^2(L)-normalised sector eigenfunctions: the eigenfunctions
of O are their (+-)-extensions to the 8 copies of L, of norm^2 8.  Hence
int_O K_O dA = sum_s Z_s = Z_O.

Level (h, p) = (0.07, 10), as the S3 export; NEV per sector as the suite.
Points: vertices of a hyperbolic-size-0.03 triangulation of L (points and
triangles are saved), pulled 1e-9 toward the centroid so that point location in
the curved solver mesh is safe.  Coordinates in the Poincare disk; hyperbolic
normalisation.

Sanity asserts (physics, not fits):
  - far from the boundary and the cone point, K_O ~ (4 pi t)^{-1} (1 - t/3)
    (heat kernel of H^2 on the diagonal; K = -1) to 1e-3 relative at t = 0.005,
    at points whose distance to the cone point exceeds 6 sqrt(t) -- neither the sides of Q
    nor its mirror axes are boundaries of the double O, they are interior geodesics, and
    the shortest geodesic loop through any point is >= the systole (e^{-sys^2/4t} < 1e-10);
  - at the cone point V (order 3) K_O -> 3/(4 pi t): ratio within 5% at t = 0.005.

Writes data/heat_kernel_diagonal_moduli.npz.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from quad import Lambert  # noqa: E402
from solve_moduli import SECTORS, NEV, K_SLICE, make_mesh, assemble  # noqa: E402

MEMBERS = (0.0, 2.8)
LEVEL = (0.07, 10)
TIMES = (0.005, 0.01, 0.02)
SAMPLE_H = 0.03


def sample_points(tau):
    import ngsolve as ngs
    L = Lambert(tau)
    fine = make_mesh(L, SAMPLE_H, 1)
    pts = np.array([v.point for v in fine.vertices])
    tris = np.array([[v.nr for v in el.vertices] for el in fine.Elements(ngs.VOL)])
    cen = pts.mean(axis=0)
    return pts, tris, cen + (pts - cen) * (1 - 1e-9)


def _job(args):
    tau, sector = args
    import ngsolve as ngs
    from solve_moduli import eigenvalues_robust
    ngs.SetNumThreads(1)
    t0 = time.time()
    pts, tris, pe = sample_points(tau)
    S = assemble(tau, sector, *LEVEL)
    lam, vec, info = eigenvalues_robust(S["A"], S["M"], NEV, k=K_SLICE, want_vectors=True)
    gf = ngs.GridFunction(S["fes"])
    fd = S["freedofs"]
    mpts = S["mesh"](pe[:, 0], pe[:, 1])
    K = np.zeros((len(TIMES), len(pts)))
    for j in range(len(lam)):
        full = np.zeros(S["fes"].ndof)
        full[fd] = vec[:, j]
        gf.vec.FV().NumPy()[:] = full
        u = gf(mpts).ravel()
        for i, tt in enumerate(TIMES):
            K[i] += np.exp(-lam[j] * tt) * u * u
    print(f"kernel tau={tau} {sector}: {len(lam)} eigenfunctions, lambda_max {lam[-1]:.0f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return tau, sector, K, lam


def main(workers=16):
    from multiprocessing import get_context
    jobs = [(tau, s) for tau in MEMBERS for s in SECTORS]
    with get_context("spawn").Pool(workers) as pool:
        res = list(pool.imap_unordered(_job, jobs))
    out = {}
    for tau in MEMBERS:
        pts, tris, _ = sample_points(tau)
        L = Lambert(tau)
        Ks = {s: K for (tt, s, K, _) in res if tt == tau}
        KO = sum(Ks[s] for s in SECTORS) / 8.0
        tag = f"tau{tau:.1f}"
        z = pts[:, 0] + 1j * pts[:, 1]
        V = complex(L.V)
        # hyperbolic distance to the cone point and to the sides of Q (the two arcs)
        # nearest cone point: V itself (the other three are mirror images of V, farther from L)
        dV = 2 * np.arctanh(np.abs((z - V) / (1 - np.conj(V) * z)))
        t = TIMES[0]
        far = dV > 6 * np.sqrt(t)
        flat = (1 - t / 3) / (4 * np.pi * t)
        relfar = np.abs(KO[0][far] / flat - 1)
        assert far.sum() > 50 and np.max(relfar) < 1e-3, (tau, far.sum(), np.max(relfar))
        iv = int(np.argmin(dV))
        ratio_cone = KO[0][iv] / (3 / (4 * np.pi * t))
        assert abs(ratio_cone - 1) < 0.05, (tau, ratio_cone)
        print(f"tau={tau}: {far.sum()} interior points within {np.max(relfar):.1e} of (1 - t/3)/(4 pi t); "
              f"K/(3/(4 pi t)) at the cone point = {ratio_cone:.4f}", flush=True)
        out[tag] = dict(points=pts, triangles=tris, t=np.array(TIMES), K_orbifold=KO,
                        **{f"K_sector_{s}": Ks[s] for s in SECTORS},
                        lambda_max_per_sector=np.array([max(l) for (tt, s, K, l) in res if tt == tau]))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "data", "heat_kernel_diagonal_moduli.npz"),
                        **{f"{tag}_{k}": v for tag, d in out.items() for k, v in d.items()},
                        level=np.array(LEVEL), sectors=np.array(SECTORS), note=np.array(
                            "K(t,x,x) of the doubled quadrilateral O(tau) at points x of the Lambert quarter L "
                            "(first quadrant, one sheet), Poincare-disk coordinates, hyperbolic normalisation "
                            "(int_O K dA = Z). K_orbifold = (1/8) sum over the 8 symmetry sectors of the sector "
                            "kernels K_sector_s (L^2(L)-normalised eigenfunctions). Values elsewhere on O follow "
                            "by the mirror symmetries."))
    print("KERNEL EXPORT DONE")


if __name__ == "__main__":
    main()

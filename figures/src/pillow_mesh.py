"""Stylised 3D pillow meshes for the Blender renders (F1, F6 and the contact sheet).

A pillow O(p,q,r) is two copies of the hyperbolic triangle T_{p,q,r} glued along their boundary.
The shape drawn is NOT an isometric embedding: each sheet is the triangle in Poincare-disk
coordinates (conformal, so the corner angles pi/m are exact), lifted to height +-c sqrt(u),
where -Lap u = 1 in the triangle and u = 0 on its sides (P1 finite elements on the committed
triangulation). The two sheets meet along the sides with a rounded seam, and the corners
(the cone points) stay sharp. Each vertex carries the value of the heat-kernel diagonal of the
pillow there; the pillow kernel is the same on both sheets.

    python3 figures/src/pillow_mesh.py            writes figures/build/mesh_<key>_t<i>.npz

Asserted: the triangulation's boundary is one closed curve through the three corners; the
interior values of 4 pi t h_t(x,x) are 1 - t/3 + O(t^2) away from the corners and the largest
value is m/(1) at the corner of largest order m to within 1%.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "figures" / "build"
KERNEL = ROOT / "numerics" / "data" / "heat_kernel_diagonal.npz"
ORDERS = {"2-8-8": (2, 8, 8), "3-3-12": (3, 3, 12)}
HEIGHT = 1.5            # c in h = c sqrt(u); one value for every pillow, so sizes are comparable


def boundary(T):
    e = Counter()
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            e[(min(u, v), max(u, v))] += 1
    edges = [k for k, v in e.items() if v == 1]
    nb = Counter(i for k in edges for i in k)
    assert all(v == 2 for v in nb.values()), "boundary is not a closed curve"
    return sorted(nb)


def poisson(P, T, bnd):
    """P1 solution of -Lap u = 1, u = 0 on the boundary vertices."""
    n = len(P)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    for tri in T:
        x = P[tri]
        B = np.array([x[1] - x[0], x[2] - x[0]]).T
        area = abs(np.linalg.det(B)) / 2
        G = np.linalg.inv(B).T @ np.array([[-1, 1, 0], [-1, 0, 1]])
        Ke = area * G.T @ G
        for i in range(3):
            rhs[tri[i]] += area / 3
            for j in range(3):
                rows.append(tri[i]); cols.append(tri[j]); vals.append(Ke[i, j])
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    free = np.setdiff1d(np.arange(n), bnd)
    u = np.zeros(n)
    u[free] = spsolve(A[free][:, free], rhs[free])
    assert u.min() >= -1e-12
    return u


def build(key, ti):
    d = np.load(KERNEL)
    P, T = d[f"{key}_points"], d[f"{key}_triangles"]
    t = float(d[f"{key}_t"][ti])
    ratio = d[f"{key}_K_pillow"][ti] * 4 * np.pi * t          # 4 pi t h_t(x,x)
    bnd = boundary(T)
    u = poisson(P, T, bnd)
    h = HEIGHT * np.sqrt(np.clip(u, 0, None))
    # corners: the three boundary vertices where the largest ratios sit, one per cone point
    m = ORDERS[key]
    far = np.array([np.min(np.hypot(*(P[bnd] - P[i]).T)) for i in range(len(P))])
    assert np.all(np.abs(ratio[far > 0.15] - (1 - t / 3)) < 0.02), "interior value 1 - t/3"
    assert abs(ratio.max() - max(m)) < 0.01 * max(m), (ratio.max(), m)
    # double: boundary vertices shared, interior vertices duplicated at -h
    n = len(P)
    interior = np.setdiff1d(np.arange(n), bnd)
    low = {int(i): n + k for k, i in enumerate(interior)}
    V = np.vstack([np.c_[P, h], np.c_[P[interior], -h[interior]]])
    vals = np.r_[ratio, ratio[interior]]
    Tb = np.array([[low.get(int(i), int(i)) for i in tri[::-1]] for tri in T])
    F = np.vstack([T, Tb])
    corners = [int(i) for i in bnd if ratio[i] > 0.9 * min(m)]
    out = BUILD / f"mesh_{key}_t{ti}.npz"
    BUILD.mkdir(parents=True, exist_ok=True)
    np.savez(out, vertices=V, faces=F, ratio=vals, t=t, orders=np.array(m), boundary=np.array(bnd))
    return out, t, float(ratio.min()), float(ratio.max())


if __name__ == "__main__":
    tis = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    for key in ORDERS:
        for ti in tis:
            out, t, lo, hi = build(key, ti)
            print(f"{out.relative_to(ROOT)}: t = {t}, 4 pi t h_t(x,x) in [{lo:.4f}, {hi:.4f}]")

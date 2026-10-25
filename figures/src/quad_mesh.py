"""Stylised 3D meshes of members O(theta) of the moduli family (0;3,3,3,3) for F6(a), with the
shortest closed geodesic as a tube.

Each member is the double of the geodesic quadrilateral Q(theta) of numerics/moduli (four angles
pi/3, symmetric under z -> conj z and z -> -conj z in the Poincare disk; its sides are arcs of the
circles listed in numerics/moduli/data/geometries.json). As for the pillows (pillow_mesh.py), each
sheet is Q in disk coordinates lifted to +-c sqrt(u), -Lap u = 1 on Q, u = 0 on its sides: a
stylised shape, not an isometric embedding. The surface is a uniform neutral (no scalar field).

The shortest closed geodesic of every member has length 4b (asserted in geometries.json): the
segment of the imaginary axis from the south to the north side of Q (hyperbolic length 2b) on
both sheets. It is drawn as a tube on the surface; its colour is chosen by SPEC section 4 rule 2.

    python3 figures/src/quad_mesh.py     writes figures/build/quad_<theta>.npz
Asserted: the quadrilateral's corners lie on two side circles each, all four interior angles are
pi/3 (from the circle tangents), the hyperbolic length of the drawn geodesic is 4b = the systole.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pillow_mesh import HEIGHT, boundary, poisson  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "figures" / "build"
MEMBERS = ["0.0", "0.8", "1.6", "2.8"]
N_SIDE = 90
H_GRID = 0.012


def geometry(theta):
    g = json.loads((ROOT / "numerics/moduli/data/geometries.json").read_text())["members"][theta]
    circles = [(np.array(c["centre"]), c["radius"]) for c in g["quad_sides_circles"]]
    Vq = [np.array(v) for v in g["quad_vertices"]]
    return g, circles, Vq


def inside(P, circles):
    return np.all([np.hypot(*(P - c).T) > r for c, r in circles], axis=0) & (np.hypot(*P.T) < 1)


def side_arc(c, r, p, q, n):
    a0, a1 = np.arctan2(*(p - c)[::-1]), np.arctan2(*(q - c)[::-1])
    if abs(a1 - a0) > np.pi:
        a1 += 2 * np.pi * (1 if a1 < a0 else -1)
    s = np.linspace(a0, a1, n, endpoint=False)
    return c + r * np.c_[np.cos(s), np.sin(s)]


def build(theta):
    g, circles, Vq = geometry(theta)
    # corners: each on its two adjacent side circles; angle pi/3 between the circle tangents
    for k, v in enumerate(Vq):
        on = [abs(np.hypot(*(v - c)) - r) < 1e-12 for c, r in circles]
        assert sum(on) == 2, (theta, k)
        (c1, _), (c2, _) = [circles[i] for i in range(4) if on[i]]
        t1, t2 = v - c1, v - c2
        ang = np.arccos(abs(t1 @ t2) / np.linalg.norm(t1) / np.linalg.norm(t2))
        assert abs(min(ang, np.pi - ang) - np.pi / 3) < 1e-9, (theta, ang)
    # boundary: the four arcs in order (vertices listed counterclockwise from the first quadrant)
    order = [(0, 1), (1, 2), (2, 3), (3, 0)]
    bpts = []
    for i, j in order:
        p, q = Vq[i], Vq[j]
        ci = next(k for k, (c, r) in enumerate(circles)
                  if abs(np.hypot(*(p - c)) - r) < 1e-12 and abs(np.hypot(*(q - c)) - r) < 1e-12)
        bpts.append(side_arc(*circles[ci], p, q, N_SIDE))
    B = np.vstack(bpts)
    xs = np.arange(-1, 1, H_GRID)
    G = np.array(np.meshgrid(xs, xs)).reshape(2, -1).T
    G = G[inside(G, circles)]
    dist = np.min(np.hypot(*(G[:, None, :] - B[None]).transpose(2, 0, 1)), axis=1)
    G = G[dist > 0.6 * H_GRID]
    P = np.vstack([B, G])
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1)
    tri = tri[inside(cen, circles)]
    bnd = boundary(tri)
    assert set(range(len(B))) <= set(bnd)
    u = poisson(P, tri, bnd)
    h = HEIGHT * np.sqrt(np.clip(u, 0, None))
    n = len(P)
    interior = np.setdiff1d(np.arange(n), bnd)
    low = {int(i): n + k for k, i in enumerate(interior)}
    V = np.vstack([np.c_[P, h], np.c_[P[interior], -h[interior]]])
    F = np.vstack([tri, [[low.get(int(i), int(i)) for i in t[::-1]] for t in tri]])
    corners = [int(np.argmin(np.hypot(*(P - v).T))) for v in Vq]
    # the geodesic: x = 0 from south to north side, on both sheets
    yN = float(np.tanh(g["b"] / 2))
    ys = np.linspace(-yN, yN, 241)
    from scipy.interpolate import LinearNDInterpolator
    hf = LinearNDInterpolator(P, h)
    hy = np.nan_to_num(hf(np.c_[np.zeros_like(ys), ys]), nan=0.0)
    hy[[0, -1]] = 0.0
    loop = np.vstack([np.c_[np.zeros_like(ys), ys, hy], np.c_[np.zeros_like(ys), ys[::-1], -hy[::-1]]])
    per_sheet = 2 * (2 * np.arctanh(yN))                 # from -i yN to i yN: twice the distance 2 artanh(yN)
    length = 2 * per_sheet                               # both sheets
    assert abs(length - 4 * g["b"]) < 1e-12 and abs(length - g["systole"]) < 1e-9
    out = BUILD / f"quad_{theta}.npz"
    BUILD.mkdir(parents=True, exist_ok=True)
    crease = np.zeros(len(V))
    crease[corners] = 1.0
    np.savez(out, vertices=V, faces=F, crease=crease, geodesic=loop, systole=g["systole"])
    return out


if __name__ == "__main__":
    for th in MEMBERS:
        print(build(th).relative_to(ROOT))

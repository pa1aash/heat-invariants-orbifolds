"""F6 (Sections 4 and 7): the moduli family (0;3,3,3,3): the shape moves, the heat expansion does not.

(a) Four members O(theta), theta = 0, 0.8, 1.6, 2.8 (stylised, quad_mesh.py; one orthographic
    scale for all four), each with its shortest closed geodesic (length 4b(theta), the systole:
    2.634, 1.831, 1.250, 0.694) drawn on the surface. Surface: uniform neutral; curve colour by
    SPEC section 4 rule 2 (the dark pillow hue or ink, whichever contrasts more with the surface).
(b) The eigenvalues lambda_j(theta) below LAMBDA_MAX (the first 40 or so of every member), at the
    eight computed moduli theta = 0, 0.4, ..., 2.8 (dots), joined along each symmetry sector
    (numerics/moduli/data/eigenvalue_flow_sectors.csv): eigenvalues of one sector are joined in
    order, so lines of different sectors may cross and lines of one sector do not.

Asserted: for every theta, the union of the eight sector spectra equals the orbifold list of
eigenvalue_flow_orbifold.csv below LAMBDA_MAX (to 1e-9); lambda_0 = 0; lambda_1 decreases from
4.1224 (theta = 0) to 0.4348 (theta = 2.8); the systoles are those of
geometries.json and strictly decrease.
"""
import json

import numpy as np

from figlib import ROOT, check_only, finish, fs, rows

THETAS = ["0.0", "0.4", "0.8", "1.2", "1.6", "2.0", "2.4", "2.8"]
SHOWN = ["0.0", "0.8", "1.6", "2.8"]
LAMBDA_MAX = 120.0
SURFACE_L = 88


def data():
    orb = rows("numerics/moduli/data/eigenvalue_flow_orbifold.csv")
    sec = rows("numerics/moduli/data/eigenvalue_flow_sectors.csv")
    branches = {}
    for th in THETAS:
        o = sorted(float(r["lambda"]) for r in orb if r["tau"] == th and float(r["lambda"]) < LAMBDA_MAX)
        s = sorted(float(r["lambda"]) for r in sec if r["tau"] == th and float(r["lambda"]) < LAMBDA_MAX)
        assert len(o) == len(s) and np.allclose(o, s, atol=1e-9, rtol=0), th
        assert o[0] == 0
    l1 = {th: sorted(float(r["lambda"]) for r in orb if r["tau"] == th)[1] for th in THETAS}
    assert abs(l1["0.0"] - 4.1224) < 1e-4 and abs(l1["2.8"] - 0.4348) < 1e-4
    for r in sec:
        branches.setdefault((r["sector"], int(r["k"])), {})[r["tau"]] = float(r["lambda"])
    lines = []
    for b, v in branches.items():
        ys = np.array([v[th] for th in THETAS])
        if ys.min() < LAMBDA_MAX:
            lines.append(ys)
    g = json.loads((ROOT / "numerics/moduli/data/geometries.json").read_text())["members"]
    sy = [g[th]["systole"] for th in SHOWN]
    assert all(a > b for a, b in zip(sy, sy[1:])) and abs(sy[0] - 2.6339) < 1e-4 and abs(sy[-1] - 0.694) < 1e-3
    return np.array([float(t) for t in THETAS]), lines, sy


def curve_colour():
    """SPEC section 4 rule 2: dark pillow hue or ink, whichever has the higher contrast ratio
    against the surface (WCAG relative-luminance contrast)."""
    import colourtools as ct
    fs.use()
    Y = lambda c: float((ct.to_linear(np.array(c)) @ np.array([0.2126729, 0.7151522, 0.0721750])))
    Ls = ((SURFACE_L + 16) / 116) ** 3
    cands = {"ink": fs.GREY["ink"], "dark pillow hue": fs.PILLOW["2,8,8"]}
    ratio = {k: (Ls + 0.05) / (Y(c) + 0.05) for k, c in cands.items()}
    best = max(ratio, key=ratio.get)
    return best, cands[best], ratio


def renders():
    import subprocess
    import sys
    import blender_jobs as bj
    subprocess.run([sys.executable, str(ROOT / "figures/src/quad_mesh.py")], check=True, stdout=subprocess.DEVNULL)
    _, rgb, _ = curve_colour()
    out = {}
    for th in SHOWN:
        mesh = bj.BUILD / f"quad_{th}.npz"
        out[th] = bj.two_pass(mesh, f"F6_{th}", ["--res", "1000x1000", "--view", "90,55", "--surface-L", str(SURFACE_L),
                                                  "--curve-rgb", ",".join(f"{c:.6f}" for c in rgb),
                                                  "--curve-radius", "0.016", "--ortho", "2.3"])
    return out


def draw(theta, lines, sy, imgs):
    import matplotlib.pyplot as plt
    fs.use()
    fig = fs.figure(118)
    import blender_jobs as bj
    crops = bj.crop_common([plt.imread(imgs[th]) for th in SHOWN])      # one scale, white ground
    for i, img in enumerate(crops):
        ax = fig.add_axes([0.03 + 0.25 * i, 0.70, 0.21, 0.29])
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
    fs.letter_at(fig, 0.005, 0.995, "a")
    bx = fig.add_axes([0.11, 0.08, 0.86, 0.56])
    for ys in lines:
        bx.plot(theta, ys, color=fs.GREY["mid"], lw=fs.LW["thin"], marker="o", ms=1.6, mfc=fs.GREY["dark"], mec="none")
    l1 = np.sort(np.array(lines), axis=0)[1]          # lambda_1 at each modulus, across the sectors
    bx.plot(theta, l1, color=fs.GREY["ink"], lw=fs.LW["heavy"], marker="o", ms=2.4, mfc=fs.GREY["ink"], mec="none",
            zorder=4)
    for th in SHOWN:
        bx.axvline(float(th), color=fs.GREY["light"], lw=fs.LW["thin"], zorder=0)
    bx.set_xlim(-0.05, 2.85)
    bx.set_yscale("function", functions=(lambda y: np.sqrt(np.clip(y, 0, None)), lambda r: r ** 2))   # square-root
    bx.set_ylim(0, LAMBDA_MAX)
    bx.set_yticks([0, 1, 5, 10, 20, 40, 60, 80, 100, 120])
    bx.set_yticklabels(["0", "1", "5", "10", "20", "40", "60", "80", "100", "120"])
    bx.minorticks_off()
    bx.set_xlabel(r"$\vartheta$")
    bx.set_ylabel(r"$\lambda_j$")
    fs.letter_at(fig, 0.005, 0.665, "b")
    return fig


if __name__ == "__main__":
    d = data()
    best, _, ratio = curve_colour()
    print(f"F6: curve colour {best} (contrast {ratio})")
    if not check_only():
        finish(draw(*d, renders()), "F6")
    print("F6: assertions passed")

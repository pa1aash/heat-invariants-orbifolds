"""F1 (Introduction): the two pillows O(2,8,8) and O(3,3,12), coloured by the heat-kernel diagonal.

Stylised shapes (not isometric embeddings; pillow_mesh.py): each sheet is the triangle in
Poincare-disk coordinates lifted to +-c sqrt(u), -Lap u = 1, so the cone points are sharp corners.
Colour: 4 pi t h_t(x,x) at t = T (numerics/data/heat_kernel_diagonal.npz), on one log scale shared
by the two pillows (the colour bar), darker = larger. Away from the cone points the value is
1 - t/3 + O(t^2); at a cone point of order m it approaches m.

    python3 figures/src/F1.py [--check]      (Blender renders, cached in figures/build/)

Asserted: on each pillow, interior value 1 - t/3 to 0.02 and maximum = largest order to 1%;
2 * int_T h_t(x,x) dA_hyp (P1 quadrature on the committed triangulation, both sheets) equals the
computed heat trace Z(t) of numerics/data/heat_trace_difference.csv to 0.5%; both pillows lie
inside the shared colour scale.
"""
import csv

import numpy as np

from figlib import ROOT, check_only, finish, fs

T_INDEX = 2
KEYS = {"2-8-8": ((2, 8, 8), "Z_2_8_8"), "3-3-12": ((3, 3, 12), "Z_3_3_12")}
VMAX = np.log(12.0)
SHADE_WEIGHT = 0.10      # light shading: the brightness of the surface is the colour scale


def data():
    d = np.load(ROOT / "numerics/data/heat_kernel_diagonal.npz")
    zrows = list(csv.DictReader((ROOT / "numerics/data/heat_trace_difference.csv").open()))
    lo = []
    for key, (m, zcol) in KEYS.items():
        P, Tri = d[f"{key}_points"], d[f"{key}_triangles"]
        t = float(d[f"{key}_t"][T_INDEX])
        K = d[f"{key}_K_pillow"][T_INDEX]
        ratio = 4 * np.pi * t * K
        assert abs(ratio.max() - max(m)) < 0.01 * max(m) and ratio.max() < 12
        assert abs(np.median(ratio) - (1 - t / 3)) < 0.02
        w = 4 / (1 - (P ** 2).sum(1)) ** 2                    # hyperbolic area density in the disk
        x = P[Tri]
        area = 0.5 * np.abs((x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (x[:, 2, 0] - x[:, 0, 0]) * (x[:, 1, 1] - x[:, 0, 1]))
        integral = 2 * np.sum(area * (K * w)[Tri].mean(1))
        Z = float(next(float(r[zcol]) for r in sorted(zrows, key=lambda r: abs(float(r["t"]) - t))))
        nearest = min(zrows, key=lambda r: abs(float(r["t"]) - t))
        assert abs(float(nearest["t"]) - t) < 1e-3 * t, (t, nearest["t"])
        assert abs(integral / float(nearest[zcol]) - 1) < 0.005, (key, integral, nearest[zcol])
        lo.append(float(np.log(ratio.min())))
    return t, min(lo)


def renders(vmin):
    import subprocess
    import sys
    import blender_jobs as bj
    subprocess.run([sys.executable, str(ROOT / "figures/src/pillow_mesh.py"), str(T_INDEX)], check=True,
                   stdout=subprocess.DEVNULL)
    out = {}
    meshes = {key: bj.BUILD / f"mesh_{key}_t{T_INDEX}.npz" for key in KEYS}
    radius = []
    for mesh in meshes.values():
        V = np.load(mesh)["vertices"]
        radius.append(float(np.max(np.linalg.norm((V - V.mean(axis=0))[:, :2], axis=1))))
    ortho = 2.25 * max(radius)                      # one orthographic scale for both pillows
    for key, mesh in meshes.items():
        out[key] = bj.two_pass(mesh, f"F1_{key}", ["--res", "1700x1400", "--view", "-20,62", "--ortho", repr(ortho),
                                                    "--vmin", repr(float(vmin)), "--vmax", repr(float(VMAX))],
                               weight=SHADE_WEIGHT)
    return out


def draw(t, vmin, imgs):
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import LogNorm
    import blender_jobs as bj
    fs.use()
    fig = fs.figure(66)
    crops = bj.crop_common([plt.imread(imgs[key]) for key in KEYS])     # one scale, white ground
    for i, img in enumerate(crops):
        ax = fig.add_axes([0.04 + 0.5 * i, 0.31, 0.44, 0.66])
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
        fs.letter_at(fig, 0.005 + 0.5 * i, 0.99, "ab"[i])
    cax = fig.add_axes([0.25, 0.21, 0.50, 0.035])
    cb = ColorbarBase(cax, cmap=fs.SEQ, norm=LogNorm(np.exp(vmin), np.exp(VMAX)), orientation="horizontal")
    cb.set_ticks([1, 2, 3, 4, 6, 8, 12])
    cb.set_ticklabels(["1", "2", "3", "4", "6", "8", "12"])
    cb.minorticks_off()
    cb.outline.set_linewidth(fs.LW["axis"])
    cb.set_label(r"$4\pi t\,\mathfrak{h}_t(x,x)$")
    return fig


if __name__ == "__main__":
    t, vmin = data()
    if not check_only():
        finish(draw(t, vmin, renders(vmin)), "F1")
    print(f"F1: t = {t}; assertions passed")

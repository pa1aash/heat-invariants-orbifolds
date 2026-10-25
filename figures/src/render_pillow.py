"""Blender render of one stylised pillow coloured by the heat-kernel diagonal.

    blender -b --factory-startup --threads 2 -P figures/src/render_pillow.py -- \
        MESH.npz OUT.png [--palette A] [--res 1200x900] [--vmin V] [--vmax V] [--view az,el]

--pass colour renders the exact vertex colours (flat, unlit); --pass shade renders a uniform grey
surface under the studio light. figstyle.shade() combines the two with a fixed, mild weight, so
the shading shows the shape without changing the colour scale by more than that weight.
MESH.npz is written by pillow_mesh.py. The colour of a vertex is the palette's sequential map at
(log ratio - vmin)/(vmax - vmin), ratio = 4 pi t h_t(x,x); vmin, vmax are passed so that several
renders share one scale. The camera is orthographic. Not part of the verification suite.
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures" / "style"))
from blender_palette import Palette, grey_of_L, setup_scene, srgb_to_linear  # noqa: E402


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    opt = {"--palette": None, "--res": "1200x900", "--vmin": None, "--vmax": None, "--view": "-35,32", "--pass": "colour",
           "--surface-L": None, "--curve-rgb": None, "--curve-radius": "0.012",
           "--ortho": None}
    pos = []
    i = 0
    while i < len(a):
        if a[i] in opt:
            opt[a[i]] = a[i + 1]
            i += 2
        else:
            pos.append(a[i])
            i += 1
    return pos, opt


def tube(sc, loop, radius, rgb, k=12):
    """A closed tube of the given radius around the polyline `loop` (closed), as a mesh with a
    constant vertex colour (sRGB `rgb`)."""
    n = len(loop)
    T = np.roll(loop, -1, 0) - np.roll(loop, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    ref = np.array([1.0, 0.0, 0.0])
    N1 = np.cross(T, ref)
    N1 /= np.linalg.norm(N1, axis=1)[:, None]
    N2 = np.cross(T, N1)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    verts = (loop[:, None, :] + radius * (np.cos(ang)[None, :, None] * N1[:, None, :]
                                         + np.sin(ang)[None, :, None] * N2[:, None, :])).reshape(-1, 3)
    faces = [[i * k + j, i * k + (j + 1) % k, ((i + 1) % n) * k + (j + 1) % k, ((i + 1) % n) * k + j]
             for i in range(n) for j in range(k)]
    me = bpy.data.meshes.new("curve")
    me.from_pydata(verts.tolist(), [], faces)
    me.update()
    col = me.color_attributes.new("heat", "FLOAT_COLOR", "POINT")
    c = srgb_to_linear(np.array(rgb))
    col.data.foreach_set("color", np.tile(np.r_[c, 1.0], len(verts)))
    me.color_attributes.active_color = col
    me.shade_smooth()
    sc.collection.objects.link(bpy.data.objects.new("curve", me))


def main():
    (mesh_path, out), opt = args()
    pal = Palette(opt["--palette"])
    d = np.load(mesh_path)
    V, F = d["vertices"], d["faces"]
    if opt["--surface-L"] is None:           # heat colouring: the palette's sequential map
        ratio = d["ratio"]
        lr = np.log(ratio)
        vmin = float(opt["--vmin"]) if opt["--vmin"] else float(lr.min())
        vmax = float(opt["--vmax"]) if opt["--vmax"] else float(lr.max())
        assert lr.min() >= vmin - 1e-9 and lr.max() <= vmax + 1e-9, "value outside the shared scale"
        surface = pal.sequential((lr - vmin) / (vmax - vmin))
    else:                                    # a uniform neutral of the given CIE L*
        g = grey_of_L(float(opt["--surface-L"]))
        surface = np.full((len(V), 3), g)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    rx, ry = (int(v) for v in opt["--res"].split("x"))
    sc = setup_scene(bpy, pal, rx, ry)

    centre = V.mean(axis=0)
    V = V - centre
    me = bpy.data.meshes.new("pillow")
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    col = me.color_attributes.new("heat", "FLOAT_COLOR", "POINT")
    rgb = srgb_to_linear(surface)
    col.data.foreach_set("color", np.c_[rgb, np.ones(len(rgb))].ravel())
    me.color_attributes.active_color = col
    me.shade_smooth()
    # sharp corners: crease the cone points so the subdivision keeps them
    crease = me.attributes.new("crease_vert", "FLOAT", "POINT")
    if "crease" in d:
        vals = d["crease"].astype(float)
    else:
        vals = np.zeros(len(V))
        bnd = d["boundary"]
        vals[bnd[ratio[bnd] > 0.9 * d["orders"].min()]] = 1.0
    crease.data.foreach_set("value", vals)
    ob = bpy.data.objects.new("pillow", me)
    sc.collection.objects.link(ob)
    mod = ob.modifiers.new("subdiv", "SUBSURF")
    mod.levels = mod.render_levels = 2

    if opt["--curve-rgb"] is not None:       # a closed curve on the surface, as a tube
        tube(sc, d["geodesic"] - centre, float(opt["--curve-radius"]),
             [float(v) for v in opt["--curve-rgb"].split(",")])

    az, el = (math.radians(float(v)) for v in opt["--view"].split(","))
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    R = float(np.max(np.linalg.norm(V[:, :2], axis=1)))
    cam_data.ortho_scale = float(opt["--ortho"]) if opt["--ortho"] else 2.25 * R
    cam = bpy.data.objects.new("cam", cam_data)
    dist = 10 * R
    cam.location = (dist * math.cos(el) * math.sin(az), -dist * math.cos(el) * math.cos(az), dist * math.sin(el))
    sc.collection.objects.link(cam)
    con = cam.constraints.new("TRACK_TO")
    target = bpy.data.objects.new("target", None)
    sc.collection.objects.link(target)
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    sc.camera = cam
    if opt["--pass"] == "colour":          # exact vertex colours: no lighting at all
        sc.display.shading.light = "FLAT"
    else:                                  # shading only: a uniform grey surface under studio light
        sc.display.shading.color_type = "SINGLE"
        sc.display.shading.single_color = (0.8, 0.8, 0.8)
    sc.render.filepath = str(Path(out).resolve())
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()

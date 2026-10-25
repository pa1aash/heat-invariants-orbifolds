"""Palette and materials for the Blender renders, from the same palette.json and tables as the
matplotlib style (figstyle.py). Runs inside Blender's Python (numpy is bundled).

Renders are flat and matte: the Workbench engine, studio lighting without specular highlights,
no outline, cavity, shadow or depth-of-field, a neutral background of palette L* 97, and the
'Standard' view transform, so a vertex colour reaches the image unchanged apart from the
diffuse shading that shows the shape.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PALETTE = json.loads((HERE / "palette.json").read_text())


def _table(name):
    return np.loadtxt(HERE / "third_party" / name)


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def grey_of_L(L):
    Y = ((L + 16) / 116) ** 3 if L > 8 else L / 903.3
    return float(np.where(Y <= 0.0031308, 12.92 * Y, 1.055 * Y ** (1 / 2.4) - 0.055))


class Palette:
    def __init__(self, key=None):
        key = key or PALETTE["chosen"]
        assert key in PALETTE["candidates"], f"no palette chosen: {key!r}"
        cand = PALETTE["candidates"][key]
        seq = _table(cand["sequential"]["table"])
        self.seq = seq[::-1] if cand["sequential"]["reverse"] else seq
        self.pillow = {}
        for sig, spec in cand["pillow"].items():
            t = _table(spec["table"])
            self.pillow[sig] = tuple(t[int(round(spec["at"] * (len(t) - 1)))])
        self.background = grey_of_L(PALETTE["neutral_Lstar"]["render_background"])
        self.key = key

    def sequential(self, x):
        """sRGB colours of values x in [0, 1] (linear interpolation in the 256-entry table)."""
        x = np.clip(np.asarray(x, float), 0, 1) * (len(self.seq) - 1)
        i = np.minimum(x.astype(int), len(self.seq) - 2)
        f = (x - i)[..., None]
        return (1 - f) * self.seq[i] + f * self.seq[i + 1]


def setup_scene(bpy, pal, res_x, res_y, threads=4):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.threads_mode = "FIXED"
    sc.render.threads = threads
    sc.render.resolution_x, sc.render.resolution_y = res_x, res_y
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    for attr in dir(sc.render):
        if attr.startswith("use_stamp"):
            try:
                setattr(sc.render, attr, False)
            except (AttributeError, TypeError):
                pass
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "VERTEX"
    sh.show_specular_highlight = False
    sh.show_cavity = False
    sh.show_object_outline = False
    sh.show_shadows = False
    sh.show_xray = False
    sh.background_type = "WORLD" if hasattr(sh, "background_type") else None
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    g = float(srgb_to_linear(pal.background))
    sc.world.color = (g, g, g)
    sc.display.render_aa = "32"
    return sc

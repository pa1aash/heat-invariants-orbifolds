"""Run Blender renders one at a time, with the machine-safety guard of SPEC section 5.

Before every render: read "System-wide memory free percentage" from `memory_pressure -Q`; render
only if it is at least 20%, otherwise re-check every 2 minutes and, after 30 minutes, give up and
raise MemoryBusy. (Swap use is not the signal: on macOS swap stays allocated after the pressure
has passed.) A render that runs longer than 10 minutes is stopped and raises RenderTimeout.
Renders are cached in figures/build/ and redone only when an input is newer than the image.
Never run in parallel; 2 threads; Workbench engine.
"""
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "figures" / "build"
FREE_MIN = 20                      # percent, "System-wide memory free percentage"
CHECK_EVERY = 2 * 60
WAIT_LIMIT = 30 * 60
RENDER_LIMIT = 10 * 60
THREADS = "2"


class MemoryBusy(RuntimeError):
    pass


class RenderTimeout(RuntimeError):
    pass


def memory_free_percent():
    out = subprocess.check_output(["memory_pressure", "-Q"]).decode()
    line = next(l for l in out.splitlines() if "System-wide memory free percentage" in l)
    return int(line.split(":")[1].strip().rstrip("%"))


def wait_for_memory():
    t0 = time.time()
    while (f := memory_free_percent()) < FREE_MIN:
        if time.time() - t0 >= WAIT_LIMIT:
            raise MemoryBusy(f"memory free {f}% below {FREE_MIN}% for {WAIT_LIMIT // 60} min: not starting Blender")
        print(f"memory free {f}% < {FREE_MIN}%: waiting", flush=True)
        time.sleep(CHECK_EVERY)
    return f


def blender(script, args):
    """blender -b --factory-startup --threads 2 -P script -- *args, after the memory guard."""
    wait_for_memory()
    cmd = ["blender", "-b", "--factory-startup", "--python-exit-code", "1", "--threads", THREADS, "-P", str(script), "--", *map(str, args)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, timeout=RENDER_LIMIT)
    except subprocess.TimeoutExpired:
        raise RenderTimeout(f"render ran longer than {RENDER_LIMIT // 60} min and was stopped: {script} {args}")


def render(script, args, out, inputs):
    """blender(script, args), writing `out` (cached)."""
    out = Path(out)
    deps = [Path(p) for p in inputs] + [Path(script), ROOT / "figures/style/palette.json",
                                         ROOT / "figures/style/blender_palette.py"]
    stamp = out.with_suffix(".args")              # the arguments are part of the cache key
    key = "\n".join(map(str, args))
    if (out.exists() and stamp.exists() and stamp.read_text() == key
            and all(out.stat().st_mtime > p.stat().st_mtime for p in deps)):
        return out
    blender(script, args)
    stamp.write_text(key)
    assert out.exists(), out
    return out


def two_pass(mesh, stem, extra, inputs=(), weight=None):
    """Colour pass and shading pass of render_pillow.py, combined by figstyle.shade()."""
    import figstyle as fs
    from PIL import Image
    script = ROOT / "figures/src/render_pillow.py"
    paths = {}
    for kind in ("colour", "shade"):
        p = BUILD / f"{stem}_{kind}.png"
        paths[kind] = render(script, [mesh, p, *extra, "--pass", kind], p, [mesh, *inputs])
    img = fs.shade(paths["colour"], paths["shade"], weight=fs.SHADE_WEIGHT if weight is None else weight)
    out = BUILD / f"{stem}.png"
    Image.fromarray((img * 255).round().astype(np.uint8)).save(out)
    return out


def crop_common(imgs, pad=10):
    """Crop renders made at one resolution and one orthographic scale to boxes of one common size,
    each centred on its object, so that they keep one scale; the flat render background is
    painted white."""
    imgs = [np.asarray(im)[..., :3].astype(float) for im in imgs]
    masks = [np.abs(im - im[2, 2]).sum(-1) > 0.02 for im in imgs]
    boxes = []
    for m in masks:
        r, c = np.where(m)
        boxes.append((r.min(), r.max(), c.min(), c.max()))
    h = max(b[1] - b[0] for b in boxes) + 2 * pad
    w = max(b[3] - b[2] for b in boxes) + 2 * pad
    out = []
    for im, m, (r0, r1, c0, c1) in zip(imgs, masks, boxes):
        im = im.copy()
        im[~m] = 1.0
        canvas = np.ones((h, w, 3))
        sub = im[r0:r1 + 1, c0:c1 + 1]
        y, x = (h - sub.shape[0]) // 2, (w - sub.shape[1]) // 2
        canvas[y:y + sub.shape[0], x:x + sub.shape[1]] = sub
        out.append(canvas)
    return out

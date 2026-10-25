"""Fetch the third-party colour material the figures use, from its original source, and write
the extracted tables to figures/style/third_party/. Nothing here is typed by hand.

    python3 figures/style/fetch_third_party.py           fetch, extract, write
    python3 figures/style/fetch_third_party.py --check   fetch again and assert the committed tables

Sources (licences and citations in third_party/SOURCES.json):
  * Crameri, Scientific colour maps 8.0.1, Zenodo record 8409685 (MIT licence): the 256-entry
    RGB tables <name>/<name>.txt of the sequential maps listed in CRAMERI.
  * BIDS/colormap (Smith and van der Walt; CC0): option_d.py, the table that became viridis.
  * Machado, Oliveira and Fernandes (2009), simulation matrices of Table 1 on the authors' page:
    protanopia, deuteranopia and tritanopia at severity 1.0.
Downloads go to figures/_fetched/ (git-ignored); only the extracted tables are committed.
"""
import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "figures" / "_fetched"
OUT = HERE / "third_party"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

CRAMERI_URL = "https://zenodo.org/api/records/8409685/files/ScientificColourMaps8.zip/content"
CRAMERI_SHA256 = "fe8ccf8e26d87b6703e96e4da1a6872f14a79e725c7025bc9db034898b8fc02c"
CRAMERI = ["batlow", "lajolla", "lipari", "davos", "oslo", "acton", "bilbao", "devon", "imola", "glasgow"]
VIRIDIS_URL = "https://raw.githubusercontent.com/BIDS/colormap/master/option_d.py"
VIRIDIS_LICENSE_URL = "https://raw.githubusercontent.com/BIDS/colormap/master/LICENSE.txt"
MACHADO_URL = "https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html"


def fetch(url, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        subprocess.run(["curl", "-sSfL", "-m", "600", "-A", UA, "-o", str(path), url], check=True)
    return path.read_bytes()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def crameri():
    raw = fetch(CRAMERI_URL, "ScientificColourMaps8.zip")
    assert sha(raw) == CRAMERI_SHA256, "Crameri archive differs from version 8.0.1"
    zf = zipfile.ZipFile(io.BytesIO(raw))
    out = {}
    for name in CRAMERI:
        txt = zf.read(f"{name}/{name}.txt").decode()
        rows = [[float(x) for x in line.split()] for line in txt.strip().splitlines()]
        assert len(rows) == 256 and all(len(r) == 3 and all(0 <= v <= 1 for v in r) for r in rows), name
        out[f"crameri_{name}.txt"] = txt
    return out, {"url": CRAMERI_URL, "sha256": CRAMERI_SHA256, "version": "8.0.1", "doi": "10.5281/zenodo.8409685",
                 "licence": "MIT"}


def viridis():
    src = fetch(VIRIDIS_URL, "bids_option_d.py").decode()
    lic = fetch(VIRIDIS_LICENSE_URL, "bids_LICENSE.txt").decode()
    assert "CC0" in lic
    body = src[src.index("cm_data = [") + len("cm_data = "):]
    body = body[:body.index("]]") + 2]
    rows = ast.literal_eval(body)
    assert len(rows) == 256, len(rows)
    txt = "".join(f"{r:.8f} {g:.8f} {b:.8f}\n" for r, g, b in rows)
    return {"viridis.txt": txt}, {"url": VIRIDIS_URL, "sha256": sha(src.encode()), "licence": "CC0 1.0",
                                  "licence_url": VIRIDIS_LICENSE_URL}


def machado():
    page = fetch(MACHADO_URL, "machado_cvd.html").decode("utf-8", "replace")
    seg = page[page.index("Simulation matrices"):]
    nums = re.findall(r">\s*(-?\d+\.\d+)\s*<", seg)[:11 * 28]
    table = {}
    for b in range(11):
        sev, vals = nums[28 * b], [float(x) for x in nums[28 * b + 1:28 * b + 28]]
        assert abs(float(sev) - b / 10) < 1e-9, sev
        table[sev] = {k: [vals[9 * i + 3 * r:9 * i + 3 * r + 3] for r in range(3)]
                      for i, k in enumerate(("protan", "deutan", "tritan"))}
    one = table["1.0"]
    for k, M in one.items():                      # rows sum to 1: white maps to white
        assert all(abs(sum(r) - 1) < 2e-6 for r in M), k
    assert table["0.0"]["protan"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    data = {"source": MACHADO_URL, "severity": 1.0, "space": "linear RGB (Machado et al. 2009, Eq. 1)",
            "protanopia": one["protan"], "deuteranopia": one["deutan"], "tritanopia": one["tritan"]}
    return {"machado2009_cvd.json": json.dumps(data, indent=1) + "\n"}, {"url": MACHADO_URL, "sha256": sha(page.encode())}


def main():
    files, sources = {}, {}
    for fn, key in ((crameri, "crameri"), (viridis, "viridis"), (machado, "machado2009")):
        f, s = fn()
        files.update(f)
        sources[key] = s
    if "--check" in sys.argv:
        for name, txt in files.items():
            assert (OUT / name).read_text() == txt, f"{name} differs from the fetched source"
        print(f"third_party: {len(files)} tables equal their fetched sources")
        return
    OUT.mkdir(exist_ok=True)
    for name, txt in files.items():
        (OUT / name).write_text(txt)
    (OUT / "SOURCES.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(files)} tables to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

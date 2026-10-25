"""Shared helpers for the figure-data generators (figures/gen/).

Every generator recomputes its table from the committed scripts of theory/ (imported, never
copied), writes a CSV to figures/data/, and asserts the table against the committed
transcript or data it must reproduce. A failed assertion exits nonzero.
"""
import ast
import contextlib
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "figures" / "data"


def import_from(directory, module):
    """Import `module` from `directory` (relative to the repository root), with its own
    directory first on sys.path, as the module's own scripts expect. Returns (module, stdout)."""
    d = str(ROOT / directory)
    if d not in sys.path:
        sys.path.insert(0, d)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = __import__(module)
    return mod, buf.getvalue()


def extract_functions(path, names, namespace):
    """Execute only the named top-level function definitions of a script (no module body).
    Used for scripts whose module body is a long verification run."""
    tree = ast.parse((ROOT / path).read_text(), filename=str(path))
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in found} == set(names), f"{path}: missing {set(names) - {n.name for n in found}}"
    code = compile(ast.Module(body=found, type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace


def write_csv(name, header, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def read_csv(path):
    with (ROOT / path).open() as fh:
        return list(csv.DictReader(fh))

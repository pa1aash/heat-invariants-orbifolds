"""Import a module from a directory of the repository with that directory first on sys.path
(as the module's own scripts expect), keeping its import-time output out of the figure log."""
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def import_from(directory, module):
    d = str(ROOT / directory)
    if d not in sys.path:
        sys.path.insert(0, d)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = __import__(module)
    return mod, buf.getvalue()

#!/usr/bin/env python3
"""Compare two JSON files for equality, ignoring named keys (at any depth).

    json_equal.py A.json B.json [--ignore KEY ...]

Used by the make steps for outputs that carry a wall-clock field (e.g. sharpness_n5_N60.json holds
"wall_seconds"): the stage is run in a scratch copy of its directory and its result is compared
with the committed file on every field except the timings. Exits 1 and prints the first
differing path otherwise.
"""
import json
import sys


def strip(o, ignore):
    if isinstance(o, dict):
        return {k: strip(v, ignore) for k, v in o.items() if k not in ignore}
    if isinstance(o, list):
        return [strip(v, ignore) for v in o]
    return o


def first_diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: present in only one file"
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    ignore = set(argv[argv.index("--ignore") + 1:]) if "--ignore" in argv else set()
    a = strip(json.load(open(argv[1])), ignore)
    b = strip(json.load(open(argv[2])), ignore)
    d = first_diff(a, b)
    if d:
        print(f"json_equal: {argv[1]} and {argv[2]} differ at {d}", file=sys.stderr)
        return 1
    print(f"json_equal: identical apart from {sorted(ignore) or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

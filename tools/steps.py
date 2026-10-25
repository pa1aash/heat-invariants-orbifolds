#!/usr/bin/env python3
"""Run the steps of one make target and report PASS, FAIL or SKIP for each.

    python3 tools/steps.py TARGET        (TARGET: verify, full, figures, tables, data-check, all)
    python3 tools/steps.py --list        (every step with its targets and expected time)

The steps are listed in tools/steps.json. A step passes when its command exits 0 and, if it has a
transcript, when its standard output equals the committed transcript (apart from printed run times).
A step also fails if it changes a file listed in SHA256SUMS: regenerating committed data is a
reproduction test, so the result must be byte-identical. A changed file is restored and reported.

Interpreters (environment variables):
    PYTHON            the core environment (default: the interpreter running this script)
    PYTHON_NUMERICS   an interpreter with NGSolve, for the finite-element steps (default: PYTHON)
    PYTHON_PARI       an interpreter with cypari2, for the PARI steps (default: PYTHON)
    BLENDER           the Blender executable, for the rendered figures (default: blender on PATH)
    CC                the C compiler, for the steps with C programs (default: cc)
Steps whose optional tool is absent are skipped with the reason; a skip is never a failure.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = json.loads((ROOT / "tools" / "steps.json").read_text(encoding="utf-8"))
SUMS = ROOT / "SHA256SUMS"
TARGETS = ("verify", "full", "figures", "tables", "data-check")


def tty(code):
    return code if sys.stdout.isatty() else ""


GREEN, RED, YELLOW, BOLD, RESET = (tty(c) for c in ("\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"))


def fmt(sec):
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums():
    out = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def interpreters():
    py = os.environ.get("PYTHON") or sys.executable
    return {
        "python": py,
        "ngsolve": os.environ.get("PYTHON_NUMERICS") or py,
        "pari": os.environ.get("PYTHON_PARI") or py,
        "blender": os.environ.get("BLENDER") or shutil.which("blender") or "",
        "cc": os.environ.get("CC") or shutil.which("cc") or "",
    }


def available(need, it):
    """Return None if the tool needed by a step is present, else the reason for skipping."""
    if need == "python":
        return None
    if need == "ngsolve":
        ok = subprocess.run([it["ngsolve"], "-c", "import ngsolve"], capture_output=True).returncode == 0
        return None if ok else ("NGSolve not installed: create the optional environment (requirements-optional.txt, "
                                "README section 'Optional components') and set PYTHON_NUMERICS")
    if need == "pari":
        ok = subprocess.run([it["pari"], "-c", "import cypari2"], capture_output=True).returncode == 0
        return None if ok else "cypari2 (PARI/GP) not installed: pip install -r requirements-pari.txt, or set PYTHON_PARI"
    if need == "blender":
        return None if it["blender"] else "Blender not found: install Blender (README, 'Optional components') or set BLENDER"
    if need == "cc":
        return None if it["cc"] else "no C compiler found: install one or set CC"
    raise SystemExit(f"unknown requirement {need!r}")


NORM = re.compile(r"\d+(\.\d+)?s(?=[\],)]|$)", re.M)


def normalise(text):
    text = NORM.sub("<t>", text)
    return "\n".join(l for l in text.splitlines() if l.strip() != "exit 0")


def run_step(step, it, logdir):
    env = dict(os.environ)
    env.update({"PYTHON": it["python"], "PYTHON_NUMERICS": it["ngsolve"], "PYTHON_PARI": it["pari"],
                "BLENDER": it["blender"], "CC": it["cc"], "LOGDIR": str(logdir)})
    cwd = ROOT / step.get("dir", ".")
    log = logdir / (re.sub(r"[^A-Za-z0-9]+", "_", step["name"]) + ".log")
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(step["cmd"], shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        fh.write(proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-15:])
        return False, f"exit {proc.returncode}; last lines:\n{tail}\n(full output in {log})"
    if step.get("transcript"):
        want = (ROOT / step["transcript"]).read_text(encoding="utf-8", errors="replace")
        if normalise(proc.stdout) != normalise(want):
            return False, f"output differs from {step['transcript']} (output in {log})"
    return True, ""


def main(argv):
    sys.stdout.reconfigure(line_buffering=True)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        for s in STEPS:
            print(f"{s['name']:<48} {','.join(s['targets']):<24} {s.get('needs', 'python'):<8} ~{s.get('minutes', 0)} min")
        return 0
    target = argv[0]
    if target not in TARGETS + ("all",):
        raise SystemExit(f"unknown target {target!r}; one of: {', '.join(TARGETS + ('all',))}")
    wanted = TARGETS if target == "all" else (target,)

    it = interpreters()
    logdir = Path(tempfile.mkdtemp(prefix=f"steps-{target}-"))
    sums = read_sums()
    backup = Path(tempfile.mkdtemp(prefix="steps-backup-"))
    results, failed = [], False
    t_all = time.time()
    print(f"{BOLD}make {target}{RESET}   python {it['python']}   logs in {logdir}\n")

    for tgt in wanted:
        if tgt == "data-check":
            t0 = time.time()
            bad = [n for n, d in sums.items() if not (ROOT / n).is_file() or sha256(ROOT / n) != d]
            ok = not bad
            name = f"data-check: {len(sums)} files against SHA256SUMS"
            results.append((name, "PASS" if ok else "FAIL", time.time() - t0, "" if ok else f"{len(bad)} differ: {bad[:5]}"))
            print(f"{GREEN if ok else RED}{'PASS' if ok else 'FAIL'}{RESET}  {name}" + ("" if ok else f"  ({', '.join(bad[:5])})"))
            failed |= not ok
            continue
        for step in [s for s in STEPS if tgt in s["targets"]]:
            if any(r[0] == step["name"] for r in results):
                continue  # a step listed under several targets runs once in 'all'
            reason = available(step.get("needs", "python"), it)
            if reason:
                results.append((step["name"], "SKIP", 0, reason))
                print(f"{YELLOW}SKIP{RESET}  {step['name']}  ({reason})")
                continue
            for n in sums:  # snapshot the committed data
                dst = backup / n
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ROOT / n, dst)  # fresh timestamps: old ones invite the temporary-folder cleaner
            t0 = time.time()
            ok, msg = run_step(step, it, logdir)
            changed = [n for n, d in sums.items() if not (ROOT / n).is_file() or sha256(ROOT / n) != d]
            for n in changed:
                shutil.copyfile(backup / n, ROOT / n)
            if changed:
                ok = False
                msg = (msg + "\n" if msg else "") + f"the step changed committed data (restored): {', '.join(changed[:5])}"
            dt = time.time() - t0
            results.append((step["name"], "PASS" if ok else "FAIL", dt, msg))
            print(f"{GREEN if ok else RED}{'PASS' if ok else 'FAIL'}{RESET}  {step['name']}  ({fmt(dt)})")
            if not ok:
                print("      " + msg.replace("\n", "\n      "))
            failed |= not ok
    shutil.rmtree(backup, ignore_errors=True)

    n = {k: sum(1 for r in results if r[1] == k) for k in ("PASS", "FAIL", "SKIP")}
    print(f"\n{BOLD}summary of make {target}{RESET}: {n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped, "
          f"total {fmt(time.time() - t_all)}")
    for name, status, _, msg in results:
        if status == "SKIP":
            print(f"  SKIP {name}: {msg}")
    if failed:
        print(f"{RED}{BOLD}FAIL{RESET}")
        return 1
    if n["PASS"] == 0:
        print(f"{RED}{BOLD}FAIL{RESET}: no step ran")
        return 1
    print(f"{GREEN}{BOLD}PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

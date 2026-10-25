"""Driver for search_T3_modular.c: the exact modular exclusion behind Remark 3.14 (the paper section 6).

    python3 search_T3_modular.py [NMAX] [JOBS]      (defaults 220 and 4; about 15 minutes at 220 on 4 cores)
    python3 search_T3_modular.py --check            (no search: re-decide the committed survivors exactly)

Steps:
  1. compile search_T3_modular.c (clang -O2) into a temporary directory;
  2. control: every 3-multiset in [1, NMAX] passes the same modular test (it must: then the cubic of
     the test is the triple's own polynomial), so the table and the formula reject no genuine split;
  3. the search over all 5-multisets V of [1, NMAX] with gcd 1, split by the least entry into JOBS
     interleaved processes, run one after another in a single batch (one heavy job);
  4. the total count is asserted against the Moebius count of primitive 5-multisets of [1, NMAX]
     (4,325,115,770 at NMAX = 220);
  5. every survivor is decided exactly: its cubic q_V (rational coefficients, sympy) is factored over
     Q, and the search passes only if no survivor's q_V splits into three rational linear factors.
Writes data/T3_modular.json (counts, primes, survivors with their factorisations) and prints a
transcript (kept as output/search_T3_modular.txt).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from fractions import Fraction as F
from math import comb

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "T3_modular.json")
PRIMES = [223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277]


def mobius(n):
    res, k, m = 1, 2, n
    while k * k <= m:
        if m % k == 0:
            m //= k
            if m % k == 0:
                return 0
            res = -res
        k += 1
    return -res if m > 1 else res


def primitive_count(nmax, size=5):
    """Number of multisets of `size` elements of [1, nmax] with gcd 1 (Moebius inversion)."""
    return sum(mobius(d) * comb(nmax // d + size - 1, size) for d in range(1, nmax + 1))


def cubic(V):
    t = sp.symbols("t")
    S = sum(V)
    C = sum(v ** 3 for v in V)
    R = sum(F(1, v) for v in V)
    e3 = F(C - S ** 3) / (3 * (1 - S * R))
    e2 = R * e3
    P = sp.Poly(t ** 3 - S * t ** 2 + sp.Rational(e2.numerator, e2.denominator) * t
                - sp.Rational(e3.numerator, e3.denominator), t)
    return P


def decide(V):
    P = cubic(V)
    fl = P.factor_list()[1]
    linear = sum(e for f, e in fl if f.degree() == 1)
    return linear, str(P.as_expr()), [str(f.as_expr()) + (f"^{e}" if e > 1 else "") for f, e in fl]


def main():
    if "--check" in sys.argv:
        rec = json.load(open(OUT))
        for s in rec["survivors"]:
            lin, _, _ = decide(s["V"])
            assert lin == s["rational_linear_factors"] and lin < 3, s
        assert rec["total"] == primitive_count(rec["nmax"]) and rec["control_rejected"] == 0
        print(f"T3 modular record: {rec['total']} primitive 5-multisets, {len(rec['survivors'])} survivors, "
              f"none splits over Q (re-decided exactly)")
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nmax = int(args[0]) if args else 220
    jobs = int(args[1]) if len(args) > 1 else 4
    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "search_T3_modular")
    subprocess.run(["clang", "-O2", "-o", exe, os.path.join(HERE, "search_T3_modular.c")], check=True)
    t0 = time.time()
    c = subprocess.run([exe, "--control", str(nmax)], capture_output=True, text=True)
    assert c.returncode == 0, c.stdout[:2000]
    ncontrol, nrej = (int(x) for x in c.stderr.split()[1::2])
    assert ncontrol == comb(nmax + 2, 3) and nrej == 0
    print(f"control: all {ncontrol} 3-multisets of [1, {nmax}] pass the modular test ({time.time() - t0:.0f} s)")
    procs = [subprocess.Popen([exe, str(nmax), str(1 + j), str(jobs)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for j in range(jobs)]
    total, survivors = 0, []
    for p in procs:
        out, err = p.communicate()
        assert p.returncode == 0, err
        total += int(err.split()[1])
        survivors += [list(map(int, line.split()[1:])) for line in out.splitlines() if line.startswith("SURV")]
    secs = time.time() - t0
    expect = primitive_count(nmax)
    assert total == expect, (total, expect)
    print(f"search: {total} primitive 5-multisets of [1, {nmax}] (= Moebius count), "
          f"{len(survivors)} split modulo all {len(PRIMES)} primes ({secs:.0f} s)")
    rec_s = []
    for V in sorted(survivors):
        lin, poly, fac = decide(V)
        assert lin < 3, ("a rational partner triple exists", V, fac)
        rec_s.append(dict(V=V, cubic=poly, factors=fac, rational_linear_factors=lin))
        print(f"  V = {V}: q_V has {lin} rational linear factors; factors {fac}")
    rec = dict(nmax=nmax, primes=PRIMES, total=total, control_triples=ncontrol, control_rejected=0,
               survivors=rec_s)
    with open(OUT, "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    print(f"no 5-multiset with entries <= {nmax} and gcd 1 has a rational partner triple: "
          f"T_g(3) = 8 is excluded in this range")


if __name__ == "__main__":
    main()

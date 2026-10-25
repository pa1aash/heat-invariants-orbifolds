#!/usr/bin/env python3
"""
Mordell-Weil rank certification for the curves C_lambda that the
recommendation relies on.

Pinned environment (the only versions these outputs were produced with):

    python3 -m venv .venv --system-site-packages
    .venv/bin/pip install cypari2==2.2.4     # bundles PARI/GP 2.17.2
    .venv/bin/python ranks.py

Without cypari2 the script prints a clear SKIP message and exits 0, so it
can sit in a suite that does not have PARI.

Model. C_lambda : (x+y+z)(xy+yz+zx) = lambda xyz, lambda = A/B, is mapped
Q-birationally
to the Bremner-Guy-Nowakowski model
    tau^2 = sigma (sigma^2 + (lambda^2 - 6 lambda - 3) sigma + 16 lambda),
and X = B^2 sigma, Y = B^3 tau gives the integral model passed to PARI:
    Y^2 = X (X^2 + (A^2 - 6AB - 3B^2) X + 16 A B^3),  i.e. ellinit([0, a2, 0, a4, 0]).

Reading the output (PARI manual, "ellrank", fetched from
pari.math.u-bordeaux.fr/dochtml/html/Elliptic_curves.html): the result is
[r1, r2, s, L] with r1 <= rank <= r2 always; r2 = C - T - s where C is the
2-Selmer rank, T the 2-torsion rank and s the rank of Sha[2]/2Sha[4], all
computed unconditionally. A rank is PROVEN when r1 = r2. The lower bound is
additionally certified here: each point of L is checked to lie on the model
exactly, and the height-pairing determinant of L is reported (non-zero means
independent).

Writes data/ranks.txt.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED_CYPARI2 = "2.2.4"
PINNED_PARI = (2, 17, 2)

CURVES = [
    (Fraction(27, 2), "base pair {(2,8,8),(3,3,12)}: isolation theorem"),
    (Fraction(155, 12), "(4,9,18),(5,6,20) at S = 31: large-fibre theorem"),
    (Fraction(68, 5), "first fibre of size 3 (S = 136) and of size 4 (S = 408)"),
    (Fraction(1849, 120), "first fibre of size 5 (S = 1849)"),
    (Fraction(230, 21), "first fibre of size 6 (S = 4600)"),
    (Fraction(169, 16), "torsion Z/12 with r = -4: the pairs of torsion type"),
]


def coefficients(lam: Fraction) -> list[int]:
    A, B = lam.numerator, lam.denominator
    return [0, A * A - 6 * A * B - 3 * B * B, 0, 16 * A * B ** 3, 0]


def main() -> int:
    try:
        import cypari2
    except ImportError:
        print("SKIP  ranks.py: cypari2 is not installed. Install the pinned version with\n"
              f"      pip install cypari2=={PINNED_CYPARI2}   (PARI/GP {'.'.join(map(str, PINNED_PARI))})\n"
              "      and rerun. No rank is asserted without it.")
        return 0
    import importlib.metadata as md

    pari = cypari2.Pari()
    pari.allocatemem(10 ** 9)
    ver = tuple(int(v) for v in pari.version())
    cyver = md.version("cypari2")
    out = [f"cypari2 {cyver}, PARI/GP {'.'.join(map(str, ver))}"]
    if cyver != PINNED_CYPARI2 or ver != PINNED_PARI:
        out.append(f"WARNING: pinned cypari2 {PINNED_CYPARI2} / PARI {PINNED_PARI}; outputs may differ")

    proven = {}
    for lam, why in CURVES:
        a = coefficients(lam)
        call = f"E = ellinit({a}); ellrank(E); elltors(E); ellrootno(E)"
        E = pari.ellinit(a)
        r = pari.ellrank(E)
        tors = pari.elltors(E)
        root = pari.ellrootno(E)
        cond = pari.ellglobalred(E)[0]          # the conductor (for comparison with the LMFDB)
        r1, r2, s = int(r[0]), int(r[1]), int(r[2])
        pts = list(r[3])
        for P in pts:
            assert bool(pari.ellisoncurve(E, P)), (lam, P)
        det = pari.matdet(pari.ellheightmatrix(E, pts)) if pts else None
        status = "PROVEN" if r1 == r2 else "BOUNDED"
        proven[lam] = (r1, r2, status)
        out += [
            "",
            f"lambda = {lam}   ({why})",
            f"  call:     {call}",
            f"  ellrank:  {r}",
            f"  elltors:  {tors}",
            f"  rootno:   {root}",
            f"  conductor: {cond}",
            f"  rank:     {r1} <= rank <= {r2}   -> {status}",
            f"  points:   {len(pts)} returned, all on the model; height-matrix det = {det}",
        ]
    r1, r2, st = proven[Fraction(169, 16)]
    assert (r1, r2, st) == (1, 1, "PROVEN"), "rank(C_{169/16}) = 1 is not proven"
    # the isolation theorem needs an unconditional upper bound of 0
    r1, r2, st = proven[Fraction(27, 2)]
    assert r2 == 0 and st == "PROVEN", "rank(C_{27/2}) = 0 is not proven"
    out += ["", "ASSERT  rank(C_{27/2}) upper bound r2 = 0 (unconditional 2-descent bound): PASS"]

    # With rank 0, C_{27/2}(Q) is its torsion group, of order 12 by elltors. Exhibit the
    # 12 points on the plane cubic with exact chord-tangent arithmetic and list the positive ones.
    sys.path.insert(0, str(HERE))
    from cubic_group import TRIVIAL, Cubic, positive_triple
    C = Cubic(Fraction(27, 2))
    assert int(pari.elltors(pari.ellinit(coefficients(Fraction(27, 2))))[0]) == 12
    P = (1, 4, 4)
    assert C.order(P) == 6
    pts = set(TRIVIAL) | {C.add(P, T) for T in TRIVIAL}
    assert len(pts) == 12 and all(C.on(Q) for Q in pts)
    positive = sorted({positive_triple(Q) for Q in pts} - {None})
    assert positive == [(1, 1, 4), (1, 4, 4)], positive
    out.append("ASSERT  C_{27/2}(Q) = 12 explicit points; positive ones are permutations of "
               "(1,4,4) and (1,1,4) only: PASS  -> isolation theorem proven")
    text = "\n".join(out) + "\n"
    print(text)
    (HERE / "data" / "ranks.txt").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

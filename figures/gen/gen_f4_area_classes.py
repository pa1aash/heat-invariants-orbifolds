"""F4 data: one row per complete area class, and the Theorem N construction points.

    python3 figures/gen/gen_f4_area_classes.py        (about one minute)

Writes
  figures/data/f4_area_classes.csv   s = Area/2pi, class size, the bound floor(2s)+4 of Theorem S,
                                     the lower bound floor(log_4(s+1))+2 of Corollary N1, the
                                     largest K_mult (and K_n, K_g) in the class, one extremal
                                     signature and a partner sharing K_mult - 1 coefficients;
  figures/data/f4_construction.csv   the Theorem N(a) pairs for L = 2..6: their s, cone counts,
                                     and the number of shared coefficients (so K_mult >= L + 1).

The per-class numbers are those computed by theory/signatures/area_classes.py, imported
unchanged: its module body enumerates every class and keeps them in `rows`. The construction
is genus_pair() of theory/signatures/genus.py, executed alone (the module body of genus.py is
a 3.5-minute verification run that is not needed here).

Asserted: the imported run prints exactly the committed transcript output/area_classes.txt;
the CSV reproduces its histograms and its "largest max K_mult" list; every class respects
max K_mult < floor(2s)+4; every construction point shares exactly L coefficients, has
s < 4^(L-1) - 1 and matches the cone counts and largest order printed in output/genus.txt.
"""
import re
import sys
from collections import Counter
from fractions import Fraction

from common import ROOT, extract_functions, import_from, write_csv

SIG = "theory/signatures"


def floor_log4_plus2(s):
    """floor(log_4(s + 1)) + 2, exactly: the largest j with 4^j <= s + 1, plus 2."""
    j = 0
    while 4 ** (j + 1) <= s + 1:
        j += 1
    return j + 2


def fmt_sig(sig):
    g, m = sig
    return f"({g};{','.join(map(str, m))})"


def main():
    # ---------------------------------------------------------------- per-class table
    ac, out = import_from(SIG, "area_classes")
    transcript = (ROOT / SIG / "output" / "area_classes.txt").read_text().splitlines()
    body = [l for l in transcript if not l.startswith(("/", "python3 -u ")) and l != "EXIT 0"]
    assert out.splitlines() == body, "area_classes.py output differs from output/area_classes.txt"
    assert ac.failures == 0

    sig_common = sys.modules["sig_common"]
    rows = sorted(ac.rows, key=lambda r: r[0])
    table = []
    for s, size, bound, kmax, kn, kg, argmax in rows:
        assert bound == int(2 * s) + 4 and kmax < bound
        sig = argmax[0]
        partner = ""
        if kmax >= 2:
            cls = ac.area_class(s)
            partner = next(o for o in cls if o != sig and sig_common.shared_int(sig, o, bound + 2) == kmax - 1)
            partner = fmt_sig(partner)
        table.append([str(s), f"{float(s):.12g}", size, bound, floor_log4_plus2(s), kmax, kn, kg,
                      fmt_sig(sig), partner])
    write_csv("f4_area_classes.csv",
              ["s", "s_float", "class_size", "upper_bound_floor_2s_plus_4", "lower_bound_floor_log4_s_plus_1_plus_2",
               "max_K_mult", "max_K_n", "max_K_g", "extremal_signature", "partner_sharing_K_mult_minus_1"], table)

    # the CSV reproduces the transcript's summary lines
    def hist_line(prefix):
        line = next(l for l in transcript if l.startswith(prefix))
        return {int(a): int(b) for a, b in re.findall(r"(\d+): (\d+)", line.split(":", 1)[1])}
    assert len(table) == 525
    assert sum(r[2] for r in table) == 33946
    assert dict(Counter(r[5] for r in table)) == hist_line("max_O K_mult over the class, histogram")
    assert dict(Counter(r[3] - r[5] for r in table)) == hist_line("bound floor(2s)+4 minus max K_mult, histogram")
    listed = [l for l in transcript if l.startswith("  s=")]
    assert len(listed) == 12
    by_s = {r[0]: r for r in table}
    for l in listed:
        s, size, bound, kmax, kn, kg = re.match(
            r"  s=(\S+): \|class\|=(\d+), bound=(\d+), max K_mult=(\d+), max K_n=(\d+), max K_g=(\d+);", l).groups()
        r = by_s[s]
        assert (r[2], r[3], r[5], r[6], r[7]) == tuple(map(int, (size, bound, kmax, kn, kg))), l

    # ---------------------------------------------------------------- Theorem N(a) points
    ns = {"Fraction": Fraction, "thue_morse": sig_common.thue_morse}
    extract_functions(f"{SIG}/genus.py", ["genus_pair", "R", "check"], ns)
    ns["failures"] = 0
    gtext = (ROOT / SIG / "output" / "genus.txt").read_text()
    cons = []
    for L in range(2, 7):
        U, V, r0, rho = ns["genus_pair"](L)
        assert sig_common.odd_balanced(U, V, L)
        a, b = sig_common.realise(U, V)
        assert a[0] == 1 and b[0] == 0
        k = sig_common.shared_int(a, b, L + 2)
        assert k == L, (L, k)
        s = sig_common.s_of(*a)
        assert s == sig_common.s_of(*b) and 0 < s < 4 ** (L - 1) - 1
        assert floor_log4_plus2(s) <= L + 1          # Corollary N1 is weaker than the point itself
        m = re.search(rf"L={L}: \(1; (\d+) cones\) vs \(0; (\d+) cones\), max order (\d+), .*shares exactly (\d+)", gtext)
        assert m and (len(a[1]), len(b[1]), max(U + V), L) == tuple(map(int, m.groups())), L
        cons.append([L, f"{float(s):.15g}", str(s) if s.denominator < 10 ** 30 else "", len(a[1]), len(b[1]),
                     k, L + 1, 4 ** (L - 1) - 1])
    write_csv("f4_construction.csv",
              ["L", "s_float", "s_exact_if_short", "cones_genus1", "cones_genus0", "shared_coefficients",
               "K_mult_at_least", "s_bound_4_pow_L_minus_1_minus_1"], cons)
    print(f"f4: {len(table)} classes, {sum(r[2] for r in table)} signatures; construction L = 2..6 at s = "
          + ", ".join(f"{float(c[1]):.4g}" for c in cons) + "; all assertions passed")


if __name__ == "__main__":
    main()

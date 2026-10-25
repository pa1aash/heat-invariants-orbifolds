#!/usr/bin/env python3
"""Regenerate the generated tables of the supplement (tables/supplement_tables.tex) from committed data.

Each table is written between the markers
    % BEGIN GENERATED TABLE <name>
    % END GENERATED TABLE <name>
in tables/supplement_tables.tex.

    python3 tools/make_tables.py           # rewrite the tables in place
    python3 tools/make_tables.py --check   # exit 1 if any table is out of date

Sources (all committed):
  overlap     theory/threshold/first_overlap_vs_collision.csv
  thresholds_full  theory/stability/threshold_results.json (Table S4)
  enum        checks/enumeration6000/tab_enum_reference.txt, re-derived here by enumeration
"""
import csv
import json
import math
import re
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sig(x, digits, mode):
    """x rounded to `digits` significant figures, down or up, as a LaTeX a\\times10^{b}."""
    d = Decimal(repr(float(x))) if not isinstance(x, Fraction) else Decimal(x.numerator) / Decimal(x.denominator)
    e = d.adjusted()
    q = Decimal(1).scaleb(e - digits + 1)
    r = d.quantize(q, rounding=ROUND_FLOOR if mode == "down" else ROUND_CEILING)
    if r.adjusted() != e:  # rounding up crossed a power of ten
        e = r.adjusted()
    mant = r.scaleb(-e)
    mant = f"{mant:.{digits - 1}f}"
    return f"{mant}\\times10^{{{e}}}"


def tex_triple(t):
    return "(" + ",".join(str(x) for x in t) + ")"


# ---------------------------------------------------------------- overlap
def table_overlap():
    rows = list(csv.DictReader(open(ROOT / "theory/threshold/first_overlap_vs_collision.csv")))
    out = []
    for r in rows:
        p = int(r["p"])
        if p > 14:
            continue
        xs = Fraction(r["x_star"])
        xs_t = str(xs.numerator) if xs.denominator == 1 else f"{xs.numerator}/{xs.denominator}"
        pair = " and ".join("$" + tex_triple(eval(t)) + "$" for t in r["triads"].split(";"))
        out.append(f"{p} & ${xs_t}$ & {r['S_star']} & {r['first_collision_S']} & {r['gap']} & {pair} \\\\")
    assert [int(r["p"]) for r in rows if int(r["p"]) <= 14] == list(range(2, 15))
    body = "\n".join(out)
    return (
        "\\begin{table}[t]\n"
        "\\caption{First overlap and first collision of the adjacent strata $p$ and $p+1$, for $2\\le p\\le14$, "
        "from the exhaustive enumeration of all hyperbolic triads with $S\\le600$. The gap is the first collision sum "
        "minus $S^*(p)$; in this table it vanishes only for $p=2$ and $p=4$ (compare Proposition~\\pref{prop:tangency}).}\\label{tab:overlap}\n"
        "\\centering\\small\n"
        "\\begin{tabular}{@{}rrrrrl@{}}\n\\toprule\n"
        "$p$ & $x^*(p)$ & $S^*(p)$ & first collision & gap & colliding pair\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"
    )


# ------------------------------------------------------------- thresholds
def threshold_rows():
    """One dict per multiset of theory/stability/threshold_results.json, every entry rounded so that
    it keeps its meaning (delta_thm, delta_cert, eps_cert and the relative precisions down, delta_up up)."""
    d = json.load(open(ROOT / "theory/stability/threshold_results.json"))
    out = []
    for key, v in d.items():
        m = eval(key)
        n = len(m)
        cert_exact = Fraction(v["delta_cert_exact"])
        up = v["delta_up"]
        thm_s = sig(v["delta_thm"], 3, "down")
        cert_s = sig(cert_exact, 4, "down")
        up_s = sig(up, 4, "up")
        ratio = float(Decimal(up_s.split("\\times")[0]).scaleb(int(up_s.split("{")[1].rstrip("}")))) / float(
            Decimal(cert_s.split("\\times")[0]).scaleb(int(cert_s.split("{")[1].rstrip("}"))))
        eps_s = sig(v["eps_cert"], 2, "down")
        # relative precision delta_cert/|c_j| of each coefficient under the uniform model, from the
        # printed (rounded-down) delta_cert and the exact c_j, rounded down
        cert_printed = Decimal(cert_s.split("\\times")[0]).scaleb(int(cert_s.split("{")[1].rstrip("}")))
        cert_printed = Fraction(str(cert_printed))
        rel = [sig(cert_printed / abs(Fraction(h)), 2, "down") for h in v["H"]]
        assert len(rel) == n
        # the relative box of radius min_j delta_cert/|c_j| lies in the uniform box of radius delta_cert
        # and the certificate is monotone in the radii, so eps_cert >= min_j delta_cert/|c_j|
        assert Fraction(repr(v["eps_cert"])) >= min(cert_exact / abs(Fraction(h)) for h in v["H"]), key
        out.append({"m": m, "thm": thm_s, "cert": cert_s, "up": up_s, "ratio": ratio, "eps": eps_s, "rel": rel})
    return out


def _e(x):          # compact a\text{e}b notation for these wide tables
    return x.replace("\\times10^", "\\text{e}")


def table_thresholds():
    body = "\n".join(f"${tex_triple(r['m'])}$ & ${_e(r['thm'])}$ & ${_e(r['cert'])}$ & ${_e(r['up'])}$ \\\\"
                      for r in threshold_rows())
    return (
        "\\begin{table}[t]\n"
        "\\caption{Thresholds for exact recovery of integer orders under the uniform model $|\\delta c_j|\\le\\delta$ "
        "on the heat invariants: the threshold $\\delta_{\\rm thm}$ of Theorem~\\pref{thm:S4}, evaluated with the exact matrix norm, the threshold "
        "$\\delta_{\\rm cert}$ certified in exact arithmetic by Proposition~\\sref{prop:S5} of the supplement, and "
        "a constructed failure $\\delta_{\\rm up}$; aeb means $a\\times10^b$, $\\delta_{\\rm thm}$ and "
        "$\\delta_{\\rm cert}$ are rounded down and $\\delta_{\\rm up}$ up. Table~\\sref{tab:thresholdsfull} "
        "adds the relative precisions.}\\label{tab:thresholds}\n"
        "\\centering\\footnotesize\n"
        "\\begin{tabular}{@{}llll@{}}\n\\toprule\n"
        "$m$ & $\\delta_{\\rm thm}$ & $\\delta_{\\rm cert}$ & $\\delta_{\\rm up}$\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"
    )


def table_thresholds_full():
    out = []
    for r in threshold_rows():
        rel_s = "$, $".join(_e(x) for x in r["rel"])      # separate math groups so the cell can wrap
        out.append(f"${tex_triple(r['m'])}$ & ${_e(r['thm'])}$ & ${_e(r['cert'])}$ & ${_e(r['up'])}$ & "
                   f"{r['ratio']:.2f} & ${_e(r['eps'])}$ & ${rel_s}$ \\\\")
    body = "\n".join(out)
    return (
        "\\begin{table}[ht]\n"
        "\\caption{Thresholds for exact recovery of integer orders under the uniform model "
        "$|\\delta c_j|\\le\\delta$ on the heat invariants: the threshold $\\delta_{\\rm thm}$, evaluated with the exact matrix norm, of "
        "Theorem~\\ref{thm:S4}, the threshold $\\delta_{\\rm cert}$ certified in exact arithmetic "
        "by Proposition~\\ref{prop:S5}, a constructed failure $\\delta_{\\rm up}$, which minimizes "
        "$\\|c(\\tilde q)-c(m)\\|_\\infty$ over real monic $\\tilde q$ with a real root, or a non-real complex pair, of real "
        "part $a\\pm\\frac12$, where rounding is a tie; then the ratio "
        "$\\delta_{\\rm up}/\\delta_{\\rm cert}$; the largest uniform relative error $\\epsilon_{\\rm cert}$ "
        "certified by Proposition~\\ref{prop:S5}, that is, for errors $|\\delta c_j|\\le\\epsilon|c_j|$ for all $j$; "
        "and $\\delta_{\\rm cert}/|c_j|$, $j=1,\\dots,n$, the relative precision that the uniform threshold "
        "$\\delta_{\\rm cert}$ demands of each coefficient (not a tolerance for one coefficient alone). The relative "
        "box of radius $\\min_j\\delta_{\\rm cert}/|c_j|$ lies inside the uniform box of radius $\\delta_{\\rm cert}$ "
        "and the certificate is monotone in the radii, so $\\epsilon_{\\rm cert}\\ge\\min_j\\delta_{\\rm cert}/|c_j|$; "
        "it is larger when the box of the relative model is not the binding one. aeb means $a\\times10^b$; "
        "$\\delta_{\\rm up}$ is rounded up, the ratio $\\delta_{\\rm up}/\\delta_{\\rm cert}$ to the nearest unit of its last digit, and every other entry down.}\\label{tab:thresholdsfull}\n"
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llllrl>{\\raggedright\\arraybackslash}p{30mm}@{}}\n\\toprule\n"
        "$m$ & $\\delta_{\\rm thm}$ & $\\delta_{\\rm cert}$ & $\\delta_{\\rm up}$ & ratio & $\\epsilon_{\\rm cert}$ & "
        "$\\delta_{\\rm cert}/|c_j|$, $j=1,\\dots,n$\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"
    )


# ------------------------------------------------------------------- enum
def table_enum():
    ref = {}
    for line in (ROOT / "checks/enumeration6000/tab_enum_reference.txt").read_text().splitlines():
        m = re.match(r"S=(\d+) \(\d+ triads\): (.*)", line)
        if m:
            ref[int(m.group(1))] = [(eval(t.split()[0]), Fraction(t.split()[1])) for t in m.group(2).split("; ")]
    # re-derive by enumeration
    for S in range(10, 19):
        mine = []
        for p in range(2, S):
            for q in range(p, S):
                r = S - p - q
                if r < q:
                    continue
                R = Fraction(1, p) + Fraction(1, q) + Fraction(1, r)
                if R < 1:
                    mine.append(((p, q, r), R))
        assert mine == ref[S], S
    assert sum(len(v) for v in ref.values()) == 83
    out = []
    for S in range(10, 19):
        items = []
        Rs = [R for _, R in ref[S]]
        for t, R in ref[S]:
            s = f"{tex_triple(t)}\\,{R.numerator}/{R.denominator}"
            items.append(f"$\\mathbf{{{s}}}$" if Rs.count(R) > 1 else f"${s}$")
        joined = "; ".join(items)
        out.append(f"{S} & {joined} \\\\")
    body = "\n".join(out)
    return (
        "\\begin{table}[ht]\n"
        "\\caption{All $83$ hyperbolic triads $(p,q,r)$ with $10\\le S_1\\le18$, each followed by its reciprocal sum "
        "$R$. The only coincidence of $R$ within a sum is the pair in bold.}\\label{tab:enum}\n"
        "\\centering\\footnotesize\n"
        "\\begin{tabular}{@{}rp{0.9\\textwidth}@{}}\n\\toprule\n"
        "$S_1$ & triads and $R$\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"
    )


TABLES = {"overlap": table_overlap, "thresholds": table_thresholds, "thresholds_full": table_thresholds_full,
          "enum": table_enum}
FILES = {ROOT / "tables" / "supplement_tables.tex": ["overlap", "enum", "thresholds_full"]}


def main():
    stale = False
    for path, names in FILES.items():
        src = path.read_text(encoding="utf-8")
        new = src
        for name in names:
            pat = re.compile(r"(% BEGIN GENERATED TABLE " + name + r"\n).*?(% END GENERATED TABLE " + name + r")", re.S)
            if not pat.search(new):
                raise SystemExit(f"markers for table {name} not found in {path.name}")
            new = pat.sub(lambda m, fn=TABLES[name]: m.group(1) + fn() + "\n" + m.group(2), new)
        if "--check" in sys.argv:
            stale |= new != src
            continue
        path.write_text(new, encoding="utf-8")
    if "--check" in sys.argv:
        print("tables are out of date; run make_tables.py" if stale else "tables up to date")
        sys.exit(1 if stale else 0)
    print("tables written")


if __name__ == "__main__":
    main()

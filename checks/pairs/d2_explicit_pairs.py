"""The explicit pairs behind Example S8.6 and Figure 3 (Table S1).

    python3 checks/pairs/d2_explicit_pairs.py            write the CSV and the supplement table
    python3 checks/pairs/d2_explicit_pairs.py --check    exit 1 if the supplement table is stale

Two sources, both re-verified here with the coefficient code of a1_fig2_kmult.py (written
independently of theory/ and figures/):
  * the 18 Fig. 2 diamonds, theory/pte/data/witnesses.json: for each L = 2..7 a genus pair
    (genus 1 vs genus 0), an equal-count genus-0 pair ("balanced"; L = 4..7 are the pairs of
    Example 3.12(iii)) and a genus-0 pair with different cone counts;
  * the 5 Fig. 2 squares, the Thue-Morse (Prouhet) genus pairs for L = 2..6, rebuilt here from
    their closed form and compared with figures/data/f4_construction.csv.
For every pair: both signatures are hyperbolic, the areas are equal, and the number of shared
heat invariants, computed exactly from b_l(m) and again from Psi_k, is exactly L.

Writes checks/pairs/output/d2_pairs.csv (full signatures, exact areas) and the table
between "% BEGIN GENERATED TABLE pairs" and "% END GENERATED TABLE pairs" in
tables/supplement_tables.tex (squares with L >= 4 by their closed form; their cone lists are in the CSV).
"""
import csv
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_fig2_kmult as a1  # noqa: E402

ROOT = a1.ROOT
OUT = Path(__file__).resolve().parent / "output"
SUPP = ROOT / "tables" / "supplement_tables.tex"
KIND = {"genus": "genus", "balanced": "equal count", "cone": "cone count"}


def shared(sig1, sig2):
    """Number of shared heat invariants c_1, c_2, ... of two signatures of equal area."""
    (g1, m1), (g2, m2) = sig1, sig2
    assert a1.s_of(g1, m1) == a1.s_of(g2, m2) > 0
    k = 1 + a1.prefix(a1.cone_vector(m1), a1.cone_vector(m2))
    assert k == 1 + a1.prefix(a1.psi_vector(m1), a1.psi_vector(m2))
    assert k <= a1.LMAX
    return k


def thue_morse(i):
    return bin(i).count("1") & 1


def prouhet_pair(L):
    """The genus pair of Fig. 2 (squares): U (genus 1) and V (genus 0), |V| = |U| + 2.

    With T_e = {0 <= i < 4^(L-1) : thue_morse(i) = e}: U0 = {2i-1 : i in T_0, i != 0},
    V0 = {2i-1 : i in T_1} + {1}, A' = {2i+1 : i in T_0}, B' = {2i+1 : i in T_1}; U0, V0 and A', B'
    have equal odd power sums up to 2L-3; r0 = R(U0) - R(V0), rho = R(A') - R(B') (swap A', B' if
    r0 < 0); p/q = rho/r0 in lowest terms (doubled if min(p,q) = 1); U = qU0 + pB', V = qV0 + pA'.
    """
    K = 2 * L - 2
    T0 = [i for i in range(2 ** K) if thue_morse(i) == 0]
    T1 = [i for i in range(2 ** K) if thue_morse(i) == 1]
    U0 = [2 * i - 1 for i in T0 if i != 0]
    V0 = [2 * i - 1 for i in T1] + [1]
    Ap, Bp = [2 * i + 1 for i in T0], [2 * i + 1 for i in T1]
    R = lambda X: sum(Fraction(1, x) for x in X)
    r0, rho = R(U0) - R(V0), R(Ap) - R(Bp)
    assert r0 != 0 and rho > 0
    if r0 < 0:
        Ap, Bp, rho = Bp, Ap, -rho
    p, q = (rho / r0).numerator, (rho / r0).denominator
    if min(p, q) == 1:
        p, q = 2 * p, 2 * q
    U = sorted([q * u for u in U0] + [p * b for b in Bp])
    V = sorted([q * v for v in V0] + [p * a for a in Ap])
    return (1, tuple(U)), (0, tuple(V)), p, q


def remark_313_partner():
    """Remark 3.13: the 3-element side U sharing P_1, P_3, R with V = {1,1,1,1,7}.

    With e_1 = P_1(V), e_2 = R e_3 and P_3 = e_1^3 - 3 e_1 e_2 + 3 e_3, e_3 = (P_3 - e_1^3)/(3 - 3 e_1 R), so U is
    the root set of z^3 - e_1 z^2 + R e_3 z - e_3: exactly one partner multiset, real and positive here.
    """
    import numpy as np
    V = [1, 1, 1, 1, 7]
    e1 = Fraction(sum(V))
    R = sum(Fraction(1, v) for v in V)
    P3 = Fraction(sum(v ** 3 for v in V))
    e3 = (P3 - e1 ** 3) / (3 - 3 * e1 * R)
    roots = np.roots([1, -float(e1), float(R * e3), -float(e3)])
    assert np.all(np.abs(roots.imag) < 1e-12) and np.all(roots.real > 0)
    U = sorted(roots.real)
    assert abs(sum(U) - float(e1)) < 1e-9 and abs(sum(u ** 3 for u in U) - float(P3)) < 1e-7
    assert abs(sum(1 / u for u in U) - float(R)) < 1e-12
    return U


def fmt_sig(sig):
    g, m = sig
    return f"({g};" + ",\\allowbreak ".join(map(str, m)) + ")"


def fmt_area(s):
    if len(str(s)) <= 22:
        return f"${s.numerator}/{s.denominator}$" if s.denominator > 1 else f"${s}$"
    n = round(s)
    d = s - n
    if d != 0 and abs(d) < Fraction(1, 10 ** 9):      # within 1e-9 of an integer: show the offset
        x = abs(d)                                     # exact exponent: float(x) can underflow to 0
        e = len(str(x.numerator)) - len(str(x.denominator))
        if x < Fraction(10) ** e:
            e -= 1
        m = f"{float(x / Fraction(10) ** e):.2f}"
        if m == "10.00":
            m, e = "1.00", e + 1
        return f"${n}{'+' if d > 0 else '-'}{m}\\times10^{{{e}}}$"
    return f"$\\approx{float(s):.10g}$"


def fmt_exact(s):
    """The exact s = N/D as text with a break point after every ten digits (rows L <= 5)."""
    def brk(n):
        d = str(n)
        return "\\allowbreak ".join(d[i:i + 10] for i in range(0, len(d), 10))
    return f"exact $s=$ {brk(s.numerator)}\\allowbreak /\\allowbreak {brk(s.denominator)}"


def main():
    rows, tex = [], []
    # -------------------------------------------------------------- diamonds
    w = json.load(open(ROOT / "theory/pte/data/witnesses.json"))
    assert len(w) == 18
    for r in w:
        a = (r["sig1"]["g"], tuple(r["sig1"]["orders"]))
        b = (r["sig2"]["g"], tuple(r["sig2"]["orders"]))
        L = int(r["L"])
        s = a1.s_of(*a)
        assert s == Fraction(r["area_over_2pi"]) and min(a[1] + b[1]) >= 2
        k = shared(a, b)
        assert k == L, (r["kind"], L, k)
        rows.append(["diamond", r["kind"], L, a[0], " ".join(map(str, a[1])), b[0], " ".join(map(str, b[1])),
                     str(s), k, r["recipe"]])
        tex.append((L, KIND[r["kind"]], s, k, fmt_sig(a), fmt_sig(b)))
    # -------------------------------------------------------------- squares
    con = list(a1.csv.DictReader(open(ROOT / "figures/data/f4_construction.csv")))
    assert [int(c["L"]) for c in con] == [2, 3, 4, 5, 6]
    for c in con:
        L = int(c["L"])
        a, b, p, q = prouhet_pair(L)
        s = a1.s_of(*a)
        assert (len(a[1]), len(b[1])) == (int(c["cones_genus1"]), int(c["cones_genus0"]))
        assert abs(float(s) - float(c["s_float"])) <= 1e-12 * float(s)
        assert len(a[1]) + len(b[1]) == 2 ** (2 * L - 1) and min(a[1] + b[1]) >= 2
        k = shared(a, b)
        assert k == L == int(c["shared_coefficients"])
        rows.append(["square", "prouhet genus", L, a[0], " ".join(map(str, a[1])), b[0], " ".join(map(str, b[1])),
                     str(s), k, f"Thue-Morse construction, p={p}, q={q}"])
        if L <= 3:
            tex.append((L, "Prouhet", s, k, fmt_sig(a), fmt_sig(b)))
        else:
            digest = hashlib.sha256((" ".join(map(str, a[1])) + "|" + " ".join(map(str, b[1]))).encode()).hexdigest()[:12]
            pq = (f"$p={p}$, $q={q}$" if len(str(p)) + len(str(q)) <= 40 else
                  f"$p,q$ of ${len(str(p))}$ and ${len(str(q))}$ digits (in the CSV)")
            tex.append((L, "Prouhet", s, k,
                        f"$(1;qU_0\\cup pB')$, ${len(a[1])}$ cone points, {pq}",
                        f"$(0;qV_0\\cup pA')$, ${len(b[1])}$ cone points (SHA-256 of both lists {digest}\\dots)"))

    OUT.mkdir(exist_ok=True)
    with open(OUT / "d2_pairs.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["marker", "kind", "L", "genus_1", "orders_1", "genus_2", "orders_2", "area_over_2pi",
                     "shared_exactly", "source"])
        wr.writerows(rows)

    body = []
    for L, kind, s, k, x, y in tex:
        exact = f"\\newline {fmt_exact(s)}" if L <= 5 and "/" not in fmt_area(s) else ""
        body.append(f"{L} & {kind} & {fmt_area(s)} & {k}\\\\*\n"
                    f"\\multicolumn{{4}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\raggedright $\\sigma_1=$ {x}\\newline $\\sigma_2=$ {y}{exact}}}\\\\[2pt]")
    table = (
        "{\\scriptsize\\setlength{\\LTcapwidth}{\\textwidth}\n"
        "\\begin{longtable}{@{}rlll@{}}\n"
        "\\caption{The explicit pairs behind Example~\\ref{ex:ptepairs} and the diamonds and squares of "
        "Figure~\\pref{fig:F4}, each rechecked exactly by a program of the code and data deposit (Section~\\ref{sec:methods}): "
        "both signatures, the area $s=\\mathrm{Area}/2\\pi$ (when an approximation is shown, "
        "the exact value is printed below the signatures for $L\\le5$ and is in the data file for $L\\ge6$) and the exact number $L$ of shared heat invariants; "
        "``equal count'' rows with $L=4,\\dots,7$ are Example~\\ref{ex:ptepairs}(iii). For the Prouhet squares with $L\\ge4$, "
        "$T_e=\\{0\\le i<4^{L-1}:\\ \\text{the binary digits of $i$ have sum}\\equiv e\\ (2)\\}$, "
        "$U_0=\\{2i-1:i\\in T_0,\\ i\\ne0\\}$, $V_0=\\{2i-1:i\\in T_1\\}\\cup\\{1\\}$, $A'=\\{2i+1:i\\in T_0\\}$, "
        "$B'=\\{2i+1:i\\in T_1\\}$, with $A'$ and $B'$ exchanged when $R(U_0)<R(V_0)$; their cone lists are in a data file of the same repository.}"
        "\\label{tab:pairs}\\\\\n"
        "\\toprule\n$L$ & kind & $s$ & shared\\\\\n\\midrule\n\\endfirsthead\n"
        "\\toprule\n$L$ & kind & $s$ & shared\\\\\n\\midrule\n\\endhead\n"
        "\\bottomrule\n\\endfoot\n"
        + "\n".join(body) + "\n\\end{longtable}}"
    )
    src = SUPP.read_text(encoding="utf-8")
    pat = re.compile(r"(% BEGIN GENERATED TABLE pairs\n).*?(% END GENERATED TABLE pairs)", re.S)
    assert pat.search(src), "markers for table pairs not found in supplement_tables.tex"
    new = pat.sub(lambda m: m.group(1) + table + "\n" + m.group(2), src)
    if "--check" in sys.argv:
        stale = new != src
        print("pairs table out of date" if stale else "pairs table up to date")
        sys.exit(1 if stale else 0)
    SUPP.write_text(new, encoding="utf-8")
    U = remark_313_partner()
    print("Remark 3.13: {1,1,1,1,7} has exactly one partner triple, of positive reals "
          + ", ".join(f"{u:.4f}" for u in U))
    print(f"D2: {len(rows)} pairs (18 diamonds, 5 squares) verified: equal areas, hyperbolic, "
          f"shared exactly L heat invariants (b_l and Psi_k agree); table written to supplement_tables.tex")


if __name__ == "__main__":
    main()

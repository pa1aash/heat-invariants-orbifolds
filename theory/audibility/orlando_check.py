"""Orlando's formula and the Hurwitz determinants, degrees 2..7, exactly.

Source (fetched): Holtz and Tyaglov, "Structured
matrices, continued fractions, and root localization of polynomials", SIAM
Review 54 (2012) 421-509, doi:10.1137/090781127, arXiv:0912.4703.
  * eq. (1.37) / Definition 1.16: Hurwitz determinants Delta_j(p) of
    p(z) = a_0 z^n + ... + a_n, the leading principal minors of the matrix with
    (i, j) entry a_(2j - i).
  * Theorem 1.17 (Orlando 1911, Math. Ann. 71, doi:10.1007/BF01456650):
        Delta_{n-1}(p) = (-1)^(n(n-1)/2) a_0^(n-1) prod_{i<j} (z_i + z_j),
    z_1, ..., z_n the zeros of p.

Checks, all exact polynomial identities in sparse integer polynomial rings:
  1. Orlando's formula in full generality (symbolic a_0 and symbolic zeros).
  2. The specialisation used in the paper: for p(z) = prod (z + m_i),
     Delta_{n-1} = + prod_{i<j} (m_i + m_j), every coefficient positive.
  3. Delta_n = a_n Delta_{n-1} (Barkovsky, arXiv:0802.1805, eq. (35)).
  4. Res_w(even part, odd part) = +/- a_0^k Delta_{n-1}: the even and odd parts
     of p are coprime exactly when no two zeros of p sum to zero.

Exits nonzero on any failure.
"""
import sys
import time

import sympy as sp
from sympy import ZZ
from sympy.polys.rings import ring

from audibility_common import evaluate_poly, generic_hurwitz_det, ring_elementary

DEGREES = range(2, 8)


def check_orlando_general(n):
    names = ["a0"] + [f"z{i}" for i in range(1, n + 1)]
    Rg, *gens = ring(",".join(names), ZZ)
    a0, zs = gens[0], gens[1:]
    e = ring_elementary(zs, Rg(1))
    a = [a0 * (-1) ** k * e[k] for k in range(n + 1)]  # p = a0 prod (z - z_i)
    lhs = evaluate_poly(generic_hurwitz_det(n, n - 1), a, Rg(0))
    rhs = Rg((-1) ** (n * (n - 1) // 2)) * a0 ** (n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            rhs *= zs[i] + zs[j]
    assert lhs == rhs, f"Orlando's formula fails at n={n}"


def check_decay_rate_form(n):
    Rg, *ms = ring(",".join(f"m{i}" for i in range(1, n + 1)), ZZ)
    e = ring_elementary(ms, Rg(1))  # p(z) = prod (z + m_i): a_k = e_k
    D = evaluate_poly(generic_hurwitz_det(n, n - 1), e, Rg(0))
    prod = Rg(1)
    for i in range(n):
        for j in range(i + 1, n):
            prod *= ms[i] + ms[j]
    assert D == prod, f"Delta_{n-1} != prod(m_i+m_j) at n={n}"
    assert all(c > 0 for c in prod.values()), "negative coefficient in prod(m_i+m_j)"
    Dn = evaluate_poly(generic_hurwitz_det(n, n), e, Rg(0))
    assert Dn == e[n] * D, f"Delta_n != a_n Delta_(n-1) at n={n}"
    return len(prod)


def check_resultant(n):
    a = sp.symbols(f"a0:{n + 1}")
    w = sp.Symbol("w")
    even = sum(a[k] * w ** ((n - k) // 2) for k in range(n + 1) if (n - k) % 2 == 0)
    odd = sum(a[k] * w ** ((n - k - 1) // 2) for k in range(n + 1) if (n - k) % 2 == 1)
    res = sp.resultant(sp.Poly(even, w), sp.Poly(odd, w)).as_expr()
    D = generic_hurwitz_det(n, n - 1).as_expr()
    ratio = sp.factor(sp.cancel(res / D))
    sign, rest = ratio.as_coeff_Mul()
    assert sign in (1, -1), f"Res/Delta has constant {sign} at n={n}"
    assert rest == 1 or rest == a[0] or (rest.is_Pow and rest.base == a[0]), (
        f"Res(even, odd)/Delta_(n-1) = {ratio} is not +/- a power of a0 (n={n})"
    )
    return ratio


def main():
    for n in DEGREES:
        t = time.time()
        check_orlando_general(n)
        nterms = check_decay_rate_form(n)
        ratio = check_resultant(n)
        print(
            f"n={n}: Orlando (general a0, zeros) OK; Delta_{n-1}(prod(z+m_i)) = "
            f"prod(m_i+m_j), {nterms} terms, all positive; Delta_n = a_n Delta_{n-1}; "
            f"Res(even,odd)/Delta_{n-1} = {ratio}   [{time.time() - t:.1f}s]",
            flush=True,
        )
    print("orlando_check: all assertions passed for degrees 2..7")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

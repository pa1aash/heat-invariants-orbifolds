"""Independent exact-arithmetic check of the cone-point heat coefficients.

Implements, from the transcription in Ucar's thesis:
  Ucar (arXiv:1711.03405) eq. (4.25):
    c_l(pi/k) = 1/(4k) * (-1)^l/((l+1)!) * 1/(2l+1)
                * sum_{j=0}^{l+1} binom(2l+2, 2j) (k^{2j}-1) B_{2j} B_{2l+2-2j}(1/2)
  Ucar eq. (4.33)/(4.34):
    b_l / kappa^l = sum_{i=0}^{l} 2/(4^i i!) * c_{l-i}(pi/k)       (k = m)
B_n are Bernoulli numbers (B_1=-1/2 convention, only even indices occur) and
B_n(x) Bernoulli polynomials, as in Ucar (3.69).
"""
from fractions import Fraction
from math import comb, factorial
import sympy as sp

m = sp.symbols('m')

# --- exact Bernoulli numbers (own recursion, B_1 = -1/2) ---
_B = {0: Fraction(1)}
def bern(n):
    if n in _B:
        return _B[n]
    s = sum(comb(n + 1, j) * bern(j) for j in range(n))
    _B[n] = -s / (n + 1)
    return _B[n]

def bern_half(n):
    # Ucar Lemma 4.14: -B_n(1/2) = B_n (1 - 1/2^{n-1})  => B_n(1/2) = (2^{1-n}-1) B_n
    return (Fraction(1, 2 ** (n - 1)) - 1) * bern(n) if n >= 1 else Fraction(1)

# cross-check against sympy's Bernoulli polynomial at 1/2 and own numbers vs sympy numbers
for n in range(0, 30, 2):
    assert sp.Rational(bern(n).numerator, bern(n).denominator) == sp.bernoulli(n), n
    assert sp.Rational(bern_half(n).numerator, bern_half(n).denominator) == sp.bernoulli(n, sp.Rational(1, 2)), n

def R(fr):
    return sp.Rational(fr.numerator, fr.denominator)

def c_ucar(l, k):
    """(4.25); k may be a sympy symbol or a number."""
    s = 0
    for j in range(l + 2):
        s += comb(2 * l + 2, 2 * j) * (k ** (2 * j) - 1) * R(bern(2 * j)) * R(bern_half(2 * l + 2 - 2 * j))
    return sp.Rational((-1) ** l, 4 * factorial(l + 1) * (2 * l + 1)) * s / k

def b_ratio(l, k):
    """(4.33)/(4.34): b_l/kappa^l for a cone point of order k."""
    return sum(sp.Rational(2, 4 ** i * factorial(i)) * c_ucar(l - i, k) for i in range(l + 1))

rows = {l: sp.simplify(b_ratio(l, m)) for l in range(0, 7)}
polys = {l: sp.expand(sp.together(rows[l] * m)) for l in rows}   # p_l(m) = m * b_l/kappa^l
for l in range(0, 5):
    print(f"l={l}: b_l/kappa^l = {sp.factor(rows[l])}")
    print(f"      p_{l}(m) = {polys[l]}")

# (a) l = 0 : manuscript cone(m)
assert sp.simplify(rows[0] - (m**2 - 1) / (12 * m)) == 0
# (b) l = 1 : the paper's eq. (4) and Schueth Remark 4.2
eq4 = sp.Rational(1, 360) * (m**3 - 1/m) + sp.Rational(1, 36) * (m - 1/m)
assert sp.simplify(rows[1] - eq4) == 0
# Schueth Remark 4.2 a_0 = (1/12)(k - 1/k)
assert sp.simplify(rows[0] - sp.Rational(1, 12) * (m - 1/m)) == 0
# (c) l = 2 : Schueth Theorem 4.1, K^2 coefficient (as transcribed)
thm41 = (sp.Rational(1, 2520) * (m**5 - 1/m) + sp.Rational(1, 720) * (m**3 - 1/m)
         + sp.Rational(1, 180) * (m - 1/m))
assert sp.simplify(rows[2] - thm41) == 0
assert sp.expand(sp.together(rows[2] - thm41)) == 0
# (d) l = 3 : stated polynomial and factored form
p3 = (3*m**8 + 8*m**6 + 14*m**4 + 32*m**2 - 57) / 30240
fact3 = (m**2 - 1) * (m**2 + 3) * (3*m**4 + 2*m**2 + 19) / (30240 * m)
assert sp.expand(polys[3] - p3) == 0, "stated p_3 differs from computed"
assert sp.simplify(rows[3] - fact3) == 0, "factored l=3 form differs from computed"
# consistency of the two stated forms of the l=3 row (reported, asserted below)
expanded_factored = sp.expand((m**2 - 1) * (m**2 + 3) * (3*m**4 + 2*m**2 + 19))
print("expand((m^2-1)(m^2+3)(3m^4+2m^2+19)) =", expanded_factored)
print("30240*p_3(m)                         =", sp.expand(30240 * p3))
assert sp.expand(expanded_factored - 30240 * p3) == 0, "two stated forms of l=3 row are inconsistent"
# (e) leading coefficient of p_l
for l in range(0, 7):
    lead = sp.Poly(polys[l], m).LC()
    assert sp.Poly(polys[l], m).degree() == 2 * l + 2
    want = abs(R(bern(2 * l + 2))) / (2 * factorial(l + 1) * (2 * l + 1))
    assert lead == want, (l, lead, want)
    print(f"l={l}: leading coeff of p_l = {lead} = |B_{2*l+2}|/(2 (l+1)! (2l+1))")
# numeric Fraction check, l = 3 and 4 (independent Fraction implementation)
def c_frac(l, k):
    k = Fraction(k)
    s = sum(comb(2*l+2, 2*j) * (k**(2*j) - 1) * bern(2*j) * bern_half(2*l+2-2*j) for j in range(l+2))
    return Fraction((-1)**l, 4 * factorial(l+1) * (2*l+1)) * s / k
def b_frac(l, k):
    return sum(Fraction(2, 4**i * factorial(i)) * c_frac(l-i, k) for i in range(l+1))
for l in (3, 4):
    for k in range(2, 13):
        sym = rows[l].subs(m, k)
        fr = b_frac(l, k)
        assert sp.Rational(fr.numerator, fr.denominator) == sym, (l, k)
    # also k=1 vanishes (Ucar: coefficients vanish for k=1)
    assert b_frac(l, 1) == 0
for k in range(2, 13):
    f3 = Fraction((k**2-1)*(k**2+3)*(3*k**4+2*k**2+19), 30240*k)
    assert b_frac(3, k) == f3
print("ALL ASSERTS PASSED")

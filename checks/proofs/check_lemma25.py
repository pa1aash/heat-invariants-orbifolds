"""Lemmas 2.7-2.10 and Remark S8.1 for every order, from the elliptic term of the trace formula.

Checks every computational claim of these statements.

  E_m(t) = sum_{j=1}^{m-1} 1/(2m sin th_j) int_R e^{-2 th_j r}/(1+e^{-2 pi r}) e^{-t(1/4+r^2)} dr,
  th_j = pi j/m  (Dryden-Strohmaier eq. (1), elliptic term, at h = h_t).

Proof chain checked here:
  (a) M_a(s) = int_R e^{-(a-s) r}/(1+e^{-2 pi r}) dr = 1/(2 sin((a-s)/2))           [quadrature]
  (b) Phi_m(u) := sum_j 1/(4m sin th_j sin(th_j - u)) = (cot u - m cot(mu))/(4m sin u)   [50 digits]
      and sum_{j=0}^{m-1} cot(x + pi j/m) = m cot(mx)                                   [50 digits]
  (c) phi_k(m) := [u^{2k}] Phi_m(u)
          = (1/4m) sum_{n=1}^{k+1} sig_{k+1-n} (-4)^n B_{2n} (1 - m^{2n})/(2n)!
      with sig_i = [u^{2i}] u/sin u                                                    [exact vs mpmath Taylor]
  (d) [t^l] E_m = (-1)^l p_l(m)/m,  p_l(m) = 4^{-l} sum_k (2k)! m phi_k(m) / (k!(l-k)!)
      against the moments of the actual integral                                       [quadrature, 30 digits]
  (e) p_l even, deg 2l+2, p_l(1)=0, leading coefficient |B_{2l+2}|/(2(l+1)!(2l+1)),
      p_l(m) > 0 for m >= 2                                                            [exact, l <= LMAX]
  (f) cross-check only: p_l equals Ucar (4.25)+(4.33) exactly (l <= LMAX; the task asks l <= 8),
      the factorised forms l <= 4 of Ucar's thesis and the paper's
      p_0, p_1, p_2 (eq. plexplicit)                                                   [exact]
  (g) the triangular expansion p_l(x)/x = sum_{k=1}^{l+1} a_{l,k} psi_k(x),
      psi_k = x^{2k-1} - 1/x, with a_{l,l+1} != 0 (Lemma 2.10, used by Theorems A-C)  [exact]
  (h) truncation: |E_m(t) - sum_{l<L} b_l t^l| = O(t^L) at small t                     [quadrature]

Everything marked exact uses fractions.Fraction; Bernoulli numbers by their own recursion.
Run from the repository root:
    python3 checks/proofs/check_lemma25.py
Writes checks/proofs/check_lemma25.txt.  Exits nonzero on any failure.
"""
import os
import sys
from fractions import Fraction as Fr
from math import comb, factorial

import mpmath as mp


def mpq(q):
    return mp.mpf(q.numerator) / q.denominator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_lemma25.txt")
LMAX = 40
log = []
fails = 0


def check(cond, msg):
    global fails
    log.append(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


# ------------------------------------------------------------------ Bernoulli numbers
_B = {0: Fr(1)}


def B(n):
    """B_n with B_1 = -1/2, from sum_{j<=n} binom(n+1, j) B_j = 0."""
    if n not in _B:
        _B[n] = -sum(comb(n + 1, j) * B(j) for j in range(n)) / (n + 1)
    return _B[n]


def B_half(n):
    """B_n(1/2) = (2^{1-n} - 1) B_n."""
    return Fr(1) if n == 0 else (Fr(2) ** (1 - n) - 1) * B(n)


# ------------------------------------------------------------------ polynomials in m
# A polynomial in m is a dict {degree: Fraction}.
def padd(p, q, w=Fr(1)):
    out = dict(p)
    for d, c in q.items():
        out[d] = out.get(d, Fr(0)) + w * c
    return {d: c for d, c in out.items() if c != 0}


def peval(p, x):
    return sum(c * Fr(x) ** d for d, c in p.items())


def sig(i):
    """[u^{2i}] u/sin u = (-1)^{i+1} (2^{2i} - 2) B_{2i}/(2i)!."""
    return Fr((-1) ** (i + 1) * (2 ** (2 * i) - 2)) * B(2 * i) / factorial(2 * i)


def m_phi(k):
    """m * phi_k(m) as a polynomial in m, phi_k = [u^{2k}] Phi_m(u)."""
    out = {}
    for n in range(1, k + 2):
        w = Fr(1, 4) * sig(k + 1 - n) * Fr((-4) ** n) * B(2 * n) / factorial(2 * n)
        out = padd(out, {0: w, 2 * n: -w})          # w (1 - m^{2n})
    return out


def p_trace(l):
    """p_l(m) from the elliptic term: 4^{-l} sum_k (2k)! m phi_k / (k! (l-k)!)."""
    out = {}
    for k in range(l + 1):
        out = padd(out, m_phi(k), Fr(factorial(2 * k), 4 ** l * factorial(k) * factorial(l - k)))
    return out


def p_ucar(l):
    """Cross-check only: Ucar (4.25) + (4.33), as transcribed in the paper (eq:ucarlune, eq:pl)."""
    def m_cS(lp):        # m * c^S_{lp}(pi/m)
        pre = Fr((-1) ** lp, 4 * factorial(lp + 1) * (2 * lp + 1))
        out = {}
        for j in range(1, lp + 2):          # j = 0 has m^0 - 1 = 0
            w = pre * comb(2 * lp + 2, 2 * j) * B(2 * j) * B_half(2 * lp + 2 - 2 * j)
            out = padd(out, {2 * j: w, 0: -w})
        return out
    out = {}
    for i in range(l + 1):
        out = padd(out, m_cS(l - i), Fr(2, 4 ** i * factorial(i)))
    return out


# ------------------------------------------------------------------ (a) moment generating function
mp.mp.dps = 30


def F_int(a, s=0, power=0):
    f = lambda r: r ** power * mp.e ** (-(a - s) * r) / (1 + mp.e ** (-2 * mp.pi * r))
    return mp.quad(f, [-mp.inf, -10, 0, 10, mp.inf])


for a in [mp.mpf(1) / 3, 1, mp.pi / 2, 2, 5, 6]:
    for s in [0, mp.mpf("0.2"), mp.mpf("-0.15")]:
        lhs = F_int(a, s)
        rhs = 1 / (2 * mp.sin((a - s) / 2))
        check(abs(lhs - rhs) < mp.mpf(10) ** -25, f"(a) M_a(s) = 1/(2 sin((a-s)/2)) at a={mp.nstr(a, 6)}, s={mp.nstr(s, 3)}")

# ------------------------------------------------------------------ (b) closed form of Phi_m
mp.mp.dps = 50
for m in [2, 3, 4, 5, 7, 12, 30]:
    for u in [mp.mpf("0.1"), mp.mpf("0.37"), mp.mpf("-0.21"), mp.mpc("0.05", "0.3")]:
        th = [mp.pi * j / m for j in range(1, m)]
        lhs = sum(1 / (4 * m * mp.sin(t) * mp.sin(t - u)) for t in th)
        rhs = (mp.cot(u) - m * mp.cot(m * u)) / (4 * m * mp.sin(u))
        check(abs(lhs - rhs) < mp.mpf(10) ** -40, f"(b) Phi_{m}({mp.nstr(u, 3)}) closed form")
        x = u + mp.mpf("0.013")
        cs = sum(mp.cot(x + mp.pi * j / m) for j in range(m))
        check(abs(cs - m * mp.cot(m * x)) < mp.mpf(10) ** -38, f"(b) sum_j cot(x + pi j/{m}) = {m} cot({m}x)")
    # Phi_m is even: j -> m - j
    th = [mp.pi * j / m for j in range(1, m)]
    u = mp.mpf("0.29")
    ev = sum(1 / (mp.sin(t) * mp.sin(t - u)) for t in th) - sum(1 / (mp.sin(t) * mp.sin(t + u)) for t in th)
    check(abs(ev) < mp.mpf(10) ** -40, f"(b) Phi_{m} is even")

# ------------------------------------------------------------------ (c) Taylor coefficients of Phi_m
for m in [2, 3, 5, 8, 12]:
    f = lambda u: (mp.cot(u) - m * mp.cot(m * u)) / (4 * m * mp.sin(u)) if u != 0 else mp.mpf(0)
    # analytic at 0 (removable); Taylor via Cauchy integral on |u| = r < pi/m
    r = mp.pi / (2 * m)
    N = 256
    pts = [r * mp.expjpi(mp.mpf(2 * i) / N) for i in range(N)]
    vals = [f(z) for z in pts]
    for k in range(0, 7):
        num = sum(v * z ** (-2 * k) for v, z in zip(vals, pts)) / N
        exact = peval(m_phi(k), m) / m
        check(abs(num - mp.mpf(exact.numerator) / exact.denominator) < mp.mpf(10) ** -30 * (1 + abs(num)),
              f"(c) [u^{2 * k}] Phi_{m} = {mp.nstr(num, 12)} matches the Bernoulli formula")
        # and against the finite csc-derivative sum: (1/4m) sum_j csc(th) csc^{(2k)}(th)
        th = [mp.pi * j / m for j in range(1, m)]
        fin = sum(mp.csc(t) * mp.diff(mp.csc, t, 2 * k) for t in th) / (4 * m) / factorial(2 * k)
        check(abs(fin - num) < mp.mpf(10) ** -25 * (1 + abs(num)), f"(c) finite cosecant-derivative sum, m={m}, k={k}")

# ------------------------------------------------------------------ (d) moments of the actual integral
mp.mp.dps = 30
for m in [2, 3, 5, 7, 12]:
    th = [mp.pi * j / m for j in range(1, m)]
    for k in range(0, 6):
        mom = sum(F_int(2 * t, 0, 2 * k) / (2 * m * mp.sin(t)) for t in th)
        exact = peval(m_phi(k), m) / m * factorial(2 * k) / Fr(4) ** k     # 4^{-k} Phi^{(2k)}(0)
        ex = mp.mpf(exact.numerator) / exact.denominator
        check(abs(mom - ex) < mp.mpf(10) ** -20 * abs(ex), f"(d) sum_j mu_{2 * k}(2 th_j)/(2m sin th_j) = 4^-k Phi^({2 * k})(0), m={m}")
    # t^l coefficient of E_m from the moments, against (-1)^l p_l(m)/m
    for l in range(0, 6):
        coef = sum(Fr(-1, 4) ** (l - k) / factorial(l - k) * Fr((-1) ** k, factorial(k))
                   * peval(m_phi(k), m) / m * factorial(2 * k) / Fr(4) ** k for k in range(l + 1))
        check(coef == Fr((-1) ** l) * peval(p_trace(l), m) / m, f"(d) [t^{l}] E_{m} = (-1)^l p_l(m)/m (exact)")

# ------------------------------------------------------------------ (e) the four properties, every l <= LMAX
for l in range(LMAX + 1):
    p = p_trace(l)
    lead = Fr(abs(B(2 * l + 2).numerator), B(2 * l + 2).denominator) / (2 * factorial(l + 1) * (2 * l + 1))
    check(all(d % 2 == 0 for d in p), f"(e) l={l}: p_l even")
    check(max(p) == 2 * l + 2, f"(e) l={l}: degree {max(p)} = 2l+2")
    check(peval(p, 1) == 0, f"(e) l={l}: p_l(1) = 0")
    check(p[2 * l + 2] == lead, f"(e) l={l}: leading coefficient = |B_{2 * l + 2}|/(2(l+1)!(2l+1)) = {lead}")
    check(all(peval(p, mm) > 0 for mm in range(2, 41)), f"(e) l={l}: p_l(m) > 0 for 2 <= m <= 40")

# positivity of every Taylor coefficient of Phi_m for m >= 2 (each term w(1-m^{2n}) has w < 0)
for k in range(LMAX + 1):
    ok = all(Fr(1, 4) * sig(k + 1 - n) * Fr((-4) ** n) * B(2 * n) / factorial(2 * n) < 0 for n in range(1, k + 2))
    check(ok and all(sig(i) > 0 for i in range(k + 1)), f"(e) k={k}: every term of m phi_k is a positive multiple of m^(2n)-1")

# ------------------------------------------------------------------ (f) cross-checks
for l in range(LMAX + 1):
    check(p_trace(l) == p_ucar(l), f"(f) l={l}: p_l (trace formula) = Ucar (4.25)+(4.33) exactly")
paper = {0: {2: Fr(1, 12), 0: Fr(-1, 12)},
         1: {4: Fr(1, 360), 2: Fr(1, 36), 0: Fr(-11, 360)},
         2: {6: Fr(1, 2520), 4: Fr(1, 720), 2: Fr(1, 180), 0: Fr(-37, 5040)}}
for l, p in paper.items():
    check(p_trace(l) == p, f"(f) l={l}: equals the paper's eq. (plexplicit)")
tables = {3: lambda m: (m * m - 1) * (m * m + 3) * (3 * m ** 4 + 2 * m * m + 19) / Fr(30240),
          4: lambda m: (m * m - 1) * (70 * m ** 8 + 235 * m ** 6 + 477 * m ** 4 + 785 * m * m + 1313) / Fr(1995840)}
for l, f in tables.items():
    check(all(peval(p_trace(l), mm) == f(Fr(mm)) for mm in range(-6, 13)),
          f"(f) l={l}: equals the factorised form in Ucar's thesis")
# Schueth Rem. 4.2 cone terms a_0, a_1 at K = -1 (via the paper, eq. b1)
for mm in range(2, 20):
    check(peval(p_trace(0), mm) / mm == Fr(1, 12) * (mm - Fr(1, mm)), f"(f) m={mm}: b_0 = (m - 1/m)/12")
    check(peval(p_trace(1), mm) / mm == Fr(1, 360) * (mm ** 3 - Fr(1, mm)) + Fr(1, 36) * (mm - Fr(1, mm)),
          f"(f) m={mm}: -b_1 = (m^3-1/m)/360 + (m-1/m)/36")

# ------------------------------------------------------------------ (g) triangular expansion
for l in range(LMAX + 1):
    p = p_trace(l)
    a = {k: p.get(2 * k, Fr(0)) for k in range(l + 2)}
    check(sum(a.values()) == 0, f"(g) l={l}: sum_k a_lk = p_l(1) = 0")
    # p_l(x)/x - sum_{k>=1} a_lk psi_k(x) vanishes at many rationals
    ok = all(peval(p, x) / x == sum(a[k] * (Fr(x) ** (2 * k - 1) - 1 / Fr(x)) for k in range(1, l + 2))
             for x in [Fr(1, 3), Fr(2), Fr(7, 5), Fr(-3)])
    check(ok and a[l + 1] != 0, f"(g) l={l}: p_l(x)/x = sum a_lk psi_k(x), a_(l,l+1) = {a[l + 1]} != 0")

# ------------------------------------------------------------------ (h) truncation error at small t
mp.mp.dps = 30
for m in [3, 8]:
    th = [mp.pi * j / m for j in range(1, m)]

    def E(t):
        return sum(mp.quad(lambda r: mp.e ** (-2 * a * r) / (1 + mp.e ** (-2 * mp.pi * r)) * mp.e ** (-t * (mp.mpf(1) / 4 + r * r)),
                           [-mp.inf, -20, 0, 20, mp.inf]) / (2 * m * mp.sin(a)) for a in th)
    L = 3
    ratios = []
    for t in [mp.mpf("0.004"), mp.mpf("0.002"), mp.mpf("0.001")]:
        ser = sum(mpq(Fr((-1) ** l) * peval(p_trace(l), m) / m) * t ** l for l in range(L))
        bL = mpq(Fr((-1) ** L) * peval(p_trace(L), m) / m)
        ratios.append((E(t) - ser) / (bL * t ** L))
    check(all(abs(x - 1) < mp.mpf("0.15") for x in ratios),
          f"(h) m={m}: (E_m(t) - sum_(l<3) b_l t^l)/(b_3 t^3) -> 1: " + ", ".join(mp.nstr(x, 6) for x in ratios))

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{len(log)} checks, {fails} failures\n")
print(f"{len(log)} checks, {fails} failures")
sys.exit(1 if fails else 0)

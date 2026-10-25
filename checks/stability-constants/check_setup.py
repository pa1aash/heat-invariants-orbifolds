"""check_setup: exact checks of the reconstruction of Theorem B and of the Prop. 6.10 ingredients.

1. My M(I), b(I) satisfy Lemma S2.2 (B = S M) and Lemma S2.1 (det) exactly; solve returns e.
2. tanh U computed by series composition equals z O(z^2)/E(z^2) (truncated).
3. H_nu from Ucar (4.25), (4.33), (4.35) (fetched text) equals F I + h0 with F = inverse of the ST.2 table.
4. ST.2 row sums l_r and diagonal; sigma_k(n), zeta_n examples of Theorem S2.
5. J of Prop. 6.10 equals the sympy Jacobian of M(I)e - b(I) (n = 2..6 symbolic), and G = -M^{-1} J F^{-1}
   equals the exact derivative of the recovery map (difference quotients).
6. SB.3 claim on the series for n <= 5, and the hypothesis U >= 0.
"""
import sys, random
from fractions import Fraction as Fr
import sympy as sp
from stab610 import *

ROWS = [(2, 8, 8), (3, 3, 12), (3, 10, 15, 30), (4, 5, 21, 28), (2, 3, 7), (4, 4, 4), (7, 7, 7), (3, 3, 4, 4),
        (5, 5, 5, 5), (2, 2, 2, 3), (2, 2, 2, 2, 3)]
fails = 0


def check(cond, msg):
    global fails
    if not cond:
        fails += 1
        print("FAIL:", msg)
    return cond

# ---------------- 1. Lemma S2.2 and S2.1
random.seed(1)
tests = [list(r) for r in ROWS] + [[Fr(random.randint(1, 40), random.randint(1, 9)) for _ in range(n)]
                                   for n in (2, 3, 4, 5, 6) for _ in range(6)]
for m in tests:
    n = len(m)
    e = esym([Fr(x) for x in m])
    I = invariants(m)
    M, b = M_b(I)
    E_ = lambda k: e[k] if 0 <= k <= n else Fr(0)
    B = [[(-1) ** (j + 1) * E_(2 * k + 1 - j) for j in range(1, n + 1)] for k in range(n)]
    S = [[Fr(0)] * n for _ in range(n)]
    for k in range(n - 1):
        for i in range(k + 1):
            S[k][i] = E_(2 * (k - i))
    S[n - 1][n - 1] = (-1) ** n * E_(n)
    check(mat_mul(S, M) == B, f"B = S M fails for {m}")
    prod = Fr(1)
    for i in range(n):
        for j in range(i + 1, n):
            prod *= Fr(m[i]) + Fr(m[j])
    check(det(M) == (-1) ** (n * (n + 1) // 2) * prod / e[n], f"det M (Lemma S2.1) fails for {m}")
    check(det(B) == (-1) ** (n * (n - 1) // 2) * prod, f"det B fails for {m}")
    check(solve_e(I) == e, f"Theorem B solve does not return e for {m}")
    # 2. tanh U = zO/E
    N = 2 * n - 2
    Sr = Ser(N)
    T = T_series(I)
    Eser = [E_(k) if k % 2 == 0 else Fr(0) for k in range(N)]
    Oser = [E_(k) if k % 2 == 1 else Fr(0) for k in range(N)]
    check(Sr.mul(T, Eser) == Oser, f"tanh U != zO/E for {m}")
print("1-2. reconstruction of Theorem B: B = S M, det M, det B, exact solve, tanh U = zO/E on", len(tests), "multisets")

# ---------------- 3. H from Ucar (fetched text, sources/ucar_1711.03405.txt lines 11580-12010)
def bern(k):
    return Fr(str(sp.bernoulli(k)))


def bernpoly_half(k):
    return Fr(str(sp.bernoulli(k, sp.Rational(1, 2))))


def cS(l, k):
    tot = sum(Fr(comb(2 * l + 2, 2 * j)) * (Fr(k) ** (2 * j) - 1) * bern(2 * j) * bernpoly_half(2 * l + 2 - 2 * j)
              for j in range(l + 2))
    return Fr(1, 4) / Fr(k) * Fr((-1) ** l) / sp.factorial(l + 1) / (2 * l + 1) * tot


def C_cone(nu, k, kappa=-1):
    return sum(Fr(2, 4 ** l) / sp.factorial(l) * cS(nu - l, k) for l in range(nu + 1)) * Fr(kappa) ** nu


def alpha(nu, kappa=-1):
    return Fr(1, int(sp.factorial(nu)) * 4 ** nu) * sum(comb(nu, l) * Fr(-4) ** l * bernpoly_half(2 * l)
                                                       for l in range(nu + 1)) * Fr(kappa) ** nu


check([alpha(j) for j in range(5)] == ALPHA, "alpha_j from (4.35) differs from ST.0 list")
print("3. alpha_j (Ucar 4.35, kappa=-1):", [str(alpha(j)) for j in range(6)])


def H_ucar(m):
    n = len(m)
    R = sum(Fr(1, x) for x in m)
    A4pi = Fr(n - 2) / 2 - R / 2
    return [A4pi] + [A4pi * alpha(nu + 1) + sum(Fr(C_cone(nu, x)) for x in m) for nu in range(n - 1)]


for m in tests:
    if all(Fr(x).denominator == 1 for x in m):
        check(H_ucar([int(x) for x in m]) == H_of_m(m), f"H(Ucar) != F I + h0 for {m}")
# also random integer multisets, n = 2..5
for n in (2, 3, 4, 5):
    for _ in range(20):
        m = [random.randint(2, 30) for _ in range(n)]
        check(H_ucar(m) == H_of_m(m), f"H(Ucar) != F I + h0 for {m}")
print("3. H(Ucar (4.25),(4.33),(4.35)) == F I + h0 (F = inverse of ST.2 table): checked")
for m in ROWS:
    print("   H", m, "=", [float(h) for h in H_of_m(m)])

# ---------------- 4. ST.2 row sums and diagonals; sigma, zeta
ell = [sum(abs(x) for x in row) for row in LINV5]
check(ell == ELL[:5], f"row sums {ell}")
diag_expected = [Fr(2)] + [Fr(2) * sp.factorial(nu + 1) * (2 * nu + 1) / abs(bern(2 * nu + 2)) for nu in range(4)]
check([abs(LINV5[r][r]) for r in range(5)] == [Fr(str(x)) for x in diag_expected], "diagonal of L^{-1}")
print("4. l_r =", [str(x) for x in ell], " diagonal =", [str(abs(LINV5[r][r])) for r in range(5)])
check(sigma(3, 3)[1:4:2] == [1, Fr(49, 3)], "sigma n=3")
check(sigma(4, 5)[1:6:2] == [1, Fr(76, 3), Fr(6628, 15)], "sigma n=4")
check([zeta(3), zeta(4), zeta(5)] == [1, Fr(79, 3), Fr(14048, 15)], "zeta_n")
print("4. sigma(3) =", [str(x) for x in sigma(3, 3)[1::2]], "sigma(4) =", [str(x) for x in sigma(4, 5)[1::2]],
      "zeta_3,4,5 =", zeta(3), zeta(4), zeta(5))

# ---------------- 5. J vs sympy, G vs difference quotients
z = sp.Symbol('z')
for n in (2, 3, 4, 5, 6):
    N = 2 * n - 2
    es = sp.symbols(f'e1:{n + 1}')
    Rs = sp.Symbol('R')
    Ps = sp.symbols(' '.join(f'P{2 * k - 1}' for k in range(1, n)))
    Ps = list(Ps) if isinstance(Ps, tuple) else [Ps]
    Isym = [Rs] + Ps
    U = sum(Ps[k - 1] * z ** (2 * k - 1) / (2 * k - 1) for k in range(1, n))
    T = sp.series(sp.tanh(U), z, 0, N).removeO() if n > 1 else 0
    T = sp.expand(T)
    Tc = [T.coeff(z, k) for k in range(N)]
    Ec = lambda k: 1 if k == 0 else (es[k - 1] if 1 <= k <= n else 0)
    Fv = []
    for j in range(n - 1):
        row = (Ec(2 * j + 1) if 2 * j + 1 <= n else 0) - sum(Tc[2 * i + 1] * Ec(2 * j - 2 * i) for i in range(j)
                                                               if 2 * j - 2 * i <= n) - Tc[2 * j + 1]
        Fv.append(row)
    Fv.append(es[n - 2] - Rs * es[n - 1])
    Jsym = sp.Matrix(Fv).jacobian(Isym)
    s = sp.expand(sp.series(1 - sp.tanh(U) ** 2, z, 0, N).removeO())
    sc = [s.coeff(z, k) for k in range(N)]
    Jf = sp.zeros(n, n)
    for j in range(n - 1):
        for k in range(1, n):
            Jf[j, k] = -sp.Rational(1, 2 * k - 1) * sum(Ec(2 * j - 2 * i) * sc[2 * i + 2 - 2 * k]
                                                        for i in range(k - 1, j + 1) if 2 * j - 2 * i <= n)
    Jf[n - 1, 0] = -es[n - 1]
    diff = (Jsym - Jf).applyfunc(sp.expand)
    check(diff == sp.zeros(n, n), f"J formula differs from sympy Jacobian, n = {n}: {diff}")
    print(f"5. n = {n}: Prop. 6.10 J == sympy d(Me-b)/dI (symbolic in e, R, P)")

h = Fr(1, 10 ** 30)
for m in ROWS:
    st = Setup(m)
    n = st.n
    for c in range(n):
        dH = [Fr(0)] * n
        dH[c] = h
        I2 = [x + y for x, y in zip(st.I, mat_vec(st.Finv, dH))]
        e2 = solve_e(I2)
        dq = [(x - y) / h for x, y in zip(e2[1:], st.e[1:])]
        err = max(abs(dq[j] - st.G[j][c]) for j in range(n))
        scale = max(abs(st.G[j][c]) for j in range(n)) + 1
        check(err < Fr(1, 10 ** 20) * scale, f"G column {c} mismatch for {m}: {float(err)}")
print("5. G = -M^{-1} J F^{-1} equals the exact difference quotient (h = 1e-30) for all 11 rows")

# ---------------- 6. SB.3 claim and U >= 0
for n in (2, 3, 4, 5):
    for m in [r for r in ROWS if len(r) == n] + [[random.randint(2, 30) for _ in range(n)] for _ in range(3)]:
        st = Setup(m)
        check(all(x >= 0 for x in st.U), "U >= 0")
        check(st.N == 2 * n - 2 and st.N <= 8, "truncation after z^{2n-3} <= z^7")
        B = st.bounds([Fr(1, 10 ** 6)] * n)
        named = dict(U=st.U, T=st.T, tanU=st.tanU, dU=B['dU'], tau_lin=B['tau_lin'], tau_rem=B['tau_rem'],
                     tau=B['tau'], sech2U=st.s, sec2U=st.sec2U)
        for name, ser in named.items():
            nz = [k for k, x in enumerate(ser) if x != 0]
            odd = all(k % 2 == 1 for k in nz)
            if name in ('sech2U', 'sec2U'):
                check(all(k % 2 == 0 for k in nz) and 0 in nz, "sech^2U, sec^2U even with s_0 = 1")
            else:
                check(odd and len(nz) <= 4, f"{name} not odd/<=4 nonzero")
print("6. U >= 0 coefficientwise (P_j > 0). For n <= 5: U, T, tan U, |dU|, tau_lin, tau_rem, tau are odd with")
print("   <= 4 nonzero coefficients; but sech^2 U and sec^2 U (used in tau_lin, tau_rem and J) are EVEN series")
print("   with s_0 = 1: the SB.3 parenthesis '(odd)' is false for them.")
for m in [(2, 2, 2, 2, 3)]:
    st = Setup(m)
    print("   (2,2,2,2,3): sech^2U coefficients =", [str(x) for x in st.s])

print("TOTAL FAILURES:", fails)
sys.exit(1 if fails else 0)

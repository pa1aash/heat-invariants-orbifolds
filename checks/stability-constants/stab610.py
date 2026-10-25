"""Independent exact implementation of Proposition S5 with the explicit formulas of Prop. 6.10.

Written from the statements of the paper only, independently of theory/stability/.
Exact rational arithmetic throughout (fractions.Fraction). No floating point is used as a certificate.

Conventions:
  I = (R, P_1, P_3, ..., P_{2n-3}),  R = sum 1/m_i = e_{n-1}/e_n,  P_j = sum m_i^j.
  U(z) = sum_k P_{2k-1} z^{2k-1}/(2k-1) = sum_i artanh(m_i z),  T = tanh U = z O(z^2)/E(z^2).
  Theorem B system M(I) e = b(I), rows j = 0..n-2 (weight 2j+1) and the last row (weight n-1):
     row j   : e_{2j+1} - sum_{i<j, 2j-2i<=n} T_{2i+1} e_{2j-2i} = T_{2j+1}
     row n-1 : e_{n-1} - R e_n = 0
  This is the sign convention fixed by J = d(Me-b)/dI (J_{n-1,0} = -e_n) and it satisfies
  Lemma S2.2 (B = S M) and Lemma S2.1 (det) exactly (checked in check_setup.py).
"""
from fractions import Fraction as Fr
from math import comb
import itertools

# ---------------------------------------------------------------- linear algebra (exact)

def mat_inv(A):
    n = len(A)
    M = [list(map(Fr, row)) + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return None
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def mat_vec(A, v):
    return [sum(a * x for a, x in zip(row, v)) for row in A]


def mabs(A):
    return [[abs(x) for x in row] for row in A]


def det(A):
    n = len(A)
    M = [list(map(Fr, r)) for r in A]
    d = Fr(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return Fr(0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return d

# ---------------------------------------------------------------- symmetric functions

def esym(roots):
    """e_0..e_n of a list of numbers (q(z) = prod (z - r) = sum (-1)^j e_j z^{n-j})."""
    e = [Fr(1)]
    for r in roots:
        e = [ (e[k] if k < len(e) else 0) + (r * e[k - 1] if k >= 1 else 0) for k in range(len(e) + 1)]
    return e


def power_sums_from_e(e, kmax):
    """Newton: p_k for k = 1..kmax from e_0..e_n (e_j = 0 for j > n)."""
    n = len(e) - 1
    E = lambda j: e[j] if 0 <= j <= n else Fr(0)
    p = [None] * (kmax + 1)
    for k in range(1, kmax + 1):
        s = Fr((-1) ** (k - 1) * k) * E(k)
        for i in range(1, k):
            s += (-1) ** (i - 1) * E(i) * p[k - i]
        p[k] = s
    return p


def invariants_from_e(e):
    n = len(e) - 1
    p = power_sums_from_e(e, max(1, 2 * n - 3))
    return [e[n - 1] / e[n]] + [p[2 * k - 1] for k in range(1, n)]


def invariants(m):
    return invariants_from_e(esym([Fr(x) for x in m]))

# ---------------------------------------------------------------- front end (table of L^{-1}, ST.2)

LINV5 = [
    [Fr(-2), 0, 0, 0, 0],
    [Fr(2), Fr(12), 0, 0, 0],
    [Fr(-18), Fr(-120), Fr(-360), 0, 0],
    [Fr(30), Fr(252), Fr(1260), Fr(2520), 0],
    [Fr(-70, 3), Fr(-240), Fr(-1680), Fr(-6720), Fr(-10080)],
]
ALPHA = [Fr(1), Fr(-1, 3), Fr(1, 15), Fr(-4, 315), Fr(1, 315)]


def Finv(n):
    return [[Fr(x) for x in row[:n]] for row in LINV5[:n]]


def F(n):
    return mat_inv(Finv(n))


def h0(n):
    return [Fr(n - 2, 2) * ALPHA[k] for k in range(n)]


def H_from_I(I):
    n = len(I)
    return [a + b for a, b in zip(mat_vec(F(n), I), h0(n))]


def H_of_m(m):
    return H_from_I(invariants(m))

# ---------------------------------------------------------------- truncated power series

class Ser:
    """Power series truncated after z^{N-1}."""
    def __init__(self, N):
        self.N = N

    def mul(self, a, b):
        N = self.N
        c = [Fr(0)] * N
        for i, x in enumerate(a):
            if x:
                for j in range(N - i):
                    if b[j]:
                        c[i + j] += x * b[j]
        return c

    def add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def sub(self, a, b):
        return [x - y for x, y in zip(a, b)]

    def compose(self, coeffs, V):
        """sum_k coeffs[k] V^k, V(0) = 0 (Horner)."""
        assert V[0] == 0
        r = [Fr(0)] * self.N
        for c in reversed(coeffs[: self.N]):
            r = self.mul(r, V)
            r[0] += c
        return r


def tan_coeffs(N):
    """Taylor coefficients t_0..t_{N-1} of tan x, from T' = 1 + T^2 (all >= 0)."""
    t = [Fr(0)] * N
    if N > 1:
        t[1] = Fr(1)
    for k in range(1, N - 1):
        sq = sum(t[i] * t[k - i] for i in range(k + 1))
        t[k + 1] = sq / (k + 1)
    return t


def tanh_coeffs(N):
    t = tan_coeffs(N)
    return [(Fr(-1) ** ((k - 1) // 2) * x if k % 2 else Fr(0)) for k, x in enumerate(t)]


def U_series(I, N):
    n = len(I)
    U = [Fr(0)] * N
    for k in range(1, n):
        if 2 * k - 1 < N:
            U[2 * k - 1] = Fr(I[k]) / (2 * k - 1)
    return U

# ---------------------------------------------------------------- Theorem B system

def T_series(I):
    n = len(I)
    N = 2 * n - 2
    S = Ser(N)
    return S.compose(tanh_coeffs(N), U_series(I, N))


def M_b(I, T=None):
    n = len(I)
    if T is None:
        T = T_series(I)
    M = [[Fr(0)] * n for _ in range(n)]
    b = [Fr(0)] * n
    for j in range(n - 1):
        if 2 * j + 1 <= n:
            M[j][2 * j] += 1
        for i in range(j):
            col = 2 * j - 2 * i
            if col <= n:
                M[j][col - 1] -= T[2 * i + 1]
        b[j] = T[2 * j + 1]
    M[n - 1][n - 2] += 1
    M[n - 1][n - 1] -= Fr(I[0])
    return M, b


def solve_e(I):
    M, b = M_b(I)
    Mi = mat_inv(M)
    if Mi is None:
        return None
    return [Fr(1)] + mat_vec(Mi, b)


def J_matrix(I, e):
    """Prop. 6.10 formula for J = d(Me - b)/dI at fixed e."""
    n = len(I)
    T = T_series(I)
    N = 2 * n - 2
    s = Ser(N).sub([Fr(1)] + [Fr(0)] * (N - 1), Ser(N).mul(T, T))
    E = lambda k: e[k] if 0 <= k <= n else Fr(0)
    J = [[Fr(0)] * n for _ in range(n)]
    for j in range(n - 1):
        for k in range(1, n):
            acc = Fr(0)
            for i in range(k - 1, j + 1):
                if 2 * j - 2 * i <= n:
                    acc += E(2 * j - 2 * i) * s[2 * i + 2 - 2 * k]
            J[j][k] = -acc / (2 * k - 1)
    J[n - 1][0] = -E(n)
    return J

# ---------------------------------------------------------------- Prop. 6.10 bounds + Prop. S5 tests

LADDER = [Fr(k, 40) for k in range(20, 0, -1)] + [Fr(1, 100), Fr(1, 1000)]


class Setup:
    """Everything that depends on m only (exact)."""
    def __init__(self, m):
        self.m = sorted(m)
        n = self.n = len(m)
        self.N = 2 * n - 2
        self.S = Ser(self.N)
        self.e = esym([Fr(x) for x in self.m])
        self.I = invariants(self.m)
        self.H = H_from_I(self.I)
        self.Finv = Finv(n)
        self.absFinv = mabs(self.Finv)
        self.U = U_series(self.I, self.N)
        self.T = T_series(self.I)
        self.s = self.S.sub([Fr(1)] + [Fr(0)] * (self.N - 1), self.S.mul(self.T, self.T))
        self.abs_s = [abs(x) for x in self.s]
        self.tanc = tan_coeffs(self.N)
        self.tanU = self.S.compose(self.tanc, self.U)
        self.sec2U = self.S.add([Fr(1)] + [Fr(0)] * (self.N - 1), self.S.mul(self.tanU, self.tanU))
        self.M, self.b = M_b(self.I, self.T)
        self.Minv = mat_inv(self.M)
        self.absMinv = mabs(self.Minv)
        self.J = J_matrix(self.I, self.e)
        self.G = [[-x for x in row] for row in mat_mul(mat_mul(self.Minv, self.J), self.Finv)]
        # distinct orders and multiplicities
        self.orders = {}
        for x in self.m:
            self.orders[x] = self.orders.get(x, 0) + 1
        # Taylor coefficients at a of p_c(z) = sum_j (-1)^j G_{jc} z^{n-j}, j = 1..n
        self.taylor = {}
        for a in self.orders:
            tc = []
            for c in range(n):
                poly = [Fr(0)] * (n + 1)  # coefficient of z^d
                for j in range(1, n + 1):
                    poly[n - j] += (-1) ** j * self.G[j - 1][c]
                # shift: p(a + w) = sum_l coef_l w^l
                coef = [sum(poly[d] * comb(d, l) * Fr(a) ** (d - l) for d in range(l, n + 1)) for l in range(n + 1)]
                tc.append([abs(x) for x in coef])
            self.taylor[a] = tc

    def bounds(self, rad):
        """Prop. 6.10 steps 1-5 for componentwise H-radii rad (length n). Returns dict or None if rho(A)<1 not certified."""
        n, N, S = self.n, self.N, self.S
        Delta = mat_vec(self.absFinv, rad)
        dU = [Fr(0)] * N
        for k in range(1, n):
            dU[2 * k - 1] = Delta[k] / (2 * k - 1)
        tau_lin = S.mul(self.abs_s, dU)
        tan_shift = S.compose(self.tanc, S.add(self.U, dU))
        tau_rem = S.sub(S.sub(tan_shift, self.tanU), S.mul(self.sec2U, dU))
        tau = S.add(tau_lin, tau_rem)
        e = self.e
        E_ = lambda k: e[k] if 0 <= k <= n else Fr(0)
        rho = [sum(tau[2 * i + 1] * E_(2 * j - 2 * i) for i in range(j + 1)) for j in range(n - 1)] + [Delta[0] * E_(n)]
        rho_rem = [sum(tau_rem[2 * i + 1] * E_(2 * j - 2 * i) for i in range(j + 1)) for j in range(n - 1)] + [Fr(0)]
        DM = [[Fr(0)] * n for _ in range(n)]
        for j in range(n - 1):
            for i in range(j):
                if 2 * j - 2 * i <= n:
                    DM[j][2 * j - 2 * i - 1] += tau[2 * i + 1]
        DM[n - 1][n - 1] += Delta[0]
        A = mat_mul(self.absMinv, DM)
        IA = [[Fr(int(i == j)) - A[i][j] for j in range(n)] for i in range(n)]
        IAinv = mat_inv(IA)
        out = dict(Delta=Delta, dU=dU, tau_lin=tau_lin, tau_rem=tau_rem, tau=tau, rho=rho, rho_rem=rho_rem,
                   DM=DM, A=A, ok_spec=False)
        if IAinv is None:
            return out
        v = mat_vec(IAinv, [Fr(1)] * n)
        if not all(x > 0 for x in v):
            return out
        Erad = mat_vec(IAinv, mat_vec(self.absMinv, rho))
        AE = mat_vec(A, Erad)
        varrho = [x + y for x, y in zip(mat_vec(self.absMinv, rho_rem), AE)]
        out.update(ok_spec=True, v=v, E=Erad, varrho=varrho)
        return out

    def lhs(self, a, r):
        val = r ** self.orders[a]
        for b, kb in self.orders.items():
            if b != a:
                d = abs(Fr(a) - b) - r
                if d <= 0:
                    return None
                val *= d ** kb
        return val

    def test_i(self, a, r, B):
        L = self.lhs(a, r)
        if L is None:
            return False
        n = self.n
        R = sum(B['E'][j - 1] * (Fr(a) + r) ** (n - j) for j in range(1, n + 1))
        return L > R

    def test_ii(self, a, r, B, rad):
        L = self.lhs(a, r)
        if L is None:
            return False
        n = self.n
        lin = sum(rad[c] * sum(tc * r ** l for l, tc in enumerate(self.taylor[a][c])) for c in range(n))
        rem = sum(B['varrho'][j - 1] * (Fr(a) + r) ** (n - j) for j in range(1, n + 1))
        return L > lin + rem

    def certify(self, rad, ladder=LADDER):
        """Returns dict: per order a, first ladder radius passing (i) and (ii); overall verdicts."""
        B = self.bounds(rad)
        res = dict(spec=B['ok_spec'], per={}, i=False, ii=False, mixed=False)
        if not B['ok_spec']:
            return res
        for a in self.orders:
            ri = next((r for r in ladder if self.test_i(a, r, B)), None)
            rii = next((r for r in ladder if self.test_ii(a, r, B, rad)), None)
            res['per'][a] = (ri, rii)
        res['i'] = all(v[0] is not None for v in res['per'].values())
        res['ii'] = all(v[1] is not None for v in res['per'].values())
        res['mixed'] = all(v[0] is not None or v[1] is not None for v in res['per'].values())
        res['B'] = B
        return res

    def certify_abs(self, delta, ladder=LADDER):
        return self.certify([Fr(delta)] * self.n, ladder)

    def certify_rel(self, eps, ladder=LADDER):
        return self.certify([Fr(eps) * abs(h) for h in self.H], ladder)

# ---------------------------------------------------------------- exact root location (Routh-Hurwitz)

def poly_shift(coef_desc, x):
    """coef_desc: coefficients highest degree first of p(z); returns those of p(z + x)."""
    n = len(coef_desc) - 1
    asc = list(reversed(coef_desc))
    out = [sum(asc[d] * comb(d, l) * x ** (d - l) for d in range(l, n + 1)) for l in range(n + 1)]
    return list(reversed(out))


def rhp_count(coef_desc):
    """Number of roots with Re z > 0 of a real polynomial (Routh array). None if degenerate."""
    c = [Fr(x) for x in coef_desc]
    n = len(c) - 1
    if c[0] == 0:
        return None
    r0 = c[0::2]
    r1 = c[1::2]
    rows = [r0, r1 + [Fr(0)] * (len(r0) - len(r1))]
    for _ in range(n - 1):
        a, b = rows[-2], rows[-1]
        if b[0] == 0:
            return None
        new = [(b[0] * a[i + 1] - a[0] * b[i + 1]) / b[0] if i + 1 < len(a) else Fr(0) for i in range(len(a) - 1)]
        new = new + [Fr(0)] * (len(a) - len(new))
        rows.append(new)
    first = [row[0] for row in rows[: n + 1]]
    if any(x == 0 for x in first):
        return None
    return sum(1 for x, y in zip(first, first[1:]) if (x > 0) != (y > 0))


def count_real_part_greater(coef_desc, x):
    return rhp_count(poly_shift(coef_desc, Fr(x)))


def strip_count(coef_desc, lo, hi):
    a = count_real_part_greater(coef_desc, lo)
    b = count_real_part_greater(coef_desc, hi)
    if a is None or b is None:
        return None
    return a - b


def poly_from_e(e):
    return [(-1) ** j * e[j] for j in range(len(e))]


def recovers(e_tilde, m):
    """Exact: rounding the real parts of the roots of q~ returns the multiset m."""
    coef = poly_from_e(e_tilde)
    orders = {}
    for x in m:
        orders[x] = orders.get(x, 0) + 1
    for a, k in orders.items():
        c = strip_count(coef, Fr(a) - Fr(1, 2), Fr(a) + Fr(1, 2))
        if c is None or c != k:
            return False
    return True

# ---------------------------------------------------------------- Theorem S2 / S3 / S4 constants

def sigma(n, kmax):
    """sigma_k(n) = [w^k] artanh(w) sec^2((n+1) artanh w), k = 0..kmax."""
    N = kmax + 1
    S = Ser(N)
    at = [Fr(0)] * N
    for k in range(1, N, 2):
        at[k] = Fr(1, k)
    tanc = tan_coeffs(N)
    inner = [(n + 1) * x for x in at]
    tn = S.compose(tanc, inner)
    sec2 = S.add([Fr(1)] + [Fr(0)] * (N - 1), S.mul(tn, tn))
    return S.mul(at, sec2)


def zeta(n):
    sg = sigma(n, 2 * n)
    best = Fr(1)
    for j in range(n - 1):
        s = sum((sg[2 * i + 1] for i in range(j) if 2 <= 2 * j - 2 * i <= n), Fr(0))
        best = max(best, s)
    return best


ELL = [Fr(2), Fr(14), Fr(498), Fr(4062), Fr(56230, 3), Fr(303654, 5)]


def delta_thm(m):
    m = sorted(m)
    n = len(m)
    mu = Fr(max(m))
    mh = [Fr(x) / mu for x in m]
    eh = esym(mh)
    Ih = invariants_from_e(eh)
    Mh, _ = M_b(Ih)
    Mhi = mat_inv(Mh)
    kappa = max(sum(abs(x) for x in row) for row in Mhi)
    sg = sigma(n, 2 * n)
    z = zeta(n)
    rho_n = max([eh[n]] + [sum(sg[2 * i + 1] * (eh[2 * j - 2 * i] if 2 * j - 2 * i <= n else 0) for i in range(j + 1))
                           for j in range(n - 1)])
    lam = max([mu * ELL[0]] + [ELL[r] * mu ** (1 - 2 * r) for r in range(1, n)])
    orders = {}
    for x in m:
        orders[x] = orders.get(x, 0) + 1
    terms = [1 / lam, 1 / (2 * kappa * z * lam)]
    for a, k in orders.items():
        Q = Fr(1)
        for b, kb in orders.items():
            if b != a:
                Q *= abs(Fr(a - b) / mu) ** kb
        terms.append(Q * Fr(2) ** (k - 1) / (3 ** n * (2 * mu) ** k * 2 * kappa * rho_n * lam))
    return min(terms), dict(kappa=kappa, zeta=z, rho_n=rho_n, lam=lam, terms=terms)

# ---------------------------------------------------------------- rounding helpers (decimal, exact)

def sig_round(x, digits, mode):
    """Round positive Fraction x to `digits` significant figures, mode 'down' or 'up'. Returns (mantissa_int, exp)."""
    x = Fr(x)
    assert x > 0
    e = 0
    while x >= 10:
        x /= 10; e += 1
    while x < 1:
        x *= 10; e -= 1
    scaled = x * 10 ** (digits - 1)
    q = scaled.numerator // scaled.denominator
    if mode == 'up' and q != scaled:
        q += 1
    if mode == 'near':
        q = int((scaled + Fr(1, 2)).numerator // (scaled + Fr(1, 2)).denominator)
    return q, e - (digits - 1)


def from_sig(q, e):
    return Fr(q) * Fr(10) ** e


def fmt(q, e, digits):
    s = str(q)
    if len(s) > digits:  # carry, e.g. 9.99 -> 10.0
        s = s[:digits]; e += 1
    return f"{s[0]}.{s[1:]}e{e + digits - 1:+03d}"


def parse_sci(s):
    mant, ex = s.lower().split('e')
    return Fr(mant) * Fr(10) ** int(ex)


def next_up(s):
    """Next 4-s.f. (or same-digit) value above the printed decimal string s (same number of digits)."""
    mant, ex = s.lower().split('e')
    d = len(mant.replace('.', '')) - 1
    return Fr(mant) * Fr(10) ** int(ex) + Fr(10) ** (int(ex) - d)

"""Exact arithmetic shared by the stability scripts (theory/stability/).

Conventions (the supplement; Ucar's thesis):
  positive Laplacian, K = kappa = -1, genus 0, n cone points of orders m_i,
  Z(t) = sum_j exp(-lambda_j t) ~ sum_{nu >= -1} H_nu t^nu,
  H_{-1} = Area/(4 pi) = (n - 2 - R)/2,
  H_nu   = (Area/(4 pi)) * alpha_{nu+1} + sum_i b_nu(m_i)          (nu >= 0),
  alpha_j = a_j^sm / vol  (smooth coefficients, Ucar (4.35)),
  b_nu(m) = (-1)^nu p_nu(m)/m,  p_nu even of degree 2 nu + 2 (Ucar (4.25)+(4.33)).
Invariants  I_n(m) = (R, P_1, P_3, ..., P_{2n-3}),  R = sum 1/m_i,  P_k = sum m_i^k.
Everything here is exact (fractions.Fraction); nothing is floating point.
"""
from fractions import Fraction as Fr
from itertools import combinations
from math import comb, factorial

# ------------------------------------------------------------------ Bernoulli
_B = {0: Fr(1)}


def bern(n):
    """Bernoulli numbers, B_1 = -1/2 (own recursion)."""
    if n not in _B:
        _B[n] = -sum(comb(n + 1, j) * bern(j) for j in range(n)) / (n + 1)
    return _B[n]


def bern_half(n):
    """B_n(1/2) = (2^{1-n} - 1) B_n  (Ucar Lemma 4.14)."""
    return Fr(1) if n == 0 else (Fr(1, 2 ** (n - 1)) - 1) * bern(n)


# ------------------------------------------------------------------ cone terms
def cone_poly(l):
    """Coefficients pi_{l,k} (k = 0..l+1) of p_l(m) = sum_k pi_{l,k} m^{2k},
    where b_l(m)/kappa^l = p_l(m)/m (Ucar (4.25) + (4.33))."""
    # c_l(pi/k) * k = (-1)^l/(4 (l+1)! (2l+1)) sum_j binom(2l+2,2j) (k^{2j}-1) B_{2j} B_{2l+2-2j}(1/2)
    def c_times_k(l):
        out = [Fr(0)] * (l + 2)
        pre = Fr((-1) ** l, 4 * factorial(l + 1) * (2 * l + 1))
        for j in range(l + 2):
            w = pre * comb(2 * l + 2, 2 * j) * bern(2 * j) * bern_half(2 * l + 2 - 2 * j)
            out[j] += w
            out[0] -= w
        return out
    tot = [Fr(0)] * (l + 2)
    for i in range(l + 1):
        c = c_times_k(l - i)
        w = Fr(2, 4 ** i * factorial(i))
        for k, v in enumerate(c):
            tot[k] += w * v
    return tot


def b_cone(l, m, kappa=-1):
    """Coefficient of t^l contributed by one cone point of order m (any nonzero rational)."""
    m = Fr(m)
    return Fr(kappa) ** l * sum(c * m ** (2 * k) for k, c in enumerate(cone_poly(l))) / m


def alpha_smooth(j, kappa=-1):
    """a_j^sm / vol, Ucar (4.35)."""
    s = sum(comb(j, l) * Fr(-4) ** l * bern_half(2 * l) for l in range(j + 1))
    return s / (factorial(j) * 4 ** j) * Fr(kappa) ** j


def alpha_smooth_selberg(j):
    """Independent derivation of a_j^sm/vol at K = -1 from the Selberg identity term
    (Area/4pi) int_R r tanh(pi r) e^{-t(1/4 + r^2)} dr
      = (Area/4pi) e^{-t/4} [ 1/t - 4 sum_k (-t)^k/k! eta_k ],
    eta_k = int_0^inf r^{2k+1}/(e^{2 pi r}+1) dr = (1 - 2^{-2k-1}) (-1)^k B_{2k+2} / (2(2k+2)).
    Returns the coefficient of t^{j-1} in e^{-t/4}[1/t - 4 sum ...], i.e. alpha_j."""
    # series of g(t) = t * (bracket) = 1 - 4 sum_k (-1)^k eta_k t^{k+1}/k!
    N = j + 1
    g = [Fr(0)] * (N + 1)
    g[0] = Fr(1)
    for k in range(N):
        eta = (1 - Fr(1, 2 ** (2 * k + 1))) * (-1) ** k * bern(2 * k + 2) / (2 * (2 * k + 2))
        if k + 1 <= N:
            g[k + 1] += -4 * Fr((-1) ** k, factorial(k)) * eta
    ex = [Fr(-1, 4) ** i / factorial(i) for i in range(N + 1)]
    return sum(ex[i] * g[j - i] for i in range(j + 1))


# ------------------------------------------------------------------ invariants
def invariants(m, n=None):
    """I_n(m) = (R, P_1, P_3, ..., P_{2n-3}), n = len(m) by default."""
    n = len(m) if n is None else n
    m = [Fr(x) for x in m]
    return [sum(1 / x for x in m)] + [sum(x ** (2 * k - 1) for x in m) for k in range(1, n)]


def heat_direct(m, N):
    """First N heat coefficients H_{-1}, ..., H_{N-2} of the genus-0 orbifold with
    cone orders m (n = len(m)), evaluated cone point by cone point."""
    n = len(m)
    R = sum(Fr(1) / Fr(x) for x in m)
    A4pi = (n - 2 - R) / 2
    H = [A4pi]
    for nu in range(0, N - 1):
        H.append(A4pi * alpha_smooth(nu + 1) + sum(b_cone(nu, x) for x in m))
    return H


def front_end(n):
    """H = L I + h0 for the first n heat coefficients (H_{-1}..H_{n-2}) and
    I = I_n = (R, P_1, ..., P_{2n-3}).  L is lower triangular; its leading
    blocks do not depend on n, only h0 does."""
    L = [[Fr(0)] * n for _ in range(n)]
    h0 = [Fr(0)] * n
    L[0][0] = Fr(-1, 2)
    h0[0] = Fr(n - 2, 2)
    for nu in range(0, n - 1):
        row = nu + 1
        p = cone_poly(nu)
        a = alpha_smooth(nu + 1)
        h0[row] = Fr(n - 2, 2) * a
        L[row][0] = -a / 2 + (-1) ** nu * p[0]
        for k in range(1, nu + 2):
            L[row][k] = (-1) ** nu * p[k]
    return L, h0


# ------------------------------------------------------------------ linear algebra (exact)
def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def mat_vec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def identity(n):
    return [[Fr(int(i == j)) for j in range(n)] for i in range(n)]


def mat_inv(A):
    n = len(A)
    M = [list(map(Fr, row)) + identity(n)[i] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def det(A):
    n = len(A)
    M = [list(map(Fr, row)) for row in A]
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
            M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return d


def abs_mat(A):
    return [[abs(x) for x in row] for row in A]


def inf_norm(A):
    return max(sum(abs(x) for x in row) for row in A)


# ------------------------------------------------------------------ power series
def ser_mul(a, b, N):
    out = [Fr(0)] * (N + 1)
    for i, x in enumerate(a[:N + 1]):
        if x:
            for j, y in enumerate(b[:N + 1 - i]):
                out[i + j] += x * y
    return out


def ser_tanh(U, N):
    """tanh(U(z)) mod z^{N+1}, U(0) = 0, via T' = (1 - T^2) U'."""
    T = [Fr(0)] * (N + 1)
    for k in range(1, N + 1):
        T2 = ser_mul(T, T, k - 1)
        s = Fr(0)
        for j in range(1, k + 1):
            s += j * U[j] * ((1 if k - j == 0 else 0) - T2[k - j])
        T[k] = s / k
    return T


def ser_tan(U, N):
    """tan(U(z)) mod z^{N+1}, U(0) = 0, via T' = (1 + T^2) U'."""
    T = [Fr(0)] * (N + 1)
    for k in range(1, N + 1):
        T2 = ser_mul(T, T, k - 1)
        s = Fr(0)
        for j in range(1, k + 1):
            s += j * U[j] * ((1 if k - j == 0 else 0) + T2[k - j])
        T[k] = s / k
    return T


def U_series(Podd, N):
    """U(z) = sum_{k odd} P_k z^k / k from a dict {k: P_k}."""
    U = [Fr(0)] * (N + 1)
    for k, v in Podd.items():
        if k <= N:
            U[k] = Fr(v) / k
    return U


# ------------------------------------------------------------------ Theorem B system
def theorem_B_system(I):
    """M e = b in the unknowns e_1..e_n, from data I = (R, P_1, ..., P_{2n-3}).
    Rows j = 0..n-2: e_{2j+1} - sum_{i=0}^{j} T_{2i+1} e_{2j-2i} = 0 (e_0 = 1 moved to b);
    row n-1: e_{n-1} - R e_n = 0."""
    n = len(I)
    R = I[0]
    Podd = {2 * k - 1: I[k] for k in range(1, n)}
    N = 2 * n - 3
    T = ser_tanh(U_series(Podd, N), N) if N >= 1 else [Fr(0)]
    M = [[Fr(0)] * n for _ in range(n)]
    b = [Fr(0)] * n
    for j in range(n - 1):
        if 2 * j + 1 <= n:
            M[j][2 * j] += 1                       # e_{2j+1}, column index 2j
        for i in range(j + 1):
            idx = 2 * j - 2 * i                    # e_{idx}
            if idx == 0:
                b[j] += T[2 * i + 1]
            elif idx <= n:
                M[j][idx - 1] -= T[2 * i + 1]
    M[n - 1][n - 2] += 1
    M[n - 1][n - 1] -= R
    return M, b, T


def elementary(m):
    """e_0..e_n of the multiset m."""
    e = [Fr(1)]
    for x in m:
        x = Fr(x)
        e = [a + x * b for a, b in zip(e + [Fr(0)], [Fr(0)] + e)]
    return e


def pair_product(m):
    out = Fr(1)
    for a, b in combinations(m, 2):
        out *= Fr(a) + Fr(b)
    return out


def c_sign(n):
    """c_n = (-1)^{n(n+1)/2}  (the paper section 2, Lemma S2.1)."""
    return (-1) ** (n * (n + 1) // 2)


def hurwitz_B(e, n):
    """B[k][j-1] = (-1)^{j+1} e_{2k+1-j}, k = 0..n-1, j = 1..n: the map
    d -> coefficients of W = E_f O_D - E_D O_f."""
    def E(i):
        return e[i] if 0 <= i <= n else Fr(0)
    return [[(-1) ** (j + 1) * E(2 * k + 1 - j) for j in range(1, n + 1)] for k in range(n)]


def hurwitz_S(e, n):
    """S[k][i] = e_{2(k-i)} (k <= n-2); last row (-1)^n e_n in column n-1
    (top coefficient of W is (-1)^n (e_n d_{n-1} - e_{n-1} d_n)).  B = S M."""
    S = [[Fr(0)] * n for _ in range(n)]
    for k in range(n - 1):
        for i in range(k + 1):
            if 2 * (k - i) <= n:
                S[k][i] = e[2 * (k - i)]
    S[n - 1][n - 1] = Fr((-1) ** n) * e[n]
    return S

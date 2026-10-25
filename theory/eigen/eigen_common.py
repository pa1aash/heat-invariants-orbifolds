"""Shared exact and high-precision routines for theory/eigen.

Conventions are those of Section 2 of the paper: c_1 = Area/(4 pi) is the t^{-1} coefficient,
c_{l+2} the t^l coefficient, b_l(m) = (-1)^l p_l(m)/m the cone term at curvature -1, alpha_k the
smooth coefficients per Area/(4 pi).  Everything that is a claim of exactness is computed with
fractions.Fraction; transcendental quantities use mpmath at a stated working precision.

Run any script of this directory with the scratch virtual environment that has mpmath, sympy, numpy.
"""
from fractions import Fraction as Fr
from functools import lru_cache
from math import comb, factorial, gcd
import itertools

import mpmath as mp


def mpq(x):
    """mpf from a Fraction (or anything mpmath accepts)."""
    if isinstance(x, Fr):
        return mp.mpf(x.numerator) / x.denominator
    return mp.mpf(x)


def check(cond, msg):
    """A real assert that survives python -O: raise and exit nonzero."""
    if not cond:
        raise AssertionError(msg)


# ------------------------------------------------------------------ exact numbers
@lru_cache(maxsize=None)
def bernoulli(n):
    """Bernoulli number B_n (B_1 = -1/2), exact, by the standard recurrence."""
    if n == 0:
        return Fr(1)
    if n == 1:
        return Fr(-1, 2)
    if n % 2 == 1:
        return Fr(0)
    s = Fr(0)
    for k in range(n):
        s += comb(n + 1, k) * bernoulli(k)
    return -s / (n + 1)


@lru_cache(maxsize=None)
def sigma_coef(i):
    """u/sin u = sum_i sigma_i u^{2i}; sigma_i = (2^{2i}-2)|B_{2i}|/(2i)! (sigma_0 = 1)."""
    if i == 0:
        return Fr(1)
    return Fr(2 ** (2 * i) - 2) * abs(bernoulli(2 * i)) / factorial(2 * i)


@lru_cache(maxsize=None)
def phi(k, m):
    """Taylor coefficient phi_k(m) of Phi_m(u) = (cot u - m cot mu)/(4 m sin u) at u^{2k}
    (the paper's eq. (phik))."""
    m = Fr(m)
    s = Fr(0)
    for n in range(1, k + 2):
        s += sigma_coef(k + 1 - n) * Fr(4 ** n) * abs(bernoulli(2 * n)) / factorial(2 * n) * (m ** (2 * n) - 1)
    return s / (4 * m)


@lru_cache(maxsize=None)
def p_poly(l, m):
    """p_l(m) of the paper's eq. (bl)."""
    m = Fr(m)
    s = Fr(0)
    for k in range(l + 1):
        s += Fr(factorial(2 * k), factorial(k) * factorial(l - k)) * m * phi(k, m)
    return s / Fr(4 ** l)


def b_cone(l, m):
    """Cone term b_l(m) = (-1)^l p_l(m)/m at curvature -1."""
    return (-1) ** l * p_poly(l, m) / Fr(m)


@lru_cache(maxsize=None)
def alpha(k):
    """Smooth coefficient alpha_k (per Area/(4 pi))."""
    s = Fr(0)
    for l in range(k + 1):
        B = (Fr(2) ** (1 - 2 * l) - 1) * bernoulli(2 * l)  # B_{2l}(1/2)
        s += comb(k, l) * Fr(-4) ** l * B
    return Fr((-1) ** k, factorial(k) * 4 ** k) * s


@lru_cache(maxsize=None)
def mu_moment(k):
    """mu_k = 4 int_0^inf r^{2k+1}/(e^{2 pi r}+1) dr = (1-2^{-2k-1})|B_{2k+2}|/(k+1)."""
    return (1 - Fr(1, 2 ** (2 * k + 1))) * abs(bernoulli(2 * k + 2)) / (k + 1)


@lru_cache(maxsize=None)
def lead_coef(l):
    """a_{l,l+1} = |B_{2l+2}|/(2 (l+1)! (2l+1)): leading coefficient of p_l (Lemma 2.8)."""
    return abs(bernoulli(2 * l + 2)) / (2 * factorial(l + 1) * (2 * l + 1))


@lru_cache(maxsize=None)
def p_coeffs(l):
    """Coefficients a_{l,k} (k = 0..l+1) of p_l(x) = sum_k a_{l,k} x^{2k}, by exact interpolation
    in x^2 at l+2 points."""
    pts = [Fr(j + 2) for j in range(l + 2)]
    # solve Vandermonde in y = x^2
    n = l + 2
    A = [[(x * x) ** k for k in range(n)] for x in pts]
    b = [p_poly(l, x) for x in pts]
    # Gaussian elimination over Q
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        b[c], b[piv] = b[piv], b[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]
                A[r] = [A[r][j] - f * A[c][j] for j in range(n)]
                b[r] -= f * b[c]
    return [b[k] / A[k][k] for k in range(n)]


def lcm_upto(M):
    L = 1
    for k in range(2, M + 1):
        L = L * k // gcd(L, k)
    return L


# ------------------------------------------------------------------ signatures
def area_over_2pi(g, orders):
    return Fr(2 * g - 2) + sum(1 - Fr(1, m) for m in orders)


def heat_coeffs(g, orders, L):
    """(c_1, ..., c_L) exactly (Proposition 2.7)."""
    s = area_over_2pi(g, orders)
    c1 = s / 2  # Area/(4 pi)
    out = [c1]
    for j in range(2, L + 1):
        out.append(alpha(j - 1) * c1 + sum(b_cone(j - 2, m) for m in orders))
    return out


def signatures(A_over_pi, M):
    """All hyperbolic signatures (g; m_1..m_n), orders in [2, M], with Area <= pi * A_over_pi
    (A_over_pi a Fraction).  Area = 2 pi s, so s <= A_over_pi/2."""
    smax = Fr(A_over_pi) / 2
    out = []
    g = 0
    while Fr(2 * g - 2) <= smax:
        # each cone point adds at least 1/2 to s
        nmax = int((smax - (2 * g - 2)) * 2) if smax - (2 * g - 2) >= 0 else -1
        for n in range(0, max(nmax, -1) + 1):
            for orders in itertools.combinations_with_replacement(range(2, M + 1), n):
                s = area_over_2pi(g, orders)
                if 0 < s <= smax:
                    out.append((g, orders))
        g += 1
    return out


# ------------------------------------------------------------------ transcendental terms
def elliptic_term(m, t, dps=30):
    """E_m(t) of Theorem 2.3 by quadrature (one cone point of order m)."""
    with mp.workdps(dps):
        t = mp.mpf(t)
        tot = mp.mpf(0)
        for j in range(1, m):
            th = mp.pi * j / m
            f = lambda r: mp.exp(-2 * th * r - t * (mp.mpf(1) / 4 + r * r)) / (1 + mp.exp(-2 * mp.pi * r))
            I = mp.quad(f, [-mp.inf, -20, -5, 0, 5, 20, mp.inf])
            tot += I / (2 * m * mp.sin(th))
        return tot


def identity_term(area, t, dps=30):
    """I(t) = Area/(4 pi) int r tanh(pi r) h_t(r) dr."""
    with mp.workdps(dps):
        t = mp.mpf(t)
        f = lambda r: r * mp.exp(-t * (mp.mpf(1) / 4 + r * r)) * mp.tanh(mp.pi * r)
        return mp.mpf(area) / (4 * mp.pi) * 2 * mp.quad(f, [0, 1, 5, mp.inf])


def G_sig(g, orders, t, dps=30):
    """G_sigma(t) = I(t) + sum_i E_{m_i}(t): the part of the heat trace fixed by the signature."""
    area = 2 * mp.pi * mp.mpf(area_over_2pi(g, orders).numerator) / area_over_2pi(g, orders).denominator
    return identity_term(area, t, dps) + sum(elliptic_term(m, t, dps) for m in orders)


def Phi_closed(m, u):
    """Closed form of Phi_m (Lemma 2.6)."""
    u = mp.mpf(u)
    return (mp.cot(u) - m * mp.cot(m * u)) / (4 * m * mp.sin(u))


# ------------------------------------------------------------------ constants
def cone_sep(eps, M):
    """d_0(eps, M): lower bound for the distance between two distinct elliptic fixed points
    (Lemma D1): min(eps/2, arccosh(1 + 2/(pi^2 M^2)))."""
    eps = mp.mpf(eps)
    return min(eps / 2, mp.acosh(1 + 2 / (mp.pi ** 2 * mp.mpf(M) ** 2)))


def ball_constants(eps, M):
    """(r_0, rho_1, v_0) of Lemma D2: every ball of radius 2 r_0 has area >= v_0."""
    d0 = cone_sep(eps, M)
    r0 = d0 / 2
    rho1 = mp.asinh(mp.sinh(r0) * mp.sin(mp.pi / M))
    v0 = 2 * mp.pi * min((mp.cosh(r0) - 1) / M, mp.cosh(rho1) - 1)
    return r0, rho1, v0


def diam_bound(A, eps, M):
    """D(A, eps, M) = 4 r_0 A / v_0 (Theorem D)."""
    r0, rho1, v0 = ball_constants(eps, M)
    return 4 * r0 * mp.mpf(A) / v0


def count_constant(A, eps, M):
    """K(A, eps, M) of the lemma on counting closed geodesics without the diameter: every orbifold of
    C(A, eps, M) has n_O(x) <= K e^x, with K = A M e^{10 r_0 + rho_1} / (2 v_0 (cosh rho_1 - 1))."""
    r0, rho1, v0 = ball_constants(eps, M)
    return mp.mpf(A) * M * mp.e ** (10 * r0 + rho1) / (2 * v0 * (mp.cosh(rho1) - 1))


# ------------------------------------------------------------------ Tasks 2-4 constants
def g_abs(k, m):
    """|g_k(m)| = (2k)!/(k! 4^k) phi_k(m): the k-th Taylor coefficient of e^{t/4} E_m(t), unsigned."""
    return Fr(factorial(2 * k), factorial(k) * 4 ** k) * phi(k, m)


def Qcone(m, K, tbar=None):
    """Remainder constant of Proposition eig:remcone (enveloping form, valid for every t > 0):
    |E_m(t) - sum_{l<K} b_l(m) t^l| <= |b_K(m)| t^K.  (tbar is accepted and ignored.)"""
    return mpq(abs(b_cone(K, m)))


def Qarea(K, tbar=None):
    """Remainder constant of Proposition eig:remarea (enveloping form, valid for every t > 0), per
    unit Area/(4 pi): |I(t) - (Area/4pi) sum_{k<=K} alpha_k t^{k-1}| <= (Area/4pi) |alpha_{K+1}| t^K."""
    return mpq(abs(alpha(K + 1)))


def Qcone_crude(m, K, tbar):
    """The first form: |g_K| + e^{tbar/4} sum_{k<K} |g_k| 4^{k-K}/(K-k)!  >= |b_K(m)|."""
    s = mpq(g_abs(K, m))
    tail = sum(mpq(g_abs(k, m)) * mpq(4) ** (k - K) / mp.factorial(K - k) for k in range(K))
    return s + mp.e ** (mpq(tbar) / 4) * tail


def hyp_bound_small(t, eps, D, area_lb=None):
    """Lemma 2.5 with ell >= eps, diam <= D, Area >= area_lb (default pi/21), valid for
    0 < t <= eps^2/(2(1+eps)):  B(eps, D, t) with the area replaced by its lower bound."""
    t, eps, D = mpq(t), mpq(eps), mpq(D)
    if area_lb is None:
        area_lb = mp.pi / 21
    check(t <= eps ** 2 / (2 * (1 + eps)), "hyp_bound_small outside its range")
    return (mp.pi * mp.e ** (3 * D) * eps * mp.e ** (eps / 2) / (area_lb * (1 - mp.e ** (-eps)))
            * (1 + 2 * t / (eps - t)) * mp.e ** (-eps ** 2 / (4 * t)) / mp.sqrt(4 * mp.pi * t))


def hyp_bound_all(t, eps, D, area_lb=None):
    """Proposition C2: for every t > 0,
    Hyp(t) <= 2 (pi e^{3D}/Area) e^{17t/4 - 1/2 - eps} / (sqrt(pi) (1 - e^{-eps}))."""
    t, eps, D = mpq(t), mpq(eps), mpq(D)
    if area_lb is None:
        area_lb = mp.pi / 21
    C = mp.pi * mp.e ** (3 * D) / area_lb
    return 2 * C * mp.e ** (17 * t / 4 - mp.mpf(1) / 2 - eps) / (mp.sqrt(mp.pi) * (1 - mp.e ** (-eps)))


def theorem_E_constants(A, eps, M, area_lb=None):
    """All constants of Theorem E for the class C(A, eps, M).  Returns a dict of mpf/ints."""
    A, eps = mpq(A), mpq(eps)
    # n_* bounds the number of cone points (Lemma eig:elem); k_* bounds the first index at which two
    # signatures of S(A, M) differ: floor(A/pi) + 4 by the separation theorem and M by the theorem on
    # bounded cone orders (the paper, part (i)), whichever is smaller
    nstar = int(mp.floor(A / mp.pi)) + 4
    kstar = min(nstar, int(M))
    LM = lcm_upto(M)
    # the lcm of the orders of two signatures in S(A, M) divides L_M and is at most M^{2 n_*}
    # (at most 2 n_* orders, each <= M), so delta_1 = 1/(2 min(L_M, M^{2 n_*}))
    LMeff = min(LM, int(M) ** (2 * nstar))
    dmin = {1: Fr(1, 2 * LMeff)}
    for k in range(2, kstar + 1):
        dmin[k] = lead_coef(k - 2)
    gamma = min(dmin.values())
    Q = {K: A / (4 * mp.pi) * Qarea(K, 1) + nstar * Qcone(M, K, 1) for K in range(0, kstar)}
    t1 = min([mp.mpf(1)] + [mpq(dmin[k]) / (4 * Q[k - 1]) for k in range(1, kstar + 1)])
    t2 = eps ** 2 / (2 * (1 + eps))
    D = diam_bound(A, eps, M)
    Kc = count_constant(A, eps, M)
    if area_lb is None:
        area_lb = mp.pi / 21
    p = kstar - mp.mpf(3) / 2
    # The count n_O(x) <= K e^x (no diameter) in Lemma 2.2's integration by parts gives, for t <= t_2,
    # Hyp(t) <= B_*(t) = K eps e^{eps/2} / (1 - e^{-eps}) * 3 * e^{-eps^2/4t} / sqrt(4 pi t)
    # (1 + 2t/(eps - t) <= 3 on t <= t_2).  area_lb is no longer used.
    pref = Kc * 3 * eps * mp.e ** (eps / 2) / ((1 - mp.e ** (-eps)) * mp.sqrt(4 * mp.pi))
    # need pref t^{-1/2} e^{-y} <= gamma t^{kstar-2}/16, y = eps^2/(4t): e^{-y} <= c1 t^p, c1 = gamma/(16 pref)
    log_c1 = mp.log(mpq(gamma)) - mp.log(16 * pref)
    y0 = 2 * (p * mp.log(2 * p / mp.e) - log_c1 - p * mp.log(eps ** 2 / 4))
    check(y0 > 0, "y0 > 0")
    t3 = eps ** 2 / (4 * y0)
    tstar = min(t1, t2, t3)
    Gam = mpq(gamma) * tstar ** (kstar - 2) / 2

    def Bstar(t):
        return pref * mp.e ** (-eps ** 2 / (4 * t)) / mp.sqrt(t)

    check(Bstar(tstar) <= Gam / 8, "B_*(t*) <= Gamma/8")
    cone0 = nstar * mpq(Fr(M * M - 1, 12 * M))

    def Zb(s):
        check(s <= tstar, "Zb only for s <= t*")
        return A / (4 * mp.pi * s) + cone0 + 1

    Lam = (2 / tstar) * mp.log(8 * Zb(tstar / 2) / Gam)
    check(Lam * tstar >= 1, "1/Lambda <= t*")
    N = int(mp.floor(mp.e * Zb(1 / Lam))) + 1
    delta = min(1 / tstar, Gam / (8 * mp.e * N * tstar))
    return dict(A=A, eps=eps, M=M, kstar=kstar, nstar=nstar, LM=LM, LMeff=LMeff, gamma=gamma, Q=Q, t1=t1, t2=t2,
                D=D, K=Kc, y0=y0, t3=t3, tstar=tstar, Gamma=Gam, Bstar_tstar=Bstar(tstar), Lambda=Lam, N=N,
                delta=delta, Zb_half=Zb(tstar / 2))


# ------------------------------------------------------------------ committed spectra
def load_triangle_spectrum(pqr, root=None):
    """Union of the Neumann and Dirichlet spectra of the triangle (= spectrum of O(p,q,r)),
    from numerics/data; returns (sorted list of (lambda, err_estimate)), lambda_complete) where
    lambda_complete = min of the largest computed N and D eigenvalue."""
    import csv, os
    if root is None:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    name = "-".join(str(x) for x in pqr)
    vals = []
    tops = []
    for side in ("N", "D"):
        path = os.path.join(root, "numerics", "data", f"eigenvalues_{name}_{side}.csv")
        rows = list(csv.DictReader(open(path)))
        v = [(float(r["lambda"]), float(r["err_estimate"])) for r in rows]
        tops.append(max(x for x, _ in v))
        vals += v
    vals.sort()
    return vals, min(tops)

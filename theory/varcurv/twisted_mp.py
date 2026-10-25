"""Donnelly's rotation coefficients b_l(phi) for a rotationally symmetric metric, l <= 3,
and the cone-point heat coefficients a_l(k) of an orbisurface cone point of order k.

Method (Schueth, arXiv:1812.06119, Sec. 3, carried one order further):
  * metric in normal coordinates at p:  g = h(r^2) dx^2 + q(r^2) (x.dx)^2,  h = (f/r)^2,
    f'' = -K f, f(0)=0, f'(0)=1, with radial curvature K = k0 + k1 r^2 + k2 r^4 (+ O(r^6));
  * Minakshisundaram-Pleijel parametrix H = (4 pi t)^{-1} e^{-sigma/2t} sum_i t^i U_i(x,y),
    sigma = dist^2/2 from the Hamilton-Jacobi equation, U_i from the transport equations
        sigma^a d_a U_i + (i + (Lap sigma - 2)/2) U_i = Lap U_{i-1},   U_0(x,x) = 1,
    everything as exact truncated power series in (x, w = y - x) with coefficients in Q[k0,k1,k2];
  * I(t) = int H(t, x, R_phi x) dA(x) = int_0^oo 2 pi f(r) H(t,(r,0),R_phi(r,0)) dr, Laplace's
    method term by term (exact Gaussian moments), giving b_l(phi) as a Laurent polynomial in
    C^2 = 2 - 2 cos(phi).
Checks (asserted): b_0, b_1 (Donnelly, as quoted by Schueth Rem. 3.2), b_2 (Schueth Thm 3.7),
the diagonal U_1, U_2 (Schueth 2.1(v), Lemma 2.2), the top C^{-8} coefficient of b_3 against the
top-coefficient theorem (top_coefficient.py), and the K^3 part of a_3(k) against Ucar's
constant-curvature polynomial (theory/cone-coefficients). Output: the t^3 cone coefficient.
"""
import sympy as sp
from sympy import Rational as Q

k0, k1, k2 = sp.symbols('k0 k1 k2')
x1, x2, w1, w2 = sp.symbols('x1 x2 w1 w2')
r, t, c, s = sp.symbols('r t c s')          # c = cos(phi), s = sin(phi)
C2 = sp.symbols('C2')                        # C^2 = 2 - 2 cos(phi)
GENS = (x1, x2, w1, w2)
NMAX = 8                                      # sigma to total degree 8 (dist^2 to r^8)


def P(e):
    return sp.Poly(sp.expand(e), *GENS, domain='QQ[k0,k1,k2]')


def trunc(p, N):
    return sp.Poly.from_dict({m: v for m, v in p.as_dict().items() if sum(m) <= N}, *GENS,
                             domain=p.domain) if p.as_dict() else p


def part(p, D):
    return {m: v for m, v in p.as_dict().items() if sum(m) == D}


# ---------- radial profile ----------
def radial_profile(order=9):
    """f(r) as odd polynomial to r^order from f'' = -K f."""
    a = {1: sp.Integer(1)}
    Kc = {0: k0, 2: k1, 4: k2}
    for n in range(3, order + 1, 2):
        # coefficient of r^{n-2} in -K f equals n(n-1) a_n
        rhs = -sum(Kc.get(j, 0) * a.get(n - 2 - j, 0) for j in range(0, n - 1))
        a[n] = sp.expand(rhs / (n * (n - 1)))
    return sum(v * r**e for e, v in a.items())


f = radial_profile(9)
# h(sigma2) where sigma2 = |y|^2 ; (f/r)^2 is even in r
fr = sp.expand(f / r)
hser = sp.expand(sp.series(fr**2, r, 0, 9).removeO())
Sq = sp.symbols('Sq')                         # Sq = r^2


def in_Sq(e):
    e = sp.expand(e)
    return sp.expand(sum(e.coeff(r, 2 * j) * Sq**j for j in range(0, 6)))


h_S = in_Sq(hser)
q_S = sp.expand(sp.cancel((1 - h_S) / Sq))                    # g = h I + q y y^T
hinv_S = in_Sq(sp.series(1 / fr**2, r, 0, 9).removeO())
p_S = sp.expand(sp.cancel((1 - hinv_S) / Sq))                  # g^{-1} = h^{-1} I + p y y^T
sqrtg_S = in_Sq(sp.series(fr, r, 0, 9).removeO())              # sqrt(det g) = f/r
isqrtg_S = in_Sq(sp.series(1 / fr, r, 0, 9).removeO())


def at_y(eS):
    y1, y2 = x1 + w1, x2 + w2
    return P(eS.subs(Sq, y1**2 + y2**2)), (y1, y2)


hinv_y, (y1, y2) = at_y(hinv_S)
p_y, _ = at_y(p_S)
sqrtg_y, _ = at_y(sqrtg_S)
isqrtg_y, _ = at_y(isqrtg_S)
Y = (P(y1), P(y2))
ginv = [[trunc(hinv_y * (1 if a == b else 0) + p_y * Y[a] * Y[b], NMAX) for b in range(2)]
        for a in range(2)]
sg_ginv = [[trunc(sqrtg_y * ginv[a][b], NMAX) for b in range(2)] for a in range(2)]
DW = (w1, w2)


def d(p, a):
    return p.diff(DW[a])


def grad_dot(A, B, N):
    """g^{ab}(y) dA/dy_a dB/dy_b, truncated."""
    dA = [d(A, a) for a in range(2)]
    dB = [d(B, b) for b in range(2)]
    out = P(0)
    for a in range(2):
        for b in range(2):
            out += trunc(ginv[a][b] * trunc(dA[a] * dB[b], N), N)
    return trunc(out, N)


def lap(A, N):
    """(1/sqrt g) d_a (sqrt g g^{ab} d_b A) in y, truncated to degree N."""
    dA = [d(A, b) for b in range(2)]
    out = P(0)
    for a in range(2):
        vec = P(0)
        for b in range(2):
            vec += trunc(sg_ginv[a][b] * dA[b], N + 1)
        out += d(vec, a)
    return trunc(isqrtg_y * trunc(out, N), N)


def wdeg(m):
    return m[2] + m[3]


# ---------- Hamilton-Jacobi for sigma ----------
sigma = P((w1**2 + w2**2) / 2)
for D in range(3, NMAX + 1):
    R = grad_dot(sigma, sigma, D) - 2 * sigma
    RD = part(trunc(R, D), D)
    new = {}
    for m, v in RD.items():
        bw = wdeg(m)
        assert bw >= 2, ("HJ consistency", m, v)
        new[m] = -v / (2 * bw - 2)
    if new:
        sigma = sigma + sp.Poly.from_dict(new, *GENS, domain=sigma.domain)
assert part(trunc(grad_dot(sigma, sigma, NMAX) - 2 * sigma, NMAX), NMAX) == {}
lap_sigma = lap(sigma, 6)
assert lap_sigma.as_dict().get((0, 0, 0, 0)) == 2


def transport(i, rhs, N):
    """Solve sigma^a d_a U + (i + (Lap sigma - 2)/2) U = rhs to degree N; U_0(x,x)=1."""
    U = P(1) if i == 0 else P(0)
    for D in range(0, N + 1):
        R = grad_dot(sigma, U, D) + trunc((i + (lap_sigma - 2) * Q(1, 2)) * U, D) - trunc(rhs, D)
        RD = part(trunc(R, D), D)
        new = {}
        for mo, v in RD.items():
            ev = wdeg(mo) + i
            if ev == 0:
                assert v == 0, ("U0 diagonal consistency", mo, v)
                continue
            new[mo] = -v / ev
        if new:
            U = U + sp.Poly.from_dict(new, *GENS, domain=U.domain)
    R = grad_dot(sigma, U, N) + trunc((i + (lap_sigma - 2) * Q(1, 2)) * U, N) - trunc(rhs, N)
    assert trunc(R, N).is_zero, ("transport residual", i)
    return U


U0 = transport(0, P(0), 6)
U1 = transport(1, lap(U0, 4), 4)
U2 = transport(2, lap(U1, 2), 2)
U3 = transport(3, lap(U2, 0), 0)
U = [U0, U1, U2, U3]

# Diagonal checks at p (x = 0, w = 0): u_1(p,p) = K/3 (Schueth (8)), u_2(p,p) = K^2/15 - Delta_g K/15
# (Schueth (7)), with Delta_g = -div grad, so Delta_g K(p) = -4 k1 for K = k0 + k1 r^2.
K_p, LapK_p = k0, -4 * k1
assert sp.expand(U1.as_expr().subs({x1: 0, x2: 0, w1: 0, w2: 0}) - K_p / 3) == 0
assert sp.expand(U2.as_expr().subs({x1: 0, x2: 0, w1: 0, w2: 0}) - (K_p**2 / 15 - LapK_p / 15)) == 0
# u_1(p, exp_p(r u)) to r^2 (Schueth Lemma 2.2, radial case: dK_p = 0, Hess K = 2 k1 g)
u1_line = sp.expand(U1.as_expr().subs({x1: 0, x2: 0, w1: r, w2: 0}))
lemma22 = K_p / 3 + (K_p**2 / 30 - LapK_p / 120 + Q(1, 20) * 2 * k1) * r**2
assert all(sp.expand((u1_line - lemma22).coeff(r, j)) == 0 for j in range(3))
# u_0(p, exp_p(r u)) to r^4 (Schueth (6), radial case)
u0_line = sp.expand(U0.as_expr().subs({x1: 0, x2: 0, w1: r, w2: 0}))
schueth6 = 1 + K_p / 12 * r**2 + (K_p**2 / 160 + Q(1, 80) * 2 * k1) * r**4
assert all(sp.expand((u0_line - schueth6).coeff(r, j)) == 0 for j in range(5))

# ---------- twisted integral ----------
sub_pt = {x1: r, x2: 0, w1: r * (c - 1), w2: r * s}


def on_orbit(p):
    e = sp.expand(p.as_expr().subs(sub_pt))
    e = sp.expand(e.subs(s**2, 1 - c**2))     # reduce sin^2
    e = sp.expand(e.subs(s**3, s * (1 - c**2)))
    return e


sig_orb = on_orbit(sigma)
sig2 = sp.expand(r**2 * (1 - c))                     # = C^2 r^2 / 2
tau = sp.expand(sig_orb - sig2)
assert sp.expand(tau.coeff(r, 2)) == 0 and tau.coeff(r, 0) == 0
assert all(sp.expand(tau.coeff(r, j)) == 0 for j in (1, 3, 5, 7))
U_orb = [on_orbit(Ui) for Ui in U]
for e in [tau] + U_orb:
    assert sp.expand(e).coeff(s) == 0, "odd sin(phi) terms must cancel by reflection symmetry"

LMAX = 3
# integrand / (Gaussian e^{-C^2 r^2/4t}) = (1/2t) f(r) exp(-tau/2t) sum_i t^i U_i
# bookkeeping variable: each r^{2j} ~ t^j; keep terms of order t^{<= LMAX} after integration
eps = sp.symbols('eps')
expo = sp.expand(-tau / (2 * t))
series_exp = 1
term = 1
for pp in range(1, LMAX + 1):
    term = sp.expand(term * expo / pp)
    series_exp += term
body = sp.expand(f * series_exp * sum(t**i * U_orb[i] for i in range(LMAX + 1)))


def gauss_r(a):
    """int_0^oo r^a e^{-C2 r^2 / 4t} dr for odd a, as expression in t, C2."""
    b = (a - 1) // 2
    return Q(1, 2) * sp.factorial(b) * (4 * t / C2)**(b + 1)


poly_r = sp.Poly(body, r)
res = 0
for (a,), coef in poly_r.terms():
    assert a % 2 == 1
    res += sp.expand(coef * gauss_r(a) / (2 * t))
res = sp.expand(res.subs(c, 1 - C2 / 2))
b = {}
for l in range(LMAX + 1):
    b[l] = sp.expand(res.coeff(t, l))
# terms with t^l, l > LMAX are incomplete (truncation) and discarded

assert sp.simplify(b[0] - 1 / C2) == 0
assert sp.simplify(b[1] - 2 * K_p / C2**2) == 0
thm37 = (12 / C2**3 - 2 / C2**2) * K_p**2 - 2 / C2**3 * LapK_p
assert sp.simplify(b[2] - thm37) == 0


# ---------- invariants: Delta_g^2 K(p) for radial K ----------
def lap_g_radial(F):
    """Delta_g F = -(F'' + (f'/f) F') for radial F(r), as a series in r."""
    fp = sp.diff(f, r)
    ratio = sp.series(fp / f, r, 0, 8).removeO()
    return sp.expand(sp.series(-(sp.diff(F, r, 2) + ratio * sp.diff(F, r)), r, 0, 6).removeO())


Kr = k0 + k1 * r**2 + k2 * r**4
L1 = lap_g_radial(Kr)
L2 = lap_g_radial(L1)
assert sp.expand(L1.subs(r, 0) - LapK_p) == 0
Lap2K_p = sp.expand(L2.subs(r, 0))
print("Delta_g^2 K(p) in radial jets:", Lap2K_p)

# express b_3 in the basis K^3, K Delta K, Delta^2 K  (|grad K|^2 = 0 at p)
KK, LK, L2K = sp.symbols('K LapK Lap2K')
A_, B_, D_ = sp.symbols('A B D')
ansatz = A_ * K_p**3 + B_ * K_p * LapK_p + D_ * Lap2K_p
eqs = sp.Poly(sp.expand(ansatz - b[3]), k0, k1, k2).coeffs()
sol = sp.solve(eqs, [A_, B_, D_], dict=True)
assert len(sol) == 1
b3 = {kk: sp.factor(sp.expand(v)) for kk, v in sol[0].items()}
print("b_3(phi) = A K^3 + B K Delta_g K + D Delta_g^2 K with")
for kk in (A_, B_, D_):
    print("  ", kk, "=", sp.expand(b3[kk]))

# top C^{-8} coefficient: beta_{3,4} = 120 K^3 + 24 K Lap K... compare with top_coefficient.py
top = {kk: sp.expand(b3[kk] * C2**4).subs(C2, 0) for kk in (A_, B_, D_)}
print("top C^-8 coefficients:", top)


# ---------- orbifold average ----------
def Pi_poly(i):
    """k * Pi_i(k) = sum_{j=1}^{k-1} (4 sin^2(pi j/k))^{-i} as a polynomial in k (exact).
    Nonzero roots of T_k(1 - S/2) - 1 are S_j = 4 sin^2(pi j/k), j=1..k-1 (with multiplicity);
    reciprocal power sums from Newton's identities; interpolate in k and verify on extra k."""
    kk = sp.symbols('kk')
    S = sp.symbols('S')

    def exact_sum(kv):
        if kv == 1:
            return sp.Integer(0)
        poly = sp.Poly(sp.expand((sp.chebyshevt(kv, 1 - S / 2) - 1) / S), S)
        # reciprocal roots are roots of the reversed polynomial
        rev = sp.Poly(list(reversed(poly.all_coeffs())), S)
        # power sums of roots of rev via Newton
        a = rev.all_coeffs()
        a = [x / a[0] for x in a]
        n = len(a) - 1
        e = [sp.Integer(1)] + [(-1)**j * a[j] for j in range(1, n + 1)]
        p = [sp.Integer(n)]
        for m in range(1, i + 1):
            val = (-1)**(m - 1) * m * (e[m] if m <= n else 0)
            for j in range(1, m):
                val += (-1)**(j - 1) * (e[j] if j <= n else 0) * p[m - j]
            p.append(sp.expand(val))
        return p[i]

    pts = list(range(1, 2 * i + 4))
    vals = [exact_sum(kv) for kv in pts]
    poly = sp.expand(sp.interpolate(list(zip(pts, vals)), kk))
    for kv in range(2 * i + 4, 2 * i + 8):
        assert poly.subs(kk, kv) == exact_sum(kv)
    return sp.expand(poly.subs(kk, sp.Symbol('k')))


k = sp.Symbol('k')
PiP = {i: Pi_poly(i) for i in range(1, 5)}
for i in PiP:
    assert sp.Poly(PiP[i], k).degree() == 2 * i
    print(f"k*Pi_{i}(k) =", sp.factor(PiP[i]))


def orbifold_average(expr):
    """(1/k) sum_j expr(C^2_j) for expr a Laurent polynomial in C2 with only negative powers."""
    e = sp.expand(expr)
    out = 0
    for i in range(0, 6):
        coef = sp.expand(e.coeff(C2, -i))
        if i == 0:
            assert coef == 0, "no C^0 term expected"
            continue
        if coef != 0:
            out += coef * PiP[i] / k
    assert sp.expand(e - sum(sp.expand(e.coeff(C2, -i)) * C2**(-i) for i in range(1, 6))) == 0
    return sp.expand(out)


a = {l: orbifold_average(b[l]) for l in range(3)}
assert sp.simplify(a[0] - (k**2 - 1) / (12 * k)) == 0
assert sp.simplify(a[1] - (Q(1, 360) * (k**3 - 1 / k) + Q(1, 36) * (k - 1 / k)) * K_p) == 0
thm41 = ((Q(1, 2520) * (k**5 - 1 / k) + Q(1, 720) * (k**3 - 1 / k) + Q(1, 180) * (k - 1 / k)) * K_p**2
         - (Q(1, 15120) * (k**5 - 1 / k) + Q(1, 1440) * (k**3 - 1 / k) + Q(1, 180) * (k - 1 / k)) * LapK_p)
assert sp.simplify(a[2] - thm41) == 0

a3 = {kk: sp.factor(orbifold_average(b3[kk])) for kk in (A_, B_, D_)}
ucar3 = (k**2 - 1) * (k**2 + 3) * (3 * k**4 + 2 * k**2 + 19) / (30240 * k)
assert sp.simplify(a3[A_] - ucar3) == 0
print("a_3(k) = A(k) K^3 + B(k) K Delta_g K + D(k) Delta_g^2 K, with")
for kk in (A_, B_, D_):
    print("  ", kk, "(k) =", a3[kk])
    print("        k*", kk, "=", sp.expand(a3[kk] * k))


def in_psi(expr):
    """write expr = sum_n c_n (k^{2n-1} - 1/k)."""
    pk = sp.Poly(sp.expand(expr * k), k)
    d = pk.as_dict()
    out = {}
    for (e,), v in d.items():
        if e > 0:
            out[(e + 1) // 2] = v
    assert sum(d.values()) == 0
    return out


for kk in (A_, B_, D_):
    print("  ", kk, "in basis (k^{2n-1}-1/k):", in_psi(a3[kk]))
print("ALL ASSERTS PASSED")

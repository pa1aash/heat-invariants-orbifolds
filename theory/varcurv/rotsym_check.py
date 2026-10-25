"""Checks for the proposition on rotationally symmetric cone points (Paper A, Section 2.4,
Proposition prop:varconerot), the case of Conjecture conj:varcone proved there.

The proof combines (A) Donnelly's form B_l(theta) = C^{-2} P(cot(theta/2)), P a polynomial, with
(B) evenness B_l(theta) = B_l(2 pi - theta) and (C) the pole bound B_l = C^{-2l-2}(F(0) + o(1)) as
theta -> 0, where F(0) is computed from v = |J_u(r)| = f(r).  This script asserts, in exact
arithmetic unless stated:
 (1) the algebra of (A)+(B)+(C): for every even P of degree 2d, C^{-2} P(cot(theta/2)) is a
     polynomial in x = C^{-2} with zero constant term and degree d+1; so the pole bound forces
     d <= l, and the top coefficient is the limit of C^{2l+2} B_l;
 (2) the limit used in (C): W = d(x, Rot_theta x)^2 / (C^2 r^2) -> f(r)^2 / r^2 as theta -> 0,
     numerically (floating point, geodesic shooting) for a rotationally symmetric metric of
     non-constant curvature;
 (3) the leading-order integral: int_0^oo e^{-C^2 v^2/4t} v^{2l+1} dv = (l!/2)(4t/C^2)^{l+1},
     so that the C^{-2l-2} coefficient is 4^l l! [v^{2l}](f^{-1})'(v);
 (4) the averaging: sum_{j=1}^{m-1} (2 sin(pi j/m))^{-2i} is (1/m)-free polynomial in m of degree
     2i vanishing at m = 1, with top coefficient |B_{2i}|/(2i)!, i <= 6, against exact values;
 (5) beta_{l,l+1}: constant-curvature value (2l)!/l! K^l and linear part 2/(l-1)! (-Delta)^{l-1}K,
     l <= 8, and the Schueth check: the m^5 coefficient of b_2 is -Delta K / 15120.
"""
import math
import sympy as sp

# (1) algebra
c, x = sp.symbols('c x')
for d in range(0, 9):
    a = sp.symbols('a0:%d' % (d + 1))
    P = sum(a[i] * c**(2 * i) for i in range(d + 1))
    # C^{-2} = (1 + cot^2)/4 and cot^2 = 4 C^{-2} - 1
    expr = sp.expand(x * P.subs(c**2, 4 * x - 1)) if d else sp.expand(x * P)
    expr = sp.expand(sp.sympify(expr).subs(c, sp.sqrt(4 * x - 1)))
    poly = sp.Poly(expr, x)
    assert poly.coeff_monomial(1) == 0
    assert poly.degree() == d + 1
    assert sp.simplify(poly.coeff_monomial(x**(d + 1)) - 4**d * a[d]) == 0
print("(1) C^-2 P(cot), P even of degree 2d <= 16: polynomial in C^-2, no constant term, degree d+1")

# (2) the limit of W, numerically: metric dr^2 + f(r)^2 dtheta^2, f = r + a r^3 + b r^5
A_, B_ = -0.3, 0.07
f = lambda s: s + A_ * s**3 + B_ * s**5
fp = lambda s: 1 + 3 * A_ * s**2 + 5 * B_ * s**4


def geodesic_end(r0, phi0, L, n=4000):
    """Unit-speed geodesic from (r0, 0) with initial angle phi0 to the radial direction, length L.
    Clairaut: f(r)^2 theta' = const; integrate (r, r', theta) with RK4."""
    h = L / n
    y = [r0, math.cos(phi0), 0.0]
    k_cl = f(r0) * math.sin(phi0)

    def rhs(y):
        r, rd, th = y
        thd = k_cl / f(r)**2
        return [rd, f(r) * fp(r) * thd**2, thd]
    for _ in range(n):
        k1 = rhs(y)
        k2 = rhs([y[i] + h / 2 * k1[i] for i in range(3)])
        k3 = rhs([y[i] + h / 2 * k2[i] for i in range(3)])
        k4 = rhs([y[i] + h * k3[i] for i in range(3)])
        y = [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)]
    return y


def dist_same_radius(r0, th):
    """d((r0,0),(r0,th)): by the reflection symmetry the minimising geodesic meets the ray th/2
    orthogonally; find the initial angle phi0 and length L with r' = 0 at theta = th/2 (Newton)."""
    phi, L = math.pi / 2 - 0.01, f(r0) * th
    for _ in range(60):
        def F(phi, L):
            y = geodesic_end(r0, phi, L / 2, 400)
            return [y[1], y[2] - th / 2]
        F0 = F(phi, L)
        e = 1e-7
        Fp = F(phi + e, L); FL = F(phi, L + e)
        J = [[(Fp[0] - F0[0]) / e, (FL[0] - F0[0]) / e], [(Fp[1] - F0[1]) / e, (FL[1] - F0[1]) / e]]
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        dphi = (-F0[0] * J[1][1] + F0[1] * J[0][1]) / det
        dL = (-F0[1] * J[0][0] + F0[0] * J[1][0]) / det
        phi += dphi; L += dL
        if abs(dphi) + abs(dL) < 1e-13:
            break
    return L


for r0 in (0.3, 0.6):
    errs = []
    for th in (0.2, 0.1, 0.05):
        C = 2 * math.sin(th / 2)
        W = dist_same_radius(r0, th)**2 / (C * r0)**2
        errs.append(abs(W - f(r0)**2 / r0**2))
    assert errs[2] < errs[1] < errs[0] and errs[2] < 1e-3, errs
    print("(2) r=%.1f: |W - f^2/r^2| = %.2e, %.2e, %.2e at theta = 0.2, 0.1, 0.05" % ((r0,) + tuple(errs)))

# (3) Gaussian moment
t, C, v = sp.symbols('t C v', positive=True)
for l in range(0, 9):
    I = sp.integrate(sp.exp(-C**2 * v**2 / (4 * t)) * v**(2 * l + 1), (v, 0, sp.oo))
    assert sp.simplify(I - sp.factorial(l) / 2 * (4 * t / C**2)**(l + 1)) == 0
    # (4 pi t)^{-1} * 2 pi * (l!/2)(4t/C^2)^{l+1}: coefficient of t^l C^{-2l-2}
    assert sp.simplify(sp.Rational(1, 2) * sp.factorial(l) * 4**(l + 1) / 2 - 4**l * sp.factorial(l)) == 0
print("(3) Gaussian moments l <= 8: t^l coefficient 4^l l! C^{-2l-2} [v^{2l}](f^{-1})'")

# (4) csc power sums (60-digit floating point; the identities are exact, the check is numerical)
import mpmath as mp
mp.mp.dps = 60
S = lambda i, k: mp.fsum((2 * mp.sin(mp.pi * j / k))**(-2 * i) for j in range(1, k))
for i in range(1, 7):
    nodes = list(range(1, 2 * i + 2))
    V = mp.matrix([[mp.mpf(k)**e for e in range(2 * i + 1)] for k in nodes])
    co = mp.lu_solve(V, mp.matrix([S(i, k) for k in nodes]))
    assert all(abs(co[e]) < mp.mpf(10)**-40 for e in range(1, 2 * i, 2))
    assert abs(sum(co) ) < mp.mpf(10)**-40                      # vanishes at m = 1
    lead = abs(sp.bernoulli(2 * i)) / sp.factorial(2 * i)
    assert abs(co[2 * i] - mp.mpf(lead.p) / lead.q) < mp.mpf(10)**-40
    for k in (2 * i + 5, 2 * i + 9):                             # out of sample
        assert abs(S(i, k) - sum(co[e] * k**e for e in range(2 * i + 1))) < mp.mpf(10)**-35
print("(4) sum_j (2 sin(pi j/m))^{-2i}: even polynomial of degree 2i, zero at m=1, top |B_2i|/(2i)!, i <= 6")

# (5) beta_{l,l+1}: constant curvature K = k0 (by homogeneity of weight 2l, k0 = 1 suffices) and the
#     linear part (K = eps r^{2l-2}, dual numbers mod eps^2); exact rational power series.
from fractions import Fraction as Fr
LM = 8


def beta_top(Kc, l, dual):
    """Kc: coefficients of Kbar in r (entries are pairs (a, b) = a + b eps if dual)."""
    N = 2 * l + 3
    Z = (Fr(0), Fr(0)) if dual else Fr(0)
    add = (lambda x, y: (x[0] + y[0], x[1] + y[1])) if dual else (lambda x, y: x + y)
    mu = (lambda x, y: (x[0] * y[0], x[0] * y[1] + x[1] * y[0])) if dual else (lambda x, y: x * y)
    sc = (lambda q, x: (q * x[0], q * x[1])) if dual else (lambda q, x: q * x)
    one = (Fr(1), Fr(0)) if dual else Fr(1)

    def smul(a, b):
        c = [Z] * (N + 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b[:N + 1 - i]):
                c[i + j] = add(c[i + j], mu(x, y))
        return c
    f = [Z] * (N + 1); f[1] = one
    for _ in range(N):                          # f = r - int int Kbar f
        g = smul(Kc, f)
        nf = [Z] * (N + 1); nf[1] = one
        for n in range(N - 1):
            nf[n + 2] = add(nf[n + 2], sc(Fr(-1, (n + 1) * (n + 2)), g[n]))
        f = nf
    # invert v = f(r): r = v + sum c_n v^n, by fixed point r = v - (f(r) - r)
    rv = [Z] * (N + 1); rv[1] = one
    for _ in range(N):
        # compose f(rv)
        comp = [Z] * (N + 1); pw = [Z] * (N + 1); pw[0] = one
        for n in range(N + 1):
            if n:
                pw = smul(pw, rv)
            comp = [add(comp[i], mu(f[n], pw[i])) for i in range(N + 1)]
        # r = v - (f(r) - r): keep the coefficient 1 of v, subtract the higher terms of f(r)
        rv = [add(sc(Fr(-1), comp[i]), rv[i]) if i != 1 else one for i in range(N + 1)]
        rv[0] = Z
    c = sc(Fr(2 * l + 1), rv[2 * l + 1])        # [v^{2l}] (f^{-1})'
    return sc(Fr(4**l * math.factorial(l)), c)


for l in range(1, LM + 1):
    N = 2 * l + 3
    Kc = [Fr(0)] * (N + 1); Kc[0] = Fr(1)
    assert beta_top(Kc, l, False) == Fr(math.factorial(2 * l), math.factorial(l))
    Kd = [(Fr(0), Fr(0))] * (N + 1); Kd[2 * l - 2] = (Fr(0), Fr(1))
    lin = beta_top(Kd, l, True)
    # linear part of (-Delta)^{l-1} K(p) for K = r^{2l-2}: 4^{l-1} ((l-1)!)^2
    assert lin == (Fr(0), Fr(2 * 4**(l - 1) * math.factorial(l - 1)**2, math.factorial(l - 1)))
print("(5) beta_{l,l+1}: constant curvature (2l)!/l! K^l and linear part 2/(l-1)! (-Delta)^{l-1} K, l <= %d" % LM)
assert abs(sp.bernoulli(6)) / sp.factorial(6) * (-2) == sp.Rational(-1, 15120)
print("    m^5 coefficient of b_2: |B_6|/6! * (-2 Delta K) = -Delta K/15120 (Schueth, Thm 4.1)")
print("ALL ASSERTS PASSED")

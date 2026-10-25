"""The part of a cone contribution that is linear in the curvature jet (Theorem VC7 in the paper).

Claim. For a rotation R by phi (C^2 = 2 - 2 cos phi) that is an isometry of a metric germ at p, the part
of Donnelly's b_l(R) that is linear in the jet of K is (2/(l-1)!) C^{-2l-2} (-Delta_g)^{l-1} K(p), l >= 1.
Averaging over R^j, j = 1..m-1, the linear part of a_l(p) is (2/(l-1)!) Pi_{l+1}(m) (-Delta_g)^{l-1} K(p),
for every order m >= 2.

Method. Write g = e^{2 eps phi}|dx|^2 (isothermal coordinates), Delta_g = e^{-2 eps phi} Delta_0 with
Delta_0 = -(d_1^2 + d_2^2). First-order Duhamel, R commuting with Delta_0:
    d/d eps tr(R^* e^{-t Delta_g}) |_{eps=0} = -t tr(R^* V e^{-t Delta_0}),  V = -2 phi Delta_0,
            = 2t int phi(x) (Delta_0 h_t)(R x - x) dx,  h_t(z) = e^{-|z|^2/4t}/(4 pi t),  |Rx - x| = C|x|.
Part 1 evaluates this exactly for every monomial phi = x^a y^b with a + b <= 2L (L = 7).
Part 2 computes the eps-linear part of (-Delta_g)^{l-1} K(0) for the same monomials, exactly.
Part 3 asserts the claim, monomial by monomial (so radial and non-radial jets alike).
Part 4 cross-checks against Donnelly's b_1, Schueth's Thm 3.7 and Thm 4.1 (arXiv:1812.06119), the t^3
coefficient VC3 (twisted_mp.py) and the attacker's b_4 (attack/ATTACK-the supplement), and the averages Pi_i
against high-precision trigonometric sums.
All arithmetic exact except the explicitly toleranced mpmath check of Pi_i (50 digits).
"""
import sympy as sp
from sympy import Rational as Q
import mpmath as mp

x, y, r, th, t, C, eps = sp.symbols('x y r theta t C epsilon', positive=True)
L = 7

# ---------- Part 1: first-order twisted trace for monomial phi ----------
# (Delta_0 h_t)(z) = h_t(z) (1/t - |z|^2/(4 t^2)), evaluated at |z| = C r.
def dtrace(a, b):
    """2t * int x^a y^b (Delta_0 h_t)(Rx - x) dx, exact, as a function of t and C."""
    ang = sp.integrate(sp.cos(th)**a * sp.sin(th)**b, (th, 0, 2*sp.pi))
    if ang == 0:
        return sp.Integer(0)
    n = a + b
    radial = sp.integrate(r**n * sp.exp(-C**2*r**2/(4*t)) / (4*sp.pi*t)
                          * (1/t - C**2*r**2/(4*t**2)) * r, (r, 0, sp.oo))
    return sp.simplify(2*t*ang*radial)

# ---------- Part 2: eps-linear part of (-Delta_g)^{l-1} K at 0 ----------
def lap0(u):
    return -(sp.diff(u, x, 2) + sp.diff(u, y, 2))

def lin_invariant(phi, l):
    """d/d eps at 0 of (-Delta_g)^{l-1} K (0) for g = e^{2 eps phi}|dx|^2."""
    # e^{-2 eps phi} truncated after eps^1: exact for the eps-linear part, since K = O(eps)
    w = 1 - 2*eps*phi
    trunc = lambda e: sp.expand(e).coeff(eps, 1)*eps
    K = trunc(w*lap0(eps*phi))               # Gauss curvature of e^{2u}|dx|^2 is e^{-2u} Delta_0 u
    u = K
    for _ in range(l - 1):
        u = trunc(-w*lap0(u))                # -Delta_g
    d = sp.diff(u, eps)
    return sp.expand(d).subs({x: 0, y: 0})

checked = 0
for l in range(1, L + 1):
    coef = Q(2, sp.factorial(l - 1))
    for n in range(1, 2*L + 1):
        for a in range(n + 1):
            b = n - a
            phi = x**a * y**b
            dt = dtrace(a, b)
            # the t^l coefficient of the first-order trace
            bl = sp.expand(dt).coeff(t, l) if dt != 0 else 0
            # nothing at half-integer or other powers beyond t^{n/2}
            if dt != 0:
                assert sp.simplify(dt - sp.expand(dt).coeff(t, sp.Rational(n, 2))*t**sp.Rational(n, 2)) == 0
            lin = lin_invariant(phi, l)
            # linear part of (-Delta)^{l-1}K(0) equals (-1)^{l-1} Delta_0^l phi (0)
            u = phi
            for _ in range(l):
                u = lap0(u)
            assert sp.simplify(lin - (-1)**(l - 1)*u.subs({x: 0, y: 0})) == 0
            assert sp.simplify(bl - coef*C**(-2*l - 2)*lin) == 0, (l, a, b, bl, lin)
            checked += 1
print(f"Part 1-3: linear part of b_l = (2/(l-1)!) C^(-2l-2) (-Delta)^(l-1) K, "
      f"checked for l <= {L} on all monomials x^a y^b, a+b <= {2*L} ({checked} cases)")

# radial monomial r^{2k}: explicit value -2 k k! 4^k t^k C^{-2k-2}
for k in range(1, L + 1):
    tot = sum(sp.binomial(k, j)*dtrace(2*j, 2*k - 2*j) for j in range(k + 1))
    assert sp.simplify(tot + 2*k*sp.factorial(k)*4**k*t**k*C**(-2*k - 2)) == 0
print("radial r^(2k): first-order trace = -2 k k! 4^k t^k C^(-2k-2), k <= %d" % L)

# ---------- Part 4: cross-checks against independent results ----------
K, DK, D2K, D3K, X = sp.symbols('K DK D2K D3K X')   # DK = Delta_g K, etc.; X = C^{-2}
def linear_in(expr, syms):
    """Part of a polynomial that is of total degree exactly 1 in syms."""
    p = sp.Poly(sp.expand(expr), *syms)
    return sum(c*sp.prod([s**e for s, e in zip(syms, m)]) for m, c in p.terms() if sum(m) == 1)

jet = (K, DK, D2K, D3K)
pred = {1: 2*K*X**2, 2: -2*DK*X**3, 3: D2K*X**4, 4: -Q(1, 3)*D3K*X**5}
known = {
    1: 2*K*X**2,                                                  # Donnelly, Schueth Rem. 3.2
    2: (12*X**3 - 2*X**2)*K**2 - 2*X**3*DK,                       # Schueth Thm 3.7
    3: (120*X**4 - 32*X**3 + Q(4, 3)*X**2)*K**3 + (-42*X**4 + 8*X**3)*K*DK + X**4*D2K,  # VC3
    4: (1680*K**4 - 904*K**2*DK + Q(98, 3)*K*D2K + Q(124, 3)*DK**2 - Q(1, 3)*D3K)*X**5
       + (-600*K**4 + 280*K**2*DK - Q(20, 3)*K*D2K - 10*DK**2)*X**4
       + (52*K**4 - Q(52, 3)*K**2*DK + Q(1, 3)*DK**2)*X**3 - Q(2, 3)*K**4*X**2,   # independent b_4
}
for l in range(1, 5):
    assert sp.expand(linear_in(known[l], jet) - pred[l]) == 0, l
    assert sp.expand(pred[l] - Q(2, sp.factorial(l - 1))*(-1)**(l - 1)*jet[l - 1]*X**(l + 1)) == 0
print("linear parts of b_1..b_4 (Donnelly, Schueth Thm 3.7, VC3, independent b_4) match")

m = sp.symbols('m', positive=True)
mPi = {1: (m**2 - 1)/12, 2: (m**2 - 1)*(m**2 + 11)/720,
       3: (m**2 - 1)*(2*m**4 + 23*m**2 + 191)/60480,
       4: (m**2 - 1)*(m**2 + 11)*(3*m**4 + 10*m**2 + 227)/3628800}
mp.mp.dps = 50
for i, f in mPi.items():
    for mm in range(2, 25):
        s = mp.fsum(mp.mpf(1)/(4*mp.sin(mp.pi*j/mm)**2)**i for j in range(1, mm))
        assert abs(s - mp.mpf(sp.Rational(f.subs(m, mm)).p)/sp.Rational(f.subs(m, mm)).q) < mp.mpf(10)**-40
# Schueth Thm 4.1 / Rem. 4.2 linear coefficients
a1_K = (m**3 - 1/m)/360 + (m - 1/m)/36
a2_DK = -((m**5 - 1/m)/15120 + (m**3 - 1/m)/1440 + (m - 1/m)/180)
assert sp.simplify(a1_K - 2*mPi[2]/m) == 0
assert sp.simplify(a2_DK + 2*mPi[3]/m) == 0
D3 = (m**2 - 1)*(m**2 + 11)*(3*m**4 + 10*m**2 + 227)/(3628800*m)   # VC3, coefficient of Delta^2 K
assert sp.simplify(D3 - mPi[4]/m) == 0
print("a_1 K-coefficient = 2 Pi_2, a_2 Delta K-coefficient = -2 Pi_3 (Schueth Thm 4.1), "
      "a_3 Delta^2 K-coefficient = Pi_4 (VC3)")
print("ALL ASSERTS PASSED")

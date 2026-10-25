"""Quadratic part of the smooth heat invariants of a compactly supported conformal bump (Lemma VC6).

For g = e^{2 lambda psi} |dx|^2 on R^2 (psi in C_c^infty), Delta_0 = -(d_1^2 + d_2^2) >= 0, Delta_g = e^{-2 lambda psi} Delta_0, and the
t^l coefficient (l >= 1) of the heat trace relative to the flat one is
    G_l(lambda psi) = lambda^2 * (1/2pi) F_{l+1} * int psi Delta_0^{l+1} psi dx + O(lambda^3),
    F_n = (-1)^n n (n-1) n! / (2n+1)!.
Derivation (second-order Duhamel, w = e^{-2 lambda psi} - 1 = -2 lambda psi + O(lambda^2)):
    second-order term = (t^2/2) int_0^1 Tr( w Delta_0 e^{-(1-a) t Delta_0} w Delta_0 e^{-a t Delta_0} ) da
                      = (1/(2 (2pi)^4)) int |w^(z)|^2 int_0^1 int |eta|^2 |eta+z|^2 e^{-t(|eta+a z|^2 + a(1-a)|z|^2)} deta da dz * t^2,
and the eta-integral is a Gaussian moment. This script redoes every step exactly:
  (a) the 2D Gaussian moment E|mu+A|^2|mu+B|^2,
  (b) F(x) = int_0^1 [2 + (1-2a)^2 x + a^2(1-a)^2 x^2] e^{-a(1-a) x} da, its Taylor coefficients F_n,
      against the closed form above (n <= 30), F_1 = 0 (no t^0 term: int K = 0 for a bump),
  (c) l = 1 against Berger's u_2(p,p) = K^2/15 - Delta K/15 (quoted in Schueth 2019, (7)):
      t^1 coefficient = (1/4pi) int K^2/15 = (1/60 pi) int (Delta_0 psi)^2 lambda^2 + O(lambda^3).
"""
import sympy as sp

a, x, tt = sp.symbols('a x t', positive=True)
m1, m2, A1, A2, B1, B2 = sp.symbols('m1 m2 A1 A2 B1 B2', real=True)

# (a) Gaussian moments with weight e^{-t|mu|^2} on R^2, normalised by pi/t
w = sp.exp(-tt * (m1**2 + m2**2))
norm = sp.integrate(w, (m1, -sp.oo, sp.oo), (m2, -sp.oo, sp.oo))
assert sp.simplify(norm - sp.pi / tt) == 0
integrand = ((m1 + A1)**2 + (m2 + A2)**2) * ((m1 + B1)**2 + (m2 + B2)**2) * w
E = sp.simplify(sp.integrate(sp.expand(integrand), (m1, -sp.oo, sp.oo), (m2, -sp.oo, sp.oo)) / norm)
claim = 2 / tt**2 + (A1**2 + A2**2 + B1**2 + B2**2) / tt + 2 * (A1 * B1 + A2 * B2) / tt \
    + (A1**2 + A2**2) * (B1**2 + B2**2)
assert sp.simplify(E - claim) == 0
# with A = -a z, B = (1-a) z  (z = (Z,0)):
Z = sp.Symbol('Z', positive=True)
Ez = sp.expand(claim.subs({A1: -a * Z, A2: 0, B1: (1 - a) * Z, B2: 0}))
assert sp.simplify(Ez - (2 / tt**2 + (1 - 2 * a)**2 * Z**2 / tt + a**2 * (1 - a)**2 * Z**4)) == 0

# (b) Taylor coefficients of F
NMAX = 30
Fx = 0
s_ = a * (1 - a)
for n in range(0, NMAX + 1):
    termn = (2 * (-s_)**n / sp.factorial(n)
             + ((1 - 2 * a)**2 * (-s_)**(n - 1) / sp.factorial(n - 1) if n >= 1 else 0)
             + (s_**2 * (-s_)**(n - 2) / sp.factorial(n - 2) if n >= 2 else 0))
    Fn = sp.integrate(sp.expand(termn), (a, 0, 1))
    if n >= 1:
        closed = (-1)**n * n * (n - 1) * sp.factorial(n) / sp.factorial(2 * n + 1)
        assert Fn == closed, (n, Fn, closed)
    if n == 1:
        assert Fn == 0
    if n >= 2:
        assert Fn != 0
print("F_2..F_6 =", [(-1)**n * n * (n - 1) * sp.factorial(n) / sp.factorial(2 * n + 1) for n in range(2, 7)])

# (c) normalisation against Berger at l = 1:
# second-order term = (pi/(2 (2pi)^4 t)) int |w^|^2 F(t|z|^2) dz, |w^|^2 = 4 lambda^2 |psi^|^2 + O(lambda^3);
# the t^l coefficient is (pi/(2(2pi)^4)) * 4 F_{l+1} int |z|^{2l+2} |psi^|^2 dz
#                     = (1/(8 pi^3)) F_{l+1} (2pi)^2 int psi Delta_0^{l+1} psi dx
#                     = (1/(2 pi)) F_{l+1} int psi Delta_0^{l+1} psi dx      (Plancherel).
pref = sp.pi / (2 * (2 * sp.pi)**4) * 4 * (2 * sp.pi)**2
assert sp.simplify(pref - 1 / (2 * sp.pi)) == 0
F2 = sp.Rational(1, 30)
# Berger: t^1 coefficient (1/4pi) int u_2 = (1/4pi) int K^2/15, K = Delta_0 psi lambda + O(lambda^2)
assert sp.simplify(pref * F2 - 1 / (60 * sp.pi)) == 0
# (d) normalisation at t^{-1}: lambda^2 part = (second-order Duhamel, F_0 = 2)
#     + (first-order Duhamel with w_2 = 2 lambda^2 psi^2: -(1/4 pi t) int w_2)
#     must equal the lambda^2 part of Area/(4 pi t) = (1/4 pi t) int (e^{2 lambda psi} - 1): 2 lambda^2 psi^2.
F0 = sp.integrate(2, (a, 0, 1))
assert F0 == 2
# per int psi^2 dx: second-order Duhamel (1/2pi) F_0 = 1/pi; first-order -(1/4pi)*2 = -1/(2pi)
assert sp.simplify(pref * F0 - 2 / (4 * sp.pi) - 2 / (4 * sp.pi)) == 0   # 1/pi - 1/(2pi) = 1/(2pi)
print("t^-1 check: second-order", pref * F0, "+ first-order", -sp.Rational(1, 2) / sp.pi,
      "= area term", sp.Rational(1, 2) / sp.pi)
print("ALL ASSERTS PASSED")

"""Top coefficient of Donnelly's b_l(phi) at an orbisurface cone point (Theorem VC2).

Claim: b_l(phi) = sum_{i=1}^{l+1} beta_{l,i} C^{-2i}  (C^2 = 2-2cos phi), and
    beta_{l,l+1} = 4^l l! [v^{2l}] (rho'(v)),  rho = f^{-1},  f = length of the Jacobi field
(radial case; in general the u-average of the same expression over unit vectors u).
Consequently the coefficient of k^{2l+1} in a_l(k) = (1/k) sum_j b_l(2 pi j/k) is
    |B_{2l+2}| / (2l+2)!  *  beta_{l,l+1}.

This script, in exact arithmetic:
 (1) computes beta_{l,l+1} for radial curvature K = k_0 + k_1 r^2 + ... + k_{l-1} r^{2l-2},
     rewrites it in K(p), Delta_g K(p), ..., Delta_g^{l-1} K(p)  (Delta_g = -div grad), l <= 8;
 (2) asserts: constant curvature value (2l)!/l! kappa^l; coefficient of Delta_g^{l-1}K equal to
     2 (-1)^{l-1}/(l-1)!; agreement with b_1 = 2K/C^4 (DGGW/Donnelly), with the C^{-6} part of
     Schueth's b_2 (Thm 3.7: 12 K^2 - 2 Delta K), and with the C^{-8} part of b_3 from
     twisted_mp.py (120 K^3 - 42 K Delta K + Delta^2 K);
 (3) asserts the k^{2l+1} coefficient identity against the constant-curvature cone polynomials
     p_l of Paper A (eq. (phik),(bl)), l <= 12, and the asymptotics of the csc power sums.
"""
import sympy as sp
from math import factorial

r, v = sp.symbols('r v')
LMAX = 7
ks = sp.symbols('k0:%d' % LMAX)


def profile(l):
    """f(r) to r^{2l+1}, f'' = -K f, K = sum_j k_j r^{2j} (j < l)."""
    a = {1: sp.Integer(1)}
    Kc = {2 * j: ks[j] for j in range(l)}
    for n in range(3, 2 * l + 2, 2):
        rhs = -sum(Kc.get(j, 0) * a.get(n - 2 - j, 0) for j in range(0, n - 1))
        a[n] = sp.expand(rhs / (n * (n - 1)))
    return sum(val * r**e for e, val in a.items())


def tr(e, var, N):
    """truncate a polynomial in var to degree <= N."""
    p = sp.Poly(sp.expand(e), var)
    return sum(cf * var**m for (m,), cf in p.terms() if m <= N)


def inverse_series(fpoly, N):
    """rho with f(rho(v)) = v + O(v^{N+1}), f odd with f'(0)=1 (fixed-point iteration,
    each step gains two orders; truncated polynomial arithmetic)."""
    coeffs = {e: fpoly.coeff(r, e) for e in range(3, N + 1, 2)}
    rho = v
    for _ in range(N // 2 + 1):
        powers, cur, comp = {1: rho}, rho, 0
        rho2 = tr(rho * rho, v, N)
        for e in range(3, N + 1, 2):
            cur = tr(cur * rho2, v, N)
            comp += coeffs[e] * cur
        rho = sp.expand(v - comp)
    # verify f(rho(v)) = v + O(v^{N+1}) by truncated composition
    rho2, cur, comp = tr(rho * rho, v, N), rho, rho
    for e in range(3, N + 1, 2):
        cur = tr(cur * rho2, v, N)
        comp += coeffs[e] * cur
    assert sp.expand(tr(comp, v, N) - v) == 0
    return rho


def lap_radial(F, fpoly, N):
    """Delta_g F = -(F'' + (f'/f) F') for radial F, truncated to r^N; f'/f = 1/r + odd series."""
    g = sp.expand(fpoly / r)                       # 1 + O(r^2), even
    # (f'/f) F' = (f'/(r g)) F' ; F' = r * (even), so (f'/f) F' = (f'/g) * (F'/r)
    ginv, cur = 1, 1
    d = sp.expand(1 - g)
    for _ in range(N // 2 + 2):
        cur = tr(cur * d, r, N + 2)
        ginv += cur
    term = tr(sp.diff(fpoly, r) * tr(ginv, r, N + 2), r, N + 2) * sp.expand(sp.diff(F, r) / r)
    return tr(-(sp.diff(F, r, 2) + term), r, N)


inv = sp.symbols('L0:%d' % LMAX)   # L_j = Delta_g^j K (p)
results = {}
for l in range(1, LMAX + 1):
    f = profile(l)
    rho = inverse_series(f, 2 * l + 1)
    c_l = sp.expand(sp.diff(rho, v)).coeff(v, 2 * l)
    beta = sp.expand(4**l * factorial(l) * c_l)
    # invariants Delta_g^j K (p), j < l, as polynomials in k_0..k_{l-1}
    Kr = sum(ks[j] * r**(2 * j) for j in range(l))
    F = Kr
    Lvals = [sp.expand(F.subs(r, 0))]
    for j in range(1, l):
        F = lap_radial(F, f, 2 * (l - j))
        Lvals.append(sp.expand(F.subs(r, 0)))
    # triangular: Delta^j K(p) = (-1)^j 4^j (j!)^2 k_j + poly(k_0..k_{j-1})
    for j in range(l):
        assert sp.expand(Lvals[j]).coeff(ks[j]) == (-1)**j * 4**j * factorial(j)**2
    # invert: express k_j through L_0..L_j
    subs = {}
    for j in range(l):
        lead = (-1)**j * 4**j * factorial(j)**2
        rest = sp.expand(Lvals[j] - lead * ks[j])
        assert rest.coeff(ks[j]) == 0
        subs[ks[j]] = sp.expand((inv[j] - rest.subs(subs)) / lead)
    beta_inv = sp.expand(beta.subs(subs))
    results[l] = beta_inv
    # (2a) constant curvature
    const = beta_inv.subs({inv[j]: 0 for j in range(1, l)})
    assert sp.expand(const - sp.Rational(factorial(2 * l), factorial(l)) * inv[0]**l) == 0
    # (2b) linear coefficient of the top jet
    lin = sp.Poly(beta_inv, *inv[:l]).as_dict()
    top_mono = tuple(1 if j == l - 1 else 0 for j in range(l))
    assert lin[top_mono] == sp.Rational(2 * (-1)**(l - 1), factorial(l - 1)), (l, lin[top_mono])
    # every other monomial has total curvature degree >= 2 (weight count: L_j has weight j+1)
    for mono, val in lin.items():
        assert sum((j + 1) * e for j, e in enumerate(mono)) == l
        if mono != top_mono:
            assert sum(mono) >= 2
    print(f"l={l}: beta_(l,l+1) = {beta_inv}")

K, LK, L2K = inv[0], inv[1], inv[2]
assert sp.expand(results[1] - 2 * K) == 0                              # DGGW / Donnelly b_1
assert sp.expand(results[2] - (12 * K**2 - 2 * LK)) == 0               # Schueth Thm 3.7, C^{-6}
assert sp.expand(results[3] - (120 * K**3 - 42 * K * LK + L2K)) == 0   # twisted_mp.py, C^{-8}

# (3) k^{2l+1} coefficient: (1/k) sum_j (4 sin^2(pi j/k))^{-(l+1)} ~ |B_{2l+2}|/(2l+2)! k^{2l+1}
k = sp.Symbol('k')
for l in range(0, 13):
    lead = abs(sp.bernoulli(2 * l + 2)) / sp.factorial(2 * l + 2)
    # constant-curvature cone polynomial p_l of Paper A, leading coefficient (Lemma conepoly)
    paperA = abs(sp.bernoulli(2 * l + 2)) / (2 * sp.factorial(l + 1) * (2 * l + 1))
    beta_const = sp.factorial(2 * l) / sp.factorial(l)
    assert sp.simplify(lead * beta_const - paperA) == 0


# Paper A's closed form for p_l (eq. (phik), (bl)), independent re-implementation, kappa = 1
def sigma_coeffs(n):
    u = sp.Symbol('u')
    ser = sp.series(u / sp.sin(u), u, 0, 2 * n + 2).removeO()
    return [ser.coeff(u, 2 * i) for i in range(n + 1)]


def p_l(l):
    sig = sigma_coeffs(l + 1)
    tot = 0
    for kk in range(l + 1):
        mphi = sp.Rational(1, 4) * sum(sig[kk + 1 - n] * 4**n * abs(sp.bernoulli(2 * n))
                                       / sp.factorial(2 * n) * (k**(2 * n) - 1) for n in range(1, kk + 2))
        tot += sp.factorial(2 * kk) / (sp.factorial(kk) * sp.factorial(l - kk)) * mphi
    return sp.expand(tot / 4**l)


for l in range(0, 9):
    pl = sp.Poly(p_l(l), k)
    assert pl.degree() == 2 * l + 2
    assert pl.LC() == abs(sp.bernoulli(2 * l + 2)) / sp.factorial(2 * l + 2) * sp.factorial(2 * l) / sp.factorial(l)
print("p_3 check:", sp.factor(p_l(3)))
assert sp.expand(p_l(3) - (k**2 - 1) * (k**2 + 3) * (3 * k**4 + 2 * k**2 + 19) / 30240) == 0
print("ALL ASSERTS PASSED")

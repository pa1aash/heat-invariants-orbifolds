"""Real solvability of the size-8, L = 3 system by shape (the paper section 6, item 3).

For shape (p, q) with p + q = 8, choose q - 3 + ... : we fix the q-element side V at random (positive reals)
and solve for the p-element side when p = 3 (closed form: U = roots of t^3 - S t^2 + R e3 t - e3,
e3 = (C - S^3)/(3(1 - S R))), counting how often U consists of three positive reals.
For shapes (2,6) and (1,7) the same elimination is impossible by Theorem 2.1 (|iota| <= T - 2L = 2);
we confirm numerically that a least-squares solver never reaches a genuine solution of those shapes.
Floating point is used only to *exhibit* real solutions; nothing proved depends on it.
Deterministic (fixed seeds).  Run time about 30 s.
"""
import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(20261005)
found = 0
trials = 100000
for _ in range(trials):
    v = np.exp(rng.normal(0, rng.uniform(0.2, 3), 5))
    S, C, R = v.sum(), (v ** 3).sum(), (1 / v).sum()
    e3 = (C - S ** 3) / (3 - 3 * S * R)
    e2 = R * e3
    r = np.roots([1, -S, e2, -e3])
    if np.all(np.abs(r.imag) < 1e-10) and np.all(r.real > 0):
        u = np.sort(r.real)
        if min(abs(a - b) / max(a, b) for a in u for b in v) > 1e-6:
            found += 1
print(f"shape (3,5): {found} of {trials} random positive V give a genuine positive real U ({100 * found / trials:.1f}%)")
assert found > trials // 10


def residual(x, p, q):
    u, v = np.exp(x[:p]), np.exp(x[p:])
    S = u.sum() + v.sum()
    return [(u.sum() - v.sum()) / S, (np.sum(u ** 3) - np.sum(v ** 3)) / S ** 3 * 10,
            (np.sum(1 / u) - np.sum(1 / v)) * S / 10, x.sum() / len(x)]


rng = np.random.default_rng(7)
for p, q in [(2, 6), (1, 7), (3, 5), (4, 4)]:
    genuine = 0
    for _ in range(300):
        sol = least_squares(residual, rng.normal(0, 1.5, p + q), args=(p, q), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.linalg.norm(residual(sol.x, p, q)[:3]) < 1e-11:
            u, v = np.exp(sol.x[:p]), np.exp(sol.x[p:])
            if min(abs(a - b) / max(a, b) for a in u for b in v) > 1e-3:
                genuine += 1
    print(f"shape {(p, q)}: genuine real solutions from 300 least-squares starts: {genuine}")
    if (p, q) in [(2, 6), (1, 7)]:
        assert genuine == 0   # Theorem 2.1 forbids iota = 4, 6 at T = 8
print("ALL CHECKS PASSED")

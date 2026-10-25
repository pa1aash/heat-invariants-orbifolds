"""Identity and elliptic terms of the Selberg trace formula for a closed orientable
hyperbolic orbisurface (Dryden-Strohmaier, arXiv:math/0504571, eq. (1)) with the
heat test function h(r) = exp(-t (1/4 + r^2)).

identity   I(t) = Area/(4 pi) int_R r tanh(pi r) h(r) dr.
           With tanh(pi r) = 1 - 2/(e^{2 pi r} + 1) on r > 0 this is
               I(t) = Area/(4 pi) [ e^{-t/4}/t - 4 int_0^inf r h(r)/(e^{2 pi r} + 1) dr ],
           a closed-form leading part plus an exponentially convergent integral.
           (A plain quadrature of r tanh(pi r) h(r) over [0, inf) is unreliable at small
           t: on some t grids it is off by ~1e-7, see the supplement.)
elliptic   E_m(t) = sum_{l=1}^{m-1} 1/(2 m sin(pi l/m)) int_R e^{-2 theta_l r}/(1 + e^{-2 pi r}) h(r) dr,
           theta_l = pi l/m; the kernel decays like e^{-2 theta r} (r -> +inf) and
           e^{-(2 pi - 2 theta)|r|} (r -> -inf).

Both are evaluated with mpmath at 30 digits (vectorised over t by a loop); the
values are cached per (m, t) because every member of a family shares them.
"""

from functools import lru_cache

import mpmath as mp
import numpy as np

DPS = 30


@lru_cache(maxsize=None)
def _identity_unit(t):
    """I(t) for Area = 4 pi, i.e. the bracket above."""
    with mp.workdps(DPS):
        t = mp.mpf(t)
        f = lambda r: r * mp.exp(-t * (mp.mpf(1) / 4 + r * r)) / (mp.exp(2 * mp.pi * r) + 1)
        J = mp.quad(f, [0, 1, 4, 12, mp.inf])
        return mp.exp(-t / 4) / t - 4 * J


@lru_cache(maxsize=None)
def _elliptic(m, t):
    with mp.workdps(DPS):
        t = mp.mpf(t)
        tot = mp.mpf(0)
        for l in range(1, m):
            th = mp.pi * l / m
            f = lambda r: mp.exp(-2 * th * r - t * (mp.mpf(1) / 4 + r * r)) / (1 + mp.exp(-2 * mp.pi * r))
            tot += mp.quad(f, [-mp.inf, -20, -5, 0, 5, 20, mp.inf]) / (2 * m * mp.sin(th))
        return tot


def identity(area, t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.array([float(area / (4 * np.pi) * _identity_unit(float(x))) for x in t])


def elliptic(m, t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.array([float(_elliptic(int(m), float(x))) for x in t])


def identity_plus_elliptic(area, orders, t):
    out = identity(area, t)
    for m in orders:
        out = out + elliptic(m, t)
    return out


def check_against_heat_coefficients(orders=(3, 3, 3, 3), area=4 * np.pi / 3, numax=3):
    """Small-t Taylor coefficients of identity + elliptic must equal the exact heat
    coefficients (numerics/theory.py: Ucar (4.33), (4.35)), which are themselves
    asserted against DGGW / Schueth there.  Checked by a Richardson-free route:
    compare at t = 1e-3 .. 4e-3 against the truncated series with the exact
    rationals; the truncation error is O(t^(numax+1))."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import theory as th  # S3 module, unmodified
    from fractions import Fraction as Fr
    assert th.check_against_paper()
    coef = {}
    coef[-1] = Fr(1, 1) * 0  # filled below with area/(4 pi) as a float
    out = []
    for t in (1e-3, 2e-3, 4e-3):
        series = area / (4 * np.pi * t)
        for nu in range(0, numax + 1):
            sm = float(th.a_smooth_over_vol(nu + 1)) * area / (4 * np.pi)
            cone = sum(float(th.b_cone(nu, m)) for m in orders)
            series += (sm + cone) * t ** nu
        exact = identity_plus_elliptic(area, orders, t)[0]
        out.append((t, exact, series, exact - series))
    return out


if __name__ == "__main__":
    for t, e, s, d in check_against_heat_coefficients():
        print(f"t={t:.0e}  identity+elliptic={e:.15f}  heat series (nu<=3)={s:.15f}  diff={d:.2e}")

"""Equiangular hyperbolic quadrilaterals with angles pi/m and their Lambert quarters.

The family.  Q(tau) is the geodesic quadrilateral in the Poincare disk that is
symmetric under z -> conj(z) and z -> -conj(z), with all four angles pi/m
(m = 3 here, signature (0; 3,3,3,3) for the double).  The two mirror axes cut Q
into four congruent Lambert quadrilaterals L with right angles at O = 0,
P = tanh(a/2) and Y = i tanh(b/2), and angle pi/m at the vertex V.  Here a is the
distance from O to the east side and b the distance from O to the north side.
The Lambert relation

        sinh(a) sinh(b) = cos(pi/m)

leaves one free parameter, the modulus

        tau = ln( sinh(a) / sinh(b) ),   sinh(a) = sqrt(c) e^{tau/2},  sinh(b) = sqrt(c) e^{-tau/2},

with c = cos(pi/m).  tau = 0 is the square; tau and -tau give congruent
quadrilaterals (rotation by pi/2), so tau >= 0 parametrises the family.

Construction (explicit, no root finding).  The east side is the geodesic
orthogonal to the real axis at P: the circle orthogonal to |z| = 1 with centre
coth(a) and radius 1/sinh(a).  The north side is the circle with centre i coth(b)
and radius 1/sinh(b).  V is their intersection in the first quadrant.

The Lambert relation is NOT used to check anything: verify() recomputes the angle
at V from the tangent vectors of the two constructed circles, the right angles
at P and Y, the side lengths by integrating 2|dz|/(1-|z|^2), and the area of L by
quadrature against Gauss-Bonnet, area(L) = pi/2 - pi/m.  The double of Q is the
orbifold O(tau) of area 2 area(Q) = 8 area(L) = 4 pi (1/2 - 1/m) * 2.

Sides of L (netgen boundary names):
    'x'  O -> P   on the real axis       (mirror of Q)
    'e'  P -> V   arc of the east circle  (side of Q)
    'n'  V -> Y   arc of the north circle (side of Q)
    'y'  Y -> O   on the imaginary axis  (mirror of Q)
"""

import mpmath as mp

mp.mp.dps = 40

M_ORDER = 3


class Lambert:
    def __init__(self, tau, m=M_ORDER):
        self.tau = mp.mpf(tau)
        self.m = m
        c = mp.cos(mp.pi / m)
        self.a = mp.asinh(mp.sqrt(c) * mp.exp(self.tau / 2))
        self.b = mp.asinh(mp.sqrt(c) * mp.exp(-self.tau / 2))
        a, b = self.a, self.b
        self.P = mp.mpc(mp.tanh(a / 2), 0)
        self.Y = mp.mpc(0, mp.tanh(b / 2))
        self.cE, self.rE = mp.mpc(mp.coth(a), 0), 1 / mp.sinh(a)
        self.cN, self.rN = mp.mpc(0, mp.coth(b)), 1 / mp.sinh(b)
        # V: |z|^2 + 1 = 2 x coth(a) = 2 y coth(b)  ->  y = x coth(a)/coth(b),
        # x^2 (1 + k^2) - 2 x coth(a) + 1 = 0, smaller root (inside the disk)
        k = mp.coth(a) / mp.coth(b)
        A2, B1 = 1 + k * k, -2 * mp.coth(a)
        x = (-B1 - mp.sqrt(B1 * B1 - 4 * A2)) / (2 * A2)
        self.V = mp.mpc(x, k * x)
        self.area_exact = mp.pi / 2 - mp.pi / m          # Gauss-Bonnet for L

    # ------------------------------------------------------------- lengths
    @staticmethod
    def _arc_length(c, r, z1, z2):
        t1, t2 = mp.arg(z1 - c), mp.arg(z2 - c)
        if abs(t2 - t1) > mp.pi:
            t2 += 2 * mp.pi * (1 if t2 < t1 else -1)
        f = lambda s: 2 * r / (1 - abs(c + r * mp.expj(s)) ** 2)
        return abs(mp.quad(f, [t1, t2]))

    def side_lengths(self):
        """Hyperbolic lengths of x, e, n, y by integration of the metric."""
        return dict(x=2 * mp.atanh(abs(self.P)), e=self._arc_length(self.cE, self.rE, self.P, self.V),
                    n=self._arc_length(self.cN, self.rN, self.V, self.Y), y=2 * mp.atanh(abs(self.Y)))

    # ------------------------------------------------------------- angles
    @staticmethod
    def _circle_tangent(c, z, toward):
        t = 1j * (z - c)
        if (t.real * (toward - z).real + t.imag * (toward - z).imag) < 0:
            t = -t
        return t

    def angles(self):
        """Interior angles at O, P, V, Y from the constructed boundary (Euclidean
        angles between tangent vectors = hyperbolic angles, the metric being
        conformal).  Tangent of an arc at an endpoint is oriented toward the other
        endpoint of that side; for these short arcs (less than a half circle) the
        chord direction fixes the orientation."""
        O, P, V, Y = mp.mpc(0), self.P, self.V, self.Y
        ang = lambda u, v: abs(mp.arg(v / u))
        aO = ang(P - O, Y - O)
        aP = ang(O - P, self._circle_tangent(self.cE, P, V))
        aV = ang(self._circle_tangent(self.cE, V, P), self._circle_tangent(self.cN, V, Y))
        aY = ang(self._circle_tangent(self.cN, Y, V), O - Y)
        return dict(O=aO, P=aP, V=aV, Y=aY)

    # ------------------------------------------------------------- area
    def area_quadrature(self):
        """Polar quadrature about O: for direction phi the ray leaves L through the
        east arc (phi < arg V) or the north arc; int_0^R 4s/(1-s^2)^2 ds = 2R^2/(1-R^2)."""
        def R(phi, c, r):
            pr = (mp.expj(-phi) * c).real
            return pr - mp.sqrt(pr ** 2 - (abs(c) ** 2 - r ** 2))
        f = lambda phi, c, r: 2 * R(phi, c, r) ** 2 / (1 - R(phi, c, r) ** 2)
        phV = mp.arg(self.V)
        return (mp.quad(lambda p: f(p, self.cE, self.rE), [0, phV])
                + mp.quad(lambda p: f(p, self.cN, self.rN), [phV, mp.pi / 2]))

    def verify(self, tol=mp.mpf(10) ** -12):
        out = {}
        # the constructed circles: orthogonal to |z| = 1, through their side's endpoints
        for c, r, pts in ((self.cE, self.rE, (self.P, self.V)), (self.cN, self.rN, (self.V, self.Y))):
            assert abs(abs(c) ** 2 - 1 - r ** 2) < tol
            for z in pts:
                assert abs(abs(z - c) - r) < tol, (self.tau, z)
        an = self.angles()
        for k, e in (("O", mp.pi / 2), ("P", mp.pi / 2), ("Y", mp.pi / 2), ("V", mp.pi / self.m)):
            assert abs(an[k] - e) < tol, (self.tau, k, an[k], e)
        A = self.area_quadrature()
        assert abs(A - self.area_exact) < tol, (self.tau, A, self.area_exact)
        s = self.side_lengths()
        assert abs(s["x"] - self.a) < tol and abs(s["y"] - self.b) < tol
        out.update(angles=an, area_quadrature=A, sides=s)
        return out

    # ------------------------------------------------------------- the double
    def quad_vertices(self):
        """The four vertices of Q (cone points of the double), counter-clockwise."""
        V = self.V
        return [V, -V.conjugate(), -V, V.conjugate()]

    def quad_sides(self):
        """The four sides of Q as (centre, radius) of circles orthogonal to |z| = 1:
        east, north, west, south."""
        return [(self.cE, self.rE), (self.cN, self.rN), (-self.cE, self.rE), (-self.cN, self.rN)]

    def summary(self):
        s = self.side_lengths()
        return dict(tau=float(self.tau), a=float(self.a), b=float(self.b),
                    V=(float(self.V.real), float(self.V.imag)),
                    side_east_Q=float(2 * s["e"]), side_north_Q=float(2 * s["n"]),
                    width_Q=float(2 * self.a), height_Q=float(2 * self.b),
                    dist_O_V=float(2 * mp.atanh(abs(self.V))),
                    area_L=float(self.area_exact), area_Q=float(4 * self.area_exact),
                    area_orbifold=float(8 * self.area_exact),
                    perimeter_Q=float(4 * (s["e"] + s["n"])))

    # ------------------------------------------------------------- floats for netgen
    def arc_control(self, c, z1, z2):
        """Intersection of the tangents at z1, z2 of the circle with centre c: the
        middle control point that makes netgen's rational quadratic spline the
        exact circular arc."""
        t1, t2 = 1j * (z1 - c), 1j * (z2 - c)
        Mx = mp.matrix([[t1.real, -t2.real], [t1.imag, -t2.imag]])
        rhs = mp.matrix([(z2 - z1).real, (z2 - z1).imag])
        s = mp.lu_solve(Mx, rhs)[0]
        p = z1 + s * t1
        return (float(p.real), float(p.imag))


# The family used for the experiment.  tau = 0 is the square; the north/south
# distance 2b shrinks as tau grows, so the systole of the double falls from
# about 2.6 to about 0.7 across the family.
TAUS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]


if __name__ == "__main__":
    import json
    for tau in TAUS:
        L = Lambert(tau)
        v = L.verify()
        an = v["angles"]
        print(f"tau={tau:.1f}  a={float(L.a):.15f}  b={float(L.b):.15f}  "
              f"|angle_V - pi/3|={mp.nstr(abs(an['V'] - mp.pi / 3), 3)}  "
              f"|area quad - exact|={mp.nstr(abs(v['area_quadrature'] - L.area_exact), 3)}  "
              f"sinh(a)sinh(b)={mp.nstr(mp.sinh(L.a) * mp.sinh(L.b), 20)}")
    print(json.dumps(Lambert(0.8).summary(), indent=1))

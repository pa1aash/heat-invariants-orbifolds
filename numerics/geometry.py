"""Hyperbolic triangle with angles pi/p, pi/q, pi/r in the Poincare disk.

Placement: vertex A (angle pi/p) at the origin, vertex B (angle pi/q) on the
positive real axis, vertex C (angle pi/r) on the ray arg z = pi/p.  Sides AB and
AC are radial segments; side BC is an arc of the circle orthogonal to the unit
circle through B and C.

Side lengths come from the hyperbolic law of cosines for angles,
    cosh(c) = (cos(gamma) + cos(alpha) cos(beta)) / (sin(alpha) sin(beta)),
with c the side opposite gamma.  The formula is not taken on trust: verify()
recomputes all three interior angles from the constructed circle and compares
them with pi/p, pi/q, pi/r, and integrates the area against Gauss-Bonnet.

The Laplace-Beltrami eigenproblem on the triangle is
    -Delta_E u = lambda * w(z) u,    w(z) = 4 / (1 - |z|^2)^2,
since the disk metric is w(z) |dz|^2 and the Laplacian is conformally covariant
in two dimensions.
"""

import mpmath as mp

mp.mp.dps = 40


class Triangle:
    def __init__(self, p, q, r):
        self.pqr = (p, q, r)
        al, be, ga = mp.pi / p, mp.pi / q, mp.pi / r
        self.angles = (al, be, ga)
        # side c = |AB| (opposite gamma), side b = |AC| (opposite beta)
        cosh_c = (mp.cos(ga) + mp.cos(al) * mp.cos(be)) / (mp.sin(al) * mp.sin(be))
        cosh_b = (mp.cos(be) + mp.cos(al) * mp.cos(ga)) / (mp.sin(al) * mp.sin(ga))
        self.c = mp.acosh(cosh_c)
        self.b = mp.acosh(cosh_b)
        # a point at hyperbolic distance d from 0 sits at Euclidean radius tanh(d/2)
        self.A = mp.mpc(0, 0)
        self.B = mp.mpc(mp.tanh(self.c / 2), 0)
        self.C = mp.tanh(self.b / 2) * mp.expjpi(1 / mp.mpf(p))
        # circle orthogonal to |z|=1 through B and C: centre z0, radius rho,
        # |z0|^2 = 1 + rho^2.  |z - z0|^2 = rho^2 gives |z|^2 + 1 = 2 Re(z conj z0),
        # a 2x2 linear system for (Re z0, Im z0).
        Bx, By = self.B.real, self.B.imag
        Cx, Cy = self.C.real, self.C.imag
        M = mp.matrix([[2 * Bx, 2 * By], [2 * Cx, 2 * Cy]])
        rhs = mp.matrix([abs(self.B) ** 2 + 1, abs(self.C) ** 2 + 1])
        sol = mp.lu_solve(M, rhs)
        self.z0 = mp.mpc(sol[0], sol[1])
        self.rho = mp.sqrt(abs(self.z0) ** 2 - 1)
        self.area_exact = mp.pi - al - be - ga
        # side a = |BC| (opposite alpha), same law of cosines
        cosh_a = (mp.cos(al) + mp.cos(be) * mp.cos(ga)) / (mp.sin(be) * mp.sin(ga))
        self.a = mp.acosh(cosh_a)
        self.perimeter = self.a + self.b + self.c

    # --- floats for the mesher -------------------------------------------
    def vertices(self):
        return [(float(z.real), float(z.imag)) for z in (self.A, self.B, self.C)]

    def arc_control(self):
        """Intersection of the arc's tangents at B and C (rational-quadratic
        control point, which makes the netgen spline an exact circular arc)."""
        B, C, z0 = self.B, self.C, self.z0
        tB = 1j * (B - z0)
        tC = 1j * (C - z0)
        # B + s tB = C + u tC
        M = mp.matrix([[tB.real, -tC.real], [tB.imag, -tC.imag]])
        rhs = mp.matrix([(C - B).real, (C - B).imag])
        s = mp.lu_solve(M, rhs)[0]
        P = B + s * tB
        return (float(P.real), float(P.imag))

    # --- independent checks ----------------------------------------------
    def interior_angles(self):
        """Angles recomputed from the constructed boundary (not from the law of
        cosines): Euclidean angles between the tangent vectors at each vertex,
        which equal hyperbolic angles because the metric is conformal."""
        A, B, C, z0 = self.A, self.B, self.C, self.z0

        def ang(u, v):
            return abs(mp.arg(v / u))

        # tangent of arc at B pointing toward C: perpendicular to (B - z0),
        # oriented so that it points to the side of C
        def arc_tangent(P, Q):
            t = 1j * (P - z0)
            # the arc from P to Q bends toward the origin; choose orientation by
            # the sign of the projection of (Q - P)
            if (t.real * (Q - P).real + t.imag * (Q - P).imag) < 0:
                t = -t
            return t

        aA = ang(B - A, C - A)
        aB = ang(A - B, arc_tangent(B, C))
        aC = ang(A - C, arc_tangent(C, B))
        return aA, aB, aC

    def area_quadrature(self):
        """Hyperbolic area by 2D quadrature of w over the triangle in polar
        coordinates around the origin: for each direction phi in [0, pi/p] the
        ray from 0 hits the arc at radius R(phi), and
            int_0^R 4 s / (1 - s^2)^2 ds = 2 R^2 / (1 - R^2)."""
        z0, rho = self.z0, self.rho

        def R(phi):
            # |s e^{i phi} - z0|^2 = rho^2  ->  s^2 - 2 s Re(e^{-i phi} z0) + |z0|^2 - rho^2 = 0
            pr = (mp.expj(-phi) * z0).real
            return pr - mp.sqrt(pr ** 2 - 1)   # smaller root, |z0|^2 - rho^2 = 1

        f = lambda phi: 2 * R(phi) ** 2 / (1 - R(phi) ** 2)
        return mp.quad(f, [0, self.angles[0]])

    def verify(self, tol=1e-12):
        angs = self.interior_angles()
        for a, e in zip(angs, self.angles):
            assert abs(a - e) < tol, (self.pqr, a, e)
        # B and C lie on the circle, and the circle is orthogonal to |z| = 1
        assert abs(abs(self.B - self.z0) - self.rho) < tol
        assert abs(abs(self.C - self.z0) - self.rho) < tol
        assert abs(abs(self.z0) ** 2 - 1 - self.rho ** 2) < tol
        A_num = self.area_quadrature()
        assert abs(A_num - self.area_exact) < tol, (A_num, self.area_exact)
        # the arc BC has hyperbolic length a: integrate 2|dz|/(1-|z|^2) along it
        # (independent of the law-of-cosines value of a)
        tB = mp.arg(self.B - self.z0)
        tC = mp.arg(self.C - self.z0)
        if abs(tC - tB) > mp.pi:
            tC += 2 * mp.pi * (1 if tC < tB else -1)
        zarc = lambda s: self.z0 + self.rho * mp.expj(s)
        a_num = mp.quad(lambda s: 2 * self.rho / (1 - abs(zarc(s)) ** 2), [tB, tC])
        assert abs(abs(a_num) - self.a) < tol, (a_num, self.a)
        # radial sides: d = 2 artanh(|z|)
        assert abs(2 * mp.atanh(abs(self.B)) - self.c) < tol
        assert abs(2 * mp.atanh(abs(self.C)) - self.b) < tol
        return {"angles": [float(a) for a in angs], "area_quadrature": A_num,
                "area_exact": self.area_exact}


def make_geometry(tri, maxh, grade=True):
    """netgen 2D spline geometry of the triangle.  BC names: 'ab' (radial, on
    the real axis), 'ac' (radial), 'bc' (geodesic arc)."""
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    (ax, ay), (bx, by), (cx, cy) = tri.vertices()
    px, py = tri.arc_control()
    pA = geo.AppendPoint(ax, ay)
    pB = geo.AppendPoint(bx, by)
    pP = geo.AppendPoint(px, py)
    pC = geo.AppendPoint(cx, cy)
    geo.Append(["line", pA, pB], bc="ab", leftdomain=1, rightdomain=0)
    geo.Append(["spline3", pB, pP, pC], bc="bc", leftdomain=1, rightdomain=0)
    geo.Append(["line", pC, pA], bc="ac", leftdomain=1, rightdomain=0)
    return geo


def make_mesh(tri, h_hyp, order_geom, grade=True):
    """Mesh with local Euclidean size ~ h_hyp * (1 - |z|^2) / 2, i.e. a uniform
    hyperbolic mesh size h_hyp.  Curved to polynomial order order_geom."""
    import numpy as np
    import ngsolve as ngs
    from netgen.meshing import MeshingParameters
    geo = make_geometry(tri, None)
    (ax, ay), (bx, by), (cx, cy) = tri.vertices()
    hmin = h_hyp * (1 - max(bx * bx + by * by, cx * cx + cy * cy)) / 2
    mp_ = MeshingParameters(maxh=h_hyp / 2, grading=0.2)
    if grade:
        # sample the triangle and restrict the size at each sample point
        n = 60
        for i in range(n + 1):
            for j in range(n + 1 - i):
                l1, l2 = i / n, j / n
                x = ax + l1 * (bx - ax) + l2 * (cx - ax)
                y = ay + l1 * (by - ay) + l2 * (cy - ay)
                hl = h_hyp * (1 - x * x - y * y) / 2
                mp_.RestrictH(x=x, y=y, z=0, h=hl)
        # the arc bulges outside the chord triangle; restrict along it too
        z0 = complex(float(tri.z0.real), float(tri.z0.imag))
        rho = float(tri.rho)
        Bz, Cz = complex(bx, by), complex(cx, cy)
        tb, tc = np.angle(Bz - z0), np.angle(Cz - z0)
        if abs(tc - tb) > np.pi:
            tc += 2 * np.pi * (1 if tc < tb else -1)
        for s in np.linspace(tb, tc, 200):
            z = z0 + rho * np.exp(1j * s)
            mp_.RestrictH(x=z.real, y=z.imag, z=0, h=h_hyp * (1 - abs(z) ** 2) / 2)
    ngmesh = geo.GenerateMesh(mp=mp_)
    mesh = ngs.Mesh(ngmesh)
    mesh.Curve(order_geom)
    return mesh


def weight_cf():
    import ngsolve as ngs
    return 4 / (1 - ngs.x * ngs.x - ngs.y * ngs.y) ** 2


PILLOWS = [(2, 8, 8), (3, 3, 12)]
BENCH = (2, 3, 8)


if __name__ == "__main__":
    for pqr in PILLOWS + [BENCH]:
        T = Triangle(*pqr)
        v = T.verify()
        print(pqr, "vertices", T.vertices(), "z0", complex(T.z0), "rho", float(T.rho))
        print("   angles", v["angles"], " area quad - exact =",
              mp.nstr(v["area_quadrature"] - v["area_exact"], 5),
              " exact area", mp.nstr(v["area_exact"], 20))

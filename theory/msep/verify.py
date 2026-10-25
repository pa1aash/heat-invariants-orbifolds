"""Exact verification for the paper (bounded cone orders: M heat invariants suffice).

Statement checked (the paper, Theorem msep): for M >= 2, (i) two closed orientable hyperbolic 2-orbifolds
whose cone orders are all at most M and which share c_1, ..., c_M have the same signature; (ii) M is
optimal there: an explicit pair with orders at most M and different signatures shares c_1, ..., c_{M-1};
(iii) against all orbifolds, K_mult(O; Sig) <= min(M + 1, 2 d_O + 2) for O with orders <= M and d_O
distinct orders, at every area (sign changes of the measure sum +-delta_{x^2}/x); (iv) M + 1 is attained.

Checks, all in exact rational arithmetic (fractions.Fraction; sympy only for an independent rank):
  1. rank of A_L(M) = (psi_k(a))_{1<=k<=L-1, 2<=a<=M}, psi_k(x) = x^(2k-1) - 1/x, equals
     min(L-1, M-1) for 1 <= L <= M+1 and 2 <= M <= MMAX, by our own Fraction elimination and by sympy;
  2. the same for the padded system of the brief, B_L(M) = (x^j)_{j in {-1,1,3,..,2L-3}, 1<=x<=M},
     rank min(L, M), so both formulations give the threshold L = M;
  3. det of the (M-1)x(M-1) matrix (a^2k - 1)_{1<=k<=M-1, 2<=a<=M} equals the Vandermonde determinant
     of (1, 2^2, ..., M^2) (the identity used in the proof);
  4. the kernel vector w_a = 1/prod_{b != a, 1<=b<=M} (a^2 - b^2) (Lagrange weights) kills the rows
     k = 0..M-2 of the Vandermonde matrix, and nu(a) = a w_a kills A_{M-1}(M);
  5. for each M the explicit sharp pair of the paper (integer scaling C, genera g, g'): orders in
     [2, M], hyperbolic, different signatures, c_1..c_{M-1} equal and c_M different, recomputed from
     the cone coefficients b_l of Proposition 2.7 (theory/eigen/eigen_common.py), and the
     difference c_M - c_M' = (-1)^M C a_{M-2} with a_l the leading coefficient of p_l;
  6. independent brute force for M = 2, 3, 4: every signature with orders <= M and area at most a
     bound; no two distinct signatures share c_1..c_M, and the largest number of shared coefficients
     among distinct signatures is M - 1 once the bound reaches the sharp pair's area (and the minimal
     area of such a pair is reported).
  7. parts (iii)-(iv) on complete equal-area classes with no bound on the orders: for every pair sharing
     c_1..c_L the measure of the proof has at least L sign changes, L <= M and L <= 2 min(d_O, d_O') + 1;
     the exact K_mult(O; Sig) <= min(M + 1, 2 d_O + 2) for every O;
  8. part (iv): for 1 <= M <= 12 and M < X <= M + 4 the pair built on the nodes 1^2..M^2, X^2 has O with
     orders <= M sharing exactly c_1..c_M with O', Delta c_(M+1) = (-1)^M C a_(M-1); X = M + 1 is part (ii)
     at M + 1; the example (0;2^10), (1;4^4) of area 6 pi with K_mult = 3, from its complete area class.
Run: python3 verify.py [MMAX]   (default MMAX = 24; a few minutes). Exit status 0 iff all checks pass.
"""
import itertools
import os
import sys
from fractions import Fraction as Fr
from math import gcd, prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eigen"))
from eigen_common import alpha, b_cone, check, lead_coef  # noqa: E402

MMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 24
check(MMAX >= 18, "the brief asks for M up to at least 18")


def rank_fraction(rows):
    """Rank over Q by Gaussian elimination on Fractions."""
    A = [list(map(Fr, r)) for r in rows]
    if not A:
        return 0
    n, m = len(A), len(A[0])
    r = 0
    for c in range(m):
        piv = next((i for i in range(r, n) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        for i in range(n):
            if i != r and A[i][c] != 0:
                f = A[i][c] / A[r][c]
                A[i] = [A[i][j] - f * A[r][j] for j in range(m)]
        r += 1
        if r == n:
            break
    return r


def det_fraction(rows):
    A = [list(map(Fr, r)) for r in rows]
    n = len(A)
    d = Fr(1)
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c] != 0), None)
        if piv is None:
            return Fr(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            d = -d
        d *= A[c][c]
        for i in range(c + 1, n):
            f = A[i][c] / A[c][c]
            A[i] = [A[i][j] - f * A[c][j] for j in range(n)]
    return d


def psi(k, x):
    x = Fr(x)
    return x ** (2 * k - 1) - 1 / x


def A_mat(L, M):
    return [[psi(k, a) for a in range(2, M + 1)] for k in range(1, L)]


def B_mat(L, M):
    js = [-1] + list(range(1, 2 * L - 2, 2))
    return [[Fr(x) ** j for x in range(1, M + 1)] for j in js]


# ---------------------------------------------------------------- 1-3. ranks and the determinant
import sympy  # noqa: E402

n_rank = 0
for M in range(2, MMAX + 1):
    for L in range(1, M + 2):
        rA = rank_fraction(A_mat(L, M))
        check(rA == min(L - 1, M - 1), f"rank A_{L}({M}) = {rA}")
        rB = rank_fraction(B_mat(L, M))
        check(rB == min(L, M), f"rank B_{L}({M}) = {rB}")
        n_rank += 2
    # sympy, independently, at the two lengths that matter: L = M (full rank) and L = M - 1
    for L in (M - 1, M):
        if L >= 2:
            S = sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in A_mat(L, M)])
            check(S.rank() == min(L - 1, M - 1), f"sympy rank A_{L}({M})")
    V = [[Fr(a * a) ** k - 1 for a in range(2, M + 1)] for k in range(1, M)]
    vdm = prod(Fr(y2 - y1) for y1, y2 in itertools.combinations([a * a for a in range(1, M + 1)], 2))
    check(det_fraction(V) == vdm, f"det identity at M = {M}")
print(f"1-3. ranks: {n_rank} exact ranks (A_L(M): min(L-1, M-1); B_L(M): min(L, M)) for 2 <= M <= {MMAX}, "
      f"1 <= L <= M+1; sympy agrees at L = M-1, M; det(a^2k - 1) = Vandermonde(1, 4, ..., M^2) for every M")


# ---------------------------------------------------------------- 4-5. kernel and the sharp pairs
def lagrange_w(M):
    return {a: Fr(1, prod(a * a - b * b for b in range(1, M + 1) if b != a)) for a in range(1, M + 1)}


def heat_mult(g, mult, L):
    """(c_1..c_L) for genus g and cone orders with multiplicities mult = {order: count}."""
    s = Fr(2 * g - 2) + sum(c * (1 - Fr(1, a)) for a, c in mult.items())
    c1 = s / 2
    out = [c1]
    for j in range(2, L + 1):
        out.append(alpha(j - 1) * c1 + sum(c * b_cone(j - 2, a) for a, c in mult.items()))
    return out, s


def sharp_pair(M):
    """the paper, part (ii): nu(a) = C a w_a with the least positive integer C making nu integral
    and s = sum nu(a)(1 - 1/a) an even integer; genera g = 2 + max(0, -s/2), g' = g + s/2."""
    w = lagrange_w(M)
    nu0 = {a: a * w[a] for a in range(2, M + 1)}
    den = 1
    for v in nu0.values():
        den = den * v.denominator // gcd(den, v.denominator)
    C = den
    nu = {a: int(C * v) for a, v in nu0.items()}
    s = sum(Fr(v) * (1 - Fr(1, a)) for a, v in nu.items())
    # least multiple k with s*k an even integer
    k = 1
    while not ((s * k).denominator == 1 and (s * k).numerator % 2 == 0):
        k += 1
    C *= k
    nu = {a: v * k for a, v in nu.items()}
    s = s * k
    m = {a: v for a, v in nu.items() if v > 0}
    mp_ = {a: -v for a, v in nu.items() if v < 0}
    # least genera with g' - g = s/2 for which the (common) area is positive
    g = max(0, -int(s) // 2)
    while Fr(2 * g - 2) + sum(c * (1 - Fr(1, a)) for a, c in m.items()) <= 0:
        g += 1
    gp = g + int(s) // 2
    return C, nu, int(s), (g, m), (gp, mp_)


rows = []
for M in range(2, MMAX + 1):
    w = lagrange_w(M)
    for kk in range(0, M - 1):
        check(sum(w[a] * Fr(a * a) ** kk for a in range(1, M + 1)) == 0, f"Lagrange weights, M={M}, k={kk}")
    check(all(w[a] != 0 for a in w), "weights nonzero")
    check(all((w[a] > 0) == ((M - a) % 2 == 0) for a in w), "sign of w_a is (-1)^(M-a)")
    nu0 = [a * w[a] for a in range(2, M + 1)]
    for row in A_mat(M - 1, M):
        check(sum(r * v for r, v in zip(row, nu0)) == 0, f"kernel of A_(M-1)({M})")
    C, nu, s, (g, m), (gp, mp_) = sharp_pair(M)
    check(all(2 <= a <= M for a in list(m) + list(mp_)), "orders in [2, M]")
    check(m != mp_ or g != gp, "different signatures")
    H1, s1 = heat_mult(g, m, M)
    H2, s2 = heat_mult(gp, mp_, M)
    check(s1 == s2 and s1 > 0, f"equal positive area at M = {M}: both hyperbolic")
    check(H1[:M - 1] == H2[:M - 1], f"c_1..c_(M-1) agree at M = {M}")
    check(H1[M - 1] != H2[M - 1], f"c_M differs at M = {M}")
    check(H1[M - 1] - H2[M - 1] == (-1) ** M * C * lead_coef(M - 2), f"Delta c_M = (-1)^M C a_(M-2) at M = {M}")
    rows.append((M, C, s, g, gp, sum(m.values()), sum(mp_.values()), s1))
print("4-5. Lagrange kernel and sharp pairs (orders <= M, different signatures, equal c_1..c_(M-1), c_M differs):")
print("   M  genera  cone points    Area/2pi        nu on {2..M}")
for (M, C, s, g, gp, n1, n2, s1) in rows[:6]:
    _, nu, _, _, _ = sharp_pair(M)
    print(f"  {M:2d}  ({g},{gp})   {n1:>6} vs {n2:<6} {float(s1):12.6g}   {[nu[a] for a in range(2, M + 1)]}")
for (M, C, s, g, gp, n1, n2, s1) in rows[6:]:
    print(f"  {M:2d}  ({g},{gp})   {n1:.3g} vs {n2:.3g} cone points, Area/2pi = {float(s1):.4g}")


# ---------------------------------------------------------------- 6. brute force, small M
def all_sigs(M, smax):
    """Every hyperbolic signature (g; multiset in [2, M]) with Area/2pi = s <= smax (Fractions)."""
    out = []
    g = 0
    while 2 * g - 2 <= smax:
        nmax = int((smax - (2 * g - 2)) * 2)
        for n in range(0, max(nmax, 0) + 1):
            for orders in itertools.combinations_with_replacement(range(2, M + 1), n):
                s = Fr(2 * g - 2) + sum(1 - Fr(1, a) for a in orders)
                if 0 < s <= smax:
                    out.append((g, orders, s))
        g += 1
    return out


def heat_key(g, orders, L):
    mult = {}
    for a in orders:
        mult[a] = mult.get(a, 0) + 1
    return tuple(heat_mult(g, mult, L)[0])


print("6. brute force (all signatures with orders <= M and Area/2pi <= bound):")
SHARP_AREA = {M: s1 for (M, C, s, g, gp, n1, n2, s1) in rows}
for M, smax in ((2, Fr(6)), (3, Fr(9)), (4, SHARP_AREA[4])):
    sigs = all_sigs(M, smax)
    keyM = {}
    for g, o, s in sigs:
        keyM.setdefault(heat_key(g, o, M), []).append((g, o))
    check(all(len(v) == 1 for v in keyM.values()), f"c_1..c_M separate all signatures, M = {M}")
    keyM1 = {}
    for g, o, s in sigs:
        keyM1.setdefault(heat_key(g, o, M - 1), []).append((g, o, s))
    clash = [v for v in keyM1.values() if len(v) > 1]
    smin = min((min(x[2] for x in v) for v in clash), default=None)
    print(f"   M = {M}: {len(sigs)} signatures with Area/2pi <= {smax}; c_1..c_{M} injective; "
          + (f"pairs sharing c_1..c_{M - 1}: {len(clash)} classes, least Area/2pi = {smin}" if clash else
             f"no pair shares c_1..c_{M - 1} below this bound"))
    if M == 2:
        check(smin == Fr(1, 2), "M = 2: (1;2) and (0;2,2,2,2,2) share c_1")
    if M == 3:
        check(clash and smin == 6, "M = 3: least area of a pair sharing c_1, c_2 is 12 pi, (1;3^9)/(0;2^16)")
        ex = [v for v in clash if any(x[0] == 1 and x[1] == (3,) * 9 for x in v)]
        check(ex and any(x[0] == 0 and x[1] == (2,) * 16 for x in ex[0]), "M = 3 example")
    if M == 4:
        check(clash and smin == SHARP_AREA[4], "M = 4: least area of a pair sharing c_1..c_3 is that of the sharp pair, 36 pi")


# ---------------------------------------------------------------- 7. parts (iii)-(iv): arbitrary competitors
def multisets_with_sum(sigma, lo=2):
    """All sorted tuples m_1 <= ... <= m_n, m_i >= lo, with sum(1 - 1/m_i) = sigma (complete)."""
    out = []

    def rec(sig, r, lo, pre):
        if r == 1:
            rest = 1 - sig
            if rest > 0 and rest.numerator == 1 and rest.denominator >= lo:
                out.append(pre + (rest.denominator,))
            return
        if not (Fr(r, 2) <= sig < r):
            return
        a = lo
        while a <= Fr(r) / (r - sig):
            rec(sig - (1 - Fr(1, a)), r - 1, a, pre + (a,))
            a += 1
    if sigma == 0:
        return [()]
    for n in range(int(sigma) + 1, int(2 * sigma) + 1):
        rec(sigma, n, lo, ())
    return out


def area_class(s):
    """Every signature (g; m) with 2g - 2 + sum(1 - 1/m_i) = s, no bound on the orders."""
    cls, g = [], 0
    while 2 * g - 2 <= s:
        cls += [(g, m) for m in multisets_with_sum(s + 2 - 2 * g)]
        g += 1
    return cls


def sign_changes(sig, oth):
    """Sign changes of the measure nu of the proof of part (iii) for the pair (sig, oth) of equal area:
    paddings by 1s (Lemma sigdata), common elements removed, weight +-1/x at x^2."""
    (g, m), (gp, mp_) = sig, oth
    d = sum(Fr(1, a) for a in mp_) - sum(Fr(1, a) for a in m)
    check(d.denominator == 1, "d = R(m') - R(m) is an integer at equal area")
    d = int(d)
    U = list(m) + [1] * max(d, 0)
    V = list(mp_) + [1] * max(-d, 0)
    cnt = {}
    for x in U:
        cnt[x] = cnt.get(x, 0) + 1
    for x in V:
        cnt[x] = cnt.get(x, 0) - 1
    signs = [1 if cnt[x] > 0 else -1 for x in sorted(cnt) if cnt[x] != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b), len(signs)


import math  # noqa: E402

KCAP = 9
# complete equal-area classes: every Area/2pi = s <= 2 attained with genus <= 1, at most 5 cone points of order
# <= 4, and s = 3, the class of the example (0;2^10), (1;4^4); no bound on the orders inside a class
svals = sorted({Fr(2 * g - 2) + sum(1 - Fr(1, a) for a in o)
                for g in range(0, 2) for n in range(0, 6) for o in itertools.combinations_with_replacement(range(2, 5), n)
                if 0 < Fr(2 * g - 2) + sum(1 - Fr(1, a) for a in o) <= 2} | {Fr(3)})
nsig, npair, worst, worst_d, attained = 0, 0, {}, {}, {}
for sv in svals:
    cls = area_class(sv)
    nsig += len(cls)
    keys = {sig: heat_key(sig[0], sig[1], KCAP) for sig in cls}
    # K_mult(O; Sig) = least k with key[:k] unique in the class (all of the class has the same area, so c_1 agrees)
    count = {}
    for sig in cls:
        for k in range(1, KCAP + 1):
            count[keys[sig][:k]] = count.get(keys[sig][:k], 0) + 1
    groups = {}
    for sig in cls:
        groups.setdefault(keys[sig][:2], []).append(sig)
    for sig in cls:
        M = max(sig[1], default=1)
        dO = len(set(sig[1]))
        K = next(k for k in range(1, KCAP + 1) if count[keys[sig][:k]] == 1)
        check(K <= M + 1, f"(iii) K_mult({sig}) = {K} > M + 1 = {M + 1}")
        check(K <= 2 * dO + 2, f"(iii) K_mult({sig}) = {K} > 2 d_O + 2 = {2 * dO + 2}")
        worst[M] = max(worst.get(M, 0), K)
        worst_d[dO] = max(worst_d.get(dO, 0), K)
        if K == M + 1:
            attained.setdefault(M, (sv, sig))
        # pairs sharing at least c_1, c_2 (L <= 1 satisfies every bound trivially): the measure of the proof
        for o in groups[keys[sig][:2]]:
            if o == sig:
                continue
            L = next(k for k in range(1, KCAP + 1) if keys[o][:k] != keys[sig][:k]) - 1
            ch, npts = sign_changes(sig, o)
            check(npts > 0, "different signatures give a nonzero measure")
            check(ch >= L, f"Lemma signs: {ch} sign changes < L = {L} for {sig}, {o}")
            check(L <= M and L <= 2 * min(dO, len(set(o[1]))) + 1, f"(iii) L = {L} for {sig}, {o}")
            npair += 1
print(f"7. part (iii): {len(svals)} complete area classes (Area/2pi <= 2 and = 3; {nsig} signatures, no bound on the "
      f"orders), {npair} ordered pairs sharing at least c_1, c_2: every such pair sharing c_1..c_L has at least L sign changes, L <= M and "
      f"L <= 2 min(d_O, d_O') + 1; K_mult(O; Sig) <= min(M + 1, 2 d_O + 2) for every O")
print(f"   largest K_mult by largest order M <= 12: {dict(sorted((k, v) for k, v in worst.items() if k <= 12))}; "
      f"largest K_mult - M over all M: {max(v - k for k, v in worst.items())}")
print(f"   largest K_mult by number of distinct orders d_O: {dict(sorted(worst_d.items()))}")
for M, (sv, sig) in sorted(attained.items()):
    print(f"   M + 1 attained at M = {M}: {sig}, Area/2pi = {sv}")


# ---------------------------------------------------------------- 8. part (iv): the bound M + 1 is attained
def lagrange_w_nodes(nodes):
    return {a: Fr(1, prod(a * a - b * b for b in nodes if b != a)) for a in nodes}


def sharp_pair_X(M, X):
    """the paper, part (iv): nodes {1..M, X}, nu(a) = -C a w_a with the least positive integer C making nu
    integral on {2..M, X} and s = sum nu(a)(1 - 1/a) even; least admissible genera."""
    nodes = list(range(1, M + 1)) + [X]
    w = lagrange_w_nodes(nodes)
    nu0 = {a: -a * w[a] for a in nodes if a != 1}
    # least C: the lcm of the denominators makes nu integral; then the least multiple making s even
    C = 1
    for v in nu0.values():
        C = C * v.denominator // gcd(C, v.denominator)
    s1 = sum(C * v * (1 - Fr(1, a)) for a, v in nu0.items())
    k = 1
    while not ((s1 * k).denominator == 1 and (s1 * k).numerator % 2 == 0):
        k += 1
    C *= k
    nu = {a: int(C * v) for a, v in nu0.items()}
    s = int(sum(Fr(v) * (1 - Fr(1, a)) for a, v in nu.items()))
    m = {a: v for a, v in nu.items() if v > 0}
    mp_ = {a: -v for a, v in nu.items() if v < 0}
    g = max(0, -s // 2)
    while Fr(2 * g - 2) + sum(c * (1 - Fr(1, a)) for a, c in m.items()) <= 0:
        g += 1
    return C, nu, s, (g, m), (g + s // 2, mp_), w


n4 = 0
for M in range(1, 13):
    for X in range(M + 1, M + 5):
        C, nu, s, (g, m), (gp, mp_), w = sharp_pair_X(M, X)
        check(all(v != 0 for v in nu.values()), "every nu(a) nonzero")
        check(X in mp_ and all(2 <= a <= M for a in m), f"(iv) X on the side of O', O in Sig_<=M (M={M}, X={X})")
        H1, s1 = heat_mult(g, m, M + 1)
        H2, s2 = heat_mult(gp, mp_, M + 1)
        check(s1 == s2 and s1 > 0 and gp >= 0, f"(iv) equal positive area, M={M}, X={X}")
        check(H1[:M] == H2[:M], f"(iv) c_1..c_M agree, M={M}, X={X}")
        check(H1[M] - H2[M] == (-1) ** M * C * lead_coef(M - 1), f"(iv) Delta c_(M+1) = (-1)^M C a_(M-1), M={M}, X={X}")
        if X == M + 1 and M >= 1:
            # the pair of part (ii) at M + 1, with its two sides exchanged
            C2, nu2, s2_, _, _ = sharp_pair(M + 1)
            check(C2 == C and all(nu2[a] == -nu[a] for a in nu), f"(iv) at X = M + 1 is (ii) at M + 1, M={M}")
        n4 += 1
C, nu, s, (g, m), (gp, mp_), w = sharp_pair_X(2, 4)
check((w[1], w[2], w[4]) == (Fr(1, 45), Fr(-1, 36), Fr(1, 180)) and C == 180 and nu == {2: 10, 4: -4} and s == 2,
      "(iv) the numbers of the example M = 2, X = 4")
check((g, m) == (0, {2: 10}) and (gp, mp_) == (1, {4: 4}), "(iv) the example is (0;2^10) against (1;4^4)")
H1, s1 = heat_mult(0, {2: 10}, 4)
H2, s2 = heat_mult(1, {4: 4}, 4)
check(s1 == s2 == 3, "(iv) the example has area 6 pi")
check(H1[:2] == H2[:2] and H1[2] != H2[2], "(iv) the example shares exactly c_1, c_2")
cls3 = area_class(Fr(3))  # 3667 signatures
k3 = {sig: heat_key(sig[0], sig[1], 4) for sig in cls3}
K = next(k for k in range(1, 5) if all(k3[o][:k] != k3[(0, (2,) * 10)][:k] for o in cls3 if o != (0, (2,) * 10)))
check(K == 3, "(iv) K_mult((0;2^10); Sig) = 3 = M + 1, from its complete area class")
# surfaces (M = 1): K_mult <= 2, attained by (2;) and (0;2^8)
Hs, ss = heat_mult(2, {}, 3)
Ho, so = heat_mult(0, {2: 8}, 3)
check(ss == so == 2 and Hs[0] == Ho[0] and Hs[1] != Ho[1], "surfaces: (2;) and (0;2^8) share exactly c_1")
print(f"8. part (iv): {n4} pairs (1 <= M <= 12, M < X <= M + 4) with O in Sig_<=M and O' containing X share exactly "
      f"c_1..c_M, Delta c_(M+1) = (-1)^M C a_(M-1); X = M + 1 is part (ii) at M + 1 with the sides exchanged; "
      f"(0;2^10) and (1;4^4), area 6 pi, share exactly c_1, c_2, and K_mult((0;2^10); Sig) = 3 exactly")
print("all checks passed")

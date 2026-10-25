"""Explicit r_rem and |dM| bounds of Proposition S3.10, checkable from the paper.

Re-implements Proposition S3.10 from the formulas printed in the paper ONLY (own Bernoulli numbers,
own tanh/tan series, own front end from the trace-formula p_l of the paper), then
  (a) the front end F of Proposition 6.2 reproduces the heat coefficients c_1..c_n directly   [exact]
  (b) at the printed delta_cert of Table 4, for all 11 cases, the vectors r_rem, the matrix
      A = |M^{-1}| |dM| and the vector E agree EXACTLY with theory/stability/threshold.py      [exact]
  (c) the printed delta_cert is certified by test (i) or test (ii) of the proposition, using
      only the formulas of the paper                                                     [exact]
  (d) the printed delta_thm is also certified (as Table 4 claims delta_thm <= delta_cert)    [exact]

Run from the repository root:
    python3 checks/proofs/check_prop610.py
Writes checks/proofs/check_prop610.txt.  Exits nonzero on any failure.
"""
import os
import sys
from fractions import Fraction as Fr
from math import comb, factorial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "check_prop610.txt")
log = []
fails = 0


def check(cond, msg):
    global fails
    log.append(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


# ------------------------------------------------------------------ constants (own)
_B = {0: Fr(1)}


def B(n):
    if n not in _B:
        _B[n] = -sum(comb(n + 1, j) * B(j) for j in range(n)) / (n + 1)
    return _B[n]


def B_half(n):
    return Fr(1) if n == 0 else (Fr(2) ** (1 - n) - 1) * B(n)


def alpha(k):
    return Fr((-1) ** k, factorial(k) * 4 ** k) * sum(comb(k, l) * Fr(-4) ** l * B_half(2 * l) for l in range(k + 1))


def sig(i):
    return Fr((-1) ** (i + 1) * (2 ** (2 * i) - 2)) * B(2 * i) / factorial(2 * i)


def p_coeffs(l):
    """a_{l,k}, k = 0..l+1, with p_l(x) = sum_k a_{l,k} x^{2k} (the paper, eq. (bl))."""
    a = [Fr(0)] * (l + 2)
    for k in range(l + 1):
        w = Fr(factorial(2 * k), 4 ** l * factorial(k) * factorial(l - k))
        for n in range(1, k + 2):
            c = w * Fr(1, 4) * sig(k + 1 - n) * Fr(4 ** n) * abs(B(2 * n)) / factorial(2 * n)
            a[n] += c
            a[0] -= c
    return a


# ------------------------------------------------------------------ linear algebra and series (own)
def inv(A):
    n = len(A)
    M = [row[:] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def mv(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def mm(A, Bm):
    return [[sum(A[i][k] * Bm[k][j] for k in range(len(Bm))) for j in range(len(Bm[0]))] for i in range(len(A))]


def absm(A):
    return [[abs(x) for x in row] for row in A]


def smul(a, b, N):
    return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(N + 1)]


def compose_odd(coef, U, N):
    """sum_j coef[j] U^{2j+1}, U with zero constant term, truncated at z^N."""
    out = [Fr(0)] * (N + 1)
    P = U[:]                       # U^1
    U2 = smul(U, U, N)
    for j in range(len(coef)):
        out = [o + coef[j] * p for o, p in zip(out, P)]
        P = smul(P, U2, N)
    return out


def tanh_coef(J):
    return [Fr(2 ** (2 * n) * (2 ** (2 * n) - 1)) * B(2 * n) / factorial(2 * n) for n in range(1, J + 2)]


def tan_coef(J):
    return [Fr((-1) ** (n - 1) * 2 ** (2 * n) * (2 ** (2 * n) - 1)) * B(2 * n) / factorial(2 * n) for n in range(1, J + 2)]


# ------------------------------------------------------------------ the recovery map's ingredients
def front_end(n):
    """F (n x n) of Proposition 6.2 and h0(n)."""
    F = [[Fr(0)] * n for _ in range(n)]
    F[0][0] = Fr(-1, 2)
    for nu in range(n - 1):
        a = p_coeffs(nu)
        F[nu + 1][0] = -alpha(nu + 1) / 2 + (-1) ** nu * a[0]
        for k in range(1, nu + 2):
            F[nu + 1][k] = (-1) ** nu * a[k]
    h0 = [Fr(n - 2, 2) * alpha(j) for j in range(n)]
    return F, h0


def invariants(m):
    n = len(m)
    return [sum(Fr(1, x) for x in m)] + [sum(Fr(x) ** (2 * k - 1) for x in m) for k in range(1, n)]


def elem(m):
    e = [Fr(1)]
    for x in m:
        e = [a + Fr(x) * b for a, b in zip(e + [Fr(0)], [Fr(0)] + e)]
    return e                                   # e[0..n]


def system(I, n):
    """M, b, T of Theorem B; rows j = 0..n-2: e_{2j+1} - sum_{i<=j} T_{2i+1} e_{2j-2i} = 0; last: e_{n-1} - R e_n = 0."""
    N = 2 * n - 3
    U = [Fr(0)] * (N + 1)
    for k in range(1, n):
        U[2 * k - 1] = I[k] / (2 * k - 1)
    T = compose_odd(tanh_coef(N // 2), U, N)
    M = [[Fr(0)] * n for _ in range(n)]
    b = [Fr(0)] * n
    for j in range(n - 1):
        if 2 * j + 1 <= n:
            M[j][2 * j] += 1
        for i in range(j + 1):
            idx = 2 * j - 2 * i
            if idx == 0:
                b[j] += T[2 * i + 1]
            elif idx <= n:
                M[j][idx - 1] -= T[2 * i + 1]
    M[n - 1][n - 2] += 1
    M[n - 1][n - 1] -= I[0]
    return M, b, T, U


def bounds(m, delta):
    """Steps 1-5 of Proposition 6.10 with the explicit formulas of the paper."""
    n = len(m)
    N = 2 * n - 3
    F, _ = front_end(n)
    Fi = inv(F)
    D = mv(absm(Fi), delta)                                   # step 1: Delta_r >= |delta I_r|
    I, e = invariants(m), elem(m)
    M, b, T, U = system(I, n)
    dU = [Fr(0)] * (N + 1)
    for k in range(1, n):
        dU[2 * k - 1] = D[k] / (2 * k - 1)
    sech2 = [Fr(int(i == 0)) - x for i, x in enumerate(smul(T, T, N))]
    tU = compose_odd(tan_coef(N // 2), U, N)
    tUd = compose_odd(tan_coef(N // 2), [x + y for x, y in zip(U, dU)], N)
    sec2 = [Fr(int(i == 0)) + x for i, x in enumerate(smul(tU, tU, N))]
    tau_lin = smul([abs(x) for x in sech2], dU, N)
    tau_rem = [x - y - z for x, y, z in zip(tUd, tU, smul(sec2, dU, N))]
    assert all(x >= 0 for x in tau_rem)
    tau = [x + y for x, y in zip(tau_lin, tau_rem)]            # step 2: tau_k >= |delta T_k|
    E_ = lambda k: e[k] if 0 <= k <= n else Fr(0)
    rho = [sum(tau[2 * i + 1] * E_(2 * j - 2 * i) for i in range(j + 1)) for j in range(n - 1)] + [D[0] * e[n]]
    rho_rem = [sum(tau_rem[2 * i + 1] * E_(2 * j - 2 * i) for i in range(j + 1)) for j in range(n - 1)] + [Fr(0)]
    DM = [[Fr(0)] * n for _ in range(n)]
    for j in range(n - 1):
        for i in range(j):
            c = 2 * j - 2 * i
            if 1 <= c <= n:
                DM[j][c - 1] += tau[2 * i + 1]
    DM[n - 1][n - 1] += D[0]                                   # step 3
    Mi = inv(M)
    A = mm(absm(Mi), DM)
    IA = [[Fr(int(i == j)) - A[i][j] for j in range(n)] for i in range(n)]
    IAi = inv(IA)
    v = mv(IAi, [Fr(1)] * n)                                   # step 4
    E = mv(IAi, mv(absm(Mi), rho))                             # step 5
    # test (ii) ingredients: J = d(Me - b)/dI, G = -M^{-1} J F^{-1}
    J = [[Fr(0)] * n for _ in range(n)]
    for j in range(n - 1):
        for k in range(1, n):
            J[j][k] = -sum(E_(2 * j - 2 * i) * sech2[2 * i + 2 - 2 * k] / (2 * k - 1)
                           for i in range(j + 1) if 2 * i + 1 >= 2 * k - 1 and 2 * j - 2 * i <= n)
    J[n - 1][0] = -e[n]
    G = mm([[-x for x in row] for row in mm(Mi, J)], Fi)
    vrho = [x + y for x, y in zip(mv(absm(Mi), rho_rem), mv(A, E))]
    return dict(n=n, v=v, E=E, A=A, rho=rho, rho_rem=rho_rem, DM=DM, G=G, vrho=vrho, F=F, Fi=Fi)


RADII = [Fr(j, 40) for j in range(20, 0, -1)] + [Fr(1, 100), Fr(1, 1000)]


def taylor_abs(desc, a):
    deg = len(desc) - 1
    asc = list(reversed(desc))
    return [abs(sum(comb(i, l) * asc[i] * Fr(a) ** (i - l) for i in range(l, deg + 1))) for l in range(deg + 1)]


def certified(m, delta):
    C = bounds(m, delta)
    n = C["n"]
    if not all(x > 0 for x in C["v"]) or not all(x >= 0 for x in C["E"]):
        return False, C
    cl = {}
    for x in m:
        cl[x] = cl.get(x, 0) + 1
    tays = [taylor_abs([Fr(0)] + [(-1) ** j * C["G"][j - 1][c] for j in range(1, n + 1)], 0) for c in range(n)]
    for a, k in cl.items():
        tays = [taylor_abs([Fr(0)] + [(-1) ** j * C["G"][j - 1][c] for j in range(1, n + 1)], a) for c in range(n)]
        ok = False
        for rad in RADII:
            low = rad ** k
            for bb, kb in cl.items():
                if bb != a:
                    low *= (abs(Fr(a - bb)) - rad) ** kb
            t1 = sum(C["E"][j - 1] * (a + rad) ** (n - j) for j in range(1, n + 1))
            t2 = (sum(delta[c] * sum(t * rad ** l for l, t in enumerate(tays[c])) for c in range(n))
                  + sum(C["vrho"][j - 1] * (a + rad) ** (n - j) for j in range(1, n + 1)))
            if low > min(t1, t2):
                ok = True
                break
        if not ok:
            return False, C
    return True, C


# ------------------------------------------------------------------ (a) front end against direct coefficients
for m in [(2, 8, 8), (3, 3, 12), (3, 10, 15, 30), (2, 2, 2, 2, 3)]:
    n = len(m)
    F, h0 = front_end(n)
    c_front = [x + y for x, y in zip(mv(F, invariants(m)), h0)]
    R = sum(Fr(1, x) for x in m)
    c_dir = [(n - 2 - R) / 2] + [alpha(nu + 1) * (n - 2 - R) / 2
                                 + sum(Fr((-1) ** nu) * sum(c * Fr(x) ** (2 * k) for k, c in enumerate(p_coeffs(nu))) / x for x in m)
                                 for nu in range(n - 1)]
    check(c_front == c_dir, f"(a) {m}: F I + h0 = (c_1..c_n) = {[str(x) for x in c_dir]}")
check([str(x) for x in [Fr(1, 8), Fr(67, 48), Fr(-1601, 480)]] ==
      [str(x) for x in [x + y for x, y in zip(mv(front_end(3)[0], invariants((2, 8, 8))), front_end(3)[1])]],
      "(a) (2,8,8): c = (1/8, 67/48, -1601/480) as in Section 2 of the paper")

# ------------------------------------------------------------------ (b)-(d)
sys.path.insert(0, os.path.join(ROOT, "theory", "stability"))
import threshold as TH  # noqa: E402

TABLE = {(2, 8, 8): ("3.80e-7", "2.341e-3"), (3, 3, 12): ("1.18e-7", "4.040e-3"),
         (3, 10, 15, 30): ("4.02e-11", "3.660e-3"), (4, 5, 21, 28): ("3.14e-11", "1.461e-3"),
         (2, 3, 7): ("4.49e-7", "3.658e-3"), (4, 4, 4): ("9.35e-7", "4.539e-4"), (7, 7, 7): ("9.97e-8", "8.068e-5"),
         (3, 3, 4, 4): ("1.48e-9", "9.597e-5"), (5, 5, 5, 5): ("4.74e-10", "3.617e-5"), (2, 2, 2, 3): ("2.03e-9", "1.858e-4"),
         (2, 2, 2, 2, 3): ("2.72e-12", "7.908e-6")}
check(sorted(TABLE) == sorted(TH.CASES), "(b) the 11 cases of Table 4 are threshold.py's CASES")
for m, (sthm, scert) in TABLE.items():
    n = len(m)
    dcert = [Fr(scert)] * n
    ok, C = certified(m, dcert)
    theirs = TH._common(m, dcert)
    same = (theirs is not None and theirs["E"] == C["E"] and theirs["A"] == C["A"] and theirs["r_rem"] == C["rho_rem"])
    check(same, f"(b) {m}: r_rem, A = |M^-1||dM| and E agree exactly with threshold.py at delta = {scert}")
    check(ok and TH.certify(m, dcert), f"(c) {m}: delta_cert = {scert} certified from the printed formulas (and by threshold.py)")
    okt, _ = certified(m, [Fr(sthm)] * n)
    check(okt, f"(d) {m}: delta_thm = {sthm} certified")

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{len(log)} checks, {fails} failures\n")
print("\n".join(log))
print(f"{len(log)} checks, {fails} failures")
sys.exit(1 if fails else 0)

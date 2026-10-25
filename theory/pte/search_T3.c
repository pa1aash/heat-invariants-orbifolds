// theory/pte/search_T3.c -- exhaustive search for a size-8 genus collision sharing 3 heat coefficients
// (control: search_T3_control.py; superseded by the exact test of search_T3_modular.c).
//
// Shape (3,5) is forced (the paper Theorem 2.1). V = {v1<=...<=v5}: positive integers, gcd 1, v5 <= NMAX.
// Given V, the 3-element side U is the root set of
//     K t^3 - K S t^2 + M Rn t - M Rd = 0,   K = 3(Rd - S Rn),  M = C - S^3,
// where S = sum v, C = sum v^3, Rn/Rd = sum 1/v (eliminate e1 = S, e2 = R e3 and e3 from p3 = C).
// A rational root p/q has q | K' (K' = K / content), so K' r must be an integer.
//
// decide() returns
//    1  all three roots pass the integrality test            -> candidate, verified exactly
//    2  (numerically) repeated root: splits over Q anyway     -> candidate, verified exactly
//    3  floating point cannot decide with certainty           -> candidate, verified exactly
//    0  certainly not three rational roots                    -> rejected
//   <0  not three positive real roots (or degenerate input)   -> rejected
// Rejection is only issued when it is certain up to the stated floating-point error model (all
// rounding and coefficient errors <= 64 LDBL_EPSILON relative, with extra safety factors; note that
// on arm64 macOS long double is IEEE double):
//  - "one real root" only if the discriminant Rr^2 - Q^3 exceeds 4x its error bound;
//  - otherwise the three roots are located in Smith's inclusion disks (radius 3|f(r_i)|/prod|r_i-r_j|,
//    doubled), required to be pairwise disjoint; each disk then holds exactly one root, and that root
//    is not in (1/K')Z if K' r_i is farther from every integer than K' times the radius.
// Anything else (near-zero discriminant, overlapping disks, radius too large) is routed to the exact
// check.
//
// Usage:  search_T3 NMAX V5_FIRST V5_LAST      (search; candidates on stdout, counters on stderr)
//         search_T3 test                       (reads lines "S C Rn Rd", prints decide() per line)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
typedef __int128 i128;
static i128 g128(i128 a, i128 b) { if (a < 0) a = -a; if (b < 0) b = -b; while (b) { i128 t = a % b; a = b; b = t; } return a; }
static long gl(long a, long b) { while (b) { long t = a % b; a = b; b = t; } return a; }

static int decide(i128 S, i128 C, i128 Rn, i128 Rd) {
    i128 K = 3 * (Rd - S * Rn), M = C - S * S * S;
    if (K == 0 || M == 0) return -1;
    i128 g = g128(g128(K, M * Rn), M * Rd);
    i128 Kp = K / g; if (Kp < 0) Kp = -Kp;
    long double Sd = (long double)S, Rv = (long double)Rn / (long double)Rd;
    long double e3 = ((long double)M) / (3.0L * (1.0L - Sd * Rv)), e2 = Rv * e3;
    if (!(e3 > 0 && e2 > 0)) return -2;      // positive roots need e1,e2,e3 > 0; then every real root is > 0
    long double a = -Sd, b = e2, c = -e3;     // monic cubic f = t^3 + a t^2 + b t + c (coefficients to ~1e-18 relative)
    const long double EPS = 64 * LDBL_EPSILON; // generous bound on relative rounding/coefficient error
                                              // (on arm64 macOS long double == double, LDBL_EPSILON ~ 2.2e-16)
    long double Q = (a * a - 3 * b) / 9, Rr = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
    long double eQ = EPS * (a * a + 3 * fabsl(b)) / 9 * 4, eR = EPS * (2 * fabsl(a * a * a) + 9 * fabsl(a * b) + 27 * fabsl(c)) / 54 * 4;
    long double D = Rr * Rr - Q * Q * Q, eD = 2 * fabsl(Rr) * eR + eR * eR + 3 * Q * Q * eQ + 3 * fabsl(Q) * eQ * eQ + eQ * eQ * eQ + EPS * (Rr * Rr + fabsl(Q * Q * Q));
    if (fabsl(D) <= 4 * eD) return 2;         // discriminant indistinguishable from 0: repeated root possible
    if (D > 0) return -3;                     // certainly one real root only
    long double th = acosl(Rr / sqrtl(Q * Q * Q)), sq = -2 * sqrtl(Q), r[3];
    r[0] = sq * cosl(th / 3) - a / 3; r[1] = sq * cosl((th + 2 * M_PI) / 3) - a / 3; r[2] = sq * cosl((th - 2 * M_PI) / 3) - a / 3;
    for (int i = 0; i < 3; i++) {
        long double x = r[i];
        for (int k = 0; k < 5; k++) { long double f = ((x + a) * x + b) * x + c, fp = (3 * x + 2 * a) * x + b; if (fp == 0) break; long double xn = x - f / fp; if (!(fabsl(xn - x) < fabsl(x))) break; x = xn; }
        r[i] = x;
    }
    // Smith's inclusion disks: all zeros lie in the union of |z - r_i| <= 3|f(r_i)| / prod_{j!=i}|r_i - r_j|,
    // and pairwise disjoint disks each contain exactly one zero. f(r_i) is bounded by the computed value plus
    // EPS * sum |a_k||r_i|^k; the radius is doubled for safety.
    long double rho[3];
    for (int i = 0; i < 3; i++) {
        long double x = r[i], f = ((x + a) * x + b) * x + c;
        long double mag = fabsl(x * x * x) + fabsl(a) * x * x + fabsl(b * x) + fabsl(c);
        long double den = 1; for (int j = 0; j < 3; j++) if (j != i) den *= fabsl(x - r[j]);
        if (!(den > 0)) return 3;
        rho[i] = 2 * 3 * (fabsl(f) + EPS * mag) / den;
    }
    for (int i = 0; i < 3; i++) for (int j = i + 1; j < 3; j++) if (rho[i] + rho[j] >= fabsl(r[i] - r[j])) return 3;
    long double Kd = (long double)Kp;
    int ok = 1;
    for (int i = 0; i < 3; i++) {
        long double y = Kd * r[i], tol = Kd * rho[i] + fabsl(y) * 8 * LDBL_EPSILON + 1e-9L;
        if (tol >= 0.2L) return 3;                       // cannot decide integrality of K' r with certainty
        if (fabsl(y - roundl(y)) > tol) ok = 0;          // the root in this disk is certainly not in (1/K')Z
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "test") == 0) {
        long long S, C, Rn, Rd;
        while (scanf("%lld %lld %lld %lld", &S, &C, &Rn, &Rd) == 4) printf("%d\n", decide(S, C, Rn, Rd));
        return 0;
    }
    if (argc < 4) { fprintf(stderr, "usage: search_T3 NMAX V5_FIRST V5_LAST | search_T3 test\n"); return 1; }
    int N = atoi(argv[1]), lo = atoi(argv[2]), hi = atoi(argv[3]);
    if (hi > N) hi = N;
    long cnt = 0, real3 = 0, cand = 0, cand_rep = 0, cand_unsure = 0;
    for (int v5 = lo; v5 <= hi; v5++) for (int v4 = 1; v4 <= v5; v4++) for (int v3 = 1; v3 <= v4; v3++)
    for (int v2 = 1; v2 <= v3; v2++) for (int v1 = 1; v1 <= v2; v1++) {
        if (gl(gl(gl(v1, v2), gl(v3, v4)), v5) != 1) continue;
        cnt++;
        int v[5] = {v1, v2, v3, v4, v5};
        i128 Rd = 1; for (int i = 0; i < 5; i++) Rd = Rd / g128(Rd, v[i]) * v[i];
        i128 Rn = 0; for (int i = 0; i < 5; i++) Rn += Rd / v[i];
        i128 S = v1 + v2 + v3 + v4 + v5, C = 0; for (int i = 0; i < 5; i++) C += (i128)v[i] * v[i] * v[i];
        int d = decide(S, C, Rn, Rd);
        if (d >= 0) real3++;
        if (d >= 1) {
            cand++; if (d == 2) cand_rep++; if (d == 3) cand_unsure++;
            printf("V %d %d %d %d %d%s\n", v1, v2, v3, v4, v5, d == 2 ? " D" : d == 3 ? " X" : "");
        }
    }
    fprintf(stderr, "N=%d v5 in [%d,%d]: V-sets %ld, 3 positive real roots %ld, candidates %ld (repeated-root %ld, undecided %ld)\n",
            N, lo, hi, cnt, real3, cand, cand_rep, cand_unsure);
    return 0;
}

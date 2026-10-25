/* Exact modular exclusion for the first open genus case (Remark 3.18 of the paper): no 3-configuration of shape (3,5) whose five-element side V, made primitive, has
 * entries <= NMAX.
 *
 * The test (with a bitset table and a control mode). For a 5-multiset V of positive integers, a partner triple U with R(U) = R(V),
 * P1(U) = P1(V), P3(U) = P3(V) must consist of the roots of the monic cubic
 *     q_V(z) = z^3 - e1 z^2 + e2 z - e3,  e1 = P1(V),  e3 = (P3(V) - e1^3) / (3 (1 - e1 R(V))),
 *     e2 = R(V) e3
 * (Newton: P3 = e1^3 - 3 e1 e2 + 3 e3 and e2 = R e3; 1 - e1 R != 0 since e1 R >= 9 for any positive
 * triple, by the arithmetic-harmonic mean inequality). If U is rational, then for every prime p at
 * which e1, e2, e3 are p-integral, q_V splits into linear factors modulo p (a rational root of a
 * monic polynomial with p-integral coefficients is p-integral). For p > NMAX, R(V) is p-integral, so
 * the coefficients are p-integral exactly when p does not divide the denominator 3(1 - e1 R) read
 * modulo p; such primes are skipped, never used to reject. A V is REJECTED only when q_V fails to
 * split completely modulo some prime of PRIMES that is used: this is sound. Survivors (V that split
 * modulo every used prime) are printed and decided exactly by search_T3_modular.py.
 *
 * Splitting mod p is read from a table of all (e1, e2, e3) mod p of the form (a+b+c, ab+bc+ca, abc),
 * 0 <= a <= b <= c < p, stored as a bitset (p^3 bits).
 *
 * usage:  search_T3_modular NMAX A0 ASTEP        V = (a <= b <= c <= d <= e <= NMAX), gcd 1,
 *                                                with a = A0, A0 + ASTEP, ...   (survivors on stdout,
 *                                                "total N survivors S" on stderr)
 *         search_T3_modular --control NMAX       every 3-multiset W in [1, NMAX] run through the same
 *                                                test with V := W (then q_V = prod (z - w) splits):
 *                                                exits nonzero if any W is rejected
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long long ll;
#define NP 12
static const int PRIMES[NP] = {223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277};
static uint8_t *tab[NP];
static int *inv[NP];

static int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }
static ll powmod(ll a, ll e, ll p) { ll r = 1; a %= p; if (a < 0) a += p; while (e) { if (e & 1) r = r * a % p; a = a * a % p; e >>= 1; } return r; }

static void build(int nmax) {
    for (int k = 0; k < NP; k++) {
        int p = PRIMES[k];
        size_t nbits = (size_t)p * p * p;
        tab[k] = calloc(nbits / 8 + 1, 1);
        inv[k] = calloc(nmax + 1, sizeof(int));
        for (int i = 1; i <= nmax; i++) inv[k][i] = (int)powmod(i, p - 2, p);
        for (int a = 0; a < p; a++)
            for (int b = a; b < p; b++)
                for (int c = b; c < p; c++) {
                    ll e1 = (a + b + c) % p, e2 = ((ll)a * b + (ll)a * c + (ll)b * c) % p, e3 = (ll)a * b % p * c % p;
                    size_t idx = ((size_t)e1 * p + e2) * p + e3;
                    tab[k][idx >> 3] |= (uint8_t)(1u << (idx & 7));
                }
    }
}

/* 1 if V (n entries, all <= nmax < every prime) survives every used prime, 0 if rejected */
static int survives(const int *x, int n) {
    for (int k = 0; k < NP; k++) {
        ll p = PRIMES[k], e1 = 0, R = 0, P3 = 0;
        for (int i = 0; i < n; i++) {
            e1 += x[i];
            R += inv[k][x[i]];
            P3 += (ll)x[i] % p * x[i] % p * x[i] % p;
        }
        e1 %= p; R %= p; P3 %= p;
        ll den = (3 - 3 * (e1 * R % p)) % p; if (den < 0) den += p;
        if (den == 0) continue;                      /* coefficients not p-integral: skip p */
        ll num = (P3 - e1 * e1 % p * e1) % p; if (num < 0) num += p;
        ll e3 = num * powmod(den, p - 2, p) % p, e2 = R * e3 % p;
        size_t idx = ((size_t)e1 * p + e2) * p + e3;
        if (!(tab[k][idx >> 3] & (1u << (idx & 7)))) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--control") == 0) {
        int nmax = atoi(argv[2]);
        build(nmax);
        ll n = 0, bad = 0;
        for (int a = 1; a <= nmax; a++)
            for (int b = a; b <= nmax; b++)
                for (int c = b; c <= nmax; c++) {
                    int w[3] = {a, b, c};
                    n++;
                    if (!survives(w, 3)) { bad++; printf("REJECTED-CONTROL %d %d %d\n", a, b, c); }
                }
        fprintf(stderr, "control %lld rejected %lld\n", n, bad);
        return bad ? 1 : 0;
    }
    if (argc != 4) { fprintf(stderr, "usage: search_T3_modular NMAX A0 ASTEP | --control NMAX\n"); return 2; }
    int nmax = atoi(argv[1]), a0 = atoi(argv[2]), st = atoi(argv[3]);
    if (nmax >= PRIMES[0]) { fprintf(stderr, "NMAX must be below %d\n", PRIMES[0]); return 2; }
    build(nmax);
    ll total = 0, surv = 0;
    for (int a = a0; a <= nmax; a += st)
        for (int b = a; b <= nmax; b++) {
            int g2 = gcd(a, b);
            for (int c = b; c <= nmax; c++) {
                int g3 = gcd(g2, c);
                for (int d = c; d <= nmax; d++) {
                    int g4 = gcd(g3, d);
                    for (int e = d; e <= nmax; e++) {
                        if (gcd(g4, e) != 1) continue;
                        total++;
                        int x[5] = {a, b, c, d, e};
                        if (survives(x, 5)) { surv++; printf("SURV %d %d %d %d %d\n", a, b, c, d, e); }
                    }
                }
            }
        }
    fprintf(stderr, "total %lld survivors %lld\n", total, surv);
    return 0;
}

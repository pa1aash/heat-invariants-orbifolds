"""The collision-free sums of Proposition S2.1, as a checkable certificate.

A sum S "carries a collision" if two distinct hyperbolic triads p <= q <= r (1/p+1/q+1/r < 1) with
p+q+r = S have the same reciprocal sum R.  Claim (the paper): for 18 <= S <= 4800 exactly the 38 sums
    19 21 22 23 24 25 27 28 29 30 33 41 44 46 47 48 49 50 51 59 65 67 81 99 115 119 123 125
    173 199 203 223 235 243 251 307 329 557
carry no collision; also no S <= 17 carries one.

  (a) for every S <= 17 and every listed S: exhaustive enumeration of the hyperbolic triads of sum S,
      reciprocal sums compared as exact fractions -- all distinct                         [exact]
  (b) for every other S in [18, 4800]: a witness pair, taken from the committed enumeration
      checks/classes4800/enum_4800.txt (first class listed at S) and re-verified here from
      scratch: two distinct sorted triads, both of sum S, both hyperbolic, equal R          [exact]
  (c) the witnesses are also produced independently of that file for every S whose witness is a
      scaled copy: if d | S and d has a witness, (S/d) x witness is one at S; the remaining
      ("primitive") sums are listed with their witnesses in check_thm513.txt

Run from the repository root:
    python3 checks/proofs/check_thm513.py
Writes checks/proofs/check_thm513.txt.  Exits nonzero on any failure.
"""
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ENUM = os.path.join(ROOT, "checks", "classes4800", "enum_4800.txt")
OUT = os.path.join(HERE, "check_thm513.txt")
log = []
fails = 0
SMAX = 4800
FREE = [19, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 41, 44, 46, 47, 48, 49, 50, 51, 59, 65, 67, 81, 99,
        115, 119, 123, 125, 173, 199, 203, 223, 235, 243, 251, 307, 329, 557]


def check(cond, msg):
    global fails
    log.append(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


def R(t):
    return Fr(1, t[0]) + Fr(1, t[1]) + Fr(1, t[2])


def triads(S):
    for p in range(2, S // 3 + 1):
        for q in range(p, (S - p) // 2 + 1):
            r = S - p - q
            if (q * r + p * r + p * q) < p * q * r:          # R < 1, exact in integers
                yield (p, q, r)


def collision_free(S):
    seen = set()
    for t in triads(S):
        k = R(t)
        if k in seen:
            return False
        seen.add(k)
    return True


def valid_witness(S, t1, t2):
    return (t1 != t2 and list(t1) == sorted(t1) and list(t2) == sorted(t2) and sum(t1) == S == sum(t2)
            and min(t1) >= 2 and min(t2) >= 2 and R(t1) < 1 and R(t2) < 1 and R(t1) == R(t2))


check(len(FREE) == 38 and FREE == sorted(set(FREE)) and max(FREE) == 557, "the list has 38 distinct sums, largest 557")
# (a)
for S in range(3, 18):
    check(collision_free(S), f"(a) S = {S}: no collision")
for S in FREE:
    n = sum(1 for _ in triads(S))
    check(collision_free(S), f"(a) S = {S}: {n} hyperbolic triads, all reciprocal sums distinct")

# (b)
wit = {}
with open(ENUM) as fh:
    for line in fh:
        if line.startswith("C "):
            parts = line.split()
            S = int(parts[1])
            if S not in wit:
                trip = [tuple(int(v) for v in x.split(",")) for x in parts[6:8]]
                wit[S] = (trip[0], trip[1])
others = [S for S in range(18, SMAX + 1) if S not in FREE]
bad = [S for S in others if S not in wit or not valid_witness(S, *wit[S])]
check(not bad, f"(b) all {len(others)} sums in [18, {SMAX}] outside the list carry a verified collision" + (f"; bad: {bad[:10]}" if bad else ""))
check(not any(S in wit for S in FREE), "(b) the committed enumeration lists no collision at any of the 38 sums")
check(len(others) + len(FREE) == SMAX - 17, f"(b) {len(others)} + 38 = {SMAX - 17} sums in [18, {SMAX}]")

# (c) witnesses from scaling alone
own = {}
for S in range(18, SMAX + 1):
    for d in range(18, S // 2 + 1):
        if S % d == 0 and d in own:
            k = S // d
            own[S] = (tuple(k * x for x in own[d][0]), tuple(k * x for x in own[d][1]))
            break
    else:
        if S in wit:
            own[S] = wit[S]
prim = [S for S in others if not (S in own and any(S % d == 0 and d in own for d in range(18, S // 2 + 1)))]
check(all(valid_witness(S, *own[S]) for S in others if S in own) and all(S in own for S in others),
      f"(c) {len(others) - len(prim)} sums get a witness by scaling a smaller one; {len(prim)} need their own")
log.append("     primitive witness sums and witnesses: " + "; ".join(f"{S}: {own[S][0]}~{own[S][1]}" for S in prim[:60]) + (" ..." if len(prim) > 60 else ""))

with open(os.path.join(HERE, "collision_witnesses.csv"), "w") as fh:
    fh.write("S,p1,q1,r1,p2,q2,r2,R_num,R_den\n")
    for S in prim:
        t1, t2 = own[S]
        fh.write(f"{S},{t1[0]},{t1[1]},{t1[2]},{t2[0]},{t2[1]},{t2[2]},{R(t1).numerator},{R(t1).denominator}\n")
check(len(prim) == 783, f"(c) {len(prim)} primitive witness pairs written to collision_witnesses.csv")

with open(OUT, "w") as fh:
    fh.write("\n".join(log) + f"\n\n{sum(1 for l in log if l[:4] in ('PASS', 'FAIL'))} checks, {fails} failures\n")
print(f"{fails} failures")
sys.exit(1 if fails else 0)

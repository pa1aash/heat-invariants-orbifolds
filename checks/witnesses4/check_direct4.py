"""Are there integer sharpness witnesses for Theorem A at n = 4 that are NOT pencil configurations?

Input: direct4_N440.txt from check_direct4.c (exhaustive: all pairs m != m' of 4-multisets of
positive integers <= 440 with equal R, P_1, P_3; output complete iff the last line says
'# N=440 ...' and every s in 4..1760 has '#done s').
For each collision: re-verify exactly; cancel common elements (none can remain: a common element
would leave two 3-sets with equal R,P1,P3, impossible by Theorem A at n=3, asserted here);
make Z = m + (-m') primitive and sign-normalised; test all 35 splits of Z into 4 + 4 for the
pencil condition (A with e1 = 0, B = -(Z minus A) with e3(A) = e3(B), e4(A) = e4(B));
compare with the 61 listed configurations.  Exits nonzero on failure of a verified fact.
"""
import sys
from fractions import Fraction as F
from math import gcd
from functools import reduce
from itertools import combinations
from collections import Counter
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from heatlib import shared

HERE = __file__.rsplit('/', 1)[0]
lines = open(HERE + '/direct4_N440.txt').read().splitlines()
assert lines[-1].startswith('# N=440'), lines[-1]
dones = {int(l.split()[1]) for l in lines if l.startswith('#done')}
assert dones == set(range(4, 1761))
pairs = []
for l in lines:
    if l.startswith('#'):
        continue
    left, right = l.split('|')
    a = list(map(int, left.split()))[2:]; b = list(map(int, right.split()))
    pairs.append((a, b))
print("collisions (m, m') with entries <= 440, equal R, P1, P3:", len(pairs))

txt = open(HERE + '/pencil_configurations.md').read()
blk = txt.split('### PW.5')[1].split('```')[1]
L61 = [tuple(int(x) for x in l.strip('[] ').split(',')) for l in blk.strip().splitlines()]

def canon(Z):
    g = reduce(gcd, Z)
    Z = [z // g for z in Z]
    return min(tuple(sorted(Z)), tuple(sorted(-z for z in Z)))
S61 = {canon(list(z)) for z in L61}

def esym(S, k):
    t = 0
    for c in combinations(S, k):
        p = 1
        for x in c:
            p *= x
        t += p
    return t

def splits(Z):
    out = []
    for A in combinations(range(8), 4):
        if 0 not in A:
            continue
        a = [Z[i] for i in A]; c = [-Z[i] for i in range(8) if i not in A]
        if sum(a) == 0 and esym(a, 3) == esym(c, 3) and esym(a, 4) == esym(c, 4):
            g1 = reduce(gcd, a); g2 = reduce(gcd, c)
            out.append((sorted(v // g1 for v in a), sorted(v // g2 for v in c)))
    return out

Zs = {}
for m, mp in pairs:
    assert m != mp and len(m) == len(mp) == 4
    assert sum(F(1, v) for v in m) == sum(F(1, v) for v in mp)
    assert sum(m) == sum(mp) and sum(v ** 3 for v in m) == sum(v ** 3 for v in mp)
    assert not (Counter(m) & Counter(mp)), (m, mp)
    k, _, _ = shared(0, m, 0, mp, 5)
    assert k == 3
    Zs.setdefault(canon(m + [-v for v in mp]), []).append((m, mp))
print("distinct primitive configurations:", len(Zs), "(the rest are integer multiples)")
nonpencil = []
inlist = 0
oneside, bothside = Counter(), Counter()
for Z, ex in sorted(Zs.items(), key=lambda kv: max(map(abs, kv[0]))):
    sp = splits(list(Z))
    if not sp:
        nonpencil.append(Z)
    if Z in S61:
        inlist += 1
mx = max(max(map(abs, z)) for z in Zs)
print("max entry among them:", mx)
print("pencil-decomposable:", len(Zs) - len(nonpencil), "; NOT pencil:", len(nonpencil))
for z in nonpencil:
    print("   NON-PENCIL witness:", z)
print("of these, in the claimed list of 61:", inlist)
listed_le440 = [z for z in S61 if max(map(abs, z)) <= 440]
print("listed configurations with max entry <= 440 (all 61 should be):", len(listed_le440),
      "; found by the direct search:", sum(1 for z in listed_le440 if z in Zs))
assert all(z in Zs for z in listed_le440)
notlisted = sorted((z for z in Zs if z not in S61), key=lambda z: max(map(abs, z)))
print("direct witnesses (max entry <= 440) not in the list:", len(notlisted))
for z in notlisted[:8]:
    print("   e.g.", z, "pencil splits (primitive A,B):", splits(list(z)))
for B in (88, 130, 220, 440):
    zz = [z for z in Zs if max(map(abs, z)) <= B]
    npz = [z for z in zz if z in nonpencil]
    print(f"  max entry <= {B}: {len(zz)} primitive witnesses, {len(npz)} not pencil, {sum(1 for z in zz if z in S61)} in the list")
z0 = (-88, -44, -37, -11, 16, 16, 74, 74)
m, mp = [16, 16, 74, 74], [11, 37, 44, 88]
assert sum(F(1, v) for v in m) == sum(F(1, v) for v in mp) == F(45, 296)
assert sum(m) == sum(mp) == 180 and sum(v ** 3 for v in m) == sum(v ** 3 for v in mp) == 818640
assert shared(0, m, 0, mp, 5)[0] == 3 and z0 in nonpencil
print("smallest non-pencil witness: (0;16,16,74,74) ~ (0;11,37,44,88): R = 45/296, P1 = 180, P3 = 818640, shares exactly 3")
print("smallest direct witness:", min(Zs, key=lambda z: (max(map(abs, z)), z)))
print("ALL OK")

#!/bin/sh
# Re-create numerics/refs_cache/ (not committed): the literature sources whose
# formulas and reference values are used here, retrieved headlessly from arXiv.
#   1812.06119  Schueth, corner contributions / cone coefficients (Thm 4.1, Rem 4.2)
#   1110.2150   Strohmaier-Uski, eigenvalues of hyperbolic surfaces (Bolza data)
#   math/0504571 Dryden-Strohmaier, Huber's theorem for orbisurfaces (trace formula)
#   1711.03405  Ucar, PhD thesis, heat invariants of orbisurfaces ((4.25),(4.33),(4.35))
set -e
cd "$(dirname "$0")"
mkdir -p refs_cache
cd refs_cache
for id in 1812.06119 1110.2150 math/0504571 1711.03405; do
  f=$(echo "$id" | tr / _)
  curl -sL -o "$f.pdf" "https://arxiv.org/pdf/$id"
done
for f in README.txt eig-bolza-refined0-1000.txt; do
  curl -sL -o "SU_$f" "https://arxiv.org/src/1110.2150v4/anc/$f"
done
../.venv/bin/python - <<'PY'
import glob, pymupdf
for f in glob.glob("*.pdf"):
    d = pymupdf.open(f)
    open(f[:-4] + ".txt", "w").write("\n".join(p.get_text() for p in d))
PY

# How many heat invariants determine a hyperbolic orbifold? Code and data

This repository reproduces every computation stated in the paper

> Palaash Gang, Akshaj Agadi, Arjun Veluri, Jerry Wang, Jeremy Barreto, Aarin Chouthaiwale, *How many heat invariants determine a hyperbolic orbifold?*,

and in its electronic supplement. The paper asks how many heat invariants determine the genus and the cone
orders of a closed hyperbolic 2-orbifold. The repository contains:

- `theory/audibility/`: Theorems 3.2-3.4, Orlando's formula, the searches of Remark S8.4;
- `theory/cone-coefficients/`, `theory/signatures/`, `theory/locality/`, `theory/curvature/`,
  `theory/divergence/`: the cone polynomials, signature data, complete area classes, the terms of the trace
  formula and the asymptotics of the heat coefficients;
- `theory/msep/verify.py`: the bounded-order theorem (Theorem 3.10), with `theory/eigen/eigen_common.py`;
- `theory/pte/`: the Prouhet–Tarry–Escott constructions, the growth bounds, the pencil search and the
  modular search of Remark S8.7 (C programs `search_T3.c`, `search_T3_modular.c`);
- `theory/threshold/`, `theory/stability/`: triangle orbifolds (Section 5) and stability (Section 6), with
  the blind recovery of the supplement (`theory/stability/blind/`, protocol in `PROTOCOL.md`);
- `theory/varcurv/`: the exact computer algebra of variable curvature (Section 2.4), and the checks of Proposition 2.13;
- `theory/diophantine/ranks.py`: the rank of $C_{27/2}$ (PARI/GP);
- `checks/proofs/`: exact checks of individual proofs (Lemmas 2.5-2.10, the descent of $C_{27/2}$,
  Propositions S2.1 and S3.10 and others), each writing its transcript;
- `checks/pairs/`: the explicit pairs of Table S1 (`d2_explicit_pairs.py`), the pencil counts
  (`d1_pencil_counts.py`) and $K_{\rm mult}$ on the area classes (`a1_fig2_kmult.py`);
- `checks/witnesses4/`: the exhaustive degree-4 search of Remark S8.4 (`check_direct4.c`) and its analysis;
- `checks/stability-constants/`: the constants $\zeta_n$ and the $\operatorname{sech}^2$ coefficients of
  Section S3;
- `numerics/`: the finite-element eigenvalues of $\mathcal O(2,8,8)$ and $\mathcal O(3,3,12)$ (`solve.py`,
  NGSolve), their validation (`validate.py`, `validate_committed.py`), heat traces and the double-window
  recomputation (`rerun_double_window.py`); `numerics/moduli/`: the moduli family of Section 4;
- `figures/`: Figures 1-6, S1 and S2 (`figures/src/F*.py`, data generators in `figures/gen/`);
- `tools/make_tables.py`: Tables S2-S4 (`tables/supplement_tables.tex`).

131 data files, 19.45 MB in all (`DATA.md`).

## Install and run

Requirements: Python 3.13 (3.9 or newer works for the checks; the byte-identical figure rebuilds need the
pinned versions), `make`, a C compiler (`cc` and `clang`) and, for the PARI step, `cypari2`.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --only-binary :all: -r requirements-pari.txt
make verify PYTHON=.venv/bin/python      # fast exact checks
make all PYTHON=.venv/bin/python         # everything
```

| target | what it runs | time |
|---|---|---|
| `make verify` | the fast exact checks | about 22 min |
| `make full` | the long computations | about 16 min (plus about 60 min with the optional NGSolve/Blender steps) |
| `make figures` | rebuild every figure from its data and compare with the committed file | about 1 min (plus about 1 min with the optional NGSolve/Blender steps) |
| `make tables` | regenerate every table and compare with the committed file | about 1 min |
| `make data-check` | compare every data file with `SHA256SUMS` | seconds |
| `make all` | all of the above | about 40 min (plus about 61 min with the optional NGSolve/Blender steps) |
| `make clean` | remove caches and compiled helpers | |

Each target prints PASS, FAIL or SKIP for every step and exits nonzero if any step fails. A step also
fails if it changes a committed data file: regenerating committed data is a reproduction test. Times were
measured on an Apple M-series laptop with 8 GB of memory. `python3 tools/steps.py --list` lists every step;
`TRACE.md` maps every checkable item of the paper to its script and target; `DATA.md` lists every data
file with its size, checksum, producing script and paper element.

## Optional components

- **NGSolve** (`requirements-optional.txt`, Python 3.13): the finite-element re-solve of the eigenvalue
  problems (`numerics/rerun_double_window.py`, about an hour). Set `PYTHON_NUMERICS` to its interpreter.
  Every check of the committed eigenvalue data runs without it. The original solves (`numerics/solve.py`,
  `numerics/moduli/solve_moduli.py` with `run_server.sh`) are included as the programs that produced the data.
- **Blender** (tested with 5.2.2): the renders of Figures 1 and 4 (`figures/src/F1.py`, `F6.py`), run in a
  scratch copy so that the committed PDFs are not replaced (a raster render is not byte-reproducible across
  machines). Their data assertions run without Blender.
- **PARI/GP** (`requirements-pari.txt`): the rank of $C_{27/2}$.
- Without any of these, the corresponding steps are skipped with a message.
- `numerics/validate.py` rebuilds the committed eigenvalue tables from the solver's raw output, which is not
  included (several GB); `numerics/fetch_refs.sh` fetches the published Bolza eigenvalues it compares with.


## Licences

- **Code** (`*.py`, `*.c`, `*.sh`, `Makefile`, `tools/`): MIT licence, `LICENSE`.
- **Data, figures and text** (every file listed in `DATA.md`, and the documentation): Creative Commons
  Attribution 4.0 International, `LICENSE-DATA`.

Copyright (c) 2026 the authors: Palaash Gang, Akshaj Agadi, Arjun Veluri, Jerry Wang, Jeremy Barreto, Aarin Chouthaiwale.

## How to cite

Cite the paper and the archived version of this repository:

- Palaash Gang, Akshaj Agadi, Arjun Veluri, Jerry Wang, Jeremy Barreto, Aarin Chouthaiwale, *How many heat invariants determine a hyperbolic orbifold?*, Preprint: to appear (arXiv).
- Palaash Gang, Akshaj Agadi, Arjun Veluri, Jerry Wang, Jeremy Barreto, Aarin Chouthaiwale, *Code and data for: How many heat invariants determine a hyperbolic orbifold?*. The archived version of this repository, with its DOI, is on Zenodo (see the DOI badge and link on the GitHub page).

`CITATION.cff` and `.zenodo.json` hold the same metadata, with the authors' ORCID iDs and affiliations.
The repository is https://github.com/pa1aash/heat-invariants-orbifolds.

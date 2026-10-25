# Trace: every checkable item of the paper

*How many heat invariants determine a hyperbolic orbifold?*. Numbering is that of the submitted version of the paper and its supplement (S-numbers). Each row names the item, the script that proves, checks or computes it, and the make target that runs it. A step name in the last column is the name printed by `make` (see `python3 tools/steps.py --list`).

| item | statement | script | make target (step) |
|---|---|---|---|
| Figure 1 | heat-kernel diagonal of $\mathcal O(2,8,8)$ and $\mathcal O(3,3,12)$ | `figures/src/F1.py` (Blender) | `make figures` (figures F1 F6: data assertions; figure F1: Blender render (scratch copy)) |
| Theorem 1.1(ii), Theorem 3.10, Corollary 3.11 | bounded cone orders: $M+1$ invariants suffice and are attained; sharp pairs; minimal areas $12\pi$, $36\pi$ | `theory/msep/verify.py` | `make verify` (bounded cone orders (Theorem 3.10)) |
| Theorem 2.3, Lemma 2.4 | terms of the trace formula; counting bound | `theory/locality/check_locality.py` | `make verify` (trace formula terms (locality)) |
| Lemma 2.5 | the constant $C_{\rm hyp}$ | `checks/proofs/check_thm12iii.py` | `make verify` (proof check check_thm12iii) |
| Lemmas 2.7, 2.9, 2.10, Proposition 2.8, Remark S8.1 | $\Phi_m$, cone polynomials $p_l$, moments $\alpha_k$, agreement with Uçar for $l\le40$ | `checks/proofs/check_lemma25.py`, `checks/proofs/check_remark412.py`, `theory/cone-coefficients/verify_cone_coefficients.py`, `theory/signatures/heat_structure.py` | `make verify` (proof check check_lemma25; proof check check_remark412; cone coefficients p_l; heat structure of a signature) |
| Section 2.3 | curvature $0$ and $+1$ | `theory/curvature/curvature_checks.py` | `make verify` (flat and spherical cases) |
| Conjecture 2.12, Proposition 2.14, Remark S8.2 | variable curvature: $\beta_{l,l+1}$ for $l\le7$, the $t^3$ coefficient, the linear part (833 cases) | `theory/varcurv/top_coefficient.py`, `theory/varcurv/twisted_mp.py`, `theory/varcurv/linear_part.py` | `make verify` (variable curvature top_coefficient; variable curvature twisted_mp; variable curvature linear_part) |
| Proposition 2.16 | second variation $\omega_n$; flat-cone examples | `theory/varcurv/quadratic_form.py`, `theory/varcurv/obstruction_check.py` | `make verify` (variable curvature quadratic_form; variable curvature obstruction_check) |
| Proposition 2.13 | rotationally symmetric cone points: the algebra of the proof, the limit $W_u\to f^2/r^2$, Gaussian moments, cosecant power sums, $\beta_{l,l+1}$ for $l\le8$ | `theory/varcurv/rotsym_check.py` | `make verify` (variable curvature rotsym_check) |
| Lemma 3.1, Theorem 3.2 | parity; $n$ invariants determine $n$ orders | `theory/audibility/verify_elimination.py` | `make verify` (Theorem 3.2 and Lemma 3.1: elimination) |
| Equation (orlando) | Orlando's formula, $n\le7$ | `theory/audibility/orlando_check.py` | `make verify` (Orlando's formula) |
| Theorem 3.3 | the linear system and its determinant, $n\le8$ | `theory/audibility/linear_system.py`, `theory/stability/lipschitz_e.py` | `make verify` (Theorem 3.3: the linear system; stability: Lipschitz constants) |
| Theorem 3.4 | $\{2,8,8\},\{3,3,12\}$ and $\{3,10,15,30\},\{4,5,21,28\}$ | `theory/audibility/sharpness_search.py` | `make verify` (Theorem 3.4 and Remark S8.4: sharpness search N=60) |
| Remark S8.4 | $n=4$, orders at most 440: 193 witnesses, 107 primitive; six not from a pencil | `checks/witnesses4/check_direct4.c`, `checks/witnesses4/check_direct4.py` | `make verify`, `make full` (degree-4 witnesses to 440: analysis; degree-4 witnesses to 440: search from scratch) |
| Remark S8.4 | pencil witnesses: 17 (14 primitive) to 130, 49 (30) to 220 | `checks/pairs/d1_pencil_counts.py` | `make verify` (pencil witness counts) |
| Remark S8.4 | $n=5$: no witness with orders at most 120 | `theory/audibility/sharpness_search.py` | `make full` (Remark S8.4: no witness for n=5 with orders <= 120) |
| Lemma 3.5 | padding and genus reduction | `theory/signatures/heat_structure.py` | `make verify` (heat structure of a signature) |
| Theorem 3.6, Figure 2 | separation of signatures | `theory/signatures/genus.py`, `figures/src/F3.py` | `make verify`, `make figures` (genus separation; figure F3) |
| Corollary 3.7, Theorem 3.16 | cone count $n\le{\rm Area}/\pi+4$; constructions | `theory/signatures/cone_count.py` | `make verify` (cone count) |
| Example 3.8 | $(1;15)$ and $(0;3,3,5,5)$ share exactly two invariants | `theory/pte/witnesses.py` | `make verify` (PTE witnesses) |
| Theorem 3.14 | the Descartes bound | `theory/pte/structure.py` | `make verify` (PTE structure) |
| Proposition 3.15, Proposition A.2 | doubling and shift identities | `theory/pte/structure.py` | `make verify` (PTE structure) |
| Theorem 3.17, Proposition 3.18, Lemma A.1 | growth bounds; constructions for $L\le7$ | `theory/pte/growth.py` | `make verify` (PTE growth) |
| Lemma 3.19 | no 4-configuration of size 10 of the stated form with entries at most 200 | `theory/pte/pencil_search.py` | `make verify` (PTE pencil_search) |
| Example S8.6, bounds on $T(L)$ (Table 2) | explicit pairs sharing exactly $L$ invariants | `theory/pte/witnesses.py`, `checks/pairs/d2_explicit_pairs.py` | `make verify` (PTE witnesses; explicit pairs (Table S1)) |
| Figure 3 | the 525 complete area classes and $K_{\rm mult}$ | `theory/signatures/area_classes.py`, `figures/gen/gen_f4_area_classes.py`, `figures/src/F4.py`, `checks/pairs/a1_fig2_kmult.py` | `make verify`, `make figures` (complete area classes; figure data F4 F7 F8; figure F4; K_mult on the complete area classes) |
| Remark S8.7 | no genus pair of size 8 with five-element side at most 220; real partners | `theory/pte/search_T3_modular.py`, `theory/pte/search_T3_control.py`, `theory/pte/real_shapes.py` | `make verify` (PTE T3 modular record; PTE T3 modular search to 220; PTE search_T3_control; PTE real_shapes) |
| Corollary 4.1 | dimension of the zero set | `theory/locality/check_locality.py` | `make verify` (trace formula terms (locality)) |
| Figure 4, Section 4 | the moduli family: 8 members, systoles, $\lambda_1$ | `numerics/moduli/validate_committed.py`, `figures/src/F6.py` (Blender) | `make verify`, `make figures`, `make full` (moduli family: committed data; moduli family: full validation; figures F1 F6: data assertions; figure F6: Blender render (scratch copy)) |
| Theorem 4.2 | the locality bound overestimates by $4.4\times10^6$ to $1.0\times10^{10}$ | `numerics/moduli/locality_ratio.py` | `make verify` (locality bound: overestimation ratios) |
| Section 4, Figure S2 | computed spectra and heat traces; the first two coefficients of the difference of traces recovered | `numerics/validate_committed.py`, `figures/src/F5.py` | `make verify`, `make figures`, `make full` (computed spectra: committed data; computed spectra: full validation; figure F5) |
| Section 4 | growth of the heat coefficients | `checks/proofs/check_remark212.py`, `theory/divergence/divergence.py` | `make verify`, `make full` (proof check check_remark212; divergence of the heat coefficients) |
| Figure 5 | tilings of the triangle orbifolds | `figures/src/F2.py` | `make figures` (figure F2) |
| Lemmas 5.4-5.6, Theorems 5.7-5.10 | chambers, first overlaps, thresholds | `theory/threshold/threshold.py` | `make verify` (first overlap and first collision (Section 5)) |
| Figure 6 | strata of the first two invariants | `figures/gen/gen_f7_strata.py`, `figures/src/F7.py` | `make figures` (figure data F4 F7 F8; figure F7) |
| Theorem 5.11 | the isolation of $\{(2,8,8),(3,3,12)\}$: descent, torsion, points | `checks/proofs/check_descent.py` | `make verify` (proof check check_descent) |
| Theorem 5.11 | rank 0 of $C_{27/2}$ (PARI `ellrank`) | `theory/diophantine/ranks.py` | `make verify` (rank of C_{27/2} (PARI ellrank)) |
| Proposition S3.1 | the front end; amplification factors | `theory/stability/front_end.py` | `make verify` (stability: front end) |
| Theorem S3.2, Proposition S3.3, Remark S3.4 | stability exponents; sharpness | `theory/stability/roots_holder.py`, `checks/proofs/check_sharpness.py` | `make verify` (stability: Hoelder exponents of roots; proof check check_sharpness) |
| Figure S1, Sections 6 and S3 | thresholds and recovery sweeps | `theory/stability/threshold.py`, `figures/gen/gen_f8_recovery.py`, `figures/src/F8.py` | `make figures`, `make full` (stability thresholds and certificates; figure data F4 F7 F8; figure F8) |
| Proposition B.2 | enveloping remainders | `checks/proofs/check_lemma25.py`, `checks/proofs/check_remark412.py` | `make verify` (proof check check_lemma25; proof check check_remark412) |
| Table S1 | the explicit pairs | `checks/pairs/d2_explicit_pairs.py` | `make verify`, `make tables` (explicit pairs (Table S1); supplement table S1 (pairs)) |
| Section S1 | the complete area classes: 33 946 signatures | `theory/signatures/area_classes.py`, `figures/gen/gen_f4_area_classes.py` | `make verify`, `make figures` (complete area classes; figure data F4 F7 F8) |
| Proposition S2.1 | 38 collision-free sums with $18\le S\le4800$; the collision witnesses | `checks/proofs/check_thm513.py` | `make verify` (proof check check_thm513) |
| Proposition C.1 | tangency gaps | `theory/threshold/threshold.py` | `make verify` (first overlap and first collision (Section 5)) |
| Table S2 | first overlap against first collision | `tools/make_tables.py` | `make tables` (supplement tables S2-S4) |
| Table S3 | the 83 triads with $10\le S\le18$ | `tools/make_tables.py` | `make tables` (supplement tables S2-S4) |
| Section S3 | Lemma S3.6 and Theorems S3.7-S3.9: conditioning and constants | `theory/stability/lipschitz_e.py`, `theory/stability/roots_holder.py` | `make verify` (stability: Lipschitz constants; stability: Hoelder exponents of roots) |
| Section S3 | $\zeta_3,\zeta_4,\zeta_5=1,79/3,14048/15$; ${\rm sech}^2U$ for $(2,2,2,2,3)$ | `checks/stability-constants/check_setup.py` | `make verify` (stability constants zeta_n and sech^2 coefficients) |
| Proposition S3.10 | the certificate | `theory/stability/threshold.py`, `checks/proofs/check_prop610.py` | `make verify`, `make full` (stability thresholds and certificates; proof check check_prop610) |
| Table S4 | thresholds, certificates, ratios | `tools/make_tables.py` (from `threshold_results.json`) | `make tables`, `make full` (supplement tables S2-S4; stability thresholds and certificates) |
| Section S4, Table S5 | method, levels, error estimates, validation, the fitted coefficients | `numerics/validate_committed.py` (data from `numerics/solve.py`, `numerics/validate.py`) | `make verify`, `make full` (computed spectra: committed data; computed spectra: full validation) |
| Section S4 | double-window recomputation agrees to $1.4\times10^{-14}$ | `numerics/rerun_double_window.py` | `make verify`, `make full` (double-window recomputation: record; computed spectra: double-window re-solve (NGSolve)) |
| Section S5, Table S6 | blind recovery from the computed spectra | `theory/stability/blind/blind_pipeline.py` | `make verify` (blind recovery from the computed spectra) |
| Section S6 | the moduli family: errors, traces, length spectra | `numerics/moduli/validate_committed.py`, `numerics/moduli/locality_ratio.py` | `make verify`, `make full` (moduli family: committed data; moduli family: full validation; locality bound: overestimation ratios) |
| Appendix E | the 2-isogeny descent of $C_{27/2}$ by hand | `checks/proofs/check_descent.py` | `make verify` (proof check check_descent) |

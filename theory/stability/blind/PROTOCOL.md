# Blind end-to-end protocol: from computed spectra to cone orders

Written and committed before the pipeline (`blind_pipeline.py`) is run on the eigenvalue data.
The commit that adds this file precedes the commit that adds any result.

## Question

Two computed spectra are given, specimen **A** and specimen **B**, in the following files:

- A: `numerics/data/eigenvalues_2-8-8_{N,D}.csv`
- B: `numerics/data/eigenvalues_3-3-12_{N,D}.csv`

The file names carry the true orders. The pipeline opens these four paths and never parses
the names. The label-to-truth map is used in exactly one place: the final comparison (step 9),
which runs after every estimate and verdict has been written.

We want to know two things:

- How precisely do the computed spectra fix the first three heat coefficients?
- Is that precise enough for the threshold of `the paper` (Theorem S4 / Proposition S5) to
  recover the cone orders exactly?

## What is assumed known (not specimen-specific)

- Each specimen is a pillow, the double of a hyperbolic triangle along its boundary. So it has
  genus 0 and exactly n = 3 cone points, and its spectrum is the union of the Neumann (N) and
  Dirichlet (D) spectra of the triangle (`the supplement` §1). This structural fact is
  used. The orders are not.
- K = −1.
- The universal front-end constants L and h0 for n = 3 (`front_end.py`): the cone polynomials
  p_0, p_1 and the smooth coefficients α_1, α_2. These are the same for every orbifold.

Nothing else is used: no area, perimeter, a₀, geodesic length or Weyl constant derived from
the orders. In particular, `numerics/heat_trace.py`'s tail bound is **not** used, because it
takes the exact area and a₀ from `numerics/theory.py`.

## Steps

1. **Traces.** For each specimen, Z(t) = Σ_N e^{−λt} + Σ_D e^{−λt}, summed over all rows of the
   two CSVs, column `lambda`. The grid is 160 log-spaced points in t ∈ [0.0015, 0.02].
   - The eigenvalue error is the per-row `err_estimate` (primary), propagated as
     t Σ_j err_j e^{−(λ_j − err_j)t}, the same formula as `numerics/heat_trace.py`.
   - The run is repeated with `err_conservative` (secondary).
2. **Truncation tail (blind).**
   - For each triangle problem, fit N(λ) ≈ aλ + b√λ + c by least squares to the computed
     counting function on the upper half of the computed range.
   - Bound the tail Σ_{λ>Λ} e^{−λt} with the closed form of `numerics/heat_trace.py`. Use
     a ↦ 1.01|a|, b ↦ |b|, and C_up = 2·max(excess of N over aλ + b√λ) + 10, all taken from
     this blind fit.
   - The tail bound is added to the data error.
3. **Fits.** Fit y(t) = tZ(t) = Σ_{j=0}^{d} c_j t^j by least squares, so c₀ = H₋₁, c₁ = H₀ and
   c₂ = H₁.
   - Windows: [0.0015,0.004], [0.0015,0.006], [0.0015,0.008], [0.002,0.006], [0.002,0.008],
     [0.002,0.01], [0.003,0.01], [0.0015,0.01], [0.0015,0.012].
   - Degrees: d = 3..12, with at least 3(d+1) grid points in the window.
   - For each coefficient and configuration:
     - *data error* = |pinv|·(error of y): the worst case over all data within the error
       bound;
     - *model error* = |c_j(d) − c_j(d−1)| in the same window;
     - u = max(data, model).
   - For each coefficient separately, the estimate is the configuration with the smallest u.
     This is the objective order selection of `the supplement` §5.
   - **Honest error bar:** the final uncertainty is
     U_j = max(u_best, ½·range of c_j over the 8 configurations with the smallest u).
4. **Area.** Area = 4π H₋₁, with uncertainty 4π U₋₁.
5. **Recovery.**
   - Ĩ = L⁻¹(H̃ − h0) for n = 3.
   - Solve the Theorem B system M(Ĩ) ẽ = b(Ĩ) in exact arithmetic, with the estimates
     converted to rationals.
   - Take the three roots of z³ − ẽ₁z² + ẽ₂z − ẽ₃ and round their real parts. The result is
     the candidate m̂.
6. **A-posteriori certificate (rigorous given the error bars).**
   - Check that m̂ is an admissible multiset: integers ≥ 2 with 1 − R > 0.
   - Run the certifier of Proposition S5 at m̂ with componentwise radius
     ρ_ν = |H̃_ν − H_ν(m̂)| + U_ν.
   - If it certifies, then |H(m_true) − H(m̂)| ≤ ρ. So the exact data of the true multiset lie
     in the certified box around H(m̂), and the recovery map applied to them returns m̂. It
     also returns m_true, by Theorem A / B. Hence m_true = m̂.
   - Verdict: **CERTIFIED** or **NOT CERTIFIED**.
   - The argument assumes the error bars U are valid bounds. They are worst-case data bounds
     plus an empirical model-error term, not certified enclosures.
7. **Exhaustive cross-check (independent of rounding).** Enumerate every hyperbolic integer
   triple 2 ≤ p ≤ q ≤ r with 1/p + 1/q + 1/r < 1 whose exact H₋₁ and H₀ lie within kU of the
   estimates, for k = 1 and k = 3.
   - The enumeration is finite: R is fixed to within the H₋₁ box, and then H₀ bounds
     P₁ = p + q + r.
   - List the *two-coefficient candidates*, then the subset whose H₁ is also within kU, the
     *three-coefficient candidates*.
8. **Two coefficients cannot separate (stated before looking).**
   - (a) Structural: by T1, H₋₁ and H₀ are functions of (R, P₁) alone. Any two triples with
     equal (R, P₁) have identical first two coefficients, at any precision.
   - (b) Empirical: report H̃_ν(A) − H̃_ν(B) for ν = −1, 0, 1 with uncertainty U(A) + U(B).
   - (c) Report whether the two-coefficient candidate sets of A and B intersect.
9. **Final comparison.** Only now is the true multiset read from the file label and compared
   with m̂. Report:
   - recovered exactly / not;
   - certified / not;
   - the true values of H₋₁, H₀, H₁ and the deviation of each estimate in units of U. This is
     a check on the honesty of the error bars, not a step of the pipeline.

## Pre-registered success criteria

- **Exact recovery from three coefficients** means both of the following:
  - step 5's rounding gives the true multiset;
  - step 6 certifies.

  If the rounding is right but the certificate fails, the result is reported as
  "recovered, not certified".
- **"Two coefficients cannot separate them"** is confirmed if both of the following hold:
  - (8a) holds, which is pure algebra;
  - the step-7 two-coefficient candidate set (k = 1) of each specimen contains at least two
    triples, including the candidate recovered for the other specimen.
- **Error-bar honesty:** each |estimate − truth| ≤ U_ν. Any violation is reported
  prominently, not explained away.

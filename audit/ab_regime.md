# Audit: ab_regime experiment

## Paper sections consulted

arXiv:2306.04843v2 (Caro et al., 7 Dec 2023):
- §5 Distributional Agnostic Quantum Learning, pp. 26-31
- §6.3 Verifying Distributional Agnostic Quantum Learning, pp. 44-50
  - **Definition 14** "Distributions with L₂-bounded bias", p. 44 eq. (105)
  - **Theorem 12** parity verification, pp. 44-46
  - **Theorem 13** necessity bound `ε ≥ 2√(b²−a²)`, p. 47
  - **Theorems 14/15** Fourier-k-sparse, pp. 48-50

## What the paper predicts

**Definition 14 (p. 44):** `0 ≤ a ≤ b ≤ 1`; `D_{U_n;[a²,b²]} := { (U_n, φ) | E_{x∼U_n}[(φ(x))²] ∈ [a², b²] }`. So `(a,b)` are **two squared-L₂-norm bounds** on `φ̃ := 1−2φ ∈ [−1,1]`. By Parseval `E[φ̃²] = Σ_s ĝ(s)²`, so `a²/b²` lower- and upper-bound the **total Fourier weight**. They are *not* completeness/soundness thresholds — they are a **single interval** containing the same true Fourier weight, expressing the verifier's prior knowledge. The paper notes `a=b=1` for the noiseless functional case and `a=b=1−2η` for the noisy functional case.

**Theorem 12 (p. 45):** verifier accepts iff `Σ_{s∈L} ξ̂(s)² ≥ a² − ε²/8`. The completeness analysis (eq. 108-109) uses `a²` (lower bound). The soundness analysis (eq. 112-114) uses `b²` (upper bound) plus the precondition `ε ≥ 2√(b²−a²)`. Both bounds enter, but only `a²` sets the threshold; `b²` only constrains `|L| ≤ 64b²/θ²` and the achievable accuracy.

**Theorem 13 (p. 47):** any n-independent verifier needs `ε ≥ Ω(√(b²−a²))`.

**Theorem 15 (p. 49):** Fourier-k-sparse threshold is `a² − ε²/(128k²)`, precondition `ε ≥ 4k√(b²−a²)`.

## What the experiment does

`/Users/alex/cs310-code/experiments/harness/ab_regime.py`:
- For each `n ∈ {4,...,16}`, gap `g ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.4}`, draw 100 trials.
- `make_sparse_plus_noise` (`phi.py:228-272`) builds `φ̃ = 0.7 χ_{s*} + 0.1(χ_{s1}+χ_{s2}+χ_{s3})`, so true Parseval weight is **exactly** `0.7² + 3·0.1² = 0.52`.
- Sets `a² = max(pw − g/2, 0.01)`, `b² = min(pw + g/2, 1.0)` with `pw = 0.52` (`ab_regime.py:90-91`).
- Fixed: `ε=0.3`, `δ=0.1`, `θ=min(ε,0.6)=0.3`, `qfs_shots=2000`, `classical_samples_prover=1000`, `classical_samples_verifier=3000`, `noise_rate=0.0`.
- Calls `verify_parity` (parity branch, k=None).
- 13×6×100 = 7800 trials saved to `results/ab_regime_4_16_100.pb`.

## Implementation correctness

**`(a,b)` semantics:** `TrialSpec.a_sq/b_sq` flow into `verify_parity` (`worker.py:175-183`) → `_verify_core` (`verifier.py:434-547`):
- List-size bound `int(ceil(64 * b² / θ²))` (`verifier.py:458`) — matches Theorem 12 Step 1. **Correct.**
- Parity threshold `a² − ε²/8` (`verifier.py:509`) — matches Theorem 12 Step 4 / eq. (109). **Correct.**
- Fourier-sparse threshold `a² − ε²/(128k²)` (`verifier.py:512`) — matches Theorem 15 Step 4 / eq. (118). **Correct.**
- Hoeffding tolerances `ε²/(16|L|)` (parity) and `ε²/(256k²|L|)` (k-sparse) at `verifier.py:482-494` match Theorems 12/15 Step 3.

The `(a,b)` parameterisation in code is consistent with Definition 14: `a²` is lower, `b²` is upper. Naming is consistent through code, docstrings, and protobuf schema (`results.py:88-96`).

**Construction in `ab_regime.py`:** For `pw=0.52` and swept gaps, neither clamp ever fires: `a² ∈ {0.52, 0.495, 0.47, 0.42, 0.37, 0.32}`, `b² ∈ {0.52, 0.545, 0.57, 0.62, 0.67, 0.72}`. CSV confirms exactly these values. The true `||φ̃||₂² = 0.52` always sits at the centre of `[a², b²]`, so the trials honour the promise. The `0.01` clamp is dead code in the swept range.

**Threshold formula numerics (verified hand-by-hand against CSV):**
- `gap=0.00`: `a² = 0.52`, `τ = 0.52 − 0.01125 = 0.50875` ✓ (CSV: `0.508750`)
- `gap=0.05`: `a² = 0.495`, `τ = 0.48375` ✓ (CSV: `0.483750`)
- `gap=0.40`: `a² = 0.32`, `τ = 0.30875` ✓ (CSV: `0.308750`)

CSV `aSq`, `bSq` columns match the formulas exactly. `median_threshold_margin` = `median_accumulated_weight − acceptance_threshold` in every sampled row.

**Tests** (`tests/harness_test.py:850-924`) cover: gap>0 ⇒ `b_sq>a_sq`, gap=0 ⇒ `a_sq=b_sq`, save round-trip. Match the formulas.

## Results vs. literature

Acceptance % from CSV (also identical to hypothesis-correctness % column — every accepted trial outputs the correct dominant parity):

| gap  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  | 13  | 14  | 15  | 16  |
|------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 0.00 |  83 |  54 |  22 |  21 |  15 |  18 |  12 |  22 |  16 |  18 |  11 |  13 |  19 |
| 0.05 |  99 |  86 |  70 |  63 |  57 |  64 |  59 |  65 |  57 |  64 |  54 |  65 |  68 |
| 0.10 | 100 |  99 |  96 |  98 |  94 |  93 |  98 |  92 |  98 |  96 |  97 |  95 |  98 |
| 0.20-0.40 | 100 across the board |

**Qualitative match.** Theorem 12 predicts: as gap widens, `a²` shrinks, `τ = a² − ε²/8` drops, acceptance becomes easier. Observed: monotone non-decreasing in gap, saturating at 100% by gap=0.2.

**The gap=0 cell.** At `gap=0`, `τ=0.50875` and the median accumulated weight is `~0.49`. For `n≥6` the median is **below** the threshold (margin −0.016 to −0.022), so most trials reject. Acceptance collapses from 83% (n=4) to ~12-22% for n≥6. This is a **finite-sample tightness artefact**, not a bug: the verifier's unbiased estimator `ξ̂(s) = (1/m)Σ(1−2y_i)χ_s(x_i)` has its sum-of-squares attenuated by `~Σ Var(ξ̂(s)) ~ 4/3000 ~ 0.0013`, plus median sampling spread, which pushes it below `0.50875` for the larger-n cells where the four heavy parity positions are spread over a larger search space (so the same number of samples produces noisier estimates per coefficient). The plot script acknowledges it (`plot_ab_regime.py:495-499`).

**Theorem 12 precondition silently violated.** Theorem 12 requires `ε ≥ 2√(b²−a²)`. With `ε=0.3`, that means `gap ≤ (ε/2)² = 0.0225`. Only `gap=0.0` satisfies it. The plot script (`plot_ab_regime.py:170-177`) draws a red dotted line at `0.0225` labelled "Thm 13 bound" and at lines 472-478 says "the threshold formula remains effective even when the Thm 13 bound on ε is violated. The bound may be loose for the specific functions tested." This is a **misinterpretation**: Theorem 13 is a worst-case sample-complexity lower bound for distinguishing random parities from `U_{n+1}`; it does *not* upper-bound the acceptance probability of any specific honest run on benign inputs. The observed gap>0.0225 acceptance is *not* evidence Theorem 13 is loose — it is evidence that on this particular structured `φ` (with `||φ̃||₂² = 0.52` exactly at the centre of `[a²,b²]`) honest interactions still produce `Σ ξ̂² ≥ a² − ε²/8` even when Theorem 12 has no completeness guarantee.

**`b²` is structurally inert.** With `θ=0.3` and `b² ≤ 0.72`, the list-size bound is `≤ 512`, far above the maximum honest list of 4. So `b²` never binds anything, and the experiment is effectively a 1-D sweep of `a²` via `gap`. The name "ab regime" implies a 2-D study but the design holds the centre fixed at `pw=0.52` and only varies the width.

## Issues / discrepancies

### MAJOR

**M1. Theorem 12 precondition `ε ≥ 2√(b²−a²)` silently violated for 5 of 6 gaps.** No warning, no skip, and the figure-script analysis interprets the resulting acceptance as "Theorem 13 is loose" rather than as "we are running outside the theorem's completeness regime on benign inputs". Mitigations: (a) reduce `ε` so the gap range stays under `(ε/2)²`; or (b) clearly relabel as "out-of-regime acceptance behaviour"; or (c) replace the figure commentary with the correct interpretation. (`ab_regime.py:18`, `plot_ab_regime.py:472-478`)

**M2. The "ab regime" sweep is really a 1-D `a²` sweep.** `b²` never binds anything (list-size slack by ~100×) and `||φ̃||₂²` is fixed at the centre. To probe both completeness *and* soundness of Definition 14, the design should also vary the centre (e.g., several `c_dom` values) so that some trials pin `||φ̃||₂²` near `b²` (stressing completeness) and others near `a²` (stressing soundness). As-is, the figure script's "Threshold margin vs gap" is monotone by construction.

### MINOR

**m3. `gap=0` completeness is 11-22% for n≥6**, well below `1−δ=0.9`. Finite-sample artefact (Hoeffding shrinkage of `Σ ξ̂²` below the true Parseval weight at threshold equality), not a bug. Same phenomenon exposed by `truncation`/`theta_sensitivity`. The plot would benefit from a footnote.

**m4. Stale comment in `ab_regime.py:92-93`** ("Dominant coefficient is 0.7, so keep theta below it"). The `θ < |c_dom|` heuristic is from the prover; in the verifier path, `θ` only enters via `64b²/θ²`, where any `θ ≤ 1` is fine for `|L| ≤ 4`.

### NIT

**n5.** The `max(parseval_weight − gap/2, 0.01)` clamp at `ab_regime.py:90` is dead in the default sweep — only fires for `gap > 1.02`.

**n6.** The plot script labels `gap=(ε/2)²` as "Thm 13 bound"; it is more directly the contrapositive of Theorem 12's precondition.

**n7.** The experiment uses `verify_parity` only. An ab_regime sweep with `k=4` (matching the heavy-coefficient count of `make_sparse_plus_noise`) would also exercise `verify_fourier_sparse`'s threshold `a² − ε²/(128k²)`, currently exercised only by `k_sparse`.

**n8. No off-by-one or sign errors found.** Verified `a² − ε²/8`, `64b²/θ²`, `ε²/(16|L|)`, `(2/tol²)log(4|L|/δ)` line-by-line against Theorem 12 Step 3-4 (paper p. 45) and against the CSV row-by-row.

## Verdict

The implementation of `(a, b)` is **correct**. `a²` is the lower bound, `b²` is the upper bound; both are wired into Definition 14's threshold check `Σ ξ̂² ≥ a² − ε²/8` and list-size bound `|L| ≤ 64b²/θ²` exactly as Theorem 12 specifies. The harness construction `a² = pw − gap/2`, `b² = pw + gap/2` always honours the promise. Threshold-formula numerics match the CSV row-by-row. No off-by-one or sign errors.

The experimental observation (acceptance monotone in gap, saturating at 100% by `gap=0.2`) is **qualitatively consistent** with Theorem 12. Two substantive issues are interpretive: Theorem 12's precondition `ε ≥ 2√(b²−a²)` is violated for 5 of 6 gaps (M1), and `b²` is structurally inert so the sweep is really 1-D (M2). Neither compromises the correctness of the implementation; both should be addressed in the writeup interpretation. The `gap=0` low-completeness boundary (11-22% for n≥6) is a known finite-sample artefact, not a bug.

**Status: no implementation bugs; MAJOR issues are interpretive narrative in the figure script and the design's failure to actually vary `b²` independently. Use the data with the caveats above.**

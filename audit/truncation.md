# Audit: truncation experiment

## Paper sections consulted

Read directly from `/Users/alex/cs310-code/papers/classical_verification_of_quantum_learning.pdf`:
- §1.2 framework (pp. 4–5)
- §5.1 Theorem 5 / Corollary 5 (pp. 26–28)
- §6.1 Theorems 7/8 (pp. 35–42)
- §6.2 noisy case (p. 43)
- §6.3 Theorem 12 (p. 45) and Theorem 13 (pp. 47–48)
- Definition 14 (p. 44)

## What the paper predicts

**Theorem 12 (p. 45)** specifies that for `D ∈ 𝔇^func_{U_n;≥ϑ} ∩ 𝔇_{U_n;[a²,b²]}`, with `ε ≥ 2√(b²−a²)`, `ϑ ∈ (2^(−n/2−3), 1)`, `δ ∈ (0,1)`:
1. List bound `|L| ≤ 64 b²/ϑ²`.
2. Verifier uses `O(|L|² log(|L|/δ)/ε⁴)` classical samples to get `(ε²/(16|L|))`-accurate `ξ̂(s)` with prob ≥ 1−δ/2.
3. Accept iff `Σ ξ̂(s)² ≥ a² − ε²/8`.
4. Output `s_out · x` where `s_out = argmax|ξ̂|`.

## What the experiment does

`/Users/alex/cs310-code/experiments/harness/truncation.py`:
- **Hard-codes `target_s = 1`** (line 87) — same parity for every trial of every cell at every n.
- η = 0.15, so a² = b² = 0.49 (line 80).
- Sweeps ε ∈ {0.1, 0.2, 0.3, 0.4, 0.5} × m_V ∈ {50, 100, 200, 500, 1000, 3000}.
- θ = min(ε, 0.5) (line 103).
- Hard-codes prover `qfs_shots = 2000`, `classical_samples_prover = 1000` (lines 19–20).
- `classical_samples_verifier = v_samples` overrides the analytic Hoeffding count via `worker.py:175–183` → `verifier.py num_samples=…`.
- 11 .pb files at `results/truncation_{n}_{n}_100.pb` for n = 4..14, each ~530 KB containing 3000 trials.

## Implementation correctness

All match Theorem 12 verbatim:
- Threshold `a² − ε²/8` at `ql/verifier.py:509`.
- List cap `64 b²/θ²` at `ql/verifier.py:458`.
- Per-coefficient tolerance `ε²/(16|L|)` at `ql/verifier.py:482`.
- Hoeffding sample formula `m ≥ (2/tol²) log(4|L|/δ)` at `ql/verifier.py:493` (correct for range-2 estimator).
- Estimator `(1/m) Σ (1−2y_i)·χ_s(x_i)` at `ql/verifier.py:599–610`.
- `a² = (1−2η)²` at `experiments/harness/truncation.py:80`.

## Results vs. literature

**The experiment is sub-Hoeffding everywhere.** Computing the analytic minimum from `verifier.py:493` with `|L| = 1`:

| ε | analytic m_required | largest grid m_V |
|---|---|---|
| 0.1 | 18,883,000 | 3,000 |
| 0.2 | 1,180,188 | 3,000 |
| 0.3 | 233,124 | 3,000 |
| 0.4 | 73,762 | 3,000 |
| 0.5 | 30,213 | 3,000 |

The grid is 3–4 orders of magnitude below the Theorem 12 prescription.

**The mechanism dominating the curves is squaring bias, not Hoeffding tail.** The estimator `ξ̂` has `E[ξ̂²] = φ̂² + Var(ξ̂) = 0.49 + 0.51/m`. The acceptance margin is `a² − τ = ε²/8 ∈ {0.00125, 0.005, 0.01125, 0.02, 0.03125}`. For ε=0.1 the bias is 8× the margin at m=50 (almost-sure accept) but 7× *smaller* than the margin at m=3000 (≈50% accept driven purely by fluctuation symmetry). This explains the inverted pattern in `truncation_summary.csv` where many `accept@50 > accept@3000` rows appear (for ε=0.1: n=4 0.97→0.59, n=10 0.96→0.64, n=11 0.68→0.57, n=12 0.62→0.58, n=13 0.57→0.54, n=14 0.59→0.47).

**The n-dependence is the prover, not the verifier.** Theorem 12's verifier sample bound is n-independent. Yet `accept_at_3000` for ε=0.1 swings 0.59→0.94→0.47 as n goes 4→8→14. The actual cause is the prover's QFS hard-coded at `qfs_shots=2000` against a Lemma-3 requirement of `≈ 15100/ε⁴` post-selected samples. For ε=θ=0.1 that's ~150M samples needed; the prover gets ~1000. So at large n + small ε the prover often fails to put `s = 1` into `L`. The plot script (`plot_truncation.py:101`) collapses `accepted` and never separates "prover failed to find target" from "verifier rejected on weight".

## Issues / discrepancies

### MAJOR

**M1. Docstring overclaim** (`truncation.py:32–47`): claims to "map the tradeoff surface predicted by Theorem 12". The grid is sub-Hoeffding by 3–4 orders of magnitude; the "knees" for ε ≤ 0.3 are squaring-bias artefacts, not Theorem-12 boundary measurements. Acceptance *decreasing* with budget would be impossible if the experiment were in the asymptotic regime the docstring claims.

**M2. Conflated failure modes**: prover budget is also far below the Corollary 5 bound for small ε. The `.pb` files record `prover_found_target` and `outcome` separately (`worker.py:215–219`) but `plot_truncation.py:101` collapses them. The minimum-viable-budget figure is therefore measuring `P[prover finds target AND verifier accepts]` and presenting it as a verifier-truncation curve. **Recommendation:** plot acceptance conditional on `prover_found_target == True`, or split into two panels.

### MINOR

**m3. Single fixed instance**: `target_s = 1` hard-coded; "100 trials" are 100 samples of the same φ. With n=14 there are 16,384 parities and the experiment uses one. n-trends are measured on a single trajectory of parities.

**m4. Missing θ precondition assertion**: Theorem 12 requires θ > 2^(−n/2−3) (e.g., 0.0884 for n=4). The experiment never violates this but `_verify_core` doesn't check.

**m5. Plot script not shard-safe**: `plot_truncation.py:60–78` reads `truncation_{n}_{n}_100.pb` directly; SLURM shards `truncation_*_shard*.pb` would require manual merge first. Fine for current artefacts (single-shard runs) but fragile.

**m6. Tight-margin caveat absent from docstring**: with a²=b²=0.49, the threshold is 0.001–0.03 below a². The whole experiment lives in this thin slab; this should be noted.

### NIT

**n7. Hoeffding comment** (`verifier.py:488–489`): "P[…] ≤ 2·exp(−2·m·tol²/4)" doesn't simplify the range-2 substitution; implementation is correct.

**n8. Plot recomputes τ** (`plot_truncation.py:393`): uses `A_SQ - eps**2/8` instead of reading the recorded `acceptance_threshold` from each trial.

**n9. `theta = min(eps, 0.5)`** at `truncation.py:103` is silently capping θ; merits a comment about the list-size implication.

## Verdict

**Verification protocol code is correct. Experimental design has MAJOR concerns**: the experiment is mislabelled relative to Theorem 12 and the figures conflate two distinct failure modes. The grid is sub-Hoeffding by 3–4 orders of magnitude, so the "knees" are squaring-bias artefacts rather than Theorem-12 boundary measurements. The minimum-viable-budget figure conflates prover-side QFS truncation with verifier-side sample truncation. **Recommended fixes (cheap, high-value):**
1. Re-caption `min_viable_budget` and `sample_budget_knee` figures as "non-asymptotic / sub-Hoeffding regime" rather than "Theorem 12 boundary".
2. Split the heatmap into prover-success-rate and conditional verifier-accept-rate panels.
3. Add an additional grid of m_V values that brackets the analytic `2/(ε²/16)² · log(4|L|/δ)`, so the experiment actually crosses the Theorem 12 boundary somewhere.
4. Randomise `target_s` per trial.

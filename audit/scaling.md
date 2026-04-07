# Audit: scaling experiment

## Paper sections consulted

Read directly via the Read tool with `pages=`:
- pages 1-5: title page, table of contents, introduction (Section 1, 1.1)
- pages 5-12: Section 1.2 "Overview of Main Results" (Theorems 1-3), 1.3-1.6
- pages 19-34: Section 4 (Functional Agnostic Quantum Learning), Lemmas 1-7, Section 4.3 (QSQ), Section 5 (Distributional Agnostic Quantum Learning), Theorem 5, Corollaries 1-9
- pages 35-50: Section 6 (Classical Verification of Agnostic Quantum Learning), Theorems 7-10 (functional case), Definitions 11-14, Theorems 11-16 (distributional case)

## What the paper predicts

The protocol exercised by the experiment is the classical-verifier / quantum-prover interactive proof for agnostic parity learning. With the verifier on classical random examples and the prover on mixture-of-superpositions examples, the relevant theorem for honest single-parity targets is:

**Theorem 12 (paper p. 45)** — "The class of n-bit parities is efficiently proper 1-agnostic verifiable w.r.t. D^{func}_{U_n,>=θ} ∩ D_{U_n,[a²,b²]} by a classical verifier V with access to classical random examples interacting with a quantum prover P with access to mixture-of-superpositions quantum examples." Resource bounds:

- prover: O(log(1/(δθ²)) / θ⁴) copies of ρ_D
- verifier: O(b⁴ log(1/(δθ²)) / (ε⁴ θ⁴)) classical random examples
- one round of communication, message ≤ O(n/θ²) bits

**Crucially, the verifier's classical sample complexity is n-independent in the leading term**; n only enters through poly-log factors in time/memory. This is the central scaling claim the experiment is meant to test.

**Theorem 12 protocol (p. 45)**:
1. V asks P for a list `L` with `|L| ≤ 64 b²/θ²`.
2. P runs Corollary 5, sends `L = { s : |~φ(s)| ≥ θ/2 }`.
3. V uses O(|L|² log(|L|/δ) / ε⁴) classical examples to obtain `(ε²/(16|L|))`-accurate estimates `^ξ(s)` for all s ∈ L.
4. Accept iff `Σ_{s∈L} ^ξ(s)² ≥ a² − ε²/8`. Output `s_out = argmax_{s∈L} ^ξ(s)`, hypothesis `h(x) = s_out · x`.

**Completeness/soundness (proof pp. 45-46)**: by union bound the joint failure probability is ≤ δ, so `Pr[accept] ≥ 1 − δ` when the prover is honest and the distribution satisfies `D ∈ D^{func}_{U_n,>=θ} ∩ D_{U_n,[a²,b²]}`.

**Corollary 5 (p. 27)**: prover needs `O(log(1/(δε²))/ε⁴)` MoS copies, with the constraint `ε > 2^{-(n/2-2)}` (satisfied for ε=0.3 once n ≥ 7).

**Theorem 5(i) (p. 26)**: post-selection rate is exactly 1/2.

For functional single parities `φ(x) = s*·x`, `~φ = χ_{s*}` so `^~φ` is concentrated on a single index, `a² = b² = E[~φ²] = 1`, the unique heavy coefficient is s* with `|^~φ(s*)| = 1`, and the protocol should always accept and recover s* provided budgets are large enough.

## What the experiment does

### Driver
`/Users/alex/cs310-code/experiments/harness/scaling.py` — `run_scaling_experiment(n_range=range(4,13), num_trials=20, epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples_prover=1000, classical_samples_verifier=3000, base_seed=42, ...)` builds, per (n, trial), a `TrialSpec` with:

- `phi` from `make_random_parity(n, rng)` (`experiments/harness/phi.py:35`), drawing `s* ~ Uniform({1,...,2^n − 1})` and producing `φ(x) = s*·x mod 2`. So `~φ = χ_{s*}`, `a² = b² = 1`.
- `noise_rate = 0`, `epsilon = delta_internal = theta = 0.3` (line 87 sets `theta=epsilon`).
- `a_sq = b_sq = 1.0`, `qfs_shots = 2000`, `classical_samples_prover = 1000`, `classical_samples_verifier = 3000`.

The CLI default `--n-min/--n-max` is `(4, 10)` but the on-disk run sweeps `n ∈ [4, 16]` with 100 trials per n (filename `scaling_4_16_100.pb`, 1300 trials total).

### Per-trial worker
`/Users/alex/cs310-code/experiments/harness/worker.py:97-235` reconstructs `MoSState`, runs `prover.run_protocol(...)`, runs `verifier.verify_parity(...)` with `seed+1_000_000` (separate RNG for soundness independence), and records `total_copies = msg.total_copies_used + classical_samples_verifier = 2000 + 1000 + 3000 = 6000` (`worker.py:224`).

### Prover (`MoSProver.run_protocol`)
`/Users/alex/cs310-code/ql/prover.py:230-364`. Steps:
1. `tau = theta**2 / 8` (line 316). Auto-`m_postselected = ceil(2 log(4/δ) / tau²)` (line 318), `qfs_shots = ceil(2.5 · m_post)` (line 321) — both bypassed by the override.
2. `QuantumFourierSampler.sample(shots=2000, mode='statevector')`.
3. Builds `SpectrumApproximation` with `extraction_threshold = theta**2 / 4` (line 428). Matches Corollary 5 step "L = {s : q~_m(s,1) ≥ ε²/4}" exactly when θ=ε.
4. Sorts by empirical weight, truncates to Parseval bound `ceil(16/θ²)` (= 178 for θ=0.3, never triggered).
5. Estimates Fourier coefficients on L from 1000 classical samples via `mean[(1−2y)·χ_s(x)]`, clipped to [-1,1]. Unbiased estimator of `^~φ(s)`.

### QFS sampler
`/Users/alex/cs310-code/mos/sampler.py:304-343` (statevector mode). Per shot: samples `f ~ F_D`, builds `|ψ_f⟩`, applies `H^{n+1}`, samples one outcome. `_postselect` (line 500-534) keeps shots with leftmost (Qiskit big-end) bit '1' (the label qubit at position n). Standard and correct.

### Verifier (`MoSVerifier.verify_parity` / `_verify_core`)
`/Users/alex/cs310-code/ql/verifier.py:300-547`:
- **List size check** (line 458): `list_size_bound = ceil(64 * b_sq / theta**2)` = 712 for θ=0.3, b²=1. **Matches Theorem 12 Step 1 exactly.**
- **Per-coefficient tolerance** (line 482): `epsilon**2 / (16 * |L|)`. **Matches Theorem 12 Step 3 exactly.**
- **Auto sample budget** (lines 487-495): Hoeffding `m ≥ (2/tol²) log(4|L|/δ)`. Bypassed by `num_samples=3000`.
- **Independent estimation** (`_estimate_coefficients_independently`, lines 553-611): draws fresh (x,y), computes `mean[(1-2y) χ_s(x)]`. Uses verifier's own RNG, so soundness is information-theoretic as in the paper.
- **Acceptance threshold** (line 509): `tau_accept = a_sq - epsilon**2/8 = 1 − 0.09/8 = 0.98875`. **Matches Theorem 12 Step 4 exactly.**
- **Hypothesis** (line 633): `s_out = argmax_{s∈L} |^ξ(s)|`. **Matches Theorem 12 Step 4.**

### Result file and figures
`/Users/alex/cs310-code/results/scaling_4_16_100.pb` — 1,300 trials (13 n × 100). Per-n statistics from `/Users/alex/cs310-code/results/figures/scaling/scaling_summary.csv`, generated from the same `.pb` by `plot_scaling.py`.

```
n  acc% [CI]    corr% [CI]   med|L|  med ps_rate  med copies  med time
 4  100 [96.3]  100  [96.3]   1      0.4985       6000        0.357 s
 5  100 [96.3]  100  [96.3]   1      0.4992       6000        0.406 s
 6  100 [96.3]  100  [96.3]   1      0.4998       6000        0.467 s
 7  100 [96.3]  100  [96.3]   1      0.4960       6000        0.552 s
 8  100 [96.3]  100  [96.3]   1      0.5010       6000        0.689 s
 9  100 [96.3]  100  [96.3]   1      0.4985       6000        0.920 s
10  100 [96.3]  100  [96.3]   1      0.5010       6000        1.335 s
11  100 [96.3]  100  [96.3]   1      0.5010       6000        2.140 s
12  100 [96.3]  100  [96.3]   1      0.5015       6000        3.801 s
13  100 [96.3]  100  [96.3]   1      0.5022       6000        9.066 s
14  100 [96.3]  100  [96.3]   1      0.5002       6000       73.644 s
15  100 [96.3]  100  [96.3]   1      0.5010       6000      946.410 s
16  100 [96.3]  100  [96.3]   1      0.4980       6000     1202.786 s
```
(`[CI]` is the Wilson 95% lower bound for k=n=100.)

`plot_scaling.py` produces five artefacts. The relevant ones for this audit are:
- `completeness_vs_n.{pdf,png}`: acceptance + correctness vs n with the `1−δ=0.9` reference line from Theorem 12 (line 200).
- `postselection_vs_n.{pdf,png}`: median post-selection vs n with reference line at 1/2 from Theorem 5 (line 232).
- `resource_scaling.{pdf,png}`: median total copies (log) and wall-clock time (log) vs n, plus a fitted theoretical curve `C · n · log(1/(δθ²)) / θ⁴` (line 270).
- `list_size_vs_n.{pdf,png}`: with reference lines at `4/θ² = 44.4` (Theorem 7 / Lemma 8 Parseval bound) and `1` (single parity).

## Implementation correctness

### Looks correct because:

- QFS circuit and post-selection match Theorem 5. Empirical post-selection rate `0.498-0.502` across all n agrees with the predicted 1/2 to sub-percent precision.
- Prover spectrum extraction threshold `θ²/4` matches Corollary 5 ("if `q~_m(s,1) ≥ ε²/4` then s ∈ L", page 28) exactly when θ=ε.
- Verifier list-size bound `64 b²/θ²` matches Theorem 12 Step 1 (paper p. 45). The repository docstring at `verifier.py:454-457` correctly explains the choice — the implementation runs the *distributional* protocol uniformly, even on noiseless functional inputs, which is conservative but valid (since `64 b²/θ² ≥ 4/θ²` whenever `b² ≥ 1/16`).
- Acceptance threshold `a² − ε²/8`: matches Theorem 12 Step 4 exactly.
- Per-coefficient tolerance `ε²/(16 |L|)`: matches Theorem 12 Step 3 exactly.
- Verifier draws *fresh* samples with a separate RNG (`worker.py:161`), preserving the independence the paper relies on for information-theoretic soundness (p. 46).
- The estimator `mean[(1−2y) χ_s(x)]` is the unbiased sample-mean estimator of `^~φ(s) = E_{(x,y)~D}[(1−2y)(−1)^{s·x}]`. For functional noiseless φ, `(1−2y) = (−1)^{f(x)} = ~φ(x)`, so the estimator equals `mean[χ_s(x) ~φ(x)] → ^~φ(s)`. Correct.
- For single parities `^~φ(s) = δ_{s,s*}`. Hoeffding gives `Pr[|^ξ(s*) − 1| > 0.1] ≤ 2 exp(−2·3000·0.01) ≈ 9·10^{−27}`, so 3000 samples are massively over-sufficient; this explains the perfect 100/100 result.

## Results vs. literature

- **Completeness.** Theorem 12 predicts `Pr[accept] ≥ 1 − δ = 0.9`. Observed: 100% accept and 100% correct for every n, Wilson 95% lower bound 96.3%. Does not contradict the bound (one-sided) and is the strongest possible empirical confirmation for this distribution class. For single parities the theory expects "near-certain completeness," and that is what is observed.

- **n-independence of resource budget.** Theorem 12 states the verifier's classical sample complexity is `O(b⁴ log(1/(δθ²)) / (ε⁴ θ⁴))` with no n in the leading order. The experiment fixes the verifier budget at 3000 across all n; it therefore *implicitly tests* n-independence in the trivial sense that 3000 samples suffice at every n in the sweep. **It does not measure** how the *minimum* sufficient budget grows with n, because the budget is hard-coded.

- **Post-selection rate.** Theorem 5(i) predicts 1/2 exactly. Empirical median 0.498-0.502 with sub-percent deviation. Essentially perfect agreement, confirming the QFS implementation.

- **List size.** For single parities exactly one Fourier coefficient is non-zero. Empirical median |L| = 1 at every n. Perfect agreement. Parseval bound `4/θ² = 44.4` shown in the figure is from Theorem 7 / Lemma 8; the implementation actually uses the looser `64 b²/θ² = 711` from Theorem 12. Neither is hit.

- **Wall clock.** 0.36 s at n=4 → 1203 s at n=16. The 8x jump n=13→14 (9 s → 74 s) and 13x jump n=14→15 (74 s → 946 s) are *simulator* artefacts, not protocol artefacts: `_sample_statevector` (`mos/sampler.py:304`) builds a `2^{n+1}`-dimensional Statevector per shot and applies a Hadamard layer via `Statevector.evolve` (O(n 2^n) per shot), with 2000 shots per trial. The super-doubling jump at n≈14-15 is plausibly cache-related (the statevector exceeds L2/L3). Doesn't affect correctness.

## Issues / discrepancies

### MAJOR

**M1. `resource_scaling.{pdf,png}` is misleading.** The figure overlays the theoretical scaling `C · n log(1/(δθ²)) / θ⁴` on a curve whose y-values are *constant 6000* across all n (because all three sample budgets are hard-coded). The fitted constant C adjusts the theory line to coincide visually with the flat data, creating the impression that resource scaling is being measured. In fact the experiment fixes the resource budget and tests whether it suffices, which is a different (and weaker) statement than what the figure suggests. Recommendation: re-run with `qfs_shots=None`, `classical_samples_prover=None`, `classical_samples_verifier=None` so that the protocol's own formulas drive the per-trial copy count (this would actually exercise the n-independent verifier formula and the n-independent prover formula); or change the figure to clearly state that the budget is fixed. This is the single most important observation in this audit, because the figure is the primary visual artefact for the experiment's headline claim.

**M2. Honest-prover sample budgets are far below the Theorem 12 defaults.** With δ=0.1, θ=ε=0.3, b²=1, the Theorem 12 verifier formula gives roughly `b⁴ log(1/(δθ²))/(ε⁴θ⁴) = log(111.1)/0.00081 ≈ 5800` samples (the constant from Step 3, `|L|² log(|L|/δ)/ε⁴`, gives ≈284 for `|L|=1`). Actual verifier budget: 3000. For the prover, the auto-formula `qfs_shots = 2.5 · ceil(2 log(4/δ)/(θ²/8)²) ≈ 145664`, vs actual 2000 — about 73x fewer QFS shots than the theoretical default. This is fine for single-parity targets where the signal is overwhelming, but it means the experiment is *not* testing the theorem in its intended regime, and 100% acceptance is not a tight test of the theorem's guarantees.

### MINOR

**m3. Distributional Theorem 12 path is used uniformly even for the noiseless functional case.** The list-size bound `64 b²/θ²` is correct for Theorem 12 but looser than Theorem 7 / Theorem 8 (which use `4/θ²` from the Parseval bound on Fourier-sparse Boolean functions, Lemma 8). The experiment is solving the noise-free functional case, where Theorem 8 would give tighter bounds. The choice is documented at `verifier.py:454-457`. Conceptually fine, but it means the figure's `4/θ²` reference line is from a different (functional) theorem than the implemented (distributional) bound.

**m4. Comment-vs-code mismatch in `prover.py:312-316`.** The comment reads `tau = epsilon^2 / 8` but the code uses `tau = theta**2 / 8.0`. Functionally identical when `θ = ε` (as in the scaling experiment) but technically wrong if a caller passes `θ ≠ ε` (as the truncation and theta_sensitivity experiments do). Should read `tau = θ²/8`.

**m5. `list_size_vs_n` reference line uses `4/θ²` but the implementation enforces `64 b²/θ²`.** The figure's reference line corresponds to the Theorem 7 / Lemma 8 Parseval bound, while the actually-tested bound is 711 not 44.4. Both are valid bounds (the implementation is conservative) but the legend should make this distinction clear.

### NIT

**n6. `make_random_parity` excludes `s* = 0`** (`phi.py:55`, `rng.integers(1, 2**n)`). The all-zeros parity is the constant function `φ ≡ 0`, with `~φ ≡ 1` and a unique non-zero Fourier coefficient at `s = 0`. Excluding it is harmless but the protocol should also work on this trivially-easy case.

**n7. `total_copies` accounting.** `total_copies = qfs_shots + classical_samples_prover + classical_samples_verifier = 6000` (`worker.py:224`). The paper's "copies of ρ_D" for the prover is just QFS shots; the verifier's *classical* random examples are simulated by computational-basis measurement of ρ_D (Lemma 1), so the accounting is defensible. But the paper treats verifier random examples as classical resources, not MoS copies. A clearer label would be "total samples consumed."

**n8. Easiest possible distribution class.** The scaling experiment only uses pure parities (the noise-free functional case). This is the easiest possible distribution for the protocol. A more discriminating sweep would either reduce the sample budget per n until 100% acceptance breaks (giving a real resource curve) or use harder distributions (`bent`, `average_case`, `k_sparse` experiments do this).

**No BLOCKERS found.**

## Verdict

The scaling experiment correctly implements the Caro et al. distributional verification protocol (Theorem 12) for honest single-parity targets, and the empirical results — 100% acceptance and 100% correct hypothesis recovery for all n ∈ [4, 16], post-selection rate 0.498-0.502, median |L| = 1 — are exactly what the theorem predicts. Theorem 5 (post-selection rate), Theorem 12 completeness (`Pr[accept] ≥ 1 − δ`), and the Parseval list-size bound are all confirmed within statistical error. **However**, the experiment's "scaling" framing is weaker than it appears: all three sample budgets (`qfs_shots = 2000`, `classical_samples_prover = 1000`, `classical_samples_verifier = 3000`) are hard-coded constants, so the `total_copies` column is `6000` at every n and the `resource_scaling.pdf` figure is plotting a fitted theoretical curve through a flat data series. The experiment therefore demonstrates "this fixed budget suffices for n up to 16 on the easiest possible target distribution," not "the verifier's sample complexity is empirically n-independent." That is the single MAJOR finding; everything else is MINOR or cosmetic. Recommend (a) re-running the scaling experiment with the hard-coded budgets removed (`qfs_shots=None`, etc.), or (b) re-labelling the resource figure to clarify that the budget is fixed by user choice and not measured.

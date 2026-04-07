# Audit: soundness_multi experiment

## Paper sections consulted

Independent reading of `papers/classical_verification_of_quantum_learning.pdf` (Caro, Hinsche, Ioannou, Nietner, Sweke; arXiv:2306.04843v2):
- pp. 1-5: Abstract, framework, mixture-of-superpositions definition.
- pp. 16-17: Definition 7 (Interactive verification of α-agnostic learning) — completeness/soundness conditions, eq. 17-18.
- pp. 35-42: Section 6.1 (Verifying functional agnostic learning); Definition 11; Theorems 7/8 (parities); **Theorems 9/10** (Fourier-k-sparse) with threshold `a^2 - eps^2/(32 k^2)` and per-coeff tolerance `eps^2/(64 k^2 |L|)`.
- pp. 44-50: Section 6.3; Definition 14 (L2-bounded bias `[a^2, b^2]`); Theorem 12 (parity); **Theorem 15** (k-sparse, distributional) Steps 3-4 give per-coeff tolerance `eps^2/(256 k^2 |L|)` and acceptance threshold `a^2 - eps^2/(128 k^2)` (eq. 118, p. 49).

## What the paper predicts

The verifier protocol is information-theoretically sound: against **any** (possibly unbounded) prover P', the verifier outputs an unsound hypothesis with probability at most δ (Definition 7, eq. 18). For the k-sparse, distributional case (Theorem 15, p. 48-49) the verifier:
1. Rejects if `|L| > 64 b^2/θ^2`.
2. Independently estimates each `xi(s)` to tolerance `eps^2/(256 k^2 |L|)` from its own classical samples (independence is the source of soundness).
3. Rejects if accumulated weight `Σ xi(s)^2 < a^2 - eps^2/(128 k^2)`.
4. Otherwise builds the randomised hypothesis from the k heaviest entries.

**The paper does not enumerate or formally classify any specific cheating-prover strategies.** Soundness is proved against an arbitrary `P'` and quantified by a single δ budget; the proofs (eq. 110-114 p. 46 and eq. 119-127 pp. 49-50) only rely on the verifier's own checks once an arbitrary `L` is received. The menu of "strategies" tested in the experiment is therefore an engineering choice, not something the paper recommends or requires.

## What the experiment does

`experiments/harness/soundness_multi.py` (lines 12-152) sweeps `(n) × (k) × (4 strategies) × num_trials`. For each cell it builds a fresh random k-sparse target via `phi.make_k_sparse` (Dirichlet(1,...,1) over k random nonzero parities) and submits a `TrialSpec` with `dishonest_strategy` set.

Default sweep on the on-disk run `results/soundness_multi_4_16_100.pb`:
- `n_range = 4..16` (13 values), `k_range = [2, 4]`, `num_trials = 100`
- `epsilon = 0.3`, `delta = 0.1`
- `theta = min(eps, max(0.01, 0.9/k))` → 0.3 for k=2, 0.225 for k=4
- `a_sq = b_sq = parseval_weight = Σ c_i^2` from `make_k_sparse`
- `classical_samples_verifier = 3000`
- Total trials: 13 × 2 × 4 × 100 = **10,400** (matches the row count of `results/figures/soundness_multi/soundness_multi_summary.csv`).

The four strategies (`experiments/harness/worker.py:273-321`, registered at lines 324-333):

| Strategy | What the prover sends |
|---|---|
| `partial_real` (L 273-284) | The weaker half of genuine heavy coefficients (`heavy_sorted[n_real:]` with `n_real=max(1,len/2)`) plus up to 3 fakes, claimed coeff 0.5 |
| `diluted_list` (L 287-298) | A single weakest real index (`heavy_sorted[-n_keep:]` with `n_keep=max(1,len/4)` ⇒ **always 1** for k∈{2,4}) plus up to 20 random padding indices |
| `shifted_coefficients` (L 301-308) | Entirely fake indices (complement of true support), claimed coeff 0.8 |
| `subset_plus_noise` (L 311-321) | The single heaviest real index plus up to 5 fakes, claimed coeff 0.3 |

Verification routes through `MoSVerifier.verify_fourier_sparse` because `spec.k > 1` (`worker.py:381-401`, `ql/verifier.py:367-428`). Acceptance threshold and list-size bound implemented at `verifier.py:458, 511-512`.

## Implementation correctness

Core machinery (the verifier) matches Theorem 15:
- Acceptance threshold `a^2 - eps^2/(128 k^2)` (`verifier.py:512`) ↔ Theorem 15, eq. 118.
- Per-coefficient tolerance `eps^2/(256 k^2 |L|)` (`verifier.py:485`) ↔ Theorem 15, Step 3.
- List-size bound `64 b^2/θ^2` (`verifier.py:458`) ↔ Step 1 of Theorem 15.
- Independent estimation in `_estimate_coefficients_independently` (`verifier.py:553-611`) draws fresh classical samples and computes the standard estimator `(1/m) Σ (1−2y_i)(−1)^{s·x_i}`. Independence preserved.
- Verifier seed `spec.seed + 1_000_000` (`worker.py:379`) decorrelates the verifier's classical samples from the prover's adversarial RNG.

Driver and per-strategy code:
- `make_k_sparse` (`experiments/harness/phi.py:133-188`) draws k distinct nonzero parities and Dirichlet weights summing to 1; `parseval_weight = Σ c_i^2` is plumbed as `a_sq = b_sq`. Correct under Definition 14.
- `theta = min(eps, max(0.01, 0.9/k))` (line 92) is an undocumented heuristic; the paper only requires `theta ∈ (2^{−(n/2−3)}, 1)`. With these values, the list-size bound `64 b^2/θ^2` is between ~700 and 1280, well above any prover's ≤20-element list (consistent with `reject_list_size = 0` in every CSV row).

Per-strategy correctness checks (eps=0.3, Dirichlet means):
- `partial_real`: drops the heaviest real coefficient(s), so accumulated true weight ≪ a_sq. Should always reject. **Empirical: 100% at every (k,n).** Consistent.
- `diluted_list`: keeps the weakest real coefficient (n_keep=1 for k≤4); padding contributes ~0. Gap from a_sq is large for both k=2 and k=4. **Empirical: 96-99% (k=2), 100% (k=4).** Consistent.
- `shifted_coefficients`: accumulated true weight ≡ 0. **Empirical: 100% at every (k,n).** Structurally inevitable.
- `subset_plus_noise`: keeps only `c_max`, so true weight ≈ c_max². For k=2, mean c_max ≈ 0.75 ⇒ c_max² ≈ 0.5-0.7, while a_sq = c_max² + c_min² mean ≈ 2/3. The gap is on the order of c_min² ≈ 0.06 — comparable to the sampling noise of 3000-sample Fourier estimates. **Empirical: 82-92% (k=2), 98-100% (k=4).** This is the only strategy that probes the actual decision boundary.

## Results vs. literature

From `results/figures/soundness_multi/soundness_multi_summary.csv` (10,400 trials):

| strategy | k | rejection range over n=4..16 | mean rej |
|---|---|---|---|
| partial_real | 2 | 1.000 | 1.000 |
| partial_real | 4 | 1.000 | 1.000 |
| diluted_list | 2 | 0.96 - 0.99 | ~0.985 |
| diluted_list | 4 | 1.000 | 1.000 |
| shifted_coefficients | 2 | 1.000 | 1.000 |
| shifted_coefficients | 4 | 1.000 | 1.000 |
| subset_plus_noise | 2 | 0.82 - 0.92 | ~0.872 |
| subset_plus_noise | 4 | 0.98 - 1.00 | ~0.992 |

False-acceptance rates (1 − rejection) compared with `delta = 0.1`:
- `partial_real`, `shifted_coefficients`: 0% everywhere.
- `diluted_list`: 1-4% (k=2), 0% (k=4).
- **`subset_plus_noise`: 8-18% (k=2)**, 0-2% (k=4).

`subset_plus_noise` at k=2 **exceeds** the experiment's claimed `delta = 0.1` budget at several n (e.g. n=6: 17%, n=7: 18%, n=9: 18%, n=12: 15%). The shortfall is attributable to the hardcoded `classical_samples_verifier = 3000` being far below the Hoeffding-derived sample budget for k>1: per-coeff tolerance for k=2, |L|=6 is `eps²/(256·k²·|L|) ≈ 1.5e-5`, and `m ≥ (2/tol²)·log(4|L|/δ)` evaluates to ~1.4×10¹⁰ — five orders of magnitude more than 3000. So the 18% empirical false-acceptance is not a logical violation of Theorem 15's `1−δ` guarantee (the Big-O hides constants), but it does mean the experiment **does not deliver** the `1 − δ = 0.9` rejection it appears to claim. `verifier._verify_core` (`verifier.py:487-497`) would have computed Hoeffding-sized `num_samples` if `num_samples=None` had been passed; the experiment overrides it (`soundness_multi.py:112` → `worker.py:392/400`).

## Issues / discrepancies

### MAJOR

**M1. `subset_plus_noise` at k=2 falsely accepts at rates up to 18%, exceeding the experiment's stated `delta = 0.1`.** Root cause: hardcoded `classical_samples_verifier = 3000` is far below the Hoeffding-derived sample budget required for the k-sparse path's tolerance `eps²/(256 k² |L|)`. The harness should either bump the sample budget, acknowledge in docstrings that this is a finite-sample effect, or widen the gap (e.g. raise eps) so the marginal regime is unambiguously rejected. As written, the experiment claims to "validate the accumulated weight check" but actually demonstrates that a marginal-cheat strategy succeeds far more than the claimed 10% of the time.

**M2. Two of the four strategies are essentially redundant or trivially weak.**
- `partial_real` and `diluted_list` overlap heavily — both leave the verifier with low true accumulated weight; both reject ≥96%.
- `shifted_coefficients` has true weight ≡ 0 by construction — rejection is structurally inevitable, no information beyond what the simpler `wrong_parity` in `soundness.py` already shows.
- This leaves `subset_plus_noise` as the only strategy that probes the decision boundary. The other three inflate the trial count by 4× without testing genuinely distinct adversarial behaviours.

### MINOR

**m1. `theta` is an undocumented heuristic.** `soundness_multi.py:92`. Affects only the (already-loose) list-size bound, but should be justified.

**m2. `phi_description` is rewritten in the dishonest path** (`worker.py:447`), losing the `_multi_` and `_k=` tags from `soundness_multi.py:114`. Downstream survives only by accident via `parse_strategy`. Worse, the comment in `plot_soundness_multi.py:87` ("k field was not populated in the proto") is **wrong**: the `k` field IS populated (`results.py:374-375`, `worker.py:448`). The plot script could read `t["k"]` directly instead of inferring it from theta.

**m3. `dishonest_strategy` field's docstring is stale.** `worker.py:65-68` lists only the four single-element strategies; the four multi-element ones added later are absent.

**m4. `_strategy_diluted_list`'s `n_keep` is degenerate at small k.** `worker.py:291`: `n_keep = max(1, len/4)` ⇒ always 1 for k ∈ {2,3,4}. The "diluted" intuition (keep all real coefficients but add noise) only kicks in for k ≥ 8. The strategy docstring (`soundness_multi.py:42-44`: "Prover includes all real heavy coefficients but pads with 20 random indices") **does not match the code**, which keeps only one.

**m5. `_strategy_partial_real`'s slicing is misleading.** `worker.py:277-278`: `n_real = max(1, len // 2); real_part = heavy_sorted[n_real:]`. For k=2 this keeps only the smaller of the two coefficients. The variable name `n_real` is the count *dropped*, not kept. Behaviour OK; naming is confusing.

**m6. Hoeffding sample budget is bypassed.** `verifier._verify_core` (`verifier.py:487-497`) computes `num_samples` from Hoeffding when `num_samples is None`, but the experiment hardcodes 3000. Root cause of M1; should at least be documented.

**m7. `subset_plus_noise` always keeps `heavy_sorted[0]`, which IS `target_s`.** Combined with the verifier's argmax-based hypothesis selection (`verifier.py:633`), when the verifier accepts, the output hypothesis tends to use the correct heaviest index. This makes the "false acceptance" framing softer — the adversary is essentially submitting a partial-but-correct list and the verifier is partially-validly accepting. Worth noting in the analysis.

### NIT

**n1.** See m2: `k` field IS populated; the plot script's comment is wrong.
**n2.** `comparison_single_vs_multi.png` pools across all 4 single + 4 multi strategies, mixing one genuinely-marginal multi strategy with three trivial single ones. The comparison is apples-to-oranges and understates the difference.

## Verdict

**The verifier code is faithful to Theorem 15**, and three of four cheating strategies (`partial_real`, `diluted_list`, `shifted_coefficients`) reject at rates indistinguishable from 1 — a meaningful sanity check, even though those three are largely interchangeable and partly redundant with the simpler `soundness` experiment. The fourth strategy, `subset_plus_noise`, is the only one that probes the decision boundary, and it reveals that the experiment's hardcoded `classical_samples_verifier = 3000` is far below the Hoeffding-derived sample budget the paper's `δ = 0.1` guarantee assumes for k ≥ 2. The resulting empirical false-acceptance rate (up to 18% at k=2) is not a soundness violation in the formal Big-O sense, but it does mean the experiment **is not validating** the paper's `1 − δ` guarantee in the regime it appears to claim to validate. Plotting infrastructure is solid; the docstring framing of what the experiment proves should be tightened. Minor implementation bugs (m4, m5) do not affect the qualitative conclusions but would matter if `k_range` were extended.

**Overall: scientifically sound implementation of the verifier, partially-misleading framing of what the experiment demonstrates. One MAJOR concern about under-sampling (M1), one MAJOR concern about strategy redundancy (M2), several MINOR cleanups.**

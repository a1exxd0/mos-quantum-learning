# Audit: k_sparse experiment

## Paper sections consulted

- pp. 1-5 (intro)
- pp. 13-18 (Definitions 2-8)
- pp. 35-39 (Theorems 7-8 parity)
- pp. 40-42 (Theorems 9-10 Fourier-k-sparse, functional)
- pp. 44-49 (Definitions 13-14, Theorems 11-15, distributional case)

## What the paper predicts

**Theorem 15 (p. 48):**
- List size bound: `|L| <= 64 b^2 / theta^2` (k-independent).
- Per-coefficient tolerance: `eps^2 / (256 k^2 |L|)`.
- Acceptance threshold: `a^2 - eps^2/(128 k^2)`.
- Verifier sample complexity: `O~(b^4 k^4 log(1/(delta theta^2)) / (eps^4 theta^4))`, i.e. **k^4** dependence.
- Pick the k heaviest |xi(s)| in L; use Lemma 14 randomised hypothesis.
- Accuracy guarantee: `eps >= 4 k sqrt(b^2 - a^2)`, `err(h) <= 2 opt + eps`.

## What the experiment does

`experiments/harness/k_sparse.py:13` sweeps `k_values = [1, 2, 4, 8]`, `experiments/harness/__main__.py:193` uses `range(n_min, n_max+1, 2)` so the saved sweep is n in {4,6,8,10,12,14,16}, 100 trials/cell. `eps=0.3`, `delta=0.1`, `theta = min(eps, max(0.01, 0.9/k))` (so theta=0.30, 0.30, 0.225, 0.1125 for k=1,2,4,8). `a_sq = b_sq = parseval_weight = sum(c_i^2)`. **k=1 is routed to `verify_parity`, not `verify_fourier_sparse`** (k_sparse.py:78).

## Implementation correctness

Every k-dependent formula in `ql/verifier.py` matches the paper line by line:
- `verifier.py:458`: `list_size_bound = ceil(64 * b_sq / theta**2)` matches Theorem 15 Step 1.
- `verifier.py:485`: `per_coeff_tolerance = eps^2 / (256 * k^2 * |L|)` matches Step 3.
- `verifier.py:512`: `acceptance_threshold = a_sq - eps^2 / (128 * k^2)` matches Step 4.
- `verifier.py:158`: randomised hypothesis `p(x) = (1-g(x))^2 / (2(1+g(x)^2))` matches Lemma 14.
- `_build_fourier_sparse_hypothesis` correctly picks the k heaviest by `|xi_hat|`.
- `make_k_sparse` (phi.py:133) draws k distinct *non-zero* parities, Dirichlet(1,...,1) coefficients summing to 1, sets `target_s = argmax(coeffs)`, and uses `parseval_weight = sum(c_i^2)` which is exactly `E[phi_tilde^2]`. `phi(x) in [0,1]` is enforced by the construction.

**No code-level bugs in the verifier or generator.**

## Results vs literature

From `results/figures/k_sparse/k_sparse_summary.csv`:

- **Acceptance rates** (per (n,k) cell, 100 trials each): k=1 always 100%; k=2 in 46-64%; k=4 in 36-58%; k=8 in 41-96% (with outlier peaks 92%/96% at n=6,8 due to spurious heavy candidates from finite-sample QFS noise at small n). **The 1-delta=0.9 completeness target is met only for k=1.**
- **All rejections are `reject_insufficient_weight`**: `rej_list_count = 0` for every cell. The list-size bound `64 b^2/theta^2` (~280-700) never bites.
- **Median accumulated weight is below the acceptance threshold for k>=2**: e.g. (n=4,k=4) median_weight=0.368 vs mean_threshold=0.402; (n=16,k=8) 0.200 vs 0.218. The slack `eps^2/(128 k^2)` is only 0.7%-0.001% of `a^2`, far too tight.
- **a^2 = 2/(k+1)** in expectation (Dirichlet variance identity); the CSV's mean_threshold values 0.987, 0.66, 0.40, 0.21 for k=1,2,4,8 match this exactly.
- **Misclassification rates**: k=1: 0.000; k=2: 0.150; k=4: 0.265; k=8: 0.347. **The k=8 cells consistently exceed `eps = 0.3`** (Theorem 15 promises `err <= 2 opt + eps = 0.3` for functional data with opt=0).
- **List sizes**: k=1 always 1; for k>=2 small at large n (median 2-5) but inflated at n=4,6,8 by finite-sample QFS noise (median up to 100 at (n=8, k=8)).

## Issues / discrepancies

### MAJOR

**M1. Figure-level interpretation issue.** The experiment is *deliberately* operating outside the paper's **Definition 13** granularity promise (`hat phi(s) != 0 => |hat phi(s)| >= theta`; §5.2, p.44), distinct from **Definition 14** (the `E[phi^2] in [a^2, b^2]` bracket, which the experiment *does* satisfy trivially since `a=b=pw`). With Dirichlet(1,...,1), the smallest coefficient is `Theta(1/k^2)` in expectation, so for k>=2 most trials have at least one Fourier coefficient below `theta = 0.9/k`, violating Definition 13. Empirical Def 13 compliance verified by direct resampling: 100% at k=1, 40.4% at k=2, 0.1% at k=4, 0.0% at k=8. The headline plot `acceptance_vs_n_by_k.png` overlays a "1-delta=0.9 (Thm 9)" line that is not actually applicable (the paper makes a promise the experiment violates, and the cited theorem is also wrong: Theorem 9 is the SQ verifier; the relevant theorems for this experiment are Theorem 10 / Theorem 15). Mitigation: either rejection-sample targets so `min(c_i) >= theta`, or relabel the dashed line and figure caption to make the off-promise stress-test interpretation explicit.

### MINOR

**m2.** Average misclassification at k=8 is 0.34-0.36 vs Theorem 15's `<= eps = 0.3`. This is conditional-on-acceptance, while the theorem bounds `Pr[reject or err <= eps] >= 1-delta`. Plausibly within slack, but the figure title `<= 2*opt + eps` is technically wrong and the per-trial distribution should be reported, not just the mean.

**m3.** `plot_k_sparse.py:259` overlays `bound = 4/theta^2` (Theorem 9 form) instead of the actually-enforced `64 b^2/theta^2` (Theorem 15 form, ~16x larger). Misleading visualisation of headroom.

**m4.** k=1 is routed to the parity path; the `verify_fourier_sparse(k=1)` code path is never exercised by this experiment.

**m5.** `a_sq = b_sq = parseval_weight` exactly, granting the verifier zero-slack knowledge of `||phi||_2^2`. Stronger than realistic deployment.

### NIT

**n6.** Plot legend says "Thm 9"; should be "Thm 10" or "Thm 15".

## Verdict

The k-sparse verifier code is a faithful, line-by-line implementation of Theorem 15 of Caro et al. The k-sparse target generator is correct. The completeness shortfall for k>=2 is *not* a bug — it is a consequence of the experiment intentionally violating the paper's theta-promise on the target distribution, combined with an extremely tight `eps^2/(128 k^2)` slack that leaves no room for finite-sample error in the verifier's accumulated weight check. The experiment is measuring something interesting (off-promise behaviour); the headline figure should make that interpretation explicit. **MAJOR (figure interpretation), MINOR (k=8 misclass slightly over eps; plot bound formula; k=1 not exercising FS path); no BLOCKER.**

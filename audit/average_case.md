# Audit: average_case experiment

## Paper sections consulted

Independent reading of `papers/classical_verification_of_quantum_learning.pdf` (Caro, Hinsche, Ioannou, Nietner, Sweke; arXiv:2306.04843v2):

- pp. 1-5 — Abstract, Introduction, Section 1.1 Framework: functional vs distributional agnostic learning, MoS examples, interactive verification of learning.
- pp. 10-12 — Section 1.4 Techniques: Eq. 4 for QFS; the distribution-class promise sum_s phi_hat(s)^2 in [a^2, b^2] introduced as Eq. 5; restriction to "well-described by a few heavy Fourier coefficients".
- pp. 29-34 — Section 5.2 Distributional Agnostic Quantum Learning: Corollaries 6, 7, 8 (parity vs Fourier-sparse paths; the k^4 log(.)/eps^4 Fourier-sparse copy bound), Section 5.3 (QSQ variant).
- pp. 35-43 — Section 6 Classical Verification: Definition 11 of D^func_{U_n; >= theta}; Lemma 8 sparsity inclusions; Theorems 7/8 for parities; Theorems 9/10 for Fourier-sparse; acceptance threshold `1 - eps^2/8` for parity, `1 - eps^2/(32 k^2)` for k-sparse; list-size bound `4/theta^2` (SQ verifier) or `64/theta^2` (random-example verifier).

## What the paper predicts

The paper does **not** define an "average-case" instance distribution. Theorems 7-15 and Corollary 7 state worst-case (uniform-over-promise-class) completeness/soundness:

- Parity (Theorems 7/8/12): completeness `>= 1 - delta` when phi in `D^func_{U_n; >= theta}`; verifier accepts iff `sum (gamma_hat(s))^2 >= a^2 - eps^2/8`; `|L| <= 4/theta^2` (SQ) or `<= 64/theta^2` (random examples).
- Fourier-k-sparse (Theorems 9/10/15): same promise class, but the verifier accepts iff `sum >= a^2 - eps^2/(32 k^2)`, per-coefficient tolerance is `eps^2/(64 k^2 |L|)`, and the hypothesis is the randomized linear combination of the k heaviest (Lemma 14).
- Lemma 8 (p. 35): `D^func_{U_n; >= theta}` is sandwiched between Fourier-`floor(2/theta)`-sparse and Fourier-`(1/theta^2)`-sparse.

The paper makes **no** claim about random Boolean (Fourier-dense) functions or "sparse + noise" mixtures. A uniformly random truth table at large n has all 2^n Fourier coefficients of magnitude `~2^{-n/2}`, so it lies *outside* `D^func_{U_n; >= theta}` for any nontrivial theta and is *not covered* by any verification theorem. There is no paper-grounded prediction of "expected list size", "expected acceptance rate", or "average-case sample complexity".

## What the experiment does

`/Users/alex/cs310-code/experiments/harness/average_case.py` — for every n in `n_range` and every family in `["k_sparse_2", "k_sparse_4", "random_boolean", "sparse_plus_noise"]`, sample `num_trials` random functions, run honest prover then verifier, aggregate accept rate / list size / copy count.

Default sweep parameters (lines 79-92): `epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples_prover=1000, classical_samples_verifier=3000, base_seed=42`.

Instance distributions (`experiments/harness/phi.py`):

- `k_sparse_2` (`phi.py:133-188`): k=2 indices uniform from `{1..2^n-1}`, Dirichlet(1,1) coeffs. `a^2 = b^2 = sum c_i^2` (varies). `theta = min(eps, 0.9/k) = 0.3`.
- `k_sparse_4`: k=4 indices uniform, Dirichlet(1,1,1,1) coeffs. `theta = min(eps, 0.9/4) = 0.225`.
- `sparse_plus_noise` (`phi.py:228-272`): one dominant coeff 0.7 plus three secondary 0.1. `a^2 = b^2 = 0.7^2 + 3*0.1^2 = 0.52`. `theta = eps = 0.3`.
- `random_boolean` (`phi.py:191-225`): uniform truth table. `a^2 = b^2 = 1.0` (functional). `theta = eps = 0.3`.

Verification path: the worker (`worker.py:163-183`) selects between `verify_parity` and `verify_fourier_sparse` based on `spec.k`. Critically, **`average_case.py` never sets `spec.k`** (verified via Grep for `k=` in `average_case.py` — no matches), so every trial — including k_sparse_2, k_sparse_4 and sparse_plus_noise — runs the parity verifier with acceptance threshold `a^2 - eps^2/8 = a^2 - 0.01125`. See `verifier.py:507-512`.

## Implementation correctness

What is implemented faithfully:
- The parity verifier (`ql/verifier.py:300-365, 434-547`) implements Theorem 12: list-size cap `|L| <= 64*b^2/theta^2`, per-coefficient Hoeffding tolerance `eps^2/(16|L|)`, threshold `a^2 - eps^2/8`. Matches paper.
- Honest prover QFS extraction (`ql/prover.py:370-497`) matches Corollary 5 with extraction threshold `theta^2/4`.
- `make_k_sparse` excludes index 0, draws Dirichlet so `sum c_i = 1`, and uses `pw = sum c_i^2` as `a^2 = b^2` per Definition 14.
- `make_sparse_plus_noise` correctly yields `tilde_phi = 0.7 chi_{s*} + 0.1 (chi_{s_2} + chi_{s_3} + chi_{s_4})`; `sum c_i = 1`, `sum c_i^2 = 0.52`.
- Random sampling uses per-experiment `default_rng(base_seed)` spawning per-trial seeds (`average_case.py:36, 156-167`) — reproducible and statistically independent.

What is **not** faithful to the paper:

1. **Verification path mismatch for k-sparse families.** Corollary 7 (p. 30) and Theorem 9 (p. 40) require the *Fourier-k-sparse* path: verify the k heaviest coefficients, build a randomized hypothesis per Lemma 14, check threshold `a^2 - eps^2/(32 k^2)`. The experiment runs `verify_parity` for k=2 and k=4 instances, which uses the parity threshold `a^2 - eps^2/8` (much looser for k>1) and outputs a single-parity hypothesis even when ground truth has 2 or 4 weights of comparable size.
2. **`random_boolean` is outside the promise class.** A uniform random truth table at n>=6 has all coefficients of magnitude `~2^{-n/2}`, so for `theta=0.3` Definition 11 is violated. The experiment sets `a_sq = b_sq = 1.0` (technically the exact `E[tilde_phi^2]`), but the sparsity promise is broken.
3. **`theta` adaptation for k-sparse is fragile.** Line 47: `theta = min(eps, 0.9/k)`. For Dirichlet(1,1) draws the heaviest coefficient is Beta(1,1)=Uniform[0,1], so a non-trivial fraction of draws have `c_max < theta` and fail GL extraction. Explains the ~70% accept ceiling for k_sparse_2 even at large n.
4. **`total_copies` is constant across trials.** `worker.py:224`: `total_copies = qfs_shots + classical_samples_prover + classical_samples_verifier = 6000` for every trial regardless of n. The "average-case resources" claim is not measured at all — the resources are inputs.

## Results vs. literature

From `results/figures/average_case/cross_family_summary.csv` produced by `plot_average_case.py` from `results/average_case_4_16_100.pb`. The CSV gives 100 trials per (family, n) for n=4..16 with accept rate (Wilson 95% CI), prover-found rate, median |L|, median total copies.

Headline numbers:

- **k-sparse (k=2)**: accept 0.76 (n=4) → 0.78 (n=10) → 0.76 (n=16); prover-found 1.00 across all n; median |L| = 2 for n>=5.
- **k-sparse (k=4)**: accept 0.81 (n=4) → 0.63 (n=10) → 0.74 (n=16); prover-found 1.00 across all n; median |L| = 3 for n>=7.
- **sparse + noise**: accept 0.78 (n=4) → 0.13 (n=10) → 0.15 (n=16); prover-found 1.00 across all n; median |L| = 1 for n>=6.
- **random Boolean**: accept 0.48 (n=4) → 0.00 for n>=6; prover-found 1.00 (n<=8) → 0.00 (n>=10); median |L| → 0 for n>=10.

Cross-checks:
- The companion `scaling` experiment with single-parity phis at the same `theta=eps=0.3, a^2=b^2=1` achieves **100% acceptance and 100% correctness for every n in 4..16** (`results/figures/scaling/scaling_summary.csv`). So **none** of the four "average case" families reach the per-paper completeness target `1 - delta = 0.9` at any n — they are all *strictly worse* than the worst-case scaling baseline. The dashed reference line at 0.9 in `acceptance_by_family.png` is not crossed by any family at any n.
- This inversion — "average case" worse than "worst case" — makes sense once you see that (i) `random_boolean` violates the promise class, (ii) the smaller `a^2` for sparse_plus_noise/k_sparse_* makes the same `eps^2/8` slack proportionally larger, and (iii) the parity verifier is being applied to k>1 instances. None of the four families is actually testing the protocol on the regime the paper covers.

Sanity check of random Boolean: at n=4, 16 coefficients of magnitude `~1/4 = 0.25` (close to `theta=0.3`); ~half the trials happen to have at least one detectable coeff → ~48% accept. At n=10, `2^{-5} ~= 0.03 << theta` → empty L → reject. Correct verifier behaviour.

Sanity check of sparse_plus_noise decay: at n=4 the verifier reaches the `eps^2/(16|L|) ~= 0.0014` Hoeffding tolerance comfortably; at larger n the QFS budget of 2000 increasingly under-resolves the secondary 0.1 coefficients (extraction threshold `theta^2/4 = 0.0225` competes with `1/2^n` baseline noise), so the prover often returns `|L|=1`. Then accumulated weight `~0.49 < a^2 - eps^2/8 = 0.5088` — **rejection by a hair**. The acceptance threshold is mis-calibrated for this family: with `a_sq = 0.52` and a single dominant coeff of squared weight `0.49`, accepting requires the prover to also resolve at least one secondary coefficient, which the QFS budget cannot do.

## Issues / discrepancies

### MAJOR

**M1. Wrong verification path for k-sparse families.** k_sparse_2, k_sparse_4 and sparse_plus_noise all set `spec.k = None`, so the worker dispatches to `verify_parity` (`worker.py:174-183`) with parity threshold `a^2 - eps^2/8` and a single-parity hypothesis. Per Corollary 7 (p. 30) and Theorem 9 (p. 40), the correct path is `verify_fourier_sparse` with threshold `a^2 - eps^2/(32 k^2)` and a Lemma-14 randomized hypothesis. The experiment is not testing the k-sparse verification path of the paper at all.

**M2. `random_boolean` is outside the promise class.** Definition 11 requires `|phi_hat(s)| >= theta` for every nonzero coefficient. Uniformly random truth tables at n>=6 violate this for any `theta > 2^{-n/2+1}`. The experiment file's docstring describes it as "the hardest case" but the figures present it as one of four representatives of "average-case behaviour", which is misleading.

### MINOR

**m3. No paper-grounded notion of "average case".** The paper makes only worst-case-over-promise-class claims; "typical instances", "expected list size", etc. are project framing, not paper claims.

**m4. theta adaptation for k-sparse is fragile.** `theta = min(eps, 0.9/k) = 0.3` for k=2 and `0.225` for k=4. Dirichlet draws have substantial spread; a meaningful fraction of draws have `c_max < theta`, automatically failing GL extraction.

**m5. `total_copies` is constant.** Every accepted trial reports `total_copies = 6000`. The "average-case resources" claim is not measured — the resources are inputs.

### NIT

**n6.** The plot title "Average case: acceptance rate by function family" is heterogeneous: two families that mis-use the parity verifier on Fourier-sparse phis, one out-of-promise dense phi, and one in-promise sparse phi.

**n7.** `phi.py:188` returns the heaviest coefficient as `target_s`, but for k_sparse_2 with Dirichlet(1,1) the two coefficients are nearly equal so `argmax` is essentially a coin flip. Harmless because the experiment uses `target_s` only for the `prover_found_target` diagnostic.

## Verdict

Implementation is technically correct, but **the experiment does not test what its name promises and does not correspond to anything in the paper.** The two findings that matter:

- The experiment runs the *parity* verifier on instances that the paper prescribes the *Fourier-sparse* verifier for (k_sparse_2, k_sparse_4, sparse_plus_noise). The acceptance thresholds, hypothesis outputs and per-coefficient tolerances all differ from Theorems 9/10/15.
- The `random_boolean` family is outside Definition 11's promise class for any n >= 6. The verifier correctly rejects these instances; zero acceptance is the *correct* outcome, not a failure of "average case".

The .pb file and figures are valid measurements of *the parity verifier applied to a heterogeneous set of phi families*, but they should not be presented as average-case validation of the paper's protocol. To match the paper this experiment should (1) set `spec.k = k` for k_sparse_*/sparse_plus_noise so the worker dispatches to `verify_fourier_sparse`, (2) drop or rename `random_boolean`, (3) drop the "average case" framing or sample explicitly from `D^func_{U_n; >= theta}` and report expected list size and expected completeness across draws.

**Overall: MAJOR issues.** Implementation runs correctly but the experiment does not test the paper's average-case protocol; the artefacts are accurate measurements of an unintended quantity.

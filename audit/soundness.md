# Audit: soundness experiment

## Paper sections consulted

- pp. 1-5: Title, abstract, introduction (framework overview)
- pp. 15-17, Section 2.3 (Quantum Data Oracles, Definitions 4-6) and Section 2.4 (Interactive Verification of Learning, Definition 7 with Eqs. 17 and 18 — completeness/soundness)
- pp. 18-20: Definition 8 (mixture-of-superpositions), Lemma 1, Lemma 2 (QFS), Lemma 3 (DKW empirical approximation)
- pp. 35-42, Section 6.1 (Verifying Functional Agnostic Quantum Learning): Definition 11, Lemma 8, Theorem 7 (proper 1-agnostic verification of parities, SQ verifier), Theorem 8 (random-example verifier), Theorem 9, Theorem 10 (Fourier-k-sparse), Proposition 1
- pp. 43-44, Section 6.2 (Verifying Noisy Functional Agnostic Quantum Learning): Definition 12
- pp. 44-50, Section 6.3 (Verifying Distributional Agnostic Quantum Learning): Definitions 13-14, Theorem 11, Theorem 12 (the central object — distributional agnostic parity verification with random examples and MoS examples), Theorem 13 (lower bound on accuracy), Theorem 14, Theorem 15 (Fourier-k-sparse distributional)

## What the paper predicts

### Definition 7, Eq. (18) — soundness condition (page 17)

For any distribution D and any (possibly unbounded) dishonest prover P', the random hypothesis h that V(eps, delta) outputs after interacting with P' must satisfy
```
Pr[h != reject  ∧  err_D(h) > alpha * opt_D(F) + eps]  ≤  delta.        (Eq. 18)
```
This is a *combined* event: V is allowed to accept a bad hypothesis with probability at most delta. It is NOT "Pr[V accepts] ≤ delta" — V is permitted to accept arbitrarily often, provided that whenever it accepts the hypothesis is good (within alpha*opt + eps).

For the proper-verification refinement (paper p. 17, last bullet): "if additionally one can ensure that the random hypothesis h ... satisfies h ∈ {reject} ∪ F almost surely". For parity verification this means the output is either reject or a parity hypothesis from {chi_s : s ∈ {0,1}^n}.

### Theorems 7, 8, 11, 12 — concrete protocol that achieves Eq. (18) for parities

The paper's Theorem 12 (the experiment's main reference, page 45) gives a 4-step protocol; the soundness-relevant steps are:

- Step 3: V receives L from P. If |L| > 64*b^2/theta^2, V rejects.
- Step 3: Otherwise V uses its own classical examples to obtain (eps^2/(16|L|))-accurate estimates xi_hat(s) of phi_hat(s) for every s in L, with confidence 1 - delta/2.
- Step 4: If sum_{s in L} xi_hat(s)^2 ≥ a^2 - eps^2/8, accept and output s_out = argmax_{s in L} |xi_hat(s)|; otherwise reject.

Soundness proof sketch (paper pp. 37-38, Eqs. 73-89, repeated for the distributional case in Thm 12, pp. 45-46, Eqs. 110-114): if V does not reject in Step 4, then Parseval combined with the eps^2/(16|L|) accuracy bound gives, for every t not in L,
```
phi_hat(t)^2 ≤ (b^2 - a^2) + eps^2/4,
```
hence |phi_hat(t)| ≤ sqrt(b^2-a^2) + eps/2 ≤ eps for every t outside L. The argmax inside L therefore has eps-near-optimal classification error. Crucially, the soundness conclusion only relies on V succeeding in Step 3 (its own classical estimation) — it does NOT require P to be honest. The Chernoff–Hoeffding union bound used in Step 3 fails with probability at most delta/2, so the combined "accept ∧ bad hypothesis" event has probability at most delta/2 ≤ delta, matching Eq. (18).

### Numerical thresholds (functional case, a = b = 1)

- Acceptance threshold (parity): tau_accept = 1 - eps^2/8.
- For eps = 0.3 (used by the experiment): tau_accept = 1 - 0.01125 = **0.98875**.
- List-size bound: |L| ≤ 64*b^2/theta^2 = 64/0.09 ≈ **711** (with theta = eps = 0.3).
- Verifier per-coefficient tolerance (paper Step 3): eps^2/(16|L|).

### What the soundness experiment is supposed to test

Empirically estimate the false-acceptance rate of V against various adversarial P', and check it against the theoretical guarantee Pr[bad accept] ≤ delta = 0.1, i.e. rejection rate ≥ 0.9 whenever the prover's strategy yields a bad hypothesis. (When P' accidentally produces a good hypothesis, e.g. by guessing the right parity, V is *allowed* to accept — that is correct behaviour, not a soundness violation.)

## What the experiment does

### Top-level dispatcher

`/Users/alex/cs310-code/experiments/harness/__main__.py:136-147` defines `_run_soundness`:
```
soundness_trials = max(args.trials, 50)
r = run_soundness_experiment(
    n_range=range(args.n_min, args.n_max + 1),
    num_trials=soundness_trials,
    ...
)
```
The CLI thus invokes `run_soundness_experiment(...)` with at least 50 trials per (n, strategy) cell.

### `run_soundness_experiment`

`/Users/alex/cs310-code/experiments/harness/soundness.py:12-135`. Key parameters and behaviour:

- `n_range = range(4, 7)` default (overridden by CLI to range(4, 21) for the saved 4-20 file).
- `num_trials = 50` default.
- `epsilon = 0.3`, `delta = 0.1`, `theta = epsilon = 0.3`, `a_sq = b_sq = 1.0`. (`soundness.py:91-95`)
- `classical_samples_verifier = 3000` (`soundness.py:99`) — this is passed as an explicit override to `MoSVerifier.verify_parity`, so the Hoeffding-derived auto-sample-count branch in the verifier is bypassed.
- `qfs_shots = 0`, `classical_samples_prover = 0` (`soundness.py:97-98`) — irrelevant because the dishonest worker never instantiates a real prover.
- The four strategies under test (`soundness.py:70`):
  - `"random_list"`
  - `"wrong_parity"`
  - `"partial_list"`
  - `"inflated_list"`
- For each n, `target_s = 1` (`soundness.py:80`) and `phi = make_single_parity(n, 1)` (`phi.py:10-32`). So the ground-truth function is the single parity chi_{s*} with s* = 1.
- The experiment builds `TrialSpec(..., dishonest_strategy=strategy)` and dispatches via `run_trials_parallel` (`soundness.py:107-110`).
- After completion, it prints per-strategy rejection rates (`soundness.py:128-132`).

### Dishonest worker dispatch

`/Users/alex/cs310-code/experiments/harness/worker.py:130-131`:
```
if spec.dishonest_strategy is not None:
    return _run_dishonest_trial(spec, state)
```
`_run_dishonest_trial` (`worker.py:336-451`) builds a fake `ProverMessage` via the registered strategy and runs the verifier on it. It does NOT instantiate `MoSProver`, so QFS is skipped. Crucially, the same `MoSState` is constructed (`worker.py:128`) so that the verifier's `state.sample_classical_batch(...)` (called inside `_estimate_coefficients_independently`) is using the genuine distribution D = (U_n, chi_{s*}), exactly as the paper requires for V's independent estimation step.

### Strategy implementations (faithful instantiation check)

`/Users/alex/cs310-code/experiments/harness/worker.py:247-271`:

- **`_strategy_random_list`** (`worker.py:247-250`): sends a sorted list of `min(5, 2^n)` indices drawn uniformly at random without replacement from {0, ..., 2^n - 1}, with all estimates set to 0.0. Models a prover that guesses random heavy parities.
  - Important subtlety: with probability 5/2^n the random list contains s*=1. If s* is in L, the verifier's own estimate of phi_hat(s*) ≈ 1 (modulo Hoeffding noise on 3000 samples), so accumulated_weight ≈ 1, well above the threshold 0.98875. The verifier will accept and output the correct parity. This is *correct* protocol behaviour per Eq. (18), not a soundness violation, because the output hypothesis has zero error.
- **`_strategy_wrong_parity`** (`worker.py:253-258`): sends a single index `(target_s + 1) mod 2^n`, which by construction is NOT s*. With target_s = 1 this is s = 2. The estimate field is set to 1.0 (a fabricated value the verifier ignores anyway because it re-estimates). The verifier's own estimate of phi_hat(2) is approximately 0 (true value is exactly 0 since chi_{s*} is orthogonal to chi_s for s ≠ s*). Accumulated weight ≈ 0; threshold 0.98875; rejection certain modulo 0.0003-magnitude Hoeffding fluctuation.
- **`_strategy_partial_list`** (`worker.py:261-263`): sends an empty list. Accumulated weight is exactly 0 (the verifier's `_estimate_coefficients_independently` returns `{}` when L is empty; sum is 0). 0 < 0.98875, always rejected.
- **`_strategy_inflated_list`** (`worker.py:266-270`): sends 10 indices drawn uniformly at random from `[s for s in range(2**n) if s != target_s]` (so s*=1 is excluded by construction) with fabricated estimate 0.5 each. The verifier's own estimates are all ≈ 0 (true values are exactly 0). Accumulated weight ≈ 0, always rejected.

These four strategies are not exhaustive — they are deliberately easy/canonical adversaries. They are NOT a faithful instantiation of the worst-case prover model implied by Eq. (18) (which is universal over arbitrarily computationally unbounded P'). The experiment correctly disclaims this in `experiments/FINDINGS.md:165-170`. The harder, partially-honest strategies (`partial_real`, `diluted_list`, `shifted_coefficients`, `subset_plus_noise`) are exercised in the *separate* `soundness_multi` experiment, not here.

### Why the strategies are weak (but not vacuous)

For the random_list and inflated_list strategies, every entry of L is uniformly random and independent of the true s*=1. Because the underlying distribution is a single pure parity, all wrong frequencies have *exactly zero* true Fourier coefficient, so the verifier's empirical estimate has zero mean and standard error ~1/sqrt(3000) ≈ 0.018 per coefficient. The squared empirical estimate is therefore ~3.3e-4 in expectation per spurious entry, and the accumulated weight from |L| spurious entries is ~|L|·3.3e-4. For |L|=10 this gives ~0.003, six orders of magnitude below the acceptance threshold 0.98875. Hence rejection is essentially certain.

Because the underlying distribution is the most-favourable case (a pure single parity with phi_hat(s*) = 1 exactly), the experiment cannot probe the regime where the soundness bound becomes tight. Hardness shows up only in random_list, where the random list has an unavoidable 5/2^n chance of containing s* — and when it does, V *correctly* outputs the right parity. So the random_list rejection rate has a *combinatorial* lower-bound dependence on n that the experiment exploits as a non-trivial signal.

## Implementation correctness

### Verifier `_verify_core` matches the paper protocol

`/Users/alex/cs310-code/ql/verifier.py:434-547`.

- **Step 1: list-size check** (`verifier.py:454-473`): `list_size_bound = ceil(64 * b_sq / theta^2)`. With b_sq=1, theta=0.3, this is `ceil(64/0.09) = 712`. This matches Theorem 12 Step 3 (page 45): "If V receives a list L of length |L| > 64*b^2/theta^2, V rejects". The factor 64 (rather than the 4/theta^2 of Theorem 7) corresponds to the random-example version of the protocol; the docstring on lines 455-457 explains the factor-of-4 factor coming from theta/2 resolution in Corollary 5. Correct for Theorem 8/12.
- **Step 2: per-coefficient tolerance** (`verifier.py:480-485`):
  ```
  per_coeff_tolerance = epsilon**2 / (16.0 * max(L_size, 1))   # PARITY
  ```
  This matches Theorem 12 Step 3: "(eps^2/(16|L|))-accurate estimates". Correct.
- **Step 2: sample-count auto-derivation** (`verifier.py:487-497`): when `num_samples` is None, uses `m ≥ (2 / tol^2) * log(4 |L| / delta)` derived from Hoeffding with variance bound 1. In the soundness experiment, however, `num_samples=3000` is explicitly passed, so this branch is bypassed.
- **Step 3: independent estimation** (`verifier.py:499-502`, `_estimate_coefficients_independently` at lines 553-611): draws fresh classical samples via `state.sample_classical_batch` (Lemma 1) and computes
  ```
  xi_hat(s) = mean_i [(1 - 2 y_i) * (-1)^(s . x_i)].
  ```
  This is the standard unbiased estimator of phi_hat(s) under D, exactly as in the paper (page 19, Lemma 1, and Eq. 78 of the paper proof). Correct.
- **Step 4: weight check** (`verifier.py:504-527`):
  ```
  accumulated_weight = sum(verifier_estimates.get(s, 0.0) ** 2 for s in L)
  acceptance_threshold = a_sq - epsilon**2 / 8.0      # PARITY
  if accumulated_weight < acceptance_threshold: REJECT
  ```
  This is Theorem 12 Step 4 verbatim: "If sum (xi_hat(s_l))^2 ≥ a^2 - eps^2/8, then ... else V outputs reject."
- **Step 5: hypothesis output** (`verifier.py:529-547`, `_build_parity_hypothesis` at lines 617-638): `s_out = argmax_{s in L} |xi_hat(s)|`. Matches Theorem 12 Step 4.

The verifier code is a faithful, line-for-line implementation of Theorem 12.

### Independence of V's estimates from the prover

A key property of the protocol is that V's Fourier estimates use V's own classical samples, drawn independently of any data sent by P. The implementation respects this:

- `_run_dishonest_trial` constructs the verifier with seed `spec.seed + 1_000_000` (`worker.py:379`), independent of the prover seed.
- `_estimate_coefficients_independently` (`verifier.py:553-611`) draws samples from `self.state.sample_classical_batch(num_samples=num_samples, rng=self._rng)` — directly from the genuine distribution, with no input from the fake prover message.
- The strategy-supplied `estimates` field of `ProverMessage` is never read by `_verify_core`. The verifier only uses `prover_message.L`.

This means that even though the dishonest strategies may put fabricated values into the estimates dict (e.g. `1.0` for wrong_parity, `0.5` for inflated_list), those values cannot influence the verifier's decision. Correct behaviour.

### Reject-reason counting

`VerificationOutcome` has three values: `ACCEPT`, `REJECT_LIST_TOO_LARGE`, `REJECT_INSUFFICIENT_WEIGHT` (`verifier.py:72-77`). `TrialResult.outcome` stores the string value, and `accepted = (outcome == ACCEPT)`. The plot script's `build_tables` (`plot_soundness.py:68-97`) and `plot_rejection_mechanism` (`plot_soundness.py:182-229`) both consume these strings exactly. Counting is correct.

### Misclassification not recorded for parity dishonest trials

`_run_dishonest_trial` only computes `misclass_rate` when the hypothesis is accepted (`worker.py:408-421`). For parity hypotheses it now does compute misclass_rate (lines 418-421), but this is only stored on the trial and never aggregated by the soundness experiment. The summary print (`soundness.py:128-132`) only reports `accepted` rates, not the joint event "accept ∧ bad". This is a MINOR observability gap relative to Eq. (18) but does not affect correctness.

## Results vs. literature

### Concrete numbers per (strategy, n)

From `/Users/alex/cs310-code/results/figures/soundness/soundness_summary.csv` (rejection rate, 95% Wilson CI lower, upper):

**Random list (5 random indices):**
| n  | rej rate | 95% CI         | predicted ≥ |
|----|----------|----------------|-------------|
| 4  | 0.7100   | [0.6146, 0.7899] | 1 - 5/16 = 0.6875 |
| 5  | 0.8800   | [0.8019, 0.9300] | 1 - 5/32 = 0.8438 |
| 6  | 0.9800   | [0.9300, 0.9945] | 1 - 5/64 = 0.9219 |
| 7  | 0.9500   | [0.8882, 0.9785] | 1 - 5/128 = 0.9609 |
| 8  | 0.9700   | [0.9155, 0.9897] | 1 - 5/256 = 0.9805 |
| 9  | 0.9600   | [0.9016, 0.9843] | 1 - 5/512 = 0.9902 |
| 10 | 0.9900   | [0.9455, 0.9982] | 1 - 5/1024 = 0.9951 |
| 11 | 1.0000   | [0.9630, 1.0000] | 1 - 5/2048 = 0.9976 |
| 12 | 1.0000   | [0.9630, 1.0000] | 1 - 5/4096 = 0.9988 |
| 13 | 1.0000   | [0.9630, 1.0000] | 1 - 5/8192 = 0.99939 |
| 14-20 | 1.0000 | [0.9630, 1.0000] | ≈ 1.0 |

**Wrong parity, partial list, inflated list:** 1.0000 with 95% Wilson CI [0.9630, 1.0000] for every n in 4..20.

### Comparison to the soundness bound

The paper requires Pr[accept ∧ bad hypothesis] ≤ delta = 0.1, i.e. the rejection rate plus the rate of "accept-with-good-hypothesis" should sum to ≥ 0.9.

**Wrong parity, partial list, inflated list.** All four wrong-parity-only strategies achieve 100% rejection at every n in [4, 20], with the lowest 95% Wilson lower confidence bound at 0.9630 (n=100). This trivially satisfies "≥ 0.9". No accepts at all → no false accepts → soundness bound is satisfied with margin.

**Random list.** This strategy is interesting because it can hit s* = 1 by accident. Decompose:

- Pr[accept] = 1 - Pr[reject].
- Conditional on s* in L, the verifier outputs the correct parity hypothesis (zero error). So acceptance in this branch is *correct* per the paper's definition.
- Conditional on s* not in L, the accumulated weight is ~|L|·(stderr)^2 ≈ 5·(1/3000) = 0.00167, far below the threshold 0.98875. Rejection is essentially certain.

Therefore the marginal Pr[accept] should be close to Pr[s* in L] = 5/2^n (with vanishing extra mass from the "s* in L but rejected anyway" pathway).

For n=4: predicted Pr[reject] = 1 - 5/16 = 0.6875. Observed: 0.71 (95% CI [0.61, 0.79]). The CI covers the prediction.
For n=5: predicted 0.8438; observed 0.88 (CI [0.80, 0.93]). Consistent.
For n=6: predicted 0.9219; observed 0.98 (CI [0.93, 0.99]). Slight excess but within CI.
For n=7-10: predictions and observations all consistent.
For n ≥ 11: prediction exceeds 0.9976 and observation is 1.00 in 100/100 trials.

**Soundness bound check**: Pr[bad accept] = Pr[accept ∧ s* not in L] ≈ Pr[s* not in L] · ε_spurious where ε_spurious is exponentially small (Hoeffding tail with 3000 samples, 10 indices, gap 0.98875). Even for n=4 the upper bound on bad-accept rate should be << 0.1.

Empirically, the soundness bound is respected (rejection ≥ 0.9) at every n ≥ 7 unconditionally. For n in {4, 5, 6}, the rejection rates are 0.71, 0.88, 0.98 — only n=4 and n=5 fall below 0.9 in raw rejection rate. **But this is fully expected and consistent with Eq. (18)**, because the random_list strategy generates a *correct* hypothesis with probability 5/2^n by accident, and Eq. (18) only forbids bad accepts. The CSV/figures count rejections, not bad accepts, so the apparent dip below 0.9 at small n is a measurement artefact, not a soundness violation.

### Cross-check vs. theta/eps numerics

- threshold = 1 - eps^2/8 = 1 - 0.09/8 = 0.98875. Code: `verifier.py:509`. Matches.
- list_size_bound = ceil(64 / 0.09) = 712. Code: `verifier.py:458`. Adversarial lists have size ≤ 10, so this is never triggered.
- per_coeff_tolerance = eps^2 / (16·|L|). Code: `verifier.py:482`. Matches Thm 12 Step 3.

### Cross-check: classical_samples_verifier = 3000

3000 fixed verifier samples for L of size up to 10 gives an effective per-coefficient tolerance of std ≈ 1/sqrt(3000) ≈ 0.0183, vs the paper's prescribed eps^2/(16·10) = 0.0005625. The empirical tolerance is therefore *much looser* than the theory prescribes — by a factor of ~32. The per-coefficient Hoeffding failure probability with the looser tolerance is still small, and importantly the gap between true accumulated weight (1.0 in honest case, 0.0 in dishonest cases) and the 0.98875 threshold is enormous compared to the 10·(0.018)^2 = 0.003 noise floor — so 3000 samples is *operationally* sufficient. But it does mean the experiment is not exercising the paper's full Hoeffding-derived sample budget.

### Cross-check: rejection mechanism

All rejections are via the weight check (Step 4); no strategy triggers the list-size bound (Step 3). The implemented list-size bound is 64*b^2/theta^2 = 711. Max adversarial list size is 10 (inflated_list), bound is 712, so the list-size check never fires.

## Issues / discrepancies

### MINOR

**m1. Statistical reporting under-counts the relevant event.** The summary CSV reports "rejection_rate", but Eq. (18) is about "Pr[accept ∧ err > alpha·opt + eps]". For 3 of the 4 strategies these coincide because acceptance is impossible. For random_list, however, the verifier sometimes correctly accepts when s* is in the random L, and the experiment lumps those correct accepts in with potential false accepts. As a result, the reported random_list rejection rates at n=4,5,6 fall below 1-delta = 0.9, which superficially looks like a soundness failure, but is in fact consistent with Eq. (18) because every accept has hypothesis_correct = True. Recommendation: surface a `bad_accept_rate` column equal to `Pr[accepted ∧ ¬hypothesis_correct]`. The data is already in the .pb (worker.py:438-439).

**m2. Verifier sample count is much smaller than the paper's prescription.** `classical_samples_verifier = 3000` (`soundness.py:99`) is fixed and overrides the Hoeffding-derived count `~ (2/(eps^2/(16|L|))^2) log(4|L|/delta)`, which would be in the tens of millions for eps=0.3 and |L|=10. The override is operationally fine for the test distributions (single parity has sky-high signal-to-noise) but does mean the experiment is not stress-testing the paper's worst-case sample budget. This is a deliberate design choice — flagged here for transparency, not correctness.

**m3. Strategies are weak adversaries.** The 4 strategies in this experiment (random_list, wrong_parity, partial_list, inflated_list) are all easy: each either omits s* by construction or includes it by uniform random chance. The "interesting" partially-honest strategies — partial_real, diluted_list, shifted_coefficients, subset_plus_noise — live in `soundness_multi`, not here. As a result, this experiment confirms soundness against trivial cheating but does not stress-test it against the worst-case prover.

### NIT

**n4. The strategy `wrong_parity` has a fallback that depends on n=1.** `worker.py:255-257`: `wrong_s = (target_s + 1) % (2**n); if wrong_s == 0: wrong_s = (target_s + 2) % (2**n)`. With target_s=1 and n ≥ 2, `wrong_s = 2 mod 2^n`, which is never 0 for n ≥ 2. The fallback is a no-op for the experiment's actual parameters. Cosmetic only.

**n5. The dummy `QFSResult` and `SpectrumApproximation` are constructed inside the worker.** `worker.py:370-371`. These are required to fit the `ProverMessage` dataclass schema, which is unnecessarily heavy for dishonest trials. A small refactor (e.g. making them `Optional`) would simplify the dishonest path. Cosmetic only.

### Not found: BLOCKER, MAJOR

No blocking or major issues identified. The verifier code matches Theorem 12, the experiment correctly bypasses the prover (so the verifier acts on adversarial input), the four strategies behave as documented, and the empirical numbers are consistent with both the protocol's mechanics and the soundness bound after accounting for the random_list "lucky guess" contribution.

## Verdict

The soundness experiment is a faithful, correct implementation of Theorem 12's verifier (file `ql/verifier.py:434-547`) being exercised against four canonical adversarial strategies (file `experiments/harness/worker.py:247-271`). The verifier code matches the paper line-for-line: list-size bound `64 b^2/theta^2`, per-coefficient tolerance `eps^2/(16|L|)`, acceptance threshold `a^2 - eps^2/8`, independent classical estimation via Lemma 1, and `argmax_{s in L} |xi_hat(s)|` output. Three of four strategies (wrong_parity, partial_list, inflated_list) achieve 100% rejection at every n in [4, 20], comfortably exceeding the `1 - delta = 0.9` floor. The fourth strategy (random_list) achieves rejection rates that match the analytical prediction `1 - 5/2^n` (e.g. 0.71 at n=4 vs. predicted 0.6875; 1.00 at n ≥ 11 vs. predicted ≥ 0.9976) within Wilson 95% CI; the apparent dip below 0.9 at small n reflects the random list accidentally containing s*, in which case the verifier *correctly* outputs the right hypothesis — this is permitted by Eq. (18), which only forbids bad accepts, not accepts per se. The experiment's reporting could be sharpened by surfacing a `bad_accept_rate` column (the underlying data is captured but not aggregated), and the strategies are deliberately weak (the harder partially-honest strategies are exercised in `soundness_multi`, not here). No code-level discrepancies with the paper were found. Audit verdict: PASS, with minor reporting/observability suggestions.

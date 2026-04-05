# Experiment Gap Analysis: Validating Against Caro et al.

This document maps every claim in the verification protocol of
[Caro et al. (2306.04843)](https://arxiv.org/abs/2306.04843) against the
current experiment harness, identifies gaps, and proposes concrete
experiments to close them.

---

## 1. Paper Protocol Summary

The paper's verification framework (Section 6) covers four settings, each
with a dedicated theorem:

| Setting | Theorem | Key parameters | Hypothesis type |
|---|---|---|---|
| Functional parity (SQ) | 7 | theta, epsilon, delta | Single parity chi_s |
| Functional parity (RE) | 8 | theta, epsilon, delta | Single parity chi_s |
| Functional k-sparse (SQ) | 9 | theta, epsilon, delta, k | Randomised k-term Fourier sum |
| Functional k-sparse (RE) | 10 | theta, epsilon, delta, k | Randomised k-term Fourier sum |
| Distributional parity (SQ) | 11 | theta, epsilon, delta, a^2, b^2 | Single parity chi_s |
| Distributional parity (RE) | 12 | theta, epsilon, delta, a^2, b^2 | Single parity chi_s |
| Distributional k-sparse (SQ) | 14 | theta, epsilon, delta, a^2, b^2, k | Randomised k-term Fourier sum |
| Distributional k-sparse (RE) | 15 | theta, epsilon, delta, a^2, b^2, k | Randomised k-term Fourier sum |

(SQ = statistical query access, RE = random example access.)

### 1.1 Shared Protocol Structure (all theorems)

Every verification protocol follows the same four steps:

1. **V asks P for a list L** of pairwise distinct n-bit strings with
   |L| <= C/theta^2, where C depends on the setting (4 for functional
   parity, 64b^2 for distributional parity, 64b^2 for distributional
   k-sparse).

2. **P runs QFS (Corollary 5)** to produce a succinctly represented
   phi_tilde, then sends L = {s : |phi_tilde(s)| >= theta/2} to V.
   The completeness guarantee is that if |phi_hat(s)| >= theta then
   s in L, and if s in L then |phi_hat(s)| >= theta/2.

3. **V checks |L| against the Parseval bound**, then independently
   estimates phi_hat(s) for all s in L using classical samples.
   Estimation tolerance depends on the setting.

4. **V checks accumulated Fourier weight** sum_{l} (phi_hat_est(s_l))^2
   against a threshold:
   - Parity: a^2 - epsilon^2/8
   - k-sparse: a^2 - epsilon^2/(128k^2)

   If the weight is sufficient, V outputs a hypothesis:
   - Parity: s_out = argmax_{s in L} |phi_hat_est(s)|
   - k-sparse: top-k from L, randomised hypothesis via Lemma 14

### 1.2 Key Theoretical Constraints

- **List size bounds**: |L| <= 4/theta^2 (Thm 7), 64/theta^2 (Thm 8),
  4/theta^2 (Thm 9), 64b^2/theta^2 (Thms 11-12, 14-15).
- **Accuracy lower bound** (Theorem 13): epsilon >= 2*sqrt(b^2 - a^2)
  is necessary for sublinear-in-n sample complexity.
- **Definition 14 (L2-bounded bias)**: The distributional case requires
  a promise that E[phi(x)^2] in [a^2, b^2]. The functional case has
  a = b = 1 (noiseless) or a = b = (1 - 2*eta) (noisy).

---

## 2. Current Experiment Coverage

### 2.1 Experiment Inventory

| Experiment | Module | Phi function(s) | Expected |L| | Verifier path | Paper theorems |
|---|---|---|---|---|---|
| scaling | `scaling.py` | `make_random_parity` | 1 | `verify_parity` | 8, 12 |
| noise | `noise.py` | `make_random_parity` + label flip | 1 | `verify_parity` | 11, 12 |
| truncation | `truncation.py` | `make_single_parity(n, 1)` | 1 | `verify_parity` | 12 |
| bent | `bent.py` | `make_bent_function` | 0 or many (flat spectrum) | `verify_parity` | Corollary 5 |
| average_case | `average_case.py` | `make_k_sparse`, `make_random_boolean`, `make_sparse_plus_noise` | >1 | `verify_parity` | 7, 9 (partially) |
| soundness | `soundness.py` | `make_single_parity(n, 1)` + dishonest prover | varies | `verify_parity` | 7 (soundness) |
| gate_noise | `gate_noise.py` | `make_random_parity` + depolarising circuit noise | 1 | `verify_parity` | empirical (beyond paper) |
| soundness_multi | `soundness_multi.py` | `make_k_sparse` + dishonest prover (multi-element L) | varies | `verify_parity` | 7, 12 (soundness, multi-element) |
| theta_sensitivity | `theta_sensitivity.py` | `make_sparse_plus_noise` | 1-4+ (theta-dependent) | `verify_parity` | Corollary 5, Gap 5 |
| ab_regime | `ab_regime.py` | `make_sparse_plus_noise` | 1 | `verify_parity` | 11, 12 (Definition 14) |

### 2.2 What Is Well-Covered

- **Single-parity completeness**: scaling experiment sweeps n = 4..12 with
  random parities, demonstrating that QFS extraction and parity
  verification work for |L| = 1 (Theorem 8/12 restricted case).

- **Label-flip noise robustness**: noise experiment sweeps eta = 0..0.4
  and adapts theta, a^2, b^2 = (1-2*eta)^2 per the paper's Definition 12.
  Tests the noisy functional case.

- **Verifier sample budget**: truncation experiment sweeps (epsilon, m_V)
  at fixed n = 6, eta = 0.15, mapping the Hoeffding-based acceptance
  frontier (Theorem 12, Step 4).

- **Basic soundness**: four dishonest prover strategies (random_list,
  wrong_parity, partial_list, inflated_list) against single-parity
  targets. Tests that independent coefficient estimation rejects
  fabricated lists.

- **Gate-level noise**: empirical investigation of QFS under depolarising
  circuit noise, beyond the paper's theoretical scope.

- **Multi-element phi functions exist**: `make_k_sparse`, `make_random_boolean`,
  and `make_sparse_plus_noise` in `phi.py` do construct functions with
  multiple nonzero Fourier coefficients, and are used in `average_case.py`.

### 2.3 Parity Function Clarification

`make_single_parity(n, target_s)` computes phi(x) = (s* . x) mod 2.
This evaluates to 1 for exactly 2^(n-1) input strings (half), not just
one. However, its Fourier transform phi_tilde = chi_{s*} has exactly
**one** nonzero coefficient: phi_hat(s*) = 1, all others = 0. The
function is Fourier 1-sparse, which is why L is always a singleton for
single-parity experiments.

---

## 3. Identified Gaps

### Gap 1: `verify_fourier_sparse()` is never called (CRITICAL)

**Location**: `ql/verifier.py:367-428`

The verifier has a complete `verify_fourier_sparse(msg, ..., k=k)` method
that implements the k-sparse verification protocol (Theorems 9/10/14/15):

- Acceptance threshold: `a^2 - epsilon^2 / (128 * k^2)`
  (vs `a^2 - epsilon^2 / 8` for parity)
- Hypothesis construction: selects top-k heaviest from L, builds a
  randomised k-term Fourier hypothesis via Lemma 14
- Different Parseval list-size bound

**No experiment ever calls this method.** The `average_case` experiment
uses k-sparse phi functions (`make_k_sparse` with k=2,4) but routes
them through `verify_parity()` in the worker (`worker.py:142`). This
means:

1. The k-sparse verification code path is entirely untested.
2. The average_case experiment measures single-parity correctness
   (argmax) against functions that genuinely have multiple heavy
   coefficients -- a mismatch with the paper's intended evaluation.
3. Any bugs in `verify_fourier_sparse()`, `_build_fourier_sparse_hypothesis()`,
   or the k-sparse threshold calculation are invisible.

**Impact**: The implementation of Theorems 9, 10, 14, 15 has zero
empirical validation.

### Gap 2: a^2 != b^2 regime never tested (HIGH) -- CLOSED

**Location**: All experiment modules set `a_sq = b_sq`.

- `scaling.py`: `a_sq = b_sq = 1.0` (implicit, functional case)
- `noise.py:85-86`: `a_sq = b_sq = effective_coeff**2`
- `average_case.py:48`: `a_sq = b_sq = pw`
- `truncation.py:99`: `a_sq = b_sq = effective_coeff**2`

The paper's Definition 14 defines the distribution class
D_{U_n;[a^2,b^2]} with distinct a and b. The acceptance threshold
`a^2 - epsilon^2/8` depends on the lower bound a^2, while the list
size bound uses b^2. When a < b:

- The acceptance threshold drops (harder to accept).
- The list size bound grows (prover may send more elements).
- The accuracy is bounded below by epsilon >= 2*sqrt(b^2 - a^2)
  (Theorem 13).

With a = b, the threshold is at its most lenient and the accuracy
bound is vacuous (epsilon >= 0). Any off-by-one errors in the
threshold or list-size calculations would be hidden.

**Impact**: Bugs in how a^2 and b^2 are used separately in the
verifier's threshold vs list-size checks are invisible.

**Status**: Closed by `experiments/harness/ab_regime.py`. The experiment
sweeps `gap = b_sq - a_sq` over `{0.0, 0.05, 0.1, 0.2, 0.3, 0.4}`
using `make_sparse_plus_noise` (pw ~ 0.52), setting
`a_sq = pw - gap/2`, `b_sq = pw + gap/2`. This exercises the verifier's
acceptance threshold (`a^2 - epsilon^2/8`) and list-size bound
(`64*b^2/theta^2`) under distinct `a_sq` and `b_sq` values, validating
the Definition 14 parameter space. Registered as CLI subcommand
`ab_regime`.

### Gap 3: Multi-element L soundness not tested (HIGH) — CLOSED

**Location**: `soundness.py` only tests against `make_single_parity(n, 1)`.

The four dishonest strategies all target a function with a single
heavy coefficient. The paper's soundness argument (e.g. equations
73-77 in Theorem 7 proof, 110-114 in Theorem 12 proof) relies on
the accumulated weight check detecting that the *total* Fourier
weight on L is insufficient. For multi-element target functions:

- An adversary could include some real heavy coefficients alongside
  fake ones, potentially accumulating enough weight to pass the
  threshold.
- The soundness proof shows this still fails because the verifier's
  independent estimates of fake coefficients will be near zero, but
  this has never been tested empirically.

**Impact**: Soundness of the accumulated weight check against
partially-correct adversarial lists is unvalidated.

**Resolution**: Implemented in `experiments/harness/soundness_multi.py` with
four multi-element dishonest strategies (`partial_real`, `diluted_list`,
`shifted_coefficients`, `subset_plus_noise`) targeting `make_k_sparse`
functions. The worker (`worker.py:_run_dishonest_trial`) was extended with
these strategies, which compute the true Fourier spectrum via WHT to
construct partially-correct adversarial lists. Fixed `_run_dishonest_trial`
to pass `a_sq`, `b_sq`, and `theta` from `TrialSpec` instead of hardcoding
defaults. CLI: `python -m experiments.harness soundness_multi`.

### Gap 4: No systematic k sweep (MEDIUM)

**Location**: `average_case.py:17` hardcodes `k_sparse_2` and `k_sparse_4`.

The paper's Theorems 9/14 have k-dependent complexity:
- Estimation tolerance: epsilon^2 / (256 * k^2 * |L|)
- Acceptance threshold: a^2 - epsilon^2 / (128 * k^2)
- Accuracy guarantee: 2-agnostic, epsilon >= 4k * sqrt(b^2 - a^2)

With only k=2 and k=4, there is no visibility into whether the
protocol degrades correctly as k grows (tighter threshold, more
samples needed, larger L).

**Impact**: Cannot detect k-scaling bugs in sample budget calculations
or threshold formulas.

### Gap 5: Theta near coefficient magnitude (MEDIUM) -- ADDRESSED

**Location**: Various theta-setting logic across experiments.

The extraction threshold `theta^2/4` determines what enters L.
When theta is near the actual magnitude of a Fourier coefficient,
the coefficient may or may not enter L depending on finite-sample
noise. The paper's Corollary 5 guarantees:
- If |phi_hat(s)| >= theta, then s in L (completeness)
- If s in L, then |phi_hat(s)| >= theta/2 (partial soundness)

The gap between theta and theta/2 is where the protocol is most
fragile. The bent experiment tests the extreme case (all coefficients
equal, at 2^{-n/2}), but no experiment systematically sweeps theta
near real coefficient magnitudes for sparse functions.

**Impact**: Cannot detect finite-sample failures at the extraction
boundary.

**Status**: Addressed by `experiments/harness/theta_sensitivity.py`.
Sweeps theta in {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50} against
`make_sparse_plus_noise` (dominant 0.7, secondary 0.1 each). At theta ~ 0.20,
secondary coefficients sit at the extraction boundary (|phi_hat| = 0.1 vs
threshold theta/2 = 0.10). Records |L|, acceptance outcome, and accumulated
weight per trial.

### Gap 6: Correctness metric mismatch for k-sparse (LOW-MEDIUM)

**Location**: `worker.py:154`

```python
hyp_s = result.hypothesis.s if result.accepted and result.hypothesis else None
correct = hyp_s == spec.target_s if hyp_s is not None else False
```

For k-sparse functions, correctness should be measured against the
paper's 2-agnostic guarantee (Corollary 7, equation 50):

  P[b != h(x)] <= 2 * min_{f: k-sparse} P[b != f(x)] + epsilon

The current metric checks whether the *single* output parity matches
`target_s` (the heaviest coefficient). This is:
- Too strict for k-sparse: a correct k-term hypothesis may have a
  different argmax than the single heaviest coefficient.
- Not measuring the agnostic guarantee: should compare
  misclassification probability, not index equality.

**Impact**: The correctness metric for multi-coefficient functions
does not reflect the paper's actual guarantee.

---

## 4. Recommended Experiments

### Experiment A: k-Sparse Verification Path (closes Gaps 1, 4, 6)

**Goal**: Exercise `verify_fourier_sparse()` and validate Theorems 9/14.

**Design**:
- Sweep k = {1, 2, 4, 8} with `make_k_sparse(n, k, rng)`.
- For each trial, run the protocol through `verify_fourier_sparse(msg, ..., k=k)`
  instead of `verify_parity()`.
- Sweep n = {4, 6, 8, 10}.
- Compare against `verify_parity()` on the same functions to detect
  divergences.

**Changes required**:
1. Add a `verification_mode` field to `TrialSpec` (or a `k_sparse` int)
   so the worker can dispatch to `verify_fourier_sparse()`.
2. Extend `TrialResult` to capture the k-sparse hypothesis
   (list of (s, coefficient) pairs) rather than a single `hypothesis_s`.
3. Add a misclassification-rate metric: sample fresh (x, y) pairs,
   evaluate h(x) = sum_l phi_hat(s_l) * chi_{s_l}(x) via Lemma 14,
   and compute empirical P[h(x) != y].
4. New experiment module `experiments/harness/k_sparse.py`.

**What bugs this finds**:
- Threshold formula errors in `verify_fourier_sparse()`.
- Bugs in `_build_fourier_sparse_hypothesis()` (top-k selection,
  randomised hypothesis construction).
- Sample budget miscalculations with k-dependent tolerances.
- Any issue where verify_fourier_sparse silently accepts/rejects
  incorrectly because it has never been run.

### Experiment B: a^2 != b^2 Regime (closes Gap 2)

**Goal**: Validate the acceptance threshold and list-size bound under
the full Definition 14 parameter space.

**Design**:
- Fix a k-sparse function (e.g., make_sparse_plus_noise with known
  Parseval weight pw ~ 0.52).
- Set `a_sq = pw - gap/2`, `b_sq = pw + gap/2` for various gap values.
- Sweep gap = {0.0, 0.05, 0.1, 0.2, 0.3, 0.4}.
- For each gap, set epsilon = 2 * sqrt(b^2 - a^2) + margin, sweeping
  margin to find the acceptance frontier.
- Compare observed acceptance rate against the theoretical prediction:
  acceptance when epsilon >= 2*sqrt(b^2 - a^2), rejection below.

**Changes required**:
1. New experiment module `experiments/harness/ab_regime.py`.
2. No changes to prover/verifier needed -- just parameter variation.

**What bugs this finds**:
- Incorrect use of a^2 vs b^2 in threshold calculation.
- List size bound using wrong parameter.
- Edge cases where the acceptance margin is near zero.

### Experiment C: Multi-Element L Soundness (closes Gap 3)

**Status**: Implemented — `experiments/harness/soundness_multi.py`.

**Goal**: Verify that the accumulated weight check rejects adversarial
lists even when the adversary includes some genuine heavy coefficients.

**Design**:
- Use `make_k_sparse(n, k=4, rng)` as the target function.
- Dishonest strategies adapted for multi-element targets:
  - `"partial_real"`: include 2 of 4 real heavy coefficients plus
    3 fake ones. Tests whether partial real weight passes threshold.
  - `"diluted_list"`: include all real coefficients but pad with
    20 random indices. Tests list-size rejection.
  - `"shifted_coefficients"`: include real indices but with
    inflated/deflated coefficient estimates. Tests whether
    independent estimation detects mismatch.
  - `"subset_plus_noise"`: include 1 real heavy coefficient plus
    several near-threshold fake ones. Tests the marginal case.
- Sweep n = {4, 6, 8}, k = {2, 4}.

**Changes required**:
1. Extend `_run_dishonest_trial()` in `worker.py` with new strategies.
2. These strategies need access to the true Fourier spectrum to
   construct partially-correct adversarial lists.
3. New experiment module `experiments/harness/soundness_multi.py`.

**What bugs this finds**:
- Accumulated weight check accepting partially-correct lists.
- List-size bound not enforced correctly for multi-element L.
- Edge cases in the independent estimation when some coefficients
  are real and some are fabricated.

### Experiment D: Theta Sensitivity (closes Gap 5) -- IMPLEMENTED

**Goal**: Map the extraction boundary where coefficients enter/exit L.

**Design**:
- Use `make_sparse_plus_noise(n, rng)`: dominant coefficient 0.7,
  secondary coefficients 0.1 each.
- Sweep theta = {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50}.
- For each theta, record:
  - |L| (how many coefficients extracted)
  - Whether dominant coefficient is in L
  - Whether secondary coefficients are in L
  - Acceptance rate and hypothesis correctness
- At theta ~ 0.1, secondary coefficients are at the extraction
  boundary (|phi_hat| = 0.1 vs threshold theta/2 = 0.05). This is
  where finite-sample noise determines inclusion.

**Changes required**:
1. New experiment module `experiments/harness/theta_sensitivity.py`.
2. Extend `TrialResult` or use a separate result type to capture
   per-coefficient extraction status.

**What bugs this finds**:
- Extraction threshold calculation errors.
- DKW-based spectrum approximation failures at boundary.
- Interaction between theta and the acceptance threshold when
  marginal coefficients affect accumulated weight.

**Implementation**: `experiments/harness/theta_sensitivity.py`
CLI: `python -m experiments.harness theta_sensitivity`

---

## 5. Implementation Priority

Ordered by bug-finding value:

1. **Experiment A (k-sparse path)**: Exercises entirely dead code.
   Highest probability of finding bugs because `verify_fourier_sparse()`
   has never been run. Requires the most implementation work
   (worker routing, new result fields).

2. **Experiment B (a^2 != b^2)**: Pure parameter variation, minimal
   code changes. Tests a subtle regime that the current experiments
   completely avoid. Quick to implement.

3. **Experiment C (multi-element soundness)**: Moderate implementation
   effort. Tests the most security-critical property (soundness)
   in the general case.

4. **Experiment D (theta sensitivity)**: Mostly diagnostic. Lower
   bug-finding priority but useful for understanding protocol
   margins.

---

## 6. Appendix: Code Pointers

| Component | File | Key lines | Notes |
|---|---|---|---|
| Heavy list extraction | `ql/prover.py` | 450-497 | Sorts by empirical weight, Parseval truncation |
| Spectrum approximation | `ql/prover.py` | 370-444 | Threshold: theta^2/4, conditional on b=1 |
| Coefficient estimation (prover) | `ql/prover.py` | 503-587 | Hoeffding: m = ceil(2/eps^2 * log(4|L|/delta)) |
| Parity verification | `ql/verifier.py` | 300-365 | Threshold: a^2 - eps^2/8, argmax hypothesis |
| k-sparse verification | `ql/verifier.py` | 367-428 | Threshold: a^2 - eps^2/(128k^2), top-k hypothesis |
| Core verify logic | `ql/verifier.py` | 434-547 | List size check, estimation, weight check |
| Parity hypothesis | `ql/verifier.py` | 617-638 | s_out = argmax |
| k-sparse hypothesis | `ql/verifier.py` | 640-664 | Top-k selection, Lemma 14 construction |
| Independent estimation | `ql/verifier.py` | 553-611 | Verifier's own classical samples |
| Worker dispatch | `experiments/harness/worker.py` | 140-150 | Always calls verify_parity, never verify_fourier_sparse |
| Phi: single parity | `experiments/harness/phi.py` | 10-32 | phi(x) = s* . x mod 2, Fourier 1-sparse |
| Phi: k-sparse | `experiments/harness/phi.py` | 133-180 | Dirichlet coefficients, returns heaviest as target_s |
| Phi: sparse+noise | `experiments/harness/phi.py` | 220-264 | Dominant 0.7 + 3 secondary 0.1 |
| Phi: random boolean | `experiments/harness/phi.py` | 183-217 | Full 2^n spectrum, WHT to find heaviest |
| Average case experiment | `experiments/harness/average_case.py` | 39-76 | Uses k-sparse phi but parity verifier |
| Soundness experiment | `experiments/harness/soundness.py` | 81 | Single parity target only |
| Trial result | `experiments/harness/results.py` | 21-94 | hypothesis_s is single int, no k-sparse fields |

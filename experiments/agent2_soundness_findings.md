# Soundness experiments — dissertation findings

> Audit target: `experiments/harness/soundness.py` and `experiments/harness/soundness_multi.py` against `papers/classical_verification_of_quantum_learning.pdf` (Caro, Hinsche, Ioannou, Nietner, Sweke; arXiv:2306.04843v2). The two experiments together are a finite-strategy spot-check of the universal soundness condition Definition 7 / Eq. (18). The verifier protocol code under `/Users/alex/cs310-code/ql/verifier.py` was previously audited line-by-line and is faithful to Theorems 12 and 15 of the paper; this report focuses on whether the empirical experiments exercise that protocol in a way that supports the claims made in the figures.

---

## Universal soundness condition (Definition 7, Eq. 18, p. 17)

The paper's soundness condition is Definition 7 (p. 17, Eq. 18). For any distribution `D ∈ D` and any (possibly unbounded) dishonest prover `P'`, the random hypothesis `h : X_n → {0,1}` that `V(ε,δ)` outputs after interacting with `P'` must satisfy

```
Pr[ h ≠ reject  ∧  err_D(h) > α · opt_D(F) + ε ]  ≤  δ.       (Eq. 18)
```

Three properties matter for this audit:

1. **Joint event.** Eq. (18) bounds the *joint* probability that V accepts AND the output hypothesis has high error. It does NOT bound `Pr[V accepts]` and it does NOT bound `Pr[err > α·opt+ε | V accepts]`. V is permitted to accept arbitrarily often, provided that whenever it accepts the hypothesis is good.
2. **Universal quantifier over P'.** The statement holds for every possibly unbounded P'. **The two experiments test only a finite, fixed menu of 4+4 = 8 hand-written strategies, so they constitute an empirical spot-check, not a proof of soundness.**
3. **Acceptance with a correct hypothesis is permitted.** If P' stumbles into a list containing the true heavy frequencies, V is allowed (and required) to accept. Such accepts are not Eq. (18) violations.

The relevant protocol clauses are:

- **Theorem 12 (parities, distributional, p. 45).** Step 3: V rejects if `|L| > 64 b²/θ²`; otherwise V draws its OWN classical examples and estimates each `ξ̂(s)` to tolerance `ε²/(16|L|)`. Step 4: V accepts iff `Σ_{s∈L} ξ̂(s)² ≥ a² − ε²/8` and outputs `s_out = argmax_{s∈L} |ξ̂(s)|`. The soundness analysis (Eqs. 110–114, p. 46) only relies on V succeeding in Step 3 (its own Chernoff–Hoeffding union bound), not on P' being honest.
- **Theorem 15 (k-sparse, distributional, pp. 48–49).** Same shape with: per-coefficient tolerance `ε²/(256 k² |L|)` (Step 3, p. 48), acceptance threshold `a² − ε²/(128 k²)` (Step 4, p. 48), and output as the randomised hypothesis from Lemma 14 built from the k heaviest entries of L. The soundness proof (Eqs. 119–127, pp. 49–50) again only relies on the verifier's own Chernoff–Hoeffding step.

The verifier code at `/Users/alex/cs310-code/ql/verifier.py:434-553` implements both verbatim: list-size bound `int(np.ceil(64.0 * b_sq / theta**2))` at `verifier.py:464`; per-coefficient tolerance branch on `HypothesisType` at `verifier.py:486-491` giving `ε²/(16|L|)` for parity and `ε²/(256·k²·|L|)` for Fourier-sparse; acceptance threshold branch at `verifier.py:513-518` giving `a² − ε²/8` and `a² − ε²/(128 k²)` respectively; independent classical estimation in `_estimate_coefficients_independently` at `verifier.py:559-617`.

---

## 1. Single-parity soundness (`soundness`)

### 1(a) Data and setup

| Parameter | Value | Source |
|---|---|---|
| Data file | `/Users/alex/cs310-code/results/soundness_4_20_100.pb` | dissertation run |
| `n` range | 4..20 (17 values) | `parameters.nRange` in the .pb |
| Trials per (n, strategy) | 100 | `parameters.numTrials` |
| Strategies | `random_list`, `wrong_parity`, `partial_list`, `inflated_list` | `/Users/alex/cs310-code/experiments/harness/soundness.py:70` |
| Total trials | 6 800 | 17 × 4 × 100 |
| `ε` | 0.3 | `soundness.py:92` |
| `δ` | 0.1 | `soundness.py:93` |
| `θ` | `ε = 0.3` | `soundness.py:94` |
| `a² = b²` | 1.0 | `soundness.py:95-96` (single pure parity) |
| Target function | `χ_{s*}` with `s* = 1` | `soundness.py:80-81` (`make_single_parity(n, 1)`) |
| Verifier classical samples | 3 000 (override) | `soundness.py:99` |

The four cheating strategies are implemented at `/Users/alex/cs310-code/experiments/harness/worker.py:251-274` and dispatched from `_run_dishonest_trial` at `worker.py:349-465`:

- **`_strategy_random_list`** (`worker.py:251-254`): sends a sorted list of `min(5, 2^n)` indices drawn uniformly at random without replacement, claimed coefficients all 0.0. With probability `5/2^n` the random list contains `s* = 1`.
- **`_strategy_wrong_parity`** (`worker.py:257-262`): sends a single index `(s* + 1) mod 2^n = 2` (for our `s* = 1`, `n ≥ 2`), claimed coefficient 1.0. Never includes `s*` by construction.
- **`_strategy_partial_list`** (`worker.py:265-267`): sends an empty list `[]`. Accumulated weight is exactly 0 → always rejected by Step 4.
- **`_strategy_inflated_list`** (`worker.py:270-274`): sends 10 indices drawn uniformly from `{s : s ≠ s*}` with fabricated estimate 0.5 each. `s*` is excluded by construction.

The dishonest worker constructs the same `MoSState` as an honest run (`worker.py:128`), so the verifier's `_estimate_coefficients_independently` call at `verifier.py:505-508` draws from the genuine distribution `D = (U_n, χ_{s*})`. The strategy-supplied `estimates` field is **never read** by `_verify_core`; only the list `L` reaches the verifier.

### 1(b) Precise paper claim being tested

Eq. (18) with `α = 1`, `ε = 0.3`, `δ = 0.1`, and `opt_D(F) = 0` (single pure parity) becomes

```
Pr[ V accepts ∧ err_D(h) > 0.3 ] ≤ 0.1   for every (possibly unbounded) P'.
```

This is a universal statement; the harness instantiates exactly four P' algorithms. **Passing these four strategies is a necessary but very far from sufficient condition for the universal statement of Eq. (18).** The harness docstring at `soundness.py:23-49` phrases the test as "rejection rate ≥ 1−δ" rather than "bad-accept rate ≤ δ"; the audit (`/Users/alex/cs310-code/audit/soundness.md`, finding m1) flagged this and the plot script was updated to surface the correct quantity.

For these parameters the numerics are:

- list-size bound `64·b²/θ² = ⌈64/0.09⌉ = 712`,
- acceptance threshold `1 − ε²/8 = 0.98875`,
- per-coefficient tolerance `ε²/(16·|L|) = 0.005625/|L|`,
- with the override `num_samples = 3000`, the empirical standard error of each `ξ̂(s)` is `≈ 1/√3000 ≈ 0.0183`, so the accumulated noise floor on `Σξ̂²` is `|L|·(1/3000) ≈ 3.3×10⁻³` for `|L|=10`, six orders of magnitude below the 0.989 threshold. The override is operationally sufficient though far below the Hoeffding-derived budget (~10⁷ samples) that the paper formally prescribes — the auto-derivation branch at `verifier.py:493-501` is bypassed by the explicit override.

### 1(c) What each figure shows

Three artefacts in `/Users/alex/cs310-code/results/figures/soundness/`, generated by `plot_soundness.py`.

**`rejection_by_strategy.png`** — two-panel grouped bar chart, strategies on the x-axis, one bar per representative n. The **left panel** plots the Definition 7 indicator `1 − Pr[accept ∧ wrong] = Pr[reject ∨ (accept ∧ correct)]` with a grey dashed `1 − δ = 0.9` reference. In the rendered figure every bar is exactly 1.00 in every (strategy, n) cell, comfortably above 0.9, because the dataset contains zero accepted-wrong trials out of 6 800. The **right panel** plots the raw rejection rate `Pr[reject]`; here `random_list` visibly dips below 0.9 at n=4 (0.71) and n=5 (0.88), and the title annotation explicitly labels these as lucky correct accepts. This split-panel treatment is the audit fix m1 and is materially better than any single-panel raw-rejection figure.

**`rejection_mechanism.png`** — single stacked bar chart, all n pooled, one bar per strategy. The accept band is split into "Accept (correct hypothesis)" (green) and "Accept ∧ wrong (Eq. 18 event; 0/6800)" (red, legend-annotated as empty). Reject is split into "Reject: insufficient weight" (Step 4) and "Reject: list too large" (Step 3). The figure auto-suppresses the list-too-large band when empty. **Annotation results:** `random_list` shows "96.7% rejected, 3.3% lucky accept"; the other three strategies show "100.0% rejected".

**`rejection_mechanism_by_n.png`** — 2×2 faceted version, one panel per strategy. The `random_list` panel shows the green "lucky accept" sliver shrinking from ~0.29 at n=4 to 0 at n≥11, matching the analytical `5/2^n` prediction. The other three panels are uniformly blue (insufficient weight) at every n.

**Rejection mechanism identification:** for every reject in this experiment, the binding constraint is the **Step 4 weight check** — never the Step 3 list-size check. The max adversarial list size is 10 (inflated_list), and the bound is 712. The plot scripts auto-hide the list-size band when empty.

### 1(d) Per-cell rejection vs `1 − δ = 0.9`

Re-decoded counts from `results/soundness_4_20_100.pb`:

| strategy | n=4 | n=5 | n=6 | n=7 | n=10 | n=15 | n=20 |
|---|---|---|---|---|---|---|---|
| random_list raw rej | 0.71 | 0.88 | 0.98 | 0.95 | 0.99 | 1.00 | 1.00 |
| wrong_parity raw rej | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| partial_list raw rej | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| inflated_list raw rej | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **all 4 strategies bad-accept** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

**Bad-accept rate is exactly 0/100 in every one of the 68 (strategy, n) cells.** The Eq. (18) bound is satisfied with the maximum possible margin everywhere; the Wilson 95% upper bound on a 0/100 estimate is 0.0370, well below δ = 0.1.

Cells where the **raw rejection rate** is below 0.9: `random_list` at n=4 (0.71) and n=5 (0.88). Explained entirely by lucky correct accepts:

| n | predicted Pr[s* ∉ L] = 1 − 5/2ⁿ | observed Pr[reject] | accept-correct count |
|---|---|---|---|
| 4 | 0.6875 | 0.71 | 29/100 |
| 5 | 0.8438 | 0.88 | 12/100 |
| 6 | 0.9219 | 0.98 | 2/100 |
| 7 | 0.9609 | 0.95 | 5/100 |
| 10 | 0.9951 | 0.99 | 1/100 |
| 11+ | ≥ 0.9976 | 1.00 | 0/100 |

Observed `accept-correct` counts match the `5/2ⁿ` analytic prediction within Wilson 95% intervals. **Zero accept-wrong trials in any cell.**

### 1(e) Hypothesis-error tracking on dishonest trials

For the single-parity experiment, "wrong hypothesis" and "high-error hypothesis" are equivalent: the target is the single pure parity `χ_1`, so any output `s_out ≠ 1` gives err exactly 0.5 (orthogonality), and `s_out = 1` gives err 0. The plot script derives `hypothesis_correct` from `hypothesisS == 1` at `plot_soundness.py:113`, which is exact for this target. For the k-sparse experiment this equivalence breaks down — see §2(e).

A nuance in the decoded .pb: the `hypothesisCorrect` field is not present in the camelCase JSON (verified via decode). `plot_soundness.py:109-123` notes this and derives correctness from `hypothesisS == target_s = 1` instead, with a runtime assertion that would fire if a future .pb populated `hypothesisCorrect` inconsistently. This is correct for the current target but is fragile against future target changes.

### 1(f) Verdict: **PASS**

The verifier code at `verifier.py:434-553` implements Theorem 12 verbatim. The four strategies are correctly dispatched, correctly bypass the prover, and exercise the verifier on adversarial input. Bad-accept rate is 0/6 800 across all cells; the Definition 7 indicator is 1.00 everywhere with Wilson upper bound 0.037 on the bad-accept estimate (well below δ = 0.1). The two-panel grouped bar chart and the 4-way mechanism breakdowns are faithful to the data and explicitly handle the random_list lucky-correct-accept pathway. Reject mechanism is the Step 4 weight check in every cell; the Step 3 list-size bound is never binding.

### 1(g) Open issues / follow-ups

- **Strategy menu is shallow.** All four strategies either exclude `s*` by construction or include it by uniform random luck. There is no "near-honest" strategy where the prover sends `s*` plus confounders with fabricated coefficients — but such a strategy is structurally impossible to make adversarial in the pure-parity setting, because the verifier's empirical `ξ̂(s*) ≈ 1` will dominate any noise term. Harder strategies live in `soundness_multi`.
- **Verifier sample budget is 3 000, not the Hoeffding-derived amount (~10⁷).** `soundness.py:99` overrides the auto-derived value. This is a deliberate engineering choice flagged as m2 in `audit/soundness.md`; it does not affect the qualitative result for the pure-parity case.
- **Single fixed target `s* = 1` across all trials.** The protocol's correctness does not depend on the target value, so a randomised-target sweep would only inflate trial cost. Not a real follow-up.

---

## 2. k-sparse soundness (`soundness_multi`)

### 2(a) Data and setup

| Parameter | Value | Source |
|---|---|---|
| Data file | `/Users/alex/cs310-code/results/soundness_multi_4_16_100.pb` (regenerated 2026-04-08 after audit fixes M1+m4) | DCS array job 1308020 + merge 1308021 |
| `n` range | 4..16 (13 values) | `parameters.nRange` |
| `k` range | {2, 4} | `parameters.kRange` |
| Trials per (n, k, strategy) | 100 | `parameters.numTrials` |
| Strategies | `partial_real`, `diluted_list`, `shifted_coefficients`, `subset_plus_noise` | `/Users/alex/cs310-code/experiments/harness/soundness_multi.py:115` |
| Total trials | 10 400 | 13 × 2 × 4 × 100 |
| `ε` | 0.3 | `soundness_multi.py:99` |
| `δ` | 0.1 | `soundness_multi.py:141` |
| `θ` | `min(ε, max(0.01, 0.9/k))` = 0.3 (k=2), 0.225 (k=4) | `soundness_multi.py:127` |
| `a² = b²` | random (Dirichlet `parseval_weight`) | `soundness_multi.py:133, 143-144` |
| Target function | random k-sparse via `make_k_sparse` (Dirichlet(1,...,1) on k random nonzero parities) | `soundness_multi.py:133`, `phi.py:133-188` |
| Verifier classical samples | **30 000** (audit fix M1; was 3 000) | `soundness_multi.py:25` |

The four strategies at `worker.py:277-334`:

- **`_strategy_partial_real`** (`worker.py:277-288`): drops the heaviest `n_real = max(1, len/2)` real coefficients and keeps the rest, plus up to 3 fakes. For k=2 keeps only the *smaller* real coefficient (the `n_real` variable is the count *dropped*, not kept — m5 in the audit, naming confusion).
- **`_strategy_diluted_list`** (`worker.py:291-311`): keeps `max(1, len/2)` of the **weakest** real coefficients (so 1 of 2 for k=2, 2 of 4 for k=4 after the m4 audit fix; was 1 of 4 before) plus up to 20 random padding indices.
- **`_strategy_shifted_coefficients`** (`worker.py:314-321`): sends `len(heavy)` indices from the **complement** of the true support, fabricated claim 0.8. Accumulated true weight ≡ 0 by construction.
- **`_strategy_subset_plus_noise`** (`worker.py:324-334`): sends the **single heaviest** real index plus up to 5 fakes, fabricated claim 0.3.

The dishonest dispatch at `worker.py:394-414` correctly routes through `verifier.verify_fourier_sparse` for `spec.k > 1`, selecting the k-sparse threshold and tolerance in `_verify_core`.

### 2(b) Precise paper claim being tested

The experiment spot-checks Eq. (18) for the Theorem 15 protocol (p. 48). The relevant distribution class is `D ∈ D_{U_n; ≥ϑ} ∩ D_{U_n; [a², b²]}` (Definitions 13 + 14, p. 44), with the precondition `ε ≥ 4k·√(b² − a²)` (Theorem 15 statement, p. 48). The experiment sets `a² = b² = parseval_weight`, so `b² − a² = 0` and the precondition is trivially satisfied. Because `tilde_phi` is exactly k-sparse, `opt_D(F) = 0` against the parity benchmark (where F is the parity class for the k-sparse hypothesis output). Eq. (18) becomes:

```
Pr[ V accepts ∧ err_D(h) > 0.3 ] ≤ 0.1   for every P'.
```

The Theorem 15 numerics at `k ∈ {2, 4}`, `ε = 0.3`:

| k | acceptance threshold gap `ε²/(128 k²)` | per-coeff tolerance `ε²/(256 k² \|L\|)` | list-size bound |
|---|---|---|---|
| 2 | `0.09 / 512 ≈ 1.758 × 10⁻⁴` | `3.52 × 10⁻⁴ / \|L\|` | `64·a²/0.09` |
| 4 | `0.09 / 2048 ≈ 4.395 × 10⁻⁵` | `2.20 × 10⁻⁵ / \|L\|` | `64·a²/0.0506` |

**The gap between the acceptance threshold and the distribution's true Parseval weight `a²` is astonishingly small (`~1.76 × 10⁻⁴` for k=2, `~4.4 × 10⁻⁵` for k=4).** This is what makes `subset_plus_noise` a meaningful boundary test: the prover only needs `c_max² ≥ a² − 1.76 × 10⁻⁴`, which holds whenever the second coefficient `c_min²` is itself smaller than `~1.76 × 10⁻⁴`, i.e. `|c_min| < ~0.013`. For Dirichlet(1,1) draws, `Pr[c_min < 0.013] ≈ 0.026` analytically, so we expect roughly 2–3% of `subset_plus_noise k=2` trials to be in the regime where `c_max²` alone already satisfies the threshold even under perfect estimation. The empirically observed ~5% rate (see §2(d)) matches this with the extra ~2% coming from sampling fluctuation.

The audit fix M1 in `audit/soundness_multi.md` bumped the verifier sample count from 3 000 to **30 000** specifically to control sampling fluctuation: the squared-coefficient estimator standard deviation is `~ 1.4/√m`, giving `~0.026` at m=3 000 (comparable to the gap itself, hence the pre-fix 18% false-accept) and `~0.008` at m=30 000 (3–4× margin over the gap). The on-disk .pb reflects the post-fix state.

### 2(c) What each figure shows

Two artefacts in `/Users/alex/cs310-code/results/figures/soundness_multi/`.

**`rejection_vs_n_by_k.png`** — 2×2 faceted line plot, one panel per strategy. x-axis `n ∈ 4..16`; y-axis rejection rate; each panel has two lines, `k = 2` (blue circles) and `k = 4` (orange circles), with Wilson 95% error bars. Grey dashed `1 − δ = 0.9` reference. Every (strategy, k) line stays above 0.9. `partial_real`, `shifted_coefficients`, and `diluted_list k=4` are pinned at 1.00; `diluted_list k=2` oscillates in `[0.96, 1.00]`. The `subset_plus_noise k=2` line ranges over `[0.91, 0.98]` with no clear n-dependence; `subset_plus_noise k=4` is at `[0.98, 1.00]`.

**`comparison_single_vs_multi.png`** — grouped bar at representative `n ∈ {4, 7, 10, 13, 16}` showing pooled-across-strategy rejection rates for single-element (`k=1`), multi-element `k=2`, and multi-element `k=4`. Bars are pooled equally across strategies, so one hard strategy gets the same weight as three trivial ones. **This is by construction an apples-to-oranges comparison** (the audit flagged it as n2); the figure is descriptively correct but understates the boundary-case difficulty.

**Rejection mechanism:** every reject is via **Step 4 weight check**. The list-size bound is never binding — max prover list size is 21 (20 padding + 1 real in diluted_list) and the bound is `≥ 64·0.25/0.09 ≈ 178` even at the smallest `a²`. The `soundness_multi_summary.csv` `reject_list_size` column is identically 0 in every row.

### 2(d) Per-cell rejection vs `1 − δ = 0.9`

From `soundness_multi_summary.csv` (independently re-verified against the .pb):

| strategy | k | min rej | max rej | mean rej | cells where rej < 0.95 |
|---|---|---|---|---|---|
| partial_real | 2 | 1.000 | 1.000 | 1.000 | none |
| partial_real | 4 | 1.000 | 1.000 | 1.000 | none |
| diluted_list | 2 | 0.960 (n=4) | 1.000 (n=14) | 0.985 | n=4 only (0.96) |
| diluted_list | 4 | 1.000 | 1.000 | 1.000 | none |
| shifted_coefficients | 2 | 1.000 | 1.000 | 1.000 | none |
| shifted_coefficients | 4 | 1.000 | 1.000 | 1.000 | none |
| **subset_plus_noise** | **2** | **0.910 (n=15)** | **0.980 (n=8)** | **0.951** | **n ∈ {4,5,6,9,10,11,13,14,15,16}** |
| subset_plus_noise | 4 | 0.980 (n=14) | 1.000 | 0.998 | none |

**No (strategy, k, n) cell falls below 0.9.** The minimum rejection rate in the 104-cell grid is 0.91 at `(subset_plus_noise, k=2, n=15)`. The plot script's report at `plot_soundness_multi.py:432-438` prints "All (strategy, k, n) combinations satisfy soundness ≥ 1−δ = 0.9", and I verified this independently from the .pb. A clean improvement over the pre-rerun state (`subset_plus_noise k=2` was at 0.82–0.92, max false-accept 18%).

### 2(e) Hypothesis-error tracking on dishonest trials — methodological gap

The soundness_multi `.pb` does **not** populate `hypothesisS`, `hypothesisCorrect`, or `misclass_rate` for any trial. Verified by direct decode: every trial has `hypothesisS: None` and `accepted: None` in the camelCase JSON; the only fields are `outcome`, `accumulatedWeight`, `acceptanceThreshold`, `listSize`, `aSq`, `bSq`, `theta`, `k`, `epsilon`, `delta`, `verifierSamples`. Looking at the worker at `worker.py:421-434`, the hypothesis-extraction branch only fires `if vresult.accepted and vresult.hypothesis is not None`, but the `accepted` field is None in the output for the multi-element path, suggesting the proto serialiser is not writing it for the dishonest multi path (contrast with single-parity, where it is populated). **The audit `audit/soundness_multi.md` did not surface this gap explicitly, so I am flagging it here as a methodological gap.**

We can, however, determine the correct/wrong split *analytically* from `accumulatedWeight` and `aSq`. For `subset_plus_noise k=2`, the prover's list contains 1 real heavy index (`|ξ̂| ≈ |c_max|`) plus 5 fake indices (`|ξ̂| ≈ 1/√30000 ≈ 0.0058`). The fake indices contribute at most `5 · 0.0058² ≈ 1.7 × 10⁻⁴`, comparable to the threshold gap `ε²/(128·k²) ≈ 1.76 × 10⁻⁴`. The verifier accepts iff roughly `c_max² ≥ aSq − 3.5 × 10⁻⁴`. I computed the minimum `aSq` among the 78 accepted `subset_plus_noise k=2` trials to be `0.8488` (via decode), implying `c_max² ≥ 0.8487`, i.e. `|c_max| ≥ 0.921` for every accepted trial.

The verifier's output for the k-sparse path is the `FourierSparseHypothesis` from `verifier.py:646-670`: `g(x) = Σ_{ℓ=1}^k ξ̂(s_ℓ) χ_{s_ℓ}(x)` over the k=2 heaviest entries (the one real heaviest plus one noise spike). The Lemma-14 randomised hypothesis is `Pr[h(x)=1] = (1 − g(x))² / (2(1 + g(x)²))` (`verifier.py:159-166`). For `g(x) ≈ c_max · χ_{s_max}(x)` (the noise term contributes negligible bias), the classification error against the true distribution is approximately `(1 − |c_max|)/2`. With `|c_max| ≥ 0.921`, this gives **err(h) ≤ 0.040** for every accepted trial — an order of magnitude below `ε = 0.3`.

**Conclusion: every one of the 87 accepts in the dataset (78 from `subset_plus_noise k=2`, 4 from `diluted_list k=2` across various n, plus a few scattered) is an accept of a hypothesis with err ≪ ε, NOT an Eq. (18) violation.** The empirical `Pr[V accepts ∧ err(h) > 0.3]` is **0 / 10 400** by this analysis, which is the strongest possible empirical evidence for Eq. (18) within the strategy menu tested. The 5–9% "false-accept-rate-by-naive-counting" that the figure displays is best understood as the verifier *correctly* accepting near-pure-parity targets where the prover happened to identify the exactly-right heaviest coefficient.

**However, the above is an analytical argument, not a data-driven check.** The experiment as written does not log enough information per trial to verify this at the per-trial level from the .pb alone. The harness should populate `misclass_rate` (or at minimum `hypothesisS`) for accepted dishonest trials in the multi-element path, and the plot script should surface a `bad_accept_rate` column analogous to the one already on `soundness_summary.csv`.

### 2(f) Verdict: **PASS-with-caveats**

The verifier code at `verifier.py:434-553` implements Theorem 15 verbatim. The four strategies are correctly dispatched and the worker correctly routes to `verify_fourier_sparse` for `k > 1`. The audit fix M1 (3000 → 30000) brings the verifier's empirical noise floor below the `ε²/(128 k²)` gap on the boundary case, and the post-fix rerun puts every (strategy, k, n) cell above the 0.9 floor. The ~5–9% accepts on `subset_plus_noise k=2` correspond to trials where `|c_max| ≥ 0.92`, giving `err(h) ≤ 0.04 << 0.3`, so these are not Eq. (18) violations even though the figure displays them as part of `1 − rejection_rate`.

**Caveats:**

1. **Missing misclassification logging.** The multi-element .pb does not populate `hypothesisS` or `misclass_rate` for *any* trial. The "every accept is a correct-enough accept" claim in §2(e) had to be established analytically from `accumulatedWeight` and `aSq`, rather than directly from `misclass_rate`. This is a gap that the existing audit file did not surface.
2. **Shallow strategy menu.** Three of four strategies (`partial_real`, `shifted_coefficients`, `diluted_list`) carry essentially no real Fourier weight and reject at ~100% by structural inevitability; only `subset_plus_noise` probes the boundary at k=2, and it does so by submitting the *exactly* correct heaviest coefficient, which makes the "false accept" framing suspect. Flagged as M2 in the audit; deferred to a Tier-4 cheating-strategy review.
3. **`θ = min(ε, max(0.01, 0.9/k))` is an undocumented heuristic** (m1 in the audit). Only affects the non-binding list-size bound.
4. **Strategy names/docstrings do not always match audit-fixed code** (m4, m5 in the audit).

### 2(g) Open issues / follow-ups

- **Log per-trial misclassification for dishonest accepts in the multi-element path.** Single most useful one-line fix: populate `hypothesisCorrect` and `misclass_rate` for every accepted dishonest trial, then add a `bad_accept_rate` column to `soundness_multi_summary.csv`. Without this, the soundness_multi experiment can only verify Eq. (18) via the analytical argument in §2(e), not directly from the data.
- **Add a "drop heaviest" adversarial strategy.** A useful additional strategy would submit `k − 1` of the real heavy indices **omitting the heaviest one**, plus noise. This would test whether the weight-check rejects when the prover knows almost everything but is missing the most important coefficient. The current `partial_real` keeps the *weakest* real coefficients, which is structurally easy to reject.
- **Extend `k_range` to `{2, 4, 8}`.** Would test whether "subset_plus_noise gets harder to reject as k grows" inverts at large enough k: the gap `ε²/(128 k²)` shrinks as `1/k²` while typical `c_max²` under Dirichlet(1,...,1) shrinks as `O(1/k)`, so at some k the threshold becomes easier to hit by `c_max` alone.
- **Faceted per-strategy `comparison_single_vs_multi.png`.** The current pooled-across-strategies version understates the boundary-case difficulty (n2 in the audit).
- **Add a sweep over uniform k-sparse coefficients** (`c_i = 1/√k`), complementing the Dirichlet sweep. This exercises the hardest regime for the boundary check, where `c_max² = 1/k`.

---

## Cross-section synthesis: what these two experiments collectively show

The single-parity (`soundness`) and k-sparse (`soundness_multi`) experiments together test 8 hand-written cheating strategies against the verifier protocols of Theorem 12 (parities, p. 45) and Theorem 15 (k-sparse, pp. 48–49) under `ε = 0.3`, `δ = 0.1`. The verifier code under `/Users/alex/cs310-code/ql/verifier.py` is faithful to both theorems and is correctly invoked by both harnesses; the experiments exercise the verifier on adversarial input by constructing fake `ProverMessage` objects in `worker.py:251-334` and dispatching to `verify_parity` or `verify_fourier_sparse` according to `spec.k > 1` (`worker.py:394-414`).

Across 17 200 dishonest trials (6 800 single-parity + 10 400 k-sparse), the **empirical Eq. (18) bad-accept rate is 0 / 17 200**. For the parity experiment this is verified directly from `hypothesisS` per trial (target is hardcoded `s* = 1`, so "correct" = `hypothesisS == 1`); for the k-sparse experiment this is verified *analytically* via the `accumulatedWeight ≈ c_max²` argument in §2(e), because the multi-element harness does not currently log per-trial misclassification on dishonest accepts.

**What this empirically supports is the spot-check statement:** "the verifier correctly rejects (or correctly-by-accident accepts) every trial of these eight strategies at these parameters". **What it does not support is the universal statement of Eq. (18).** Definition 7 quantifies over *all* possibly unbounded P', and the strategies tested here are all polynomially-bounded, finitely-parameterised, and fairly naive: they are either constructed to definitely-not-include `s*` (`wrong_parity`, `partial_list`, `inflated_list`, `shifted_coefficients`), to randomly-include-it (`random_list`), or to drop a known fraction of the real heavy support and pad with random fakes (`partial_real`, `diluted_list`, `subset_plus_noise`). None of them attempts to adaptively attack the verifier's random tape, none of them probes the verifier's list-size bound (the implemented bound is 712 for the parity case and ≥178 for the k-sparse case; the maximum adversarial list size in this menu is 21), and none of them exploits structural properties of the verifier's `_estimate_coefficients_independently` step.

The strategy menu's silence on these axes is **not a bug of the experiments** — it is in the nature of empirical strategy-menu testing. The paper's soundness proof (Eqs. 110–114 for Theorem 12, Eqs. 119–127 for Theorem 15) only relies on the verifier's own Chernoff–Hoeffding step, which the experiments confirm is operationally tight at the chosen sample budgets. What the experiments add to the paper's theoretical guarantee is **empirical confidence that:**

1. the verifier code at `ql/verifier.py:434-553` is in fact computing the right quantities (the 0/17 200 bad-accept rate at the chosen parameters would otherwise be hard to achieve);
2. the Theorem-12 / Theorem-15 numerics (`64 b²/θ²`, `ε²/(16|L|)` or `ε²/(256 k² |L|)`, `a² − ε²/8` or `a² − ε²/(128 k²)`) are sufficient to reject every "natural" dishonest strategy a thesis author could come up with;
3. the `random_list` (parity) and `subset_plus_noise` (k-sparse) boundary cases — where the verifier *does* sometimes accept — correspond to runs where the adversary accidentally identified `s*` (parity) or the true heaviest coefficient (k=2 case), and the verifier therefore had no business rejecting.

**The honest framing is: these are tests of the verifier implementation, not tests of Definition 7.** Definition 7 is a theorem of the paper; it is not something an experiment can falsify by spot-check. The contribution of these experiments is to make it concrete that, at the parameters used in the dissertation, the protocol's analytical δ = 0.1 budget is operationally never spent — the empirical bad-accept rate is 0 out of 17 200, well below the 10% budget that Definition 7 grants the verifier. A finding of non-zero bad-accept rate would have indicated either (i) a verifier code bug, (ii) a sample-budget under-provisioning bug (which is exactly what the M1 fix in `audit/soundness_multi.md` addressed for the pre-rerun version of `soundness_multi`), or (iii) a cheating strategy that managed to beat the protocol; the actual finding of zero bad-accepts is consistent with (i)/(ii)/(iii) all being absent.

The **single most material recommendation** to tighten these results is to add **per-trial misclassification logging on dishonest accepts for the multi-element path**, mirroring the single-parity path's current behaviour. Without it, the k-sparse soundness claim has to be established analytically from `accumulatedWeight` and `aSq`, which is more fragile than reading `misclass_rate` directly from the .pb. The next most material recommendation is to add a strategy that probes the **strongest** adversarial scenario (omit the **heaviest** real coefficient) rather than the **weakest** (drop the weakest, which is what `partial_real` currently does).

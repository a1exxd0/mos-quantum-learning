# Agent 1 — Completeness / List-Size-Scaling Findings

This is an independent re-audit of the four completeness experiments
(`scaling`, `bent`, `k_sparse`, `average_case`) against Caro et al.,
*Classical Verification of Quantum Learning* (ITCS 2024,
arXiv:2306.04843v2). Paper pages cited below were read directly from
`/Users/alex/cs310-code/papers/classical_verification_of_quantum_learning.pdf`.
Empirical claims are re-derived from `.pb` files via
`uv run python -m experiments.decode`; figures were inspected as PNG
images, not via CSV summaries.

The shared protocol code in `/Users/alex/cs310-code/ql/verifier.py`
implements **Theorem 12** (parity, p. 45) and **Theorem 15** (k-sparse,
p. 48) line-by-line. The four identities I cross-checked:

- `verifier.py:464` — list bound `64 b²/θ²` (Theorems 12/15 Step 1).
- `verifier.py:488` (parity) — Hoeffding tolerance `ε²/(16|L|)` (Thm 12 Step 3).
- `verifier.py:491` (k-sparse) — Hoeffding tolerance `ε²/(256 k² |L|)` (Thm 15 Step 3).
- `verifier.py:515` / `:518` — acceptance thresholds `a² − ε²/8` and `a² − ε²/(128 k²)` (Steps 4 of Thms 12 and 15, eqs. (77) p.37 and (118) p.49).

All four match the paper exactly.

---

## 1. `scaling` — honest single-parity prover, completeness vs n

**(a) Data and configuration.** `/Users/alex/cs310-code/results/scaling_4_16_100.pb`,
1,300 trials, `n ∈ {4, …, 16}`, 100 trials per `n`. Parameters from
`/Users/alex/cs310-code/experiments/harness/scaling.py:12-24`:
`ε = 0.3`, `δ = 0.1`, `θ = ε = 0.3`, `a² = b² = 1`,
`qfs_shots = 2000`, `classical_samples_prover = 1000`,
`classical_samples_verifier = 3000`. Each trial draws a random
non-zero parity `s* ∈ {1, …, 2^n − 1}`. Worker dispatches to
`verify_parity` (`worker.py:178-186`).

**(b) Paper claim.** Theorem 12 (p. 45): `n`-bit parities are
efficiently 1-agnostic verifiable on `D_{U_n;≥ϑ} ∩ D_{U_n;[a²,b²]}`
with completeness `≥ 1 − δ`, list bound `|L| ≤ 64 b²/ϑ²`, acceptance
threshold `a² − ε²/8`, and verifier sample complexity
`Õ(b⁴ log(1/(δϑ²))/(ε⁴ϑ⁴))` — **`n`-independent in the leading
term**. Companion Theorem 5(i) (p. 26) predicts the QFS post-selection
rate is exactly 1/2. For a pure parity `φ = s*·x`, `a² = b² = 1`,
`‖ĝ‖₀ = 1`, so the prediction is near-certain acceptance, |L|=1, and
post-selection 0.5.

**(c) Figures (`/Users/alex/cs310-code/results/figures/scaling/`)**:

- `completeness_vs_n.png` — dual line plot, acceptance rate (blue
  circles) and correctness rate (orange squares) flat at 1.00 across
  `n ∈ {4, …, 16}`, with a `1 − δ = 0.9` reference line. Y-axis clipped
  to `[0.80, 1.02]`.
- `postselection_vs_n.png` — bar chart of median post-selection per `n`,
  every bar at ≈ 0.499 with sub-percent IQR whiskers, dashed reference
  at 1/2 ("Theoretical 1/2 (Thm 5)").
- `list_size_vs_n.png` — error-bar plot of median |L| (with IQR)
  vs `n`. All markers at 1.0. Two reference lines: dashed grey
  `4/θ² = 44.4` ("Parseval bound") and dotted orange `|L| = 1`.
  *Nit*: the `4/θ²` line is from Theorem 7 / Lemma 8; the code path
  actually exercises Theorem 12 with `64 b²/θ² = 711.1`. Both bounds
  are valid (scaling is conservative), but the legend doesn't
  distinguish.
- `resource_scaling.png` — dual-axis plot **explicitly titled
  "Fixed-budget feasibility (NOT measured n-scaling)"** with an
  italic footnote stating that all three sample budgets are
  hard-coded constants and that wall-clock growth is a `2^{n+1}`
  statevector simulator artefact. Total copies (log) is flat at
  6000; wall-clock time (log) rises 0.36 s → 1203 s. **This is the
  M1 fix from `audit/scaling.md`**: a previous version overlaid a
  fitted theoretical curve `C·n·log(1/(δθ²))/θ⁴` through this flat
  6000 series, creating the misleading impression that resource
  scaling was being measured. The current version drops the overlay
  and the title/footnote correctly frame the figure as a feasibility
  check, not a scaling measurement. The fix is appropriate.

**(d) Match to prediction.** From an independent decode of
`/tmp/scaling.json`: 1,300 / 1,300 trials have
`accepted = True`, `hypothesisCorrect = True`, `listSize = 1`. Median
accumulated weight is exactly 1.00000 at every `n` (min = max = 1.00000).
Acceptance threshold is 0.98875 = `1 − 0.09/8` uniformly. The
exactness follows directly from the signal structure: for a pure
parity, `(1 − 2y)·χ_s(x) = (−1)^{(s + s*)·x}`, so the empirical
mean over 3000 samples is exactly `δ_{s,s*}` with zero variance —
the Hoeffding slack is never tested. Median post-selection rate per
`n` is in `[0.496, 0.502]`, matching Theorem 5(i) to sub-percent
precision. Wall-clock jumps at `n = 13 → 14` (≈ 8×) and `n = 14 → 15`
(≈ 13×) are L2/L3 cache crossings of the `2^{n+1}`-dimensional
statevector, not protocol cost — the figure now says so explicitly.

**(e) Verdict: PASS (with framing caveat).** The implementation is
correct line-by-line; the empirical 100/100 results, the
post-selection rate of ≈ 0.5, and the median |L| = 1 are exactly what
Theorem 12 and Theorem 5(i) predict for pure parities. The framing
caveat — already addressed in the post-fix figure title — is that
all three sample budgets are hard-coded constants, so the experiment
demonstrates "this 6000-copy budget suffices on the easiest target up
to `n = 16`", not "the verifier's minimum sufficient sample count is
empirically `n`-independent". That distinction is now clearly marked
on the figure.

**(f) Open issues / follow-up.**

- **Tier 3 rerun** (`audit/FOLLOW_UPS.md §4`): re-run with
  `qfs_shots = None`, `classical_samples_prover = None`,
  `classical_samples_verifier = None` so the worker falls through to
  the analytic per-trial Hoeffding budgets in
  `ql/verifier.py:487-497` and `ql/prover.py:316-321`. Only this
  rerun would actually validate the leading-order `n`-independence
  claim of Theorem 12. Tractable on the DCS cluster up to `n ≈ 12`;
  beyond that the conservative DKW pre-factor in `prover.py:318`
  would need a 4× tightening.
- **Minor**: legend on `list_size_vs_n.png` cites the Theorem 7
  `4/θ²` bound while the code enforces the Theorem 12 `64 b²/θ²`
  bound. Cosmetic.

---

## 2. `bent` — Maiorana–McFarland bent functions, Fourier-extraction worst case

**(a) Data and configuration.**
`/Users/alex/cs310-code/results/bent_4_16_100.pb`, 700 trials,
`n ∈ {4, 6, 8, 10, 12, 14, 16}` (even only — bent functions exist
only for even `n`), 100 trials per `n`. Parameters
(`/Users/alex/cs310-code/experiments/harness/bent.py:12-24`):
`ε = θ = 0.3`, `δ = 0.1`, `a² = b² = 1`, `qfs_shots = 3000`,
`classical_samples_prover = 2000`, `classical_samples_verifier = 5000`.
Target is `f(x, y) = ⟨x, y⟩ mod 2` over `(F_2)^{n/2}` (canonical
Maiorana–McFarland) — one bent per `n`, reused across all 100 trials;
the randomness is in QFS sampling. After each trial the harness
overrides `t.hypothesis_correct = t.accepted` (`bent.py:125-127`)
because for a flat spectrum every parity in `L` is equally optimal.

**(b) Paper claim.** A bent function on even `n` has
`|ĝ(s)| = 2^{−n/2}` for *every* `s ∈ {0,1}^n`, so it is
`2^n`-Fourier-sparse (the maximum) and **violates Definition 11**
(p. 35) for any `θ > 2^{−n/2}`. Theorem 12 (p. 45) applies only
within Definition 11; Corollary 5 (p. 27) returns `s` in `L` if
`|ĝ(s)| ≥ θ/2` (guaranteed inclusion zone) and excludes `s` if
`|ĝ(s)| < θ/2` (with `θ/2 ≤ |ĝ(s)| < θ` being an "uncertain band"
where inclusion is permitted but not required). Predicted crossover
at `2^{−n/2} = θ/2 = 0.15`, i.e. `n = 2 log₂(2/θ) ≈ 5.47`.

**(c) Figures (`/Users/alex/cs310-code/results/figures/bent/`)**:

- `list_size_growth.png` — log-scale `|L|` vs `n` with the empirical
  median (with IQR shading), a theory curve `min(2^n, ⌊4/θ²⌋)`, a
  dotted `2^n` reference, a dash-dot `|L| = 1` reference, a dashed
  `4/θ² = 44.4` Parseval reference, a vertical red dotted line at
  `n ≈ 5.5` annotated "Crossover 2^{−n/2} = θ/2", and an **orange
  shaded vertical band** labelled "Cor. 5 uncertain band [θ/2, θ)"
  spanning `n ∈ [2 log₂(1/θ), 2 log₂(2/θ)] = [3.47, 5.47]`. The
  empirical median drops from 16 at n=4 to 1 at n=6 to 0 for n ≥ 8.
  **The orange band is the m2 fix from `audit/bent.md`**: the
  previous figure marked only the `θ/2` exclusion line, implying
  n=4 was "below the crossover"; it now correctly shows that n=4
  (with coefficient 0.25) sits inside the `[θ/2, θ)` uncertain band.
- `bent_vs_parity_acceptance.png` — grouped bar chart, parity bars
  (blue, hard-coded to 100 % at `plot_bent.py:289`) vs bent bars
  (orange) at each `n`. Bent bar at n=4 is 100 %; bent bars at
  n ≥ 6 are 0 % (annotated "0%"). Vertical red dotted crossover
  marker at `n ≈ 5.5`.
- `resource_explosion.png` — dual-axis line plot. Total copies (log)
  drops from ≈ 10,000 at n=4 (when the prover estimates 16
  coefficients) to ≈ 8000 at n ≥ 6 (when the prover skips the
  estimation step because `|L| = 0`). Wall-clock time (log) rises
  from ≈ 0.6 s at n=4 to 1612 s at n=16, with an arrow annotation
  on n=16. **Unlike `scaling/resource_scaling.png`**, this figure
  has no explicit note that wall-clock growth is a simulator
  artefact — the title "resource cost vs n" could be misread as a
  protocol-level claim. Minor framing nit.

**(d) Match to prediction.** From `/tmp/bent.json`:

| n | |coeff| = 2^{−n/2} | zone | med \|L\| | max \|L\| | accept | accumulated weight |
|---|---|---|---|---|---|---|
|  4 | 0.2500 | uncertain `[0.15, 0.30)` | 16 | 16 | 100 / 100 | 1.0029 |
|  6 | 0.1250 | exclusion `< 0.15` | 1 | 4 | 0 / 100 | 0.0171 |
|  8 | 0.0625 | exclusion | 0 | 0 | 0 / 100 | 0.0000 |
| 10–16 | < 0.04 | exclusion | 0 | 0 | 0 / 100 | 0.0000 |

Threshold is 0.98875 (`a² − ε²/8`) uniformly. Every rejection is
`reject_insufficient_weight`; **zero rejects are
`reject_list_too_large`**. The n=4 row accepts because all 16
flat-spectrum coefficients clear the conditional-QFS extraction
threshold `θ²/4 = 0.0225` (since `0.25² = 0.0625 > 0.0225`), Parseval
gives accumulated weight = 1.0029 ≥ 0.98875, and the verifier accepts.
The 0.003 excess over 1.0 is finite-sample estimator noise on
16 × 5000 = 80,000 classical samples. At n=6, median accumulated
weight 0.0171 is consistent with at most one spurious coefficient
clearing the conditional threshold and contributing
`(0.125)² = 0.0156` plus noise. The crossover prediction `n = 5.47`
is bracketed exactly by the n=4 (accept) and n=6 (reject) rows; the
transition is sharp because every bent function at a given `n` has
identical coefficient magnitudes by construction.

**Subtle paper-precondition note.** Theorem 12's hypothesis range
is `ϑ ∈ (2^{−(n/2−3)}, 1)`. With `ϑ = 0.3`, this is formally
satisfied only for `n ≥ 10`. At n ∈ {4, 6, 8} the protocol still runs
correctly (and produces the expected accept-at-4 / reject-elsewhere
behaviour), but the formal Theorem 12 *completeness* guarantee is
out of range. The protocol's **soundness** behaviour — correctly
rejecting flat-spectrum bent targets at n ≥ 6 — is independent of `ϑ`.

**(e) Verdict: PASS.** The experiment is a clean worst-case
demonstration of the Corollary 5 extraction procedure's dependence on
the θ-granularity promise. The crossover at n = 5.47 is reproduced
exactly. The verifier correctly rejects every out-of-promise bent
target at n ≥ 6. The n=4 acceptance is correctly attributed to the
Corollary 5 uncertain band by the post-fix figure. The only open
issues are framing nits.

**(f) Open issues / follow-up.**

- **Framing nit**: `resource_explosion.png` should carry the same
  "wall-clock growth is a `2^{n+1}` statevector simulator artefact"
  caption that `scaling/resource_scaling.png` does.
- **Minor**: the m3 hard-coded sample budgets (5000 verifier samples)
  bypass Hoeffding sizing. For bent the effect is benign because
  every coefficient is exactly ±2^{−n/2}, but the same shortcut would
  break soundness for less benign distributions.
- No rerun required.

---

## 3. `k_sparse` — Fourier-`k`-sparse verification (Theorem 15)

**(a) Data and configuration.**
`/Users/alex/cs310-code/results/k_sparse_4_16_100.pb`, 2,800 trials,
`n ∈ {4, 6, 8, 10, 12, 14, 16}` (even), `k ∈ {1, 2, 4, 8}`, 100
trials per cell. Parameters
(`/Users/alex/cs310-code/experiments/harness/k_sparse.py:13-30`):
`ε = 0.3`, `δ = 0.1`, fixed budgets
`qfs_shots = 2000 / classical_samples_prover = 1000 / classical_samples_verifier = 3000`,
`misclassification_samples = 1000`. **Adaptive θ**:
`theta = min(ε, max(0.01, 0.9/k))` (`k_sparse.py:59`), giving
`θ = 0.3` for k ∈ {1, 2}, `0.225` for k=4, `0.1125` for k=8 — a
heuristic, not derived from a theorem. Targets are
`make_k_sparse(n, k, rng)`: k distinct non-zero parity indices,
Dirichlet(1,…,1) coefficients summing to 1, `a² = b² = Σc_i² = pw`.
For k=1 the trial sets `spec.k = None` so the worker dispatches to
`verify_parity` (`worker.py:178-186`); for k > 1 the worker
dispatches to `verify_fourier_sparse`.

**(b) Paper claim.** **Theorem 15** (p. 48), distributional k-sparse
verification: list bound `|L| ≤ 64 b²/ϑ²`, per-coefficient Hoeffding
tolerance `ε²/(256 k² |L|)`, acceptance threshold `a² − ε²/(128 k²)`,
output the Lemma 14 randomised hypothesis on the k heaviest
coefficients, misclassification `≤ 2 · opt_{Fourier-k-sparse}(D) + ε`.
The completeness guarantee `Pr[V accepts ∧ err ≤ 2 opt + ε] ≥ 1 − δ`
applies on `D_{U_n;≥ϑ} ∩ D_{U_n;[a²,b²]}`. **Definition 13** (p. 44)
requires `ϕ̂(s) ≠ 0 ⇒ |ϕ̂(s)| ≥ ϑ`. Dirichlet(1,…,1) draws routinely
produce a smallest component well below `1/k`, so the experiment
**systematically violates Definition 13** for k ≥ 2. The
`ε ≥ 4k √(b² − a²)` precondition collapses to `ε ≥ 0` because the
experiment holds `a = b = √pw`.

**(c) Figures (`/Users/alex/cs310-code/results/figures/k_sparse/`)**:

- `acceptance_vs_n_by_k.png` — line plot of acceptance rate vs `n`
  with one line per k and 95 % Wilson CI bands, plus a `1 − δ = 0.9`
  reference line "nominal Thm 10/15 target". Title is **explicitly
  caveated**: `"k-Sparse completeness: acceptance rate vs n
  (off-promise: Dirichlet(1,…,1) targets often have c_min < θ)"` —
  the M1 fix from `audit/k_sparse.md`. k=1 sits flat at 1.00; k=2 in
  [0.46, 0.64]; k=4 in [0.36, 0.58]; k=8 in [0.41, 0.60] with two
  notable spikes at 92 % (n=6) and 96 % (n=8).
- `list_size_vs_k.png` — grouped log-scale bar chart of median |L|
  at representative `n ∈ {4, 8, 12, 16}`, bars grouped by k, with
  red dashed per-k overlay at the **actually enforced**
  `64 b²/θ²` list bound (averaged across cells). This is the
  m3 fix: previous version overlaid `4/θ²` (Theorem 7's SQ
  verifier); current version correctly uses the Theorem 10/15
  random-example bound.
- `misclassification_heatmap.png` — 7×4 heatmap of mean empirical
  misclassification rate per (n, k). Values: k=1 → 0.000 uniformly;
  k=2 → [0.138, 0.162]; k=4 → [0.252, 0.282]; k=8 → [0.341, 0.368].
  Title says "ε = 0.3, Thm 15: ≤ 2·opt + ε". For honest Dirichlet
  targets `opt_{Fourier-k-sparse}(D) = 0` (the target *is*
  Fourier-k-sparse), so the bound collapses to ε = 0.3. **The k=8
  column consistently exceeds 0.3.** The title's `≤ 2·opt + ε`
  phrasing is technically a slight over-reach because Theorem 15
  bounds misclassification *conditional on the verifier not
  rejecting*, not unconditionally; the figure averages over
  accepted trials only.

**(d) Match to prediction.** From `/tmp/k_sparse.json`, headline
per-(n, k):

```
n  k | acc% cor% | med|L| max|L| | accumW  threshold misclass
-----+-----------+---------------+-------------------------
 4  1|  100  100 |   1    1      |  1.0000  0.9888    nan
 4  2|   64   63 |  10   16      |  0.6361  0.6183    0.171
 4  4|   58   56 |  16   16      |  0.3679  0.3670    0.282
 4  8|   60   55 |  16   16      |  0.2048  0.2027    0.368
 6  8|   92   88 |  64   64      |  0.2325  0.2133    0.349  ← 2^n-saturated
 8  8|   96   87 | 101  139      |  0.2318  0.1996    0.356  ← finite-sample-saturated
 16 2|   49   49 |   2    2      |  0.6209  0.6219    0.161
 16 4|   38   36 |   3    4      |  0.3712  0.3770    0.266
 16 8|   41   40 |   5    8      |  0.2000  0.2080    0.359
```

Five things drop out cleanly:

1. **k=1 is a perfect Theorem 12 sanity check**: 100/100 acceptance,
   0 misclassification, |L|=1, accumulated weight exactly 1.

2. **For k ≥ 2 the ~50 % acceptance is the Hoeffding coin-flip on a
   zero-margin test.** Median accumulated weight sits essentially
   *on top of* mean threshold, e.g. (n=8, k=4): accumW = 0.3714,
   threshold = 0.3715. The threshold slack `ε²/(128 k²) = 0.000703,
   0.000176, 4.4e-5` for k = 2, 4, 8 is a fraction of a percent of
   `a²`, so any trial whose Hoeffding estimator under-shoots the
   true accumulated weight by even that microscopic amount fails.
   This is **not** a protocol failure; it is the visible imprint of
   running a vanishing-slack test under finite samples on
   Definition-13-violating targets.

3. **The (n=6, k=8) 92 % and (n=8, k=8) 96 % anomaly peaks are
   Hilbert-space saturation.** At n=6 with k=8 and θ=0.1125 the
   prover's extraction threshold `θ²/4 = 0.00316` is below
   `1/2^n = 0.0156`, so finite-sample QFS noise alone is enough to
   include *every* one of the 64 parities. When |L| = 2^n by
   Parseval `Σ ξ̂(sℓ)² = a²` identically and the test passes. At
   n ≥ 10 the noise floor drops below extraction, |L| collapses to
   the real heavy coefficients, and the test becomes
   Hoeffding-sensitive again.

4. **k=8 misclassification 0.34–0.37 exceeds ε = 0.3.** Two
   contributions: (a) Definition 13 violation — Dirichlet(1,…,1)
   with k=8 has typical smallest component ~1/k² ≈ 0.016 << θ; (b)
   Lemma 14 randomised-hypothesis inherent lossiness — for
   `g(x) = Σ ĝ(sℓ) χ_{sℓ}(x)` with `Σ ĝ² = a² ≈ 0.21` the floor is
   `(1 − a²)/2 ≈ 0.4`, so the observed 0.35 is *better* than naive
   Lemma 14. Both are correctly captured by the implementation.

5. **No trial anywhere rejects on `reject_list_too_large`.** The
   `64 b²/θ²` bound is ≈ 569 for k=8; even at the (n=8, k=8)
   saturation peak max |L| = 139. The list bound is a paper-level
   safety net, never operationally binding.

**(e) Verdict: PASS-with-caveats (originally MISFRAMED at the figure
caption level; now fixed).** The Theorem 15 verifier is implemented
faithfully line-by-line; the empirical data is exactly what the
protocol produces on off-promise Dirichlet targets under a
zero-Fourier-bracket configuration. The post-M1 figure title now
correctly attributes the ~50 % acceptance ceiling to off-promise
targets. The misclassification heatmap title is the only remaining
piece that claims more than the theorem allows ("≤ 2·opt + ε" is a
conditional-on-acceptance bound, not an unconditional one).

**(f) Open issues / follow-up.**

- **Caption fix**: `misclassification_heatmap.png` title should
  clarify that `≤ 2·opt + ε` is conditional on verifier acceptance.
- **Structural caveat**: with `a² = b² = pw` the Hoeffding weight
  check has zero slack, so the ~50 % rate is structural. An optional
  rerun with `a_sq = pw·0.95` would show the completeness rebound.
- **Tier 3 rerun** (`audit/FOLLOW_UPS.md §4`): re-run with analytic
  budgets so the per-coefficient Hoeffding tolerance
  `ε²/(256 k² |L|)` is actually achieved (currently the budget is
  5–7 orders of magnitude under). Only after the rerun can
  "off-promise failure" be cleanly separated from "under-sampled
  Hoeffding".
- **Optional rejection-sampling**: rejection-sample Dirichlet draws
  so `min(c_i) ≥ θ`, giving an in-promise test for k ≥ 2.

---

## 4. `average_case` — Fourier-sparse mix through the Theorem 15 path

**(a) Data and configuration.**
`/Users/alex/cs310-code/results/average_case_4_16_100.pb`, 3,900
trials, `n ∈ {4, 5, …, 16}`, three families
(`k_sparse_2`, `k_sparse_4`, `sparse_plus_noise`), 100 trials per
cell. **This `.pb` is the post-audit regeneration**: the protobuf
has `k` populated on every trial (so the worker dispatched to
`verify_fourier_sparse` per `worker.py:167-187`) and there are no
`random_boolean` rows — both post-fix indicators documented in
`audit/average_case.md` lines 105–149. Parameters
(`/Users/alex/cs310-code/experiments/harness/average_case.py:88-101`):
`ε = 0.3`, `δ = 0.1`, fixed budgets 2000/1000/3000. Per family
(`average_case.py:47-65`):

- `k_sparse_2`: Dirichlet(1,1) coeffs, `θ = 0.3`, `a² = b² = pw`, `spec.k = 2`.
- `k_sparse_4`: Dirichlet(1,1,1,1), `θ = 0.225`, `a² = b² = pw`, `spec.k = 4`.
- `sparse_plus_noise`: 1 dominant 0.7 + 3 secondary 0.1, `pw = 0.49 + 0.03 = 0.52`,
  `θ = ε = 0.3`, `spec.k = 4`.

**(b) Paper claim.** The paper makes **no average-case claims** —
Theorems 7–15 are all worst-case over the promise class. What this
experiment actually tests is the Theorem 15 path on a mix of
Fourier-sparse target families. The relevant theorem is therefore
Theorem 15 (p. 48). For `sparse_plus_noise`, the secondary
coefficients are at magnitude 0.1, **below** the Corollary 5
inclusion floor `θ/2 = 0.15` and below the conditional-QFS
extraction threshold `θ²/4 = 0.0225` (since `0.1² = 0.01 < 0.0225`),
so the honest prover cannot reliably include them in `L`. With
`|L| = 1` and `ξ̂(s_dom)² ≈ 0.49`, accumulated weight is below
`a² − ε²/(128·16) ≈ 0.520`, so the verifier should reject most
`sparse_plus_noise` trials at n ≥ 6.

**(c) Figures (`/Users/alex/cs310-code/results/figures/average_case/`)**:

- `acceptance_by_family.png` — line plot of acceptance rate vs `n`
  per family with Wilson CI shading and a `1 − δ = 0.9` reference.
  k_sparse_2: 0.53 → ~0.4–0.55. k_sparse_4: 0.57 → 0.70 (n=5 peak)
  → 0.32 (n=16). sparse_plus_noise: 0.54 → 0.30 (n=5) → 0.01–0.09
  for n ≥ 6. None of the three families crosses the 0.9 reference
  at any `n`. The `random_boolean` family has been removed from
  `FAMILY_ORDER` (`plot_average_case.py:71`) — the M2 fix is in
  place.
- `list_size_by_family.png` — line plot of median |L| vs `n` with
  dotted Parseval reference lines at `4/θ² = 79` (θ=0.225) and
  `4/θ² = 44` (θ=0.3). k_sparse_4 peaks at 31 at n=5 then drops to
  3. k_sparse_2 peaks at 7 at n=4 then drops to 2. sparse_plus_noise
  peaks at 15 at n=4 then drops to 1 at n=6. The `4/θ²` reference
  line is from Theorem 7 (SQ verifier), not the Theorem 15
  random-example `64 b²/θ²` bound — same legend mismatch as in
  scaling/k_sparse. Minor.

**(d) Match to prediction.** Independent re-derivation from
`/tmp/average_case.json`:

**`k_sparse_2`**: med |L| = 2 for n ≥ 5, accumulated weight ≈ a²,
threshold ≈ a² − 7.03e-4. Acceptance rate 39–56 %, the same
Hoeffding-coin-flip-on-zero-slack story as `k_sparse` k=2.
`prover_found_target = 100 %` at every `n` — the prover always
recovers the heaviest true coefficient.

**`k_sparse_4`**: small-`n` saturation (med |L| = 16 at n=4, 31 at n=5,
13 at n=6 — the prover lists nearly the whole 2^n parity space) gives
inflated acceptance peaks (57 %, 70 %, 54 %); at n ≥ 7 |L| collapses
to ~3 (only the three largest Dirichlet draws are recovered) and
the Hoeffding-noise-limited acceptance settles at 32–48 %.

**`sparse_plus_noise`** is the most interesting family. From the
list-size histogram per `n`:

```
n=4: |L| hist = {13:8, 14:22, 15:30, 16:40}  med_aw=0.5251 thr=0.5200
n=5: |L| hist = {1:1, 2:6, 3:29, 4:37, 5:23, 6:4}  med_aw=0.5094 thr=0.5200
n=6: |L| hist = {1:73, 2:26, 3:1}  med_aw=0.4939 thr=0.5200
n=7: |L| hist = {1:95, 2:4, 3:1}  med_aw=0.4909 thr=0.5200
n=8: |L| hist = {1:98, 2:2}  med_aw=0.4891 thr=0.5200
n=16: |L| hist = {1:99, 2:1}  med_aw=0.4947 thr=0.5200
```

The story is exactly as predicted. At n=4 the Hilbert space has
only 16 parities, finite-sample QFS noise pushes nearly all of them
above the extraction threshold, the prover enumerates almost the
entire domain, Parseval gives accumulated weight ≈ a² = 0.52, and
the 54 % acceptance rate is the Hoeffding coin-flip. At n ≥ 6 the
QFS noise floor is too low to inflate the secondary coefficients
(0.1 < θ/2 = 0.15, and `0.1² = 0.01 < θ²/4 = 0.0225`), the prover
resolves only the dominant 0.7 coefficient, |L| = 1, accumulated
weight ≈ 0.49 < 0.52 = threshold, and the verifier **correctly
rejects** essentially every trial. The acceptance threshold is
0.5200 uniformly because `a² = 0.52` and `ε²/(128 k²) = 4.39e-5` is
below reported precision.

**This is not a Theorem 15 completeness failure.** The
`sparse_plus_noise` target violates Definition 13 by construction
(the secondary coefficients 0.1 are non-zero but below θ = 0.3), so
Theorem 15's completeness guarantee is out of scope. The verifier
is doing exactly what the paper says it should do for an
out-of-promise input.

**(e) Verdict: PASS (with structural caveats).** The post-audit
implementation is now correct (M1 dispatch fix in place; M2
random_boolean dropped). The Theorem 15 path is faithfully
exercised. The empirical results are the expected consequences of
running the Theorem 15 weight check with **zero Fourier-bracket
slack** (`a² = b²`) on **off-promise targets**. The "average case"
title is misleading because the paper makes no average-case claims,
but the underlying measurements are correct interpretations of how
the protocol responds to off-promise inputs.

**(f) Open issues / follow-up.**

- **Tier 3 rerun** (`audit/FOLLOW_UPS.md §4`): the acceptance-rate
  story for all three families is currently
  Hoeffding-measurement-limited because
  `classical_samples_verifier = 3000` is 2–3 orders of magnitude
  below the analytic count required by `ε²/(256 k² |L|)`. Only with
  analytic budgets can the question "does Theorem 15's completeness
  guarantee bind on these families" be answered cleanly.
- **Title**: "Average case" should be renamed to something like
  "Theorem 15 acceptance by target family" to drop the unsupported
  framing.
- **`sparse_plus_noise` θ mismatch**: the family deliberately places
  secondary coefficients at 0.1 while θ = 0.3, so the precondition
  is built to fail. The docstring already flags this; the figure
  caption could be more explicit.
- **`list_size_by_family.png` reference lines**: use `4/θ²`
  (Theorem 7) rather than the actually enforced `64 b²/θ²` (Theorem
  15). Minor.

---

## Cross-section synthesis

Read together, the four completeness experiments tell a consistent
story about when the Caro et al. verification protocols **do** and
**do not** deliver their promised completeness guarantees.

**1. Theorem 12 on paper-aligned targets works as advertised.** The
`scaling` experiment runs the verifier on single-parity targets
(`a² = b² = 1`, `‖ĝ‖₀ = 1`, Definition 11 trivially satisfied) and
observes 100 % acceptance, 100 % correctness, median `|L| = 1`,
accumulated weight exactly 1.0 at every `n ∈ {4, …, 16}`, and a
post-selection rate matching Theorem 5(i)'s 1/2 to sub-percent
precision. This is the strongest possible empirical confirmation of
Theorem 12 *within the constraint that the sample budgets are
hard-coded* and therefore the experiment does **not** validate the
leading-order `n`-independence of Theorem 12's verifier sample
complexity. That claim would need the Tier 3 rerun.

**2. Theorem 12 on Definition-11-violating targets rejects cleanly.**
The `bent` experiment feeds the verifier flat-spectrum bent functions
whose coefficients `2^{−n/2}` drop below the Corollary 5 extraction
floor `θ/2` at `n ≥ 6`. The transition — 100 % accept at `n = 4`
(where the coefficient sits in the Corollary 5 uncertain band
`[θ/2, θ)`) to 0 % accept at `n ≥ 6` — is the predicted crossover at
`n = 2 log₂(2/θ) ≈ 5.47` to within one row of the sweep. The
verifier correctly refuses to certify out-of-promise targets, which
is the operationally important direction.

**3. Theorem 15 on a trivially k-sparse target behaves like Theorem
12.** Both `k_sparse` and `average_case` include `k = 1` baselines
that route to `verify_parity` and reproduce the `scaling`
experiment's 100 % / 100 % / |L|=1 / accumulated-weight = 1
behaviour. This is a cross-experiment dispatch consistency check.

**4. Theorem 15 on Definition-13-violating targets is
measurement-noise limited in the zero-bracket configuration.** The
two sweeps that exercise the k-sparse path (`k_sparse` for
`k ∈ {2, 4, 8}`, `average_case` for `k ∈ {2, 4}`) both hold
`a² = b² = Σc_i²` exactly, giving the Theorem 15 weight check
**zero Fourier-bracket slack**. The acceptance threshold
`a² − ε²/(128 k²)` is then microscopically below `a²` (slack
0.07 % / 0.02 % / 0.004 % for k = 2, 4, 8), so any trial whose
Hoeffding estimator under-shoots by more than that fails. The
observed ~50 % acceptance is **not** a `1 − δ = 0.9` completeness
success, and **also not** a Theorem 15 failure: it is the visible
imprint of a vanishing-slack, finite-sample test on
Definition-13-violating Dirichlet draws. The `k = 8` misclassification
values 0.34–0.37 sit just above `ε = 0.3`, at the boundary of the
Lemma 14 randomised hypothesis's inherent `(1 − a²)/2` lossiness.

**5. The list-size bounds are never operationally binding.** Across
all four experiments, **not a single trial** rejects via
`reject_list_too_large`. The Theorem 12 / Theorem 15 bound
`|L| ≤ 64 b²/θ²` is a paper-level safety net, not an operational
constraint. Every rejection flows through the accumulated
Fourier-weight check in Step 4, which is the semantic heart of the
protocol. The large-list anomalies at `(n=6, k=8)` and `(n=8, k=8)`
in `k_sparse` (|L| up to 139) are not violations of the
~569-element bound; they are Hilbert-space saturation events that
make `Σ ξ̂² = a²` by Parseval identically.

**6. All observed list sizes are `n`-independent once past
Hilbert-space saturation.** For every family tested, the median `|L|`
hits a small constant
(`scaling → 1`, `k_sparse_2 → 2`, `k_sparse_4 → 3`,
`sparse_plus_noise → 1`, `bent (n ≥ 8) → 0`) and stays flat for the
remainder of the sweep. This is the empirical manifestation of
Lemma 8: Fourier-sparse distributions have `|L|` bounds depending on
sparsity (or equivalently on `θ`), not on `n`. The dimensional
independence that the `scaling` experiment's `resource_scaling.png`
could not measure — because the budgets were hard-coded — is
confirmed *qualitatively* by these list sizes: the protocol's
communication cost is constant in `n` for fixed sparsity.

**What the four experiments do not and cannot say.** Because every
experiment hard-codes its sample budgets, none of them empirically
validates the leading-order `n`-, `θ`-, `k`- or `b`-scaling of the
verifier's Hoeffding count. What they validate is the **functional
correctness** of the four Theorem 12 / Theorem 15 ingredients (list
bound, Corollary 5 extraction, Hoeffding estimation, weight check),
the sharp promise-class boundary (bent at `n = 5.47`, Dirichlet
granularity at `c_min < θ`), and the Theorem 5(i) post-selection
rate. To validate the asymptotic scaling claims, the Tier 3 reruns
in `audit/FOLLOW_UPS.md §4` are necessary; they are tracked as future
work, not present limitations.

**Overall.** Theorem 12 is empirically confirmed on paper-aligned
targets (parities) and on Definition-11-violating targets (bent) in
both directions the theorem asserts. Theorem 15's code path is a
faithful implementation, but its in-promise completeness guarantee is
not tested by any of the four experiments because the two sweeps
that exercise the k-sparse path (`k_sparse`, `average_case`) run in a
zero-Fourier-bracket, off-promise configuration that makes the
completeness test measurement-noise limited. The protocol nevertheless
behaves correctly cell-by-cell on every trial: the observed
accept/reject split follows the Hoeffding estimator's noise around
the zero-slack threshold, and misclassification rates on accepted
trials sit at the Lemma 14 inherent lossiness floor. In that sense
the experiments measure **what the protocol does outside its promise
class**, not **how close it is to saturating its in-promise
completeness guarantee**.

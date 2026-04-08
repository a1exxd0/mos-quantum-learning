# Parameter-sensitivity audit: theta_sensitivity and truncation

Auditor: Agent 4. Two parameter-sensitivity experiments in `/Users/alex/cs310-code` audited against Caro et al. (ITCS 2024), "Classical Verification of Quantum Learning" (arXiv:2306.04843, 81 pp.). Both exercise the §6 verification protocol; the protocol code (`ql/prover.py`, `ql/verifier.py`) was previously audited line-by-line and is correct. All findings here are about experimental framing, hard-coded sample budgets, and figure interpretation. Every empirical claim was re-verified against the raw `.pb` artefacts via `uv run python -m experiments.decode`; nothing leans on the pre-computed CSVs.

---

## 1. theta_sensitivity — extraction boundary as a function of θ

### 1.1 Data, parameters, hard-coded constants

- **Data file**: `/Users/alex/cs310-code/results/theta_sensitivity_4_16_100.pb` (5 600 trials).
- **Sweep**: `n ∈ {4, 6, 8, 10, 12, 14, 16}` × `θ ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50}` × 100 trials/cell.
- **Fixed**: `ε = 0.3`, `δ = 0.1`.
- **Target**: `make_sparse_plus_noise` (`/Users/alex/cs310-code/experiments/harness/phi.py:228`) — one dominant Fourier coefficient at magnitude 0.7 plus three secondaries at magnitude 0.1, true sparsity = 4, Parseval weight `0.49 + 0.03 = 0.52`. Per-trial re-randomisation of the parity indices (`phi.py:259`), so per-cell variance includes target-distribution variance.
- **Hard-coded budgets** (`experiments/harness/theta_sensitivity.py:21-23`): `qfs_shots = 2000`, `classical_samples_prover = 1000`, `classical_samples_verifier = 3000`. These override the analytic θ-dependent formulas at `ql/prover.py:316-321` and `ql/verifier.py:493-501`.
- **Verifier branch used**: parity (`verify_parity`), because `spec.k is None` at `experiments/harness/worker.py:179`. So:
  - List bound applied: `|L| ≤ 64 b²/θ² = 33.28/θ²` (`ql/verifier.py:464`).
  - Per-coefficient tolerance: `ε²/(16|L|) = 0.09/(16|L|)` (`ql/verifier.py:488`).
  - Acceptance threshold: `τ = a² − ε²/8 = 0.52 − 0.01125 = 0.50875` (`ql/verifier.py:515`).

### 1.2 The precise paper claims being tested

- **(i) List-size bound**. Corollary 1 (p. 21) gives `‖ĝ‖_0 ≤ 4/ε²`; Corollary 5 (p. 28) gives `‖φ̃‖_0 ≤ 16/ε²`; Theorem 8 (p. 38) enforces `|L| ≤ 4/ϑ²`; Theorem 12 (p. 45) — the one the code implements — relaxes this to `|L| ≤ 64 b²/ϑ²` (factor-of-4 slack from working with θ/2 resolution in Corollary 5).
- **(ii) Corollary 5 / Theorem 4 extraction boundary** (pp. 25, 27-28). For each candidate s:
  - guaranteed in: `|ĝ(s)| ≥ ϑ` ⇒ `s ∈ L`
  - guaranteed out: `s ∈ L` ⇒ `|ĝ(s)| ≥ ϑ/2`
  - uncertain: `ϑ/2 ≤ |ĝ(s)| < ϑ`

  For `sparse_plus_noise`: at θ ≤ 0.10 the four true coefficients are guaranteed in (`|L| ≥ 4`); at 0.10 < θ ≤ 0.20 the secondaries are in the uncertain zone; at θ > 0.20 the secondaries are guaranteed out (only the dominant can appear, `|L| ∈ {0, 1}`).
- **(iii) Theorem 5(i) postselection rate = 1/2** (p. 26), independent of D and of θ.
- **(iv) Verifier weight check** (Theorem 12, p. 45 Step 4): `Σ_{s∈L} ξ̂(s)² ≥ a² − ε²/8`. With `a² = 0.52`, `ε = 0.3`, `τ = 0.50875`. If `|L| = 1` (only dominant), accumulated weight → `0.49 < τ` → reject.

**Out-of-promise disclaimer**: Definition 11 (p. 35) requires every nonzero Fourier coefficient `≥ ϑ`. The target has secondaries at exactly 0.1, so for `θ > 0.10` the distribution is **outside** `D^func_{U_n;≥ϑ}` and Theorems 7/8/12 do not formally apply. The harness docstring (`theta_sensitivity.py:69-76`) discloses this — the sweep is intentionally probing the out-of-promise regime to map verifier behaviour.

**What the experiment does NOT test**: the analytic `1/θ⁴` prover sample-complexity scaling of Corollary 5, nor the `O(|L|² log(|L|/δ)/ε⁴)` verifier Hoeffding scaling of Theorem 12. The hard-coded budgets flatten the θ-dependence.

### 1.3 Figures (in `/Users/alex/cs310-code/results/figures/theta_sensitivity/`)

- **`acceptance_heatmap.png`** (7×8 grid, n × θ, RdYlGn colour, percentage labels). Left block (θ ≤ 0.15) uniformly green/yellow (69–100%); θ = 0.20 yellow band (39–82%); θ ∈ {0.30, 0.50} uniformly red (10–22%) for n ≥ 6. The n=4 row is anomalous (still ~79% at θ=0.30) because at n=4 we have only 16 parities and the prover pulls them all into L, inflating accumulated weight via squaring bias by ~0.005, just enough to clear the 0.009 margin. A dashed navy vertical at θ=0.20 is annotated "extraction boundary θ=0.20".
- **`list_size_vs_theta.png`** (log-y line plot, median |L| vs θ for n ∈ {4, 8, 16}, IQR shading, dashed `4/θ²` overlay). The dashed line is correctly labelled "`4/θ²` (Thm 7/9 SQ-verifier bound)" — audit fix m3 against the original "Parseval bound" wording. The n=16 curve drops from ~484 at θ=0.05 through 4 at θ ∈ [0.10, 0.15] to 1 at θ ≥ 0.30.
- **`tradeoff_curve.png`** (dual y-axis line plot for n ∈ {8, 16}: left = acceptance, right = log median |L|). Cleanest narrative: as θ grows, communication |L| falls monotonically but acceptance collapses past θ ≈ 0.20.

**Visual overclaims**: minimal. The `4/θ²` legend is correct. Missed opportunity: no figure plots the tighter `64 b²/θ² ≈ 33.28/θ²` bound the verifier actually enforces; no figure splits `reject_list_too_large` from `reject_insufficient_weight`.

### 1.4 Independent verification against the decoded `.pb`

Re-decoded to `/tmp/theta_sens.json`. Cross-checks of all four claims:

**(i) List-size bound**. Across all 56 cells, both `4/θ²` and `64 b²/θ²` are **never violated**. Worst case n=16, θ=0.05: max |L| = 540 vs `4/θ² = 1600` and `64 b²/θ² = 13312` (3× and 25× headroom).

**(ii) Corollary 5 three-regime structure**. |L| histograms at n=16:

| θ | |L| histogram | Corollary-5 regime |
|---|---|---|
| 0.05 | range [429, 540], median 484 | guaranteed in, DKW-limited |
| 0.08 | {4: 24, 5: 25, 6: 22, 7: 16, 8: 8, 9: 5} | guaranteed in, tight |
| 0.10 | {3: 2, 4: 98} | **all true sparsity recovered** |
| 0.12 | {3: 6, 4: 94} | uncertain zone, mostly in |
| 0.15 | {2: 1, 3: 10, 4: 89} | uncertain zone, mostly in |
| 0.20 | {1: 16, 2: 44, 3: 34, 4: 6} | uncertain, ≈2.3 average |
| 0.30 | {1: 100} | **secondaries guaranteed out → only dominant** |
| 0.50 | {1: 100} | dominant only |

Every row matches Corollary 5. The 540-element worst case at θ=0.05 is DKW noise — extraction threshold `θ²/4 = 0.000625` at ~1000 postselected QFS samples pulls in roughly half of the 2¹⁶ parities once.

**(iii) Postselection rate**. Median across all 56 cells ranges 0.496–0.505; mean over all 5600 trials is 0.4998. **Theorem 5(i) confirmed to three decimal places**, independent of θ.

**(iv) Single-coefficient rejection at θ ≥ 0.30**. Threshold τ = 0.50875. With `|L|=1`, `ξ̂(s_dom)²` is unbiased for 0.49 with std `2|0.7|·sqrt(0.51/3000) = 0.01825`. Gaussian tail predicts accept rate `1 − Φ((0.49 − 0.50875)/0.01825) = 15.2%`. Empirical (averaged over n=6..16) 13–19%. Wilson CIs always contain 15.4%. This is the verifier correctly refusing to certify a distribution that has broken the Definition 14 bracket promise.

**Plateau cross-check at θ=0.10**: cells with `|L|=4` have empirical mean accumulated weight 0.520–0.523 — matches the true Parseval mass 0.52 to 3 decimal places.

**Analytic budgets at the swept θ**:

Prover (`ql/prover.py:318`):

| θ | analytic m_shots | used | factor short |
|---|---|---|---|
| 0.05 | **1.89 × 10⁸** | 2 000 | **94 435×** |
| 0.10 | 1.18 × 10⁷ | 2 000 | 5 902× |
| 0.20 | 737 778 | 2 000 | 369× |
| 0.30 | 145 735 | 2 000 | 73× |
| 0.50 | 18 888 | 2 000 | 9× |

Verifier at ε=0.3, δ=0.1, with empirical median |L| at n=16:

| θ | median |L| | analytic m_V | used | factor short |
|---|---|---|---|---|
| 0.05 | 484 | **1.46 × 10¹¹** | 3 000 | **4.9 × 10⁷×** |
| 0.10 | 4 | 5.13 × 10⁶ | 3 000 | 1 710× |
| 0.20 | 2 | 1.23 × 10⁶ | 3 000 | 410× |
| 0.30 | 1 | 2.33 × 10⁵ | 3 000 | 78× |
| 0.50 | 1 | 2.33 × 10⁵ | 3 000 | 78× |

The experiment is between ~10× and ~5 × 10⁷× under the Hoeffding budgets — it cannot validate the `1/θ⁴` scaling.

### 1.5 Verdict — PASS-with-caveats

The experiment is a faithful, quantitatively precise empirical map of Corollary 5's extraction boundary and Theorem 12's weight-check mechanism. All four claims (list bound, three-regime |L|, postselection rate, single-coefficient rejection) hold to within statistical precision. The 16% accept-rate floor at θ ∈ {0.30, 0.50} is exactly the Gaussian-tail prediction.

The two audit caveats (`audit/theta_sensitivity.md`) M1 (sub-Hoeffding budgets) and M2 (out-of-promise distribution at θ > 0.10) are correctly disclosed in the harness docstring. Verifier-side budgets are 78× to 5×10⁷× short of Theorem 12; the experiment maps the *boundary*, not the *scaling*.

### 1.6 Open issues / follow-ups

- **Tier-3 rerun** (`audit/FOLLOW_UPS.md`): θ-driven analytic budget sweep with `qfs_shots = None`, `classical_samples_verifier = None` at small n (≤ 8) to actually validate the `1/θ⁴` scaling.
- Add a heatmap that splits `reject_list_too_large` vs `reject_insufficient_weight` (data already in `TrialResult.outcome`).
- Overlay the Theorem-12 `64 b²/θ²` bound in `list_size_vs_theta.png` alongside the existing `4/θ²` bound.

---

## 2. truncation — verifier sample-budget sweep

### 2.1 Data, parameters, hard-coded constants

- **Data files**: `/Users/alex/cs310-code/results/truncation_{n}_{n}_100.pb` for n ∈ {4..14}; 33 000 trials total across 11 × 5 × 6 = 330 cells.
- **Sweep per n**: `ε ∈ {0.1, 0.2, 0.3, 0.4, 0.5}` × `m_V ∈ {50, 100, 200, 500, 1000, 3000}` × 100 trials.
- **Fixed**: noise rate `η = 0.15` → `a² = b² = (1 − 2η)² = 0.49`. `θ = min(ε, 0.5)` (`truncation.py:141`) — **θ is coupled to ε**, audit MINOR m6/n9.
- **Target hard-coded**: `target_s = 1` (`truncation.py:125`) for **every** trial of every cell at every n — audit MINOR m3 (single fixed instance).
- **Hard-coded prover budgets**: `qfs_shots = 2000`, `classical_samples_prover = 1000`. **Verifier budget is the swept axis** (`m_V`).
- **Verifier branch**: parity (`verify_parity`), so list bound `64 b²/θ² ≈ 31.36/θ²`, per-coeff tolerance `ε²/(16|L|)`, threshold `τ = 0.49 − ε²/8 ∈ {0.48875, 0.485, 0.47875, 0.47, 0.45875}`. Margin `a² − τ = ε²/8` ranges from **0.00125** (ε=0.1) to **0.03125** (ε=0.5).

### 2.2 The precise paper claim being tested

Theorem 12 (p. 45) verifier sample complexity: `O(|L|² log(|L|/δ)/ε⁴)` classical random examples to obtain simultaneously `(ε²/(16|L|))`-accurate `ξ̂(s)`. The concrete formula at `ql/verifier.py:497-500` is

```
m ≥ (2/tol²) · log(4|L|/δ),  tol = ε²/(16|L|)
```

This is the **completeness budget**: it guarantees the weight check `Σ ξ̂² ≥ a² − ε²/8` fires correctly for honest provers.

**What the experiment claims to test** (`truncation.py:32-47`): "the completeness frontier" — the minimum m_V at which the protocol achieves high acceptance for each (n, ε).

**What the experiment in fact tests**: NOT Theorem 12. The grid is 3–4 orders of magnitude below the analytic budget for the `|L|=1` best case:

| ε | tol = ε²/16 | analytic m_V (\|L\|=1) | grid max | factor short |
|---|---|---|---|---|
| 0.1 | 0.000625 | **18 887 063** | 3 000 | 6 296× |
| 0.2 | 0.0025 | 1 180 442 | 3 000 | 393× |
| 0.3 | 0.005625 | 233 174 | 3 000 | 78× |
| 0.4 | 0.01 | 73 778 | 3 000 | 25× |
| 0.5 | 0.015625 | **30 220** | 3 000 | 10× |

Even the largest ε is 10× short. For small-n cells where the prover's under-resolved DKW produces |L| ≈ 2ⁿ − 1 (e.g. |L| ≈ 63 at n=6, ε=0.1), the analytic m_V at `|L|=63, ε=0.1` is `1.6 × 10¹¹` — every cell of the grid is in the **sub-Hoeffding regime**.

### 2.3 Figures (in `/Users/alex/cs310-code/results/figures/truncation/`)

- **`heatmap_acceptance.png`** — 6-panel small-multiples (n ∈ {4, 6, 8, 10, 12, 14}), each panel a 5×6 grid (ε rows × m_V cols), RdYlGn colour. Critical visual feature: at ε=0.1, small-n panels have a bright-green **left** column (m_V=50 → 97% accept at n=4) fading to yellow/green on the right (m_V=3000 → 59% at n=4). At ε=0.5 the pattern reverses (left red, right green) — the only "Hoeffding-looking" regime. No caption warns the reader that the ε=0.1 left-column green is a squaring-bias artefact.
- **`sample_budget_knee.png`** — two side-by-side line plots (ε=0.3, ε=0.5) of acceptance vs log(m_V), one line per n, Wilson CIs, dashed 90% line. ε=0.3 panel: curves nearly flat at 50–80%, no clear knee. ε=0.5 panel: every n curve rises monotonically to 95–100% at m_V=3000. **The ε=0.5 panel is the only one in the experiment that looks like a Hoeffding curve.**
- **`min_viable_budget.png`** — two-panel: (a) min m_V for ≥90% accept vs n, log-y, one line per ε; (b) max acceptance vs n. The ε=0.1 line in (a) sits at m_V=50 for n ∈ {4, 5, 6} (dangerously misleading: it reads as "you only need 50 samples"; in reality at small n the squaring bias `|L|/m_V ≈ 16/50 = 0.32` saturates acceptance) and jumps to the 3000 ceiling for n ≥ 7.

**Visual overclaims**: the figure titles are technically neutral, but the **interpretation risk** is severe. A reader who hasn't read the docstring will conclude "protocol completeness easily achieved with 50 samples at small ε" when the truth is the opposite — that's the squaring bias artefact. None of the figures overlays the squaring-bias prediction `a² + (|L| − a²)/m_V`, which would make the artefact unambiguous.

### 2.4 Independent verification

**Inversion check** across all 330 cells (Wilson 95% CIs):

| ε | n | accept@50 | accept@3000 | inversion |
|---|---|---|---|---|
| 0.1 | 4 | 97 [92, 99] | 59 [49, 68] | **YES (CIs disjoint)** |
| 0.1 | 5 | 100 | 74 | YES |
| 0.1 | 6 | 100 | 88 | YES |
| 0.1 | 8 | 100 | 94 | YES |
| 0.1 | 9 | 100 | 83 | YES |
| 0.1 | 10 | 96 | 64 | YES |
| 0.1 | 14 | 59 | 47 | YES |
| 0.3 | 4 | 98 | 83 | YES |
| 0.5 | all n | monotone↑ | 95–100 | none (normal) |

The (n=4, ε=0.1) inversion is **statistically rock-solid** — CIs do not overlap.

**Squaring-bias prediction** `pred = 0.49 + (|L| − 0.49)/m_V` vs empirical median accumulated weight (representative cells, **±4% accuracy in every row**):

| n | ε | m_V | med \|L\| | pred | obs | err |
|---|---|---|---|---|---|---|
| 4 | 0.1 | 50 | 16 | 0.8002 | 0.8320 | 4.0% |
| 4 | 0.1 | 3000 | 16 | 0.4952 | 0.4925 | −0.5% |
| 6 | 0.1 | 50 | 63 | 1.7402 | 1.7640 | 1.4% |
| 6 | 0.1 | 3000 | 63 | 0.5108 | 0.5127 | 0.4% |
| 8 | 0.1 | 50 | 82 | 2.1302 | 2.1496 | 0.9% |
| 8 | 0.1 | 3000 | 82 | 0.5172 | 0.5194 | 0.4% |
| 10 | 0.1 | 3000 | 16 | 0.4952 | 0.4965 | 0.3% |

Conclusive: **the figures are tracing the squaring-bias landscape of the unbiased estimator**, not a Theorem 12 sample-complexity boundary.

**Gaussian-tail accept prediction at ε=0.5, |L|=1** (the "normal" regime, target only): `E[ξ̂²] = 0.49 + 0.51/m_V`, `Var(ξ̂²) ≈ 1/m_V`, threshold `τ = 0.45875`:

| m_V | pred mean | pred std | pred accept | obs accept |
|---|---|---|---|---|
| 50 | 0.5002 | 0.1414 | 61.5% | 69.4% |
| 100 | 0.4951 | 0.1000 | 64.2% | 72.0% |
| 500 | 0.4910 | 0.0447 | 76.5% | 73.4% |
| 1000 | 0.4905 | 0.0316 | 84.2% | 85.6% |
| 3000 | 0.4902 | 0.0183 | **95.7%** | **97.8%** |

ε=0.5 traces a smooth approach to the Hoeffding knee. The analytic m_V is 30 220, the experiment's 3 000 is 10× short, but the wider `ε²/8 = 0.03125` margin lets variance reduction beat the disappearing bias.

**`proverFoundTarget` audit**: `True` in **100.00%** of all 33 000 trials. This rules out "prover failed to find target" as a cause of the rejection pattern — the n-dependence is purely |L|-size variability driven by the prover's under-resolved DKW.

### 2.5 Verdict — MISFRAMED

The protocol code is correct (`ql/verifier.py:464, 488, 497-500, 515` all match Theorem 12 verbatim) and the data are clean, but the experimental framing is wrong: the m_V grid is 3–4 OOM below the Theorem 12 budget in every cell. The figures do not measure a Theorem 12 frontier — they measure the squaring-bias landscape of `ξ̂²` when read by a threshold sitting only `ε²/8` below the true Parseval mass. The bias `(|L| − a²)/m_V` is comparable to or larger than the acceptance margin for every ε ≤ 0.3 cell, which is exactly why the inversion (acceptance *decreasing* with budget) is observed in 8+ ε=0.1 cells.

Both audit findings reproduced numerically:
- **M1** (sub-Hoeffding by 3–4 OOM): verified against `ql/verifier.py:497-500` formula.
- **M2** (squaring bias is dominant mechanism): verified to ±4% precision across 20+ cell-by-cell comparisons against `pred = a² + (|L| − a²)/m_V`.

What the experiment is good for:
1. Reproducing Theorem 12 soundness in the ε=0.5 regime (the only place the protocol approaches its asymptotic curve);
2. Demonstrating the squaring-bias failure mode of naïve sub-Hoeffding deployments;
3. A non-asymptotic acceptance-surface map that is qualitatively informative.

What it cannot do: claim to have measured the Theorem 12 sample-complexity frontier. Dissertation text must caption the figures as "non-asymptotic / sub-Hoeffding regime" (the harness docstring already does so).

### 2.6 Open issues / follow-ups

- **Tier-3 rerun** (`audit/FOLLOW_UPS.md`): extend `verifier_sample_range` to `[3 000, 10 000, 30 000, 100 000, 300 000]` at ε ∈ {0.4, 0.5}, n ≤ 8. At (ε=0.5, |L|=1) this crosses the 30 220 analytic knee and would actually validate the soft Hoeffding transition. ε ≤ 0.3 needs 10⁵–10⁷ samples — cluster only.
- Randomise `target_s` per trial (audit m3).
- Overlay the squaring-bias prediction `a² + (|L| − a²)/m_V` on a new accumulated_weight panel.
- Re-caption `min_viable_budget.png` and `sample_budget_knee.png` as "non-asymptotic / sub-Hoeffding regime".
- Decouple θ from ε (currently `θ = min(ε, 0.5)` partly contaminates the ε=0.5, |L|=1 regime).

---

## 3. Cross-section synthesis

The two experiments probe Theorem 12 (verifier completeness) and Corollary 5 / Theorem 15 (prover extraction) from perpendicular axes: theta_sensitivity fixes budgets and sweeps θ, truncation fixes θ (coupled to ε) and sweeps m_V.

**On the analytic Theorem 12 sample budgets**:

| experiment | sweep | analytic budget range | actual | factor short |
|---|---|---|---|---|
| theta_sensitivity | θ ∈ [0.05, 0.50] | m_V ∈ [2.3 × 10⁵, 1.5 × 10¹¹] | 3 000 | 78× – 5 × 10⁷× |
| truncation | m_V ∈ [50, 3 000] | m_V ∈ [3.0 × 10⁴, 1.9 × 10⁷] | 3 000 | 10× – 6 300× |

**Neither experiment is in the Theorem 12 asymptotic regime.** The `O(|L|² log(|L|/δ)/ε⁴)` bound from `ql/verifier.py:497-500` is unreachable on a laptop for ε ≤ 0.3.

The `1/ε⁴` scaling is visible **qualitatively but not quantitatively**: at ε=0.1 the 100% completeness target requires m_V ≈ 1.9 × 10⁷ (6 300× the 3 000 used); at ε=0.5 it requires m_V ≈ 3 × 10⁴ (10× the 3 000 used). That's a factor of ~630 in required m_V for a 5× shrink in ε, consistent with the `1/ε⁴ = 5⁴ = 625` prediction — but only as endpoints, not as a measured trend.

The tightness of the `a² − ε²/8` threshold is the load-bearing mechanism behind the `1/ε⁴`: shrinking ε by 2× narrows the margin by 4×, forcing variance (∝ 1/m_V) to shrink by 16×. This is direct margin-vs-variance arithmetic, not a statistical accident.

**On the role of θ in Corollary 5 / Theorem 15**:

- **Corollary 5's three-regime structure is empirically crisp**. At θ ≤ 0.10, |L| ≥ 4 in 98+% of n=16 trials. At θ ∈ (0.10, 0.20] |L| smoothly interpolates between 1 and 4. At θ > 0.20, |L| = 1 in 100% of trials at every n ∈ {6..16}. Cleanest possible confirmation of the extraction boundary.
- **θ and ε interact multiplicatively** in the combined Theorem 12 budget `O(b⁴ log(1/δθ²) / (ε⁴ θ⁴))`. theta_sensitivity exposes the `1/θ⁴` factor (through `|L|² ∝ 1/θ⁴`); truncation exposes the `1/ε⁴` factor (through `tol⁻²`). Going from `(θ, ε) = (0.5, 0.5)` to `(0.05, 0.1)` blows up the budget by `10⁴ · 5⁴ = 6.25 × 10⁶` — exactly what makes the experiments infeasible.
- **Neither experiment exercises Theorem 15's k-sparse threshold** `a² − ε²/(128 k²)`. Even though the theta_sensitivity target is 4-sparse, the harness calls `verify_parity` (`worker.py:179`, because `spec.k is None`) with the loose parity threshold `a² − ε²/8 = 0.50875`, not the much tighter k=4 threshold `≈ 0.5196`. A `spec.k = 4` follow-up would test Theorem 15 proper and would expose the squaring-bias artefact much earlier.

**Bottom line**: the two experiments collectively (a) confirm Corollary 5's extraction guarantee in every testable regime, (b) confirm Theorem 12's weight-check mechanism with Gaussian-tail accuracy, and (c) **do not reach Theorem 12's analytic sample complexity** anywhere — that requires Tier-3 cluster reruns. The dissertation text should describe these as **boundary-mapping** and **mechanism-validation** experiments rather than **scaling-validation** experiments.

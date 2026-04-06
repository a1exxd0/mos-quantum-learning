# Experiment Results: Presentation & Analysis Plan

Date: 2026-04-05

## Context

This project implements and evaluates the MoS (mixture-of-superpositions) verification
protocol from Caro et al. (arXiv:2306.04843). The results chapter needs to present
findings from 11 experiments in a way that is clear, visually compelling, and ties
empirical observations back to the paper's theoretical predictions.

**Key constraint:** This plan does not presume experiment outcomes. Each section
describes *what to measure*, *what theoretical benchmarks to compare against*, and
*what questions the analysis should answer*. Conclusions are drawn from data, not
assumed.

**Data sources:** All visualisations use the re-run data exclusively (100 trials per cell,
n=4--20 where applicable). See `RERUN_PLAN.md` for the full list of `.pb` files.

---

## Chapter Narrative Structure

The results are organised around four investigative questions, each mapping to
properties of the verification protocol:

1. **Completeness** -- Does the protocol accept honest provers, and how does it scale?
   - Experiments: scaling, average_case, k_sparse
   - Paper: Theorems 7, 8, 9, 10

2. **Soundness** -- Does the protocol reject dishonest provers?
   - Experiments: soundness, soundness_multi
   - Paper: Theorems 7, 8 (soundness parts), 12

3. **Robustness** -- How does performance degrade under noise and distributional assumptions?
   - Experiments: noise, gate_noise, ab_regime
   - Paper: Theorems 11, 12, Definition 14

4. **Sensitivity & practical limits** -- How do protocol parameters and function structure
   affect behaviour?
   - Experiments: bent, theta_sensitivity, truncation
   - Paper: Corollary 5, Theorem 12

---

## Non-Graph Artefacts

These artefacts provide context for interpreting experiment results and should appear
*before* the per-experiment sections.

### A1. Protocol Flowchart

**Type:** Diagram (e.g. TikZ, draw.io, or programmatic)

**Content:** A visual depiction of the 4-step verifier-prover interaction from Theorems 7/8:

```
Verifier V                          Prover P
    |                                   |
    |--- "Send list L of heavy          |
    |     Fourier coefficients" ------->|
    |                                   |
    |                              P uses QFS
    |                              (Thm 4 / Cor 1)
    |                              to build L
    |                                   |
    |<-------- L = {s_1, ..., s_|L|} ---|
    |                                   |
    |  Step 3: Check |L| <= 4/theta^2   |
    |  If too large: REJECT             |
    |                                   |
    |  Step 3: Estimate g_hat(s)        |
    |  for each s in L using            |
    |  classical SQs / random examples  |
    |                                   |
    |  Step 4: Weight check             |
    |  sum(g_hat(s_l)^2) >= threshold?  |
    |  If yes: ACCEPT, output h         |
    |  If no:  REJECT                   |
```

**Purpose:** Gives the reader a concrete mental model of what "the protocol" means
before they see any data. Label each step with the experiment(s) that probe it.

### A2. Quantum Fourier Sampling Circuit

**Type:** Circuit diagram (Qiskit drawing or TikZ)

**Content:** The QFS circuit for n qubits:
- n+1 qubit register
- Prepare MoS state rho_D
- Single-qubit Hadamard layer on label qubit
- Computational basis measurement
- Post-selection on label qubit = 1

**Purpose:** Shows the quantum subroutine at the heart of the prover. Annotate with
the post-selection probability (~1/2, Equation 4 in the paper) and the resulting
sampling distribution over {0,1}^n.

### A3. Verification Protocol Pseudocode

**Type:** Algorithm figure (algorithmic/algorithm2e style)

**Content:** Pseudocode for the full protocol as implemented, covering both the
parity (k=1) and k-sparse verification paths. Annotate with:
- Which parameters are configurable (epsilon, delta, theta, qfs_shots, etc.)
- Where the list-size bound |L| <= 4/theta^2 (or 64b^2/theta^2) is enforced
- Where the weight threshold tau = a^2 - epsilon^2/8 is applied

**Purpose:** Bridges the gap between the paper's mathematical description and the
implementation. The reader can trace each experiment's parameter back to a line
in this pseudocode.

---

## Per-Experiment Presentation

### Exp 4. Scaling -- Honest Baseline (Completeness)

**Data source:** `scaling_4_20_100.pb`

**Metrics to extract per n:** acceptance rate, hypothesis correctness, median |L|,
postselection rate, median total_copies, median total_time_s, median prover_time_s,
median verifier_time_s.

| Artefact | Type | What it shows |
|---|---|---|
| Table: Baseline summary | Table | Per-n (4,6,8,...,20): acceptance %, correctness %, median \|L\|, postselection rate, median copies, median time |
| Plot: Completeness vs n | Line plot | Acceptance rate and correctness vs n. Theoretical prediction: both should be >= 1-delta for honest prover (Thm 8) |
| Plot: Postselection rate vs n | Bar chart | Postselection rate per n. Theoretical benchmark: ~0.5 (Equation 4). Deviation signals implementation or finite-sample effects |
| Plot: Resource scaling | Dual-axis line (log-linear) | Primary: median total_copies vs n. Secondary: median wall-clock time. Overlay theoretical bound O(n * log(1/delta*theta^2) / theta^4) from Thm 8 |
| Plot: List size vs n | Line plot (log-linear) | Median \|L\| vs n. For single parities, \|L\| should be O(1). Overlay upper bound 4/theta^2. Distribution box-whiskers if variance is notable |

**Theoretical benchmarks:**
- Postselection rate: 1/2 (Eq. 4)
- |L| upper bound: 4/theta^2 (Thm 7 Step 1)
- Total copies: O(n * log(1/(delta*theta^2)) / theta^4) (Thm 8)
- Acceptance threshold: a^2 - epsilon^2/8

**Analysis questions:**
- Does acceptance remain high as n grows to 20?
- Does postselection rate stay near 0.5?
- How does empirical resource usage compare to theoretical worst-case bounds?
- At what n (if any) does performance begin to degrade?

---

### Exp 7. Average-Case Performance (Completeness, Generalisation)

**Data source:** `average_case_4_20_100.pb`

**Metrics to extract per (n, family):** acceptance rate, correctness, median |L|,
median total_copies.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Acceptance by family | Line plot with error bands | Acceptance rate vs n, one line per function family. 95% CI bands (Wilson interval) |
| Plot: List size by family | Line plot | Median \|L\| vs n, one line per family. Shows how spectral complexity affects communication |
| Table: Cross-family summary | Table | Per (n, family): acceptance %, correctness %, median \|L\|, median copies |

**Theoretical benchmarks:**
- For k-sparse families: |L| should scale with k, not n (Thm 9)
- Acceptance should remain high for all families if targets lie in the protocol's promise class

**Analysis questions:**
- Does the protocol generalise beyond single parities?
- Which function families (if any) cause degraded performance?
- Does |L| growth track with the Fourier sparsity of each family?

---

### G1. k-Sparse Verification (Completeness, 2-Agnostic Bound)

**Data source:** `k_sparse_4_20_100.pb`

**Metrics to extract per (n, k):** acceptance rate, correctness, misclassification_rate,
median |L|, median total_copies, hypothesis_coefficients accuracy.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Acceptance vs n by k | Line plot | One line per k in {1,2,4,8}. Shows whether completeness degrades with sparsity |
| Plot: List size vs k | Grouped bar chart | Median \|L\| at fixed n values, grouped by k. Theoretical prediction: \|L\| <= 4/(theta^2), and the list should contain the k heaviest coefficients |
| Plot: Misclassification rate | Heatmap (n x k) | Empirical misclassification rate. Thm 9 guarantees 2-agnostic learning; compare observed rate to the 2*opt + epsilon bound |
| Table: k-sparse summary | Table | Per (n,k): acceptance %, \|L\|, misclassification rate, copies |

**Theoretical benchmarks:**
- 2-agnostic guarantee: misclassification <= 2 * opt + epsilon (Thm 9)
- |L| <= 4/theta^2 (Thm 9 Step 1)
- Verifier sample complexity: O(1/(theta^2)) SQs (Thm 9)
- Communication: O(n/theta^2) bits

**Analysis questions:**
- Does the factor-of-2 agnostic loss from Thm 9 materialise empirically?
- How does |L| grow with k? Is it dominated by k or by theta?
- At what k does the protocol struggle (if at all)?

---

### Exp 1. Soundness -- Single-Parity Dishonest Prover

**Data source:** `soundness_4_20_100.pb`

**Metrics to extract per (n, strategy):** rejection rate, outcome breakdown
(reject_list_too_large vs reject_insufficient_weight).

| Artefact | Type | What it shows | Status |
|---|---|---|---|
| Plot: Rejection by strategy | Grouped bar chart | Rejection rate (y) by strategy (x-groups), bars for each n. Four strategies: random_list, wrong_parity, partial_list, inflated_list | DONE |
| Plot: Rejection mechanism breakdown | Stacked bar chart | For each strategy: proportion rejected by list-size check (Step 3) vs weight check (Step 4). Shows *how* the protocol catches each cheat | DONE |
| Plot: Rejection mechanism by n | Faceted stacked bar | Per-strategy breakdown across all n values (bonus artefact) | DONE |
| Table: Soundness summary | Table (CSV + LaTeX) | Per (strategy, n): rejection rate with 95% Wilson CI | DONE |

**Output directory:** `results/figures/soundness/`

**Script:** `results/figures/soundness/plot_soundness.py`

**Theoretical benchmarks:**
- Soundness guarantee: rejection probability >= 1 - delta (Thms 7, 8)
- random_list / wrong_parity: should be caught by weight check (Step 4)
- inflated_list: should be caught by list-size bound (Step 3)
- partial_list: should be caught by weight check (accumulated weight too low)

**Analysis questions:**
- Do all strategies get rejected at the theoretical rate?
- Which rejection mechanism (list-size vs weight) fires for each strategy?
- Does soundness strengthen or weaken as n grows?

**Findings:**
- Wrong parity, partial list, inflated list: 100% rejection at all n (exceeds 1-delta=0.9).
- Random list: rejection rises from 71% (n=4) to 100% (n>=11), matching 1-5/2^n collision probability.
- All rejections are via weight check (Step 4); no strategy triggers the list-size bound (Step 3).
  This is expected: all lists are <=10 entries vs the bound 4/theta^2 ~ 44.

---

### G3. Soundness -- Multi-Element (k-Sparse Dishonest Prover)

**Data source:** `soundness_multi_4_20_100.pb`

**Metrics to extract per (n, k, strategy):** rejection rate, outcome breakdown.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Rejection vs n by k | Line plot | One line per k, showing rejection rate vs n for each strategy (faceted by strategy) |
| Plot: Comparison with single-element | Grouped bar chart | Side-by-side: single-element soundness (Exp 1) vs multi-element soundness at matched n values. Shows whether k-sparse targets make cheating easier |
| Table: Multi-element soundness | Table | Per (strategy, k, n): rejection rate |

**Theoretical benchmarks:**
- Same soundness guarantees as Thms 7/8 but for k-sparse targets
- Weight threshold adjusted for k: tau = a^2 - epsilon^2/(32k^2) (Thm 9 Step 4)

**Analysis questions:**
- Is soundness maintained for multi-element targets?
- Does increasing k make any strategy harder to reject?
- Are there (n, k) combinations where soundness degrades?

---

### Exp 2. Noise Sweep -- Label-Flip Noise (Robustness)

**Data source:** `noise_sweep_4_20_100.pb`

**Metrics to extract per (n, eta):** acceptance rate, correctness, median |L|,
accumulated_weight.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Noise heatmap | Heatmap (n x eta) | Acceptance rate as colour. Rows = n (4..20), columns = eta (0.0, 0.05, ..., 0.4). Reveals the joint effect of noise and dimension |
| Plot: Acceptance & correctness vs eta | Line plot (faceted by n) | Two or three representative n values. Solid = acceptance, dashed = correctness. Vertical line at theoretical threshold eta_max. Shade theoretically-safe region |
| Plot: Fourier weight attenuation | Line plot | Accumulated Fourier weight vs eta for fixed n. Overlay theoretical prediction: weight attenuated by (1-2*eta)^2 (Definition 12). Shows the mechanism of noise-induced failure |
| Table: Breakdown points | Table | Per n: empirical eta at which acceptance drops below 50%, vs theoretical threshold from Thm 11/12 |

**Theoretical benchmarks:**
- Noisy Fourier weight: (1-2*eta)^2 * noiseless_weight (Def 12, Eq. 103)
- Acceptance threshold becomes: (1-2*eta)^2 * a^2 - epsilon^2/8
- Theoretical noise tolerance from Thm 11: protocol works when noise is known and accounted for

**Analysis questions:**
- At what eta does the protocol break down, and does this match (1-2*eta)^2 attenuation?
- Is there an n-dependence in the breakdown point?
- Does the protocol degrade gradually (graceful) or abruptly (phase transition)?

---

### Exp 3. Gate-Level Noise (Robustness -- Empirical)

**Data source:** `gate_noise_4_8_50.pb`

**Note:** This experiment is *not* a headline result. It provides supplementary
empirical evidence about a noise model (depolarising circuit noise) that the paper
does not analyse theoretically. Present concisely.

**Metrics to extract per (n, gate_error_rate):** acceptance rate, correctness.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Gate noise acceptance | Line plot | Acceptance rate vs depolarising error rate p, one line per n (4..8). Compare shape with label-flip curves from Exp 2 at equivalent effective noise |
| Table: Gate noise summary | Small table | Per (n, p): acceptance rate, correctness. Note where protocol breaks down completely |

**Theoretical benchmarks:**
- None directly from the paper (this is novel/empirical)
- Qualitative comparison: gate noise is "worse" than label-flip noise because it corrupts the QFS circuit itself, not just the labels

**Analysis questions:**
- At what gate error rate does the protocol fail?
- How does gate noise compare to label-flip noise at equivalent rates?
- Is there a sharp n threshold beyond which any gate noise is fatal?

---

### G2. a^2 != b^2 Regime (Robustness, Distributional Setting)

**Data source:** `ab_regime_4_20_100.pb`

**Metrics to extract per (n, gap):** acceptance rate, correctness, accumulated_weight,
acceptance_threshold.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Acceptance vs gap | Line plot | Acceptance rate vs gap value (b^2 - a^2), one line per n. Shows how loosening the L2-bounded bias promise affects verification |
| Plot: Threshold margin | Line plot | (accumulated_weight - acceptance_threshold) vs gap. Positive = accepted, negative = rejected. Shows how much "headroom" the verifier has |
| Plot: Accuracy bound | Scatter / line | Empirical accuracy vs theoretical lower bound epsilon >= 2*sqrt(b^2 - a^2) (Thm 13). Tests whether the bound is tight |
| Table: ab_regime summary | Table | Per (n, gap): acceptance %, accuracy, threshold margin |

**Theoretical benchmarks:**
- Definition 14: D in D_{U_n, [a^2, b^2]} requires total Fourier weight in [a^2, b^2]
- Theorem 11: verification works when epsilon >= 2*sqrt(b^2 - a^2)
- Theorem 13: accuracy lower bound -- epsilon cannot be smaller than 2*sqrt(b^2 - a^2)
- Weight threshold: a^2 - epsilon^2/8 (Step 4)

**Analysis questions:**
- How does the gap between a^2 and b^2 affect acceptance?
- Is the epsilon >= 2*sqrt(b^2 - a^2) lower bound tight or loose?
- Does the verifier's weight check become the binding constraint as the gap widens?

---

### Exp 5. Bent Functions (Sensitivity -- Fourier Density)

**Data source:** `bent_4_20_100.pb`

**Metrics to extract per n (even only):** acceptance rate, median |L|,
median total_copies, median total_time_s.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: List size growth | Line plot (log scale y) | Median \|L\| vs n for bent functions. Overlay theoretical prediction \|L\| = 2^(n/2) (all Fourier coefficients have magnitude 2^(-n/2) >= theta/2 for small theta). On same axes, overlay single-parity \|L\| from Exp 4 for comparison |
| Plot: Bent vs parity acceptance | Grouped bar chart | Acceptance rate side-by-side: bent vs single-parity at each n |
| Plot: Resource explosion | Line plot (log scale y) | Total copies and wall-clock time vs n for bent functions. Shows the practical cost of Fourier-dense targets |
| Table: Bent summary | Table | Per n: \|L\|, acceptance %, copies, time. Theoretical \|L\| vs observed |

**Theoretical benchmarks:**
- Bent function Fourier spectrum: all 2^n coefficients have magnitude 2^(-n/2)
- Expected |L|: all coefficients with |g_hat(s)| >= theta/2 are included, so |L| ~ 2^n when theta < 2^(1-n/2)
- Corollary 5: approximate QFS extraction threshold

**Analysis questions:**
- Does |L| grow as 2^(n/2) as predicted?
- At what n does the protocol become impractical for bent functions?
- Does this confirm that Fourier sparsity is load-bearing for protocol efficiency?

---

### G5. Theta Sensitivity (Sensitivity -- Resolution Threshold)

**Data source:** `theta_sensitivity_4_20_100.pb`

**Metrics to extract per (n, theta):** acceptance rate, correctness, median |L|,
postselection rate.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Acceptance heatmap | Heatmap (n x theta) | Acceptance rate as colour. Shows the joint effect of resolution threshold and dimension |
| Plot: List size vs theta | Line plot | Median \|L\| vs theta at representative n values. Theoretical: \|L\| <= 4/theta^2, so smaller theta => larger lists |
| Plot: Trade-off curve | Line plot | At fixed n: acceptance rate (solid) and \|L\| (dashed, log scale) vs theta. Visualises the precision-communication trade-off |
| Table: Theta sensitivity summary | Table | Per (n, theta): acceptance %, \|L\|, correctness |

**Theoretical benchmarks:**
- |L| <= 4/theta^2 (Thm 7 Step 1 / Thm 9 Step 1)
- Corollary 5: approximate QFS succeeds with postselection probability ~1/2 for theta in (2^(-(n/2 - 3)), 1)
- Smaller theta resolves finer Fourier structure but increases |L| and sample complexity

**Analysis questions:**
- Is there a theta "sweet spot" that balances acceptance with communication cost?
- Does the |L| <= 4/theta^2 bound hold empirically, and how tight is it?
- How does the optimal theta shift with n?

---

### Exp 6. Verifier Truncation (Practical Limits)

**Data source:** `truncation_{N}_{N}_100.pb` for N in 4..14

**Metrics to extract per (n, epsilon, verifier_samples):** acceptance rate, correctness.

| Artefact | Type | What it shows |
|---|---|---|
| Plot: Heatmap grid (one per n) | Heatmap (epsilon x verifier_samples) | Acceptance rate as colour. Small multiples for n in {4, 6, 8, 10, 12, 14} show how the "safe region" shrinks with n |
| Plot: Correctness heatmap | Heatmap (same grid) | Hypothesis correctness. May differ from acceptance if the verifier accepts but outputs a wrong hypothesis |
| Plot: Sample budget knee | Line plot | Fix epsilon (e.g. 0.3), plot acceptance vs verifier_samples for multiple n. Identify the minimum sample budget for reliable verification -- the "knee" of the curve |
| Plot: Minimum viable budget vs n | Line plot | Extract the knee point from each n, plot vs n. Shows how verifier cost scales with problem dimension |
| Table: Truncation summary | Table | Per (n, epsilon): minimum verifier_samples for >= 90% acceptance |

**Theoretical benchmarks:**
- Verifier sample complexity: O(|L|^2 * log(|L|/delta) / epsilon^4) from Thm 8 Step 3
- Each SQ needs tolerance epsilon^2 / (16|L|) for Chernoff-Hoeffding
- As n grows, |L| may grow, requiring more verifier samples

**Analysis questions:**
- What is the minimum verifier sample budget for reliable verification at each n?
- How does this budget scale with n? Polynomial? Super-polynomial?
- Is there an epsilon below which the protocol always fails regardless of budget?
- Does correctness track acceptance, or are there "false accepts"?

---

## Cross-Experiment Synthesis

These artefacts combine data from multiple experiments to tell the overall story.

### S1. Summary Overview Figure

**Type:** Multi-panel figure (2x3 or 3x2)

**Panels:**
1. Completeness: acceptance rate vs n (from scaling, overlaid with k_sparse k=1)
2. Soundness: rejection rate by strategy (from soundness, best/worst n)
3. Noise robustness: acceptance heatmap (from noise sweep)
4. Fourier density: bent vs parity |L| growth (from bent + scaling)
5. k-sparse generalisation: acceptance by k (from k_sparse)
6. Verifier cost: minimum budget vs n (from truncation)

**Purpose:** A single figure that captures the headline finding from each experimental
axis. Suitable for referencing in an abstract or introduction.

### S2. Theory-vs-Empirics Comparison Table

**Type:** Summary table

**Content:** One row per theorem/corollary tested. Columns:
- Theorem reference (e.g. "Thm 7")
- Property (e.g. "Soundness, functional, SQ")
- Theoretical prediction (e.g. "Rejection >= 1-delta")
- Experiment(s) that test it
- Empirical finding (filled in after analysis)
- Verdict: confirmed / partially confirmed / not confirmed / inconclusive

**Purpose:** Directly maps between the paper's claims and the experimental evidence.
This is the core deliverable of the results chapter.

### S3. Protocol Parameter Sensitivity Summary

**Type:** Table or radar chart

**Content:** For each tuneable parameter (epsilon, delta, theta, n, k, eta, verifier_samples),
summarise:
- Which experiments vary it
- Observed sensitivity (high/medium/low)
- Whether theoretical predictions about its effect are confirmed

---

## Experiment-to-Theorem Mapping

| Paper theorem | Property | Experiments |
|---|---|---|
| Thm 5 (approx QFS) | Postselection ~1/2 | scaling, theta_sensitivity |
| Thm 7 (functional parity, SQ) | Completeness + soundness | scaling, soundness, average_case |
| Thm 8 (functional parity, RE) | Completeness + soundness | scaling |
| Thm 9 (functional k-sparse, SQ) | 2-agnostic verifiability | k_sparse, average_case |
| Thm 10 (functional k-sparse, RE) | 2-agnostic verifiability | k_sparse |
| Thm 11 (distributional parity, SQ) | Noisy verification | noise, ab_regime |
| Thm 12 (distributional parity, RE) | Noisy verification | scaling, noise, truncation, ab_regime |
| Thm 13 (accuracy lower bound) | epsilon >= 2*sqrt(b^2-a^2) | ab_regime |
| Thm 14 (distributional k-sparse, SQ) | k-sparse distributional | k_sparse |
| Thm 15 (distributional k-sparse, RE) | k-sparse distributional | k_sparse |
| Corollary 5 (approx QFS) | Extraction threshold | bent, theta_sensitivity |
| Definition 14 (L2-bounded bias) | a^2 != b^2 regime | ab_regime |
| Soundness (multi-element L) | Weight check vs adversary | soundness_multi |

---

## Data Files Summary

| Experiment | Output file | n range | Trials | Grid dimensions |
|---|---|---|---|---|
| scaling | `scaling_4_20_100.pb` | 4--20 | 100 | 17 n-values |
| soundness | `soundness_4_20_100.pb` | 4--20 | 100 | 17n x 4 strategies |
| soundness_multi | `soundness_multi_4_20_100.pb` | 4--20 | 100 | 17n x 2k x 4 strategies |
| noise | `noise_sweep_4_20_100.pb` | 4--20 | 100 | 17n x 9 eta |
| gate_noise | `gate_noise_4_8_50.pb` | 4--8 | 50 | 5n x 7 rates |
| bent | `bent_4_20_100.pb` | 4--20 (even) | 100 | 9 n-values |
| truncation | `truncation_{N}_{N}_100.pb` | 4--14 | 100 | 30 grid x 11 files |
| average_case | `average_case_4_20_100.pb` | 4--20 | 100 | 17n x 4 families |
| k_sparse | `k_sparse_4_20_100.pb` | 4--20 (even) | 100 | 9n x 4 k |
| theta_sensitivity | `theta_sensitivity_4_20_100.pb` | 4--20 (even) | 100 | 9n x 8 theta |
| ab_regime | `ab_regime_4_20_100.pb` | 4--20 | 100 | 17n x 6 gaps |

---

## Implementation: Plotting Module

**File:** `experiments/plot.py` (new)

**Dependencies:** `matplotlib`, `seaborn`, `numpy` (add to pyproject.toml)

**Structure:**
```
experiments/plot.py
  load_results(path) -> dict              # deserialise .pb via decode module
  # Per-experiment plot functions
  plot_scaling(data)                       # Exp 4
  plot_average_case(data)                  # Exp 7
  plot_k_sparse(data, scaling_data)        # G1
  plot_soundness(data)                     # Exp 1
  plot_soundness_multi(data, single_data)  # G3
  plot_noise_sweep(data)                   # Exp 2
  plot_gate_noise(data, noise_data)        # Exp 3
  plot_ab_regime(data)                     # G2
  plot_bent(data, scaling_data)            # Exp 5
  plot_theta_sensitivity(data)             # G5
  plot_truncation(data_dict)              # Exp 6 (dict of n -> data)
  # Cross-experiment synthesis
  plot_summary(all_results)                # S1
  build_theory_table(all_results)          # S2
  # Entry point
  main()  # CLI: uv run python -m experiments.plot [all|scaling|noise|...]
```

**Style guidelines:**
- Use `seaborn` with `paper` context and `colorblind` palette
- Export as both PDF (for LaTeX / print) and PNG (for previewing / slides)
- Figure widths: single-column (3.5in) and double-column (7in)
- Consistent axis labels using LaTeX math: `$n$`, `$\eta$`, `$|L|$`, `$\vartheta$`
- Error bars / bands: 95% CI via Wilson interval (proportions) or bootstrap (medians)
- Heatmaps: diverging colourmap (e.g. RdYlGn) with explicit annotation of cell values

**Output directory:** `results/figures/`

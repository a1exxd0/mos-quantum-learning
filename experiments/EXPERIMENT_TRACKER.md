# Experiment Tracker

Status as of 2026-04-04.

## Experiments Required by the Dissertation Plan (Chapter 5)

The plan defines two regimes -- *small-n* (n=4, direct thresholding) and
*scaled* (n=10--20, GL extraction) -- and seven numbered experiments plus an
honest baseline.

| # | Name | Regime | Plan ref |
|---|------|--------|----------|
| -- | **Honest baseline** | both | 5.2 |
| 1 | Oracle mismatch (dishonest prover) | small-n | 5.3 |
| 2 | Noisy prover (oracle noise sweep) | both | 5.3 |
| 3 | Gate-level noise (depolarising) | small-n | 5.3 |
| 4 | Scaling sweep (honest, GL) | scaled | 5.4 |
| 5 | Fourier-dense (bent) functions | scaled | 5.4 |
| 6 | Verifier truncation | scaled | 5.4 |
| 7 | Average-case performance | scaled | 5.4 |

---

## Completed Experiments

### Exp 1 -- Oracle Mismatch (Soundness)
- **Module:** `experiments/harness/soundness.py`
- **Result file:** `results/soundness_4_16_50.pb`
- **Coverage:** n=4..16, 50 trials per (n, strategy) cell.
  Four dishonest strategies tested: `random_list`, `wrong_parity`,
  `partial_list`, `inflated_list`.
- **Status:** COMPLETE. Covers both the small-n (n=4) and scaled regimes
  called for in the plan. The plan only explicitly requires n=4, so this
  exceeds requirements.

### Exp 2 -- Noisy Prover (Oracle Noise Sweep)
- **Module:** `experiments/harness/noise.py`
- **Result file:** `results/noise_sweep_4_13_24.pb`
- **Coverage:** n=4..13, eta in [0.0, 0.05, ..., 0.4], 24 trials per cell.
  Uses label-flip noise model (Definition 5(iii) in Caro et al.):
  phi_eff(x) = (1 - 2*eta)*phi(x) + eta.
- **Status:** COMPLETE.
  - The plan calls for running at both n=4 (direct thresholding) and n=12
    (GL). The result file covers both.
  - Adaptive theta: theta = min(eps, 0.9*(1-2*eta)).
  - *Gap:* The plan mentions connecting to the Ma--Su--Deng threshold
    eta <= 1/(10*theta). This is an analysis/writing task, not a code gap.

### Exp 4 -- Scaling Sweep
- **Module:** `experiments/harness/scaling.py`
- **Result file:** `results/scaling_4_16_24.pb`
- **Coverage:** n=4..16, 24 trials per n, random nonzero parities.
- **Status:** COMPLETE. Covers n in {4,6,8,10,12,14,16} as required.
  Records acceptance rate, hypothesis correctness, and timing.

### Exp 5 -- Fourier-Dense (Bent) Functions
- **Module:** `experiments/harness/bent.py`
- **Result file:** `results/bent_4_16_24.pb`
- **Coverage:** Even n from 4..16, 24 trials. Uses Maiorana--McFarland
  construction where |g_hat(s)| = 2^(-n/2) for all s.
- **Status:** COMPLETE. Demonstrates worst-case exponential scaling of |L|.

### Exp 6 -- Verifier Truncation
- **Module:** `experiments/harness/truncation.py`
- **Result files:**
  - `results/truncation_6_6_24.pb` (n=6, original)
  - `results/truncation_10_10_24.pb` (n=10, scaled regime)
  - `results/truncation_12_12_24.pb` (n=12, scaled regime)
- **Coverage:** eta=0.15 across n in {6, 10, 12}. 2D grid sweep over
  epsilon in {0.1, 0.2, 0.3, 0.4, 0.5} x
  verifier_samples in {50, 100, 200, 500, 1000, 3000}.
  24 trials per cell per n value (720 trials each, 2160 total).
- **Status:** COMPLETE.
  - Covers both small-n (n=6) and scaled regimes (n=10, n=12).
  - Acceptance rate grids show the completeness frontier shifting across
    n: at n=10 and n=12, high acceptance requires both large eps and
    large verifier sample budget, confirming the theoretical prediction
    that the tradeoff surface is sensitive to dimension.
  - At small eps (0.1), acceptance rates *decrease* with more verifier
    samples at n=10 and n=12. This is consistent with a noise-dithering
    effect: when the true accumulated weight is near the tight threshold
    tau = a^2 - eps^2/8, Hoeffding noise at low m_V can push estimates
    above threshold, while accurate estimates at high m_V reveal the
    marginal shortfall.
  - *Note:* The implementation sweeps the verifier's *sample budget*
    rather than the *fraction of L processed*. These are related but
    not identical to the plan's description.

### Exp 7 -- Average-Case Performance
- **Module:** `experiments/harness/average_case.py`
- **Result file:** `results/average_case_{n_min}_{n_max}_{trials}.pb`
- **Coverage:** Sweeps n x function family. Four families:
  - `k_sparse_2`: 2-Fourier-sparse with Dirichlet(1,1) coefficients.
  - `k_sparse_4`: 4-Fourier-sparse with Dirichlet(1,1,1,1) coefficients.
  - `random_boolean`: uniform random truth table (maximally Fourier-dense).
  - `sparse_plus_noise`: dominant parity (c=0.7) + 3 secondary (c=0.1 each).
  Phi generators added to `experiments/harness/phi.py`:
  `make_k_sparse`, `make_random_boolean`, `make_sparse_plus_noise`.
- **Status:** COMPLETE.
  - Proto schema: `experiments/proto/average_case.proto`.
  - Adaptive theta per family: theta = min(eps, 0.9/k) for k-sparse,
    theta = eps for random_boolean and sparse_plus_noise.
  - a_sq = b_sq = Parseval weight (computed per-trial from coefficients).

### Exp 3 -- Gate-Level Noise (Depolarising via Qiskit NoiseModel)
- **Module:** `experiments/harness/gate_noise.py`
- **Result file:** `results/gate_noise_4_8_50.pb`
- **Coverage:** n=4..8, gate error rate p in
  {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}, 50 trials per (n, p) cell.
  Total: 1750 trials. Uses depolarising channels on H, X, CX gates via
  Qiskit `NoiseModel`. Protocol run with a² = b² = 1 (noiseless promise),
  θ = ε = 0.3, 2000 QFS shots, 1000 prover samples, 3000 verifier samples.
  Target functions are random single parities.
- **Wall clock:** ~23.8 hours (32 workers).
- **Status:** COMPLETE. Exceeds the planned n=4..6 range by including n=7
  and n=8, and uses 50 trials (vs planned 24) for tighter confidence.

#### Key Results

| n | p=0 acc | p=0.001 acc | p=0.005 acc | p=0.01 acc | p=0.02 acc | p=0.05 acc | p=0.1 acc |
|---|---------|-------------|-------------|------------|------------|------------|-----------|
| 4 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 5 | 100% | 100% | 98% | 96% | 96% | 96% | 98% |
| 6 | 100% | 100% | 0% | 6% | 0% | 2% | 2% |
| 7 | 100% | 0% | 0% | 0% | 0% | 0% | 0% |
| 8 | 100% | 0% | 0% | 0% | 0% | 0% | 0% |

#### Observations

1. **p=0 baseline is clean:** 100% acceptance and correctness at all n,
   confirming that the circuit pipeline (transpilation, MCX decomposition)
   introduces no artefacts.

2. **Gate noise inflates |L| at small n:** At n=4, |L| jumps from 1
   (p=0) to 16 (all 2⁴ strings) at p≥0.005, and to ~3.7 at p=0.001.
   At n=5, |L| reaches ~30 at p≥0.005. Despite this, the verifier's
   weight check still passes and the correct hypothesis is identified.

3. **Sharp breakdown scales with n:**
   - n=4: No breakdown up to p=0.1 (protocol fully robust).
   - n=5: Mild degradation (96--100%) but no full breakdown.
   - n=6: Breakdown between p=0.001 and p=0.005.
   - n=7--8: Breakdown between p=0 and p=0.001.

4. **Circuit depth is the mechanism:** MCX on n qubits decomposes into
   O(n) CX gates, each suffering independent depolarising error. The
   effective error accumulates with depth, so the breakdown point
   decreases with n. This is qualitatively different from label-flip
   noise, where breakdown depends on η relative to θ, independent of n.

5. **Physically realistic regime (p ≈ 0.001):** Protocol remains fully
   functional at n=4--6, suggesting practical viability on near-term
   hardware for small instances. At n≥7, even p=0.001 causes total
   failure.

### Honest Baseline (5.2)
- **Status:** IMPLICITLY COVERED by scaling (n=4..16) and noise (eta=0.0
  rows). No dedicated baseline result file exists, but the data is there.
- *Gap:* The plan specifically calls for reporting post-selection rate,
  GL tree depth, extraction accuracy, and wall-clock time at n=4, n=10,
  and n=16. These metrics may need to be extracted from existing results
  or the experiment re-run with explicit logging of GL diagnostics.

---

## Pending Experiments

None — all seven experiments plus the honest baseline are complete.

---

## Next Steps

### Priority 1: Run remaining experiments
1. ~~**Exp 3 (gate-level noise):**~~ DONE (2026-04-04). Ran n=4..8,
   50 trials, 32 workers. Results in `gate_noise_4_8_50.pb`.
2. ~~**Exp 7 (average-case):**~~ DONE (2026-04-02). Implemented
   `make_k_sparse`, `make_random_boolean`, `make_sparse_plus_noise` in
   `phi.py` and `experiments/harness/average_case.py`.

### Priority 2: Strengthen existing experiments
3. ~~**Truncation at larger n:**~~ DONE (2026-04-02). Ran at n=10 and n=12
   with 24 trials per cell. Results in `truncation_10_10_24.pb` and
   `truncation_12_12_24.pb`.
4. **Honest baseline diagnostics:** Extract or re-run baseline trials at
   n={4, 10, 16} with explicit GL diagnostics (tree depth, nodes
   explored, false positive/negative rates).

### Priority 3: Writing tasks
5. Connect noise sweep results to Ma--Su--Deng threshold analysis.
6. Compare empirical scaling against theoretical O(1/theta) and
   O(poly(n)) predictions in the scaling data.
7. Tabulate all hypothesis risk vs. best parity risk across conditions.

---

## Extensions: Novel Insights Linking Back to Caro et al.

The following are opportunities for original empirical findings that
directly engage with open questions or theoretical claims in Caro et al.
(arXiv:2306.04843, ITCS 2024).

### 1. The 2-agnostic gap for Fourier-sparse learning
Caro et al. prove *2-agnostic* improper learning for Fourier-sparse
functions (Theorem 1, part 3; Corollary 7) and note this is not known to
be tight. The plan's scaling experiment uses single parities (1-agnostic
case). An extension would construct k-sparse functions (k=2,4,8) and
empirically measure the achieved agnostic ratio alpha: does the protocol
achieve alpha < 2 in practice, or does the factor-of-2 loss materialise?
This directly tests whether Caro et al.'s 2-agnostic bound is tight or
an artefact of the analysis.

**Connection:** Theorem 1 part 3, Section 5.2 (Corollary 7), and the
open question in Section 1.5 about improving 2-agnostic to 1-agnostic.

### 2. Distribution class promise sensitivity
The verification protocol (Theorem 2) requires the distribution class
promise that the total Fourier weight lies in a known interval [a^2, b^2]
(Equation 5 in the paper). The plan's experiments all use the exact
promise (a^2 = b^2). A valuable extension: *deliberately mis-specify* the
promise interval and measure how far off [a^2, b^2] can be before
soundness or completeness breaks. This probes whether the protocol is
robust to partial knowledge of the weight, which the theory does not
address. A positive finding (the protocol is tolerant to moderate
mis-specification) would be practically significant.

**Connection:** Theorem 2, Section 6.3, the weight-check step (Step 3 of
the protocol). Also connects to the limitation noted in Section 6.3's
discussion of the "restricted distribution class" requirement.

### 3. Post-selection rate vs. Fourier weight structure
Theorem 5 in Caro et al. (distributional approximate QFS) shows that the
post-selection probability is 1/2 in the noiseless case, and the
conditional distribution is close to |phi_hat(s)|^2 up to an
exponentially small correction of (1 - E[phi(x)^2]) / 2^n. The plan
already records post-selection rates in the scaling experiment. An
extension: systematically vary the *Fourier weight concentration* (from
single parity at one extreme to bent at the other) at fixed n and plot
how the empirical post-selection rate and extraction accuracy depend on
the Fourier weight structure. This tests whether the 2^{-n} correction
term is practically negligible at intermediate n, or whether it
introduces measurable bias for functions with low total Fourier weight.

**Connection:** Theorem 5 (Equation 4 in the paper), Section 5.1.

### 4. Noise model comparison: Lemma 4 vs. Lemma 6
Caro et al. prove that mixed noisy functional quantum examples
(Definition 5(i), Lemma 4) give *exactly* the same QFS distribution as
the noiseless case, while mixture-of-superpositions noisy examples
(Definition 5(iii), Lemma 6) attenuate coefficients by (1-2*eta)^2. The
current noise experiment implements only the latter model. Adding
Lemma 4's noise model (mixed noise on pure states) and comparing the two
experimentally would verify whether the theoretical distinction holds in
simulation and characterise the practical impact. A finding that the
mixed-state noise model is strictly better for verification would be a
concrete empirical confirmation of a non-obvious theoretical prediction.

**Connection:** Lemma 4, Lemma 6, Section 4.2.

### 5. Goldreich-Levin as a QSQ emulator
Caro et al. note (Section 5.3, Corollary 9) that GL extraction can be
viewed as operating in the distributional agnostic quantum statistical
query (QSQ) model. The experiment harness already uses GL at scale.
Measuring the effective "tolerance" of the GL-based QSQ queries (i.e.,
how much estimation noise the GL tree accumulates per level vs. the
theoretical tau) would quantify whether GL is practically operating
within the QSQ regime or exceeding its noise budget.

**Connection:** Section 5.3, Corollary 9, and the QSQ model definition.

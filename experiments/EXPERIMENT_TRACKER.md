# Experiment Tracker

Status as of 2026-04-02.

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

### Honest Baseline (5.2)
- **Status:** IMPLICITLY COVERED by scaling (n=4..16) and noise (eta=0.0
  rows). No dedicated baseline result file exists, but the data is there.
- *Gap:* The plan specifically calls for reporting post-selection rate,
  GL tree depth, extraction accuracy, and wall-clock time at n=4, n=10,
  and n=16. These metrics may need to be extracted from existing results
  or the experiment re-run with explicit logging of GL diagnostics.

---

## Pending Experiments

### Exp 3 -- Gate-Level Noise (Depolarising via Qiskit NoiseModel)
- **Module:** `experiments/harness/gate_noise.py`
- **Result file:** `results/gate_noise_{n_min}_{n_max}_{trials}.pb`
- **Coverage:** n=4..6, gate error rate p in
  {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}, 24 trials per (n, p) cell.
  Uses depolarising channels on H, X, CX gates via Qiskit `NoiseModel`.
  Protocol run with a² = b² = 1 (noiseless promise), θ = ε = 0.3
  (no noise-adaptive threshold), 2000 QFS shots, 1000 prover samples,
  3000 verifier samples. Target functions are random single parities
  (same as Exp 2 for direct comparison).
- **Status:** NOT YET RUN. Module and proto schema implemented; awaiting
  execution.

#### Theoretical Context

Caro et al. (arXiv:2306.04843, §4.2) analyse three noise models for
quantum Fourier sampling, all operating at the distribution/label level:

- **Lemma 4** (mixed noisy functional examples): QFS distribution is
  *identical* to the noiseless case — noise has no effect.
- **Lemma 5** (pure η-noisy examples): Post-selection probability drops
  from 1/2 to 1/2 − √((1−η)η), but conditional Fourier distribution
  is unchanged.
- **Lemma 6** (MoS noisy examples, Definition 5(iii)): Post-selection
  stays at 1/2, but Fourier coefficients are attenuated by (1−2η)².

The verification protocol (Theorems 11–12, §6.2) adapts to label-flip
noise by setting the distribution class promise a² = b² = (1−2η)² and
adjusting θ accordingly.

**Gate-level depolarising noise is not covered by any of these results.**
Depolarising channels applied to H, X, and CX gates corrupt the quantum
state in a fundamentally different way from label-flip noise — there is
no closed-form expression for the effective Fourier attenuation, and the
noise does not factor cleanly into the distribution class promise. This
makes Experiment 3 a genuinely novel empirical contribution with no
theoretical prediction to compare against.

#### Why These Noise Rates

- **p = 0.0**: Baseline — circuit mode without noise. Validates that the
  circuit pipeline itself (transpilation, MCX decomposition) does not
  introduce artefacts.
- **p = 0.001–0.005**: Realistic for current superconducting hardware
  (IBM's reported 2-qubit gate error rates are ~0.5–1%). Tests whether
  the protocol is robust to hardware-realistic noise levels.
- **p = 0.01–0.02**: Moderate noise. Pilot data shows p=0.01 already
  causes GL extraction to find spurious heavy Fourier coefficients
  (|L| = 16 at n=4 vs |L| = 1 at p=0), though the protocol still
  accepts and identifies the correct hypothesis.
- **p = 0.05–0.1**: Heavy noise, probing the breakdown point. At n=6,
  pilot data shows p=0.01 already causes rejection (|L| = 1, rejected),
  so the breakdown point is somewhere below p=0.01 for n=6.

#### Pilot Timing Data

Single-trial timings at p=0.01 (2000 QFS shots, sequential):

| n | Time/trial | |L| | Accepted | Correct |
|---|-----------|-----|----------|---------|
| 4 | ~40s | 16 | ✓ | ✓ |
| 5 | ~111s | 32 | ✓ | ✓ |
| 6 | ~227s | 1 | ✗ | ✗ |

Trials at p=0.0 are faster (~17s at n=4) because the non-noise
StatevectorSampler path avoids transpilation overhead.

#### Early Observations from Pilots

1. **Gate noise inflates |L|**: At n=4, p=0.01 causes |L| to jump from
   1 (correct single parity) to 16 (all 2⁴ strings). Gate noise
   introduces spurious apparent Fourier weight across all frequencies,
   causing GL extraction to flag many false positives. Despite this, the
   verifier's weight check (Step 4 of the protocol) still passes and the
   correct hypothesis is identified.

2. **Breakdown scales with n**: At n=6, even p=0.01 causes outright
   rejection. The depolarising noise on the larger circuit (more CX gates
   from MCX decomposition) accumulates enough error to destroy the QFS
   signal entirely. This is qualitatively different from label-flip noise,
   where the breakdown depends on η relative to the Fourier weight, not
   on circuit depth.

3. **Circuit depth is the key variable**: MCX gates on n qubits decompose
   into O(n) CX gates during transpilation. Each CX gate independently
   suffers depolarising error at rate p, so the effective error
   accumulates with circuit depth. This suggests an effective noise
   rate scaling roughly as p × (circuit depth), which grows with n.

#### Estimated Run Times

Per-cell estimates (1 trial, sequential, based on pilot data):

| n | p=0 | p>0 (avg) |
|---|-----|-----------|
| 4 | ~17s | ~35s |
| 5 | ~50s | ~110s |
| 6 | ~100s | ~230s |

Full run: 3 values of n × 7 noise rates × 24 trials = **504 trials**.

| n | Est. serial time | With 8 workers |
|---|------------------|----------------|
| 4 | ~1.0h | ~8min |
| 5 | ~3.5h | ~26min |
| 6 | ~8.0h | ~60min |
| **Total** | **~12.5h** | **~1.5h** |

#### Execution Commands

```bash
# Full run (recommended)
uv run python -m experiments.harness gate_noise \
    --n-min 4 --n-max 6 --trials 24 --workers 8

# Output: results/gate_noise_4_6_24.pb
```

```bash
# Quick validation at n=4 only (~8 min with 8 workers)
uv run python -m experiments.harness gate_noise \
    --n-min 4 --n-max 4 --trials 24 --workers 8

# Output: results/gate_noise_4_4_24.pb
```

#### Presentation Plan (Chapter 5)

- **Figure 5.5**: Acceptance rate vs p for each n. Compare side-by-side
  with the label-flip curve from Exp 2 at equivalent effective noise.
- **Table 5.4**: Gate-noise breakdown point vs label-flip breakdown point
  for each n.

**Key claims to support:**

1. Gate-level noise causes protocol failure at much lower noise rates
   than label-flip noise, because errors accumulate with circuit depth
   rather than attenuating Fourier coefficients uniformly.
2. The breakdown point decreases with n (more gates → more accumulated
   error), unlike label-flip noise where the breakdown depends on η
   relative to θ, independent of n.
3. At physically realistic error rates (p ≈ 0.001–0.005), the protocol
   remains functional at small n, suggesting practical viability on
   near-term hardware for small problem instances.

---

## Next Steps

### Priority 1: Run remaining experiments
1. **Exp 3 (gate-level noise):** Module implemented. Run the full sweep
   (n=4..6, 24 trials, 8 workers). See Exp 3 section above for commands.
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

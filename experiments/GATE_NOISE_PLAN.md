# Experiment 3: Gate-Level Depolarising Noise — Execution Plan

## Theoretical Context

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

## Experiment Design

### What We Measure

The protocol is run with `a² = b² = 1` (noiseless promise) and
`θ = ε` (no noise-adaptive threshold), since gate noise does not
attenuate Fourier coefficients in the analytically tractable way that
label-flip noise does. The key question is: **at what gate error rate p
does the QFS circuit degrade enough to cause verification failure?**

This directly contrasts with Experiment 2 (label-flip noise), where
the breakdown point is theoretically predicted by the Ma–Su–Deng
threshold η ≤ 1/(10θ) and the attenuation factor (1−2η)².

### Parameters

| Parameter | Value | Justification |
|---|---|---|
| n | {4, 5, 6} | Small-n regime (tracker §5.3). Circuit simulation cost grows exponentially; n=6 is practical upper limit for circuit mode. |
| p (gate error rate) | {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1} | Spans from no noise to substantial noise. p=0.001–0.01 is the physically realistic regime for near-term hardware; p=0.05–0.1 probes deep into failure. |
| ε | 0.3 | Consistent with all other experiments. |
| θ | 0.3 (= ε) | No noise adaptation — gate noise has no theory to guide θ adjustment. |
| a², b² | 1.0 | Noiseless promise. Gate noise is purely at the circuit level, not the distribution level. |
| Trials per (n, p) cell | 24 | Matches Exps 2, 4, 5, 6, 7. Gives Wilson 95% CI width of ~±20% at 50% acceptance. |
| QFS shots | 2000 | Standard across experiments. |
| Classical samples (prover) | 1000 | Standard. |
| Classical samples (verifier) | 3000 | Standard. |
| Target functions | Random single parities | Same as Exp 2 (noise sweep), enabling direct comparison. |

### Why These Noise Rates

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

## Pilot Timing Data

Single-trial timings at p=0.01 (2000 QFS shots, sequential):

| n | Time/trial | |L| | Accepted | Correct |
|---|-----------|-----|----------|---------|
| 4 | ~40s | 16 | ✓ | ✓ |
| 5 | ~111s | 32 | ✓ | ✓ |
| 6 | ~227s | 1 | ✗ | ✗ |

Trials at p=0.0 are faster (~17s at n=4) because the non-noise
StatevectorSampler path avoids transpilation overhead.

### Early Observations from Pilots

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

## Estimated Run Times

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

Worker parallelism applies across trials (each trial is an independent
process). Qiskit AerSimulator circuit simulation within a trial is
single-threaded.

## Execution Commands

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

## Presentation Plan (Chapter 5)

From PRESENTATION_PLAN.md, the gate noise results should produce:

- **Figure 5.5**: Acceptance rate vs p for each n. Compare side-by-side
  with the label-flip curve from Exp 2 at equivalent effective noise.
- **Table 5.4**: Gate-noise breakdown point vs label-flip breakdown point
  for each n.

### Key Claims to Support

1. Gate-level noise causes protocol failure at much lower noise rates
   than label-flip noise, because errors accumulate with circuit depth
   rather than attenuating Fourier coefficients uniformly.
2. The breakdown point decreases with n (more gates → more accumulated
   error), unlike label-flip noise where the breakdown depends on η
   relative to θ, independent of n.
3. At physically realistic error rates (p ≈ 0.001–0.005), the protocol
   remains functional at small n, suggesting practical viability on
   near-term hardware for small problem instances.

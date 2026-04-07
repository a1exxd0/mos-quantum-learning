# Audit: gate_noise experiment

## Paper sections consulted
Intro (pp. 1-5); Future Work 1.5 (pp. 11-12); 4.2 Noisy Functional QL with Lemmas 4-6 (pp. 22-24); 5.1 Theorem 5 (pp. 26-28); 6.1 Theorems 7-10 (pp. 35-42); 6.2 Definition 12 (pp. 43-44); 6.3 Definitions 13-14 + Theorems 11-13 (pp. 44-47).

## What the paper predicts
The paper considers exactly **one** noise model: classical label-flip noise on the data oracle (Definition 12, p. 43): `phi = eta + (1 - 2*eta) f`, each label flipped i.i.d. with prob `eta < 1/2`. Lemmas 4-6 (pp. 23-24) describe how this shifts the QFS distribution by `(1 - 2*eta)^2` (pure case) or by a poly-small extra perturbation (mixture case). §6.2 promotes these to a verification protocol assuming `eta` known. The paper does NOT discuss per-gate depolarising/Pauli/dephasing/amplitude-damping noise, threshold theorems, fault tolerance, or NISQ implementations of QFS. The only hardware-noise mention is §1.5 future work (p. 12) on NISQ-friendly variational protocols -- a different research direction. **No theoretical prediction exists for the gate_noise experiment; it is exploratory.**

## What the experiment does
`gate_noise.py:14-144` sweeps `n` and a per-gate depolarising rate `p`, runs the honest protocol on a random parity for each `(n, p, trial)` cell. Defaults: `n_range=range(4,7)` (line 15), 12 rates `[0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]` (line 78), 24 trials (line 17), `epsilon=0.3`, `theta=epsilon`, `qfs_shots=2000`, `classical_samples_prover=1000`, `classical_samples_verifier=3000` (lines 18-22), `a_sq=b_sq=1.0`, `noise_rate=0.0` (lines 96-103) -- distribution is noiseless, only the prover's circuit is noisy. `qfs_mode="circuit"` is forced (line 110). The on-disk artefact `results/gate_noise_4_8_50.pb` was run at `n in [4,8]` with 50 trials/cell.

Noise injection (`worker.py:133-144`):
```
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(p,1), ["h","x"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(p,2), ["cx"])
```
Handed only to `MoSProver`/`QuantumFourierSampler` (`worker.py:148`), consumed by `AerSimulator(noise_model=...)` in `mos/sampler.py:387-406`. Each circuit transpiled to AerSimulator basis -- multi-controlled X gates from `_circuit_oracle_f` (`mos/__init__.py:230-269`) decompose into many CX + 1q gates. Verifier (`worker.py:160-184`) runs without the noise model and consumes classical samples directly from `MoSState`, which is unperturbed. Noise is correctly single-sided (prover-only).

**Noise model summary:** symmetric depolarising channel of rate `p` after every `h`, `x` (1q) and `cx` (2q) gate. No noise on barriers, measurements, idle qubits, or other basis gates the transpiler may emit (`sx`, `rz`, `u3`, `t`, ...).

## Implementation correctness
**Right:**
1. `depolarizing_error(p,k)` is the correct Qiskit API.
2. Noise model attached only to the prover side (`worker.py:148`).
3. `qfs_mode="circuit"` forced; `mos/sampler.py:387` is the only path that respects the noise model.
4. `noise_rate=0.0`, `a_sq=b_sq=1.0` keeps the functional-case promise consistent with a noise-free distribution.
5. Each trial reseeds from a master rng (`gate_noise.py:90`); AerSimulator gets a per-circuit `seed_simulator` (`mos/sampler.py:397-401`).
6. No off-by-one in `range(args.n_min, args.n_max+1)` (`__main__.py:181`).

**Concerns:**
1. **MAJOR -- Domain truncation undocumented.** On-disk file is `n in [4,8]`, 50 trials/cell, vs sibling experiments at `n in [4,16]` and 100 trials. `gate_noise.py` defaults are smaller still (`range(4,7)`, 24 trials). `mos/sampler.py:367-372` warns circuit mode "will be very slow (up to 2^n multi-controlled gates per copy)" for `n>12`; with `qfs_shots=2000` plus AerSimulator + noise, going to `n=16` would multiply work by ~256x and require simulating a 17-qubit noisy circuit. Almost certainly the practical reason, but **not flagged anywhere**.

2. **MAJOR -- The reported "threshold" is dominated by truth-table oracle synthesis, not the protocol.** `_circuit_oracle_f` (`mos/__init__.py:230-269`) emits up to `2^n` MCX gates, each transpiling into `O(n)` CX gates -- circuit depth exponential in `n`, expected errors per shot `~O(p * n * 2^n)`. The paper's complexity bounds (Theorem 12, p. 45) give the prover `O(n log(...))` *single-qubit* gates total, so the experiment is hitting an exponential blow-up that does not exist in the theory. The figure caption's "the protocol transitions sharply" should be reframed as a property of the simulator, not the protocol.

3. **MAJOR -- Sharp drops at n=7,8 from p=1e-4 to p=5e-4 are suspiciously sharp.** From `gate_noise_summary.csv`:
   - n=7, p=0.0001 -> 100%, p=0.0005 -> 2%
   - n=8, p=0.0001 -> 100%, p=0.0005 -> 0%
   A 98-100 point drop across half a decade in `p` is much sharper than compounding depolarising noise predicts. AerSimulator seeds derive from the trial rng (`mos/sampler.py:397`), so the cliff might partly reflect a small number of effective seeds. Should be verified with independent reruns and >=100 trials before being interpreted as a physical threshold.

4. **MAJOR -- p > 0.1 is unphysical.** Real superconducting/trapped-ion/photonic gates have 2q error rates ~1e-4 to ~1e-2. At p=0.5 the depolarising channel maps any state to within 1/4 of the maximally mixed state in one application -- this measures verifier behaviour against random garbage, not hardware. The mild recovery at n=5 for large p (98%, 92%, 94%, 98%) is consistent: at very high noise the prover's "list" is essentially random, and for small n a random `s` may still accidentally clear the slack `epsilon^2/8 = 0.01125` threshold.

5. **MINOR -- Noise model only covers `h, x, cx`.** After Aer transpilation other basis gates (`sx`, `rz`, `u3`, `t`, `tdg`) may carry zero error, so the effective per-logical-op error is smaller than nominal `p`. The `p` axis is therefore only loosely physical.

6. **MINOR -- `epsilon = 0.3` is unjustified.** Acceptance threshold is `1 - 0.01125`, very slack. Larger epsilon makes the protocol *more* tolerant of noise, masking the true robustness budget. Internally consistent with sibling experiments but not optimal for this study.

7. **MINOR -- DKW shot count and Parseval list bound (`prover.py:316-321`, `prover.py:490-495`) inherit from the noiseless analysis (Corollary 5/Theorem 8); they have no theoretical justification under gate noise.** Worth noting in the docstring.

8. **NIT -- Substring matching `f"p={p}" in t.phi_description`** (`gate_noise.py:138`) is fragile against future sweep edits; current sweep values are safe (no value is a prefix of another up to `_`).

9. **NIT -- Hadamard noise applied to both state-prep H's and QFS H's; experiment doesn't separate these.**

10. **NIT -- The docstring (`gate_noise.py:42-44`) acknowledges "this goes beyond the noise models analysed in Caro et al."** -- good. The figure caption says the same. But this caveat is buried.

## Results vs. literature
No literature value to compare against. CSV data:

| n | first p with acc<50% | acceptance just before / just after |
|---|---|---|
| 4 | none | always 100% across full sweep |
| 5 | none (min 90%) | near-flat 98-100% throughout |
| 6 | 0.002 | 100% -> 8% |
| 7 | 0.0005 | 100% -> 2% |
| 8 | 0.0005 | 100% -> 0% |

The plot script's printed analysis (`plot_gate_noise.py:288-295`) -- "sharp threshold, larger n more sensitive, qualitatively worse than label-flip" -- matches the CSV but is unsupported by theory. The exponential `n`-sensitivity is a quirk of the truth-table oracle synthesis, not the verification protocol.

## Issues / discrepancies
- **MAJOR** -- "Threshold" is dominated by truth-table oracle synthesis, not the protocol.
- **MAJOR** -- Domain truncated to `n in [4,8]`, 50 trials, with no in-file justification; inconsistent with sibling experiments.
- **MAJOR** -- High-`p` end of sweep (`{0.2, 0.5}`) is unphysical and uninformative.
- **MAJOR** -- Sharp cliffs at n=7,8 between p=1e-4 and p=5e-4 need verification with independent seeds and more trials.
- **MINOR** -- Noise model only covers `h, x, cx`; transpiled basis includes uncovered gates.
- **MINOR** -- `epsilon=0.3` gives a slack acceptance threshold.
- **NIT** -- Substring matching in per-`p` aggregation is fragile.
- **NIT** -- DKW/Parseval bounds inherited from noiseless analysis with no theoretical justification under gate noise.

**No BLOCKER issues.** The experiment runs, noise is applied to the right side, results are internally consistent.

## Verdict
**Exploratory, methodologically sound at the level of "we wired Aer's depolarising channel into the QFS circuit", but the headline finding is largely an artefact of a truth-table oracle synthesis that the paper itself does not require, run on too small a domain with too few trials, with a sweep range that extends well past physical relevance.** The conclusion "the verification protocol fails sharply under gate noise" is **NOT supported** without isolating circuit-implementation cost from protocol robustness.

**Recommended actions:**
1. Restrict `p` to `[1e-5, 1e-2]`.
2. >=100 trials with independently re-seeded reruns to verify the n=7,8 cliff at p~1e-4.
3. Add a comment explaining the `n<=8` ceiling, or extend the sweep to higher `n` on GPU.
4. Replace `_circuit_oracle_f` with a structured parity oracle (multi-controlled Z conjugated by Hadamards is `O(n)` gates, matching the paper) so the experiment actually tests the protocol's robustness instead of truth-table synthesis cost.
5. Add a figure-caption sentence stating the paper makes no prediction here and that `p` is per-gate depolarising on `h, x, cx` only.
6. Optional: add a unit test in `tests/` that runs noiseless + `p=1e-3` trials at `n=4` and asserts qualitative behaviour.

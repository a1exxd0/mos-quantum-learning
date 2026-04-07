# Audit summary: results/ vs. Caro et al. paper

This directory contains independent per-experiment audits of every result in `results/` and `results/figures/`, performed by 11 parallel agents that each read `papers/classical_verification_of_quantum_learning.pdf`, the experiment harness, the protocol code under `ql/` and `mos/`, the on-disk `.pb` results, and the plot scripts. This file is the cross-experiment synthesis.

## Audit files

| Experiment | File | Verdict |
|---|---|---|
| scaling | [scaling.md](scaling.md) | PASS impl, MAJOR figure framing |
| bent | [bent.md](bent.md) | PASS |
| truncation | [truncation.md](truncation.md) | PASS impl, MAJOR experiment design |
| soundness | [soundness.md](soundness.md) | PASS |
| soundness_multi | [soundness_multi.md](soundness_multi.md) | PASS impl, MAJOR sample-budget |
| average_case | [average_case.md](average_case.md) | PASS impl, MAJOR wrong dispatch path |
| k_sparse | [k_sparse.md](k_sparse.md) | PASS impl, MAJOR off-promise |
| noise_sweep | [noise_sweep.md](noise_sweep.md) | PASS impl, MAJOR sweep range |
| gate_noise | [gate_noise.md](gate_noise.md) | exploratory, MAJOR off-paper |
| theta_sensitivity | [theta_sensitivity.md](theta_sensitivity.md) | PASS impl, MAJOR fixed budgets |
| ab_regime | [ab_regime.md](ab_regime.md) | PASS impl, MAJOR figure interpretation |

**No BLOCKER issues in any experiment.** Every audit is "implementation correct, experimental design or framing has issues".

## What is correct (do not touch)

- **`ql/verifier.py`** — faithful line-by-line implementation of Theorems 12 and 15:
  - list-size bound `64 b²/θ²` (`verifier.py:458`)
  - parity acceptance threshold `a² − ε²/8` (`verifier.py:509`)
  - k-sparse acceptance threshold `a² − ε²/(128 k²)` (`verifier.py:512`)
  - per-coefficient Hoeffding tolerances `ε²/(16|L|)` (parity) and `ε²/(256 k²|L|)` (k-sparse) (`verifier.py:482, 485`)
  - independent classical estimation via Lemma 1 (`verifier.py:553-611`)
  - Hoeffding sample count auto-derivation (`verifier.py:487-497`)
- **`ql/prover.py`** — Corollary 5 extraction (`prover.py:316, 428`), Parseval list cap `16/θ²` (`prover.py:490`).
- **`mos/__init__.py`, `mos/sampler.py`** — MoS state, label-flip noise, statevector and circuit QFS modes. Empirical post-selection rate 0.498-0.502 across all experiments matches Theorem 5's 1/2 to sub-percent precision.
- **`tests/`** — unit tests pin down the right algebraic invariants (e.g. `mos_state_test.py:189-204` locks in Lemma 6 attenuation). Do not rewrite.

## Two recurring MAJOR themes

### Theme A — Hard-coded sample budgets bypass the paper's analytic formulas

Most experiments fix `qfs_shots=2000`, `classical_samples_prover=1000`, `classical_samples_verifier=3000`. These constants override the Hoeffding-derived budgets in `verifier.py:487-497` and `prover.py:316-321`. For ε=0.3, δ=0.1 the analytic prover budget is ~145k QFS shots; the analytic verifier budget at |L|=10 is ~38M classical samples. The hard-coded values are 3-5 orders of magnitude below.

**Where it matters:**
- **scaling** (M1 in scaling.md): `resource_scaling.{pdf,png}` overlays a fitted theoretical scaling curve through a y-axis that is constant 6000 across all n. The figure is the headline visual artifact and is genuinely misleading.
- **truncation** (M1, M2): the entire grid is sub-Hoeffding by 3-4 orders. The "knees" are squaring-bias artefacts, not Theorem 12 boundary measurements. Some `accept@50 > accept@3000` rows appear.
- **theta_sensitivity** (M1): maps the accept/reject boundary correctly but does NOT validate the `1/θ⁴` sample-complexity scaling.
- **soundness_multi** (M1): `subset_plus_noise` falsely accepts at up to 18% at k=2, exceeding the stated δ=0.1.
- **bent** (m3), **k_sparse** (m5), **average_case** (m5), **soundness** (m2) — same shortcut, but the signal is overwhelming for those experiments so the result is operationally fine.

**Where it does NOT matter:** scaling, soundness, bent — the underlying signal-to-noise ratio is so high that 3000 samples is operationally sufficient. The framing is wrong but the data is fine.

### Theme B — Experiments operate outside the paper's promise classes

Several experiments run on distributions that violate Definition 11 / Definition 14 / the precondition `ε ≥ 2√(b²−a²)` of Theorems 12/15. The verifier *correctly* either rejects or fails to certify these instances; the experiments then interpret this as "average-case failure" or "Theorem 13 looseness", which misframes the result.

**Where it matters:**
- **average_case** (M1, M2): the worst offender. `k_sparse_2`, `k_sparse_4`, `sparse_plus_noise` all set `spec.k = None`, so the worker dispatches to `verify_parity` instead of `verify_fourier_sparse`. Theorems 9/10/15 prescribe the latter, with threshold `a² − ε²/(32 k²)` and Lemma-14 randomized hypothesis. Plus `random_boolean` violates Definition 11 at n≥6 by construction.
- **k_sparse** (M1): `make_k_sparse` uses Dirichlet(1,..,1) which routinely produces `c_min < θ`, violating Definition 11. Acceptance rate ceiling is structural, not protocol failure.
- **ab_regime** (M1): `ε ≥ 2√(b²−a²)` is satisfied only for `gap ≤ 0.0225`, i.e. just `gap=0.0` of the 6 swept gaps. The plot script (`plot_ab_regime.py:472-478`) misinterprets the resulting honest-input acceptance as "Theorem 13 is loose".
- **gate_noise** (M1): the paper has zero predictions about per-gate depolarizing noise. The "threshold" the experiment measures is dominated by exponential truth-table oracle synthesis cost in `_circuit_oracle_f`, not protocol robustness.

**Where it does NOT matter:** the paper-aligned distributions in scaling, soundness, bent, noise_sweep all satisfy their respective promises and behave as predicted.

## Triaged fix list

### Tier 1 — documentation-only (no rerun, ~half a day)

The data is fine, only the figure captions and plot-script narratives need updating.

| Fix | File | Action |
|---|---|---|
| Scaling resource figure | `results/figures/scaling/plot_scaling.py` | Drop the fitted theoretical-scaling overlay or relabel as "fixed-budget feasibility, not measured scaling". |
| ab_regime interpretation | `results/figures/ab_regime/plot_ab_regime.py:472-478` | Replace "Thm 13 is loose" with "outside Thm 12 completeness regime; honest interactions still pass on benign inputs". |
| k_sparse legend | `results/figures/k_sparse/plot_k_sparse.py` | Change "1−δ=0.9 (Thm 9)" → cite Theorem 10/15; change `4/θ²` reference line to the actually-enforced `64 b²/θ²`. |
| gate_noise caption | `results/figures/gate_noise/plot_gate_noise.py` | Add: "the paper makes no prediction about gate noise; this experiment is exploratory; `p` is per-gate depolarizing on `h, x, cx` only". |
| Soundness bad-accept column | `results/figures/soundness/plot_soundness.py` | Add `bad_accept_rate = Pr[accepted ∧ ¬hypothesis_correct]`; data already in `worker.py:438-439`. No rerun needed. |
| theta_sensitivity legend | `results/figures/theta_sensitivity/plot_theta_sensitivity.py:184` | Relabel `4/θ²` as "Theorem 7/9 SQ-verifier bound" or use the actually-enforced `64 b²/θ²`. |
| soundness_multi plot comment | `results/figures/soundness_multi/plot_soundness_multi.py:87` | Wrong: `k` field IS populated (`results.py:374-375`). Read `t["k"]` directly. |
| Docstring updates | `experiments/harness/{scaling,truncation,theta_sensitivity,soundness_multi}.py` | Add a paragraph each acknowledging the hard-coded-budget framing. |

### Tier 2 — small harness fix + rerun at same scale (1-2 lines + a few hours of compute)

Highest-value fixes per unit of effort.

| Fix | File | Action |
|---|---|---|
| **average_case wrong dispatch** | `experiments/harness/average_case.py` | Pass `k=k` into `TrialSpec` for `k_sparse_2`, `k_sparse_4`, `sparse_plus_noise` so the worker dispatches to `verify_fourier_sparse`. Drop or rename `random_boolean`. **Single highest-value fix in the audit.** |
| soundness_multi under-sampling | `experiments/harness/soundness_multi.py:112` | Bump `classical_samples_verifier` to ~30k OR widen ε so `subset_plus_noise` is unambiguously rejected. Trim redundant strategies (`shifted_coefficients` is structurally trivial; `partial_real` overlaps `diluted_list`). |
| noise_sweep range | `experiments/harness/noise.py:79` | Extend η to {0.42, 0.44, 0.46, 0.48} so theoretical breakdown η ≈ 0.447 is visible. |
| noise_sweep adaptive-θ confound | `experiments/harness/noise.py:92` | Hold θ fixed across η sweep (don't vary two parameters along one axis). |
| diluted_list bug | `experiments/harness/worker.py:291` | `n_keep = max(1, len/4)` is degenerate (=1) for k ≤ 4. Either fix the formula or restrict `soundness_multi` to k ≥ 8 for this strategy. |

### Tier 3 — larger reruns (cluster jobs)

These actually validate the paper's scaling claims rather than just the boundary positions.

| Fix | Action |
|---|---|
| **scaling, theta_sensitivity, k_sparse** | Re-run with `qfs_shots=None`, `classical_samples_prover=None`, `classical_samples_verifier=None` so the analytic formulas in `verifier.py:487-497` and `prover.py:316-321` drive per-trial budgets. Validates n-, θ-, k-independence claims of Theorems 12 and 15. Probably tractable on DCS for n ≤ 12; n ≥ 13 may require constant-factor surgery. |
| truncation grid | Either extend `m_V` upward to cross the analytic Hoeffding count, or extend ε downward so the existing grid crosses the boundary. Split the heatmap into prover-success-rate and conditional verifier-accept-rate panels. |
| truncation single-instance | Randomise `target_s` per trial instead of hard-coding `target_s=1` (`truncation.py:87`). |

### Tier 4 — leave as-is, document the limitation

| Item | Action |
|---|---|
| gate_noise | Either restrict `p` to physical `[1e-5, 1e-2]` and re-run, or accept it as exploratory and explain in the writeup. Don't try to make it match a theorem that doesn't exist. |
| bent at n=4 | Note that this point is in the "uncertain band" `[θ/2, θ)` of Corollary 5, not strictly "below the crossover". |
| Cheating-strategy menu | The paper proves soundness against arbitrary P' (Definition 7); no enumerated strategy menu is required. The strategy menus in `soundness` and `soundness_multi` are engineering choices, not paper claims. |

## Cross-cutting things that are NOT broken

- **Verifier and prover protocol code.** No edits needed.
- **MoS state and noise injection.** Empirical post-selection rate matches Theorem 5's 1/2; Fourier-weight attenuation matches Lemma 6 exactly.
- **Tests.** Existing unit tests pin down the right invariants. Don't rewrite.
- **Reproducibility.** All experiments use seeded RNGs, per-trial seeds spawned from a base seed, and `MoSState` reconstruction in workers. No flakiness reported.
- **`bent`, `noise_sweep` (algebraic part), `soundness`, `scaling` (numerics)** — these pass cleanly. Only their *framing* in figures/captions has issues.

## Suggested order of operations

If under time pressure for the dissertation/presentation, **Tier 1 + Tier 2 alone** gets the project to a defensible position, because:

1. The protocol implementation is already correct (Theorems 12 and 15 implemented line-by-line).
2. The misleading figures are mostly fixable without rerunning anything.
3. The single biggest narrative win (`average_case` dispatch fix) is one line.

Recommended sequence:
1. **Tier 2 / average_case** — one-line fix, biggest narrative improvement, modest rerun (4 families × 13 n × 100 trials).
2. **All Tier 1** doc-only fixes in one sitting.
3. **Tier 2 / soundness_multi** + **noise_sweep** small reruns.
4. Submit **Tier 3** reruns to the DCS cluster as background jobs.
5. Treat **Tier 4** as writeup notes.

After Tier 1+2, every figure either (a) reflects what the paper predicts and the experiment measures, or (b) carries an honest caveat about why it doesn't. After Tier 3, the experiment suite would also empirically validate the paper's asymptotic scaling claims, not just their boundary positions.

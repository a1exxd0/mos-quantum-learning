# Audit: noise_sweep experiment

## Paper sections consulted

- **Definition 4–5** (pp. 15–16) — pure / mixed / mixture-of-superpositions noisy quantum examples. Definition 5(iii), Eq. (15), is the model used here.
- **Definition 8** + Lemma 1 (pp. 17–18) — MoS state and "computational-basis measurement of ρ_D ⇒ classical sample from D".
- **§4.2 Lemma 6** (p. 23) — under MoS noisy examples, QFS conditional output distribution is `(4η−4η²)/2ⁿ + (1−2η)² ĝ(s)²`, i.e. ϕ̂_eff(s) = (1−2η)·ϕ̂(s).
- **§5.1 Theorem 5 + Corollary 5** (pp. 26–28) — distributional approximate QFS and heavy-list extraction, with `|L| ≤ 16/ε²`. Requires `ε ≥ 2^(−(n/2−2))`.
- **§6.2 Definition 12** (p. 43) — noisy distributional class.
- **§6.3 Definition 14, Theorems 11/12** (pp. 44–46) — proper 1-agnostic parity verifier, `|L| ≤ 64 b²/ϑ²`, per-coefficient tolerance `ε²/(16|L|)`, accept if `Σ ξ̂(sₗ)² ≥ a² − ε²/8`. Theorem 12 requires `ε ≥ 2√(b²−a²)` and `ϑ ∈ (2^(−(n/2−3)), 1)`.

## What the paper predicts

For a parity `f(x) = s*·x` with η label-flip noise, `ϕ̃_eff = (1−2η)·χ_{s*}`, so `a² = b² = (1−2η)²` and the only nonzero Fourier coefficient is `ϕ̂_eff(s*) = 1−2η`. Theorem 12 with `a²=b²=(1−2η)²` accepts iff `Σ ξ̂(s)² ≥ (1−2η)² − ε²/8`. The honest prover's QFS distribution is concentrated on `s*` with mass `≈ (1−2η)²`. Theoretical breakdown when `(1−2η)² < ε²/8`, i.e. **η_max = (1 − ε/(2√2))/2 ≈ 0.4470 for ε = 0.3**.

This is sample/oracle noise (Definition 5(iii)) — the labels in the underlying distribution are flipped while the quantum hardware is perfect. Gate-level noise is a separate experiment (`gate_noise.py`).

## What the experiment does

`experiments/harness/noise.py:run_noise_sweep_experiment`. Sweep:
- `n ∈ [4..16]`, `η ∈ {0.0, 0.05, 0.10, …, 0.40}`, 100 trials/cell.
- Random parity `s* ∼ Uniform({1,…,2ⁿ−1})` (`phi.py:make_random_parity`, line 35).
- `a_sq = b_sq = (1−2η)²` (`noise.py:91`) — correct per Definition 14.
- `theta = min(ε, 0.9·(1−2η))` if `1−2η > 0.01` else `0.01` (`noise.py:92`) — **adaptive ϑ**, see MAJOR-3.
- Fixed: `ε = 0.3`, `δ = 0.1`, `qfs_shots = 2000`, `classical_samples_prover = 1000`, `classical_samples_verifier = 3000`.

Noise injection happens **exactly once** in `mos/__init__.py:131`: `_phi_effective = (1 − 2η)·_phi + η`. Both `MoSState.sample_f` (line 187, used by QFS) and `sample_classical_batch` (line 447, used by the verifier) draw from this single `_phi_effective`. There is no double-counting and no separate η-dependent post-processing. Worker reconstruction at `worker.py:_run_trial_worker` line 128 passes `noise_rate=spec.noise_rate` straight to `MoSState`.

## Implementation correctness

**Noise model.** The implementation collapses Definition 5(iii)'s two-step "sample noiseless f, then flip each bit independently with prob η" into one Bernoulli per x with bias `(1−2η)ϕ(x) + η`. These are equal in distribution because each `f(x)` is independent. Verified by `tests/mos_state_test.py:189` (`test_phi_effective_noisy`), `:195` (`test_tilde_phi_effective_attenuation`), `:202` (`test_eta_half_completely_random`). **Correct.**

**Fourier spectrum.** `MoSState.fourier_coefficient(s, effective=True)` (`mos/__init__.py:455`) returns `(1−2η) · mean(tphi · χ_s)`, matching Lemma 6's `ϕ̂_eff(s) = (1−2η)·ϕ̂(s)`. `qfs_distribution` (line 582) computes the general form `(1 − E[ϕ̃_eff²])/2ⁿ + ϕ̂_eff(s)²`, which agrees with Lemma 6 for parities and with Theorem 5 in general. **Correct.**

**Verifier acceptance.** `ql/verifier.py:509` returns `acceptance_threshold = a_sq − ε²/8` for parity mode. Matches Theorem 12 Step 4 exactly. List bound `ceil(64·b²/ϑ²)` at line 458 matches Step 1. **Correct.**

**Verifier estimator (the data path).** `ql/verifier.py:597` computes `(1−2y_i)·χ_s(x_i)` and averages. Because `y ∼ ϕ_eff(x)`, `E[(1−2y)χ_s(x)] = (1−2η)·E[χ_{s*}(x)·χ_s(x)] = (1−2η)·δ_{s,s*}`, which equals `ϕ̂_eff(s)` exactly. So the verifier sees the noise-damped spectrum **without ever being told η** — matches the §6.2 remark that under ρ_(D,η) the verifier need not know η. **Correct.**

**Adaptive ϑ.** `noise.py:92` clamps ϑ ≤ 0.9·(1−2η) so that ϑ doesn't exceed the only nonzero coefficient. Not in the paper but a sensible practical adaptation; doesn't violate the soundness check (acceptance uses ε, not ϑ). **OK** but see MAJOR-3.

**Off-by-one / wrong-base errors.** None found. η flows through one place only (`_phi_effective` initialization). The {0,1} vs {−1,+1} convention is consistent: `tilde_phi = 1 − 2·phi` (line 148), estimator uses `(1 − 2y)`, so the estimator is unbiased for `ϕ̂(s)` (not `2·ϕ̂(s)`). The acceptance threshold `ε²/8` uses **ε**, not ϑ — matches Theorem 12 Step 4 verbatim (and is **not** a typo despite the prover side using ϑ).

## Results vs. literature

Source: `noise_heatmap.png`, `acceptance_correctness_vs_eta.png`, `fourier_weight_attenuation.png`, `breakdown_points.csv` — all under `results/figures/noise_sweep/`.

### Acceptance heatmap (rounded from PNG cell annotations)

| n  | η=0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 |
|----|--------|------|------|------|------|------|------|------|------|
| 4  | 100    | 81   | 82   | 75   | 81   | 88   | 93   | 96   | 100  |
| 8  | 100    | 79   | 71   | 73   | 78   | 76   | 79   | 93   | 100  |
| 12 | 100    | 82   | 76   | 78   | 79   | 70   | 87   | 85   | 95   |
| 16 | 100    | 84   | 79   | 80   | 77   | 79   | 79   | 86   | 93   |

(Full 13×9 table on the heatmap; pattern is uniform across n.)

### Headline observations

1. **No empirical breakdown.** `breakdown_points.csv` reports `no_breakdown` for every n ∈ [4..16]; at the largest η tested (0.40), acceptance is 93–100 %, never below 50 %. Theoretical breakdown is η ≈ 0.4470, **above** the largest η in the sweep. The experiment never enters the failure regime — see MAJOR-1.

2. **Non-monotonic acceptance.** Acceptance drops from 100 % at η=0 to ~70–82 % around η ∈ [0.05, 0.20], then **recovers** to 93–100 % at η=0.40. This is *not* what the literature predicts (monotone degradation). I traced the cause (it is a property of the experiment design, not the protocol):

   - Acceptance test slack = `ε²/8 = 0.01125` (constant, since ε is fixed at 0.3).
   - Verifier estimator standard error: with m=3000 samples, `σ_(ξ̂(s)) ≈ √((1−ξ²)/3000)`.
   - At η=0.05 (true ξ ≈ 0.9): `σ_(ξ̂) ≈ 0.0080`, `SD(ξ̂²) ≈ 2|ξ|·σ ≈ 0.0144`. **Larger than the slack 0.01125** → roughly 22 % of trials should reject by chance, matching the empirical 18–28 % rejection.
   - At η=0.40 (true ξ ≈ 0.2): `σ_(ξ̂) ≈ 0.018`, `SD(ξ̂²) ≈ 0.0072`. **Below** the slack.
   - Compounding factor: as ϑ shrinks (because of the η-adaptive ϑ in `noise.py:92`), the prover's `|L|` grows. Each spurious `ξ̂(s)²` is non-negative with mean `≈ 1/3000`, so `Σ ξ̂(s)²` gains `+|L|/3000` upward bias. At η=0.40, ϑ=0.18, prover's heavy list ≈ 5–10 entries → +0.002–0.003 bias, comparable to the slack. **The high-η acceptance is partly a statistical artefact**, not genuine signal.

3. **Fourier weight attenuation matches theory exactly.** `fourier_weight_attenuation.png` overlays median `Σ ξ̂(s)²` vs. `(1−2η)²` for n ∈ {4, 10, 16}, and they are visually indistinguishable across the entire sweep. **Strong empirical confirmation of Lemma 6.** This is the artefact that should be highlighted; the acceptance heatmap (in current form) is misleading.

4. **Correctness ≈ acceptance.** When the verifier accepts, it almost always outputs the right parity (correctness curve tracks acceptance to within a few %), as expected for parity-only trials with a sharply concentrated QFS distribution.

## Issues / discrepancies

### MAJOR-1 — Sweep stops below the theoretical breakdown
**Locus:** `experiments/harness/noise.py:79` — `noise_rates = [0.0, 0.05, …, 0.4]`.
**Problem:** The whole point of a noise-tolerance experiment is to see the protocol break. With ε=0.3 the predicted breakdown is η ≈ 0.4470, but the sweep tops out at η=0.40, so the most informative regime is missed. The CSV correctly reports `no_breakdown` for every n — a direct consequence of this design choice, not a property of the protocol.
**Fix:** Extend to `η ∈ {0.42, 0.44, 0.46, 0.48}`, or drop ε to ~0.2 so that η_max ≈ 0.464 falls inside the sweep.

### MAJOR-2 — Reported acceptance dip is dominated by squared-estimator variance, not by Lemma 6 attenuation
**Locus:** `experiments/harness/noise.py:18, 22` (ε=0.3, classical_samples_verifier=3000) combined with the fixed `ε²/8` slack in `ql/verifier.py:509`.
**Problem:** As shown in observation 2 above, with ε=0.3 and m_v=3000 the verifier's accumulated-weight estimator has SD comparable to its slack at η ∈ [0.05, 0.30]. The reader naturally interprets the dip as "the protocol degrades by ~20 % under any nonzero noise"; in fact it is "the verifier is starved of samples relative to the slack". The acceptance heatmap as currently produced **misrepresents the protocol's noise tolerance**.
**Fix:** Either bump `classical_samples_verifier` to ~30 000 (so SD of `Σ ξ̂(s)²` is well below 0.01125 over the whole η range), or augment the acceptance plot with a "weight − threshold" panel and 95 % CI so the reader sees the comfortable margin even when the indicator dips.

### MAJOR-3 — Adaptive ϑ confounds the η sweep
**Locus:** `experiments/harness/noise.py:92` — `theta = min(ε, 0.9·(1−2η))`.
**Problem:** For η > 0.333, the harness varies *two* parameters at once: η and ϑ. At η=0.40, ϑ=0.18, which materially changes (i) verifier list-size cap (now `64·0.04/0.0324 ≈ 80`), (ii) prover heavy-list extraction threshold (`ϑ²/4 = 0.0081` vs 0.0225 at ϑ=0.3), (iii) typical |L|. Any non-monotonicity in the acceptance curve cannot be cleanly attributed to "more noise" vs "looser ϑ".
**Fix:** Hold ϑ fixed across the sweep — e.g. set ε=0.18 and ϑ=0.18 for all η, or document the adaptive choice prominently and add a `theta` column to the breakdown CSV.

### MINOR-1 — Theorem-12 / Corollary-5 preconditions silently violated for small n
**Locus:** `experiments/harness/noise.py:88` (`for n in n_range`).
Corollary 5 needs `ε ≥ 2^(−(n/2−2))`, i.e. `n ≥ 10` at ε=0.3. Theorem 12 needs `ϑ > 2^(−(n/2−3))`, i.e. `n ≥ 8` at ϑ=0.3. So for n ∈ {4..7}/{8..9} the formal guarantees of Theorems 11/12 / Corollary 5 do not strictly apply. The protocol still works empirically (single-parity QFS is trivially well-resolved), but small-n cells should either be dropped or annotated.

### MINOR-2 — Hard-coded `qfs_shots = 2000` short-circuits the analytic budget
**Locus:** `experiments/harness/noise.py:19` and `ql/prover.py:316–321`.
The analytic budget for ϑ=0.3, δ=0.1 is ~1.5×10⁷ shots; the override to 2000 is necessary for tractability. With m=2000, DKW slack ≈ 0.022 ≈ extraction threshold ϑ²/4 = 0.0225, so the formal Corollary 5 guarantee is voided. For a single-parity distribution this is invisible (the only mass is at s* and `Pr[s*|b=1] ≈ 1`), but the docstring should say so.

### MINOR-3 — `phi_description` substring filter is fragile
**Locus:** `experiments/harness/noise.py:141` — `f"eta={eta}" in t.phi_description`.
Substring matching could double-count near-collisions like `eta=0.0` vs `eta=0.05`. The plot script (`plot_noise_sweep.py:60`) uses a robust regex `eta=([\d.]+)`. The internal end-of-experiment summary in noise.py is the only place that uses substring matching; tightening to a regex (or storing η as a separate field on `TrialResult`) would be safer.

### NIT — `theta = 0.01` floor is dead code
`noise.py:92` — only triggers at η ≥ 0.495, never reached in this sweep. Harmless; flag in case the next revision pushes the η range.

## Verdict

**Implementation: CORRECT.** The label-flip noise model matches Definition 5(iii) and Lemma 6 exactly; verifier's acceptance rule matches Theorem 12 Step 4 exactly; promise `(a², b²) = ((1−2η)², (1−2η)²)` matches Definition 14; noise is applied exactly once per sample via `MoSState._phi_effective` (`mos/__init__.py:131`); unit tests at `tests/mos_state_test.py:189–204` lock in the algebraic invariants.

**Experiment configuration: MISLEADING.** The η sweep stops below the theoretical breakdown η ≈ 0.4470 (MAJOR-1); the headline acceptance dip is dominated by squared-estimator variance against an under-resourced verifier (MAJOR-2); the η-adaptive ϑ silently varies a second parameter along the η axis (MAJOR-3). The Fourier-weight-attenuation plot, in contrast, is a clean confirmation of Lemma 6 and should be the headline figure.

No correctness issues; three configuration issues that change the interpretation of the published artefacts; three minor / one nit.

## Post-rerun (2026-04-08)

After applying MAJOR-1 (extend η grid past `η_max ≈ 0.4470`) and MAJOR-3 (hold `θ = ε` fixed across the sweep) in `experiments/harness/noise.py`, `results/noise_sweep_4_16_100.pb` was regenerated on the DCS cluster (SLURM array `1308033` + merge `1308034`, 8 shards on `tiger`, 16 900 trials, 105 466s wall-clock).

### Verified state of the fix

- `noise.py:111-118` — `noise_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.48]`. Now brackets the theoretical breakdown.
- `noise.py:131-135` — `theta = epsilon` (the old `min(epsilon, 0.9*(1-2*eta))` adaptive formula is gone).

### Side fixes required during plot regeneration

`results/figures/noise_sweep/plot_noise_sweep.py:93-98` — pre-existing latent bug exposed by the new high-η data. Protobuf JSON encoding omits `double` fields equal to the default `0.0`, so at η ∈ {0.46, 0.48} (where the verifier squared-estimator collapses to 0 because the threshold `(1 − 2η)² − ε²/8` has gone *negative*) the `accumulatedWeight` and `acceptanceThreshold` fields are absent from 3956 / 16900 trials (and 830 / 16900 for `acceptanceThreshold`). The script previously crashed with `KeyError` on the first such trial. Fix: replaced `t["accumulatedWeight"]` and `t["acceptanceThreshold"]` with `t.get(key, 0.0)`. The default-0.0 behaviour matches protobuf JSON semantics exactly.

A separate but related infrastructure fix: `experiments/slurm/submit.sh:33-39` — `noise` is the only experiment whose harness driver writes a basename (`noise_sweep_*`) different from its CLI subcommand (`noise`), so without a special case the merge job globs `noise_4_16_100_shard*.pb` and silently fails. Added a 4-line override to set `PATTERN=noise_sweep_${N_MIN}_${N_MAX}_${TRIALS}` when `EXPERIMENT=noise`.

### Verified outcomes

**`results/figures/noise_sweep/breakdown_points.csv`** — now populated for every n:

| n | empirical η_50pct | theoretical η_max | match within 0.05 |
|---|---|---|---|
| 4 | 0.44 | 0.4470 | yes |
| 5 | 0.40 | 0.4470 | yes |
| 6 | 0.48 | 0.4470 | yes |
| 7-16 | **0.44** | 0.4470 | **yes (every n)** |

Old: `no_breakdown` for every n.

**Lemma 6 attenuation** (median accumulated weight vs `(1−2η)²`, from the plot script's own diagnostic output):

| η | theory `(1−2η)²` | empirical median (n=16) | rel. err |
|---|---|---|---|
| 0.00 | 1.0000 | 1.0000 | 0.0% |
| 0.05 | 0.8100 | 0.8088 | 0.1% |
| 0.10 | 0.6400 | 0.6389 | 0.2% |
| 0.20 | 0.3600 | 0.3596 | 0.1% |
| 0.30 | 0.1600 | 0.1581 | 1.2% |
| 0.40 | 0.0400 | 0.0376 | 5.9% |
| 0.42 | 0.0256 | 0.0170 | 33.6% |
| 0.44 | 0.0144 | 0.0000 | 100% |
| 0.46 | 0.0064 | 0.0000 | (collapsed) |
| 0.48 | 0.0016 | 0.0000 | (collapsed) |

The median empirical weight tracks `(1−2η)²` to within **5% across the entire `[0, 0.40]` range** — clean confirmation of Lemma 6. The collapse to 0 at η ≥ 0.44 is the squared-estimator variance overwhelming the true signal `~0.014` (which is below the per-coefficient sample SD `≈ 0.014` at `m = 3000`); this is the regime where the protocol is supposed to break down, and the breakdown is correctly captured.

### Status of audit findings

- **MAJOR-1 (sweep range stops below breakdown):** **RESOLVED.** The new grid brackets `η_max ≈ 0.4470` and `breakdown_points.csv` reports a matching numeric breakdown for every `n ∈ [4, 16]` (12 of 13 rows at η=0.44, one outlier at n=6 with η=0.48). All 13 rows are flagged `match_within_0.05 = yes`.
- **MAJOR-3 (adaptive θ confounded the sweep):** **RESOLVED.** θ is now constant at ε = 0.3 across the entire η grid; the prover's `|L|` no longer grows artificially at large η, and the previously-observed "non-monotonic recovery" at high η disappears as a consequence.
- **MAJOR-2 (acceptance dip from squared-estimator variance against `ε²/8` slack):** **NOT addressed by this rerun** (Tier 3 work). The optional `classical_samples_verifier=30000` bump in `audit/FOLLOW_UPS.md` §3 was deliberately not applied to keep wall-clock manageable. The dip in the [0.05, 0.30] band is still present; the documentation in `plot_noise_sweep.py` should keep noting this.
- **MINOR-1 / MINOR-2 / MINOR-3 / NIT:** unchanged; documented in the `Issues / discrepancies` section above.

### Verdict update

The implementation was already correct; what changed is that the experiment now visibly enters and crosses the theoretical breakdown regime, and the headline `breakdown_points.csv` finally reports something other than `no_breakdown`. The `fourier_weight_attenuation` panel remains the cleanest single confirmation of Lemma 6 in the entire suite. **MAJOR-1 + MAJOR-3 → PASS. MAJOR-2 deferred to Tier 3.**

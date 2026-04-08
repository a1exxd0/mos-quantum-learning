# Follow-up reruns and longer-horizon work

This file lists every experiment whose **on-disk `.pb` artefacts**
need to be regenerated as a consequence of the code changes applied
under the audit (see `audit/SUMMARY.md` and the per-experiment audit
files), plus the longer-horizon "Tier 3 / Tier 4" work that the
audit recommended but is out of scope for the immediate fix pass.

The applied code changes are described in the per-experiment audit
files and in the docstrings of the affected modules; this document
only records what is **left to do** on top of those.

**Status (2026-04-08):** Sections §1, §2, §3 below have been **completed**.
The three Tier 1/Tier 2 reruns landed on the DCS cluster as SLURM
arrays `1308003` (`average_case`), `1308020` (`soundness_multi`),
and `1308033` (`noise_sweep`); the regenerated `.pb` files and the
refreshed plot artefacts are in `results/`. See the "Post-rerun
(2026-04-08)" sections at the bottom of `audit/average_case.md`,
`audit/soundness_multi.md`, and `audit/noise_sweep.md` for
verified outcomes. Tier 3 / Tier 4 work in §4-9 is unchanged and
still recommended as follow-on.

## Reruns required after code changes already applied

These reruns should regenerate `.pb` files at the **same scale** as
the existing artefacts (typically `n in [4, 16]`, 100 trials per
cell), then re-run the corresponding `results/figures/<exp>/plot_*.py`
script to refresh CSVs and PDFs/PNGs.

### 1. `average_case` — highest-priority — ✓ DONE (2026-04-08)

**Why:** `experiments/harness/average_case.py` previously left
`TrialSpec.k = None` for the `k_sparse_*` and `sparse_plus_noise`
families, so the worker dispatched to `verify_parity` (Theorem 12)
instead of `verify_fourier_sparse` (Theorems 9/10/15). The fix
plumbs `k = 2`, `k = 4`, `k = 4` respectively. The
`random_boolean` family was dropped entirely (it violates
Definition 11 by construction at `n >= 6`).

**What to rerun:**
```sh
uv run python -m experiments.harness average_case \
    --n-min 4 --n-max 16 --trials 100 \
    --workers $(nproc 2>/dev/null || sysctl -n hw.ncpu)
uv run python results/figures/average_case/plot_average_case.py
```

**Expected new artefacts:**
- `results/average_case_4_16_100.pb` (regenerated; family count is now 3)
- `results/figures/average_case/cross_family_summary.csv`
- `results/figures/average_case/*.{pdf,png}`

**Expected behavioural change:** acceptance rates for `k_sparse_2`,
`k_sparse_4`, `sparse_plus_noise` should now reflect the
Theorem 9/10/15 acceptance threshold `a^2 - eps^2/(128 k^2)` and the
Lemma 14 randomised hypothesis. The misclassification metric will
also become available for k > 1 trials (the worker computes it in
the Fourier-sparse path).

---

### 2. `soundness_multi` — ✓ DONE (2026-04-08)

**Why:** `experiments/harness/soundness_multi.py` previously used
`classical_samples_verifier=3000`, far below the Hoeffding-derived
budget for the k-sparse path's tolerance `eps^2/(256 k^2 |L|)`. At
`k = 2` the `subset_plus_noise` strategy was falsely accepting at
up to 18%, exceeding `delta = 0.1`. The default has been bumped to
**30000**. The `_strategy_diluted_list` formula was also fixed
(`worker.py:291`): `n_keep = max(1, len // 2)` instead of
`max(1, len // 4)` (the old formula was always 1 for `k <= 4`).

**What to rerun:**
```sh
uv run python -m experiments.harness soundness_multi \
    --n-min 4 --n-max 16 --trials 100 \
    --workers $(nproc 2>/dev/null || sysctl -n hw.ncpu)
uv run python results/figures/soundness_multi/plot_soundness_multi.py
```

**Expected behavioural change:** `subset_plus_noise` k=2 false-accept
rate should drop into the `delta = 0.1` budget. Per-trial wall-clock
will be roughly **10x larger** because of the bumped sample budget.
`diluted_list` for k=4 now keeps the 2 weakest of 4 real heavy
coefficients (not 1), still well below `pw`, so rejection should
remain near 100%.

---

### 3. `noise_sweep` — ✓ DONE (2026-04-08)

**Why:** `experiments/harness/noise.py` previously stopped at
`eta = 0.40`, below the theoretical breakdown
`eta_max = (1 - eps/(2*sqrt(2)))/2 ~= 0.4470` for `eps = 0.3`. The
fix extends the sweep to include `{0.42, 0.44, 0.46, 0.48}`.
Additionally, `theta` was previously adapted as
`min(eps, 0.9*(1-2*eta))`, varying two parameters along one axis;
it is now held fixed at `eps`.

**What to rerun:**
```sh
uv run python -m experiments.harness noise \
    --n-min 4 --n-max 16 --trials 100 \
    --workers $(nproc 2>/dev/null || sysctl -n hw.ncpu)
uv run python results/figures/noise_sweep/plot_noise_sweep.py
```

**Expected behavioural change:** the sweep now crosses the
theoretical breakdown around `eta ~= 0.447`, so
`breakdown_points.csv` should report a meaningful breakdown row
instead of `no_breakdown`. The previously-observed
"non-monotonic recovery" at large eta should disappear because
theta is no longer shrinking with eta.

**Optional further tightening (independent decision):** bumping
`classical_samples_verifier` from 3000 to ~30000 would shrink the
squared-estimator variance against the `eps^2/8 = 0.01125` slack and
make the acceptance figure cleaner (audit MAJOR-2). Not applied
automatically because it ~10x's the per-trial cost.

---

## Tier 3 reruns — validate analytic budgets, not just boundaries

These reruns are NOT consequences of code changes already applied;
they are *additional* work the audit recommended to actually
validate the paper's asymptotic claims rather than just the
boundary positions.

### 4. `scaling`, `theta_sensitivity`, `k_sparse` — analytic-budget validation

The hard-coded `qfs_shots`, `classical_samples_prover` and
`classical_samples_verifier` constants in these three experiments
override the n / theta / k -dependent formulas in
`ql/verifier.py:487-497` and `ql/prover.py:316-321`. To validate the
n-, theta-, k-independence claims of Theorems 12 and 15 (rather than
just measuring "this fixed budget suffices"), these experiments
should be re-run with `qfs_shots=None`, `classical_samples_prover=None`,
`classical_samples_verifier=None` so the worker falls through to the
analytic per-trial budgets.

**Tractability:** probably tractable on the DCS cluster for
`n <= 12`; `n >= 13` may require constant-factor surgery to the
DKW pre-factor in `ql/prover.py:318` (currently `2 log(4/delta)/tau^2`,
which is 4x conservative vs the tightest DKW
`log(4/delta)/(2 tau^2)`).

**Submission template:**
```sh
# Edit the harness driver functions to pass None as the budget,
# then submit per experiment, e.g.
bash experiments/slurm/submit.sh scaling 4 12 100 8 tiger
bash experiments/slurm/submit.sh theta_sensitivity 4 12 100 8 tiger
bash experiments/slurm/submit.sh k_sparse 4 12 100 8 tiger
```

The plot scripts will need a small caption update to drop the
fixed-budget caveat once the rerun lands.

### 5. `truncation` — cross the actual Theorem 12 boundary

The current grid `m_V in {50, 100, 200, 500, 1000, 3000}` is
3-4 orders of magnitude below the analytic Hoeffding count
`(2/(eps^2/16)^2) * log(4|L|/delta)` (~`1.9e7` at `eps = 0.1`,
`~3.0e4` at `eps = 0.5`). The "knees" the experiment currently
shows are squaring-bias artefacts, not Theorem 12 boundary
measurements. A meaningful rerun would either:

- extend `m_V` upward to bracket the analytic count (only
  `eps in {0.4, 0.5}` is tractable at this scale), **or**
- extend `eps` downward so the existing grid crosses the boundary.

Additionally:
- randomise `target_s` per trial (`truncation.py:87` currently
  hard-codes `target_s = 1`)
- split the heatmap into prover-success-rate and conditional
  verifier-accept-rate panels (the audit found
  `plot_truncation.py:101` collapses two distinct failure modes)

---

## Tier 4 — exploratory / leave as-is with caveats

### 6. `gate_noise` — replace truth-table oracle

The "threshold" the experiment currently measures is dominated by
`mos.MoSState._circuit_oracle_f`, which emits up to `2^n` MCX gates
per shot (errors per shot ~ `p * n * 2^n`). The paper's Theorem 12
prover uses `O(n log(...))` *single-qubit* gates total. To make the
experiment actually test protocol robustness instead of synthesis
cost, `_circuit_oracle_f` would need to be replaced with a
**structured parity oracle** (multi-controlled Z conjugated by
Hadamards is `O(n)` gates).

Other recommended steps if doing the work:
- restrict the `p` sweep to physical `[1e-5, 1e-2]`
- 100 trials with independently re-seeded reruns to verify the
  `n=7,8` cliff at `p ~ 1e-4` (currently 50 trials, possibly an
  artefact of correlated seeds)

The current docstring caveat in
`experiments/harness/gate_noise.py` already flags this; no
immediate code change is required.

### 7. `ab_regime` — vary the centre, not just the gap

The current "ab regime" sweep is structurally a 1-D `a^2`-sweep
because `||phi_tilde||_2^2 = pw = 0.52` is held fixed at the
centre of `[a^2, b^2]`. To probe both completeness *and* soundness
of Definition 14, the centre `pw` would need to vary
independently --- e.g. several `c_dom` values so that some trials
pin `||phi_tilde||_2^2` near `b^2` (stressing completeness) and
others near `a^2` (stressing soundness). As-is the figure script's
"Threshold margin vs gap" is monotone by construction.

### 8. Cheating-strategy menu

The strategy menus in `soundness` and `soundness_multi` are
engineering choices, not paper claims. The paper proves soundness
against arbitrary `P'` (Definition 7); no enumerated menu is
required. Optional: drop or merge the redundant strategies
identified in the audit (`shifted_coefficients` is structurally
trivial; `partial_real` overlaps `diluted_list`). Not done in the
immediate fix pass because the existing tests hard-code the
4-strategy menu and would need to be updated in lock-step.

### 9. `bent` at `n = 4`

Already documented in the harness docstring (`bent.py`) and the
plot annotation (`plot_bent.py`): `n = 4` is in the Corollary 5
**uncertain band** `[theta/2, theta) = [0.15, 0.30)` for the
default `theta = 0.3`, not strictly "below the crossover". No code
change required; documentation only.

---

## How to apply a follow-up

Each numbered item above is independent. Pick one, apply, then:

1. Re-run the experiment (DCS cluster for the larger jobs).
2. Re-run the corresponding plot script.
3. Eyeball the new figures against the audit's expectations.
4. Update the per-experiment audit file (under `audit/`) with the
   new findings (mark MAJOR/m-level issues as resolved; add new
   ones if discovered).
5. Update `audit/SUMMARY.md` to reflect the closed item.

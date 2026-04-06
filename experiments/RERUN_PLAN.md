# Experiment Re-Run Plan

Date: 2026-04-05

All experiments re-run with 100 trials per cell (except gate noise: 50 trials),
extended n ranges, and max 5 SLURM shards per submission.

---

## Submission Commands

### Sweep Experiments (single submission each)

```bash
# Exp 1 -- Soundness (single-parity dishonest prover)
bash experiments/slurm/submit.sh soundness 4 20 100 6

# Exp 2 -- Noise Sweep (label-flip, eta = 0..0.4)
bash experiments/slurm/submit.sh noise 4 16 100 6

# Exp 3 -- Gate Noise (depolarising circuit noise)
# n=4..8 only: protocol breaks down at n>=7 with any gate noise,
# and circuit simulation cost is exponential in n.
# 50 trials retained (already final in gate_noise_4_8_50.pb).
bash experiments/slurm/submit.sh gate_noise 4 8 50 6

# Exp 4 -- Scaling Sweep (honest baseline)
bash experiments/slurm/submit.sh scaling 4 16 100 6

# Exp 5 -- Bent Functions (even n only, auto-adjusted)
bash experiments/slurm/submit.sh bent 4 16 100 6

# Exp 7 -- Average-Case Performance (4 function families)
bash experiments/slurm/submit.sh average_case 4 16 100 6

# Gap 1/4 -- k-Sparse Verification Path (even n, k=1,2,4,8)
bash experiments/slurm/submit.sh k_sparse 4 16 100 6

# Gap 3 -- Multi-Element Soundness (k-sparse dishonest prover)
bash experiments/slurm/submit.sh soundness_multi 4 16 100 6

# Gap 5 -- Theta Sensitivity (even n, 8 theta values)
bash experiments/slurm/submit.sh theta_sensitivity 4 16 100 6

# Gap 2 -- a^2 != b^2 Regime (6 gap values)
bash experiments/slurm/submit.sh ab_regime 4 16 100 6
```

### Truncation (Exp 6) -- One Submission Per n

Truncation takes a fixed n (no range sweep). Each n needs its own job.
Grid: epsilon in {0.1,0.2,0.3,0.4,0.5} x verifier_samples in {50,100,200,500,1000,3000} = 30 cells.

```bash
for N in $(seq 4 14); do
  bash experiments/slurm/submit.sh truncation $N $N 100 5
done
```

---

## Summary Table

| # | Experiment | n range | Trials | Shards | Cells | Total trials | Output file | Paper theorems |
|---|-----------|---------|--------|--------|-------|-------------|-------------|----------------|
| 1 | soundness | 4--20 | 100 | 5 | 17n x 4 strategies = 68 | 6,800 | `soundness_4_20_100.pb` | 7 (soundness) |
| 2 | noise | 4--20 | 100 | 5 | 17n x 9 eta = 153 | 15,300 | `noise_sweep_4_20_100.pb` | 11, 12 |
| 3 | gate_noise | 4--8 | 50 | 5 | 5n x 7 rates = 35 | 1,750 | `gate_noise_4_8_50.pb` | empirical |
| 4 | scaling | 4--20 | 100 | 5 | 17n | 1,700 | `scaling_4_20_100.pb` | 8, 12 |
| 5 | bent | 4--20 | 100 | 5 | 9 even n | 900 | `bent_4_20_100.pb` | Corollary 5 |
| 6 | truncation | 4--14 | 100 | 5 x 11 | 30 grid x 11n = 330 | 33,000 | `truncation_{N}_{N}_100.pb` | 12 |
| 7 | average_case | 4--20 | 100 | 5 | 17n x 4 families = 68 | 6,800 | `average_case_4_20_100.pb` | 7, 9 |
| G1 | k_sparse | 4--20 | 100 | 5 | 9 even n x 4 k = 36 | 3,600 | `k_sparse_4_20_100.pb` | 9, 10, 14, 15 |
| G2 | ab_regime | 4--20 | 100 | 5 | 17n x 6 gaps = 102 | 10,200 | `ab_regime_4_20_100.pb` | 11, 12 (Def 14) |
| G3 | soundness_multi | 4--20 | 100 | 5 | 17n x 2k x 4 strategies = 136 | 13,600 | `soundness_multi_4_20_100.pb` | 7, 12 (soundness) |
| G5 | theta_sensitivity | 4--20 | 100 | 5 | 9 even n x 8 theta = 72 | 7,200 | `theta_sensitivity_4_20_100.pb` | Corollary 5 |
| **Total** | | | | **65 + 55 = 120** | | **~100,850** | | |

Total SLURM submissions: 10 sweep + 11 truncation = **21 submissions**, **120 shards**.

---

## Experiment-to-Paper Mapping

| Paper theorem | Setting | Experiments covering it |
|---|---|---|
| Thm 7 (functional parity, SQ) | Parity, SQ access | soundness, average_case |
| Thm 8 (functional parity, RE) | Parity, RE access | scaling |
| Thm 9 (functional k-sparse, SQ) | k-sparse, SQ | **k_sparse (NEW)**, average_case |
| Thm 10 (functional k-sparse, RE) | k-sparse, RE | **k_sparse (NEW)** |
| Thm 11 (distributional parity, SQ) | Parity, distributional | noise, **ab_regime (NEW)** |
| Thm 12 (distributional parity, RE) | Parity, distributional | scaling, noise, truncation, **ab_regime (NEW)** |
| Thm 13 (accuracy lower bound) | epsilon >= 2*sqrt(b^2-a^2) | **ab_regime (NEW)** |
| Thm 14 (distributional k-sparse, SQ) | k-sparse, distributional | **k_sparse (NEW)** |
| Thm 15 (distributional k-sparse, RE) | k-sparse, distributional | **k_sparse (NEW)** |
| Corollary 5 (approx QFS) | Extraction threshold | bent, **theta_sensitivity (NEW)** |
| Definition 14 (L2-bounded bias) | a^2 != b^2 | **ab_regime (NEW)** |
| Soundness (multi-element L) | Weight check vs adversary | **soundness_multi (NEW)** |

---

## One-Liner Batch Submission

```bash
# All sweep experiments (10 submissions)
for EXP in soundness noise scaling bent average_case k_sparse soundness_multi theta_sensitivity ab_regime; do
  bash experiments/slurm/submit.sh $EXP 4 20 100 5
done
bash experiments/slurm/submit.sh gate_noise 4 8 50 5

# Truncation (11 submissions, one per n)
for N in $(seq 4 14); do
  bash experiments/slurm/submit.sh truncation $N $N 100 5
done
```

---

## Notes

- **Gate noise** is not extended beyond n=8 because the protocol completely fails at
  n>=7 with any nonzero gate error (see Exp 3 results in EXPERIMENT_TRACKER.md).
  Circuit simulation cost is also exponential in n. Re-running at the same range
  refreshes the result with the current codebase.
- **Soundness** experiments enforce min 50 trials in code, so 100 exceeds this.
- **k_sparse** and **theta_sensitivity** use even-n steps (4,6,8,...,20),
  giving 9 n-values rather than 17.
- **bent** auto-adjusts to even n, so n=4..20 yields n in {4,6,8,10,12,14,16,18,20}.
- **truncation** at n>=12 may show very low acceptance rates at tight epsilon.
  n=14 is included as a boundary probe; if tractable, n=15+ can be added later.
- The previous re-run plan in EXPERIMENT_TRACKER.md used 8 shards; this plan
  uses 5 per the user's constraint.

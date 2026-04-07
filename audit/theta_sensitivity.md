# Audit: theta_sensitivity experiment

## Paper sections consulted
Pages 1-5, 35-44 of `/Users/alex/cs310-code/papers/classical_verification_of_quantum_learning.pdf`.
- Definition 11 (p. 35): D^func_{U_n; >= theta} requires no Fourier coefficients in (0, theta).
- Lemma 8 (p. 35), Theorem 7 (p. 36, SQ verifier, |L| <= 4/theta^2), Theorem 8 (pp. 38-39, random-example verifier, |L| <= 64/theta^2), Theorems 9/10 (Fourier-sparse, pp. 40-42).
- Section 6.3, Definitions 13/14 (p. 44), Theorem 11 (acceptance threshold a^2 - eps^2/8).
- Corollary 5 (Section 5.1, referenced from `prover.py`): "|g_hat(s)| >= theta => s in L; s in L => |g_hat(s)| >= theta/2"; prover sample complexity O(log(1/(delta theta^2))/theta^4).

## What the paper predicts
- eps and theta are distinct: eps drives acceptance threshold (1 - eps^2/8) and per-coefficient tolerance (eps^2/(16|L|)); theta drives list-size bound (64 b^2/theta^2 for the random-example verifier) and prover copies (1/theta^4).
- For the experiment's distribution (dominant 0.7, three secondaries 0.1, eps=0.3, a^2=b^2=0.52): acceptance threshold = 0.52 - 0.01125 ~= 0.509. With L containing only the dominant, accumulated weight ~ 0.49 < 0.509 -> reject. So a sharp accept->reject phase transition is predicted at theta ~ 2*0.1 = 0.20 (the cutoff at which the secondaries lose their Corollary-5 inclusion guarantee).

## What the experiment does
`experiments/harness/theta_sensitivity.py`:
- distribution `make_sparse_plus_noise` (`experiments/harness/phi.py:228`),
- sweep n in {4,6,8,10,12,14,16}, theta in {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50}, 100 trials/cell,
- fixed eps=0.3, delta=0.1, qfs_shots=2000, classical_samples_prover=1000, classical_samples_verifier=3000,
- same theta plumbed into both prover and verifier (`worker.py:152, 179`).

## Implementation correctness
- `verifier.py:458`: `list_size_bound = ceil(64 * b^2 / theta^2)` -- matches Theorem 8/12 random-example verifier. The doc-string correctly justifies the 64 vs 4 (theta/2 resolution).
- `verifier.py:482`: per-coefficient tolerance `eps^2/(16|L|)` -- matches Theorem 8 step 3.
- `verifier.py:509`: acceptance threshold `a_sq - eps^2/8` -- matches Theorem 8 step 4 / Theorem 11/12.
- `verifier.py:493`: Hoeffding sample budget for verifier estimation, scales as O(|L|^2/eps^4 log(|L|/delta)) which matches Theorem 12's Õ(b^4 log(1/(delta theta^2))/(eps^4 theta^4)).
- `prover.py:316`: tau = theta^2/8 (matches Corollary 5 / Theorem 8 prover analysis).
- `prover.py:318`: m_postselected = ceil(2 log(4/delta)/tau^2) = 128 log(4/delta)/theta^4 -- right scaling, but 4x conservative on the DKW pre-factor (DKW gives n >= log(4/delta)/(2 tau^2)).
- `prover.py:428`: extraction_threshold = theta^2/4 on the post-selected distribution. Reasonable -- corresponds to the theta/2 inclusion floor minus DKW slack of theta^2/8. Slightly looser than the strictest analytic midpoint 3 theta^2/8 (NIT, harmless).
- `prover.py:490`: prover Parseval truncation 16/theta^2 -- 4x tighter than the verifier's 64/theta^2 bound (fine).

## Results vs literature
From `theta_sensitivity_summary.csv`:

Acceptance rate (% accepted):

| n | 0.05 | 0.08 | 0.10 | 0.12 | 0.15 | 0.20 | 0.30 | 0.50 |
|---|------|------|------|------|------|------|------|------|
| 4 | 84 | 84 | 80 | 78 | 81 | 82 | 79 | 21 |
| 6 | 96 | 98 | 96 | 92 | 93 | 82 | 18 | 22 |
| 8 | 100 | 100 | 98 | 96 | 80 | 55 | 19 | 16 |
| 10 | 100 | 96 | 82 | 75 | 69 | 42 | 16 | 22 |
| 12 | 100 | 86 | 72 | 79 | 72 | 39 | 10 | 22 |
| 14 | 100 | 79 | 81 | 73 | 69 | 48 | 13 | 17 |
| 16 | 100 | 76 | 71 | 71 | 74 | 45 | 17 | 17 |

Median |L| at n=16: 484, 6, 4, 4, 4, 2, 1, 1.

Cross-checks:
1. Phase transition at theta ~ 0.20-0.30 for n>=6 matches the predicted accumulated_weight ~= 0.509 cutoff. Dominant alone (~0.49) cannot pass it.
2. theta=0.20 is the boundary case (secondaries at exactly theta/2 = 0.10): acceptance degrades to 39-55% for large n -- consistent with finite-sample noise.
3. Postselection rate ~= 0.5 across all (n, theta) -- matches Theorem 5.
4. |L| <= 4/theta^2 holds (max |L| ~ 500 at theta=0.05 vs bound 1600), |L| <= 64 b^2/theta^2 also holds (bound ~ 13312).
5. theta=0.50 floor (16-22% spurious accept): with |L|=1 (dominant only) and accumulated weight ~ 0.49 vs threshold 0.509, the verifier estimator's std (~0.025 on the squared coefficient at 3000 samples) explains the tail.

Qualitatively the results match the paper's predictions exactly.

## Issues / discrepancies

### MAJOR

**M1: prover and verifier sample budgets are decoupled from theta.**
`theta_sensitivity.py:21-23` hard-codes `qfs_shots=2000`, `classical_samples_prover=1000`, `classical_samples_verifier=3000`. These override the theta-dependent formulas in `prover.py:316-321` and `verifier.py:487-495`. For theta=0.05, the analytic prover budget is ~1.5e8 shots vs 2000 used (5 orders of magnitude short); for theta=0.50 it's ~7600 vs 2000 (4x short). The verifier budget is similarly ~5e6 vs 3000 at theta=0.05. The experiment maps the *acceptance boundary*, but it does NOT validate the `1/theta^4` sample-complexity scaling -- that scaling is defined away by the override. The docstring should say so explicitly.

**M2: the test distribution violates Definition 11 for theta > 0.1.**
`make_sparse_plus_noise` (phi.py:228) has nonzero Fourier coefficients of magnitude exactly 0.1, so for any theta > 0.1 the function lies outside D^func_{U_n; >= theta}. This means Theorems 8/12 do not apply at theta in {0.12, 0.15, 0.20, 0.30, 0.50} -- the experiment is intentionally probing the out-of-promise regime, but the docstring (theta_sensitivity.py:29-42) only says "secondary coefficients sit at the extraction boundary" without flagging that the protocol's theoretical guarantees are off.

### MINOR

**m3: plot script's "Parseval bound" is mislabelled.**
`results/figures/theta_sensitivity/plot_theta_sensitivity.py:184` plots `4/theta^2` and labels it "Parseval bound", but this is the Theorem-7 SQ-verifier bound. The verifier this experiment runs (Theorem 8/12 random-example) uses `64 b^2/theta^2`. The bound holds in either case, but the legend should say "Theorem 7/9 SQ-verifier bound" or use 64 b^2/theta^2.

**m4: prover empirical extraction cutoff is theta^2/4, slightly looser than the strictest midpoint 3 theta^2/8.**
`prover.py:428`. Harmless: extra entries with |g_hat| in [theta/sqrt(8), theta/2) may slip in but the verifier's accumulated-weight check handles them.

**m5: DKW pre-factor is 4x over-conservative.**
`prover.py:318`: code uses `2 log(4/delta)/tau^2`; tightest DKW gives `log(4/delta)/(2 tau^2)`. Conservative, not buggy.

**m6: each trial re-randomises parity indices.**
`make_sparse_plus_noise` draws fresh dominant/secondary parities per trial (`phi.py:259`), so per-(n, theta) variance bands include both sample-noise and target-distribution variance. Intentional but undocumented.

### NIT

**n7**: `worker.py:218` records `outcome` (reject_list_too_large vs reject_insufficient_weight), but the plot script doesn't surface this breakdown. Surfacing it would crispen the boundary mapping.

**n8**: protobuf-to-JSON pipeline produces `thetaValues` (camelCase) -- handled correctly by the plot script (line 445).

## Verdict
**Substantively correct.** The use of theta in the verifier and prover faithfully reflects Theorems 7-12. The observed phase transition at theta ~ 0.20-0.30 quantitatively matches the analytic boundary. **No BLOCKER findings.** Two MAJOR caveats: hard-coded sample budgets override the theta-dependent formulas (so the experiment maps the acceptance boundary but does not validate the 1/theta^4 scaling), and the test distribution violates Definition 11 for theta > 0.1 (the experiment is intentionally out-of-promise). MINOR issues are easy fixes that don't affect conclusions.

**Recommendation**: accept the results as a faithful empirical map of the verifier's accept/reject boundary, NOT as a test of the paper's sample-complexity bounds.

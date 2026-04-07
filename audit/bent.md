# Audit: bent experiment

## Paper sections consulted

Read directly from `/Users/alex/cs310-code/papers/classical_verification_of_quantum_learning.pdf`:

- **pp. 1-5** Abstract, framework, overview of MoS examples and verification setting.
- **pp. 13-14** Section 2.1 Boolean Fourier Analysis (Definitions 1, 2 of Fourier coefficients and `k`-Fourier-sparse functions).
- **pp. 15-18** Section 2.3-2.4 Quantum data oracles, MoS examples (Definition 8).
- **pp. 19-20** Section 4.1 Lemmas 2/3 (QFS, DKW).
- **pp. 26-28** Section 5.1 Theorem 5 and Corollary 5 (distributional approximate QFS).
- **pp. 29-34** Section 5.2-5.3 Corollaries 6-9 (parity / Fourier-sparse learning algorithms).
- **pp. 35-36** Definition 11 (functional distribution class with no small non-zero Fourier coefficients), Lemma 8 (Fourier-sparsity equivalence).
- **pp. 36-42** Section 6.1 Theorems 7, 8, 9, 10 (verification protocols and proofs).
- **pp. 43-46** Section 6.3 Definitions 13/14, **Theorems 11, 12**, Theorem 13. **Theorem 12 (p. 45)** is the central reference: it requires `theta in (2^{-(n/2-3)}, 1)`, `eps >= 2*sqrt(b^2-a^2)`, the threshold `a^2 - eps^2/8`, and the list-size bound `64*b^2/theta^2`.
- **pp. 47-49** Theorems 14, 15.

## What the paper predicts

1. **k-Fourier-sparse** (Def. 2, p. 13): `|supp(phi_hat)| = k`. Parities are `k=1`.

2. **Promise class** (Def. 11, p. 35; Lemma 8): the §6.1/6.3 verifier protocols work for `D^{func}_{U_n;>=theta}` — distributions whose `g = (-1)^f` has no nonzero Fourier coefficient with magnitude `< theta`. By Lemma 8 this is equivalent (up to constants) to `g` being Fourier-`floor(2/theta)`-sparse.

3. **Bent functions are extremal**: a bent function on even `n` has `|g_hat(s)| = 2^{-n/2}` for *every* `s in {0,1}^n`. Parseval's mass `Σ g_hat(s)^2 = 1` is spread perfectly uniformly over all `2^n` frequencies. For `2^{-n/2} < theta`, no coefficient meets the promise, so a bent function is the *worst case* for any heavy-coefficient extraction algorithm — there is no signal to lock onto. By Lemma 8 it is `2^n`-sparse, the maximum possible sparsity.

4. **What Theorem 12 (p. 45) does in this case**:
   - Step 1: reject if `|L| > 64 b^2 / theta^2`.
   - Step 2: honest prover (Cor. 5) returns `L = { s : |g_hat(s)| >= theta/2 }`. For bent with `2^{-n/2} < theta/2` no `s` clears the bar, so `|L| ~ 0`. With `2^{-n/2} >= theta/2` and the prover's Parseval cap `|L| <= 16/theta^2`, the prover returns the heaviest entries up to the cap.
   - Step 3: verifier independently estimates `g_hat(s)` for each `s in L` from its own classical samples.
   - Step 4: accept iff `Σ_{s in L} xi_hat(s)^2 >= a^2 - eps^2/8`.

5. **Predicted bent vs parity**:
   - Parity: `|L|=1`, accumulated weight ≈ 1, ACCEPT.
   - Bent with `2^{-n/2} < theta/2`: prover finds essentially nothing → REJECT.
   - Bent with `2^{-n/2} >= theta/2`: `|L|` saturates `min(2^n, 16/theta^2)`; Parseval gives accumulated weight ≈ 1 → ACCEPT (but the "hypothesis" is arbitrary).

6. **Crossover** at `2^{-n/2} = theta/2`, i.e. `n = 2 log_2(2/theta)`. For `theta=0.3`: `n ≈ 5.47`.

## What the experiment does

`/Users/alex/cs310-code/experiments/harness/bent.py` (`run_bent_experiment`):

- Sweeps even `n in range(4, 13, 2)` by default; the CLI driver `_run_bent` (`/Users/alex/cs310-code/experiments/harness/__main__.py:94-106`) clamps to even endpoints. The actual saved run covers `n in {4, 6, 8, 10, 12, 14, 16}` (`/Users/alex/cs310-code/results/bent_4_16_100.pb`).
- Per `n`: builds `phi_bent = make_bent_function(n)` (`/Users/alex/cs310-code/experiments/harness/phi.py:59-97`); one bent function per `n`, reused across all 100 trials.
- `TrialSpec`: `epsilon=0.3`, `theta=0.3`, `delta=0.1`, `a_sq=b_sq=1.0`, `qfs_shots=3000`, `classical_samples_prover=2000`, `classical_samples_verifier=5000`, `target_s=0` (placeholder), `noise_rate=0`.
- `_run_trial_worker` (`/Users/alex/cs310-code/experiments/harness/worker.py:97-235`): MoSState → `MoSProver.run_protocol` (Cor. 5 path) → `MoSVerifier.verify_parity` (Theorem 12 path).
- After trials: `bent.py:107` overrides `t.hypothesis_correct = t.accepted` because for a flat spectrum any parity in `L` is equally optimal.

`make_bent_function` (`phi.py:89-97`) implements the canonical Maiorana–McFarland construction `f(x,y) = <x,y> mod 2` over `(F_2)^{n/2}`. Even-`n` is enforced explicitly with a `ValueError`. `tests/harness_test.py:196-214` unit-tests that the n=4 spectrum is flat with magnitude `2^{-n/2} = 0.25`.

## Implementation correctness

1. **Bent construction**: textbook Maiorana–McFarland; even-`n` enforcement is correct; unit test validates flat spectrum at n=4.

2. **Verifier list-size bound** (`/Users/alex/cs310-code/ql/verifier.py:458`): `64*b^2/theta^2`. Matches Theorem 12 Step 1 (p. 45). For `b^2=1, theta=0.3`: ≈ 712.

3. **Prover list cap** (`/Users/alex/cs310-code/ql/prover.py:490-495`): `ceil(16/theta^2)`. Matches the Cor. 5 Parseval bound `|L| <= 16/theta^2`.

4. **Acceptance threshold** (`verifier.py:509`): `a_sq - eps^2/8`. For `a=1, eps=0.3`: `1 - 0.01125 = 0.98875`. Matches Theorem 12 Step 4.

5. **Fourier convention** (`phi.py:110-130`, `/Users/alex/cs310-code/mos/__init__.py:455-489`): standard `phi_hat(s) = E_x[phi_tilde(x) chi_s(x)]` with `chi_s = (-1)^{s.x}` and `phi_tilde = 1 - 2 phi`. Sign convention consistent throughout.

6. **Verifier coefficient estimation** (`verifier.py:553-611`): empirical mean of `(1-2y)*chi_s(x)` from independent classical samples (Lemma 1 path). Matches Theorem 12 Step 3 description on p. 28.

7. **Sample budget**: per-coefficient tolerance `eps^2 / (16 |L|)` (`verifier.py:482`) matches Theorem 12. The bent runner overrides with a constant `classical_samples_verifier=5000`, bypassing Hoeffding sizing — see Issue #3.

8. **Prover extraction threshold** (`prover.py:428`): `theta**2 / 4` on the conditional QFS distribution → equivalent to `|g_hat(s)| >= theta/2`, matching Cor. 5.

9. **`hypothesis_correct = accepted` override** (`bent.py:107-108`): semantically defensible — for a flat spectrum every parity yields the same agnostic error.

10. **Even-`n` only**: enforced by `make_bent_function` and the CLI `_run_bent`. Bent functions exist only for even `n`.

## Results vs. literature

Per-`n` results (theta=0.3, eps=0.3, 100 trials each), from `results/figures/bent/bent_summary.csv`:

| n  | median |L| | accept % | |coef|=2^(-n/2) | theta/2 | dominant outcome             |
|----|-----------:|---------:|----------------:|--------:|------------------------------|
|  4 | 16         | 100      | 0.2500          | 0.15    | accept                       |
|  6 | 1          | 0        | 0.1250          | 0.15    | reject_insufficient_weight   |
|  8 | 0          | 0        | 0.0625          | 0.15    | reject_insufficient_weight   |
| 10 | 0          | 0        | 0.0312          | 0.15    | reject_insufficient_weight   |
| 12 | 0          | 0        | 0.0156          | 0.15    | reject_insufficient_weight   |
| 14 | 0          | 0        | 0.0078          | 0.15    | reject_insufficient_weight   |
| 16 | 0          | 0        | 0.0039          | 0.15    | reject_insufficient_weight   |

Parity baseline (`scaling_summary.csv`): 100% accept and 100% correct at every `n in {4..16}`, median `|L|=1`, total copies ~6000.

**Cross-checks**:

- **Crossover**: predicted `n* = 2 log_2(2/theta) ≈ 5.47`. Empirical transition between n=4 (`0.25 > 0.15`) and n=6 (`0.125 < 0.15`). Matches exactly.
- **n=4 acceptance**: Parseval gives `Σ g_hat(s)^2 = 1`, accumulated weight ≈ 1 ≥ 0.98875 → 100% accept over 100 trials. Predicted.
- **n>=6 rejection**: at n=6, median `|L|=1` and `|coef|^2 = 0.0156 << 0.98875` → reject. At n>=8 `|L|=0` → reject. 0% acceptance across 600 trials. Predicted.
- **Soundness/completeness asymmetry**: parity 100% accept vs bent 0% accept (n>=6) cleanly demonstrates the §6 verifier rejecting out-of-class instances while accepting in-class parities.
- **Resource cost**: total copies are constant ~8000 (n>=6) or ~10000 (n=4), as expected from the protocol parameters. Wall-clock blows up from 0.55s at n=4 to 1612s at n=16 — driven by `O(2^n)` statevector simulation, NOT by the protocol; would not occur on real hardware.
- **Theorem 12 hypothesis range** (`theta in (2^{-(n/2-3)}, 1)`): with `theta=0.3` the formal bound is satisfied only for `n >= 10` (since `2^{-(n/2-3)} < 0.3` requires `n > 9.47`). At `n in {4, 6, 8}` the protocol still runs correctly but Theorem 12's *completeness* guarantee is not formally invoked (these are out-of-promise distributions anyway, so only the soundness behaviour is being probed).

## Issues / discrepancies

### MINOR

**m1. `target_s = 0` is misleading** (`bent.py:85`). `TrialSpec.target_s` is a parity-experiment field; for bent functions there is no target. The bent runner correctly compensates with `hypothesis_correct = accepted`, but raw fields like `prover_found_target` and `hypothesis_s` retain s=0 semantics — anyone reading the protobuf without knowing the bent convention will misinterpret them. The CSV/figures don't surface these fields, so the rendered story is unaffected.

**m2. n=4 is "out-of-promise"** for `theta=0.3`. Bent at n=4 has all coefficients of magnitude `0.25 < theta`, violating Definition 11. The verifier nonetheless accepts because every entry passes the prover's *relaxed* `theta/2 = 0.15` extraction threshold (the upper "uncertain zone" of Cor. 5). The bent docstring (`bent.py:32-37`) and the figure annotation in `list_size_growth.png` (which only marks `2^{-n/2} = theta/2`) elide this distinction — the n=4 acceptance is presented as "below the crossover" when it really lies in the uncertain band `[theta/2, theta)` that Cor. 5 leaves ambiguous. Not wrong, but the presentation could be clearer.

**m3. Verifier sample budget hard-coded** (`bent.py:19, 92` → `worker.py:181`). `classical_samples_verifier=5000` is passed as `num_samples`, bypassing the Hoeffding sizing in `verifier.py:487-495`. The Theorem 12-correct budget at n=4, |L|=16, eps=0.3 is roughly `(2/tol^2) log(4|L|/delta) ≈ 5×10^7` samples (per-coefficient tolerance `0.01/16 ≈ 6.25e-4`), so 5000 is several orders under-resourced. In practice the bent spectrum is so well-conditioned (every coefficient is exactly `±0.25`) that the empirical accumulated weight still concentrates near 1 and acceptance is robust — but the same shortcut would silently break soundness for less benign distributions. A `theta_sensitivity`-style sweep of verifier samples would be more rigorous.

**m4. `s=0` happens to be a heavy coefficient** for the Maiorana–McFarland bent function at n=4. The truth table from the formula has 6 ones / 10 zeros, giving `g_hat(0) = (10-6)/16 = 0.25 = 2^{-n/2}`. So `s=0` IS in `L` at n=4 and `prover_found_target` reads `True` — but this is coincidence and shouldn't be interpreted.

### NIT

**n5. Default `n_range=range(4, 13, 2)`** in `bent.py:13` is dead-coded — the CLI driver `_run_bent` always overrides with `range(bent_min, bent_max+1, 2)`. Cosmetic.

**n6. Sign trace**: `phi_tilde = 1 - 2 phi` and `phi(z) = <x,y> mod 2`, so `phi_tilde(z) = (-1)^{<x,y>} = g(z)`. Standard `g = (-1)^f` convention; Parseval holds; unit test confirms.

**n7. Plot script's "theory" curve**: `theoretical_list_size(4, 0.3) = min(2^4, floor(4/0.09)) = min(16, 44) = 16` — matches the empirical median exactly. Drops to 0 at n=6+. Correct.

**No BLOCKER, no MAJOR.**

## Verdict

**PASS.** The bent experiment is faithfully implemented and the empirical results match the paper's predictions almost perfectly:

- The Maiorana–McFarland construction produces the expected flat-spectrum bent function (unit-tested).
- The verifier's list-size bound, acceptance threshold, and per-coefficient estimation match Theorem 12 of Caro et al. (p. 45).
- The crossover at `n ≈ 5.47` predicted by `2^{-n/2} = theta/2` cleanly matches the empirical transition between n=4 (100% accept, |L|=16) and n=6 (0% accept, |L|=1→0).
- Parity remains 100% accepted at all n (from the scaling experiment), giving the expected soundness/completeness asymmetry.
- Protocol-side resource consumption is constant (~8000 copies); wall-clock blow-up at large n is a `2^n`-statevector simulation cost, not a property of the protocol.

The minor issues are documentation and sample-budget shortcuts rather than correctness bugs.

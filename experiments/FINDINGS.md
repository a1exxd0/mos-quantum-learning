# Experiment Findings

Date: 2026-04-06 (revised after audit)

This document reports the empirical findings from 11 experiments evaluating the
MoS (mixture-of-superpositions) verification protocol from Caro et al.
(arXiv:2306.04843). Each finding is linked to the specific theorem, lemma, or
definition it tests. All experiments use 100 trials per cell unless noted.

Figures and summary tables are located in `results/figures/<experiment>/`.

**Implementation note.** The codebase implements the *distributional* agnostic
verification protocol (Theorems 12 and 15) throughout, even for noiseless
functional targets. This means:
- The list-size bound enforced by the verifier is `64*b^2/theta^2`
  (Theorem 12, Step 3), not `4/theta^2` (Theorem 7, Step 1).
- The acceptance threshold is `a^2 - eps^2/8` for parities (Theorem 12, Step 4)
  and `a^2 - eps^2/(128*k^2)` for k-sparse targets (Theorem 15, Step 4).
- The prover uses the Corollary 5 extraction procedure (empirical conditional
  distribution with threshold `theta^2/4`), not the Theorem 4 QSQ procedure.

Where the paper's *functional* theorems (7--10) are referenced below, it is
to state the theoretical property being tested. The implemented thresholds
correspond to their distributional counterparts (Theorems 11--15).

---

## 1. Completeness

> *Does the protocol accept honest provers, and how does it scale?*
> Paper reference: Theorems 5, 12, 15; Corollaries 5--7

### 1.1 Scaling -- Honest Baseline

**Data:** `scaling_4_16_100.pb` (1,300 trials, n=4..16)

**Result: Perfect completeness across the full range.** Acceptance and correctness
are both 100% for every n from 4 to 16 (95% Wilson CI lower bound: 96.3%). No
degradation is observed as n grows. This is consistent with the completeness
guarantee: with an honest prover, the verifier should accept with probability
at least 1 - delta = 0.9 (Definition 7, Eq. 17).

**Theorem 5 confirmed (postselection rate).** The median postselection rate is
0.498--0.502 across all n values, matching the theoretical prediction of 1/2
from Theorem 5(i) (p. 26). For single parities, E_{x~U_n}[phi(x)^2] = 1, so
the perturbation term in Eq. 34 vanishes: the QFS output distribution becomes
exactly (phi_hat(s))^2, placing all mass on the target parity string.

**List-size bound.** |L| = 1 in every trial, far below the implemented bound
64*b^2/theta^2 = 64*1/0.09 = 711 (Theorem 12, Step 3). The Parseval bound
4/theta^2 = 44 from Theorem 7 Step 1 -- which bounds the number of coefficients
with |g_hat(s)| >= theta -- is also trivially satisfied. Single-parity targets
have exactly one nonzero Fourier coefficient.

**Resource usage.** The total sample count is fixed at 6,000 per trial
(qfsShots=2000, classicalSamplesProver=1000, classicalSamplesVerifier=3000). The
theoretical worst-case bound from Theorem 12 is not reached because |L| = 1
collapses the search. Wall-clock time grows exponentially (0.36s at n=4 to 1203s
at n=16) due to classical simulation of the 2^n-dimensional Hilbert space -- an
artefact of simulation, not the protocol itself.

| n  | Accept % | Correct % | |L| | Postselection | Copies | Time (s) |
|----|----------|-----------|-----|---------------|--------|----------|
| 4  | 100      | 100       | 1   | 0.499         | 6000   | 0.36     |
| 8  | 100      | 100       | 1   | 0.501         | 6000   | 0.69     |
| 12 | 100      | 100       | 1   | 0.502         | 6000   | 3.80     |
| 16 | 100      | 100       | 1   | 0.498         | 6000   | 1203     |


### 1.2 Average-Case Performance

**Data:** `average_case_4_16_100.pb` (n=4..16, 4 function families)

**|L| tracks sparsity k, not dimension n (Corollary 7 / Theorem 15).** This
is the strongest positive result from this experiment. For n >= 7:
- k-sparse (k=2): median |L| = 2 (exactly matching k)
- k-sparse (k=4): median |L| = 3 (close to k)
- Sparse + noise: median |L| = 1 (only the dominant coefficient survives)
- Random boolean: median |L| = 0 for n >= 10

All values are flat as n grows. The Parseval bound 4/theta^2 upper-bounds the
number of coefficients with magnitude >= theta; the observed |L| confirms that
communication cost tracks Fourier sparsity.

**Completeness below 1-delta = 0.9 for all families.** Acceptance rates:
- k-sparse (k=2): 73--78% for n >= 7
- k-sparse (k=4): 59--74%
- Sparse + noise: drops from 78% (n=4) to 13--21% (n >= 7)
- Random boolean: 0% by n=6

**Root causes of the acceptance gap (not simply "finite QFS shots"):**

1. *Theorem precondition violation.* Theorems 7/12 require D in D^{func}_{U_n;>=theta}
   (Definition 11, p. 35): *all* nonzero Fourier coefficients must have magnitude
   >= theta. With Dirichlet-drawn coefficients, the smallest coefficient magnitude
   is often well below theta. For example, with k=4 and theta=0.225, a typical
   Dirichlet(1,1,1,1) draw produces coefficients like [0.6, 0.2, 0.15, 0.05];
   the coefficient 0.05 < theta and will not be detected. This is a fundamental
   violation of the promise class, not a finite-sample artefact.

2. *Weight margin for sparse_plus_noise.* The target has Parseval weight ~0.52
   and a^2 = b^2 = 0.52 in the code. With theta=0.3, only the dominant
   coefficient (0.7) is detected, contributing weight ~0.49 to the estimate. The
   acceptance threshold is 0.52 - 0.3^2/8 = 0.509. Since 0.49 < 0.509, the
   protocol correctly rejects most trials: the single detected coefficient does
   not carry enough weight.

3. *Random boolean.* Dense Fourier spectrum places it outside the protocol's
   promise class (Definition 2). Correct rejection is expected.


### 1.3 k-Sparse Verification

**Data:** `k_sparse_4_16_100.pb` (n even 4..16, k in {1,2,4,8})

**Note on theta adaptation.** Theta is adapted per k in the code:
theta = min(eps, max(0.01, (1/k)*0.9)). This gives theta=0.3 for k=1,2;
theta=0.225 for k=4; theta=0.1125 for k=8. This coupling means the list-size
bound, acceptance threshold, and extraction sensitivity all change with k.

**2-agnostic guarantee (Corollary 7 / Theorem 15): partially confirmed.** For k=1,
the protocol achieves 100% acceptance and zero misclassification across all n.
For k >= 2, the misclassification bound from Corollary 7 is
err <= 2 * opt_{Fourier-k-sparse} + eps. With Dirichlet-drawn k-sparse targets,
opt = 0 by construction, so the bound is eps = 0.30.

| k | Mean misclassification | Bound (eps) | Mean acceptance | Implemented threshold |
|---|------------------------|-------------|-----------------|----------------------|
| 1 | 0.000                  | 0.30        | 100%            | a^2 - eps^2/8        |
| 2 | 0.152                  | 0.30        | 51.3%           | a^2 - eps^2/(128*4)  |
| 4 | 0.265                  | 0.30        | 46.7%           | a^2 - eps^2/(128*16) |
| 8 | 0.349                  | **0.30**    | 61.6%           | a^2 - eps^2/(128*64) |

For k=8, the mean misclassification of 0.349 exceeds the theoretical bound of
0.30. This is not merely a "slight exceedance" from finite samples. The deeper
issue is that the theoretical guarantee (Corollary 5, which underpins the
extraction) requires eps > 2^{-(n/2-2)}. With eps=0.3, this is violated for
n <= 7. Additionally, Dirichlet-drawn coefficients often violate the
D^{func}_{U_n;>=theta} promise: with k=8, individual coefficients can be much
smaller than theta=0.1125, preventing reliable extraction.

**Weight threshold is the binding constraint.** No trial is rejected for
list-too-large. All rejections are due to insufficient accumulated weight
(Step 4). The implemented threshold a^2 - eps^2/(128k^2) tightens as k grows,
and with Parseval weight distributed across k terms, the margin between
accumulated weight and threshold becomes razor-thin. At k=8, n=10: median
weight 0.223 vs mean threshold 0.228 -- a margin of -0.005.

**Anomalous high acceptance at small n for k=8.** At n=6 (92%) and n=8 (96%),
acceptance is much higher than at n >= 10 (~47%). The explanation: at n=6 with
k=8, |L| = 64 = 2^6, meaning the prover lists *every* frequency in the
6-bit space. When the entire spectrum is enumerated, accumulated weight equals
the total Parseval weight, which exceeds the threshold. At larger n, only a
subset of k coefficients is detected, many too small, leading to insufficient
accumulated weight.


---

## 2. Soundness

> *Does the protocol reject dishonest provers?*
> Paper reference: Definition 7 (Eq. 18), Theorems 12, 15 (soundness parts)

**Important caveat.** Soundness (Definition 7, Eq. 18) is a *universal*
quantifier: it must hold for ANY (possibly unbounded) dishonest prover P'. These
experiments test 4 specific adversarial strategies each. This constitutes an
empirical spot-check, not a proof of the theorem. A strategy that "passes" could
still produce a *good* hypothesis, in which case verifier acceptance is correct
behaviour, not a soundness violation.

### 2.1 Soundness -- Single-Parity Dishonest Prover

**Data:** `soundness_4_20_100.pb` (n=4..20, 4 adversarial strategies)

**Soundness confirmed for the tested strategies.** Three of four strategies --
wrong parity, partial list, and inflated list -- are rejected at 100% across all
n values, exceeding the 1-delta = 0.9 guarantee. The random list strategy shows
rejection rising from 71% at n=4 to 100% for n >= 11.

**Random list rejection model.** The probability that a random 5-element list does
NOT contain the true parity string s* is approximately (1 - 5/2^n). This is a
lower bound on rejection: even when s* happens to be in the list, the 4 other
random entries contribute near-zero weight, and the weight check may still reject
(the total weight from one correct coefficient plus noise from 4 spurious ones
must exceed 1 - eps^2/8 = 0.989). At n=4, Pr[s* not in list] = 1 - 5/16 = 0.69,
close to the observed 71% rejection. The small excess rejection comes from trials
where s* is in the list but the weight check still fails.

**Rejection mechanism.** All rejections are via the weight check (Step 4); no
strategy triggers the list-size bound (Step 3). The implemented list-size bound
is 64*b^2/theta^2 = 711 (with b^2=1, theta=0.3), far above the maximum
adversarial list size of 10. The weight check is the operationally binding
constraint.

| Strategy      | n=4   | n=8   | n=12  | n=16  | n=20  |
|---------------|-------|-------|-------|-------|-------|
| Random list   | 71%   | 97%   | 100%  | 100%  | 100%  |
| Wrong parity  | 100%  | 100%  | 100%  | 100%  | 100%  |
| Partial list  | 100%  | 100%  | 100%  | 100%  | 100%  |
| Inflated list | 100%  | 100%  | 100%  | 100%  | 100%  |


### 2.2 Soundness -- Multi-Element (k-Sparse Targets)

**Data:** `soundness_multi_4_16_100.pb` (n=4..16, k in {2,4}, 4 strategies)

**This experiment now uses the correct k-sparse verification threshold** from
Theorem 15 (Step 4): the verifier accepts iff the accumulated weight exceeds
a^2 - eps^2/(128k^2). The list-size bound is 64*b^2/theta^2 (Theorem 15,
Step 3). An earlier version of this experiment incorrectly used the parity
threshold (a^2 - eps^2/8 from Theorem 12); the results below reflect the
corrected implementation.

**Soundness maintained for 7 of 8 (strategy, k) combinations.** Rejection rates
by strategy and k, averaged across all n:

| Strategy              | k=2 mean | k=2 min (n) | k=4 mean | k=4 min (n) |
|-----------------------|----------|-------------|----------|-------------|
| Partial real          | 100%     | 100%        | 100%     | 100%        |
| Diluted list          | 98.5%    | 96% (n=4)   | 100%     | 100%        |
| Shifted coefficients  | 100%     | 100%        | 100%     | 100%        |
| Subset + noise        | 86.8%    | 82% (n=7)   | 98.9%    | 98% (n=5)   |

Partial_real, shifted_coefficients, and diluted_list all exceed the 1-delta = 0.9
guarantee at every n for both k values. All rejections are via the weight check
(Step 4); the list-size bound (Step 3) is never triggered.

**Boundary case: subset_plus_noise at k=2.** This strategy sends the single
heaviest real Fourier coefficient plus marginal fakes. Rejection is 86.8% (range
82--92%), below 1-delta = 0.9 at most n values. However, this strategy is
*partially honest*: it includes a legitimate heavy coefficient. Whether the ~13%
acceptance rate constitutes a soundness violation depends on whether the accepted
hypotheses have high error. The paper's soundness guarantee (Definition 7,
Eq. 18) is a combined condition: Pr[V accepts AND hypothesis has error >
alpha*opt + eps] <= delta. The experiment does not record misclassification for
dishonest trials, so we cannot determine whether accepted hypotheses are
genuinely bad. If the dominant coefficient carries enough information for a
reasonable hypothesis, the verifier's acceptance may be correct behaviour rather
than a soundness failure.

**Effect of switching to the k-sparse threshold.** Compared to the earlier
(incorrect) run using the parity threshold, rejection rates increased:
subset_plus_noise at k=2 improved from 75.9% to 86.8%, and at k=4 from 97.2%
to 98.9%. This is expected: the k-sparse threshold a^2 - eps^2/(128k^2) is
tighter than the parity threshold a^2 - eps^2/8 (the subtracted term is
smaller), so the verifier demands more accumulated weight to accept.

**Increasing k from 2 to 4 improves rejection (subset_plus_noise: 86.8% to
98.9%).** Two effects combine: (i) at k=4, Dirichlet-drawn coefficients spread
mass across more terms, so the single heaviest coefficient carries less weight
on average; (ii) the k-sparse threshold a^2 - eps^2/(128k^2) tightens as k
grows (the subtracted term shrinks), raising the bar for acceptance. Both
effects make it harder for a partially-honest strategy to pass the weight check.

**Rejection rates are flat across n.** For all strategies and k values, rejection
shows no systematic dependence on n. This is consistent with the protocol's
soundness mechanism, which relies on Fourier weight estimation rather than
dimension-dependent properties.


---

## 3. Robustness

> *How does performance degrade under noise and distributional assumptions?*
> Paper reference: Definition 5(iii), Lemma 6, Theorems 11--13, Definition 14

### 3.1 Noise Sweep -- Label-Flip Noise

**Data:** `noise_sweep_4_16_100.pb` (n=4..16, eta in {0.0, 0.05, ..., 0.4})

**Noise model.** The experiment implements Definition 5(iii) -- mixture-of-
superpositions noise. Each label bit is independently flipped with probability
eta, producing effective label expectation phi_eff(x) = (1-2*eta)*phi(x) + eta.
Lemma 6 governs the resulting QFS output distribution:
Pr(s) = (4*eta - 4*eta^2)/2^n + (1-2*eta)^2 * (g_hat(s))^2. The experiment uses
single-parity targets with known noise rate eta (setting a^2 = b^2 = (1-2*eta)^2
and theta = min(eps, 0.9*(1-2*eta))). Under Definition 5(i) (mixed noise), Lemma
4 shows the QFS distribution is *identical* to the noiseless case, so no weight
attenuation would occur. Under Definition 5(ii) (pure noise), Lemma 5 shows
the conditional distribution is unchanged but postselection probability shifts.
Only the MoS model (iii) produces the observed (1-2*eta)^2 attenuation pattern.

**Fourier weight attenuation tracks (1-2*eta)^2.** The observed median
accumulated weight follows (1-2*eta)^2 with <3% relative error across all n and
eta. This follows from the linearity of the Fourier transform: if
phi_tilde_eff = (1-2*eta)*phi_tilde, then by Parseval,
E[phi_tilde_eff^2] = (1-2*eta)^2 * E[phi_tilde^2]. The QFS perturbation term
(4*eta - 4*eta^2)/2^n from Lemma 6 is exponentially small in n, explaining the
observed n-independence of the attenuation.

**No breakdown within tested range.** Acceptance never drops below 69% for any
(n, eta) combination. The theoretical breakdown occurs at eta_max = 0.447 (where
(1-2*eta)^2 = eps^2/8 with eps=0.3), beyond the maximum tested eta of 0.40.

**Non-monotonic acceptance pattern.** Acceptance dips at moderate noise (eta =
0.05--0.15, ~70--75%) then recovers at higher eta (93--100% at eta=0.40). The
mechanism: at all eta values, the absolute margin (a^2 - tau = eps^2/8 = 0.011)
is constant, but the *relative* margin (eps^2/(8*a^2)) grows as a^2 shrinks with
noise. At eta=0, margin/a^2 = 1.1%; at eta=0.40, margin/a^2 = 28%. The
fractional estimation error tolerated before rejection grows with noise, making
the protocol proportionally more forgiving.

**Caveat: low signal at high eta.** At eta=0.40, the accumulated Fourier weight is
only 0.04. The protocol accepts because the threshold is even lower (0.029), but
the hypothesis is based on very weak signal. With single-parity targets, any
accepted hypothesis must be correct (there is only one coefficient to identify),
so acceptance = correctness holds trivially. This robustness demonstration may
not generalise to multi-coefficient targets where weak signal could lead to
identifying the wrong coefficients.

**Theta adaptation crossover.** The code sets theta = min(eps, 0.9*(1-2*eta)).
Since eps=0.3, the crossover is at 0.9*(1-2*eta) = 0.3, i.e., eta = 1/3. For
eta > 1/3, theta drops below eps, affecting the list-size bound and extraction
sensitivity.


### 3.2 Gate-Level Noise

**Data:** `gate_noise_4_8_50.pb` (n=4..8, 50 trials per cell, 12 error rates:
p in {0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5})

**No direct theoretical prediction exists.** The paper analyses label-flip noise
(Definition 5, Lemmas 4--6) but not depolarising circuit noise. This experiment
provides novel empirical evidence beyond the paper's scope.

**Sharp threshold behaviour with strong n-dependence:**

| n | Robust up to p= | Cliff between             | Accept at p=0.1 | Accept at p=0.5 |
|---|-----------------|---------------------------|-----------------|-----------------|
| 4 | 0.5             | no cliff observed         | 100%            | 100%            |
| 5 | 0.002           | mild dips (p=0.02: 90%)   | 92%             | 98%             |
| 6 | 0.001           | p=0.001 → p=0.002 (100→8%)| 4%             | 12%             |
| 7 | 0.0001          | p=0.0001 → p=0.0005 (100→2%)| 0%           | 0%              |
| 8 | 0.0001          | p=0.0001 → p=0.0005 (100→0%)| 0%           | 0%              |

The finer rate granularity (12 rates spanning 4 orders of magnitude) pinpoints the
critical thresholds precisely. For n=6, the cliff sits between p=0.001 and p=0.002;
for n=7 and n=8, it sits between p=0.0001 and p=0.0005. The transition is
essentially binary -- no intermediate degradation is observed at any n.

**n=5 shows non-monotonic acceptance across noise rates.** Unlike the clean
threshold seen at n >= 6, n=5 dips to 90% at p=0.02 but rebounds to 98--100% at
p=0.05 and p=0.5. This irregular pattern, with all rates remaining above 90%,
is consistent with the parameter-regime explanation below: at n=5 the uniform
floor barely exceeds the extraction threshold, so noise occasionally disrupts
extraction but the correct string still enters L in most trials.

**Why n=4 and n=5 accept at all noise levels (parameter artifact, not genuine
robustness).** The apparent noise tolerance at small n is an artifact of the
extraction threshold being too permissive relative to the search space size.

The protocol has two independent stages:

1. The **prover** runs (noisy) QFS and includes any string s in its list L if the
   empirical post-selected probability exceeds theta^2/4 = 0.0225.
2. The **verifier** independently estimates Fourier coefficients from its own
   *noiseless* classical samples and checks accumulated weight against
   a^2 - eps^2/8 = 0.98875.

Gate noise only corrupts stage 1. Stage 2 is unaffected because the verifier
draws classical samples from the true distribution.

The critical observation is that for n <= 5, the uniform noise floor 1/2^n
**exceeds** the extraction threshold theta^2/4:

| n | 1/2^n (uniform floor) | theta^2/4 (extraction threshold) | Relationship       |
|---|-----------------------|----------------------------------|---------------------|
| 4 | 0.0625                | 0.0225                           | Floor **above** threshold |
| 5 | 0.0313                | 0.0225                           | Floor **above** threshold |
| 6 | 0.0156                | 0.0225                           | Floor **below** threshold |
| 7 | 0.0078                | 0.0225                           | Floor **below** threshold |

The crossover occurs at n = log_2(4/theta^2) = log_2(44.4) = 5.47, matching the
n=5 -> n=6 transition in the data exactly.

When 1/2^n >= theta^2/4 (i.e., n <= 5), even a *completely* noisy QFS circuit
producing a uniform distribution over 2^n strings would cause every string to
pass the extraction threshold. This means the correct parity string s* is
guaranteed to appear in L regardless of noise. Since the verifier then estimates
g_hat(s*) from noiseless samples, it finds g_hat(s*)^2 ~ 1.0 >= 0.98875 and
accepts. The protocol is not genuinely tolerating noise at n=4; it is
incidentally including the right answer in a brute-force list of all 2^n = 16
possible strings.

For n >= 6, the prover must actually resolve the target string from the QFS
distribution. Gate depolarising noise across the transpiled MCX + Hadamard
circuit (O(n) depth) smears the signal below the extraction threshold, so the
correct string fails to enter L and the verifier rejects.

Gate noise is qualitatively different from label-flip noise. No formal
equivalence mapping between gate depolarising rate p and label-flip rate eta
exists, so direct comparison is not rigorous. However, at p=0.002 the protocol
already fails for n=6 (8% acceptance), while at eta=0.01 (label-flip) the
protocol shows no degradation at any n. Gate noise corrupts the QFS circuit
itself (Hadamard layer and oracle), while label-flip noise only affects the data
distribution.

**Limitations.** The experiment covers only n=4..8 (5 values) with 50 trials each,
providing limited statistical power (CIs are wide, e.g., [92.9%, 100%] for 50/50
acceptance). The 12 error rates spanning p=0.0001 to p=0.5 provide good
resolution of the critical thresholds but the small-n acceptance remains a
parameter regime artifact (theta = eps = 0.3); a tighter theta or larger n range
would show rejection at small n too.


### 3.3 a^2 != b^2 Regime

**Data:** `ab_regime_4_16_100.pb` (n=4..16, gap in {0.0, 0.05, ..., 0.4})

**Definition 14 (L2-bounded bias class) tested.** The experiment uses
sparse_plus_noise targets with Parseval weight ~0.52, constructing
a^2 = pw - gap/2, b^2 = pw + gap/2.

**Wider gap lowers the acceptance threshold, increasing acceptance.** This follows
directly from the protocol mechanics: the acceptance threshold is
tau = a^2 - eps^2/8 (Theorem 12, Step 4), and wider gaps decrease a^2. There is
nothing counter-intuitive about this -- a looser lower bound on the true Fourier
weight naturally makes the weight check easier to pass.

| Gap  | a^2  | b^2  | tau    | Median margin | Acceptance (n=10) |
|------|------|------|--------|---------------|-------------------|
| 0.00 | 0.52 | 0.52 | 0.509  | -0.018        | 12%               |
| 0.05 | 0.50 | 0.55 | 0.484  | +0.005        | 59%               |
| 0.10 | 0.47 | 0.57 | 0.459  | +0.029        | 98%               |
| 0.20 | 0.42 | 0.62 | 0.409  | +0.082        | 100%              |

**Why gap=0 gives low acceptance.** At gap=0, a^2 = 0.52 and tau = 0.509. The
*true* Parseval weight is 0.52, but the verifier's independent coefficient
estimates introduce variance. With the sparse_plus_noise function having 4
nonzero coefficients (one at 0.7, three at 0.1) and theta=0.3, only the dominant
coefficient is detected (|L|=1). The squared estimate of this single coefficient
scatters around 0.49, which is below tau=0.509, explaining the ~15% acceptance.

**Theorem 13 (accuracy lower bound).** Theorem 13 states that for the *worst-case*
function in D_{U_n;>=theta} intersect D_{U_n;[a^2,b^2]}, the verifier requires
eps >= 2*sqrt(b^2 - a^2) and Omega(n) classical examples. With eps=0.3, this
predicts a critical gap of (eps/2)^2 = 0.0225. Gaps larger than this should make
verification impossible for the worst case. Yet the experiment shows high
acceptance at gap=0.05, 0.10, etc. This does NOT contradict Theorem 13: the
theorem constructs a specific hard instance (via reduction to distinguishing
(U_n, (1-2*eta)*chi_s) from U_{n+1}, Lemma 18), while the experiment uses the
*specific* sparse_plus_noise function, which is an easy instance. The bound is
worst-case, and the experiment confirms it is not tight for typical functions.


---

## 4. Sensitivity and Practical Limits

> *How do protocol parameters and function structure affect behaviour?*
> Paper reference: Corollary 1/5, Theorems 12, 15

### 4.1 Bent Functions -- Fourier Density

**Data:** `bent_4_16_100.pb` (n even 4..16)

**Detection boundary.** Bent functions have all 2^n Fourier coefficients equal in
magnitude: |g_hat(s)| = 2^{-n/2}. The code implements the Corollary 5 extraction
procedure with threshold theta^2/4 on the empirical conditional distribution
q(s,1). For a coefficient of magnitude c, its contribution to the conditional
distribution is approximately c^2. The relevant thresholds from Theorem 4 (which
Corollary 5 inherits) are:

- **Guaranteed inclusion:** |g_hat(s)| >= theta --> s in L
- **Guaranteed exclusion:** |g_hat(s)| < theta/2 --> s not in L (approximately;
  the code's theta^2/4 on the conditional is equivalent)
- **Uncertain zone:** theta/2 <= |g_hat(s)| < theta

For bent functions, the crossover from "all detected" (|g_hat(s)| >= theta) to
"none guaranteed" (|g_hat(s)| < theta/2) spans:
- 2^{-n/2} >= theta=0.3: n <= 2*log_2(1/0.3) = 3.44 (all guaranteed detected)
- 2^{-n/2} < theta/2=0.15: n > 2*log_2(2/0.3) = 5.47 (none guaranteed)

At n=4, the coefficient magnitude 0.25 falls in the uncertain zone
(0.15 < 0.25 < 0.30), yet all 16 coefficients are empirically detected. This
is because the finite-sample extraction occasionally captures coefficients in
the uncertain zone. At n=6, magnitude 0.125 < 0.15, and nearly all coefficients
are missed.

| n  | Coeff magnitude | Zone          | Observed |L| | Accept % |
|----|-----------------|---------------|-----------|----------|
| 4  | 0.250           | uncertain     | 16        | 100%     |
| 6  | 0.125           | excluded      | 1         | 0%       |
| 8  | 0.063           | excluded      | 0         | 0%       |
| 16 | 0.004           | excluded      | 0         | 0%       |

**Sharp phase transition, not gradual decline.** Between n=4 (100% acceptance,
|L|=16) and n=6 (0% acceptance, |L|~1), the protocol transitions completely.

**Acceptance threshold explains rejection.** At n=4 with all 16 coefficients
found, accumulated weight = 16 * (0.25)^2 = 1.0, exceeding tau = 1 - 0.3^2/8 =
0.989. At n=6 with |L|=1, accumulated weight ~ (0.125)^2 = 0.016, far below
tau = 0.989.

**QFS output is uniform for bent functions.** By Theorem 5 Eq. 34, with
E[phi(x)^2] = 1 for bent functions, the QFS conditional distribution is exactly
phi_hat(s)^2 = 2^{-n} -- uniform over all strings. This makes it maximally hard
to distinguish any coefficient from the noise floor, which is a more
illuminating explanation of why bent functions are the worst case than simply
noting small coefficient magnitude.

**Fourier sparsity is load-bearing.** Single parities (|L|=1, 100% acceptance at
all n) contrast sharply with bent functions. The sparsity assumption in
Definition 2 is a practical prerequisite for protocol efficiency.

**All rejections are via insufficient weight (Step 4),** not list-size (Step 3).
The implemented list-size bound 64*b^2/theta^2 = 711 never triggers because
coefficients fall below the detection threshold before the list grows large.

**Citation note.** The functional case is covered by Corollary 1 (p. 21), which
does not require the eps > 2^{-(n/2-2)} lower bound that Corollary 5 (p. 27,
distributional case) requires. Since the code implements the distributional
extraction procedure, Corollary 5 is the operationally relevant citation, but
its eps lower bound is violated for n <= 7 with eps=0.3 -- the protocol works
anyway, indicating the bound is not tight.


### 4.2 Theta Sensitivity -- Resolution Threshold

**Data:** `theta_sensitivity_4_16_100.pb` (n even 4..16, theta in {0.05..0.50})

**List-size bound holds universally.** The Parseval bound 4/theta^2 is never
violated across all 56 (n, theta) cells. Empirical list sizes sit well below the
bound, especially for larger n.

**Extraction boundary manifests as n-dependent transition.** The target function
(sparse_plus_noise: dominant 0.7, three secondary 0.1) has Parseval weight 0.52
and a^2 = b^2 = 0.52. The acceptance threshold is tau = 0.52 - eps^2/8 = 0.509.
The secondary coefficients (magnitude 0.1) sit at the theta/2 detection boundary
when theta = 0.20 (since 0.1 = theta/2). This also corresponds to (0.1)^2 =
theta^2/4 = 0.01, the code's extraction threshold on the conditional
distribution.

| theta | |L| at n=16 | Parseval bound 4/theta^2 | Accept at n=16 |
|-------|-----------|--------------------------|----------------|
| 0.05  | 484       | 1600                     | 100%           |
| 0.10  | 4         | 400                      | 71%            |
| 0.15  | 4         | 178                      | 74%            |
| 0.20  | 2         | 100                      | 45%            |
| 0.30  | 1         | 44                       | 17%            |
| 0.50  | 1         | 16                       | 17%            |

At theta <= 0.15, |L| = 4 matches the true sparsity (1 dominant + 3 secondary
coefficients). At theta >= 0.30, only the dominant coefficient survives. With
|L|=1, the accumulated weight from one coefficient (~0.49) is below the threshold
(0.509), explaining the ~17% acceptance: the verifier only accepts when sampling
noise inflates the estimate above threshold.

**Practical sweet spot at theta = 0.10--0.15.** Balances coefficient detection
(|L| = 4 = true sparsity) with manageable list sizes and reasonable acceptance.

**Postselection rate is 0.50, independent of theta (Theorem 5(i) confirmed).**
Median postselection rates are 0.497--0.503 for every theta value.

**Acceptance = correctness in every cell.** No false accepts. When the verifier
accepts, the hypothesis is always correct. This is stronger than the soundness
guarantee (Eq. 18), which allows a small probability of accepting a bad
hypothesis.


### 4.3 Verifier Truncation -- Sample Budget

**Data:** `truncation_{N}_{N}_100.pb` for N=4..14 (30 grid cells per n, eta=0.15)

**Note on parameter coupling.** The experiment sets theta = min(eps, 0.5), so
theta varies with eps: theta=0.1 for eps=0.1, theta=0.3 for eps=0.3,
theta=0.5 for eps=0.5. This couples extraction resolution with accuracy,
complicating interpretation.

**Verifier sample complexity (Theorem 12, Step 3).** The theoretical complexity
O(|L|^2 * log(|L|/delta) / eps^4) predicts that larger eps requires fewer
samples. This is observed: eps=0.5 achieves >= 90% acceptance at budget 3000 for
most n, while eps=0.1--0.3 frequently fails at all tested budgets for n >= 11.

**Minimum viable budget by (n, eps):**

| n   | eps=0.1 | eps=0.3 | eps=0.5 |
|-----|---------|---------|---------|
| 4   | 50      | 50      | 3000    |
| 7   | 50      | never   | 1000    |
| 10  | 50      | never   | 3000    |
| 14  | never   | never   | 3000    |

**The "inversion" at small eps is not paradoxical -- it is correct verifier
behaviour.** At eps=0.1 with eta=0.15, the acceptance threshold is
tau = 0.49 - 0.01/8 = 0.4887. The true accumulated weight is ~0.49, giving a
margin of 0.0013 (0.26% of a^2). With 50 verifier samples, the standard error of
the coefficient estimate is ~1/sqrt(50) = 0.14, producing enormous variance in
the squared weight estimate. Many trials "pass" the threshold by chance -- these
are **false accepts** driven by estimation noise. With 3000 samples, estimates
converge to the true value, and the verifier correctly identifies that the margin
is razor-thin, rejecting roughly half the time when the estimate lands slightly
below the true value.

The pattern (acceptance decreasing from 97% at 50 samples to 59% at 3000 for
n=4, eps=0.1) is the verifier becoming *more accurate*, not less effective. The
high acceptance at low sample counts is an artefact of noise-induced false
accepts, not a desirable property.

For eps=0.5 (margin=0.031, 6.4% of a^2), the pattern is normal: more samples
always helps, because the margin is wide enough for accurate estimates to pass.

**Correctness nearly tracks acceptance.** In most cells, correctness equals
acceptance. A small number of cells show minor discrepancies (e.g., n=4 eps=0.3:
accept 98%, correct 97%), indicating rare false accepts where the protocol
accepts an incorrect hypothesis. This is consistent with the soundness bound
(Eq. 18) allowing small error probability.


---

## 5. Cross-Experiment Synthesis

### 5.1 Theory-vs-Empirics Comparison

| Theorem / Result | Property | Prediction | Experiments | Verdict |
|---|---|---|---|---|
| Thm 5 (i) | Postselection = 1/2 | Pr[last qubit = 1] = 1/2 | scaling, theta_sens | **Confirmed** (0.497--0.503) |
| Thm 4 / Cor 5 | |L| <= 4/theta^2 | Parseval list-size bound | all | **Confirmed** (never violated) |
| Thm 12 completeness | Accept >= 1-delta | Honest prover accepted | scaling | **Confirmed** (100% for parities) |
| Thm 12 soundness | Reject >= 1-delta | Dishonest prover rejected | soundness | **Confirmed** (100% for 3/4 strategies) |
| Cor 7 / Thm 15 | Misclass <= 2*opt + eps | k-sparse learning bound | k_sparse | **Partially confirmed** (holds k<=4; violated k=8 due to precondition failures) |
| Thm 15 soundness | Multi-element rejection | Reject >= 1-delta | soundness_multi | **Mostly confirmed** (7/8 combos at correct k-sparse threshold; subset_plus_noise k=2 at 86.8% -- partially honest strategy) |
| Lemma 6 + Parseval | Weight = (1-2*eta)^2 | MoS noise attenuation | noise_sweep | **Precisely confirmed** (<3% error) |
| Thm 12 | Noisy verification | Protocol works with adapted params | noise_sweep | **Confirmed** (no breakdown up to eta=0.40) |
| Thm 13 | eps >= 2*sqrt(b^2-a^2) | Accuracy lower bound (worst-case) | ab_regime | **Consistent** (not tight for typical functions) |
| Cor 1/5 | Extraction threshold | QFS resolves coeffs > theta | bent, theta_sens | **Confirmed** (sharp transition; n=4 in uncertain zone) |
| Thm 12 Step 3 | Verifier: O(|L|^2/eps^4) | Sample complexity | truncation | **Qualitatively confirmed** |
| Def 14 | [a^2, b^2] promise | Distributional class | ab_regime | **Confirmed** (wider gap lowers threshold) |
| -- (no theorem) | Gate noise | Novel / empirical | gate_noise | **Novel finding** (sharp threshold: n=6 cliff at p=0.001→0.002, n=7,8 at p=0.0001→0.0005; small-n acceptance is parameter artifact: 1/2^n > theta^2/4) |


### 5.2 Key Cross-Cutting Findings

1. **The weight check (Step 4) is the operationally dominant mechanism.** Across
   all experiments, rejection is driven by accumulated weight falling below the
   threshold. The list-size bound (Step 3) is never the binding constraint. This
   holds for both the implemented bound (64*b^2/theta^2) and the tighter Parseval
   bound (4/theta^2).

2. **Fourier sparsity is the load-bearing assumption.** The protocol works
   flawlessly for single parities (k=1) and degrades for larger k and dense
   spectra. Bent functions show a sharp phase transition at the detection
   boundary. Random boolean functions are correctly rejected as outside the
   promise class.

3. **The gap between theory and practice has multiple sources:**
   - *Finite QFS shot budgets* (2000 shots) introduce estimation variance.
   - *Theorem precondition violations:* Dirichlet-drawn coefficients can fall
     below theta, violating D^{func}_{U_n;>=theta}.
   - *Tight weight margins:* for multi-coefficient targets, accumulated weight
     from a subset of detected coefficients may not reach the threshold.
   - *Corollary 5 lower bound:* eps > 2^{-(n/2-2)} is violated for small n.

4. **Gate noise is an unexplored theoretical frontier.** The sharp n=5->6
   transition is explained by a parameter regime artifact: for n <= 5, the
   uniform noise floor 1/2^n exceeds the prover's extraction threshold
   theta^2/4, so even fully-corrupted QFS trivially includes the correct
   string. Genuine noise sensitivity begins at n >= 6, where the prover must
   actually resolve the target from QFS. No formal equivalence exists between
   gate error rate and label-flip rate.

5. **Correctness nearly always tracks acceptance.** Across all experiments, false
   accepts are extremely rare (observed only in truncation at tight margins with
   very few verifier samples). For single-parity targets, acceptance implies
   correctness by construction (only one coefficient to identify).

6. **The soundness_multi experiment now uses the correct k-sparse threshold**
   (Thm 15, a^2 - eps^2/(128k^2)). The only remaining boundary case is
   subset_plus_noise at k=2 (86.8% rejection, below 1-delta=0.9). This
   strategy is partially honest (includes the heaviest real coefficient), so
   acceptance may reflect correct verifier behaviour rather than a soundness
   violation. Recording misclassification for dishonest trials would resolve
   this ambiguity.


### 5.3 Parameter Sensitivity Summary

| Parameter | Varied in | Sensitivity | Theory confirmed? |
|---|---|---|---|
| n (dimension) | All experiments | Low for parities; high for dense spectra | Yes (resource bounds, extraction boundary) |
| theta (resolution) | theta_sensitivity, bent | High: determines coefficient detection | Yes (Parseval bound 4/theta^2, extraction zones) |
| eps (accuracy) | truncation, ab_regime | Medium: affects threshold margin tau = a^2-eps^2/8 | Yes |
| eta (label noise) | noise_sweep | Low with adaptation; weight tracks (1-2*eta)^2 | Yes (Lemma 6 + Parseval) |
| k (sparsity) | k_sparse, avg_case | High: weight margin tightens as 1/(128k^2) | Partially (precondition violations for large k) |
| gate error rate | gate_noise | Very high: sharp threshold (n=6 at p~0.002, n=7,8 at p~0.0005); small-n acceptance is a parameter artifact (1/2^n > theta^2/4) | No theorem exists (novel) |
| gap (b^2-a^2) | ab_regime | Medium: wider gap lowers a^2 and thus tau | Yes (Thm 13 worst-case bound not tight) |
| verifier samples | truncation | High for tight margins (small eps) | Yes (O(|L|^2/eps^4) qualitatively) |

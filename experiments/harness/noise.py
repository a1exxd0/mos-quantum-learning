"""Experiment 4: Noise sweep."""

import time
from typing import Optional

import numpy as np
from numpy.random import default_rng

from experiments.harness.phi import make_random_parity
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_noise_sweep_experiment(
    n_range: range = range(4, 7),
    noise_rates: Optional[list[float]] = None,
    num_trials: int = 20,
    epsilon: float = 0.3,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Noise sweep: verification under increasing label-flip noise.

    For each :math:`n` in *n_range* and each noise rate :math:`\eta`,
    the MoS state is constructed from the effective label probabilities
    (Definition 5(iii)):

    .. math::

        \varphi_{\mathrm{eff}}(x) = (1 - 2\eta)\,\varphi(x) + \eta

    The effective Fourier coefficient becomes
    :math:`\hat{\tilde\phi}_{\mathrm{eff}}(s) = (1 - 2\eta)\,
    \hat{\tilde\phi}(s)`, and the distribution class promise is
    :math:`a^2 = b^2 = (1 - 2\eta)^2`.

    As :math:`\eta \to 0.5`, the signal :math:`(1 - 2\eta) \to 0` and
    the protocol should eventually fail.  The experiment measures the
    empirical acceptance and correctness rates as functions of
    :math:`\eta`, testing the noise-robust verification results of §6.2.

    The Fourier resolution threshold :math:`\vartheta` is held
    **fixed** at :math:`\vartheta = \varepsilon` across the entire
    sweep (audit fix MAJOR-3 in ``audit/noise_sweep.md``).  Previously
    :math:`\vartheta` was adapted per noise level
    (:math:`\vartheta = \min(\varepsilon,\, 0.9 \cdot (1 - 2\eta))`),
    which silently varied two parameters along the same axis and
    confounded the interpretation of any non-monotonicity in the
    acceptance curve.

    .. note::

       **Audit fixes** (``audit/noise_sweep.md``):

       - **MAJOR-1:** the :math:`\eta` range now extends to
         :math:`\{0.42, 0.44, 0.46, 0.48\}` so the sweep crosses the
         theoretical breakdown :math:`\eta_{\max} \approx 0.4470`
         (set by :math:`(1 - 2\eta)^2 = \varepsilon^2/8`).  Previously
         the sweep stopped at :math:`\eta = 0.40`, never entering the
         failure regime.
       - **MAJOR-2** (not fixed in this revision): the headline
         acceptance dip in :math:`\eta \in [0.05, 0.30]` is dominated
         by squared-estimator variance against the slack
         :math:`\varepsilon^2/8 = 0.01125`, not Lemma 6 attenuation.
         For a sharper acceptance figure ``classical_samples_verifier``
         could be bumped to :math:`\sim 30000`; the cleanest empirical
         confirmation of Lemma 6 is already
         ``fourier_weight_attenuation.png``.
       - **MAJOR-3:** :math:`\vartheta` is now held fixed across the
         sweep (see above).
       - The hard-coded ``qfs_shots=2000``,
         ``classical_samples_prover=1000``,
         ``classical_samples_verifier=3000`` are still below the
         analytic Hoeffding budget; these are documented limitations
         tracked in ``audit/FOLLOW_UPS.md``.

       The on-disk ``results/noise_sweep_*.pb`` was generated under
       the old configuration and is invalid until re-run.

    Parameters
    ----------
    n_range : range
        Range of :math:`n` values to sweep.
    noise_rates : list[float] or None
        Values of :math:`\eta` to sweep.
        Default: ``[0.0, 0.05, 0.1, ..., 0.4]``.
    num_trials : int
        Trials per :math:`(n, \eta)` cell.
    epsilon : float
        Accuracy parameter :math:`\varepsilon`.
    qfs_shots : int
        QFS copy budget per trial.
    classical_samples_prover : int
        Classical samples for the prover.
    classical_samples_verifier : int
        Classical samples for the verifier.
    base_seed : int
        Base random seed.
    max_workers : int
        Number of parallel worker processes.

    Returns
    -------
    ExperimentResult
    """
    if noise_rates is None:
        # Audit fix MAJOR-1 (audit/noise_sweep.md): extended past
        # eta_max = (1 - eps/(2*sqrt(2)))/2 ~= 0.4470 for eps=0.3 so the
        # sweep crosses the theoretical breakdown.
        noise_rates = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
            0.42, 0.44, 0.46, 0.48,
        ]

    print(
        f"=== Noise Sweep: n in {list(n_range)}, eta in {noise_rates}, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for eta in noise_rates:
            effective_coeff = 1.0 - 2.0 * eta
            a_sq = effective_coeff**2
            # Audit fix MAJOR-3 (audit/noise_sweep.md): hold theta fixed
            # at epsilon across the entire eta sweep.  Previously theta
            # was adapted as ``min(epsilon, 0.9*(1-2*eta))``, which
            # silently varied a second parameter along the eta axis.
            theta = epsilon

            for _ in range(num_trials):
                seed = int(rng.integers(0, 2**31))
                trial_rng = default_rng(seed)
                phi, target_s = make_random_parity(n, trial_rng)
                specs.append(
                    TrialSpec(
                        n=n,
                        phi=phi,
                        noise_rate=eta,
                        target_s=target_s,
                        epsilon=epsilon,
                        delta=0.1,
                        theta=theta,
                        a_sq=a_sq,
                        b_sq=a_sq,
                        qfs_shots=qfs_shots,
                        classical_samples_prover=classical_samples_prover,
                        classical_samples_verifier=classical_samples_verifier,
                        seed=seed,
                        phi_description=f"noisy_parity_eta={eta}_s={target_s}",
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="noise",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="noise_sweep",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "noise_rates": noise_rates,
            "num_trials": num_trials,
            "epsilon": epsilon,
        },
    )

    # Print per-eta summary
    print(f"\n  {'eta':>5s} {'eff_coeff':>9s} {'accept%':>8s} {'correct%':>9s}")
    for eta in noise_rates:
        et = [t for t in trials if f"eta={eta}" in t.phi_description]
        ar = np.mean([t.accepted for t in et]) if et else 0.0
        cr = np.mean([t.hypothesis_correct for t in et]) if et else 0.0
        print(f"  {eta:5.2f} {1 - 2 * eta:9.2f} {ar:8.0%} {cr:9.0%}")

    print(f"  Wall-clock time: {wall:.1f}s")
    return result

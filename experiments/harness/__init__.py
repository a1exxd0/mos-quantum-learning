r"""
Experiment harness for the MoS verification protocol.

Runs instrumented experiments that produce the data needed for
dissertation figures and analysis, with optional process-level
parallelism via :class:`~concurrent.futures.ProcessPoolExecutor`.

Covers seven experimental directions from Caro et al. [ITCS2024]_:

1. **Scaling** (:math:`n = 4 \to 16{+}`):
   Sweep :math:`n` with Goldreich--Levin extraction, recording copy
   complexity, post-selection rate, completeness probability, and
   wall-clock time.

2. **Bent function worst-case**:
   Bent functions have maximally flat Fourier spectra
   (:math:`|\hat{\tilde\phi}(s)| = 2^{-n/2}` for all :math:`s`),
   the hardest case for heavy coefficient extraction (Corollary 5).

3. **Verifier truncation tradeoffs**:
   Vary the verifier's classical sample budget :math:`m_V` and accuracy
   parameter :math:`\varepsilon` to map the completeness/soundness
   tradeoff surface (Theorems 8, 12).  Uses a fixed :math:`n` (via
   ``--n``) because the sweep is already 2-D; adding an :math:`n` axis
   would produce a prohibitively large 3-D trial grid.

4. **Noise sweep** (:math:`n \times \eta`):
   Random :math:`\varphi` functions drawn from the noisy parity ensemble
   at varying label-flip rate :math:`\eta`, testing the effective
   coefficient regime :math:`\hat{\tilde\phi}_{\mathrm{eff}}(s) =
   (1-2\eta)\,\hat{\tilde\phi}(s)` (Definition 5(iii), §6.2).  Sweeps
   both :math:`n` and :math:`\eta`, producing a 2-D grid.

5. **Soundness verification** (:math:`n \times \text{strategy}`):
   Inject dishonest provers with adversarial strategies and measure
   empirical rejection rates against the information-theoretic soundness
   guarantee (Definition 7).  Sweeps :math:`n` against four fixed
   adversarial strategies.

6. **Average-case performance** (:math:`n \times \text{family}`):
   Test the protocol on diverse function families beyond single parities:
   :math:`k`-Fourier-sparse (Dirichlet coefficients), random Boolean
   (uniform truth tables), and sparse-plus-noise (dominant parity with
   secondary coefficients).  Probes the regime of Corollary 7
   (2-agnostic Fourier-sparse learning).

7. **Gate-level noise** (:math:`n \times p`):
   Apply depolarising noise channels to H, X, and CX gates in the QFS
   circuit, sweeping the error rate :math:`p`.  Goes beyond the
   label-flip noise model of Definition 5(iii); results are inherently
   empirical with no theoretical prediction.

All experiments write results to Protocol Buffer binary files with
per-experiment schemas (see ``experiments/proto/``).

Usage
-----
Run individual experiments::

    python -m experiments.harness scaling --n-min 4 --n-max 12 --workers 8

Run all experiments::

    python -m experiments.harness all --n-min 4 --n-max 12 --workers 4

``--n-min`` / ``--n-max`` control the :math:`n` range for all
experiments.  ``--n`` overrides the fixed dimension used by the
truncation experiment (which sweeps other axes instead of :math:`n`);
it defaults to ``--n-min`` when omitted.

Programmatic use::

    from experiments.harness import run_scaling_experiment
    results = run_scaling_experiment(
        n_range=range(4, 13), num_trials=20, max_workers=8
    )
    results.save("scaling_results.pb")

.. [ITCS2024] M.\,C. Caro, M. Hinsche, M. Ioannou, A. Nietner, and
   R. Sweke, "Classical Verification of Quantum Learning," *ITCS 2024*,
   :doi:`10.4230/LIPIcs.ITCS.2024.24`.
"""

from experiments.harness.average_case import run_average_case_experiment
from experiments.harness.bent import run_bent_experiment
from experiments.harness.gate_noise import run_gate_noise_experiment
from experiments.harness.k_sparse import run_k_sparse_experiment
from experiments.harness.noise import run_noise_sweep_experiment
from experiments.harness.phi import (
    make_bent_function,
    make_k_sparse,
    make_random_boolean,
    make_random_parity,
    make_single_parity,
    make_sparse_plus_noise,
)
from experiments.harness.results import ExperimentResult, TrialResult
from experiments.harness.scaling import run_scaling_experiment
from experiments.harness.soundness import run_soundness_experiment
from experiments.harness.truncation import run_truncation_experiment
from experiments.harness.sharding import merge_shard_files, shard_specs
from experiments.harness.worker import TrialSpec, run_trials_parallel

__all__ = [
    "ExperimentResult",
    "TrialResult",
    "TrialSpec",
    "merge_shard_files",
    "shard_specs",
    "make_bent_function",
    "make_k_sparse",
    "make_random_boolean",
    "make_random_parity",
    "make_single_parity",
    "make_sparse_plus_noise",
    "run_average_case_experiment",
    "run_bent_experiment",
    "run_gate_noise_experiment",
    "run_k_sparse_experiment",
    "run_noise_sweep_experiment",
    "run_scaling_experiment",
    "run_soundness_experiment",
    "run_trials_parallel",
    "run_truncation_experiment",
]

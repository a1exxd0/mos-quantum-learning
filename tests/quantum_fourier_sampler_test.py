r"""
Test suite for QuantumFourierSampler.

Tests are organised into five categories:

1. **Theoretical distribution** — Verify that ``theoretical_distribution()``
   matches hand-computed Theorem 5 predictions for known phi.
2. **Post-selection rate** — Theorem 5(i): the label qubit marginal is 1/2,
   regardless of phi.
3. **Empirical convergence** — Sampled distributions converge to the
   Theorem 5 prediction as shots increase (DKW / Corollary 5).
4. **Mode consistency** — Statevector, circuit, and batched modes agree.
5. **Edge cases and validation** — Degenerate phi, input errors, container
   properties.

References
----------
- Caro et al., "Classical Verification of Quantum Learning", ITCS 2024.
- Theorem 5 (Section 5.1): QFS conditional distribution.
- Theorem 5(i): label qubit marginal = 1/2.
- Lemma 2 (Section 4.1): functional QFS (special case of Theorem 5).
- Corollary 5 (Section 5.1): DKW-based spectrum approximation.
- Lemma 6 (Section 4.2): noisy MoS Fourier sampling.

Running
-------
    pytest test_quantum_fourier_sampler.py -v

Or standalone:
    python test_quantum_fourier_sampler.py
"""

import sys
import pytest
import numpy as np

sys.path.insert(0, ".")

from mos import MoSState
from mos.sampler import QuantumFourierSampler, QFSResult


# =====================================================================
# Helpers
# =====================================================================


def _make_parity_phi(n: int, s: int) -> np.ndarray:
    r"""
    Construct :math:`\phi` for a pure parity :math:`\chi_s`.

    :math:`f(x) = \bigoplus_{i \in S} x_i` (inner product mod 2),
    so :math:`\phi(x) = f(x) \in \{0,1\}` and
    :math:`\tilde\phi = (-1)^f = \chi_s`.
    """
    dim = 2**n
    phi = np.array([bin(s & x).count("1") % 2 for x in range(dim)], dtype=np.float64)
    return phi


def _make_distributional_phi(n: int, coeffs: dict[int, float]) -> np.ndarray:
    r"""
    Construct :math:`\phi` from specified Fourier coefficients.

    Given ``coeffs = {s: c_s, ...}``, builds

    .. math::

        \tilde\phi(x) = \sum_{s} c_s \chi_s(x)

    and returns :math:`\phi(x) = (1 - \tilde\phi(x))/2`.

    Raises ValueError if any :math:`\phi(x) \notin [0,1]`.
    """
    dim = 2**n
    tilde_phi = np.zeros(dim)
    for x in range(dim):
        for s, c in coeffs.items():
            parity = bin(s & x).count("1") % 2
            tilde_phi[x] += c * (1.0 - 2.0 * parity)
    phi = (1.0 - tilde_phi) / 2.0
    if not (np.all(phi >= -1e-12) and np.all(phi <= 1.0 + 1e-12)):
        raise ValueError(f"phi out of [0,1]: [{phi.min()}, {phi.max()}]")
    return np.clip(phi, 0.0, 1.0)


def _l_inf(a: np.ndarray, b: np.ndarray) -> float:
    """L-infinity distance."""
    return float(np.max(np.abs(a - b)))


# =====================================================================
# 1. Theoretical distribution (Theorem 5 formula)
# =====================================================================


class TestTheoreticalDistribution:
    """Verify theoretical_distribution() against hand computations."""

    def test_pure_parity_n3_s1(self):
        r"""
        :math:`\chi_1` on 3 bits.

        :math:`\tilde\phi = \chi_1`, so :math:`\hat{\tilde\phi}(1)=1`,
        all others zero.  :math:`E[\tilde\phi^2]=1`.

        :math:`\Pr[s|b{=}1] = 0 + \delta_{s,1} = \delta_{s,1}`.
        """
        phi = _make_parity_phi(3, s=1)
        state = MoSState(n=3, phi=phi)
        qfs = QuantumFourierSampler(state)
        dist = qfs.theoretical_distribution()

        assert dist.shape == (8,)
        assert np.isclose(dist.sum(), 1.0)
        assert np.isclose(dist[1], 1.0)
        for s in [0, 2, 3, 4, 5, 6, 7]:
            assert np.isclose(dist[s], 0.0)

    def test_pure_parity_n3_s5(self):
        r"""
        :math:`\chi_5 = \chi_{101}` on 3 bits (:math:`f(x) = x_0 \oplus x_2`).

        Same structure: delta at s=5.
        """
        phi = _make_parity_phi(3, s=5)
        state = MoSState(n=3, phi=phi)
        dist = QuantumFourierSampler(state).theoretical_distribution()

        assert np.isclose(dist[5], 1.0)
        assert np.isclose(dist.sum(), 1.0)

    def test_uniform_labels(self):
        r"""
        :math:`\phi \equiv 0.5` (no signal).

        :math:`\tilde\phi \equiv 0`, all Fourier coefficients zero.
        QFS distribution is uniform: :math:`\Pr[s|b{=}1] = 1/2^n`.
        """
        state = MoSState(n=3, phi=np.full(8, 0.5))
        dist = QuantumFourierSampler(state).theoretical_distribution()

        assert np.allclose(dist, 1.0 / 8)

    def test_two_coefficients(self):
        r"""
        :math:`\tilde\phi = 0.6\chi_1 + 0.2\chi_3`.

        :math:`E[\tilde\phi^2] = 0.36 + 0.04 = 0.40`.

        :math:`\Pr[s=1|b{=}1] = (1-0.40)/8 + 0.36 = 0.435`.
        :math:`\Pr[s=3|b{=}1] = (1-0.40)/8 + 0.04 = 0.115`.
        :math:`\Pr[\text{other}|b{=}1] = (1-0.40)/8 = 0.075`.
        """
        phi = _make_distributional_phi(3, {1: 0.6, 3: 0.2})
        state = MoSState(n=3, phi=phi)
        dist = QuantumFourierSampler(state).theoretical_distribution()

        assert np.isclose(dist[1], 0.435)
        assert np.isclose(dist[3], 0.115)
        for s in [0, 2, 4, 5, 6, 7]:
            assert np.isclose(dist[s], 0.075)
        assert np.isclose(dist.sum(), 1.0)

    def test_noisy_parity_lemma6(self):
        r"""
        Noisy parity :math:`\chi_1` with :math:`\eta = 0.1` (Lemma 6).

        :math:`\hat{\tilde\phi}_{\text{eff}}(1) = 0.8`.
        :math:`E[\tilde\phi_{\text{eff}}^2] = 0.64`.

        :math:`\Pr[s=1|b{=}1] = (1-0.64)/8 + 0.64 = 0.685`.
        :math:`\Pr[\text{other}|b{=}1] = (1-0.64)/8 = 0.045`.
        """
        phi = _make_parity_phi(3, s=1)
        state = MoSState(n=3, phi=phi, noise_rate=0.1)
        dist = QuantumFourierSampler(state).theoretical_distribution()

        assert np.isclose(dist[1], 0.685)
        for s in [0, 2, 3, 4, 5, 6, 7]:
            assert np.isclose(dist[s], 0.045)

    def test_sums_to_one(self):
        """Theorem 5 distribution sums to 1 for arbitrary phi."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            phi = rng.random(16)
            state = MoSState(n=4, phi=phi, noise_rate=rng.uniform(0, 0.3))
            dist = QuantumFourierSampler(state).theoretical_distribution()
            assert np.isclose(dist.sum(), 1.0, atol=1e-12)
            assert np.all(dist >= -1e-15)

    def test_noiseless_vs_noisy_coefficients(self):
        r"""
        ``fourier_coefficient(s, effective=True)`` should equal
        :math:`(1-2\eta)` times the noiseless coefficient.
        """
        phi = _make_distributional_phi(3, {1: 0.7, 5: 0.2})
        eta = 0.15
        state = MoSState(n=3, phi=phi, noise_rate=eta)
        qfs = QuantumFourierSampler(state)

        for s in range(8):
            c_eff = qfs.fourier_coefficient(s, effective=True)
            c_raw = qfs.fourier_coefficient(s, effective=False)
            assert np.isclose(c_eff, (1 - 2 * eta) * c_raw, atol=1e-14)

    def test_constant_zero_phi(self):
        r"""
        :math:`\phi \equiv 0` (:math:`f(x) = 0` always).

        :math:`\tilde\phi \equiv 1 = \chi_0`, so
        :math:`\hat{\tilde\phi}(0) = 1`.
        QFS: delta at s=0.
        """
        state = MoSState(n=3, phi=np.zeros(8))
        dist = QuantumFourierSampler(state).theoretical_distribution()
        assert np.isclose(dist[0], 1.0)

    def test_constant_one_phi(self):
        r"""
        :math:`\phi \equiv 1` (:math:`f(x) = 1` always).

        :math:`\tilde\phi \equiv -1 = -\chi_0`, so
        :math:`\hat{\tilde\phi}(0) = -1`.
        QFS: :math:`\hat{\tilde\phi}(0)^2 = 1`, delta at s=0.
        """
        state = MoSState(n=3, phi=np.ones(8))
        dist = QuantumFourierSampler(state).theoretical_distribution()
        assert np.isclose(dist[0], 1.0)


# =====================================================================
# 2. Post-selection rate (Theorem 5(i))
# =====================================================================


class TestPostselectionRate:
    """
    Theorem 5(i): Pr[b=1] = 1/2, independent of phi.

    With enough shots, the empirical post-selection rate should
    concentrate around 0.5.
    """

    SHOTS = 3000
    TOL = 0.05  # generous for 3000 shots

    @pytest.fixture(
        params=[
            ("pure_parity", _make_parity_phi(3, s=1), 0.0),
            ("uniform", np.full(8, 0.5), 0.0),
            ("constant_0", np.zeros(8), 0.0),
            ("constant_1", np.ones(8), 0.0),
            ("biased", np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]), 0.0),
            ("noisy_parity", _make_parity_phi(3, s=3), 0.2),
            (
                "noisy_distributional",
                _make_distributional_phi(3, {2: 0.5, 7: 0.3}),
                0.1,
            ),
        ],
        ids=lambda p: p[0],
    )
    def phi_case(self, request):
        return request.param

    def test_rate_near_half(self, phi_case):
        name, phi, eta = phi_case
        state = MoSState(n=3, phi=phi, noise_rate=eta, seed=hash(name) % 2**31)
        qfs = QuantumFourierSampler(state, seed=hash(name) % 2**31)
        result = qfs.sample(shots=self.SHOTS, mode="statevector")
        assert abs(result.postselection_rate - 0.5) < self.TOL, (
            f"{name}: ps_rate={result.postselection_rate:.3f}"
        )

    def test_rate_batched_mode(self):
        """Batched mode should also give ps_rate ~ 0.5."""
        phi = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        state = MoSState(n=3, phi=phi, seed=77)
        qfs = QuantumFourierSampler(state, seed=77)
        result = qfs.sample(shots=self.SHOTS, mode="batched")
        assert abs(result.postselection_rate - 0.5) < self.TOL


# =====================================================================
# 3. Empirical convergence to Theorem 5
# =====================================================================


class TestEmpiricalConvergence:
    """
    Sampled distributions should converge to the Theorem 5 prediction.

    By the DKW theorem (used in Corollary 5), the L-inf error between
    empirical and true distributions decays as O(sqrt(log(1/delta)/M))
    where M is the number of post-selected shots.
    """

    def _run_convergence(self, state, mode, shots, atol):
        qfs = QuantumFourierSampler(state, seed=shots)
        theory = qfs.theoretical_distribution()
        result = qfs.sample(shots=shots, mode=mode)
        emp = result.empirical_distribution()
        err = _l_inf(emp, theory)
        assert err < atol, (
            f"L-inf={err:.4f} exceeds {atol} at {shots} shots "
            f"(ps_rate={result.postselection_rate:.3f})"
        )
        return err

    def test_parity_statevector_convergence(self):
        """Pure parity: empirical dist should be exact (delta function)."""
        phi = _make_parity_phi(3, s=1)
        state = MoSState(n=3, phi=phi, seed=10)
        err = self._run_convergence(state, "statevector", shots=500, atol=0.01)
        assert err < 1e-10  # should be exactly zero

    def test_distributional_statevector_convergence(self):
        """Distributional phi with two heavy coefficients."""
        phi = _make_distributional_phi(3, {1: 0.6, 3: 0.2})
        state = MoSState(n=3, phi=phi, seed=20)
        self._run_convergence(state, "statevector", shots=10000, atol=0.03)

    def test_distributional_batched_convergence(self):
        """Same distributional phi, batched mode."""
        phi = _make_distributional_phi(3, {1: 0.6, 3: 0.2})
        state = MoSState(n=3, phi=phi, seed=30)
        self._run_convergence(state, "batched", shots=10000, atol=0.03)

    def test_noisy_convergence(self):
        """Noisy parity at eta=0.15."""
        phi = _make_parity_phi(3, s=5)
        state = MoSState(n=3, phi=phi, noise_rate=0.15, seed=40)
        self._run_convergence(state, "statevector", shots=8000, atol=0.03)

    def test_monotonic_improvement(self):
        """
        L-inf error should generally decrease with more shots.

        We don't require strict monotonicity (stochastic), but the
        error at 10000 shots should be smaller than at 500 shots
        with high probability.
        """
        phi = _make_distributional_phi(3, {2: 0.5, 6: 0.3})
        state = MoSState(n=3, phi=phi, seed=50)

        errors = []
        for shots in [500, 2000, 10000]:
            qfs = QuantumFourierSampler(state, seed=shots * 7)
            theory = qfs.theoretical_distribution()
            result = qfs.sample(shots=shots, mode="statevector")
            emp = result.empirical_distribution()
            errors.append(_l_inf(emp, theory))

        # The error at the highest shot count should be below the error
        # at the lowest shot count (with very high probability)
        assert errors[-1] < errors[0], f"Error did not decrease: {errors}"

    def test_n4_convergence(self):
        """Slightly larger problem: n=4, 16-dimensional distribution."""
        phi = _make_distributional_phi(4, {3: 0.4, 10: 0.3, 15: 0.2})
        state = MoSState(n=4, phi=phi, seed=60)
        self._run_convergence(state, "statevector", shots=15000, atol=0.04)

    def test_heavy_coefficient_identification(self):
        r"""
        The QFS peak should correspond to the heaviest Fourier
        coefficient.  This is the key property exploited by the prover
        in Corollary 6 (distributional agnostic parity learning).
        """
        phi = _make_distributional_phi(4, {7: 0.6, 3: 0.2, 12: 0.1})
        state = MoSState(n=4, phi=phi, seed=70)
        qfs = QuantumFourierSampler(state, seed=70)

        theory = qfs.theoretical_distribution()
        assert np.argmax(theory) == 7, "Theoretical peak should be at s=7"

        result = qfs.sample(shots=5000, mode="statevector")
        emp = result.empirical_distribution()
        assert np.argmax(emp) == 7, "Empirical peak should be at s=7"


# =====================================================================
# 4. Mode consistency
# =====================================================================


class TestModeConsistency:
    """
    All three simulation modes implement the same physical protocol.
    Their empirical distributions should agree (up to sampling noise).
    """

    def test_parity_all_modes(self):
        """Pure parity: all modes should produce delta at s=1."""
        phi = _make_parity_phi(2, s=1)
        state = MoSState(n=2, phi=phi, seed=100)

        distributions = {}
        for mode in ["statevector", "circuit", "batched"]:
            qfs = QuantumFourierSampler(state, seed=100)
            result = qfs.sample(shots=1000, mode=mode)
            distributions[mode] = result.empirical_distribution()

        for mode, emp in distributions.items():
            assert np.argmax(emp) == 1, f"{mode}: peak not at s=1"
            assert emp[1] > 0.99, f"{mode}: Pr[s=1] = {emp[1]}"

    def test_distributional_sv_vs_batched(self):
        """
        Statevector and batched modes for distributional phi.

        Both use the internal RNG for measurement, so with independent
        seeds they should still converge to the same theoretical dist.
        """
        phi = _make_distributional_phi(3, {1: 0.5, 4: 0.3})
        state = MoSState(n=3, phi=phi, seed=200)
        theory = QuantumFourierSampler(state).theoretical_distribution()

        for mode in ["statevector", "batched"]:
            qfs = QuantumFourierSampler(state, seed=hash(mode) % 2**31)
            result = qfs.sample(shots=8000, mode=mode)
            emp = result.empirical_distribution()
            err = _l_inf(emp, theory)
            assert err < 0.04, f"{mode}: L-inf={err:.4f}"

    def test_circuit_vs_statevector_noisy(self):
        """
        Circuit and statevector modes for noisy distributional phi.

        Both should converge to the Theorem 5 / Lemma 6 prediction.
        Each circuit gets a fresh StatevectorSampler seed drawn from the
        QFS instance's RNG, ensuring independent measurement outcomes
        even for duplicate circuits.
        """
        phi = _make_distributional_phi(2, {1: 0.7, 3: 0.2})
        theory = QuantumFourierSampler(
            MoSState(n=2, phi=phi, noise_rate=0.1)
        ).theoretical_distribution()

        for mode in ["statevector", "circuit"]:
            state = MoSState(n=2, phi=phi, noise_rate=0.1, seed=300)
            qfs = QuantumFourierSampler(state, seed=hash(mode) % 2**31)
            result = qfs.sample(shots=3000, mode=mode)
            emp = result.empirical_distribution()
            err = _l_inf(emp, theory)
            assert err < 0.06, f"{mode}: L-inf={err:.4f}"
            assert abs(result.postselection_rate - 0.5) < 0.08, (
                f"{mode}: ps_rate={result.postselection_rate:.3f}"
            )

    def test_raw_counts_bitstring_lengths(self):
        """
        All modes should produce (n+1)-bit raw and n-bit postselected
        bitstrings.
        """
        n = 3
        phi = _make_distributional_phi(n, {2: 0.4})
        state = MoSState(n=n, phi=phi, seed=400)

        for mode in ["statevector", "batched"]:
            qfs = QuantumFourierSampler(state, seed=400)
            result = qfs.sample(shots=200, mode=mode)

            for bs in result.raw_counts:
                assert len(bs) == n + 1, (
                    f"{mode}: raw bitstring {bs!r} has length {len(bs)}"
                )
            for bs in result.postselected_counts:
                assert len(bs) == n, f"{mode}: ps bitstring {bs!r} has length {len(bs)}"


# =====================================================================
# 5. Edge cases, validation, and QFSResult properties
# =====================================================================


class TestEdgeCases:
    """Degenerate inputs, boundary conditions, error handling."""

    def test_n1_parity(self):
        """Smallest non-trivial case: n=1, parity chi_1 = NOT gate."""
        phi = np.array([0.0, 1.0])
        state = MoSState(n=1, phi=phi, seed=500)
        qfs = QuantumFourierSampler(state, seed=500)

        dist = qfs.theoretical_distribution()
        assert dist.shape == (2,)
        assert np.isclose(dist[1], 1.0)

        result = qfs.sample(shots=500, mode="statevector")
        emp = result.empirical_distribution()
        assert emp[1] > 0.99

    def test_n1_uniform(self):
        """n=1, uniform: Pr[s=0|b=1] = Pr[s=1|b=1] = 0.5."""
        state = MoSState(n=1, phi=np.array([0.5, 0.5]), seed=510)
        dist = QuantumFourierSampler(state).theoretical_distribution()
        assert np.allclose(dist, 0.5)

    def test_maximum_noise(self):
        r"""
        :math:`\eta = 0.5`: all Fourier coefficients damped to zero.

        :math:`\tilde\phi_{\text{eff}} \equiv 0`, so QFS is uniform.
        """
        phi = _make_parity_phi(3, s=1)
        state = MoSState(n=3, phi=phi, noise_rate=0.5, seed=520)
        qfs = QuantumFourierSampler(state, seed=520)

        dist = qfs.theoretical_distribution()
        assert np.allclose(dist, 1.0 / 8), "At eta=0.5, QFS should be uniform"

        result = qfs.sample(shots=3000, mode="statevector")
        emp = result.empirical_distribution()
        assert _l_inf(emp, dist) < 0.05

    def test_single_shot(self):
        """Sampling with exactly 1 shot should work."""
        state = MoSState(n=2, phi=np.array([0.0, 1.0, 0.0, 1.0]), seed=530)
        qfs = QuantumFourierSampler(state, seed=530)
        result = qfs.sample(shots=1, mode="statevector")

        assert result.total_shots == 1
        assert result.postselected_shots in (0, 1)
        assert sum(result.raw_counts.values()) == 1


class TestInputValidation:
    """Error handling for invalid inputs."""

    def test_shots_zero(self):
        state = MoSState(n=2, phi=np.array([0.0, 1.0, 0.0, 1.0]))
        qfs = QuantumFourierSampler(state)
        with pytest.raises(ValueError, match="shots must be >= 1"):
            qfs.sample(shots=0)

    def test_shots_negative(self):
        state = MoSState(n=2, phi=np.array([0.0, 1.0, 0.0, 1.0]))
        qfs = QuantumFourierSampler(state)
        with pytest.raises(ValueError, match="shots must be >= 1"):
            qfs.sample(shots=-5)

    def test_unknown_mode(self):
        state = MoSState(n=2, phi=np.array([0.0, 1.0, 0.0, 1.0]))
        qfs = QuantumFourierSampler(state)
        with pytest.raises(ValueError, match="Unknown mode"):
            qfs.sample(shots=10, mode="quantum_magic")

    def test_mode_case_sensitive(self):
        state = MoSState(n=2, phi=np.array([0.0, 1.0, 0.0, 1.0]))
        qfs = QuantumFourierSampler(state)
        with pytest.raises(ValueError, match="Unknown mode"):
            qfs.sample(shots=10, mode="Statevector")


class TestQFSResult:
    """QFSResult dataclass behaviour."""

    @pytest.fixture
    def result(self):
        state = MoSState(n=3, phi=_make_distributional_phi(3, {1: 0.5}), seed=600)
        qfs = QuantumFourierSampler(state, seed=600)
        return qfs.sample(shots=2000, mode="statevector")

    def test_frozen(self, result):
        """QFSResult is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            result.total_shots = 999

    def test_total_shots(self, result):
        assert result.total_shots == 2000

    def test_shot_counts_consistent(self, result):
        """Raw counts sum to total_shots."""
        assert sum(result.raw_counts.values()) == result.total_shots

    def test_postselected_shots_consistent(self, result):
        """Postselected counts sum to postselected_shots."""
        assert sum(result.postselected_counts.values()) == result.postselected_shots

    def test_postselected_leq_total(self, result):
        assert result.postselected_shots <= result.total_shots

    def test_mode_recorded(self, result):
        assert result.mode == "statevector"

    def test_n_recorded(self, result):
        assert result.n == 3

    def test_empirical_distribution_shape(self, result):
        emp = result.empirical_distribution()
        assert emp.shape == (8,)

    def test_empirical_distribution_sums_to_one(self, result):
        emp = result.empirical_distribution()
        assert np.isclose(emp.sum(), 1.0)

    def test_empirical_distribution_nonnegative(self, result):
        emp = result.empirical_distribution()
        assert np.all(emp >= 0)

    def test_empirical_distribution_empty_postselection(self):
        """If no shots survive, empirical_distribution returns zeros."""
        res = QFSResult(
            raw_counts={"000": 10},
            postselected_counts={},
            total_shots=10,
            postselected_shots=0,
            n=2,
            mode="test",
        )
        emp = res.empirical_distribution()
        assert emp.shape == (4,)
        assert np.allclose(emp, 0.0)

    def test_postselection_rate_zero_shots(self):
        """Edge case: total_shots = 0 gives rate 0."""
        res = QFSResult(
            raw_counts={},
            postselected_counts={},
            total_shots=0,
            postselected_shots=0,
            n=2,
            mode="test",
        )
        assert res.postselection_rate == 0.0


class TestPostselectionMechanics:
    """Verify the post-selection logic itself (bitstring parsing)."""

    def test_label_bit_is_leftmost(self):
        """
        The label qubit (qubit n) should be the leftmost character
        in the Qiskit bitstring convention.

        For n=2: bitstring 'abc' has a=qubit2 (label), b=qubit1, c=qubit0.
        Post-selection keeps strings starting with '1'.
        """
        state = MoSState(n=2, phi=np.array([0.0, 1.0, 0.0, 1.0]), seed=700)
        qfs = QuantumFourierSampler(state, seed=700)
        result = qfs.sample(shots=500, mode="statevector")

        for bs in result.raw_counts:
            if bs[0] == "1":
                s_bits = bs[1:]
                assert s_bits in result.postselected_counts

    def test_postselected_bitstring_length(self):
        """Post-selected bitstrings should be exactly n bits."""
        for n in [1, 2, 3, 4]:
            phi = np.full(2**n, 0.5)
            state = MoSState(n=n, phi=phi, seed=710 + n)
            qfs = QuantumFourierSampler(state, seed=710 + n)
            result = qfs.sample(shots=200, mode="statevector")
            for bs in result.postselected_counts:
                assert len(bs) == n, f"n={n}: got length {len(bs)}"


class TestReproducibility:
    """Same seed should produce identical results."""

    def test_statevector_deterministic(self):
        phi = _make_distributional_phi(3, {1: 0.4, 5: 0.3})
        state = MoSState(n=3, phi=phi, seed=800)

        qfs1 = QuantumFourierSampler(state, seed=42)
        r1 = qfs1.sample(shots=500, mode="statevector")

        # Reset MoSState RNG too
        state2 = MoSState(n=3, phi=phi, seed=800)
        qfs2 = QuantumFourierSampler(state2, seed=42)
        r2 = qfs2.sample(shots=500, mode="statevector")

        assert r1.raw_counts == r2.raw_counts

    def test_batched_deterministic(self):
        phi = _make_distributional_phi(3, {1: 0.4, 5: 0.3})
        state1 = MoSState(n=3, phi=phi, seed=810)
        state2 = MoSState(n=3, phi=phi, seed=810)

        r1 = QuantumFourierSampler(state1, seed=99).sample(500, "batched")
        r2 = QuantumFourierSampler(state2, seed=99).sample(500, "batched")

        assert r1.raw_counts == r2.raw_counts

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) give different results."""
        phi = _make_distributional_phi(3, {1: 0.4, 5: 0.3})

        state1 = MoSState(n=3, phi=phi, seed=820)
        r1 = QuantumFourierSampler(state1, seed=1).sample(500, "statevector")

        state2 = MoSState(n=3, phi=phi, seed=830)
        r2 = QuantumFourierSampler(state2, seed=2).sample(500, "statevector")

        assert r1.raw_counts != r2.raw_counts


# =====================================================================
# Standalone runner
# =====================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

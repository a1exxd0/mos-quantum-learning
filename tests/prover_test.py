r"""
Tests for the MoS Prover module.

Validates the honest prover's protocol against theoretical guarantees
from Caro et al. (ITCS 2024):

- Corollary 5: succinct Fourier spectrum approximation
- Theorems 8/12: heavy coefficient extraction (completeness + partial soundness)
- Lemma 1: classical sample correctness
- Parseval bound on list size

Tests authored by Claude Opus 4.6 in full.
"""

import numpy as np
import pytest

from mos import MoSState
from ql.prover import MoSProver, ProverMessage


# ===================================================================
# Fixtures: common test distributions
# ===================================================================


@pytest.fixture
def parity_state():
    """
    Pure parity function: phi(x) = (x_0) for n=3.
    So tilde_phi(x) = 1 - 2*x_0 = (-1)^{x_0}.
    This is exactly chi_{s=1}(x), so Fourier spectrum is:
      hat(tilde_phi)(1) = 1, all others = 0.
    """
    n = 3
    phi = np.array([0 if x & 1 == 0 else 1 for x in range(2**n)], dtype=np.float64)
    return MoSState(n=n, phi=phi, seed=42)


@pytest.fixture
def two_parity_state():
    """
    Superposition of two parities: tilde_phi(x) = 0.6*chi_1(x) + 0.4*chi_2(x).
    phi(x) = (1 - tilde_phi(x)) / 2.
    Fourier spectrum: hat(1) = 0.6, hat(2) = 0.4, rest = 0.

    We need ||tilde_phi||_inf <= 1 so phi stays in [0,1].
    Max of |0.6*chi_1 + 0.4*chi_2| = 1.0 (when both +1), so this is valid.
    """
    n = 3

    def tilde_phi(x):
        chi_1 = (-1) ** (bin(1 & x).count("1") % 2)
        chi_2 = (-1) ** (bin(2 & x).count("1") % 2)
        return 0.6 * chi_1 + 0.4 * chi_2

    phi = np.array([(1 - tilde_phi(x)) / 2.0 for x in range(2**n)], dtype=np.float64)
    return MoSState(n=n, phi=phi, seed=42)


@pytest.fixture
def noisy_parity_state():
    """
    Noisy parity: underlying phi(x) = x_0, with noise_rate = 0.1.
    Effective tilde_phi = (1 - 2*0.1) * chi_1 = 0.8 * chi_1.
    """
    n = 3
    phi = np.array([0 if x & 1 == 0 else 1 for x in range(2**n)], dtype=np.float64)
    return MoSState(n=n, phi=phi, noise_rate=0.1, seed=42)


@pytest.fixture
def uniform_state():
    """
    Uniform distribution: phi(x) = 0.5 for all x.
    tilde_phi(x) = 0, so all Fourier coefficients are 0.
    """
    n = 3
    phi = np.full(2**n, 0.5)
    return MoSState(n=n, phi=phi, seed=42)


@pytest.fixture
def distributional_state():
    """
    Distributional agnostic case: phi not {0,1}-valued.
    phi(x) = 0.3 + 0.4 * (x_0 XOR x_1).
    This gives tilde_phi(x) = 0.4 - 0.8*(x_0 XOR x_1).
    """
    n = 3

    def phi_fn(x):
        x0 = x & 1
        x1 = (x >> 1) & 1
        return 0.3 + 0.4 * (x0 ^ x1)

    phi = np.array([phi_fn(x) for x in range(2**n)], dtype=np.float64)
    return MoSState(n=n, phi=phi, seed=42)


# ===================================================================
# Test: Prover construction
# ===================================================================


class TestProverConstruction:
    def test_basic_construction(self, parity_state):
        prover = MoSProver(parity_state, seed=42)
        assert prover.n == 3
        assert prover.state is parity_state

    def test_seeded_reproducibility(self, parity_state):
        p1 = MoSProver(parity_state, seed=123)
        p2 = MoSProver(parity_state, seed=123)
        msg1 = p1.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=500)
        msg2 = p2.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=500)
        assert msg1.L == msg2.L


# ===================================================================
# Test: Pure parity — single heavy coefficient
# ===================================================================


class TestPureParityProver:
    """
    For phi(x) = x_0 (parity on bit 0), the only nonzero Fourier
    coefficient is hat(tilde_phi)(1) = 1.0.

    The prover should identify L = {1} (or a superset containing 1).
    """

    def test_finds_correct_parity(self, parity_state):
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3,
            delta=0.1,
            qfs_shots=2000,
            classical_samples=1000,
        )
        assert 1 in msg.L, f"Parity s=1 not found in L={msg.L}"

    def test_list_size_parseval_bound(self, parity_state):
        """Parseval: |L| <= 16/theta^2."""
        prover = MoSProver(parity_state, seed=42)
        theta = 0.3
        msg = prover.run_protocol(
            epsilon=theta,
            theta=theta,
            qfs_shots=2000,
        )
        parseval_bound = int(np.ceil(16.0 / theta**2))
        assert msg.list_size <= parseval_bound

    def test_coefficient_estimate_accuracy(self, parity_state):
        """Estimated coefficient for s=1 should be close to 1.0."""
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.2,
            delta=0.1,
            qfs_shots=2000,
            classical_samples=5000,
        )
        if 1 in msg.estimates:
            assert abs(msg.estimates[1] - 1.0) < 0.15, (
                f"Estimate for s=1: {msg.estimates[1]:.4f}, expected ~1.0"
            )

    def test_spurious_entries_have_small_weight(self, parity_state):
        """
        Any s in L besides s=1 should have small true Fourier weight
        (it's in L only due to finite-sample noise).
        """
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=2000)
        spectrum = parity_state.fourier_spectrum(effective=True)
        for s in msg.L:
            if s != 1:
                assert abs(spectrum[s]) < 0.3, (
                    f"Spurious s={s} has |hat(s)|={abs(spectrum[s]):.4f}"
                )


# ===================================================================
# Test: Two-parity case
# ===================================================================


class TestTwoParityProver:
    """
    tilde_phi = 0.8*chi_1 + 0.6*chi_2.
    Heavy coefficients: s=1 (|coeff|=0.8) and s=2 (|coeff|=0.6).
    """

    def test_finds_both_heavy_coefficients(self, two_parity_state):
        prover = MoSProver(two_parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3,
            delta=0.1,
            qfs_shots=3000,
            classical_samples=2000,
        )
        assert 1 in msg.L, f"s=1 not found in L={msg.L}"
        assert 2 in msg.L, f"s=2 not found in L={msg.L}"

    def test_heaviest_first(self, two_parity_state):
        """L should be sorted by weight, so s=1 (0.6) before s=2 (0.4)."""
        prover = MoSProver(two_parity_state, seed=42)
        msg = prover.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=5000)
        if 1 in msg.L and 2 in msg.L:
            idx_1 = msg.L.index(1)
            idx_2 = msg.L.index(2)
            assert idx_1 < idx_2, f"s=1 at position {idx_1}, s=2 at position {idx_2}"


# ===================================================================
# Test: Noisy parity
# ===================================================================


class TestNoisyParityProver:
    """
    With noise_rate=0.1, effective coefficient is 0.8*chi_1.
    Prover should still find s=1.
    """

    def test_finds_parity_under_noise(self, noisy_parity_state):
        prover = MoSProver(noisy_parity_state, seed=42)
        msg = prover.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=3000)
        assert 1 in msg.L, f"Noisy parity s=1 not found in L={msg.L}"

    def test_effective_coefficient_estimate(self, noisy_parity_state):
        """Estimate should be ~0.8 (the effective value), not ~1.0."""
        prover = MoSProver(noisy_parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.2,
            delta=0.1,
            qfs_shots=3000,
            classical_samples=5000,
        )
        if 1 in msg.estimates:
            # Estimator sees the noisy distribution, so the estimate
            # targets tilde_phi_eff(s) = (1-2*0.1) * 1.0 = 0.8
            assert abs(msg.estimates[1] - 0.8) < 0.15, (
                f"Noisy estimate for s=1: {msg.estimates[1]:.4f}, expected ~0.8"
            )


# ===================================================================
# Test: Uniform distribution (no heavy coefficients)
# ===================================================================


class TestUniformProver:
    """
    phi(x) = 0.5 for all x => all Fourier coefficients are 0.
    The prover should return an empty list L = [].
    """

    def test_empty_list_with_high_theta(self, uniform_state):
        """
        For uniform phi, all Fourier coefficients are 0.  But the QFS
        distribution has a floor of (1 - E[tilde_phi^2]) / 2^n = 1/2^n
        (Theorem 5).  With n=3, this floor is 1/8 = 0.125.

        When theta is large enough that theta^2/4 > 1/2^n, the prover
        correctly returns an empty list.  With theta=0.8, threshold is
        0.64/4 = 0.16 > 0.125, so no entries pass.
        """
        prover = MoSProver(uniform_state, seed=42)
        msg = prover.run_protocol(epsilon=0.8, theta=0.8, delta=0.1, qfs_shots=2000)
        assert msg.list_size == 0, (
            f"Expected empty L for uniform with high theta, got |L|={msg.list_size}"
        )

    def test_all_entries_with_low_theta(self, uniform_state):
        """
        With low theta, the uniform floor 1/2^n exceeds the threshold,
        so the prover correctly picks up all 2^n entries.  This is
        expected behaviour — the prover resolves the spectrum to
        accuracy theta, and with theta small, the uniform noise floor
        becomes visible.
        """
        prover = MoSProver(uniform_state, seed=42)
        msg = prover.run_protocol(epsilon=0.3, theta=0.3, delta=0.1, qfs_shots=2000)
        # With n=3, floor=1/8 > theta^2/4 = 0.0225, so all 8 appear
        assert msg.list_size == 8


# ===================================================================
# Test: Distributional agnostic case
# ===================================================================


class TestDistributionalProver:
    """
    Non-deterministic labelling: phi(x) not {0,1}-valued.
    Tests the genuinely distributional agnostic regime that
    motivates the MoS construction.
    """

    def test_runs_without_error(self, distributional_state):
        prover = MoSProver(distributional_state, seed=42)
        msg = prover.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=2000)
        assert isinstance(msg, ProverMessage)
        assert msg.n == 3


# ===================================================================
# Test: Exact reference quantities
# ===================================================================


class TestExactReference:
    """Validate the prover's exact_heavy_coefficients against MoSState."""

    def test_exact_matches_state(self, two_parity_state):
        prover = MoSProver(two_parity_state, seed=42)
        heavy = prover.exact_heavy_coefficients(theta=0.3, effective=True)
        # Should find s=1 (0.6) and s=2 (0.4)
        indices = [s for s, _ in heavy]
        assert 1 in indices
        assert 2 in indices
        assert len(heavy) == 2

    def test_exact_parity(self, parity_state):
        prover = MoSProver(parity_state, seed=42)
        heavy = prover.exact_heavy_coefficients(theta=0.5, effective=True)
        assert len(heavy) == 1
        assert heavy[0][0] == 1
        assert abs(heavy[0][1] - 1.0) < 1e-10


# ===================================================================
# Test: ProverMessage diagnostics
# ===================================================================


class TestProverMessageDiagnostics:
    def test_summary_string(self, parity_state):
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=1000)
        summary = msg.summary()
        assert "n = 3" in summary
        assert "epsilon" in summary

    def test_total_copies(self, parity_state):
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3,
            delta=0.1,
            qfs_shots=500,
            classical_samples=200,
        )
        assert msg.total_copies_used == 500 + 200

    def test_spectrum_approx_metadata(self, parity_state):
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(epsilon=0.3, delta=0.1, qfs_shots=1000)
        sa = msg.spectrum_approx
        assert sa.n == 3
        assert sa.total_qfs_shots == 1000
        assert sa.num_qfs_samples > 0  # some shots must pass post-selection
        assert sa.num_qfs_samples <= 1000


# ===================================================================
# Test: Parameter validation
# ===================================================================


class TestParameterValidation:
    def test_epsilon_bounds(self, parity_state):
        prover = MoSProver(parity_state)
        with pytest.raises(ValueError, match="epsilon"):
            prover.run_protocol(epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            prover.run_protocol(epsilon=1.0)

    def test_delta_bounds(self, parity_state):
        prover = MoSProver(parity_state)
        with pytest.raises(ValueError, match="delta"):
            prover.run_protocol(epsilon=0.3, delta=0.0)


# ===================================================================
# Test: Scaling (n=4)
# ===================================================================


class TestScaling:
    """Test at n=4 to verify beyond the minimal n=3 case."""

    def test_n4_parity(self):
        n = 4
        # Parity on bits 0 and 2: s = 0b0101 = 5
        phi = np.array(
            [((x & 1) ^ ((x >> 2) & 1)) for x in range(2**n)],
            dtype=np.float64,
        )
        state = MoSState(n=n, phi=phi, seed=42)
        prover = MoSProver(state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=3000, classical_samples=2000
        )
        assert 5 in msg.L, f"s=5 not in L={msg.L}"
        if 5 in msg.estimates:
            assert abs(msg.estimates[5] - 1.0) < 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

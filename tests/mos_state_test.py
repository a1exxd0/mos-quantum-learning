"""
Tests for MoSState — the Mixture-of-Superpositions quantum example state.

Organised by theoretical guarantee from Caro et al. (2306.04843):

  - Construction & validation (Definition 8 basics)
  - Sampling f ~ F_D (Definition 8)
  - State preparation |psi_{U_n, f}> (Definition 4 / §3)
  - Circuit preparation equivalence
  - Density matrix rho_D (Definition 8, Eq. 20)
  - Classical sampling / Lemma 1
  - Fourier analysis (Definition 1, Parseval)
  - Noise model (Definition 5(iii), Lemma 6)
  - Edge cases and input validation

Tests authored by Claude Opus 4.6 in full.
"""

import numpy as np
import pytest
from numpy.random import default_rng
from qiskit.quantum_info import Statevector, state_fidelity

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mos import MoSState


# ======================================================================
# Fixtures: reusable test distributions
# ======================================================================


@pytest.fixture
def deterministic_state():
    """Functional case: f(x) = x_0 (LSB parity), phi(x) = f(x).
    This is a deterministic labelling, so MoS = pure superposition example."""
    phi = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)  # n=2
    return MoSState(n=2, phi=phi, seed=42)


@pytest.fixture
def parity_biased_state():
    """Distributional case: phi biased by full parity chi_7.
    phi(x) = 0.5 + 0.25*(-1)^{|x|}, so tilde_phi(x) = -0.5 * chi_7(x).
    Single nonzero Fourier coefficient at s=7 with value -0.5."""

    def phi_parity(x):
        parity = bin(x).count("1") % 2
        return 0.5 + 0.25 * (1 - 2 * parity)

    return MoSState(n=3, phi=phi_parity, seed=42)


@pytest.fixture
def uniform_state():
    """Completely random labels: phi(x) = 0.5 for all x.
    tilde_phi = 0, all Fourier coefficients are zero."""
    phi = np.full(4, 0.5)
    return MoSState(n=2, phi=phi, seed=42)


@pytest.fixture
def noisy_state():
    """Noisy version of the parity-biased state with eta=0.1."""

    def phi_parity(x):
        parity = bin(x).count("1") % 2
        return 0.5 + 0.25 * (1 - 2 * parity)

    return MoSState(n=3, phi=phi_parity, noise_rate=0.1, seed=42)


@pytest.fixture
def multi_fourier_state():
    """Distribution with multiple nonzero Fourier coefficients.
    phi(x) = 0.5 + 0.2*(-1)^{x_0} + 0.1*(-1)^{x_1} for n=2.
    tilde_phi has coefficients at s=1 (from x_0) and s=2 (from x_1)."""
    # x=0(00): 0.5 + 0.2 + 0.1 = 0.8
    # x=1(01): 0.5 - 0.2 + 0.1 = 0.4
    # x=2(10): 0.5 + 0.2 - 0.1 = 0.6
    # x=3(11): 0.5 - 0.2 - 0.1 = 0.2
    phi = np.array([0.8, 0.4, 0.6, 0.2])
    return MoSState(n=2, phi=phi, seed=42)


# ======================================================================
# §1: Construction and input validation
# ======================================================================


class TestConstruction:
    """Test MoSState initialisation and input validation."""

    def test_basic_dimensions(self, parity_biased_state):
        s = parity_biased_state
        assert s.n == 3
        assert s.dim_x == 8
        assert s.dim_total == 16

    def test_phi_from_callable(self):
        state = MoSState(n=2, phi=lambda x: x / 3.0)
        expected = np.array([0.0, 1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(state.phi, expected)

    def test_phi_from_array(self):
        phi = np.array([0.1, 0.9, 0.5, 0.5])
        state = MoSState(n=2, phi=phi)
        np.testing.assert_array_equal(state.phi, phi)

    def test_phi_array_is_copied(self):
        """Modifying the input array should not affect the state."""
        phi = np.array([0.5, 0.5, 0.5, 0.5])
        state = MoSState(n=2, phi=phi)
        phi[0] = 0.0
        assert state.phi[0] == 0.5

    def test_invalid_n_zero(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            MoSState(n=0, phi=np.array([0.5]))

    def test_invalid_n_negative(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            MoSState(n=-1, phi=np.array([0.5]))

    def test_invalid_phi_out_of_range(self):
        with pytest.raises(ValueError, match="phi values must be in"):
            MoSState(n=1, phi=np.array([0.5, 1.5]))

    def test_invalid_phi_negative(self):
        with pytest.raises(ValueError, match="phi values must be in"):
            MoSState(n=1, phi=np.array([-0.1, 0.5]))

    def test_invalid_phi_wrong_length(self):
        with pytest.raises(ValueError, match="phi must have length"):
            MoSState(n=2, phi=np.array([0.5, 0.5]))

    def test_invalid_noise_rate_too_high(self):
        with pytest.raises(ValueError, match="noise_rate must be in"):
            MoSState(n=1, phi=np.array([0.5, 0.5]), noise_rate=0.6)

    def test_invalid_noise_rate_negative(self):
        with pytest.raises(ValueError, match="noise_rate must be in"):
            MoSState(n=1, phi=np.array([0.5, 0.5]), noise_rate=-0.1)

    def test_repr(self, parity_biased_state):
        assert "MoSState" in repr(parity_biased_state)
        assert "n=3" in repr(parity_biased_state)

    def test_repr_noisy(self, noisy_state):
        assert "noise_rate=0.1" in repr(noisy_state)


# ======================================================================
# §2: Properties — phi / tilde_phi conventions (§2.1)
# ======================================================================


class TestProperties:
    """Test phi ↔ tilde_phi conversion follows paper convention: phi = 1 - 2*varphi."""

    def test_tilde_phi_convention(self, deterministic_state):
        """tilde_phi(x) = 1 - 2*phi(x), matching §2.1 of the paper."""
        s = deterministic_state
        np.testing.assert_allclose(s.tilde_phi, 1.0 - 2.0 * s.phi)

    def test_tilde_phi_range(self, parity_biased_state):
        """tilde_phi should be in [-1, 1]."""
        tp = parity_biased_state.tilde_phi
        assert np.all(tp >= -1.0) and np.all(tp <= 1.0)

    def test_deterministic_tilde_phi_is_boolean(self, deterministic_state):
        """For deterministic f, tilde_phi should be {-1, +1}-valued."""
        tp = deterministic_state.tilde_phi
        assert np.all(np.isin(tp, [-1.0, 1.0]))

    def test_uniform_tilde_phi_is_zero(self, uniform_state):
        """When phi=0.5 everywhere, tilde_phi=0 everywhere."""
        np.testing.assert_allclose(uniform_state.tilde_phi, 0.0)

    def test_phi_effective_noiseless(self, parity_biased_state):
        """With no noise, phi_effective = phi."""
        np.testing.assert_array_equal(
            parity_biased_state.phi_effective, parity_biased_state.phi
        )

    def test_phi_effective_noisy(self, noisy_state):
        """phi_eff(x) = (1-2*eta)*phi(x) + eta."""
        eta = noisy_state.noise_rate
        expected = (1 - 2 * eta) * noisy_state.phi + eta
        np.testing.assert_allclose(noisy_state.phi_effective, expected)

    def test_tilde_phi_effective_attenuation(self, noisy_state):
        """tilde_phi_eff = (1-2*eta) * tilde_phi (Lemma 6 attenuation)."""
        eta = noisy_state.noise_rate
        expected = (1 - 2 * eta) * noisy_state.tilde_phi
        np.testing.assert_allclose(noisy_state.tilde_phi_effective, expected)

    def test_max_noise_gives_uniform(self):
        """At eta=0.5, phi_effective = 0.5 for all x (completely random)."""
        state = MoSState(n=2, phi=np.array([0.0, 1.0, 0.3, 0.7]), noise_rate=0.5)
        np.testing.assert_allclose(state.phi_effective, 0.5)


# ======================================================================
# §3: Sampling f ~ F_D (Definition 8)
# ======================================================================


class TestSampleF:
    """Test sampling Boolean functions from F_D."""

    def test_output_shape_and_dtype(self, parity_biased_state):
        f = parity_biased_state.sample_f()
        assert f.shape == (parity_biased_state.dim_x,)
        assert f.dtype == np.uint8

    def test_output_is_boolean(self, parity_biased_state):
        f = parity_biased_state.sample_f()
        assert np.all(np.isin(f, [0, 1]))

    def test_deterministic_phi_gives_fixed_f(self):
        """When phi is {0,1}-valued, F_D is a point mass on f."""
        phi = np.array([0.0, 1.0, 0.0, 1.0])
        state = MoSState(n=2, phi=phi, seed=42)
        for _ in range(10):
            f = state.sample_f()
            np.testing.assert_array_equal(f, np.array([0, 1, 0, 1]))

    def test_marginal_probabilities(self, parity_biased_state):
        """E[f(x)] should approximate phi_eff(x) over many samples."""
        rng = default_rng(123)
        num_samples = 10000
        accum = np.zeros(parity_biased_state.dim_x)
        for _ in range(num_samples):
            f = parity_biased_state.sample_f(rng)
            accum += f
        empirical = accum / num_samples
        np.testing.assert_allclose(
            empirical, parity_biased_state.phi_effective, atol=0.03
        )

    def test_independence_across_x(self, parity_biased_state):
        """f(x) and f(x') should be independent for x != x'.
        Check via correlation between f(0) and f(1)."""
        rng = default_rng(456)
        num_samples = 10000
        f0_vals = []
        f1_vals = []
        for _ in range(num_samples):
            f = parity_biased_state.sample_f(rng)
            f0_vals.append(f[0])
            f1_vals.append(f[1])
        corr = np.corrcoef(f0_vals, f1_vals)[0, 1]
        assert abs(corr) < 0.05, f"Correlation {corr} too high, expect ~0"

    def test_reproducibility_with_seed(self):
        phi = np.array([0.3, 0.7])
        s1 = MoSState(n=1, phi=phi, seed=99)
        s2 = MoSState(n=1, phi=phi, seed=99)
        np.testing.assert_array_equal(s1.sample_f(), s2.sample_f())

    def test_external_rng(self, parity_biased_state):
        """Passing an external RNG should give reproducible results."""
        rng1 = default_rng(77)
        rng2 = default_rng(77)
        f1 = parity_biased_state.sample_f(rng1)
        f2 = parity_biased_state.sample_f(rng2)
        np.testing.assert_array_equal(f1, f2)


# ======================================================================
# §4: Statevector preparation |psi_{U_n, f}> (Definition 4)
# ======================================================================


class TestStatevector:
    """Test pure state preparation for a fixed f."""

    def test_is_normalised(self, parity_biased_state):
        f = parity_biased_state.sample_f()
        sv = parity_biased_state.statevector_f(f)
        assert abs(np.linalg.norm(sv.data) - 1.0) < 1e-12

    def test_correct_dimension(self, parity_biased_state):
        f = parity_biased_state.sample_f()
        sv = parity_biased_state.statevector_f(f)
        assert len(sv.data) == parity_biased_state.dim_total

    def test_uniform_amplitudes(self, deterministic_state):
        """All nonzero amplitudes should be 1/sqrt(2^n)."""
        f = np.array([0, 1, 0, 1], dtype=np.uint8)
        sv = deterministic_state.statevector_f(f)
        expected_amp = 1.0 / np.sqrt(4)
        nonzero = sv.data[np.abs(sv.data) > 1e-12]
        np.testing.assert_allclose(np.abs(nonzero), expected_amp)

    def test_correct_nonzero_indices(self):
        """For f = [0,1,1,0], nonzero at |0,0>, |1,1>, |2,1>, |3,0>.
        Indices: 0+0*4=0, 1+1*4=5, 2+1*4=6, 3+0*4=3."""
        state = MoSState(n=2, phi=np.array([0.5, 0.5, 0.5, 0.5]))
        f = np.array([0, 1, 1, 0], dtype=np.uint8)
        sv = state.statevector_f(f)
        nonzero_idx = set(np.where(np.abs(sv.data) > 1e-12)[0])
        assert nonzero_idx == {0, 5, 6, 3}

    def test_exactly_2n_nonzero(self, parity_biased_state):
        """Should have exactly 2^n nonzero entries (one per x)."""
        f = parity_biased_state.sample_f()
        sv = parity_biased_state.statevector_f(f)
        num_nonzero = np.sum(np.abs(sv.data) > 1e-12)
        assert num_nonzero == parity_biased_state.dim_x

    def test_is_pure_state(self, parity_biased_state):
        """Statevector for a single f should be a pure state."""
        f = parity_biased_state.sample_f()
        sv = parity_biased_state.statevector_f(f)
        assert sv.is_valid()

    def test_different_f_give_different_states(self, deterministic_state):
        f1 = np.array([0, 0, 0, 0], dtype=np.uint8)
        f2 = np.array([1, 1, 1, 1], dtype=np.uint8)
        sv1 = deterministic_state.statevector_f(f1)
        sv2 = deterministic_state.statevector_f(f2)
        # These should be orthogonal (different label qubits)
        fid = state_fidelity(sv1, sv2)
        assert fid < 1e-10

    def test_n1_minimal(self):
        """Minimal case: n=1, f=[0,1]. State = (|0,0> + |1,1>)/sqrt(2)."""
        state = MoSState(n=1, phi=np.array([0.5, 0.5]))
        f = np.array([0, 1], dtype=np.uint8)
        sv = state.statevector_f(f)
        # Index 0: |x=0, b=0> = |00>
        # Index 3: |x=1, b=1> = |11> -> 1 + 1*2 = 3
        expected = np.zeros(4, dtype=np.complex128)
        expected[0] = 1.0 / np.sqrt(2)
        expected[3] = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(sv.data, expected)


# ======================================================================
# §5: Circuit preparation equivalence
# ======================================================================


class TestCircuitPreparation:
    """Test that circuit-prepared states match statevector construction."""

    def test_oracle_circuit_matches_statevector(self, deterministic_state):
        """circuit_prepare_f should produce the same state as statevector_f."""
        f = np.array([0, 1, 0, 1], dtype=np.uint8)
        sv_direct = deterministic_state.statevector_f(f)
        qc = deterministic_state.circuit_prepare_f(f)
        sv_circuit = Statevector.from_instruction(qc)
        fid = state_fidelity(sv_direct, sv_circuit)
        assert fid > 1 - 1e-10

    def test_initialize_circuit_matches_statevector(self, deterministic_state):
        """circuit_prepare_f_initialize should match statevector_f."""
        f = np.array([0, 1, 0, 1], dtype=np.uint8)
        sv_direct = deterministic_state.statevector_f(f)
        qc = deterministic_state.circuit_prepare_f_initialize(f)
        sv_circuit = Statevector.from_instruction(qc)
        fid = state_fidelity(sv_direct, sv_circuit)
        assert fid > 1 - 1e-10

    def test_both_circuits_agree(self, parity_biased_state):
        """Oracle and initialize circuits should produce the same state."""
        f = parity_biased_state.sample_f()
        qc_oracle = parity_biased_state.circuit_prepare_f(f)
        qc_init = parity_biased_state.circuit_prepare_f_initialize(f)
        sv_oracle = Statevector.from_instruction(qc_oracle)
        sv_init = Statevector.from_instruction(qc_init)
        fid = state_fidelity(sv_oracle, sv_init)
        assert fid > 1 - 1e-10

    def test_circuit_correct_qubit_count(self, parity_biased_state):
        f = parity_biased_state.sample_f()
        qc = parity_biased_state.circuit_prepare_f(f)
        assert qc.num_qubits == parity_biased_state.n + 1

    def test_all_zero_f(self):
        """Oracle for f=0 everywhere should be identity (no gates on label)."""
        state = MoSState(n=2, phi=np.array([0.5] * 4))
        f = np.array([0, 0, 0, 0], dtype=np.uint8)
        sv_direct = state.statevector_f(f)
        qc = state.circuit_prepare_f(f)
        sv_circuit = Statevector.from_instruction(qc)
        fid = state_fidelity(sv_direct, sv_circuit)
        assert fid > 1 - 1e-10

    def test_all_one_f(self):
        """Oracle for f=1 everywhere."""
        state = MoSState(n=2, phi=np.array([0.5] * 4))
        f = np.array([1, 1, 1, 1], dtype=np.uint8)
        sv_direct = state.statevector_f(f)
        qc = state.circuit_prepare_f(f)
        sv_circuit = Statevector.from_instruction(qc)
        fid = state_fidelity(sv_direct, sv_circuit)
        assert fid > 1 - 1e-10

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_random_f_multiple_n(self, n):
        """Circuit matches statevector for random f at various n."""
        phi = np.full(2**n, 0.5)
        state = MoSState(n=n, phi=phi, seed=42)
        f = state.sample_f()
        sv_direct = state.statevector_f(f)
        qc = state.circuit_prepare_f(f)
        sv_circuit = Statevector.from_instruction(qc)
        fid = state_fidelity(sv_direct, sv_circuit)
        assert fid > 1 - 1e-10


# ======================================================================
# §6: Density matrix rho_D (Definition 8, Eq. 20)
# ======================================================================


class TestDensityMatrix:
    """Test the Monte Carlo density matrix approximation."""

    def test_is_valid_density_matrix(self, parity_biased_state):
        rho = parity_biased_state.density_matrix(num_samples=200)
        assert abs(rho.trace() - 1.0) < 1e-6
        # Eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(rho.data)
        assert np.all(eigenvalues > -1e-6)

    def test_correct_dimension(self, parity_biased_state):
        rho = parity_biased_state.density_matrix(num_samples=50)
        assert rho.data.shape == (
            parity_biased_state.dim_total,
            parity_biased_state.dim_total,
        )

    def test_deterministic_phi_gives_pure_rho(self, deterministic_state):
        """When phi is {0,1}-valued, F_D is a point mass, so rho is pure."""
        rho = deterministic_state.density_matrix(num_samples=100)
        assert abs(rho.purity() - 1.0) < 0.01

    def test_distributional_phi_gives_mixed_rho(self, parity_biased_state):
        """For non-deterministic phi, rho should be genuinely mixed."""
        rho = parity_biased_state.density_matrix(num_samples=500)
        assert rho.purity() < 0.95

    def test_uniform_phi_purity(self, uniform_state):
        """For phi=0.5 (maximally mixed labels), purity should be low."""
        rho = uniform_state.density_matrix(num_samples=500)
        assert rho.purity() < 0.8

    def test_hermitian(self, parity_biased_state):
        rho = parity_biased_state.density_matrix(num_samples=100)
        np.testing.assert_allclose(rho.data, rho.data.conj().T, atol=1e-12)

    def test_reproducibility(self):
        phi = np.array([0.3, 0.7])
        s1 = MoSState(n=1, phi=phi, seed=42)
        s2 = MoSState(n=1, phi=phi, seed=42)
        rho1 = s1.density_matrix(num_samples=50)
        rho2 = s2.density_matrix(num_samples=50)
        np.testing.assert_allclose(rho1.data, rho2.data)


# ======================================================================
# §6b: Density matrix — analytic correctness
# ======================================================================


def _analytic_rho(phi: np.ndarray) -> np.ndarray:
    """
    Compute the exact analytic rho_D for given phi.

    From the MoS definition and independence of f(x) across x:

        <x,b|rho|x',b'> = (1/2^n) * P[f(x)=b] * P[f(x')=b']   if x != x'
                         = (1/2^n) * P[f(x)=b] * delta_{b,b'}    if x == x'

    where P[f(x)=1] = phi(x), P[f(x)=0] = 1 - phi(x).
    """
    dim_x = len(phi)
    dim_total = 2 * dim_x

    def prob_b(x, b):
        return phi[x] if b == 1 else (1 - phi[x])

    rho = np.zeros((dim_total, dim_total), dtype=np.float64)
    for x in range(dim_x):
        for b in range(2):
            for xp in range(dim_x):
                for bp in range(2):
                    row = x + b * dim_x
                    col = xp + bp * dim_x
                    if x == xp:
                        if b == bp:
                            rho[row, col] = (1 / dim_x) * prob_b(x, b)
                    else:
                        rho[row, col] = (1 / dim_x) * prob_b(x, b) * prob_b(xp, bp)
    return rho


class TestDensityMatrixCorrectness:
    """Test that rho_D has the correct matrix entries, not just valid properties.

    The analytic formula follows from Definition 8 and the independence
    structure of F_D:

        <x,b|rho_D|x',b'> = (1/2^n) * P_F[f(x)=b] * P_F[f(x')=b']  (x != x')
        <x,b|rho_D|x,b'>  = (1/2^n) * P_F[f(x)=b] * delta_{b,b'}

    The diagonal entries recover Lemma 1: <x,b|rho|x,b> = D(x,b).
    """

    def test_diagonal_matches_distribution_n1(self):
        """Diagonal entries <x,b|rho|x,b> = (1/2^n) * P[f(x)=b] = D(x,b)."""
        phi = np.array([0.3, 0.7])
        state = MoSState(n=1, phi=phi, seed=42)
        rho = state.density_matrix(num_samples=50000)

        # |0,0> -> idx 0: D(0,0) = 0.5 * 0.7 = 0.35
        # |1,0> -> idx 1: D(1,0) = 0.5 * 0.3 = 0.15
        # |0,1> -> idx 2: D(0,1) = 0.5 * 0.3 = 0.15
        # |1,1> -> idx 3: D(1,1) = 0.5 * 0.7 = 0.35
        expected_diag = np.array([0.35, 0.15, 0.15, 0.35])
        np.testing.assert_allclose(np.diag(rho.data).real, expected_diag, atol=0.01)

    def test_diagonal_matches_distribution_n2(self):
        """Diagonal entries for n=2."""
        phi = np.array([0.8, 0.4, 0.6, 0.2])
        state = MoSState(n=2, phi=phi, seed=42)
        rho = state.density_matrix(num_samples=50000)

        expected_diag = np.zeros(8)
        for x in range(4):
            expected_diag[x] = (1 / 4) * (1 - phi[x])  # b=0
            expected_diag[x + 4] = (1 / 4) * phi[x]  # b=1
        np.testing.assert_allclose(np.diag(rho.data).real, expected_diag, atol=0.01)

    def test_off_diagonal_x_neq_xprime(self):
        """Off-diagonal: <x,b|rho|x',b'> = (1/2^n)*P[f(x)=b]*P[f(x')=b']
        for x != x'."""
        phi = np.array([0.3, 0.7])
        state = MoSState(n=1, phi=phi, seed=42)
        rho = state.density_matrix(num_samples=50000)

        # <0,0|rho|1,0> = 0.5 * P[f(0)=0] * P[f(1)=0] = 0.5 * 0.7 * 0.3 = 0.105
        np.testing.assert_allclose(rho.data[0, 1].real, 0.105, atol=0.01)

        # <0,1|rho|1,1> = 0.5 * P[f(0)=1] * P[f(1)=1] = 0.5 * 0.3 * 0.7 = 0.105
        np.testing.assert_allclose(rho.data[2, 3].real, 0.105, atol=0.01)

        # <0,0|rho|1,1> = 0.5 * P[f(0)=0] * P[f(1)=1] = 0.5 * 0.7 * 0.7 = 0.245
        np.testing.assert_allclose(rho.data[0, 3].real, 0.245, atol=0.01)

    def test_off_diagonal_same_x_different_b_is_zero(self):
        """<x,0|rho|x,1> = 0 for all x (f(x) can't be both 0 and 1)."""
        phi = np.array([0.3, 0.7])
        state = MoSState(n=1, phi=phi, seed=42)
        rho = state.density_matrix(num_samples=50000)

        # <0,0|rho|0,1>: indices 0 and 2
        np.testing.assert_allclose(rho.data[0, 2], 0.0, atol=0.005)
        # <1,0|rho|1,1>: indices 1 and 3
        np.testing.assert_allclose(rho.data[1, 3], 0.0, atol=0.005)

    def test_full_matrix_matches_analytic_n1(self):
        """Full matrix comparison for n=1."""
        phi = np.array([0.3, 0.7])
        state = MoSState(n=1, phi=phi, seed=42)
        rho_mc = state.density_matrix(num_samples=100000)
        rho_exact = _analytic_rho(phi)
        np.testing.assert_allclose(rho_mc.data.real, rho_exact, atol=0.005)

    def test_full_matrix_matches_analytic_n2(self):
        """Full matrix comparison for n=2."""
        phi = np.array([0.8, 0.4, 0.6, 0.2])
        state = MoSState(n=2, phi=phi, seed=42)
        rho_mc = state.density_matrix(num_samples=100000)
        rho_exact = _analytic_rho(phi)
        np.testing.assert_allclose(rho_mc.data.real, rho_exact, atol=0.005)

    def test_full_matrix_matches_analytic_n3(self):
        """Full matrix comparison for n=3 (parity-biased)."""

        def phi_parity(x):
            parity = bin(x).count("1") % 2
            return 0.5 + 0.25 * (1 - 2 * parity)

        phi = np.array([phi_parity(x) for x in range(8)])
        state = MoSState(n=3, phi=phi, seed=42)
        rho_mc = state.density_matrix(num_samples=100000)
        rho_exact = _analytic_rho(phi)
        np.testing.assert_allclose(rho_mc.data.real, rho_exact, atol=0.01)

    def test_deterministic_phi_matches_pure_outer_product(self):
        """For phi in {0,1}, rho = |psi_f><psi_f| exactly."""
        phi = np.array([0.0, 1.0, 0.0, 1.0])
        state = MoSState(n=2, phi=phi, seed=42)
        rho_mc = state.density_matrix(num_samples=100)

        f = np.array([0, 1, 0, 1], dtype=np.uint8)
        sv = state.statevector_f(f)
        rho_pure = np.outer(sv.data, sv.data.conj())

        np.testing.assert_allclose(rho_mc.data, rho_pure, atol=1e-10)

    def test_imaginary_parts_are_zero(self):
        """rho_D should be real-valued for real phi (no phase structure)."""
        phi = np.array([0.3, 0.7, 0.5, 0.5])
        state = MoSState(n=2, phi=phi, seed=42)
        rho = state.density_matrix(num_samples=10000)
        np.testing.assert_allclose(rho.data.imag, 0.0, atol=1e-10)

    def test_mc_convergence(self):
        """Error should decrease as M increases (roughly as 1/sqrt(M))."""
        phi = np.array([0.3, 0.7])
        rho_exact = _analytic_rho(phi)

        errors = []
        for M in [100, 1000, 10000]:
            state = MoSState(n=1, phi=phi, seed=42)
            rho_mc = state.density_matrix(num_samples=M)
            err = np.max(np.abs(rho_mc.data.real - rho_exact))
            errors.append(err)

        # Each 10x increase in M should roughly halve the error (1/sqrt(10) ~ 0.32)
        assert errors[1] < errors[0], "More samples should reduce error"
        assert errors[2] < errors[1], "More samples should reduce error"

    def test_noisy_rho_diagonal(self):
        """With noise, diagonal should reflect phi_eff, not phi."""
        phi = np.array([0.3, 0.7])
        eta = 0.1
        state = MoSState(n=1, phi=phi, noise_rate=eta, seed=42)
        rho = state.density_matrix(num_samples=50000)

        phi_eff = state.phi_effective
        expected_diag = np.array(
            [
                0.5 * (1 - phi_eff[0]),
                0.5 * (1 - phi_eff[1]),
                0.5 * phi_eff[0],
                0.5 * phi_eff[1],
            ]
        )
        np.testing.assert_allclose(np.diag(rho.data).real, expected_diag, atol=0.01)

    def test_noisy_full_matrix(self):
        """Full matrix under noise should match analytic formula with phi_eff."""
        phi = np.array([0.3, 0.7])
        eta = 0.1
        state = MoSState(n=1, phi=phi, noise_rate=eta, seed=42)
        rho_mc = state.density_matrix(num_samples=100000)

        # The analytic rho uses phi_effective (since sample_f uses phi_eff)
        rho_exact = _analytic_rho(state.phi_effective)
        np.testing.assert_allclose(rho_mc.data.real, rho_exact, atol=0.005)


# ======================================================================
# §7: Classical sampling — Lemma 1
# ======================================================================


class TestClassicalSampling:
    """Test that comp. basis measurement of rho_D recovers (x,y) ~ D (Lemma 1)."""

    def test_single_sample_valid(self, parity_biased_state):
        x, y = parity_biased_state.sample_classical()
        assert 0 <= x < parity_biased_state.dim_x
        assert y in (0, 1)

    def test_batch_shapes(self, parity_biased_state):
        xs, ys = parity_biased_state.sample_classical_batch(100)
        assert xs.shape == (100,)
        assert ys.shape == (100,)

    def test_batch_ranges(self, parity_biased_state):
        xs, ys = parity_biased_state.sample_classical_batch(1000)
        assert np.all(xs >= 0) and np.all(xs < parity_biased_state.dim_x)
        assert np.all(np.isin(ys, [0, 1]))

    def test_uniform_marginal_over_x(self, parity_biased_state):
        """x should be uniformly distributed (known marginal U_n)."""
        xs, _ = parity_biased_state.sample_classical_batch(20000)
        counts = np.bincount(xs, minlength=parity_biased_state.dim_x)
        expected = 20000 / parity_biased_state.dim_x
        # Chi-squared-like check: each bin within 20% of expected
        assert np.all(counts > expected * 0.8)
        assert np.all(counts < expected * 1.2)

    def test_conditional_label_distribution(self, parity_biased_state):
        """E[y|x] should match phi_effective(x) (Lemma 1)."""
        xs, ys = parity_biased_state.sample_classical_batch(50000)
        for x in range(parity_biased_state.dim_x):
            mask = xs == x
            if mask.sum() > 100:
                empirical_phi = ys[mask].mean()
                true_phi = parity_biased_state.phi_effective[x]
                assert abs(empirical_phi - true_phi) < 0.05, (
                    f"x={x}: empirical={empirical_phi:.3f}, expected={true_phi:.3f}"
                )

    def test_deterministic_labels(self, deterministic_state):
        """For phi in {0,1}, labels should be deterministic."""
        xs, ys = deterministic_state.sample_classical_batch(1000)
        for i in range(len(xs)):
            expected_y = int(deterministic_state.phi[xs[i]] > 0.5)
            assert ys[i] == expected_y

    def test_noisy_conditional_distribution(self, noisy_state):
        """With noise, E[y|x] should match phi_effective, not phi."""
        xs, ys = noisy_state.sample_classical_batch(50000)
        for x in range(noisy_state.dim_x):
            mask = xs == x
            if mask.sum() > 100:
                empirical_phi = ys[mask].mean()
                true_phi_eff = noisy_state.phi_effective[x]
                assert abs(empirical_phi - true_phi_eff) < 0.05


# ======================================================================
# §8: Fourier analysis (Definition 1, Parseval)
# ======================================================================


class TestFourierAnalysis:
    """Test Fourier coefficient computation and Parseval's identity."""

    def test_parseval_identity(self, parity_biased_state):
        """sum_s hat{tilde_phi}(s)^2 = E[tilde_phi(x)^2] (Parseval)."""
        fourier_sum, expected_sq = parity_biased_state.parseval_check()
        np.testing.assert_allclose(fourier_sum, expected_sq, atol=1e-10)

    def test_parseval_deterministic(self, deterministic_state):
        """For Boolean tilde_phi, E[tilde_phi^2] = 1, so sum = 1."""
        fourier_sum, expected_sq = deterministic_state.parseval_check()
        np.testing.assert_allclose(fourier_sum, 1.0, atol=1e-10)
        np.testing.assert_allclose(expected_sq, 1.0, atol=1e-10)

    def test_parseval_uniform(self, uniform_state):
        """For phi=0.5, tilde_phi=0, so all Fourier coefficients are 0."""
        fourier_sum, expected_sq = uniform_state.parseval_check()
        np.testing.assert_allclose(fourier_sum, 0.0, atol=1e-10)
        np.testing.assert_allclose(expected_sq, 0.0, atol=1e-10)

    def test_parseval_multi_fourier(self, multi_fourier_state):
        fourier_sum, expected_sq = multi_fourier_state.parseval_check()
        np.testing.assert_allclose(fourier_sum, expected_sq, atol=1e-10)

    def test_single_parity_fourier(self, parity_biased_state):
        """phi biased by chi_7 should have one nonzero coefficient at s=7."""
        spectrum = parity_biased_state.fourier_spectrum()
        assert abs(spectrum[7]) > 0.4  # Should be -0.5
        # All others should be ~0
        for s in range(8):
            if s != 7:
                assert abs(spectrum[s]) < 1e-10

    def test_parity_coefficient_value(self, parity_biased_state):
        """hat{tilde_phi}(7) = -0.5 for our parity-biased phi."""
        coeff = parity_biased_state.fourier_coefficient(7)
        np.testing.assert_allclose(coeff, -0.5, atol=1e-10)

    def test_zero_coefficient(self, parity_biased_state):
        """All non-7 coefficients should be zero."""
        for s in [0, 1, 2, 3, 4, 5, 6]:
            coeff = parity_biased_state.fourier_coefficient(s)
            np.testing.assert_allclose(coeff, 0.0, atol=1e-10)

    def test_multi_fourier_coefficients(self, multi_fourier_state):
        """Check specific coefficient values for the multi-Fourier state.
        tilde_phi = 1 - 2*phi:
          x=0: 1 - 2*0.8 = -0.6
          x=1: 1 - 2*0.4 =  0.2
          x=2: 1 - 2*0.6 = -0.2
          x=3: 1 - 2*0.2 =  0.6
        hat{tilde_phi}(0) = mean(tilde_phi) = (-0.6+0.2-0.2+0.6)/4 = 0
        hat{tilde_phi}(1) = mean(tilde_phi * chi_1):
          chi_1 = [(-1)^0, (-1)^1, (-1)^0, (-1)^1] = [1, -1, 1, -1]
          = (-0.6*1 + 0.2*(-1) + (-0.2)*1 + 0.6*(-1))/4 = -1.6/4 = -0.4
        hat{tilde_phi}(2) = mean(tilde_phi * chi_2):
          chi_2 = [1, 1, -1, -1]
          = (-0.6 + 0.2 + 0.2 - 0.6)/4 = -0.8/4 = -0.2
        """
        s = multi_fourier_state
        np.testing.assert_allclose(s.fourier_coefficient(0), 0.0, atol=1e-10)
        np.testing.assert_allclose(s.fourier_coefficient(1), -0.4, atol=1e-10)
        np.testing.assert_allclose(s.fourier_coefficient(2), -0.2, atol=1e-10)

    def test_fourier_spectrum_shape(self, parity_biased_state):
        spectrum = parity_biased_state.fourier_spectrum()
        assert spectrum.shape == (parity_biased_state.dim_x,)

    def test_fourier_coefficient_s_zero(self):
        """hat{tilde_phi}(0) = E[tilde_phi] = mean of tilde_phi."""
        phi = np.array([0.3, 0.7, 0.5, 0.5])
        state = MoSState(n=2, phi=phi)
        coeff_0 = state.fourier_coefficient(0)
        expected = np.mean(1.0 - 2.0 * phi)
        np.testing.assert_allclose(coeff_0, expected, atol=1e-10)


# ======================================================================
# §9: Noise model (Definition 5(iii), Lemma 6)
# ======================================================================


class TestNoiseModel:
    """Test noise attenuation: hat{tilde_phi_eff}(s) = (1-2*eta) * hat{tilde_phi}(s)."""

    def test_effective_coefficient_attenuation(self, noisy_state):
        """Lemma 6: noise attenuates Fourier coefficients by (1-2*eta)."""
        eta = noisy_state.noise_rate
        clean_coeff = noisy_state.fourier_coefficient(7)
        effective_coeff = noisy_state.fourier_coefficient_effective(7)
        np.testing.assert_allclose(effective_coeff, (1 - 2 * eta) * clean_coeff)

    def test_all_coefficients_attenuated(self):
        """Every Fourier coefficient should be attenuated by the same factor."""
        phi = np.array([0.8, 0.4, 0.6, 0.2])
        eta = 0.15
        state = MoSState(n=2, phi=phi, noise_rate=eta)
        for s in range(4):
            clean = state.fourier_coefficient(s)
            effective = state.fourier_coefficient_effective(s)
            np.testing.assert_allclose(effective, (1 - 2 * eta) * clean, atol=1e-12)

    def test_zero_noise_no_attenuation(self, parity_biased_state):
        """With eta=0, effective = clean."""
        for s in range(8):
            clean = parity_biased_state.fourier_coefficient(s)
            effective = parity_biased_state.fourier_coefficient_effective(s)
            np.testing.assert_allclose(effective, clean)

    def test_max_noise_kills_coefficients(self):
        """At eta=0.5, all effective Fourier coefficients are zero."""
        phi = np.array([0.8, 0.2, 0.6, 0.4])
        state = MoSState(n=2, phi=phi, noise_rate=0.5)
        for s in range(4):
            effective = state.fourier_coefficient_effective(s)
            np.testing.assert_allclose(effective, 0.0, atol=1e-12)

    def test_noisy_sampling_marginals(self, noisy_state):
        """sample_f with noise should give E[f(x)] = phi_eff(x)."""
        rng = default_rng(789)
        num_samples = 10000
        accum = np.zeros(noisy_state.dim_x)
        for _ in range(num_samples):
            f = noisy_state.sample_f(rng)
            accum += f
        empirical = accum / num_samples
        np.testing.assert_allclose(empirical, noisy_state.phi_effective, atol=0.03)

    @pytest.mark.parametrize("eta", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    def test_attenuation_factor_parametric(self, eta):
        """Verify attenuation across a range of noise rates."""
        phi = np.array([0.9, 0.1, 0.7, 0.3])
        state = MoSState(n=2, phi=phi, noise_rate=eta)
        for s in range(4):
            clean = state.fourier_coefficient(s)
            effective = state.fourier_coefficient_effective(s)
            np.testing.assert_allclose(effective, (1 - 2 * eta) * clean, atol=1e-12)


# ======================================================================
# §10: Distributional vs functional collapse (special cases of Def 8)
# ======================================================================


class TestSpecialCases:
    """Test that MoS reduces to known special cases."""

    def test_functional_case_pure_state(self):
        """For deterministic phi in {0,1}, rho_D = |psi_f><psi_f| (pure).
        This is the collapse to Definition 4."""
        phi = np.array([1.0, 0.0, 1.0, 0.0])
        state = MoSState(n=2, phi=phi, seed=42)
        rho = state.density_matrix(num_samples=200)
        np.testing.assert_allclose(float(rho.purity().real), 1.0, atol=0.01)

    def test_functional_case_unique_f(self):
        """For deterministic phi, every sampled f should be identical."""
        phi = np.array([1.0, 0.0, 1.0, 0.0])
        state = MoSState(n=2, phi=phi)
        f_ref = state.sample_f()
        for _ in range(20):
            f = state.sample_f()
            np.testing.assert_array_equal(f, f_ref)

    def test_distributional_parseval_less_than_one(self, parity_biased_state):
        """In the distributional case, E[tilde_phi^2] < 1 (not Boolean)."""
        _, expected_sq = parity_biased_state.parseval_check()
        assert expected_sq < 1.0

    def test_n1_exhaustive(self):
        """Exhaustively check n=1 with phi=[0.3, 0.8]."""
        phi = np.array([0.3, 0.8])
        state = MoSState(n=1, phi=phi, seed=42)

        # tilde_phi = [0.4, -0.6]
        np.testing.assert_allclose(state.tilde_phi, [0.4, -0.6])

        # hat{tilde_phi}(0) = mean = (0.4 + (-0.6))/2 = -0.1
        # hat{tilde_phi}(1) = mean(tilde_phi * chi_1) = (0.4*1 + (-0.6)*(-1))/2 = 0.5
        np.testing.assert_allclose(state.fourier_coefficient(0), -0.1, atol=1e-10)
        np.testing.assert_allclose(state.fourier_coefficient(1), 0.5, atol=1e-10)

        # Parseval: (-0.1)^2 + 0.5^2 = 0.01 + 0.25 = 0.26
        # E[tilde_phi^2] = (0.16 + 0.36)/2 = 0.26
        fourier_sum, expected_sq = state.parseval_check()
        np.testing.assert_allclose(fourier_sum, 0.26, atol=1e-10)
        np.testing.assert_allclose(expected_sq, 0.26, atol=1e-10)

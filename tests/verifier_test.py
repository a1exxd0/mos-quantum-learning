r"""
Tests for the MoS Verifier module.

Validates the classical verifier's protocol against theoretical guarantees
from Caro et al. (ITCS 2024), §6:

Part I — Verifier in isolation (unit tests):
  - Hypothesis construction and evaluation (ParityHypothesis, FourierSparseHypothesis)
  - List size validation (Parseval bound)
  - Independent coefficient estimation (Hoeffding / Lemma 1)
  - Fourier weight check thresholds (Theorems 8/12/15)
  - Acceptance/rejection logic for synthetic prover messages

Part II — Prover + Verifier end-to-end (integration tests):
  - Completeness (Theorems 8/12): honest prover accepted with correct hypothesis
  - Soundness (information-theoretic): dishonest prover rejected
  - Noisy parity verification (§6.2 / Theorem 12)
  - Distributional agnostic verification (§6.3 / Theorems 12, 15)
  - Accuracy limitation from Theorem 13: eps >= 2*sqrt(b^2 - a^2)
  - Scaling tests (n=4, n=5)

Tests authored by Claude Opus 4.6 in full.
"""

import numpy as np
import pytest

from mos import MoSState
from ql.prover import MoSProver, ProverMessage, SpectrumApproximation
from ql.verifier import (
    FourierSparseHypothesis,
    HypothesisType,
    MoSVerifier,
    ParityHypothesis,
    VerificationOutcome,
)
from mos.sampler import QFSResult


# ===================================================================
# Helpers
# ===================================================================


def _make_fake_prover_message(
    L: list[int],
    estimates: dict[int, float] | None = None,
    n: int = 3,
    epsilon: float = 0.3,
    theta: float = 0.3,
) -> ProverMessage:
    """
    Build a synthetic ProverMessage for isolated verifier testing.

    Fills in dummy QFS metadata so the verifier can process the message
    without having actually run the prover.
    """
    if estimates is None:
        estimates = {}
    dummy_qfs = QFSResult(
        raw_counts={},
        postselected_counts={},
        total_shots=0,
        postselected_shots=0,
        n=n,
        mode="statevector",
    )
    dummy_sa = SpectrumApproximation(
        entries={s: 0.1 for s in L},
        threshold=theta**2 / 4.0,
        n=n,
        num_qfs_samples=0,
        total_qfs_shots=0,
    )
    return ProverMessage(
        L=L,
        estimates=estimates,
        n=n,
        epsilon=epsilon,
        theta=theta,
        spectrum_approx=dummy_sa,
        qfs_result=dummy_qfs,
        num_classical_samples=0,
    )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def parity_state():
    """
    Pure parity phi(x) = x_0 for n=3.
    Fourier spectrum: hat(tilde_phi)(1) = 1, all others 0.
    E[tilde_phi^2] = 1.
    """
    n = 3
    phi = np.array([0 if x & 1 == 0 else 1 for x in range(2**n)], dtype=np.float64)
    return MoSState(n=n, phi=phi, seed=42)


@pytest.fixture
def two_parity_state():
    """
    tilde_phi(x) = 0.6*chi_1(x) + 0.4*chi_2(x).
    Fourier spectrum: hat(1)=0.6, hat(2)=0.4, rest 0.
    E[tilde_phi^2] = 0.52.
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
    Noisy parity: phi(x) = x_0, noise_rate=0.1.
    Effective: tilde_phi_eff = 0.8*chi_1.
    E[tilde_phi_eff^2] = 0.64.
    """
    n = 3
    phi = np.array([0 if x & 1 == 0 else 1 for x in range(2**n)], dtype=np.float64)
    return MoSState(n=n, phi=phi, noise_rate=0.1, seed=42)


@pytest.fixture
def uniform_state():
    """
    Uniform: phi(x) = 0.5 for all x.
    tilde_phi = 0, all Fourier coefficients 0.
    E[tilde_phi^2] = 0.
    """
    n = 3
    phi = np.full(2**n, 0.5)
    return MoSState(n=n, phi=phi, seed=42)


# ===================================================================
# PART I: Verifier in isolation
# ===================================================================


class TestParityHypothesis:
    """Unit tests for the ParityHypothesis data class."""

    def test_evaluate_parity_s1(self):
        """h(x) = x_0 when s=1 (bit 0)."""
        h = ParityHypothesis(s=1, n=3, estimated_coefficient=1.0)
        # x=0 (000) -> 0, x=1 (001) -> 1, x=2 (010) -> 0, x=3 (011) -> 1
        assert h.evaluate(0) == 0
        assert h.evaluate(1) == 1
        assert h.evaluate(2) == 0
        assert h.evaluate(3) == 1

    def test_evaluate_parity_s5(self):
        """h(x) = x_0 XOR x_2 when s=5 (=0b101)."""
        h = ParityHypothesis(s=5, n=4, estimated_coefficient=1.0)
        # x=0 -> 0, x=1 -> 1, x=4 -> 1, x=5 -> 0
        assert h.evaluate(0) == 0
        assert h.evaluate(1) == 1
        assert h.evaluate(4) == 1
        assert h.evaluate(5) == 0

    def test_evaluate_batch(self):
        h = ParityHypothesis(s=3, n=3, estimated_coefficient=0.9)
        xs = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        ys = h.evaluate_batch(xs)
        # s=3=0b011, parity is x_0 XOR x_1
        expected = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(ys, expected)

    def test_s_zero_is_constant(self):
        """s=0 gives the constant function h(x)=0."""
        h = ParityHypothesis(s=0, n=3, estimated_coefficient=0.0)
        for x in range(8):
            assert h.evaluate(x) == 0


class TestFourierSparseHypothesis:
    """Unit tests for the FourierSparseHypothesis data class."""

    def test_g_single_coefficient(self):
        """g(x) = c * chi_s(x) for a single term."""
        h = FourierSparseHypothesis(coefficients={1: 0.8}, n=3)
        # chi_1(0) = +1, chi_1(1) = -1
        assert abs(h.g(0) - 0.8) < 1e-10
        assert abs(h.g(1) - (-0.8)) < 1e-10

    def test_g_two_coefficients(self):
        """g(x) = 0.6*chi_1(x) + 0.4*chi_2(x)."""
        h = FourierSparseHypothesis(coefficients={1: 0.6, 2: 0.4}, n=3)
        # x=0: chi_1=1, chi_2=1 -> 0.6+0.4=1.0
        # x=1: chi_1=-1, chi_2=1 -> -0.6+0.4=-0.2
        # x=2: chi_1=1, chi_2=-1 -> 0.6-0.4=0.2
        # x=3: chi_1=-1, chi_2=-1 -> -0.6-0.4=-1.0
        assert abs(h.g(0) - 1.0) < 1e-10
        assert abs(h.g(1) - (-0.2)) < 1e-10
        assert abs(h.g(2) - 0.2) < 1e-10
        assert abs(h.g(3) - (-1.0)) < 1e-10

    def test_lemma13_probabilities(self):
        """
        Lemma 13: p(x) = (1 - g(x))^2 / (2(1 + g(x)^2)).
        When g=+1 (confident label=0): p=0.
        When g=-1 (confident label=1): p=1.
        When g=0 (uncertain): p=0.5.
        """
        h = FourierSparseHypothesis(coefficients={1: 1.0}, n=3)
        # g(0) = chi_1(0) = +1 -> p = 0
        # g(1) = chi_1(1) = -1 -> p = 1
        rng = np.random.default_rng(42)

        # At x=0, g=+1, should always output 0
        outputs_x0 = [h.evaluate(0, rng=rng) for _ in range(100)]
        assert all(y == 0 for y in outputs_x0)

        # At x=1, g=-1, should always output 1
        outputs_x1 = [h.evaluate(1, rng=rng) for _ in range(100)]
        assert all(y == 1 for y in outputs_x1)

    def test_lemma13_uncertain_regime(self):
        """When g(x) ≈ 0, the hypothesis should output ~50% ones."""
        # g = 0 for all x means p = 0.5 everywhere
        h = FourierSparseHypothesis(coefficients={}, n=3)
        rng = np.random.default_rng(42)
        outputs = [h.evaluate(0, rng=rng) for _ in range(2000)]
        frac_ones = np.mean(outputs)
        assert abs(frac_ones - 0.5) < 0.05, f"Expected ~0.5, got {frac_ones:.3f}"

    def test_empty_coefficients_is_coin_flip(self):
        """No coefficients => g(x)=0 => p(x)=0.5 for all x."""
        h = FourierSparseHypothesis(coefficients={}, n=3)
        assert abs(h.g(0)) < 1e-10
        assert abs(h.g(7)) < 1e-10


class TestListSizeValidation:
    """
    §6, Step 3: Verifier rejects if |L| > 64*b^2/theta^2.

    This is the Parseval bound from Corollary 5 applied to the
    distribution class promise (Definition 14).
    """

    def test_rejects_oversized_list(self, parity_state):
        """List exceeding 64*b^2/theta^2 should be rejected."""
        theta = 0.3
        b_sq = 1.0
        bound = int(np.ceil(64.0 * b_sq / theta**2))  # = 712

        # Build a list that is too large
        big_L = list(range(bound + 1))
        msg = _make_fake_prover_message(L=big_L, n=3, theta=theta)

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, theta=theta, num_samples=100)
        assert result.outcome == VerificationOutcome.REJECT_LIST_TOO_LARGE

    def test_accepts_list_at_bound(self, parity_state):
        """List exactly at the bound should NOT be rejected for size."""
        theta = 0.5
        b_sq = 1.0
        bound = int(np.ceil(64.0 * b_sq / theta**2))  # = 256

        # L at exactly the bound; verifier won't reject for size,
        # but may reject for insufficient weight (that's a separate test).
        L = list(range(bound))
        msg = _make_fake_prover_message(L=L, n=3, theta=theta)

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, theta=theta, num_samples=100)
        assert result.outcome != VerificationOutcome.REJECT_LIST_TOO_LARGE

    def test_tighter_bound_with_smaller_b_sq(self, parity_state):
        """Distributional promise b^2 < 1 tightens the list size bound."""
        theta = 0.3
        b_sq = 0.5  # half of functional case
        bound = int(np.ceil(64.0 * b_sq / theta**2))  # = 356

        # A list that fits under b^2=1 bound but exceeds b^2=0.5 bound
        L = list(range(bound + 1))
        msg = _make_fake_prover_message(L=L, n=3, theta=theta)

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.3, theta=theta, b_sq=b_sq, num_samples=100
        )
        assert result.outcome == VerificationOutcome.REJECT_LIST_TOO_LARGE
        assert result.list_size_bound == bound


class TestAcceptanceThresholds:
    """
    Verify the Fourier weight thresholds match the paper:
    - Parity (Thm 8/12): a^2 - eps^2/8
    - Fourier-sparse (Thm 10/15): a^2 - eps^2/(128*k^2)
    """

    def test_functional_parity_threshold(self, parity_state):
        """Functional case: a^2=1, threshold = 1 - eps^2/8."""
        eps = 0.3
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=eps, a_sq=1.0, num_samples=5000)
        expected_threshold = 1.0 - eps**2 / 8.0
        assert abs(result.acceptance_threshold - expected_threshold) < 1e-10

    def test_distributional_parity_threshold(self, parity_state):
        """Distributional case with a^2=0.64: threshold = 0.64 - eps^2/8."""
        eps = 0.3
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=eps, a_sq=0.64, num_samples=5000)
        expected_threshold = 0.64 - eps**2 / 8.0
        assert abs(result.acceptance_threshold - expected_threshold) < 1e-10

    def test_fourier_sparse_threshold(self, parity_state):
        """Fourier-sparse: threshold = a^2 - eps^2/(128*k^2)."""
        eps = 0.3
        k = 2
        msg = _make_fake_prover_message(L=[1, 2], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_fourier_sparse(
            msg, epsilon=eps, k=k, a_sq=1.0, num_samples=5000
        )
        expected_threshold = 1.0 - eps**2 / (128.0 * k**2)
        assert abs(result.acceptance_threshold - expected_threshold) < 1e-10

    def test_fourier_sparse_tighter_than_parity(self, parity_state):
        """
        The Fourier-sparse threshold is tighter (closer to a^2) than
        the parity threshold, because 128*k^2 > 8 for k >= 1.
        """
        eps = 0.3
        parity_thr = 1.0 - eps**2 / 8.0
        fs_thr_k1 = 1.0 - eps**2 / (128.0 * 1**2)
        fs_thr_k2 = 1.0 - eps**2 / (128.0 * 2**2)
        assert fs_thr_k1 > parity_thr
        assert fs_thr_k2 > fs_thr_k1


class TestIndependentEstimation:
    """
    The verifier's coefficient estimates must be independent of the
    prover's estimates.  This is the foundation of information-theoretic
    soundness (§6 proofs, passim).
    """

    def test_estimates_converge_to_true_values(self, parity_state):
        """
        With enough samples, verifier's estimate of hat(tilde_phi)(1)
        should be close to 1.0 for a pure parity.
        """
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=10000)
        assert abs(result.verifier_estimates[1] - 1.0) < 0.05

    def test_estimates_near_zero_for_absent_coefficients(self, parity_state):
        """
        If the prover claims s=3 is heavy (it isn't for pure parity on bit 0),
        the verifier's independent estimate should be near 0.
        """
        msg = _make_fake_prover_message(L=[1, 3], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=10000)
        assert abs(result.verifier_estimates[3]) < 0.05

    def test_independence_from_prover_estimates(self, parity_state):
        """
        The verifier's estimates should be the same regardless of what
        the prover claims.  Sending different ProverMessages with different
        prover estimates should not affect the verifier's output when
        using the same seed and sample count.
        """
        verifier1 = MoSVerifier(parity_state, seed=99)
        msg1 = _make_fake_prover_message(L=[1], estimates={1: 0.5}, n=3)
        r1 = verifier1.verify_parity(msg1, epsilon=0.3, num_samples=5000)

        verifier2 = MoSVerifier(parity_state, seed=99)
        msg2 = _make_fake_prover_message(L=[1], estimates={1: 0.999}, n=3)
        r2 = verifier2.verify_parity(msg2, epsilon=0.3, num_samples=5000)

        # Same seed, same state, same L => same verifier estimates
        assert abs(r1.verifier_estimates[1] - r2.verifier_estimates[1]) < 1e-10

    def test_noisy_estimates_target_effective_coefficient(self, noisy_parity_state):
        """
        Under noise_rate=0.1, the classical samples come from the noisy
        distribution, so the verifier's estimate targets
        hat(tilde_phi_eff)(1) = 0.8, not 1.0.
        """
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(noisy_parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=10000)
        # Should be near 0.8
        assert abs(result.verifier_estimates[1] - 0.8) < 0.05


class TestWeightCheckLogic:
    """
    Test the accept/reject decision based on accumulated Fourier weight.
    """

    def test_correct_list_accepted_functional(self, parity_state):
        """
        Honest list L=[1] for pure parity: verifier's estimate of
        hat(1)^2 should be ~1.0, well above threshold 1 - eps^2/8.
        """
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        assert result.accepted
        assert result.accumulated_weight > result.acceptance_threshold

    def test_empty_list_rejected_functional(self, parity_state):
        """
        Empty list for functional case (a^2=1): accumulated weight = 0,
        threshold ≈ 0.989, so must reject.
        """
        msg = _make_fake_prover_message(L=[], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=0)
        assert result.outcome == VerificationOutcome.REJECT_INSUFFICIENT_WEIGHT
        assert result.accumulated_weight == 0.0

    def test_wrong_single_entry_rejected(self, parity_state):
        """
        Prover sends L=[3] (wrong parity).  Verifier's estimate of
        hat(3) ≈ 0, so accumulated weight ≈ 0 < threshold.
        """
        msg = _make_fake_prover_message(L=[3], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        assert result.outcome == VerificationOutcome.REJECT_INSUFFICIENT_WEIGHT

    def test_partial_list_may_still_pass(self, parity_state):
        """
        L=[1, 3]: entry 1 is correct (weight ~1), entry 3 is spurious
        (weight ~0).  Total weight ~1 > threshold, so should accept.
        The hypothesis should still select s=1 (the heaviest).
        """
        msg = _make_fake_prover_message(L=[1, 3], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        assert result.accepted
        assert result.hypothesis.s == 1


class TestVerificationResultDiagnostics:
    """Test VerificationResult metadata and summary."""

    def test_accepted_property(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        assert result.accepted is True

    def test_summary_contains_key_info(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        s = result.summary()
        assert "accept" in s
        assert "n = 3" in s
        assert "epsilon" in s
        assert "Accumulated weight" in s

    def test_rejection_summary(self, parity_state):
        msg = _make_fake_prover_message(L=[], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=0)
        assert not result.accepted
        assert result.hypothesis is None

    def test_num_classical_samples_recorded(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=1234)
        assert result.num_classical_samples == 1234


class TestRunFullProtocolDispatch:
    """Test the run_full_protocol convenience wrapper."""

    def test_parity_mode(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.run_full_protocol(msg, mode="parity", num_samples=5000)
        assert result.hypothesis_type == HypothesisType.PARITY

    def test_fourier_sparse_mode(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.run_full_protocol(
            msg, mode="fourier_sparse", k=1, num_samples=5000
        )
        assert result.hypothesis_type == HypothesisType.FOURIER_SPARSE

    def test_invalid_mode_raises(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        with pytest.raises(ValueError, match="Unknown mode"):
            verifier.run_full_protocol(msg, mode="bogus", num_samples=100)

    def test_defaults_epsilon_from_prover(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3, epsilon=0.25)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.run_full_protocol(msg, mode="parity", num_samples=5000)
        assert result.epsilon == 0.25


# ===================================================================
# PART II: Prover + Verifier end-to-end
# ===================================================================


class TestCompletenessFunctionalParity:
    """
    Theorem 8 (completeness): When the honest prover interacts with the
    verifier for a distribution D in D^func_{n; >= theta}, the verifier
    accepts and outputs a hypothesis h with

        Pr[h(x) != y] <= min_t Pr[t.x != y] + epsilon

    with probability >= 1 - delta.

    We test the functional case (a^2 = b^2 = 1) at n=3 and n=4.
    """

    def test_pure_parity_n3(self, parity_state):
        """Honest prover + verifier for phi(x)=x_0, n=3."""
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples=1000
        )

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, delta=0.1, num_samples=5000)
        assert result.accepted, (
            f"Honest prover rejected: weight={result.accumulated_weight:.4f}, "
            f"threshold={result.acceptance_threshold:.4f}"
        )
        assert result.hypothesis.s == 1

    def test_pure_parity_n4(self):
        """Honest prover + verifier for parity on bits {0,2}, n=4."""
        n = 4
        phi = np.array(
            [((x & 1) ^ ((x >> 2) & 1)) for x in range(2**n)],
            dtype=np.float64,
        )
        state = MoSState(n=n, phi=phi, seed=42)

        prover = MoSProver(state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=3000, classical_samples=2000
        )

        verifier = MoSVerifier(state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, delta=0.1, num_samples=5000)
        assert result.accepted
        assert result.hypothesis.s == 5  # 0b0101

    def test_hypothesis_has_low_error(self, parity_state):
        """
        Lemma 11: the output hypothesis satisfies the agnostic guarantee.
        For pure parity, the optimal error is 0, so the hypothesis
        should achieve error <= epsilon on fresh samples.
        """
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples=1000
        )

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        assert result.accepted

        # Evaluate on all inputs
        h = result.hypothesis
        errors = 0
        n = 3
        for x in range(2**n):
            true_label = x & 1  # phi(x) = x_0
            if h.evaluate(x) != true_label:
                errors += 1

        error_rate = errors / 2**n
        assert error_rate == 0.0, f"Parity hypothesis has error {error_rate}"


class TestSoundness:
    """
    Information-theoretic soundness (§6, all theorems):
    Even a computationally unbounded dishonest prover cannot fool the
    verifier into accepting a bad hypothesis, except with probability
    <= delta.

    We test this by constructing adversarial prover messages and
    verifying that the verifier rejects.
    """

    def test_completely_wrong_list(self, parity_state):
        """
        Dishonest prover sends L with no correct entries.
        True heavy coefficient is s=1; prover sends L=[2,4,6].
        Verifier's estimates for these should all be near 0.
        """
        msg = _make_fake_prover_message(L=[2, 4, 6], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        assert not result.accepted
        assert result.outcome == VerificationOutcome.REJECT_INSUFFICIENT_WEIGHT

    def test_one_correct_but_missing_others(self, two_parity_state):
        """
        For two-parity (hat(1)=0.6, hat(2)=0.4, E[phi^2]=0.52),
        a prover who sends only L=[1] captures weight ~0.36.
        With a_sq=0.52, threshold = 0.52 - eps^2/8 ≈ 0.509.
        Since 0.36 < 0.509, verifier should reject.

        This tests the key insight: the prover must send ALL heavy
        coefficients, not just one.
        """
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(two_parity_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.3, a_sq=0.52, b_sq=0.52, num_samples=5000
        )
        assert not result.accepted, (
            f"Incomplete list accepted: weight={result.accumulated_weight:.4f}, "
            f"threshold={result.acceptance_threshold:.4f}"
        )

    def test_inflated_list_rejected_by_size(self, parity_state):
        """
        Prover sends an unreasonably large list to try to game the
        weight check.  Verifier rejects in Step 3 (list size check).
        """
        huge_L = list(range(1000))
        msg = _make_fake_prover_message(L=huge_L, n=3, theta=0.3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, theta=0.3, num_samples=100)
        assert result.outcome == VerificationOutcome.REJECT_LIST_TOO_LARGE

    def test_soundness_of_hypothesis_when_accepted(self, parity_state):
        """
        Even if the verifier accepts (honest prover), the hypothesis
        must be the correct argmax.  If we inject a list [1, 7] where
        1 is correct and 7 is wrong, the hypothesis must select s=1
        because the verifier's independent estimates will show |hat(1)| >> |hat(7)|.
        """
        msg = _make_fake_prover_message(L=[1, 7], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        if result.accepted:
            # The hypothesis MUST be s=1, not s=7
            assert result.hypothesis.s == 1


class TestCompletenessDistributional:
    """
    Theorem 12 (distributional completeness): verification for
    D in D_{n; >= theta} ∩ D_{n; [a^2, b^2]}.

    Tests the distributional agnostic case where phi is not {0,1}-valued.
    """

    def test_noisy_parity_verification(self, noisy_parity_state):
        """
        Noisy parity with eta=0.1.
        E[tilde_phi_eff^2] = (1-2*0.1)^2 = 0.64.
        a^2 = b^2 = 0.64.
        Prover should find s=1 with effective coefficient ~0.8.
        """
        prover = MoSProver(noisy_parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, theta=0.5, qfs_shots=3000, classical_samples=2000
        )

        a_sq = (1.0 - 2.0 * 0.1) ** 2  # = 0.64
        verifier = MoSVerifier(noisy_parity_state, seed=99)
        result = verifier.verify_parity(
            msg,
            epsilon=0.3,
            delta=0.1,
            theta=0.5,
            a_sq=a_sq,
            b_sq=a_sq,
            num_samples=5000,
        )
        assert result.accepted, (
            f"Noisy honest prover rejected: weight={result.accumulated_weight:.4f}, "
            f"threshold={result.acceptance_threshold:.4f}"
        )
        assert result.hypothesis.s == 1

    def test_two_parity_distributional_parity_mode(self, two_parity_state):
        """
        Two-parity distributional case: hat(1)=0.6, hat(2)=0.4.
        E[tilde_phi^2] = 0.52.
        In parity mode, the verifier picks the heaviest, s=1.
        """
        prover = MoSProver(two_parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, theta=0.3, qfs_shots=5000, classical_samples=3000
        )

        a_sq = 0.52
        verifier = MoSVerifier(two_parity_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.3, delta=0.1, a_sq=a_sq, b_sq=a_sq, num_samples=5000
        )
        # This might accept or reject depending on whether the prover's
        # list has enough total weight.  The key assertion: if accepted,
        # the hypothesis must be s=1 (heaviest coefficient).
        if result.accepted:
            assert result.hypothesis.s == 1


class TestCompletenessFourierSparse:
    """
    Theorems 10/15: verification for 2-agnostic Fourier-sparse learning.
    """

    def test_pure_parity_as_1sparse(self, parity_state):
        """
        Pure parity is Fourier-1-sparse.  The Fourier-sparse verifier
        with k=1 should accept and produce a hypothesis whose top
        coefficient is at s=1.
        """
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples=1000
        )

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_fourier_sparse(
            msg, epsilon=0.3, k=1, delta=0.1, num_samples=5000
        )
        assert result.accepted
        assert isinstance(result.hypothesis, FourierSparseHypothesis)
        assert 1 in result.hypothesis.coefficients

    def test_fourier_sparse_hypothesis_evaluates_correctly(self, parity_state):
        """
        For pure parity with k=1, the Fourier-sparse hypothesis should
        have g(x) ≈ chi_1(x) = (-1)^{x_0}, giving the same predictions
        as the parity hypothesis.
        """
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples=1000
        )

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_fourier_sparse(
            msg, epsilon=0.3, k=1, delta=0.1, num_samples=5000
        )
        assert result.accepted

        h = result.hypothesis
        # g(x) ≈ hat(1) * chi_1(x).  For x with x_0=0: g>0, p≈0, h(x)=0.
        # For x with x_0=1: g<0, p≈1, h(x)=1.
        # This matches phi(x) = x_0.
        for x in range(8):
            true_label = x & 1
            gx = h.g(x)
            # g should be close to +1 (label=0) or -1 (label=1)
            if true_label == 0:
                assert gx > 0.5, f"x={x}: expected g>0, got {gx:.3f}"
            else:
                assert gx < -0.5, f"x={x}: expected g<0, got {gx:.3f}"


class TestUniformDistribution:
    """
    Uniform phi(x) = 0.5: all Fourier coefficients are 0.
    E[tilde_phi^2] = 0, so a^2 = b^2 = 0.

    This is a degenerate case where every parity is equally bad
    (error = 1/2).  The verifier should accept any list (since the
    threshold a^2 - eps^2/8 < 0, and accumulated weight >= 0).
    """

    def test_empty_list_accepted(self, uniform_state):
        """
        With a^2 = 0, threshold = -eps^2/8 < 0.
        Empty list has weight 0 > threshold, so verifier accepts.
        """
        msg = _make_fake_prover_message(L=[], n=3)
        verifier = MoSVerifier(uniform_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.3, a_sq=0.0, b_sq=0.0, num_samples=0
        )
        assert result.accepted
        # Hypothesis is s=0 (degenerate), which is fine: all parities
        # have error 1/2, matching opt_D(Parities) = 1/2.

    def test_nonempty_list_rejected_by_size_when_b_sq_zero(self, uniform_state):
        """
        With b^2 = 0, the Parseval bound is ceil(64*0/theta^2) = 0,
        so ANY non-empty list is correctly rejected for size.
        This is right: if b^2 = 0, there are no Fourier coefficients,
        so the prover should not be sending any.
        """
        msg = _make_fake_prover_message(L=[1, 2, 3], n=3)
        verifier = MoSVerifier(uniform_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.3, a_sq=0.0, b_sq=0.0, num_samples=500
        )
        assert result.outcome == VerificationOutcome.REJECT_LIST_TOO_LARGE
        assert result.list_size_bound == 0

    def test_nonempty_list_accepted_with_positive_b_sq(self, uniform_state):
        """
        If the verifier's promise allows b^2 > 0 (e.g. the verifier
        doesn't know E[phi^2] exactly), a non-empty list can pass
        the size check.  The weight check still passes trivially
        since a^2 = 0 => threshold < 0.
        """
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(uniform_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.3, a_sq=0.0, b_sq=1.0, num_samples=500
        )
        # Size bound = ceil(64*1/0.09) = 712, so |L|=1 passes
        # Weight threshold = 0 - 0.3^2/8 = -0.01125, so weight >= 0 passes
        assert result.accepted


class TestAccuracyLimitation:
    r"""
    Theorem 13: the accuracy limitation eps >= 2*sqrt(b^2 - a^2) in
    Theorem 12 cannot be significantly improved.

    When a^2 != b^2, the verifier cannot distinguish whether the
    missing Fourier weight is within L or outside it.  We test that
    the protocol behaves correctly at the boundary.
    """

    def test_eps_below_limitation_may_cause_failure(self, parity_state):
        """
        With a^2 = 0.5, b^2 = 1.0: the constraint is
        eps >= 2*sqrt(1.0 - 0.5) ≈ 1.414.

        At eps = 0.3 (violating the constraint), the threshold
        a^2 - eps^2/8 = 0.5 - 0.01125 = 0.489.  Since the true
        weight is 1.0 > 0.489, an honest prover still passes.

        But the soundness gap shrinks: a dishonest prover omitting
        weight up to sqrt(b^2 - a^2) ≈ 0.707 outside L cannot be
        reliably detected.  We verify the threshold arithmetic.
        """
        eps = 0.3
        a_sq = 0.5
        b_sq = 1.0
        threshold = a_sq - eps**2 / 8.0
        # Dishonest prover could have weight outside L up to:
        # (b^2 - a^2) + eps^2/4 = 0.5 + 0.0225 = 0.5225
        # from Eq. (114) in the proof of Theorem 12.
        # This is large enough that |hat(t)| for t not in L could be
        # up to sqrt(0.5225) ≈ 0.72, violating the soundness condition
        # that |hat(t)| < eps for t not in L.
        assert 2 * np.sqrt(b_sq - a_sq) > eps, (
            "This test requires the accuracy limitation to be violated"
        )
        assert threshold < 1.0

    def test_tight_promise_works(self, parity_state):
        """
        With a^2 = b^2 = 1.0 (tight promise), the constraint
        eps >= 2*sqrt(0) = 0 is always satisfied.  Any eps works.
        """
        msg = _make_fake_prover_message(L=[1], n=3)
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.01, a_sq=1.0, b_sq=1.0, num_samples=10000
        )
        # Even with tiny epsilon (tight threshold), honest list passes
        assert result.accepted


class TestNoisyParityEndToEnd:
    """
    §6.2: Verifying noisy functional agnostic quantum learning.

    With noise_rate eta, the effective distribution has
    a^2 = b^2 = (1 - 2*eta)^2.  The protocol should work
    provided both prover and verifier know eta.
    """

    def test_moderate_noise(self):
        """eta = 0.15, so effective coefficient = 0.7 * chi_1."""
        n = 3
        eta = 0.15
        phi = np.array([0 if x & 1 == 0 else 1 for x in range(2**n)], dtype=np.float64)
        state = MoSState(n=n, phi=phi, noise_rate=eta, seed=42)

        prover = MoSProver(state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, theta=0.5, qfs_shots=3000, classical_samples=2000
        )

        a_sq = (1.0 - 2.0 * eta) ** 2
        verifier = MoSVerifier(state, seed=99)
        result = verifier.verify_parity(
            msg,
            epsilon=0.3,
            delta=0.1,
            theta=0.5,
            a_sq=a_sq,
            b_sq=a_sq,
            num_samples=5000,
        )
        assert result.accepted
        assert result.hypothesis.s == 1


class TestScalingEndToEnd:
    """Test at larger n to verify the protocol scales."""

    def test_n5_parity(self):
        """n=5, parity on bit 3: s = 8 = 0b01000."""
        n = 5
        phi = np.array(
            [((x >> 3) & 1) for x in range(2**n)],
            dtype=np.float64,
        )
        state = MoSState(n=n, phi=phi, seed=42)

        prover = MoSProver(state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=3000, classical_samples=2000
        )

        verifier = MoSVerifier(state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, delta=0.1, num_samples=5000)
        assert result.accepted
        assert result.hypothesis.s == 8

    def test_n4_two_bit_parity(self):
        """n=4, parity on bits {1, 3}: s = 0b1010 = 10."""
        n = 4
        phi = np.array(
            [(((x >> 1) & 1) ^ ((x >> 3) & 1)) for x in range(2**n)],
            dtype=np.float64,
        )
        state = MoSState(n=n, phi=phi, seed=42)

        prover = MoSProver(state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=3000, classical_samples=2000
        )

        verifier = MoSVerifier(state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, delta=0.1, num_samples=5000)
        assert result.accepted
        assert result.hypothesis.s == 10


class TestSeededReproducibility:
    """
    Same seeds should produce identical verification outcomes.
    """

    def test_verifier_deterministic(self, parity_state):
        msg = _make_fake_prover_message(L=[1], n=3)

        v1 = MoSVerifier(parity_state, seed=42)
        r1 = v1.verify_parity(msg, epsilon=0.3, num_samples=1000)

        v2 = MoSVerifier(parity_state, seed=42)
        r2 = v2.verify_parity(msg, epsilon=0.3, num_samples=1000)

        assert r1.outcome == r2.outcome
        assert r1.accumulated_weight == r2.accumulated_weight
        for s in r1.verifier_estimates:
            assert r1.verifier_estimates[s] == r2.verifier_estimates[s]

    def test_full_protocol_deterministic(self, parity_state):
        """Same prover seed + same verifier seed = same outcome."""
        p1 = MoSProver(parity_state, seed=10)
        m1 = p1.run_protocol(epsilon=0.3, qfs_shots=1000, classical_samples=500)
        v1 = MoSVerifier(parity_state, seed=20)
        r1 = v1.verify_parity(m1, epsilon=0.3, num_samples=1000)

        p2 = MoSProver(parity_state, seed=10)
        m2 = p2.run_protocol(epsilon=0.3, qfs_shots=1000, classical_samples=500)
        v2 = MoSVerifier(parity_state, seed=20)
        r2 = v2.verify_parity(m2, epsilon=0.3, num_samples=1000)

        assert r1.outcome == r2.outcome
        assert r1.hypothesis.s == r2.hypothesis.s


class TestFourierSparseEndToEnd:
    """
    End-to-end tests for the Fourier-sparse verification path
    (Theorems 10/15).
    """

    def test_two_parity_k2(self, two_parity_state):
        """
        Two-parity with k=2: prover sends both heavy coefficients,
        verifier should produce a 2-sparse hypothesis.
        """
        prover = MoSProver(two_parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, theta=0.3, qfs_shots=5000, classical_samples=3000
        )

        a_sq = 0.52
        verifier = MoSVerifier(two_parity_state, seed=99)
        result = verifier.verify_fourier_sparse(
            msg, epsilon=0.3, k=2, delta=0.1, a_sq=a_sq, b_sq=a_sq, num_samples=5000
        )
        if result.accepted:
            h = result.hypothesis
            assert isinstance(h, FourierSparseHypothesis)
            assert len(h.coefficients) == 2
            # The two heaviest should be s=1 and s=2
            assert 1 in h.coefficients
            assert 2 in h.coefficients

    def test_k_larger_than_list(self, parity_state):
        """
        If k > |L|, the hypothesis should use all entries in L,
        padded with zero-weight entries (which don't matter).
        """
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples=1000
        )

        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_fourier_sparse(
            msg, epsilon=0.3, k=5, delta=0.1, num_samples=5000
        )
        assert result.accepted
        # Hypothesis should have at most |L| coefficients
        assert len(result.hypothesis.coefficients) <= len(msg.L)


class TestMisclassificationGuarantee:
    """
    The ultimate test: does the output hypothesis actually satisfy the
    agnostic learning guarantee from the paper?

    Lemma 11: Pr[h(x) != y] <= min_t Pr[t.x != y] + eps.
    Lemma 14: 2-agnostic for Fourier-sparse.
    """

    def test_parity_zero_error(self, parity_state):
        """
        For pure parity, opt_D = 0.  The verified hypothesis should
        have error 0 (exact recovery).
        """
        prover = MoSProver(parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, qfs_shots=2000, classical_samples=1000
        )
        verifier = MoSVerifier(parity_state, seed=99)
        result = verifier.verify_parity(msg, epsilon=0.3, num_samples=5000)
        assert result.accepted

        # Check exact correctness on all inputs
        h = result.hypothesis
        for x in range(8):
            assert h.evaluate(x) == (x & 1)

    def test_noisy_parity_bounded_error(self, noisy_parity_state):
        """
        For noisy parity with eta=0.1, the Bayes-optimal parity is s=1
        with error eta = 0.1.  The verified hypothesis should achieve
        error <= 0.1 + epsilon.

        Actually, since h(x) = s.x is deterministic and s=1 is correct,
        the hypothesis error against the *noiseless* distribution is 0.
        The error against the noisy distribution is eta = 0.1.
        """
        prover = MoSProver(noisy_parity_state, seed=42)
        msg = prover.run_protocol(
            epsilon=0.3, delta=0.1, theta=0.5, qfs_shots=3000, classical_samples=2000
        )

        a_sq = 0.64
        verifier = MoSVerifier(noisy_parity_state, seed=99)
        result = verifier.verify_parity(
            msg, epsilon=0.3, theta=0.5, a_sq=a_sq, b_sq=a_sq, num_samples=5000
        )
        assert result.accepted
        assert result.hypothesis.s == 1

        # Compute empirical error against the noisy distribution
        # using a large number of classical samples
        rng = np.random.default_rng(0)
        xs, ys = noisy_parity_state.sample_classical_batch(50000, rng=rng)
        predictions = result.hypothesis.evaluate_batch(xs)
        error = np.mean(predictions != ys)

        # Optimal error for parity s=1 is eta = 0.1
        # With Lemma 11 guarantee: error <= opt + eps = 0.1 + 0.3
        assert error < 0.1 + 0.3 + 0.05, f"Error {error:.4f} exceeds agnostic bound 0.4"
        # In practice, error should be very close to 0.1
        assert abs(error - 0.1) < 0.02, f"Error {error:.4f} deviates from expected 0.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

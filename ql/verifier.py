r"""
Classical Verifier for Quantum Learning — §6 of Caro et al. (ITCS 2024).

Implements the verifier side of the interactive verification protocol for
distributional agnostic quantum parity and Fourier-sparse learning.

**Protocol overview** (verifier's role):

1. **Receive** the prover's message: a list :math:`L = \{s_1, \ldots, s_{|L|}\}`
   of candidate heavy Fourier coefficient indices (and optionally estimated
   coefficient values).

2. **Validate list size**: reject if :math:`|L|` exceeds the Parseval bound.

3. **Independently estimate** the Fourier coefficients
   :math:`\hat{\tilde\phi}(s)` for each :math:`s \in L` using the verifier's
   own classical random example access (Lemma 1 / Hoeffding bound).

4. **Check accumulated Fourier weight**: verify that

   .. math::

       \sum_{s \in L} \hat{\xi}(s)^2 \geq \tau_{\text{accept}}

   where :math:`\tau_{\text{accept}}` depends on the L\ :sup:`2`-bracket
   promise (**Definition 14**, :math:`\mathbb{E}[\phi^2] \in [a^2, b^2]`).
   The protocol *also* assumes the granularity promise (**Definition 11**
   in the functional case, **Definition 13** in the distributional case:
   :math:`\hat\phi(s) \neq 0 \Rightarrow |\hat\phi(s)| \geq \vartheta`),
   which is a *separate* constraint from Definition 14 — Def 14 brackets
   total Fourier mass while Def 11/13 forbids small-but-nonzero
   coefficients.  For the functional case (:math:`a = b = 1`), the
   acceptance threshold is :math:`1 - \varepsilon^2/8`; for the
   distributional case it is :math:`a^2 - \varepsilon^2/8`.

5. **Output hypothesis**:

   - *Parity learning* (Theorems 7/8/11/12):
     :math:`s_{\text{out}} = \arg\max_{s \in L} |\hat{\xi}(s)|`,
     hypothesis :math:`h(x) = s_{\text{out}} \cdot x`.

   - *Fourier-sparse learning* (Theorems 9/10/14/15):
     pick :math:`k` heaviest from :math:`L`, build randomized hypothesis
     per Lemma 14.

**Soundness** is information-theoretic: the verifier's checks guarantee
correctness regardless of the prover's strategy.  In particular, if the
verifier accepts, the output hypothesis satisfies the agnostic learning
guarantee with high probability — even against a computationally
unbounded dishonest prover.

**Copy complexity** (verifier): :math:`O(b^4 \log(1/\delta\vartheta^2) /
(\varepsilon^4 \vartheta^4))` classical random examples (Theorem 12).

References
----------
- Caro et al., "Classical Verification of Quantum Learning", ITCS 2024.
  §6.1 (Theorems 7–10), §6.2 (noisy), §6.3 (Theorems 11–16).
- Definitions 13, 14 for distribution class promises.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
from numpy.random import Generator, default_rng

from mos import MoSState
from ql.prover import ProverMessage


# ===================================================================
# Result types
# ===================================================================


class VerificationOutcome(Enum):
    """Outcome of the verification protocol."""

    ACCEPT = "accept"
    REJECT_LIST_TOO_LARGE = "reject_list_too_large"
    REJECT_INSUFFICIENT_WEIGHT = "reject_insufficient_weight"


class HypothesisType(Enum):
    """Type of hypothesis output by the verifier."""

    PARITY = "parity"
    FOURIER_SPARSE = "fourier_sparse"


@dataclass(frozen=True)
class ParityHypothesis:
    r"""
    Parity hypothesis :math:`h(x) = s \cdot x \mod 2`.

    Attributes
    ----------
    s : int
        The parity vector, as an integer in :math:`\{0, \ldots, 2^n - 1\}`.
    n : int
        Number of input bits.
    estimated_coefficient : float
        The verifier's estimate of :math:`\hat{\tilde\phi}(s)`.
    """

    s: int
    n: int
    estimated_coefficient: float

    def evaluate(self, x: int) -> int:
        r"""Evaluate :math:`h(x) = s \cdot x \mod 2`."""
        return bin(self.s & x).count("1") % 2

    def evaluate_batch(self, xs: np.ndarray) -> np.ndarray:
        """Evaluate the hypothesis on a batch of inputs."""
        return np.array(
            [bin(self.s & int(x)).count("1") % 2 for x in xs],
            dtype=np.uint8,
        )


@dataclass(frozen=True)
class FourierSparseHypothesis:
    r"""
    Randomised Fourier-sparse hypothesis per Lemma 14.

    Given estimated coefficients :math:`\tilde\phi(s_\ell)` for
    :math:`\ell = 1, \ldots, k`, defines
    :math:`g(x) = \sum_\ell \tilde\phi(s_\ell) \chi_{s_\ell}(x)`
    and the randomised hypothesis

    .. math::

        h(x) = 1 \text{ with probability }
        p(x) = \frac{(1 - g(x))^2}{2(1 + g(x)^2)}

    Attributes
    ----------
    coefficients : dict[int, float]
        Maps each :math:`s_\ell` to its estimated coefficient.
    n : int
        Number of input bits.
    """

    coefficients: dict[int, float]
    n: int

    def g(self, x: int) -> float:
        r"""Evaluate :math:`g(x) = \sum_\ell \hat\xi(s_\ell) \chi_{s_\ell}(x)`."""
        val = 0.0
        for s, coeff in self.coefficients.items():
            parity = bin(s & x).count("1") % 2
            chi_s = 1.0 - 2.0 * parity
            val += coeff * chi_s
        return val

    def evaluate(self, x: int, rng: Optional[Generator] = None) -> int:
        """Evaluate the randomised hypothesis at x."""
        if rng is None:
            rng = default_rng()
        gx = self.g(x)
        p = (1.0 - gx) ** 2 / (2.0 * (1.0 + gx**2))
        p = np.clip(p, 0.0, 1.0)
        return int(rng.random() < p)

    def evaluate_batch(
        self, xs: np.ndarray, rng: Optional[Generator] = None
    ) -> np.ndarray:
        """Evaluate the hypothesis on a batch of inputs."""
        if rng is None:
            rng = default_rng()
        return np.array([self.evaluate(int(x), rng=rng) for x in xs], dtype=np.uint8)


@dataclass(frozen=True)
class VerificationResult:
    r"""
    Complete result of the classical verification protocol.

    Attributes
    ----------
    outcome : VerificationOutcome
        Whether the verifier accepted or rejected (and why).
    hypothesis : ParityHypothesis | FourierSparseHypothesis | None
        The output hypothesis (None if rejected).
    hypothesis_type : HypothesisType | None
        The type of hypothesis produced.
    verifier_estimates : dict[int, float]
        The verifier's independent Fourier coefficient estimates
        :math:`\hat\xi(s)` for each :math:`s \in L`.
    accumulated_weight : float
        :math:`\sum_{s \in L} \hat\xi(s)^2`.
    acceptance_threshold : float
        The threshold that accumulated_weight was compared against.
    list_received : list[int]
        The list :math:`L` received from the prover.
    list_size_bound : int
        The Parseval-derived bound on :math:`|L|`.
    n : int
        Number of input bits.
    epsilon : float
        Accuracy parameter.
    num_classical_samples : int
        Number of classical samples the verifier consumed.
    """

    outcome: VerificationOutcome
    hypothesis: Optional[Union[ParityHypothesis, FourierSparseHypothesis]]
    hypothesis_type: Optional[HypothesisType]
    verifier_estimates: dict[int, float]
    accumulated_weight: float
    acceptance_threshold: float
    list_received: list[int]
    list_size_bound: int
    n: int
    epsilon: float
    num_classical_samples: int

    @property
    def accepted(self) -> bool:
        return self.outcome == VerificationOutcome.ACCEPT

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Verification Result — {self.outcome.value}",
            f"  n = {self.n}, epsilon = {self.epsilon:.4f}",
            f"  |L| = {len(self.list_received)} (bound: {self.list_size_bound})",
            f"  Accumulated weight: {self.accumulated_weight:.6f}",
            f"  Acceptance threshold: {self.acceptance_threshold:.6f}",
            f"  Classical samples used: {self.num_classical_samples}",
        ]
        if self.accepted and self.hypothesis is not None:
            if isinstance(self.hypothesis, ParityHypothesis):
                bits = format(self.hypothesis.s, f"0{self.n}b")
                lines.append(
                    f"  Hypothesis: parity s={self.hypothesis.s} ({bits}), "
                    f"est coeff={self.hypothesis.estimated_coefficient:+.6f}"
                )
            elif isinstance(self.hypothesis, FourierSparseHypothesis):
                lines.append(
                    f"  Hypothesis: Fourier-sparse with "
                    f"{len(self.hypothesis.coefficients)} terms"
                )
        return "\n".join(lines)


# ===================================================================
# Verifier
# ===================================================================


class MoSVerifier:
    r"""
    Classical verifier for the interactive quantum learning protocol.

    Implements the verifier side of the verification protocols from §6
    of Caro et al. (ITCS 2024).  The verifier has access to classical
    random examples from the distribution :math:`D` (obtained by
    computational-basis measurement of :math:`\rho_D`, per Lemma 1)
    and interacts with a (potentially dishonest) quantum prover.

    The verifier's checks are sufficient to guarantee:

    - **Completeness** (Theorems 8/12): when interacting with the honest
      prover, the verifier accepts and outputs a good hypothesis with
      probability :math:`\geq 1 - \delta`.

    - **Soundness** (information-theoretic): even against a computationally
      unbounded dishonest prover, if the verifier accepts, the output
      hypothesis satisfies the agnostic learning guarantee.

    Parameters
    ----------
    mos_state : MoSState
        The MoS state encoding the distribution :math:`D`.  The verifier
        uses this only to draw classical random examples via
        computational-basis measurement (:meth:`MoSState.sample_classical_batch`).
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    In a real deployment, the verifier would have access only to a
    classical random example oracle — not to the full MoSState.  Here
    we pass the MoSState to enable classical sampling (Lemma 1), which
    is the verifier's only use of it.
    """

    def __init__(
        self,
        mos_state: MoSState,
        seed: Optional[int] = None,
    ):
        self.state = mos_state
        self.n = mos_state.n
        self._seed = seed
        self._rng: Generator = default_rng(seed)

    # ------------------------------------------------------------------
    # Main verification entry points
    # ------------------------------------------------------------------

    def verify_parity(
        self,
        prover_message: ProverMessage,
        epsilon: float,
        delta: float = 0.1,
        theta: Optional[float] = None,
        a_sq: float = 1.0,
        b_sq: float = 1.0,
        num_samples: Optional[int] = None,
    ) -> VerificationResult:
        r"""
        Verify agnostic parity learning (Theorems 7/8/11/12).

        Implements the verifier's protocol for 1-agnostic proper
        parity verification.

        **Protocol steps** (Theorems 8/12):

        1. Receive :math:`L` from prover; reject if :math:`|L| > 64b^2/\vartheta^2`.
        2. Estimate :math:`\hat{\tilde\phi}(s)` for each :math:`s \in L`
           using classical random examples.
        3. Check :math:`\sum_{s \in L} \hat\xi(s)^2 \geq a^2 - \varepsilon^2/8`.
        4. Output :math:`h(x) = s_{\text{out}} \cdot x` where
           :math:`s_{\text{out}} = \arg\max_{s \in L} |\hat\xi(s)|`.

        Parameters
        ----------
        prover_message : ProverMessage
            The message received from the (potentially dishonest) prover.
        epsilon : float
            Accuracy parameter :math:`\varepsilon \in (0, 1)`.
        delta : float
            Confidence parameter :math:`\delta \in (0, 1)`.
        theta : float, optional
            Fourier resolution threshold :math:`\vartheta`.
            Defaults to ``epsilon``.
        a_sq : float
            Lower bound on :math:`\mathbb{E}[\tilde\phi(x)^2]`
            (Definition 14).  For the functional case, :math:`a^2 = 1`.
            For the distributional case with noise rate :math:`\eta`,
            :math:`a^2 = (1 - 2\eta)^2`.
        b_sq : float
            Upper bound on :math:`\mathbb{E}[\tilde\phi(x)^2]`
            (Definition 14).  For the functional case, :math:`b^2 = 1`.
        num_samples : int, optional
            Override the number of classical samples.  If not provided,
            computed from the Hoeffding bound.

        Returns
        -------
        VerificationResult
        """
        if theta is None:
            theta = epsilon

        return self._verify_core(
            prover_message=prover_message,
            epsilon=epsilon,
            delta=delta,
            theta=theta,
            a_sq=a_sq,
            b_sq=b_sq,
            hypothesis_type=HypothesisType.PARITY,
            k=None,
            num_samples=num_samples,
        )

    def verify_fourier_sparse(
        self,
        prover_message: ProverMessage,
        epsilon: float,
        k: int,
        delta: float = 0.1,
        theta: Optional[float] = None,
        a_sq: float = 1.0,
        b_sq: float = 1.0,
        num_samples: Optional[int] = None,
    ) -> VerificationResult:
        r"""
        Verify agnostic Fourier-sparse learning (Theorems 9/10/14/15).

        Implements the verifier's protocol for 2-agnostic improper
        Fourier-:math:`k`-sparse verification.

        **Protocol steps** (Theorems 10/15):

        1. Receive :math:`L` from prover; reject if :math:`|L| > 64b^2/\vartheta^2`.
        2. Estimate :math:`\hat{\tilde\phi}(s)` for each :math:`s \in L`.
        3. Check :math:`\sum_{s \in L} \hat\xi(s)^2 \geq a^2 - \varepsilon^2/(128k^2)`.
        4. Pick :math:`k` heaviest from :math:`L`, build randomised hypothesis
           per Lemma 14.

        Parameters
        ----------
        prover_message : ProverMessage
            The message received from the prover.
        epsilon : float
            Accuracy parameter.
        k : int
            Fourier sparsity parameter (number of terms).
        delta : float
            Confidence parameter.
        theta : float, optional
            Fourier resolution threshold.  Defaults to ``epsilon``.
        a_sq : float
            Lower bound on :math:`\mathbb{E}[\tilde\phi^2]`.
        b_sq : float
            Upper bound on :math:`\mathbb{E}[\tilde\phi^2]`.
        num_samples : int, optional
            Override the number of classical samples.

        Returns
        -------
        VerificationResult
        """
        if theta is None:
            theta = epsilon

        return self._verify_core(
            prover_message=prover_message,
            epsilon=epsilon,
            delta=delta,
            theta=theta,
            a_sq=a_sq,
            b_sq=b_sq,
            hypothesis_type=HypothesisType.FOURIER_SPARSE,
            k=k,
            num_samples=num_samples,
        )

    # ------------------------------------------------------------------
    # Core verification logic
    # ------------------------------------------------------------------

    def _verify_core(
        self,
        prover_message: ProverMessage,
        epsilon: float,
        delta: float,
        theta: float,
        a_sq: float,
        b_sq: float,
        hypothesis_type: HypothesisType,
        k: Optional[int],
        num_samples: Optional[int],
    ) -> VerificationResult:
        r"""
        Core verification protocol shared by parity and Fourier-sparse modes.

        Follows the structure of Theorems 8/10/12/15 in §6.
        """
        n = self.n
        L = prover_message.L

        # ---- Step 1: Validate list size (§6, Step 3) ----
        # Parseval bound: |L| <= 16 * E[tilde_phi^2] / theta^2 <= 16*b^2 / theta^2
        # The proofs in §6.3 use 64*b^2/theta^2 to accommodate the factor
        # of 4 from working with theta/2 resolution in Corollary 5.
        list_size_bound = int(np.ceil(64.0 * b_sq / theta**2))

        if len(L) > list_size_bound:
            return VerificationResult(
                outcome=VerificationOutcome.REJECT_LIST_TOO_LARGE,
                hypothesis=None,
                hypothesis_type=None,
                verifier_estimates={},
                accumulated_weight=0.0,
                acceptance_threshold=0.0,
                list_received=L,
                list_size_bound=list_size_bound,
                n=n,
                epsilon=epsilon,
                num_classical_samples=0,
            )

        # ---- Step 2: Estimate Fourier coefficients independently ----
        # The verifier uses its OWN classical samples — this is the key
        # to information-theoretic soundness.
        L_size = len(L)

        if hypothesis_type == HypothesisType.PARITY:
            # Theorem 12, Step 3: tolerance eps^2 / (16 * |L|)
            per_coeff_tolerance = epsilon**2 / (16.0 * max(L_size, 1))
        else:
            # Theorem 15, Step 3: tolerance eps^2 / (256 * k^2 * |L|)
            per_coeff_tolerance = epsilon**2 / (256.0 * k**2 * max(L_size, 1))

        if num_samples is None:
            # Hoeffding: P[|mean - E| > tol] <= 2*exp(-2*m*tol^2/4)
            # Want this <= delta / (2 * |L|) for union bound.
            # => m >= (2 / tol^2) * log(4 * |L| / delta)
            if L_size > 0:
                num_samples = int(
                    np.ceil(2.0 / per_coeff_tolerance**2 * np.log(4.0 * L_size / delta))
                )
                num_samples = max(num_samples, 100)
            else:
                num_samples = 0

        verifier_estimates = self._estimate_coefficients_independently(
            L=L,
            num_samples=num_samples,
        )

        # ---- Step 3: Check accumulated Fourier weight ----
        accumulated_weight = sum(verifier_estimates.get(s, 0.0) ** 2 for s in L)

        if hypothesis_type == HypothesisType.PARITY:
            # Theorem 12, Step 4: threshold a^2 - eps^2/8
            acceptance_threshold = a_sq - epsilon**2 / 8.0
        else:
            # Theorem 15, Step 4: threshold a^2 - eps^2/(128*k^2)
            acceptance_threshold = a_sq - epsilon**2 / (128.0 * k**2)

        if accumulated_weight < acceptance_threshold:
            return VerificationResult(
                outcome=VerificationOutcome.REJECT_INSUFFICIENT_WEIGHT,
                hypothesis=None,
                hypothesis_type=None,
                verifier_estimates=verifier_estimates,
                accumulated_weight=accumulated_weight,
                acceptance_threshold=acceptance_threshold,
                list_received=L,
                list_size_bound=list_size_bound,
                n=n,
                epsilon=epsilon,
                num_classical_samples=num_samples,
            )

        # ---- Step 4: Construct hypothesis ----
        if hypothesis_type == HypothesisType.PARITY:
            hypothesis = self._build_parity_hypothesis(L, verifier_estimates)
        else:
            hypothesis = self._build_fourier_sparse_hypothesis(L, verifier_estimates, k)

        return VerificationResult(
            outcome=VerificationOutcome.ACCEPT,
            hypothesis=hypothesis,
            hypothesis_type=hypothesis_type,
            verifier_estimates=verifier_estimates,
            accumulated_weight=accumulated_weight,
            acceptance_threshold=acceptance_threshold,
            list_received=L,
            list_size_bound=list_size_bound,
            n=n,
            epsilon=epsilon,
            num_classical_samples=num_samples,
        )

    # ------------------------------------------------------------------
    # Independent coefficient estimation (verifier's own classical data)
    # ------------------------------------------------------------------

    def _estimate_coefficients_independently(
        self,
        L: list[int],
        num_samples: int,
    ) -> dict[int, float]:
        r"""
        Estimate Fourier coefficients using the verifier's own classical
        random examples — independent of the prover.

        For each :math:`s \in L`:

        .. math::

            \hat\xi(s) = \frac{1}{m} \sum_{i=1}^{m}
            (1 - 2y_i)(-1)^{s \cdot x_i}

        where :math:`(x_i, y_i) \sim D` are i.i.d. classical samples.

        This independence is the foundation of information-theoretic
        soundness: the verifier's estimates are uncontaminated by
        anything the prover does.

        Parameters
        ----------
        L : list[int]
            Frequency indices to estimate.
        num_samples : int
            Number of classical samples to draw.

        Returns
        -------
        estimates : dict[int, float]
            The verifier's independent estimates.
        """
        if len(L) == 0 or num_samples == 0:
            return {s: 0.0 for s in L}

        # Draw classical samples from D (Lemma 1)
        xs, ys = self.state.sample_classical_batch(
            num_samples=num_samples,
            rng=self._rng,
        )

        # Compute signed labels: (1 - 2y)
        signed_labels = 1.0 - 2.0 * ys.astype(np.float64)

        estimates: dict[int, float] = {}
        for s in L:
            # chi_s(x) = (-1)^{popcount(s & x)}
            parities = np.array(
                [bin(s & int(x)).count("1") % 2 for x in xs],
                dtype=np.float64,
            )
            chi_s = 1.0 - 2.0 * parities
            est = float(np.mean(signed_labels * chi_s))
            est = np.clip(est, -1.0, 1.0)
            estimates[s] = est

        return estimates

    # ------------------------------------------------------------------
    # Hypothesis construction
    # ------------------------------------------------------------------

    def _build_parity_hypothesis(
        self,
        L: list[int],
        verifier_estimates: dict[int, float],
    ) -> ParityHypothesis:
        r"""
        Build parity hypothesis :math:`h(x) = s_{\text{out}} \cdot x`
        where :math:`s_{\text{out}} = \arg\max_{s \in L} |\hat\xi(s)|`.

        This is Step 4 of Theorems 7/8/11/12.
        """
        if not L:
            # Degenerate case: empty list passed checks (shouldn't happen
            # with sensible parameters, but handle gracefully)
            return ParityHypothesis(s=0, n=self.n, estimated_coefficient=0.0)

        s_out = max(L, key=lambda s: abs(verifier_estimates.get(s, 0.0)))
        return ParityHypothesis(
            s=s_out,
            n=self.n,
            estimated_coefficient=verifier_estimates.get(s_out, 0.0),
        )

    def _build_fourier_sparse_hypothesis(
        self,
        L: list[int],
        verifier_estimates: dict[int, float],
        k: int,
    ) -> FourierSparseHypothesis:
        r"""
        Build Fourier-sparse randomised hypothesis per Lemma 14.

        Pick :math:`k` heaviest from :math:`L` (by :math:`|\hat\xi(s)|`),
        construct :math:`g(x) = \sum_{\ell=1}^{k} \hat\xi(s_\ell) \chi_{s_\ell}(x)`.

        This is Step 4 of Theorems 9/10/14/15.
        """
        # Sort L by |estimated coefficient| descending
        sorted_L = sorted(
            L, key=lambda s: abs(verifier_estimates.get(s, 0.0)), reverse=True
        )

        # Take the k heaviest
        top_k = sorted_L[:k]

        coefficients = {s: verifier_estimates.get(s, 0.0) for s in top_k}

        return FourierSparseHypothesis(coefficients=coefficients, n=self.n)

    # ------------------------------------------------------------------
    # Convenience: full protocol (prover + verifier)
    # ------------------------------------------------------------------

    def run_full_protocol(
        self,
        prover_message: ProverMessage,
        mode: str = "parity",
        epsilon: Optional[float] = None,
        delta: float = 0.1,
        theta: Optional[float] = None,
        k: int = 1,
        a_sq: float = 1.0,
        b_sq: float = 1.0,
        num_samples: Optional[int] = None,
    ) -> VerificationResult:
        r"""
        Run the full verification given a prover message.

        Convenience wrapper that dispatches to :meth:`verify_parity`
        or :meth:`verify_fourier_sparse`.

        Parameters
        ----------
        prover_message : ProverMessage
            The prover's message.
        mode : str
            ``"parity"`` or ``"fourier_sparse"``.
        epsilon : float, optional
            Accuracy parameter.  Defaults to the prover's epsilon.
        delta : float
            Confidence parameter.
        theta : float, optional
            Fourier resolution threshold.
        k : int
            Fourier sparsity (only used in ``"fourier_sparse"`` mode).
        a_sq : float
            Lower bound on :math:`\mathbb{E}[\tilde\phi^2]`.
        b_sq : float
            Upper bound on :math:`\mathbb{E}[\tilde\phi^2]`.
        num_samples : int, optional
            Override classical sample count.

        Returns
        -------
        VerificationResult
        """
        if epsilon is None:
            epsilon = prover_message.epsilon

        if mode == "parity":
            return self.verify_parity(
                prover_message=prover_message,
                epsilon=epsilon,
                delta=delta,
                theta=theta,
                a_sq=a_sq,
                b_sq=b_sq,
                num_samples=num_samples,
            )
        elif mode == "fourier_sparse":
            return self.verify_fourier_sparse(
                prover_message=prover_message,
                epsilon=epsilon,
                k=k,
                delta=delta,
                theta=theta,
                a_sq=a_sq,
                b_sq=b_sq,
                num_samples=num_samples,
            )
        else:
            raise ValueError(
                f"Unknown mode {mode!r}; expected 'parity' or 'fourier_sparse'"
            )

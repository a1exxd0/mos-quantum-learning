r"""
Honest Quantum Prover for Classical Verification of Quantum Learning.

Implements the prover side of the interactive verification protocol from
Caro et al. (ITCS 2024), Theorems 8, 10, 12, and 15.

**Protocol overview** (prover's role):

1. **Quantum Fourier Sampling** (Theorem 5):  Given copies of the MoS
   state :math:`\rho_D`, apply :math:`H^{\otimes(n+1)}`, measure,
   post-select on the label qubit being 1.  Each accepted shot yields
   a sample :math:`s \in \{0,1\}^n` from the distribution

   .. math::

       \Pr[s \mid b{=}1]
       = \frac{1 - \mathbb{E}[\tilde\phi_{\text{eff}}(x)^2]}{2^n}
       + \hat{\tilde\phi}_{\text{eff}}(s)^2

2. **Empirical spectrum approximation** (Corollary 5 via Lemma 3 / DKW):
   From :math:`m` post-selected QFS samples, build the empirical
   distribution :math:`\tilde{q}_m` over :math:`\{0,1\}^n`.  By
   DKW, :math:`m = O(\log(1/\delta)/\varepsilon^4)` samples suffice
   for :math:`\|\tilde{q}_m - q\|_\infty \leq \varepsilon^2/8` with
   probability :math:`\geq 1 - \delta/2`.

3. **Heavy coefficient extraction**: Identify the list

   .. math::

       L = \{s \in \{0,1\}^n : \tilde{q}_m(s,1) \geq \varepsilon^2/4\}

   By the analysis in Corollary 5:

   - If :math:`|\hat{\tilde\phi}(s)| \geq \varepsilon`, then :math:`s \in L`.
   - If :math:`s \in L`, then :math:`|\hat{\tilde\phi}(s)| \geq \varepsilon/4`.

4. **Fourier coefficient estimation** (optional): For each :math:`s \in L`,
   estimate :math:`\hat{\tilde\phi}(s)` from classical samples of
   :math:`D` (obtained by computational-basis measurement of
   :math:`\rho_D`, per Lemma 1).

5. **Send** :math:`L` (and optionally the estimates) to the verifier.

The prover is *honest*: it follows the protocol faithfully.  Soundness
holds against *any* prover — the verifier's checks ensure correctness
regardless.

**Copy complexity**: The prover uses :math:`O(\log(1/\delta\varepsilon^2)/\varepsilon^4)`
copies of :math:`\rho_D` for QFS (Corollary 5), plus
:math:`O(\log(1/\delta\varepsilon^2)/\varepsilon^4)` copies for classical
estimation (via computational-basis measurement).

References
----------
- Caro et al., "Classical Verification of Quantum Learning", ITCS 2024.
  §5.1 (Corollary 5), §6 (Theorems 7–15).
- Lemma 3 (DKW-based empirical approximation).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator, default_rng

from mos import MoSState
from mos.sampler import QuantumFourierSampler, QFSResult


# ===================================================================
# Result containers
# ===================================================================


@dataclass(frozen=True)
class SpectrumApproximation:
    r"""
    Succinct approximation to the Fourier spectrum (Corollary 5).

    Attributes
    ----------
    entries : dict[int, float]
        Sparse representation: maps frequency index :math:`s` to
        the estimated squared-coefficient-related quantity
        :math:`\tilde{q}_m(s)`.  Only entries above the extraction
        threshold are stored.
    threshold : float
        The extraction threshold used (typically :math:`\varepsilon^2/4`).
    n : int
        Number of input bits.
    num_qfs_samples : int
        Number of post-selected QFS samples used to build the
        empirical distribution.
    total_qfs_shots : int
        Total QFS shots consumed (before post-selection).
    """

    entries: dict[int, float]
    threshold: float
    n: int
    num_qfs_samples: int
    total_qfs_shots: int


@dataclass(frozen=True)
class ProverMessage:
    r"""
    The message sent from the honest prover to the classical verifier.

    This implements the communication in Step 2 of the verification
    protocols (Theorems 7–15): a list :math:`L` of candidate heavy
    Fourier coefficient indices, optionally with estimated coefficient
    values.

    Attributes
    ----------
    L : list[int]
        List of frequency indices :math:`s` identified as having
        non-negligible Fourier weight.  Sorted by estimated weight
        (descending).
    estimates : dict[int, float]
        For each :math:`s \in L`, an estimate of
        :math:`\hat{\tilde\phi}(s)` obtained from classical samples.
        Empty if ``estimate_coefficients=False`` was used.
    n : int
        Number of input bits.
    epsilon : float
        Accuracy parameter used by the prover.
    theta : float
        Fourier coefficient resolution threshold :math:`\vartheta`.
    spectrum_approx : SpectrumApproximation
        The intermediate Fourier spectrum approximation (for diagnostics).
    qfs_result : QFSResult
        Raw QFS result (for diagnostics / post-hoc analysis).
    num_classical_samples : int
        Number of classical samples used for coefficient estimation.
    """

    L: list[int]
    estimates: dict[int, float]
    n: int
    epsilon: float
    theta: float
    spectrum_approx: SpectrumApproximation
    qfs_result: QFSResult
    num_classical_samples: int

    @property
    def list_size(self) -> int:
        """Number of candidate heavy coefficients."""
        return len(self.L)

    @property
    def total_copies_used(self) -> int:
        """Total MoS copies consumed (QFS + classical estimation)."""
        return self.spectrum_approx.total_qfs_shots + self.num_classical_samples

    def summary(self) -> str:
        """Human-readable summary of the prover's message."""
        lines = [
            "Prover Message (§6 protocol)",
            f"  n = {self.n}",
            f"  epsilon = {self.epsilon:.4f}, theta = {self.theta:.4f}",
            f"  |L| = {self.list_size}  (Parseval bound: {int(np.ceil(16 / self.theta**2))})",
            f"  QFS copies used: {self.spectrum_approx.total_qfs_shots}",
            f"  Post-selected QFS samples: {self.spectrum_approx.num_qfs_samples}",
            f"  Classical samples for estimation: {self.num_classical_samples}",
            f"  Total copies: {self.total_copies_used}",
        ]
        if self.estimates:
            lines.append("  Estimated coefficients:")
            for s in self.L:
                bits = format(s, f"0{self.n}b")
                est = self.estimates.get(s, float("nan"))
                lines.append(f"    s={s} ({bits}): est={est:+.6f}")
        else:
            lines.append("  Coefficient estimates: not computed")
            lines.append(f"  L = {self.L}")
        return "\n".join(lines)


# ===================================================================
# Prover
# ===================================================================


class MoSProver:
    r"""
    Honest quantum prover for the verification protocol.

    Follows the prover side of Theorems 8/10/12/15: uses MoS copies
    for Quantum Fourier Sampling, builds a succinct Fourier spectrum
    approximation, extracts heavy coefficients, and optionally
    estimates their values from classical samples.

    Parameters
    ----------
    mos_state : MoSState
        The MoS state to work with.  The prover has quantum access
        to copies of :math:`\rho_D`.
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    The prover's computational complexity is dominated by QFS
    (Section 5.1): :math:`O(n \cdot m)` single-qubit gates and
    :math:`\tilde{O}(n \cdot m)` classical processing, where
    :math:`m = O(\log(1/\delta)/\varepsilon^4)` is the number of
    QFS copies.
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
    # Main protocol entry point
    # ------------------------------------------------------------------

    def run_protocol(
        self,
        epsilon: float,
        delta: float = 0.1,
        theta: Optional[float] = None,
        estimate_coefficients: bool = True,
        qfs_mode: str = "statevector",
        qfs_shots: Optional[int] = None,
        classical_samples: Optional[int] = None,
    ) -> ProverMessage:
        r"""
        Execute the prover's side of the verification protocol.

        **Step 1**: Perform QFS to obtain post-selected samples.

        **Step 2**: Build the empirical spectrum approximation
        (Corollary 5 / Lemma 3).

        **Step 3**: Extract the heavy coefficient list :math:`L`.

        **Step 4** (optional): Estimate the Fourier coefficients
        for each :math:`s \in L` using classical samples.

        **Step 5**: Package and return the message.

        Parameters
        ----------
        epsilon : float
            Accuracy parameter :math:`\varepsilon \in (0, 1)`.
            The prover resolves the Fourier spectrum to accuracy
            :math:`\varepsilon` in :math:`\ell_\infty`-norm.
        delta : float
            Confidence parameter :math:`\delta \in (0, 1)`.
            The protocol succeeds with probability
            :math:`\geq 1 - \delta`.
        theta : float, optional
            Fourier coefficient resolution threshold
            :math:`\vartheta`.  If not provided, defaults to
            ``epsilon`` (appropriate for the functional agnostic
            case per Theorem 8).  For the distributional case
            (Theorem 12), should be set according to the
            promise on the distribution class.
        estimate_coefficients : bool
            If True (default), estimate :math:`\hat{\tilde\phi}(s)`
            for each :math:`s \in L` using classical samples
            (computational-basis measurement of :math:`\rho_D`).
            This is needed for the verifier's Fourier weight check.
        qfs_mode : str
            QFS simulation mode (``"statevector"``, ``"circuit"``,
            or ``"batched"``).
        qfs_shots : int, optional
            Override the number of QFS shots.  If not provided,
            computed from the DKW bound (Lemma 3):
            :math:`m = \lceil 2\log(4/\delta) / (\varepsilon^2/8)^2 \rceil`.
            Note: since post-selection succeeds with probability 1/2,
            we double this to get the expected number of accepted samples.
        classical_samples : int, optional
            Override the number of classical samples for coefficient
            estimation.  If not provided, computed from Hoeffding:
            :math:`m_2 = O(|L| \cdot \log(|L|/\delta) / \varepsilon^2)`.

        Returns
        -------
        ProverMessage
            The prover's message to the verifier.

        Raises
        ------
        ValueError
            If parameters are out of range.
        """
        # --- Parameter validation ---
        if not 0 < epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        if theta is None:
            theta = epsilon
        if not 0 < theta < 1:
            raise ValueError(f"theta must be in (0, 1), got {theta}")

        # --- Step 1: Compute required QFS shots ---
        # From Corollary 5: need m = O(log(1/delta) / tau^2) post-selected
        # samples where tau = epsilon^2 / 8.
        # DKW (Lemma 3): m = ceil(2 * log(2/delta_1) / tau^2) with delta_1 = delta/2
        tau = theta**2 / 8.0
        if qfs_shots is None:
            m_postselected = int(np.ceil(2.0 * np.log(4.0 / delta) / tau**2))
            # Post-selection succeeds ~1/2 the time, so need ~2m total shots.
            # Add a safety margin for finite-sample fluctuation.
            qfs_shots = int(np.ceil(2.5 * m_postselected))

        # --- Step 2: Run QFS ---
        sampler = QuantumFourierSampler(
            self.state,
            seed=int(self._rng.integers(0, 2**31)),
        )
        qfs_result = sampler.sample(shots=qfs_shots, mode=qfs_mode)

        # --- Step 3: Build empirical spectrum approximation ---
        spectrum_approx = self._build_spectrum_approximation(
            qfs_result=qfs_result,
            theta=theta,
        )

        # --- Step 4: Extract heavy coefficient list ---
        L = self._extract_heavy_list(
            spectrum_approx=spectrum_approx,
            theta=theta,
        )

        # --- Step 5: Estimate Fourier coefficients from classical samples ---
        estimates: dict[int, float] = {}
        num_classical = 0

        if estimate_coefficients and len(L) > 0:
            estimates, num_classical = self._estimate_coefficients(
                L=L,
                epsilon=epsilon,
                delta=delta,
                num_samples_override=classical_samples,
            )

        return ProverMessage(
            L=L,
            estimates=estimates,
            n=self.n,
            epsilon=epsilon,
            theta=theta,
            spectrum_approx=spectrum_approx,
            qfs_result=qfs_result,
            num_classical_samples=num_classical,
        )

    # ------------------------------------------------------------------
    # Step 3: Empirical spectrum approximation (Corollary 5 / Lemma 3)
    # ------------------------------------------------------------------

    def _build_spectrum_approximation(
        self,
        qfs_result: QFSResult,
        theta: float,
    ) -> SpectrumApproximation:
        r"""
        Build a succinct empirical approximation to the QFS distribution.

        From the post-selected QFS samples, compute the empirical
        distribution :math:`\tilde{q}_m(s, 1)` for each observed
        frequency :math:`s`.

        By Lemma 3 (DKW), with :math:`m` post-selected samples:

        .. math::

            \|\tilde{q}_m - q\|_\infty \leq \tau

        with probability :math:`\geq 1 - 2\exp(-m\tau^2/2)`.

        The empirical distribution :math:`\tilde{q}_m(s, 1)` relates
        to the Fourier coefficients via:

        .. math::

            \tilde{q}_m(s, 1) \approx q(s, 1)
            = \frac{1}{2}\Bigl[
              \frac{1 - \mathbb{E}[\tilde\phi^2]}{2^n}
              + \hat{\tilde\phi}(s)^2
            \Bigr]

        We store the (sparse) empirical distribution and use it to
        identify heavy coefficients.

        Parameters
        ----------
        qfs_result : QFSResult
            Output from the QFS procedure.
        theta : float
            Resolution threshold.

        Returns
        -------
        SpectrumApproximation
        """
        n = self.n

        # Build empirical distribution from post-selected counts
        # qfs_result.postselected_counts maps n-bit strings to counts
        ps_total = qfs_result.postselected_shots

        # Compute extraction threshold: epsilon^2 / 4
        # In terms of q(s, 1) = (1/2) * Pr[s | b=1], the threshold
        # for the full (n+1)-bit distribution is epsilon^2 / 8.
        # But since we're working with the conditional distribution
        # Pr[s | b=1] directly (post-selected), the threshold on the
        # conditional distribution is epsilon^2 / 4.
        # (See Corollary 5 proof: if q~_m(s,1) >= eps^2/4 then s in L)
        extraction_threshold = theta**2 / 4.0

        entries: dict[int, float] = {}
        if ps_total > 0:
            for bitstring, count in qfs_result.postselected_counts.items():
                s = int(bitstring, 2)
                empirical_prob = count / ps_total
                if empirical_prob >= extraction_threshold:
                    entries[s] = empirical_prob

        return SpectrumApproximation(
            entries=entries,
            threshold=extraction_threshold,
            n=n,
            num_qfs_samples=ps_total,
            total_qfs_shots=qfs_result.total_shots,
        )

    # ------------------------------------------------------------------
    # Step 4: Extract heavy coefficient list
    # ------------------------------------------------------------------

    def _extract_heavy_list(
        self,
        spectrum_approx: SpectrumApproximation,
        theta: float,
    ) -> list[int]:
        r"""
        Extract the list :math:`L` of candidate heavy Fourier
        coefficient indices.

        From Corollary 5:

        - **Completeness**: If :math:`|\hat{\tilde\phi}(s)| \geq \vartheta`,
          then :math:`s \in L`.
        - **Partial soundness**: If :math:`s \in L`, then
          :math:`|\hat{\tilde\phi}(s)| \geq \vartheta/4`.

        The list size is bounded by Parseval:
        :math:`|L| \leq 16/\vartheta^2`.

        Parameters
        ----------
        spectrum_approx : SpectrumApproximation
            The empirical spectrum approximation.
        theta : float
            Resolution threshold :math:`\vartheta`.

        Returns
        -------
        L : list[int]
            Sorted by empirical weight (descending).
        """
        # L consists of all s with empirical probability >= theta^2 / 4
        # These are exactly the entries stored in spectrum_approx
        L = sorted(
            spectrum_approx.entries.keys(),
            key=lambda s: spectrum_approx.entries[s],
            reverse=True,
        )

        # Sanity check: Parseval bound
        parseval_bound = int(np.ceil(16.0 / theta**2))
        if len(L) > parseval_bound:
            # This shouldn't happen with enough samples, but can occur
            # due to finite-sample noise.  Truncate to Parseval bound,
            # keeping the heaviest entries.
            L = L[:parseval_bound]

        return L

    # ------------------------------------------------------------------
    # Step 5: Classical coefficient estimation
    # ------------------------------------------------------------------

    def _estimate_coefficients(
        self,
        L: list[int],
        epsilon: float,
        delta: float,
        num_samples_override: Optional[int] = None,
    ) -> tuple[dict[int, float], int]:
        r"""
        Estimate Fourier coefficients for each :math:`s \in L`
        from classical random examples.

        By Lemma 1, computational-basis measurement of :math:`\rho_D`
        yields classical samples from :math:`D`.  For each :math:`s`,

        .. math::

            \hat{\tilde\phi}(s) = \mathbb{E}_{(x,y) \sim D}
            [(1 - 2y)(-1)^{s \cdot x}]

        so we estimate this expectation via sample mean.  By Hoeffding,
        :math:`m_2 = O(\log(|L|/\delta) / \varepsilon^2)` samples
        suffice for simultaneous :math:`\varepsilon`-accuracy across
        all :math:`s \in L`.

        Parameters
        ----------
        L : list[int]
            Frequency indices to estimate.
        epsilon : float
            Desired accuracy per coefficient.
        delta : float
            Overall confidence parameter.
        num_samples_override : int, optional
            Override the computed sample count.

        Returns
        -------
        estimates : dict[int, float]
            ``estimates[s]`` is the empirical estimate of
            :math:`\hat{\tilde\phi}(s)` for each :math:`s \in L`.
        num_samples : int
            Number of classical samples used.
        """
        L_size = len(L)
        if L_size == 0:
            return {}, 0

        # Hoeffding bound: for each s, need m2 samples for eps-accuracy
        # with failure probability delta / (2 * |L|).
        # Since |(1-2y)(-1)^{s.x}| <= 1, the range is 2.
        # Hoeffding: P[|mean - E| > eps] <= 2*exp(-2*m2*eps^2 / 4)
        #   = 2*exp(-m2*eps^2/2)
        # Want this <= delta / (2*|L|), so:
        #   m2 >= (2 / eps^2) * log(4*|L| / delta)
        if num_samples_override is not None:
            num_samples = num_samples_override
        else:
            num_samples = int(np.ceil(2.0 / epsilon**2 * np.log(4.0 * L_size / delta)))
            # Minimum sensible sample count
            num_samples = max(num_samples, 100)

        # Draw classical samples via computational-basis measurement
        xs, ys = self.state.sample_classical_batch(
            num_samples=num_samples,
            rng=self._rng,
        )

        # Compute (1 - 2y) for all samples
        signed_labels = 1.0 - 2.0 * ys.astype(np.float64)  # shape (m2,)

        # For each s in L, compute the empirical mean of (1-2y)*chi_s(x)
        estimates: dict[int, float] = {}
        for s in L:
            # chi_s(x) = (-1)^{popcount(s & x)}
            parities = np.array(
                [bin(s & int(x)).count("1") % 2 for x in xs],
                dtype=np.float64,
            )
            chi_s = 1.0 - 2.0 * parities
            est = float(np.mean(signed_labels * chi_s))
            # Project to [-1, 1] for safety
            est = np.clip(est, -1.0, 1.0)
            estimates[s] = est

        return estimates, num_samples

    # ------------------------------------------------------------------
    # Convenience: direct access to exact quantities (for validation)
    # ------------------------------------------------------------------

    def exact_heavy_coefficients(
        self,
        theta: float,
        *,
        effective: bool = True,
    ) -> list[tuple[int, float]]:
        r"""
        Return the exact list of heavy Fourier coefficients
        (for validation / comparison with the empirical list).

        Parameters
        ----------
        theta : float
            Threshold: returns all :math:`s` with
            :math:`|\hat{\tilde\phi}(s)| \geq \vartheta`.
        effective : bool
            If True, use noise-adjusted coefficients.

        Returns
        -------
        heavy : list[tuple[int, float]]
            Pairs :math:`(s, \hat{\tilde\phi}(s))`, sorted by
            absolute value descending.
        """
        spectrum = self.state.fourier_spectrum(effective=effective)
        heavy = [
            (s, float(spectrum[s]))
            for s in range(self.state.dim_x)
            if abs(spectrum[s]) >= theta
        ]
        heavy.sort(key=lambda t: abs(t[1]), reverse=True)
        return heavy

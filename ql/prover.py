"""
Prover-side heavy Fourier coefficient extraction for the MoS protocol.

Implements the quantum prover's role from Caro et al. "Classical Verification
of Quantum Learning" (2306.04843). The prover:

  1. Receives copies of the MoS state rho = E_f[|psi_f><psi_f|]
  2. Performs approximate quantum Fourier sampling via H^{n+1} + measure
  3. Post-selects on the label qubit being 1
  4. Collects frequency counts to identify heavy Fourier coefficients
  5. Returns the list L of heavy coefficients to the classical verifier

The key theoretical result (Theorem 5 / Corollary 5 of the paper) is that
each MoS copy, upon Hadamard measurement and post-selection on b=1,
produces a sample from a distribution close to:

    Pr[s | b=1] ∝ hat{tilde_phi}(s)^2

where tilde_phi = 1 - 2*phi. So repeatedly sampling and counting gives
empirical estimates of |hat{tilde_phi}(s)|^2, from which we extract the
heavy coefficients (those with |hat{tilde_phi}(s)|^2 >= theta).

Two extraction strategies are provided:

  - Direct thresholding: simple empirical frequency threshold on the
    post-selected s-distribution. Works well for moderate n.

  - Goldreich-Levin / Kushilevitz-Mansour style: iterative bisection
    of the Fourier spectrum to find heavy coefficients in poly(n) time.
    Essential for large n where 2^n enumeration is infeasible.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, NamedTuple
from dataclasses import dataclass, field
from numpy.random import Generator, default_rng


@dataclass
class HeavyCoefficient:
    """A Fourier coefficient identified as heavy by the prover."""
    s: int                  # The frequency index s in {0,...,2^n - 1}
    s_bits: str             # Binary representation of s
    estimated_weight: float # Empirical estimate of |hat{tilde_phi}(s)|^2
    count: int              # Raw count from post-selected measurements
    total_postselected: int # Total post-selected shots (b=1)


@dataclass
class ProverResult:
    """Complete output from the prover's heavy coefficient extraction."""
    heavy_coefficients: List[HeavyCoefficient]
    theta: float                    # Threshold used
    n: int                          # Number of input bits
    total_shots: int                # Total MoS copies consumed
    total_postselected: int         # Shots where b=1
    postselection_rate: float       # Fraction of shots with b=1
    all_s_counts: Dict[int, int]    # Full count dictionary (for diagnostics)

    @property
    def heavy_list(self) -> List[int]:
        """The list L = {s1, ..., s_|L|} sent to the verifier."""
        return [hc.s for hc in self.heavy_coefficients]

    @property
    def heavy_weights(self) -> Dict[int, float]:
        """Estimated |hat{tilde_phi}(s)|^2 for each heavy s."""
        return {hc.s: hc.estimated_weight for hc in self.heavy_coefficients}

    def summary(self) -> str:
        lines = [
            f"Prover Result (n={self.n})",
            f"  Total MoS copies used: {self.total_shots}",
            f"  Post-selected (b=1):   {self.total_postselected} "
            f"({self.postselection_rate:.3f})",
            f"  Threshold theta:       {self.theta:.6f}",
            f"  Heavy coefficients:    {len(self.heavy_coefficients)}",
        ]
        if self.heavy_coefficients:
            lines.append(f"  {'s':>6} {'bits':>12} {'|φ̂(s)|²':>12} {'count':>8}")
            lines.append(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
            for hc in sorted(self.heavy_coefficients,
                             key=lambda h: h.estimated_weight, reverse=True):
                lines.append(
                    f"  {hc.s:>6} {hc.s_bits:>12} "
                    f"{hc.estimated_weight:>12.6f} {hc.count:>8}"
                )
        return "\n".join(lines)


class MoSProver:
    """
    Quantum prover for the MoS verification protocol.

    Takes an MoSSimulator instance and performs the Hadamard measurement
    experiment to extract heavy Fourier coefficients.

    Parameters
    ----------
    simulator : MoSSimulator
        The simulator instance (provides MoS state preparation and measurement).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, simulator, seed: Optional[int] = None):
        self.sim = simulator
        self.n = simulator.n
        self.dim_x = simulator.dim_x
        self.rng = default_rng(seed)

    def _run_fourier_sampling(
        self,
        num_copies: int,
        mode: str = "statevector",
        **kwargs
    ) -> Tuple[Dict[int, int], int, int]:
        """
        Run the Hadamard measurement experiment and post-select on b=1.

        Each MoS copy is: sample f ~ F_D, prepare |psi_f>, apply H^{n+1},
        measure. Keep only outcomes where the label qubit b=1.

        Parameters
        ----------
        num_copies : int
            Number of MoS copies (shots) to consume.
        mode : str
            Simulation mode passed to simulator.sample_hadamard_measure.

        Returns
        -------
        s_counts : dict
            Counts of each s value from post-selected measurements.
        total_postselected : int
            Number of shots where b=1.
        total_shots : int
            Total shots consumed.
        """
        # Run the Hadamard measurement experiment
        raw_counts = self.sim.sample_hadamard_measure(
            shots=num_copies,
            mode=mode,
            rng=self.rng,
            **kwargs
        )

        # Use the simulator's analysis to extract post-selected s-distribution
        analysis = self.sim.analyze_counts(raw_counts)

        return (
            analysis['s_counts'],
            analysis['shots_last_1'],
            analysis['total_shots']
        )

    def extract_heavy_coefficients_threshold(
        self,
        theta: float,
        num_copies: int,
        mode: str = "statevector",
        confidence_boost: float = 1.0,
        **kwargs
    ) -> ProverResult:
        """
        Extract heavy Fourier coefficients by direct thresholding.

        A coefficient hat{tilde_phi}(s) is declared "heavy" if its estimated
        squared magnitude exceeds the threshold theta.

        The theoretical distribution (Theorem 5) gives:
            Pr[s | b=1] = (1 - E[tilde_phi^2]) / 2^n  +  hat{tilde_phi}(s)^2

        The first term is a uniform "noise floor" of magnitude ~ 1/2^n.
        We estimate |hat{tilde_phi}(s)|^2 from the empirical post-selected
        distribution and threshold at theta.

        This approach enumerates all 2^n possible s values, so it's only
        practical for moderate n (say n <= 20).

        Parameters
        ----------
        theta : float
            Heaviness threshold. A coefficient s is reported if the
            estimated |hat{tilde_phi}(s)|^2 >= theta.
            Typical choice: theta = epsilon^2 for learning accuracy epsilon.
        num_copies : int
            Number of MoS copies to use.
        mode : str
            Simulation mode ("statevector", "circuit", or "batched").
        confidence_boost : float
            Multiplicative factor for num_copies to improve confidence.
            The total shots used will be int(num_copies * confidence_boost).

        Returns
        -------
        result : ProverResult
            The extracted heavy coefficients and diagnostics.
        """
        total_shots = int(num_copies * confidence_boost)

        s_counts, total_postselected, actual_shots = self._run_fourier_sampling(
            total_shots, mode=mode, **kwargs
        )

        if total_postselected == 0:
            return ProverResult(
                heavy_coefficients=[],
                theta=theta,
                n=self.n,
                total_shots=actual_shots,
                total_postselected=0,
                postselection_rate=0.0,
                all_s_counts=s_counts,
            )

        # Estimate |hat{tilde_phi}(s)|^2 from empirical frequencies.
        #
        # From Theorem 5:
        #   Pr[s | b=1] = base + hat{tilde_phi}(s)^2
        # where base = (1 - E[tilde_phi^2]) / 2^n.
        #
        # We don't know E[tilde_phi^2] exactly, but we can either:
        #   (a) Estimate it from the data
        #   (b) Use the raw empirical Pr[s|b=1] as a proxy for
        #       hat{tilde_phi}(s)^2 (conservative: includes the base term)
        #
        # Approach (a): The noise floor is base = (1 - E[tilde_phi^2]) / 2^n.
        # We can estimate it as the median or minimum of the empirical
        # distribution (since most s will have small Fourier coefficients).
        #
        # We implement approach (a) with a robust floor estimator, then
        # subtract it to get cleaner weight estimates.

        empirical_probs = {}
        for s in range(self.dim_x):
            empirical_probs[s] = s_counts.get(s, 0) / total_postselected

        # Estimate the noise floor from the bottom quartile of frequencies.
        # For truly Fourier-sparse phi, most s values sit near the floor.
        all_probs = sorted(empirical_probs.values())
        quartile_idx = max(1, len(all_probs) // 4)
        estimated_floor = np.median(all_probs[:quartile_idx])

        # Extract heavy coefficients
        heavy = []
        for s in range(self.dim_x):
            # Subtract floor to estimate |hat{tilde_phi}(s)|^2
            raw_prob = empirical_probs[s]
            estimated_weight = max(0.0, raw_prob - estimated_floor)

            if estimated_weight >= theta:
                heavy.append(HeavyCoefficient(
                    s=s,
                    s_bits=format(s, f'0{self.n}b'),
                    estimated_weight=estimated_weight,
                    count=s_counts.get(s, 0),
                    total_postselected=total_postselected,
                ))

        postselection_rate = (
            total_postselected / actual_shots if actual_shots > 0 else 0.0
        )

        return ProverResult(
            heavy_coefficients=heavy,
            theta=theta,
            n=self.n,
            total_shots=actual_shots,
            total_postselected=total_postselected,
            postselection_rate=postselection_rate,
            all_s_counts=s_counts,
        )

    def extract_heavy_coefficients_gl(
        self,
        theta: float,
        num_copies_per_query: int = 1000,
        mode: str = "statevector",
        delta: float = 0.05,
        **kwargs
    ) -> ProverResult:
        """
        Extract heavy Fourier coefficients via Goldreich-Levin / Kushilevitz-
        Mansour style iterative bisection.

        This avoids enumerating all 2^n values of s and instead builds up
        heavy coefficients bit by bit in O(poly(n/epsilon)) time.

        The key idea: for a prefix p of length k, define the "bucket weight"
            W(p) = sum_{s: s starts with p} |hat{tilde_phi}(s)|^2

        If W(p) < theta, no extension of p can be heavy, so we prune.
        Otherwise we branch into p||0 and p||1 and recurse.

        To estimate W(p) from Fourier samples, we use the fact that under
        the post-selected distribution, Pr[s has prefix p | b=1] is
        approximately proportional to W(p).

        For large n (say n > 20), this is far more efficient than direct
        thresholding.

        Parameters
        ----------
        theta : float
            Heaviness threshold for |hat{tilde_phi}(s)|^2.
        num_copies_per_query : int
            MoS copies per "bucket weight query". More copies = better
            estimates but more total cost.
        mode : str
            Simulation mode.
        delta : float
            Confidence parameter. We use a slightly lower threshold
            internally (theta/2) to avoid missing heavy coefficients
            due to estimation noise, at the cost of potentially including
            some spurious ones (which the verifier will catch).

        Returns
        -------
        result : ProverResult
            The extracted heavy coefficients and diagnostics.
        """
        # We need a pool of Fourier samples to draw from.
        # Run a large batch and reuse samples for bucket queries.
        #
        # Total samples needed: O(n / theta) for GL-style,
        # but we want enough to estimate bucket weights reliably.
        # Use Hoeffding: to estimate a probability p to within epsilon
        # with confidence 1-delta, need O(log(1/delta) / epsilon^2) samples.

        # Conservative estimate of total samples needed
        # Each level of recursion queries the sample pool, max depth = n
        # At most 2/theta buckets survive at each level (by Parseval)
        # So total queries ~ 2n/theta, each needs accurate estimation
        estimation_accuracy = theta / 4
        samples_for_confidence = int(
            np.ceil(2 * np.log(2 * self.n / delta) / estimation_accuracy**2)
        )
        total_copies = max(
            num_copies_per_query * 4,
            samples_for_confidence
        )

        # Generate the sample pool
        s_counts, total_postselected, actual_shots = self._run_fourier_sampling(
            total_copies, mode=mode, **kwargs
        )

        if total_postselected == 0:
            return ProverResult(
                heavy_coefficients=[],
                theta=theta,
                n=self.n,
                total_shots=actual_shots,
                total_postselected=0,
                postselection_rate=0.0,
                all_s_counts=s_counts,
            )

        # Build an array of post-selected s samples for fast bucket queries
        s_samples = []
        for s, count in s_counts.items():
            s_samples.extend([s] * count)
        s_samples = np.array(s_samples, dtype=np.int64)
        N = len(s_samples)

        # Estimate noise floor from the empirical distribution
        empirical_probs = {}
        for s in range(self.dim_x):
            empirical_probs[s] = s_counts.get(s, 0) / N
        all_probs = sorted(empirical_probs.values())
        quartile_idx = max(1, len(all_probs) // 4)
        estimated_floor = np.median(all_probs[:quartile_idx])

        def estimate_bucket_weight(prefix: int, prefix_len: int) -> float:
            """
            Estimate W(prefix) = sum_{s with given prefix} |hat{phi}(s)|^2.

            We count how many samples s have the given prefix in their
            top prefix_len bits, divide by N to get an empirical probability,
            then subtract the expected floor contribution from the bucket.
            """
            if prefix_len == 0:
                # The entire spectrum: W = sum_s |hat{phi}(s)|^2 = E[tilde_phi^2]
                return 1.0  # Upper bound; will be refined by recursion

            # Mask: top prefix_len bits of s (in an n-bit representation)
            shift = self.n - prefix_len
            masked = s_samples >> shift
            count_in_bucket = int(np.sum(masked == prefix))
            empirical_prob = count_in_bucket / N

            # Subtract floor: the uniform component contributes
            # (2^(n - prefix_len)) / 2^n = 1/2^prefix_len per bucket
            bucket_size = 2 ** (self.n - prefix_len)
            floor_contribution = estimated_floor * bucket_size
            estimated_weight = max(0.0, empirical_prob - floor_contribution)

            return estimated_weight

        # Goldreich-Levin style iterative bisection
        # Start with the empty prefix (all of {0,...,2^n-1})
        # Frontier: list of (prefix_value, prefix_length) pairs
        internal_theta = theta / 2  # Lower threshold to avoid false negatives

        frontier = [(0, 0)]  # (prefix_value, prefix_length)
        heavy_candidates = []

        while frontier:
            prefix, depth = frontier.pop()

            if depth == self.n:
                # We've determined all n bits: prefix IS the candidate s
                s = prefix
                # Get final weight estimate
                prob = s_counts.get(s, 0) / N
                weight = max(0.0, prob - estimated_floor)
                if weight >= internal_theta:
                    heavy_candidates.append((s, weight))
                continue

            # Branch into prefix||0 and prefix||1
            for bit in [0, 1]:
                child_prefix = (prefix << 1) | bit
                child_depth = depth + 1

                w = estimate_bucket_weight(child_prefix, child_depth)
                if w >= internal_theta:
                    frontier.append((child_prefix, child_depth))

        # Final filtering at the actual threshold
        heavy = []
        for s, weight in heavy_candidates:
            if weight >= theta:
                heavy.append(HeavyCoefficient(
                    s=s,
                    s_bits=format(s, f'0{self.n}b'),
                    estimated_weight=weight,
                    count=s_counts.get(s, 0),
                    total_postselected=total_postselected,
                ))

        postselection_rate = (
            total_postselected / actual_shots if actual_shots > 0 else 0.0
        )

        return ProverResult(
            heavy_coefficients=heavy,
            theta=theta,
            n=self.n,
            total_shots=actual_shots,
            total_postselected=total_postselected,
            postselection_rate=postselection_rate,
            all_s_counts=s_counts,
        )

    def extract_heavy_coefficients(
        self,
        theta: float,
        num_copies: int = 10000,
        method: str = "auto",
        mode: str = "statevector",
        **kwargs
    ) -> ProverResult:
        """
        Unified interface for heavy Fourier coefficient extraction.

        Parameters
        ----------
        theta : float
            Heaviness threshold for |hat{tilde_phi}(s)|^2.
        num_copies : int
            Total MoS copies budget.
        method : str
            "threshold" - direct enumeration (practical for n <= ~20)
            "gl" - Goldreich-Levin style (efficient for large n)
            "auto" - choose based on n
        mode : str
            Simulation mode for the underlying Hadamard measurements.

        Returns
        -------
        result : ProverResult
        """
        if method == "auto":
            # GL becomes worthwhile when 2^n is large
            method = "threshold" if self.n <= 16 else "gl"

        if method == "threshold":
            return self.extract_heavy_coefficients_threshold(
                theta=theta,
                num_copies=num_copies,
                mode=mode,
                **kwargs
            )
        elif method == "gl":
            return self.extract_heavy_coefficients_gl(
                theta=theta,
                num_copies_per_query=num_copies // 4,
                mode=mode,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_ground_truth_heavy(self, theta: float) -> List[Tuple[int, float]]:
        """
        Compute the true heavy Fourier coefficients (for validation).

        Uses the simulator's exact Fourier coefficient computation.

        Parameters
        ----------
        theta : float
            Threshold.

        Returns
        -------
        heavy : list of (s, |hat{tilde_phi}(s)|^2) pairs
        """
        heavy = []
        for s in range(self.dim_x):
            fc = self.sim.fourier_coefficient(s)
            weight = fc ** 2
            if weight >= theta:
                heavy.append((s, weight))
        return sorted(heavy, key=lambda t: t[1], reverse=True)
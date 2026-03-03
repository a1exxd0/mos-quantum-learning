"""
Classical Verifier and Interactive Protocol for MoS Quantum Learning.

Implements the classical verifier's role and the full interactive protocol
from Caro et al. "Classical Verification of Quantum Learning" (2306.04843).

The protocol:
  1. Prover (quantum): uses MoS copies + Hadamard measurement to extract
     heavy Fourier coefficients, sends list L = {s1,...,s_|L|} to verifier.
  2. Verifier (classical): using only random examples (x,y) ~ D:
     a) Estimates each hat{tilde_phi}(s_i)^2 for s_i in L
     b) Estimates the total squared Fourier weight sum_L hat{tilde_phi}(s)^2
     c) Checks this against the a priori known total weight interval [a^2, b^2]
     d) Accepts and builds hypothesis if weight check passes; rejects otherwise.
  3. Hypothesis construction: if accepted, the verifier builds
     h(x) = sign(sum_{s in L} hat{tilde_phi}_est(s) * chi_s(x))
     (thresholded to {0,1}) as the agnostic parity / Fourier-sparse hypothesis.

Classical Fourier coefficient estimation:
  For any s, the verifier can estimate hat{tilde_phi}(s) from random examples
  (x,y) ~ D = (U_n, phi) using:
    hat{tilde_phi}(s) = E_x[tilde_phi(x) * chi_s(x)]
                       = E_{(x,y)}[(1 - 2y) * (-1)^{<s,x>}]
  This is just the sample mean of (1-2y)*(-1)^{<s,x>} over random examples.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from numpy.random import Generator, default_rng


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class VerificationDecision(Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass
class FourierEstimate:
    """Verifier's estimate of a single Fourier coefficient."""
    s: int
    s_bits: str
    estimated_coeff: float      # hat{tilde_phi}(s) estimate
    estimated_weight: float     # |hat{tilde_phi}(s)|^2 estimate
    std_error: float            # Standard error of coefficient estimate
    num_samples: int            # Classical samples used for this estimate


@dataclass
class VerifierResult:
    """Output from the classical verifier."""
    decision: VerificationDecision
    reason: str
    estimated_list_weight: float        # sum_{s in L} |hat{phi}(s)|^2
    expected_weight_interval: Tuple[float, float]  # [a^2, b^2]
    fourier_estimates: List[FourierEstimate]
    hypothesis_coefficients: Dict[int, float]  # s -> hat{tilde_phi}(s) for h
    num_classical_samples: int
    epsilon: float
    delta: float

    def summary(self) -> str:
        lines = [
            f"Verifier Result: {self.decision.value}",
            f"  Reason: {self.reason}",
            f"  Classical samples used: {self.num_classical_samples}",
            f"  epsilon={self.epsilon}, delta={self.delta}",
            f"  Estimated list weight:  {self.estimated_list_weight:.6f}",
            f"  Expected interval:      "
            f"[{self.expected_weight_interval[0]:.6f}, "
            f"{self.expected_weight_interval[1]:.6f}]",
            f"  Fourier estimates ({len(self.fourier_estimates)}):",
        ]
        if self.fourier_estimates:
            lines.append(
                f"    {'s':>6} {'bits':>12} {'φ̂(s)':>10} "
                f"{'|φ̂(s)|²':>10} {'±stderr':>10}"
            )
            lines.append(f"    {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
            for fe in sorted(self.fourier_estimates,
                             key=lambda f: abs(f.estimated_coeff), reverse=True):
                lines.append(
                    f"    {fe.s:>6} {fe.s_bits:>12} "
                    f"{fe.estimated_coeff:>+10.6f} "
                    f"{fe.estimated_weight:>10.6f} "
                    f"{fe.std_error:>10.6f}"
                )
        return "\n".join(lines)


@dataclass
class ProtocolTranscript:
    """Full transcript of the interactive protocol."""
    n: int
    epsilon: float
    delta: float
    theta: float

    # Round 1: Prover -> Verifier
    prover_list: List[int]                # L = {s1,...,s_|L|}
    prover_weights: Dict[int, float]      # Prover's weight estimates

    # Round 2: Verifier's checks
    verifier_result: 'VerifierResult'

    # Final output
    hypothesis: Optional[Dict[int, float]]  # s -> coefficient, or None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PROTOCOL TRANSCRIPT",
            "=" * 60,
            f"Parameters: n={self.n}, epsilon={self.epsilon}, "
            f"delta={self.delta}, theta={self.theta}",
            "",
            "--- Round 1: Prover -> Verifier ---",
            f"  Prover sends list L of {len(self.prover_list)} "
            f"heavy coefficients: {sorted(self.prover_list)}",
        ]
        if self.prover_weights:
            lines.append("  Prover's weight estimates:")
            for s in sorted(self.prover_weights):
                lines.append(
                    f"    s={s}: |φ̂(s)|² ≈ {self.prover_weights[s]:.6f}"
                )

        lines.extend([
            "",
            "--- Round 2: Verifier checks ---",
            self.verifier_result.summary(),
        ])

        if self.hypothesis is not None:
            lines.extend([
                "",
                "--- Hypothesis ---",
                "  h(x) = sign(sum_s c_s * chi_s(x)), thresholded to {0,1}",
                "  Coefficients:",
            ])
            for s in sorted(self.hypothesis):
                lines.append(f"    s={s}: c_s = {self.hypothesis[s]:+.6f}")
        else:
            lines.extend(["", "--- No hypothesis produced (rejected) ---"])

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classical random example oracle
# ---------------------------------------------------------------------------

class ClassicalExampleOracle:
    """
    Classical random example oracle for D = (U_n, phi).

    Samples (x, y) where x ~ Uniform({0,1}^n), y ~ Bernoulli(phi(x)).
    This is the verifier's data access model.
    """

    def __init__(self, n: int, phi: np.ndarray, seed: Optional[int] = None):
        self.n = n
        self.dim_x = 2 ** n
        self._phi = np.asarray(phi, dtype=np.float64)
        self.rng = default_rng(seed)

    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw num_samples i.i.d. random examples (x, y).

        Returns
        -------
        xs : np.ndarray of shape (num_samples,), dtype=int
            Uniformly random inputs.
        ys : np.ndarray of shape (num_samples,), dtype=int
            Labels y ~ Bernoulli(phi(x)).
        """
        xs = self.rng.integers(0, self.dim_x, size=num_samples)
        probs = self._phi[xs]
        ys = (self.rng.random(num_samples) < probs).astype(np.int64)
        return xs, ys


# ---------------------------------------------------------------------------
# Classical Verifier
# ---------------------------------------------------------------------------

class MoSVerifier:
    """
    Classical verifier for the MoS verification protocol.

    The verifier has access only to classical random examples (x, y) ~ D.
    It receives a list L of claimed heavy Fourier coefficients from the prover
    and must verify the claim using classical statistics.

    Parameters
    ----------
    oracle : ClassicalExampleOracle
        Source of classical random examples.
    n : int
        Number of input bits.
    """

    def __init__(self, oracle: ClassicalExampleOracle, n: int):
        self.oracle = oracle
        self.n = n
        self.dim_x = 2 ** n

    def _parity(self, s: int, x: int) -> int:
        """Compute <s, x> mod 2 = popcount(s & x) mod 2."""
        return bin(s & x).count('1') % 2

    def estimate_fourier_coefficient(
        self,
        s: int,
        num_samples: int
    ) -> FourierEstimate:
        """
        Estimate hat{tilde_phi}(s) from classical random examples.

        Uses the identity:
            hat{tilde_phi}(s) = E_{(x,y)~D}[(1 - 2y) * (-1)^{<s,x>}]

        Each sample gives a term in {-1, +1}, and we take the mean.

        Parameters
        ----------
        s : int
            The frequency to estimate.
        num_samples : int
            Number of classical random examples to use.

        Returns
        -------
        estimate : FourierEstimate
        """
        xs, ys = self.oracle.sample(num_samples)

        # Compute (1 - 2y) * (-1)^{<s,x>} for each sample
        tilde_y = 1 - 2 * ys  # maps y in {0,1} to {+1,-1}

        # Vectorised parity computation
        # For each x, compute popcount(s & x) mod 2
        sx = s & xs  # bitwise AND
        # popcount mod 2: XOR fold
        parities = np.zeros(len(xs), dtype=np.int64)
        temp = sx.copy()
        while np.any(temp > 0):
            parities ^= (temp & 1)
            temp >>= 1
        chi_s = 1 - 2 * parities  # (-1)^{<s,x>}

        terms = tilde_y * chi_s
        estimated_coeff = float(np.mean(terms))
        std_error = float(np.std(terms, ddof=1) / np.sqrt(num_samples))

        return FourierEstimate(
            s=s,
            s_bits=format(s, f'0{self.n}b'),
            estimated_coeff=estimated_coeff,
            estimated_weight=estimated_coeff ** 2,
            std_error=std_error,
            num_samples=num_samples,
        )

    def verify(
        self,
        prover_list: List[int],
        weight_interval: Tuple[float, float],
        epsilon: float = 0.1,
        delta: float = 0.05,
        samples_per_coefficient: Optional[int] = None,
    ) -> VerifierResult:
        """
        Verify the prover's claimed list of heavy Fourier coefficients.

        Verification procedure:
        1. For each s in prover_list L, estimate hat{tilde_phi}(s) using
           classical random examples.
        2. Compute the estimated total squared Fourier weight of L:
           W_L = sum_{s in L} hat{tilde_phi}(s)^2
        3. Check whether W_L falls within the expected interval [a^2, b^2]
           (with tolerance for estimation error).
        4. Accept if the weight check passes; reject otherwise.

        The soundness argument: if the prover omitted a truly heavy
        coefficient or included a spurious one, the total weight W_L will
        deviate from the known interval [a^2, b^2].

        Parameters
        ----------
        prover_list : list of int
            The list L of claimed heavy Fourier coefficient indices.
        weight_interval : (float, float)
            The a priori known interval [a^2, b^2] for the total squared
            Fourier weight sum_s hat{tilde_phi}(s)^2. In the paper, this
            comes from the restricted distribution class D.
        epsilon : float
            Accuracy parameter for the agnostic learning guarantee.
        delta : float
            Confidence parameter.
        samples_per_coefficient : int, optional
            Classical samples per coefficient estimate. If None, computed
            from epsilon and delta for the theoretical guarantee:
            O(log(|L|/delta) / epsilon^2).

        Returns
        -------
        result : VerifierResult
        """
        L = prover_list
        a_sq, b_sq = weight_interval

        # Determine samples per coefficient for desired accuracy
        if samples_per_coefficient is None:
            # By Hoeffding, to estimate each coefficient to accuracy
            # epsilon / (2 * sqrt(|L|)) with confidence delta / |L|,
            # need O(|L| * log(|L|/delta) / epsilon^2) samples.
            # This ensures the total weight estimate is accurate to ~ epsilon.
            safe_L = max(len(L), 1)
            samples_per_coefficient = int(np.ceil(
                2 * safe_L * np.log(2 * safe_L / delta) / (epsilon ** 2)
            ))
            # Minimum for reasonable estimates
            samples_per_coefficient = max(samples_per_coefficient, 500)

        total_samples = samples_per_coefficient * len(L) if L else 0

        # Step 1: Estimate each Fourier coefficient in L
        fourier_estimates = []
        for s in L:
            fe = self.estimate_fourier_coefficient(s, samples_per_coefficient)
            fourier_estimates.append(fe)

        # Step 2: Compute total estimated squared Fourier weight of L
        estimated_list_weight = sum(fe.estimated_weight for fe in fourier_estimates)

        # Estimation error bound for total weight
        # Each coefficient estimate has variance <= 1/num_samples,
        # so the squared coefficient has variance that we bound via
        # the delta method. Conservative bound:
        total_stderr = sum(
            2 * abs(fe.estimated_coeff) * fe.std_error
            for fe in fourier_estimates
        )

        # Step 3: Weight check
        # The prover's list should capture essentially all the Fourier
        # weight. Allow tolerance for estimation error.
        tolerance = 3 * total_stderr + epsilon  # 3-sigma + epsilon slack

        # Check: is the estimated weight consistent with [a^2, b^2]?
        weight_lower = estimated_list_weight - tolerance
        weight_upper = estimated_list_weight + tolerance

        if weight_upper < a_sq:
            # The list has too little weight — prover likely omitted
            # heavy coefficients
            decision = VerificationDecision.REJECT
            reason = (
                f"Estimated list weight {estimated_list_weight:.6f} "
                f"(± {tolerance:.6f}) falls below expected lower bound "
                f"{a_sq:.6f}. Prover likely omitted heavy coefficients."
            )
        elif weight_lower > b_sq:
            # The list has too much weight — prover included spurious
            # coefficients or the distribution is outside the class
            decision = VerificationDecision.REJECT
            reason = (
                f"Estimated list weight {estimated_list_weight:.6f} "
                f"(± {tolerance:.6f}) exceeds expected upper bound "
                f"{b_sq:.6f}. Prover may have included spurious coefficients."
            )
        else:
            decision = VerificationDecision.ACCEPT
            reason = (
                f"Estimated list weight {estimated_list_weight:.6f} "
                f"(± {tolerance:.6f}) is consistent with expected interval "
                f"[{a_sq:.6f}, {b_sq:.6f}]."
            )

        # Step 4: If accepted, build hypothesis coefficients
        hypothesis_coefficients = {}
        if decision == VerificationDecision.ACCEPT:
            for fe in fourier_estimates:
                hypothesis_coefficients[fe.s] = fe.estimated_coeff

        return VerifierResult(
            decision=decision,
            reason=reason,
            estimated_list_weight=estimated_list_weight,
            expected_weight_interval=weight_interval,
            fourier_estimates=fourier_estimates,
            hypothesis_coefficients=hypothesis_coefficients,
            num_classical_samples=total_samples,
            epsilon=epsilon,
            delta=delta,
        )


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

class FourierHypothesis:
    """
    Hypothesis constructed from verified Fourier coefficients.

    h(x) = 1  if  sum_{s in L} c_s * chi_s(x) >= 0
    h(x) = 0  otherwise

    where c_s = hat{tilde_phi}(s) and chi_s(x) = (-1)^{<s,x>}.

    For parity learning (|L| = 1), this reduces to h(x) = chi_{s*}(x)
    mapped to {0,1}.
    """

    def __init__(self, n: int, coefficients: Dict[int, float]):
        """
        Parameters
        ----------
        n : int
            Number of input bits.
        coefficients : dict
            {s: c_s} mapping frequency indices to estimated coefficients.
        """
        self.n = n
        self.dim_x = 2 ** n
        self.coefficients = coefficients

    def predict(self, x: int) -> int:
        """Predict label for a single input x."""
        val = 0.0
        for s, c in self.coefficients.items():
            parity = bin(s & x).count('1') % 2
            chi_s = 1 - 2 * parity
            val += c * chi_s
        # tilde_phi > 0 means phi < 0.5 means y=0 more likely
        # tilde_phi < 0 means phi > 0.5 means y=1 more likely
        # h predicts the more likely label
        return 0 if val >= 0 else 1

    def predict_batch(self, xs: np.ndarray) -> np.ndarray:
        """Predict labels for a batch of inputs."""
        vals = np.zeros(len(xs), dtype=np.float64)
        for s, c in self.coefficients.items():
            sx = s & xs
            parities = np.zeros(len(xs), dtype=np.int64)
            temp = sx.copy()
            while np.any(temp > 0):
                parities ^= (temp & 1)
                temp >>= 1
            chi_s = 1 - 2 * parities
            vals += c * chi_s
        return (vals < 0).astype(np.int64)

    def evaluate_risk(
        self,
        oracle: ClassicalExampleOracle,
        num_samples: int = 10000,
    ) -> float:
        """
        Estimate the risk Pr_{(x,y)~D}[h(x) != y] using random examples.
        """
        xs, ys = oracle.sample(num_samples)
        predictions = self.predict_batch(xs)
        return float(np.mean(predictions != ys))

    def evaluate_excess_risk(
        self,
        oracle: ClassicalExampleOracle,
        best_parity_risk: float,
        num_samples: int = 10000,
    ) -> float:
        """
        Estimate excess risk: risk(h) - risk(best parity).
        """
        risk = self.evaluate_risk(oracle, num_samples)
        return risk - best_parity_risk


# ---------------------------------------------------------------------------
# Interactive Protocol Orchestrator
# ---------------------------------------------------------------------------

class MoSProtocol:
    """
    Orchestrates the full interactive verification protocol.

    Ties together:
      - MoSProver (quantum side): extracts heavy Fourier coefficients
      - MoSVerifier (classical side): verifies the list and builds hypothesis
      - ClassicalExampleOracle: provides random examples to verifier

    Parameters
    ----------
    simulator : MoSSimulator
        The MoS quantum simulator.
    phi : np.ndarray
        The conditional probability function (also given to oracle).
    weight_interval : (float, float)
        A priori known interval [a^2, b^2] for sum_s hat{tilde_phi}(s)^2.
        This models the restricted distribution class D from the paper.
        If None, it's computed from the true phi (for simulation/testing).
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        simulator,
        phi: np.ndarray,
        weight_interval: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ):
        self.sim = simulator
        self.n = simulator.n
        self.phi = np.asarray(phi, dtype=np.float64)

        rng = default_rng(seed)
        seeds = rng.integers(0, 2**31, size=3)

        # Import prover here to avoid circular dependency at module level
        from ql.prover import MoSProver

        self.prover = MoSProver(simulator, seed=int(seeds[0]))
        self.oracle = ClassicalExampleOracle(
            self.n, self.phi, seed=int(seeds[1])
        )
        self.verifier = MoSVerifier(self.oracle, self.n)

        # Compute or use provided weight interval
        if weight_interval is not None:
            self.weight_interval = weight_interval
        else:
            self.weight_interval = self._compute_true_weight_interval()

        self._eval_rng_seed = int(seeds[2])

    def _compute_true_weight_interval(
        self,
        margin: float = 0.05
    ) -> Tuple[float, float]:
        """
        Compute the true total squared Fourier weight and return an interval.

        In practice, this interval comes from the distribution class promise.
        For simulation, we compute it from the true phi and add a small margin.
        """
        tilde_phi = 2 * self.phi - 1
        total_weight = float(np.mean(tilde_phi ** 2))
        lower = max(0.0, total_weight - margin)
        upper = min(1.0, total_weight + margin)
        return (lower, upper)

    def run(
        self,
        epsilon: float = 0.1,
        delta: float = 0.05,
        theta: Optional[float] = None,
        prover_copies: int = 50000,
        prover_method: str = "auto",
        prover_mode: str = "statevector",
        verifier_samples_per_coeff: Optional[int] = None,
    ) -> ProtocolTranscript:
        """
        Execute the full interactive protocol.

        Parameters
        ----------
        epsilon : float
            Learning accuracy parameter.
        delta : float
            Confidence parameter.
        theta : float, optional
            Heaviness threshold. Default: epsilon^2 / 4 (sufficient for
            the agnostic guarantee).
        prover_copies : int
            MoS copies for the prover.
        prover_method : str
            Prover extraction method ("threshold", "gl", "auto").
        prover_mode : str
            Simulation mode for Hadamard measurements.
        verifier_samples_per_coeff : int, optional
            Classical samples per Fourier coefficient for verifier.

        Returns
        -------
        transcript : ProtocolTranscript
        """
        if theta is None:
            theta = epsilon ** 2 / 4

        # ============================================================
        # ROUND 1: Prover extracts heavy coefficients and sends list L
        # ============================================================
        prover_result = self.prover.extract_heavy_coefficients(
            theta=theta,
            num_copies=prover_copies,
            method=prover_method,
            mode=prover_mode,
        )

        prover_list = prover_result.heavy_list
        prover_weights = prover_result.heavy_weights

        # ============================================================
        # ROUND 2: Verifier checks the list using classical examples
        # ============================================================
        verifier_result = self.verifier.verify(
            prover_list=prover_list,
            weight_interval=self.weight_interval,
            epsilon=epsilon,
            delta=delta,
            samples_per_coefficient=verifier_samples_per_coeff,
        )

        # ============================================================
        # OUTPUT: Build hypothesis if accepted
        # ============================================================
        hypothesis = None
        if verifier_result.decision == VerificationDecision.ACCEPT:
            hypothesis = verifier_result.hypothesis_coefficients

        return ProtocolTranscript(
            n=self.n,
            epsilon=epsilon,
            delta=delta,
            theta=theta,
            prover_list=prover_list,
            prover_weights=prover_weights,
            verifier_result=verifier_result,
            hypothesis=hypothesis,
        )

    def evaluate_hypothesis(
        self,
        transcript: ProtocolTranscript,
        num_samples: int = 50000,
    ) -> Optional[Dict]:
        """
        Evaluate the hypothesis produced by the protocol.

        Returns
        -------
        evaluation : dict or None
            None if the protocol rejected (no hypothesis).
            Otherwise dict with risk, excess risk, etc.
        """
        if transcript.hypothesis is None:
            return None

        h = FourierHypothesis(self.n, transcript.hypothesis)
        eval_oracle = ClassicalExampleOracle(
            self.n, self.phi, seed=self._eval_rng_seed
        )

        risk = h.evaluate_risk(eval_oracle, num_samples)

        # Compute best parity risk for comparison
        tilde_phi = 2 * self.phi - 1
        best_parity_risk = 1.0
        best_s = 0
        dim_x = 2 ** self.n
        for s in range(dim_x):
            # Risk of parity chi_s (mapped to {0,1}):
            # Pr[chi_s(x) != y] = (1 - hat{tilde_phi}(s)) / 2
            fc = self.sim.fourier_coefficient(s)
            parity_risk = (1 - abs(fc)) / 2
            if parity_risk < best_parity_risk:
                best_parity_risk = parity_risk
                best_s = s

        excess_risk = risk - best_parity_risk

        return {
            'risk': risk,
            'best_parity_risk': best_parity_risk,
            'best_parity_s': best_s,
            'excess_risk': excess_risk,
            'epsilon': transcript.epsilon,
            'agnostic_bound_met': excess_risk <= transcript.epsilon + 0.02,
        }
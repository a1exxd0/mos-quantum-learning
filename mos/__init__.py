"""
Mixture-of-Superpositions (MoS) State — Definition 8 of Caro et al.

Represents the mixed quantum example state:

    rho_D = E_{f ~ F_D} [ |psi_{U_n, f}><psi_{U_n, f}| ]

where F_D is the distribution over Boolean functions induced by independently
sampling f(x) ~ Bernoulli(phi(x)) for each x in {0,1}^n, and

    |psi_{U_n, f}> = (1/sqrt(2^n)) sum_x |x, f(x)>

This class handles ONLY state-level concerns:
  - Storing the distribution D = (U_n, phi)
  - Sampling f ~ F_D
  - Preparing |psi_f> as a Statevector or QuantumCircuit
  - Approximating rho_D via Monte Carlo
  - Recovering classical samples via computational basis measurement (Lemma 1)

It does NOT handle Hadamard measurement, post-selection, Fourier sampling,
heavy coefficient extraction, or anything verification-related.

Conventions:
  - phi(x) = Pr[y=1 | x] in [0, 1]          (the {0,1}-valued label expectation)
  - tilde_phi(x) = 1 - 2*phi(x) in [-1, 1]  (the {-1,1}-valued label expectation)
  - Qiskit little-endian: integer x = sum_i x_i * 2^i
  - Qubits 0..n-1 hold x, qubit n holds the label bit b
"""

import numpy as np
from typing import Callable, Union, Optional, Tuple
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, DensityMatrix


class MoSState:
    """
    Mixture-of-Superpositions quantum example state (Definition 8).

    Parameters
    ----------
    n : int
        Number of input bits (dimension of X_n = {0,1}^n).
    phi : callable or array-like
        The conditional probability function phi(x) = Pr[y=1 | x].
        If callable: phi(x: int) -> float in [0, 1].
        If array: phi[x] for x in 0..2^n - 1, values in [0, 1].
    noise_rate : float
        Label-flip noise rate eta in [0, 0.5]. When eta > 0, each label
        is independently flipped with probability eta before state preparation.
        This corresponds to the MoS noisy functional setting, Definition 5(iii).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n: int,
        phi: Union[Callable[[int], float], np.ndarray],
        noise_rate: float = 0.0,
        seed: Optional[int] = None,
    ):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not 0.0 <= noise_rate <= 0.5:
            raise ValueError(f"noise_rate must be in [0, 0.5], got {noise_rate}")

        self.n: int = n
        self.dim_x: int = 2**n
        self.dim_total: int = 2 ** (n + 1)
        self.noise_rate: float = noise_rate
        self._rng: Generator = default_rng(seed)

        # Store phi as array
        if callable(phi):
            self._phi = np.array([phi(x) for x in range(self.dim_x)], dtype=np.float64)
        else:
            self._phi = np.asarray(phi, dtype=np.float64).copy()
            if len(self._phi) != self.dim_x:
                raise ValueError(
                    f"phi must have length 2^n = {self.dim_x}, got {len(self._phi)}"
                )

        if not np.all((self._phi >= 0.0) & (self._phi <= 1.0)):
            raise ValueError("All phi values must be in [0, 1]")

        # Precompute effective phi under noise:
        #   phi_eff(x) = (1 - 2*eta) * phi(x) + eta
        # This is the effective Pr[y=1|x] after independent label flips.
        eta = self.noise_rate
        self._phi_effective: np.ndarray = (1 - 2 * eta) * self._phi + eta

    # ------------------------------------------------------------------
    # Properties: access phi in both {0,1} and {-1,1} conventions
    # ------------------------------------------------------------------

    @property
    def phi(self) -> np.ndarray:
        """phi(x) = Pr[y=1|x] in [0, 1] for all x (noiseless)."""
        return self._phi

    @property
    def tilde_phi(self) -> np.ndarray:
        """tilde_phi(x) = 1 - 2*phi(x) in [-1, 1] for all x (noiseless)."""
        return 1.0 - 2.0 * self._phi

    @property
    def phi_effective(self) -> np.ndarray:
        """Effective phi after noise: (1 - 2*eta)*phi(x) + eta."""
        return self._phi_effective

    @property
    def tilde_phi_effective(self) -> np.ndarray:
        """Effective tilde_phi after noise: (1 - 2*eta) * tilde_phi(x)."""
        return 1.0 - 2.0 * self._phi_effective

    # ------------------------------------------------------------------
    # Sampling f ~ F_D (Definition 8)
    # ------------------------------------------------------------------

    def sample_f(self, rng: Optional[Generator] = None) -> np.ndarray:
        """
        Sample a random Boolean function f ~ F_D.

        For each x in {0,1}^n, independently sample f(x) ~ Bernoulli(phi_eff(x)).
        When noise_rate > 0, phi_eff incorporates the label-flip noise.

        Parameters
        ----------
        rng : Generator, optional
            NumPy random generator. Uses internal RNG if not provided.

        Returns
        -------
        f : np.ndarray of shape (2^n,), dtype=np.uint8
            f[x] is the value f(x) in {0, 1}.
        """
        if rng is None:
            rng = self._rng
        return (rng.random(self.dim_x) < self._phi_effective).astype(np.uint8)

    # ------------------------------------------------------------------
    # Pure state preparation: |psi_{U_n, f}>
    # ------------------------------------------------------------------

    def statevector_f(self, f: np.ndarray) -> Statevector:
        """
        Construct the Qiskit Statevector |psi_{U_n, f}> for a fixed function f.

            |psi_{U_n, f}> = (1/sqrt(2^n)) * sum_x |x, f(x)>

        In Qiskit's little-endian convention, |x, b> maps to index x + b * 2^n
        since qubit n (the label) is the highest-index qubit.

        Parameters
        ----------
        f : np.ndarray of shape (2^n,), dtype=np.uint8
            Boolean function values.

        Returns
        -------
        sv : Statevector
            The (n+1)-qubit state |psi_{U_n, f}>.
        """
        sv_data = np.zeros(self.dim_total, dtype=np.complex128)
        amp = 1.0 / np.sqrt(self.dim_x)

        for x in range(self.dim_x):
            idx = x + int(f[x]) * self.dim_x
            sv_data[idx] = amp

        return Statevector(sv_data)

    # ------------------------------------------------------------------
    # Circuit preparation of |psi_{U_n, f}>
    # ------------------------------------------------------------------

    def _circuit_oracle_f(self, f: np.ndarray) -> QuantumCircuit:
        """
        Build an oracle circuit U_f mapping |x>|0> -> |x>|f(x)>.

        For each x where f(x) = 1, applies a multi-controlled X gate on the
        label qubit, controlled on the input register being |x>.

        Parameters
        ----------
        f : np.ndarray
            Boolean function values.

        Returns
        -------
        qc : QuantumCircuit
            Oracle circuit on n+1 qubits.
        """
        qr = QuantumRegister(self.n + 1, "q")
        qc = QuantumCircuit(qr, name="oracle_f")

        for x in range(self.dim_x):
            if f[x] == 1:
                ctrl_state = format(x, f"0{self.n}b")
                if self.n == 1:
                    # Single-controlled X
                    qc.cx(0, 1, ctrl_state=ctrl_state)
                else:
                    qc.mcx(
                        control_qubits=list(range(self.n)),
                        target_qubit=self.n,
                        ctrl_state=ctrl_state,
                    )

        return qc

    def circuit_prepare_f(self, f: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit preparing |psi_{U_n, f}> via H^n + oracle.

            |0>^{n+1}  --[H^n ⊗ I]-->  |+>^n|0>  --[U_f]-->  |psi_{U_n, f}>

        This is more hardware-friendly than arbitrary state initialisation.

        Parameters
        ----------
        f : np.ndarray
            Boolean function values.

        Returns
        -------
        qc : QuantumCircuit
            Circuit on n+1 qubits that prepares |psi_{U_n, f}>.
        """
        qr = QuantumRegister(self.n + 1, "q")
        qc = QuantumCircuit(qr, name="prepare_psi_f")

        # Uniform superposition on the x-register
        for i in range(self.n):
            qc.h(qr[i])

        # Apply oracle to entangle label register
        oracle = self._circuit_oracle_f(f)
        qc.compose(oracle, inplace=True)

        return qc

    def circuit_prepare_f_initialize(self, f: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit preparing |psi_{U_n, f}> via Qiskit's Initialize.

        Exact but synthesises an arbitrary state preparation unitary —
        less portable to real hardware.

        Parameters
        ----------
        f : np.ndarray
            Boolean function values.

        Returns
        -------
        qc : QuantumCircuit
            Circuit on n+1 qubits.
        """
        sv = self.statevector_f(f)
        qr = QuantumRegister(self.n + 1, "q")
        qc = QuantumCircuit(qr, name="prepare_psi_f_init")
        qc.initialize(sv, qr)
        return qc

    # ------------------------------------------------------------------
    # Density matrix: rho_D = E_{f ~ F_D}[|psi_f><psi_f|]
    # ------------------------------------------------------------------

    def density_matrix(
        self,
        num_samples: int = 1000,
        rng: Optional[Generator] = None,
    ) -> DensityMatrix:
        """
        Approximate rho_D by Monte Carlo averaging over sampled f.

            rho_D ≈ (1/M) sum_{m=1}^{M} |psi_{f_m}><psi_{f_m}|

        Parameters
        ----------
        num_samples : int
            Number of functions f to sample (M).
        rng : Generator, optional
            NumPy random generator.

        Returns
        -------
        rho : DensityMatrix
            Monte Carlo estimate of the MoS density matrix.
        """
        if rng is None:
            rng = self._rng

        rho_data = np.zeros((self.dim_total, self.dim_total), dtype=np.complex128)

        for _ in range(num_samples):
            f = self.sample_f(rng)
            sv = self.statevector_f(f)
            rho_data += np.outer(sv.data, sv.data.conj())

        rho_data /= num_samples
        return DensityMatrix(rho_data)

    # ------------------------------------------------------------------
    # Classical sampling: computational basis measurement (Lemma 1)
    # ------------------------------------------------------------------

    def sample_classical(
        self,
        rng: Optional[Generator] = None,
    ) -> Tuple[int, int]:
        """
        Draw a classical sample (x, y) ~ D by measuring rho_D in the
        computational basis.

        By Lemma 1, this is equivalent to drawing (x, y) ~ D directly:
          1. Sample f ~ F_D
          2. Prepare |psi_{U_n, f}>
          3. Measure in computational basis
          -> yields (x, f(x)) with x ~ U_n

        Equivalently (and more efficiently), we can just sample x ~ U_n
        and y ~ Bernoulli(phi_eff(x)) directly.

        Parameters
        ----------
        rng : Generator, optional
            NumPy random generator.

        Returns
        -------
        x : int
            Input in {0, ..., 2^n - 1}.
        y : int
            Label in {0, 1}.
        """
        if rng is None:
            rng = self._rng

        x = rng.integers(0, self.dim_x)
        y = int(rng.random() < self._phi_effective[x])
        return x, y

    def sample_classical_batch(
        self,
        num_samples: int,
        rng: Optional[Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw a batch of classical samples (x_i, y_i) ~ D.

        Parameters
        ----------
        num_samples : int
            Number of samples.
        rng : Generator, optional
            NumPy random generator.

        Returns
        -------
        xs : np.ndarray of shape (num_samples,), dtype=int
            Input values.
        ys : np.ndarray of shape (num_samples,), dtype=int
            Label values.
        """
        if rng is None:
            rng = self._rng

        xs = rng.integers(0, self.dim_x, size=num_samples)
        ys = (rng.random(num_samples) < self._phi_effective[xs]).astype(np.uint8)
        return xs, ys

    # ------------------------------------------------------------------
    # Fourier analysis (for validation / ground truth)
    # ------------------------------------------------------------------

    def fourier_coefficient(self, s: int) -> float:
        """
        Compute the exact Fourier coefficient hat{tilde_phi}(s).

            hat{tilde_phi}(s) = E_{x ~ U_n}[tilde_phi(x) * chi_s(x)]

        where chi_s(x) = (-1)^{s · x} and tilde_phi = 1 - 2*phi (noiseless).

        Parameters
        ----------
        s : int
            Frequency index in {0, ..., 2^n - 1}.

        Returns
        -------
        coeff : float
            The Fourier coefficient hat{tilde_phi}(s).
        """
        tphi = self.tilde_phi
        # Compute (-1)^{popcount(s & x)} for all x
        parities = np.array([bin(s & x).count("1") % 2 for x in range(self.dim_x)])
        chi_s = 1.0 - 2.0 * parities  # (-1)^{s·x}
        return float(np.mean(tphi * chi_s))

    def fourier_coefficient_effective(self, s: int) -> float:
        """
        Compute hat{tilde_phi_eff}(s) = (1 - 2*eta) * hat{tilde_phi}(s).

        This is the Fourier coefficient of the noise-adjusted tilde_phi,
        which governs the actual sampling distribution from Theorem 5
        when noise_rate > 0.

        Parameters
        ----------
        s : int
            Frequency index in {0, ..., 2^n - 1}.

        Returns
        -------
        coeff : float
            The effective Fourier coefficient.
        """
        return (1.0 - 2.0 * self.noise_rate) * self.fourier_coefficient(s)

    def fourier_spectrum(self) -> np.ndarray:
        """
        Compute the full Fourier spectrum {hat{tilde_phi}(s)} for all s.

        Returns
        -------
        spectrum : np.ndarray of shape (2^n,)
            spectrum[s] = hat{tilde_phi}(s).
        """
        return np.array([self.fourier_coefficient(s) for s in range(self.dim_x)])

    def parseval_check(self) -> Tuple[float, float]:
        """
        Verify Parseval's identity: sum_s hat{tilde_phi}(s)^2 = E[tilde_phi(x)^2].

        Returns
        -------
        fourier_sum : float
            sum_s hat{tilde_phi}(s)^2
        expected_sq : float
            E_{x ~ U_n}[tilde_phi(x)^2]
        """
        spectrum = self.fourier_spectrum()
        fourier_sum = float(np.sum(spectrum**2))
        expected_sq = float(np.mean(self.tilde_phi**2))
        return fourier_sum, expected_sq

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        noise_str = f", noise_rate={self.noise_rate}" if self.noise_rate > 0 else ""
        return f"MoSState(n={self.n}{noise_str})"

    def summary(self) -> str:
        """Human-readable summary of the MoS state."""
        tphi = self.tilde_phi
        fourier_sum, expected_sq = self.parseval_check()

        lines = [
            "MoS State (Definition 8)",
            f"  n = {self.n}, dim = 2^n = {self.dim_x}",
            f"  noise_rate = {self.noise_rate}",
            f"  E[tilde_phi^2] = {expected_sq:.6f}",
            f"  Parseval check: sum hat(s)^2 = {fourier_sum:.6f}",
            f"  phi range: [{self._phi.min():.4f}, {self._phi.max():.4f}]",
            f"  tilde_phi range: [{tphi.min():.4f}, {tphi.max():.4f}]",
        ]

        # Show nonzero Fourier coefficients if manageable
        if self.dim_x <= 64:
            spectrum = self.fourier_spectrum()
            nonzero = [
                (s, spectrum[s]) for s in range(self.dim_x) if abs(spectrum[s]) > 1e-10
            ]
            lines.append(f"  Nonzero Fourier coefficients: {len(nonzero)}")
            for s, coeff in sorted(nonzero, key=lambda t: abs(t[1]), reverse=True):
                bits = format(s, f"0{self.n}b")
                lines.append(f"    s={s} ({bits}): {coeff:+.6f}")

        return "\n".join(lines)

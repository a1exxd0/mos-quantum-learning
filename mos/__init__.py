r"""
Mixture-of-Superpositions (MoS) State - Definition 8 of Caro et al.

Represents the mixed quantum example state:

.. math::

    \rho_D = \mathbb{E}_{f \sim F_D}
    \bigl[|\psi_{U_n, f}\rangle\langle\psi_{U_n, f}|\bigr]

where :math:`F_D` is the distribution over Boolean functions induced by
independently sampling :math:`f(x) \sim \text{Bernoulli}(\phi_{\text{eff}}(x))`
for each :math:`x \in \{0,1\}^n`, and

.. math::

    |\psi_{U_n, f}\rangle = \frac{1}{\sqrt{2^n}} \sum_x |x, f(x)\rangle

**Noise model (Definition 5(iii)).**  When ``noise_rate`` :math:`\eta > 0`,
each label is independently flipped with probability :math:`\eta` before
state preparation.  The *effective* conditional label probability becomes

.. math::

    \phi_{\text{eff}}(x) = (1 - 2\eta)\,\phi(x) + \eta

and the effective :math:`\{-1,1\}`-valued label expectation is

.. math::

    \tilde\phi_{\text{eff}}(x) = (1 - 2\eta)\,\tilde\phi(x).

The MoS state :math:`\rho_D` is constructed from :math:`\phi_{\text{eff}}`,
so computational-basis measurement yields samples from the *noisy*
distribution (Lemma 1), and Quantum Fourier Sampling (Theorem 5)
produces outcomes governed by the *effective* Fourier coefficients
:math:`\hat{\tilde\phi}_{\text{eff}}(s) = (1 - 2\eta)\,\hat{\tilde\phi}(s)`.

All Fourier-analytic methods accept an ``effective`` flag (default
``True``) that controls whether the returned quantities incorporate the
noise factor :math:`(1 - 2\eta)`.  Set ``effective=False`` to obtain the
noiseless ground-truth coefficients of :math:`\tilde\phi`.

This class handles ONLY state-level concerns:

- Storing the distribution :math:`D = (U_n, \phi)` and its noise model
- Sampling :math:`f \sim F_D`
- Preparing :math:`|\psi_f\rangle` as a Statevector or QuantumCircuit
- Approximating :math:`\rho_D` via Monte Carlo
- Recovering classical samples via computational basis measurement (Lemma 1)
- Exact Fourier analysis of :math:`\tilde\phi` (noiseless or effective)

It does NOT handle Hadamard measurement, post-selection, Fourier sampling,
heavy coefficient extraction, or anything verification-related.

Conventions:

- :math:`\phi(x) = \Pr[y{=}1 \mid x] \in [0, 1]` — the {0,1}-valued label expectation
- :math:`\tilde\phi(x) = 1 - 2\phi(x) \in [-1, 1]` — the {-1,1}-valued label expectation
- Qiskit little-endian: integer :math:`x = \sum_i x_i \cdot 2^i`
- Qubits 0..n-1 hold x, qubit n holds the label bit b
"""

import numpy as np
from typing import Callable, Union, Optional, Tuple
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, DensityMatrix


class MoSState:
    r"""
    Mixture-of-Superpositions quantum example state (Definition 8).

    Parameters
    ----------
    n : int
        Number of input bits (dimension of :math:`X_n = \{0,1\}^n`).
    phi : callable or array-like
        The conditional probability function :math:`\phi(x) = \Pr[y{=}1 \mid x]`.
        If callable: ``phi(x: int) -> float`` in [0, 1].
        If array: ``phi[x]`` for x in 0..2^n - 1, values in [0, 1].
    noise_rate : float
        Label-flip noise rate :math:`\eta \in [0, 0.5]`. When :math:`\eta > 0`,
        each label is independently flipped with probability :math:`\eta` before
        state preparation.  This corresponds to the MoS noisy functional setting,
        Definition 5(iii).  The MoS state is constructed from the *effective*
        label probabilities :math:`\phi_{\text{eff}}(x) = (1-2\eta)\phi(x)+\eta`,
        so all quantum operations (state preparation, QFS, classical sampling)
        see the noisy distribution.
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

        # Precompute noise damping factor for Fourier coefficients
        self._noise_damping: float = 1.0 - 2.0 * eta

    # ------------------------------------------------------------------
    # Properties: access phi in both {0,1} and {-1,1} conventions
    # ------------------------------------------------------------------

    @property
    def phi(self) -> np.ndarray:
        r""":math:`\phi(x) = \Pr[y{=}1 \mid x]` in [0, 1] for all x (noiseless)."""
        return self._phi

    @property
    def tilde_phi(self) -> np.ndarray:
        r""":math:`\tilde\phi(x) = 1 - 2\phi(x)` in [-1, 1] for all x (noiseless)."""
        return 1.0 - 2.0 * self._phi

    @property
    def phi_effective(self) -> np.ndarray:
        r"""Effective phi after noise: :math:`(1 - 2\eta)\phi(x) + \eta`."""
        return self._phi_effective

    @property
    def tilde_phi_effective(self) -> np.ndarray:
        r"""Effective tilde_phi after noise: :math:`(1 - 2\eta) \tilde\phi(x)`."""
        return 1.0 - 2.0 * self._phi_effective

    # ------------------------------------------------------------------
    # Sampling f ~ F_D (Definition 8)
    # ------------------------------------------------------------------

    def sample_f(self, rng: Optional[Generator] = None) -> np.ndarray:
        r"""
        Sample a random Boolean function :math:`f \sim F_D`.

        For each :math:`x \in \{0,1\}^n`, independently sample
        :math:`f(x) \sim \text{Bernoulli}(\phi_{\text{eff}}(x))`.

        When ``noise_rate > 0``, :math:`\phi_{\text{eff}}` incorporates
        the label-flip noise, so the sampled :math:`f` is drawn from
        the noisy MoS distribution.

        Parameters
        ----------
        rng : Generator, optional
            NumPy random generator. Uses internal RNG if not provided.

        Returns
        -------
        f : np.ndarray of shape (2^n,), dtype=np.uint8
            ``f[x]`` is the value f(x) in {0, 1}.
        """
        if rng is None:
            rng = self._rng
        return (rng.random(self.dim_x) < self._phi_effective).astype(np.uint8)

    # ------------------------------------------------------------------
    # Pure state preparation: |psi_{U_n, f}>
    # ------------------------------------------------------------------

    def statevector_f(self, f: np.ndarray) -> Statevector:
        r"""
        Construct the Qiskit Statevector :math:`|\psi_{U_n, f}\rangle` for a
        fixed function f.

        .. math::

            |\psi_{U_n, f}\rangle
            = \frac{1}{\sqrt{2^n}} \sum_x |x,\, f(x)\rangle

        In Qiskit's little-endian convention, :math:`|x, b\rangle` maps to
        index :math:`x + b \cdot 2^n` since qubit n (the label) is the
        highest-index qubit.

        Parameters
        ----------
        f : np.ndarray of shape (2^n,), dtype=np.uint8
            Boolean function values.

        Returns
        -------
        sv : Statevector
            The (n+1)-qubit state :math:`|\psi_{U_n, f}\rangle`.
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
        r"""
        Build an oracle circuit :math:`U_f` mapping
        :math:`|x\rangle|0\rangle \to |x\rangle|f(x)\rangle`.

        For each x where f(x) = 1, applies a multi-controlled X gate on the
        label qubit, controlled on the input register being :math:`|x\rangle`.

        Note: This constructs up to :math:`2^n` multi-controlled gates and is
        intended for simulation only (impractical for n > ~10).

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
                # ctrl_state is big-endian: bit string specifies which
                # computational basis state |x> activates the gate.
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
        r"""
        Build a circuit preparing :math:`|\psi_{U_n, f}\rangle` via
        :math:`H^{\otimes n}` + oracle.

        .. math::

            |0\rangle^{\otimes(n+1)}
            \xrightarrow{H^{\otimes n} \otimes I}
            |{+}\rangle^{\otimes n}|0\rangle
            \xrightarrow{U_f}
            |\psi_{U_n, f}\rangle

        This is more hardware-friendly than arbitrary state initialisation.

        Parameters
        ----------
        f : np.ndarray
            Boolean function values.

        Returns
        -------
        qc : QuantumCircuit
            Circuit on n+1 qubits that prepares
            :math:`|\psi_{U_n, f}\rangle`.
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
        r"""
        Build a circuit preparing :math:`|\psi_{U_n, f}\rangle` via Qiskit's
        Initialize.

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
        r"""
        Approximate :math:`\rho_D` by Monte Carlo averaging over sampled f.

        .. math::

            \rho_D \approx \frac{1}{M}
            \sum_{m=1}^{M} |\psi_{f_m}\rangle\langle\psi_{f_m}|

        The functions :math:`f_m` are sampled using :math:`\phi_{\text{eff}}`,
        so the resulting density matrix incorporates any label-flip noise.

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
        r"""
        Draw a classical sample :math:`(x, y)` by measuring
        :math:`\rho_D` in the computational basis.

        By Lemma 1, this is equivalent to sampling from the distribution
        :math:`D` encoded by the MoS state.  When ``noise_rate > 0``, the
        sampled labels reflect the noisy distribution (i.e. :math:`y` is
        drawn from :math:`\phi_{\text{eff}}(x)`).

        Equivalently (and more efficiently), we sample
        :math:`x \sim U_n` and :math:`y \sim \text{Bernoulli}(\phi_{\text{eff}}(x))`
        directly.

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
        r"""
        Draw a batch of classical samples :math:`(x_i, y_i)`.

        Labels are drawn from :math:`\phi_{\text{eff}}` (see
        :meth:`sample_classical`).

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
    # Fourier analysis
    # ------------------------------------------------------------------

    def fourier_coefficient(self, s: int, *, effective: bool = True) -> float:
        r"""
        Compute the Fourier coefficient :math:`\hat{\tilde\phi}(s)`.

        .. math::

            \hat{\tilde\phi}(s)
            = \mathbb{E}_{x \sim U_n}[\tilde\phi(x) \cdot \chi_s(x)]

        where :math:`\chi_s(x) = (-1)^{s \cdot x}` and
        :math:`\tilde\phi = 1 - 2\phi`.

        Parameters
        ----------
        s : int
            Frequency index in {0, ..., 2^n - 1}.
        effective : bool
            If True (default), return the noise-adjusted coefficient
            :math:`\hat{\tilde\phi}_{\text{eff}}(s) = (1-2\eta)\hat{\tilde\phi}(s)`,
            which governs the actual QFS sampling distribution (Theorem 5 /
            Lemma 6).  If False, return the noiseless coefficient.

        Returns
        -------
        coeff : float
            The Fourier coefficient.
        """
        tphi = self.tilde_phi  # always compute from noiseless
        # Compute (-1)^{popcount(s & x)} for all x
        parities = np.array([bin(s & x).count("1") % 2 for x in range(self.dim_x)])
        chi_s = 1.0 - 2.0 * parities  # (-1)^{s·x}
        coeff = float(np.mean(tphi * chi_s))
        if effective:
            coeff *= self._noise_damping
        return coeff

    def fourier_spectrum(self, *, effective: bool = True) -> np.ndarray:
        r"""
        Compute the full Fourier spectrum for all s.

        Parameters
        ----------
        effective : bool
            If True (default), return the noise-adjusted spectrum
            :math:`\{\hat{\tilde\phi}_{\text{eff}}(s)\}_s`.
            If False, return the noiseless spectrum
            :math:`\{\hat{\tilde\phi}(s)\}_s`.

        Returns
        -------
        spectrum : np.ndarray of shape (2^n,)
            ``spectrum[s]`` is the Fourier coefficient at frequency s.
        """
        # Compute noiseless spectrum first (avoids repeated damping)
        tphi = self.tilde_phi
        spectrum = np.empty(self.dim_x, dtype=np.float64)
        for s in range(self.dim_x):
            parities = np.array([bin(s & x).count("1") % 2 for x in range(self.dim_x)])
            chi_s = 1.0 - 2.0 * parities
            spectrum[s] = float(np.mean(tphi * chi_s))
        if effective:
            spectrum *= self._noise_damping
        return spectrum

    def parseval_check(self, *, effective: bool = True) -> Tuple[float, float]:
        r"""
        Verify Parseval's identity:
        :math:`\sum_s \hat{\tilde\phi}(s)^2 = \mathbb{E}[\tilde\phi(x)^2]`.

        When ``effective=True``, both sides are computed with the noise-adjusted
        :math:`\tilde\phi_{\text{eff}}`, so the identity remains valid.

        Parameters
        ----------
        effective : bool
            If True (default), check Parseval for the effective (noise-adjusted)
            spectrum and :math:`\tilde\phi_{\text{eff}}`.
            If False, check for the noiseless quantities.

        Returns
        -------
        fourier_sum : float
            :math:`\sum_s \hat{\tilde\phi}(s)^2` (or effective variant).
        expected_sq : float
            :math:`\mathbb{E}_{x \sim U_n}[\tilde\phi(x)^2]` (or effective variant).
        """
        spectrum = self.fourier_spectrum(effective=effective)
        fourier_sum = float(np.sum(spectrum**2))
        if effective:
            expected_sq = float(np.mean(self.tilde_phi_effective**2))
        else:
            expected_sq = float(np.mean(self.tilde_phi**2))
        return fourier_sum, expected_sq

    # ------------------------------------------------------------------
    # QFS sampling distribution (Theorem 5)
    # ------------------------------------------------------------------

    def qfs_probability(self, s: int) -> float:
        r"""
        Probability of observing frequency :math:`s` from Quantum Fourier
        Sampling, conditioned on the last qubit being 1 (Theorem 5).

        .. math::

            \Pr[s \mid b{=}1] = \frac{1}{2^n}
            \bigl(1 - \mathbb{E}_{x}[\tilde\phi_{\text{eff}}(x)^2]\bigr)
            + \hat{\tilde\phi}_{\text{eff}}(s)^2

        This always uses the effective (noise-adjusted) coefficients,
        since the QFS circuit acts on the physical MoS state.

        Parameters
        ----------
        s : int
            Frequency index in {0, ..., 2^n - 1}.

        Returns
        -------
        prob : float
            Conditional probability of observing s.
        """
        tphi_eff = self.tilde_phi_effective
        E_sq = float(np.mean(tphi_eff**2))
        coeff = self.fourier_coefficient(s, effective=True)
        return (1.0 - E_sq) / self.dim_x + coeff**2

    def qfs_distribution(self) -> np.ndarray:
        r"""
        Full QFS conditional distribution over :math:`\{0,1\}^n`,
        conditioned on the last qubit being 1 (Theorem 5).

        Always uses effective (noise-adjusted) coefficients.

        Returns
        -------
        dist : np.ndarray of shape (2^n,)
            ``dist[s]`` is the conditional probability of observing s.
        """
        spectrum_eff = self.fourier_spectrum(effective=True)
        tphi_eff = self.tilde_phi_effective
        E_sq = float(np.mean(tphi_eff**2))
        return (1.0 - E_sq) / self.dim_x + spectrum_eff**2

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        noise_str = f", noise_rate={self.noise_rate}" if self.noise_rate > 0 else ""
        return f"MoSState(n={self.n}{noise_str})"

    def summary(self, *, effective: bool = True) -> str:
        """
        Human-readable summary of the MoS state.

        Parameters
        ----------
        effective : bool
            If True (default), report the noise-adjusted Fourier spectrum
            that governs actual QFS outcomes.  If False, report noiseless
            ground-truth coefficients.
        """
        label = "effective" if effective else "noiseless"
        tphi = self.tilde_phi_effective if effective else self.tilde_phi
        fourier_sum, expected_sq = self.parseval_check(effective=effective)

        lines = [
            f"MoS State (Definition 8) — Fourier analysis: {label}",
            f"  n = {self.n}, dim = 2^n = {self.dim_x}",
            f"  noise_rate = {self.noise_rate}",
            f"  E[tilde_phi^2] = {expected_sq:.6f}",
            f"  Parseval check: sum hat(s)^2 = {fourier_sum:.6f}",
            f"  phi range: [{self._phi.min():.4f}, {self._phi.max():.4f}]",
            f"  tilde_phi range: [{tphi.min():.4f}, {tphi.max():.4f}]",
        ]

        if effective and self.noise_rate > 0:
            lines.append(f"  noise damping (1 - 2η) = {self._noise_damping:.6f}")

        # Show nonzero Fourier coefficients if manageable
        if self.dim_x <= 64:
            spectrum = self.fourier_spectrum(effective=effective)
            nonzero = [
                (s, spectrum[s]) for s in range(self.dim_x) if abs(spectrum[s]) > 1e-10
            ]
            lines.append(f"  Nonzero Fourier coefficients: {len(nonzero)}")
            for s, coeff in sorted(nonzero, key=lambda t: abs(t[1]), reverse=True):
                bits = format(s, f"0{self.n}b")
                lines.append(f"    s={s} ({bits}): {coeff:+.6f}")

        return "\n".join(lines)

"""
Mixture-of-Superpositions with Qiskit Circuits

This module implements a reusable simulator for the MoS quantum example model
using proper Qiskit circuits that can run on simulators or real quantum hardware.

Bit-ordering convention (Qiskit little-endian):
    Integer x is encoded as |x> where x = sum_i x_i * 2^i.
    Qubits 0..n-1 hold x, qubit n holds the label bit b.

Convention for phi:
    phi(x) ∈ [0,1] represents Pr[y=1 | x].
    Internally we convert to tilde_phi(x) = 2*phi(x) - 1 ∈ [-1,1] where needed.
"""

import numpy as np
from typing import Callable, Union, Optional
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class MoSSimulator:
    """
    Simulator for the Mixture-of-Superpositions (MoS) quantum example model.
    
    Parameters
    ----------
    n : int
        Number of input bits (dimension of X_n = {0,1}^n).
    phi : callable or array-like
        The conditional probability function phi(x) = Pr[y=1 | x].
        If callable: phi(x: int) -> float in [0,1].
        If array: phi[x] for x in 0..2^n-1, values in [0,1].
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n: int,
        phi: Union[Callable[[int], float], np.ndarray],
        seed: Optional[int] = None
    ):
        self.n = n
        self.dim_x = 2 ** n
        self.dim_total = 2 ** (n + 1)
        self.rng = default_rng(seed)
        
        # Store phi as array for efficiency
        if callable(phi):
            self._phi = np.array([phi(x) for x in range(self.dim_x)], dtype=np.float64)
        else:
            self._phi = np.asarray(phi, dtype=np.float64)
            assert len(self._phi) == self.dim_x, f"phi must have length 2^n = {self.dim_x}"
        
        assert np.all((self._phi >= 0) & (self._phi <= 1)), "phi values must be in [0,1]"
        
        # Default backend
        self._backend = AerSimulator()
    
    @property
    def phi(self) -> np.ndarray:
        """phi(x) = Pr[y=1|x] in [0,1] for all x."""
        return self._phi
    
    @property
    def tilde_phi(self) -> np.ndarray:
        """tilde_phi(x) = 2*phi(x) - 1 in [-1,1] for all x."""
        return 2 * self._phi - 1
    
    def set_backend(self, backend):
        """Set the Qiskit backend for circuit execution."""
        self._backend = backend
    
    def sample_f(self, rng: Optional[Generator] = None) -> np.ndarray:
        """
        Sample a random Boolean function f ~ F_D.
        
        For each x in {0,1}^n, independently sample f(x) ~ Bernoulli(phi(x)).
        
        Returns
        -------
        f : np.ndarray of shape (2^n,), dtype=np.uint8
            f[x] is the value f(x) in {0, 1}.
        """
        if rng is None:
            rng = self.rng
        return (rng.random(self.dim_x) < self._phi).astype(np.uint8)
    
    def statevector_from_f(self, f: np.ndarray) -> Statevector:
        """
        Construct the Qiskit Statevector |phi_(U_n, f)> for a fixed function f.
        
        |phi_(U_n, f)> = (1/sqrt(2^n)) * sum_x |x, f(x)>
        """
        sv_data = np.zeros(self.dim_total, dtype=np.complex128)
        amp = 1.0 / np.sqrt(self.dim_x)
        
        for x in range(self.dim_x):
            idx = x + int(f[x]) * self.dim_x
            sv_data[idx] = amp
        
        return Statevector(sv_data)
    
    def circuit_prepare_state(self, f: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that prepares |phi_(U_n, f)> for a fixed f.
        
        Uses Qiskit's Initialize instruction which synthesizes the state
        preparation circuit automatically.
        
        Parameters
        ----------
        f : np.ndarray
            The Boolean function values.
        
        Returns
        -------
        qc : QuantumCircuit
            Circuit on n+1 qubits that prepares the state.
        """
        sv = self.statevector_from_f(f)
        
        qr = QuantumRegister(self.n + 1, 'q')
        qc = QuantumCircuit(qr, name='prepare_phi_f')
        
        # Initialize to the target statevector
        qc.initialize(sv, qr)
        
        return qc
    
    def circuit_prepare_uniform_superposition(self) -> QuantumCircuit:
        """
        Build a circuit that prepares uniform superposition over x: |+>^n ⊗ |0>.
        
        This is the first step before applying the oracle.
        """
        qr = QuantumRegister(self.n + 1, 'q')
        qc = QuantumCircuit(qr, name='uniform_superposition')
        
        # Apply H to first n qubits to get uniform superposition over x
        for i in range(self.n):
            qc.h(qr[i])
        
        return qc
    
    def circuit_oracle_f(self, f: np.ndarray) -> QuantumCircuit:
        """
        Build an oracle circuit U_f that maps |x>|0> -> |x>|f(x)>.
        
        This implements the standard phase-free oracle using multi-controlled X gates.
        For each x where f(x)=1, flip the target qubit conditioned on input being x.
        
        Parameters
        ----------
        f : np.ndarray
            The Boolean function values.
        
        Returns
        -------
        qc : QuantumCircuit
            Oracle circuit on n+1 qubits.
        """
        qr = QuantumRegister(self.n + 1, 'q')
        qc = QuantumCircuit(qr, name='oracle_f')
        
        # For each x where f(x) = 1, apply MCX controlled on |x>
        for x in range(self.dim_x):
            if f[x] == 1:
                # Determine which qubits need X gates (for 0 bits in x)
                ctrl_state = format(x, f'0{self.n}b')[::-1]  # reversed for Qiskit ordering
                
                if self.n == 0:
                    qc.x(qr[0])
                else:
                    qc.mcx(
                        control_qubits=list(range(self.n)),
                        target_qubit=self.n,
                        ctrl_state=ctrl_state
                    )
        
        return qc
    
    def circuit_prepare_with_oracle(self, f: np.ndarray) -> QuantumCircuit:
        """
        Build circuit that prepares |phi_(U_n,f)> using H gates + oracle.
        
        |0>^{n+1} --H^n⊗I-- |+>^n|0> --U_f-- (1/sqrt(2^n)) sum_x |x,f(x)>
        
        This is more hardware-friendly than arbitrary state initialization.
        """
        qr = QuantumRegister(self.n + 1, 'q')
        qc = QuantumCircuit(qr, name='prepare_phi_f')
        
        # Uniform superposition on x register
        for i in range(self.n):
            qc.h(qr[i])
        
        # Apply oracle
        oracle = self.circuit_oracle_f(f)
        qc.compose(oracle, inplace=True)
        
        return qc
    
    def circuit_hadamard_measure(
        self,
        f: np.ndarray,
        use_oracle: bool = True
    ) -> QuantumCircuit:
        """
        Build complete circuit: prepare |phi_f>, apply H^{n+1}, measure.
        
        Parameters
        ----------
        f : np.ndarray
            The Boolean function values.
        use_oracle : bool
            If True, use H+oracle preparation (hardware-friendly).
            If False, use Initialize instruction (exact but less portable).
        
        Returns
        -------
        qc : QuantumCircuit
            Complete circuit with measurements.
        """
        qr = QuantumRegister(self.n + 1, 'q')
        cr = ClassicalRegister(self.n + 1, 'c')
        qc = QuantumCircuit(qr, cr, name='mos_hadamard_measure')
        
        # State preparation
        if use_oracle:
            # H on x-register, then oracle
            for i in range(self.n):
                qc.h(qr[i])
            oracle = self.circuit_oracle_f(f)
            qc.compose(oracle, qubits=qr, inplace=True)
        else:
            # Direct initialization
            sv = self.statevector_from_f(f)
            qc.initialize(sv, qr)
        
        qc.barrier()
        
        # Apply H^{n+1}
        for i in range(self.n + 1):
            qc.h(qr[i])
        
        qc.barrier()
        
        # Measure all qubits
        qc.measure(qr, cr)
        
        return qc
    
    def rho_estimate(self, M: int, rng: Optional[Generator] = None) -> DensityMatrix:
        """
        Approximate the MoS density matrix by Monte Carlo.
        
        rho_hat = (1/M) * sum_{m=1}^M |phi_m><phi_m|
        """
        if rng is None:
            rng = self.rng
        
        rho_data = np.zeros((self.dim_total, self.dim_total), dtype=np.complex128)
        
        for _ in range(M):
            f = self.sample_f(rng)
            sv = self.statevector_from_f(f)
            rho_data += np.outer(sv.data, sv.data.conj())
        
        rho_data /= M
        return DensityMatrix(rho_data)
    
    def sample_hadamard_measure_statevector(
        self,
        shots: int,
        rng: Optional[Generator] = None
    ) -> dict:
        """
        Simulate Hadamard+measure using statevector sampling (per-shot mixture).
        
        Fast simulation that samples f per shot and uses Statevector.sample_counts.
        
        Returns
        -------
        counts : dict
            Measurement counts as {bitstring: count}.
        """
        if rng is None:
            rng = self.rng
        
        # Precompute H^{n+1} operator
        H = Operator([[1, 1], [1, -1]]) / np.sqrt(2) # type: ignore
        H_all = H
        for _ in range(self.n):
            H_all = H_all.tensor(H)
        
        counts = {}
        for _ in range(shots):
            f = self.sample_f(rng)
            sv = self.statevector_from_f(f)
            sv_h = sv.evolve(H_all)
            
            # Sample one measurement
            result = sv_h.sample_counts(1)
            bitstring = list(result.keys())[0]
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def sample_hadamard_measure_circuit(
        self,
        shots: int,
        use_oracle: bool = True,
        rng: Optional[Generator] = None,
        backend=None
    ) -> dict:
        """
        Simulate Hadamard+measure using circuit execution (per-shot mixture).
        
        Each shot samples a new f and runs the corresponding circuit.
        This matches the operational MoS ensemble exactly.
        
        Parameters
        ----------
        shots : int
            Number of measurements.
        use_oracle : bool
            Use oracle-based preparation (hardware-friendly).
        rng : Generator, optional
            Random generator for sampling f.
        backend : optional
            Qiskit backend. If None, uses AerSimulator.
        
        Returns
        -------
        counts : dict
            Measurement counts as {bitstring: count}.
        """
        if rng is None:
            rng = self.rng
        if backend is None:
            backend = self._backend
        
        counts = {}
        
        for _ in range(shots):
            f = self.sample_f(rng)
            qc = self.circuit_hadamard_measure(f, use_oracle=use_oracle)
            
            # Transpile and run
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            transpiled = pm.run(qc)
            
            job = backend.run(transpiled, shots=1)
            result = job.result()
            shot_counts = result.get_counts()
            
            bitstring = list(shot_counts.keys())[0]
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def sample_hadamard_measure_batched(
        self,
        shots: int,
        batch_size: int = 100,
        use_oracle: bool = True,
        rng: Optional[Generator] = None,
        backend=None
    ) -> dict:
        """
        Batched circuit execution for efficiency.
        
        Groups shots into batches where each batch uses a single sampled f.
        This is an approximation but much faster for large shot counts.
        
        Parameters
        ----------
        shots : int
            Total number of measurements.
        batch_size : int
            Shots per sampled f. Smaller = more accurate MoS, larger = faster.
        use_oracle : bool
            Use oracle-based preparation.
        
        Returns
        -------
        counts : dict
            Measurement counts.
        """
        if rng is None:
            rng = self.rng
        if backend is None:
            backend = self._backend
        
        counts = {}
        remaining = shots
        
        while remaining > 0:
            batch = min(batch_size, remaining)
            f = self.sample_f(rng)
            qc = self.circuit_hadamard_measure(f, use_oracle=use_oracle)
            
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            transpiled = pm.run(qc)
            
            job = backend.run(transpiled, shots=batch)
            result = job.result()
            batch_counts = result.get_counts()
            
            for bitstring, count in batch_counts.items():
                counts[bitstring] = counts.get(bitstring, 0) + count
            
            remaining -= batch
        
        return counts
    
    def sample_hadamard_measure(
        self,
        shots: int,
        mode: str = "statevector",
        **kwargs
    ) -> dict:
        """
        Unified interface for Hadamard+measure experiment.
        
        Parameters
        ----------
        shots : int
            Number of measurement samples.
        mode : str
            "statevector" - fast simulation using Statevector class
            "circuit" - per-shot circuit execution (exact MoS)
            "batched" - batched circuits (faster approximation)
        **kwargs
            Passed to underlying method (e.g., batch_size, use_oracle, backend).
        
        Returns
        -------
        counts : dict
            Measurement counts as {bitstring: count}.
        """
        if mode == "statevector":
            return self.sample_hadamard_measure_statevector(shots, kwargs.get('rng'))
        elif mode == "circuit":
            return self.sample_hadamard_measure_circuit(shots, **kwargs)
        elif mode == "batched":
            return self.sample_hadamard_measure_batched(shots, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def analyze_counts(self, counts: dict) -> dict:
        """
        Analyze measurement counts from Hadamard+measure experiment.
        
        Parameters
        ----------
        counts : dict
            Measurement counts as {bitstring: count}.
        
        Returns
        -------
        analysis : dict
            'prob_last_1': empirical Pr[last qubit = 1]
            's_distribution': Pr[s | last=1]
            etc.
        """
        total = sum(counts.values())
        n_last_1 = 0
        n_last_0 = 0
        s_counts = {}
        
        for bitstring, count in counts.items():
            # Qiskit bitstrings are big-endian: rightmost bit is qubit 0
            # Last qubit (index n) is at position 0 from the left
            last_bit = int(bitstring[0])  # leftmost char = highest qubit = qubit n
            s_bits = bitstring[1:]  # remaining bits = qubits 0..n-1
            s = int(s_bits, 2) if s_bits else 0
            
            if last_bit == 1:
                n_last_1 += count
                s_counts[s] = s_counts.get(s, 0) + count
            else:
                n_last_0 += count
        
        s_dist = {}
        if n_last_1 > 0:
            for s, c in s_counts.items():
                s_dist[s] = c / n_last_1
        
        return {
            'prob_last_1': n_last_1 / total if total > 0 else 0.0,
            'prob_last_0': n_last_0 / total if total > 0 else 0.0,
            's_distribution': s_dist,
            's_counts': s_counts,
            'total_shots': total,
            'shots_last_1': n_last_1
        }
    
    def fourier_coefficient(self, s: int) -> float:
        """
        Compute Fourier coefficient hat{tilde_phi}(s).
        """
        tphi = self.tilde_phi
        parities = np.array([bin(s & x).count('1') % 2 for x in range(self.dim_x)])
        signs = 1 - 2 * parities
        return np.mean(tphi * signs) # type: ignore
    
    def theoretical_s_distribution(self) -> np.ndarray:
        """
        Compute theoretical Pr[s | last=1] for all s.
        """
        tphi = self.tilde_phi
        expected_tphi_sq = np.mean(tphi ** 2)
        base = (1 - expected_tphi_sq) / self.dim_x
        
        probs = np.zeros(self.dim_x)
        for s in range(self.dim_x):
            fc = self.fourier_coefficient(s)
            probs[s] = base + fc ** 2
        
        return probs


if __name__ == "__main__":
    print("=" * 60)
    print("MoS Simulator Demo (Qiskit Circuits)")
    print("=" * 60)
    
    n = 3
    
    # Parity-biased phi
    def phi_parity(x: int) -> float:
        parity = bin(x).count('1') % 2
        return 0.5 + 0.25 * (1 - 2 * parity)
    
    print(f"\nExample: n={n}, phi biased by parity")
    sim = MoSSimulator(n, phi_parity, seed=43)
    
    # Show a sample circuit
    print("\n--- Sample Circuit (oracle-based) ---")
    f_sample = sim.sample_f()
    print(f"Sampled f = {[int(b) for b in f_sample]}")
    qc = sim.circuit_hadamard_measure(f_sample, use_oracle=True)
    print(qc.draw())
    
    # Verify statevector preparation
    print("\n--- Statevector Verification ---")
    sv = sim.statevector_from_f(f_sample)
    print(f"Statevector norm: {np.linalg.norm(sv.data):.6f}")
    print(f"Non-zero amplitudes at indices: ", end="")
    nonzero = np.where(np.abs(sv.data) > 1e-10)[0]
    print(nonzero)
    
    # Run Hadamard+measure experiment
    shots = 10000
    print(f"\n--- Hadamard+Measure ({shots} shots, statevector mode) ---")
    counts = sim.sample_hadamard_measure(shots, mode="statevector")
    analysis = sim.analyze_counts(counts)
    
    print(f"Pr[last=1] = {analysis['prob_last_1']:.4f} (theory: 0.5)")
    print(f"Pr[last=0] = {analysis['prob_last_0']:.4f} (theory: 0.5)")
    
    # Compare distributions
    print("\nConditional distribution Pr[s | last=1]:")
    theory_dist = sim.theoretical_s_distribution()
    print(f"{'s':>4} {'bits':>6} {'empirical':>10} {'theory':>10}")
    print("-" * 36)
    for s in range(2**n):
        bits = format(s, f'0{n}b')
        emp = analysis['s_distribution'].get(s, 0.0)
        print(f"{s:>4} {bits:>6} {emp:>10.4f} {theory_dist[s]:>10.4f}")
    
    # Test circuit mode (slower but uses actual circuit execution)
    print(f"\n--- Circuit Mode Test (100 shots) ---")
    counts_circ = sim.sample_hadamard_measure(100, mode="circuit", use_oracle=True)
    analysis_circ = sim.analyze_counts(counts_circ)
    print(f"Pr[last=1] = {analysis_circ['prob_last_1']:.4f}")
    
    # Test batched mode
    print(f"\n--- Batched Mode Test (1000 shots, batch_size=50) ---")
    counts_batch = sim.sample_hadamard_measure(1000, mode="batched", batch_size=50)
    analysis_batch = sim.analyze_counts(counts_batch)
    print(f"Pr[last=1] = {analysis_batch['prob_last_1']:.4f}")
    
    # Show density matrix
    print("\n--- Density Matrix (M=100 samples) ---")
    rho = sim.rho_estimate(100)
    print(f"Trace: {rho.trace():.6f}")
    print(f"Purity: {rho.purity():.6f}")

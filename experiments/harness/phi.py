r"""Phi generators for constructing label-bias functions.

Functions that build :math:`\varphi(x) = \Pr[y{=}1 \mid x]` vectors
for use as inputs to the MoS verification protocol experiments.
"""

import numpy as np


def make_single_parity(n: int, target_s: int) -> list[float]:
    r"""Construct :math:`\varphi` for a pure parity function.

    .. math::

        \varphi(x) = s^* \cdot x \bmod 2

    so that :math:`\tilde\phi = \chi_{s^*}` and the Fourier spectrum has
    a single nonzero coefficient :math:`\hat{\tilde\phi}(s^*) = 1`.

    Parameters
    ----------
    n : int
        Number of input bits.
    target_s : int
        Parity index :math:`s^* \in \{0, \ldots, 2^n - 1\}`.

    Returns
    -------
    list[float]
        :math:`\varphi(x)` for :math:`x = 0, \ldots, 2^n - 1`.
    """
    return [float(bin(target_s & x).count("1") % 2) for x in range(2**n)]


def make_random_parity(n: int, rng: np.random.Generator) -> tuple[list[float], int]:
    r"""Construct :math:`\varphi` for a uniformly random nonzero parity.

    Draws :math:`s^* \sim \mathrm{Uniform}(\{1, \ldots, 2^n - 1\})` and
    returns the corresponding :math:`\varphi` via :func:`make_single_parity`.

    Parameters
    ----------
    n : int
        Number of input bits.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    phi : list[float]
        Label-bias function.
    target_s : int
        The sampled parity index.
    """
    s = int(rng.integers(1, 2**n))
    return make_single_parity(n, s), s


def make_bent_function(n: int) -> list[float]:
    r"""Construct :math:`\varphi` for a Maiorana--McFarland bent function.

    For even :math:`n`, defines :math:`f(x, y) = \langle x, y \rangle \bmod 2`
    where :math:`x, y \in \{0,1\}^{n/2}`.  The resulting
    :math:`g = (-1)^f` has all Fourier coefficients equal in magnitude:

    .. math::

        |\hat{g}(s)| = 2^{-n/2} \quad \forall\, s \in \{0,1\}^n

    This is the worst case for heavy coefficient extraction because no
    coefficient dominates: Parseval gives :math:`\sum_s \hat{g}(s)^2 = 1`
    spread uniformly over all :math:`2^n` frequencies.

    Parameters
    ----------
    n : int
        Number of input bits (must be even).

    Returns
    -------
    list[float]
        :math:`\varphi(x)` for :math:`x = 0, \ldots, 2^n - 1`.

    Raises
    ------
    ValueError
        If *n* is odd.
    """
    if n % 2 != 0:
        raise ValueError(f"Bent functions require even n, got {n}")
    half = n // 2
    phi = []
    for z in range(2**n):
        x_bits = z & ((1 << half) - 1)
        y_bits = (z >> half) & ((1 << half) - 1)
        phi.append(float(bin(x_bits & y_bits).count("1") % 2))
    return phi


def _parity_value(s: int, x: int) -> int:
    r"""Compute :math:`s \cdot x \bmod 2`."""
    return bin(s & x).count("1") % 2


def _chi(s: int, x: int) -> float:
    r"""Compute :math:`\chi_s(x) = (-1)^{s \cdot x}`."""
    return 1.0 - 2.0 * _parity_value(s, x)


def walsh_hadamard(phi_tilde: np.ndarray) -> np.ndarray:
    r"""In-place Walsh--Hadamard transform (unnormalised).

    Given :math:`\tilde\phi(x) \in \{-1, +1\}^{2^n}`, returns
    :math:`\hat{\tilde\phi}(s) = 2^{-n} \sum_x \tilde\phi(x)\,\chi_s(x)`.

    Uses the standard butterfly algorithm in :math:`O(n \cdot 2^n)`.
    """
    a = phi_tilde.copy().astype(np.float64)
    N = len(a)
    n = int(np.log2(N))
    h = 1
    for _ in range(n):
        for j in range(0, N, h * 2):
            for i in range(j, j + h):
                u, v = a[i], a[i + h]
                a[i] = u + v
                a[i + h] = u - v
        h *= 2
    a /= N
    return a


def make_k_sparse(
    n: int, k: int, rng: np.random.Generator
) -> tuple[list[float], int, float]:
    r"""Construct :math:`\varphi` for a random :math:`k`-Fourier-sparse function.

    Draws :math:`k` distinct nonzero parity indices and random
    coefficients from the Dirichlet(1, ..., 1) distribution on the
    :math:`k`-simplex, so that :math:`\sum_i c_i = 1`.  The resulting
    :math:`\tilde\phi` satisfies :math:`|\tilde\phi(x)| \le 1` by the
    triangle inequality, keeping :math:`\varphi(x) \in [0, 1]`.

    This goes beyond single parities (Exp 1) by testing the protocol on
    functions with multiple non-zero Fourier coefficients, probing the
    regime of Corollary 7 (2-agnostic Fourier-sparse learning).

    Parameters
    ----------
    n : int
        Number of input bits.
    k : int
        Fourier sparsity.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    phi : list[float]
        Label-bias function.
    target_s : int
        The index of the heaviest Fourier coefficient.
    parseval_weight : float
        :math:`\sum_i c_i^2 = \mathbb{E}[\tilde\phi(x)^2]`, used as
        :math:`a^2 = b^2` for the distribution class promise.
    """
    indices = rng.choice(2**n - 1, size=k, replace=False) + 1
    coeffs = rng.dirichlet(np.ones(k))

    N = 2**n
    phi = []
    for x in range(N):
        phi_tilde_x = sum(
            coeffs[i] * _chi(int(indices[i]), x) for i in range(k)
        )
        # Dirichlet coefficients sum to 1 analytically so
        # |phi_tilde_x| <= 1 by the triangle inequality, but floating-point
        # roundoff in sum(coeffs) can exceed 1 by O(machine epsilon).
        val = (1.0 - phi_tilde_x) / 2.0
        assert -1e-9 < val < 1.0 + 1e-9, (
            f"make_k_sparse: phi[{x}] = {val} is too far outside [0, 1] "
            f"(phi_tilde_x = {phi_tilde_x})"
        )
        phi.append(np.clip(val, 0.0, 1.0))

    target_s = int(indices[np.argmax(coeffs)])
    parseval_weight = float(np.sum(coeffs**2))
    return phi, target_s, parseval_weight


def make_random_boolean(
    n: int, rng: np.random.Generator
) -> tuple[list[float], int]:
    r"""Construct :math:`\varphi` from a uniform random truth table.

    Each :math:`\varphi(x)` is drawn independently from
    :math:`\{0, 1\}`, producing a maximally Fourier-dense function
    where every coefficient is generically nonzero.  Since
    :math:`\tilde\phi \in \{-1, +1\}`, we have
    :math:`\mathbb{E}[\tilde\phi(x)^2] = 1` exactly, so
    :math:`a^2 = b^2 = 1`.

    The heaviest Fourier coefficient is identified via the
    Walsh--Hadamard transform.

    Parameters
    ----------
    n : int
        Number of input bits.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    phi : list[float]
        Label-bias function.
    target_s : int
        Index of the largest :math:`|\hat{\tilde\phi}(s)|`.
    """
    N = 2**n
    phi_arr = rng.integers(0, 2, size=N).astype(np.float64)
    phi_tilde = 1.0 - 2.0 * phi_arr
    spectrum = walsh_hadamard(phi_tilde)
    target_s = int(np.argmax(np.abs(spectrum)))
    return phi_arr.tolist(), target_s


def make_sparse_plus_noise(
    n: int, rng: np.random.Generator
) -> tuple[list[float], int, float]:
    r"""Construct :math:`\varphi` with one dominant parity plus secondary coefficients.

    The function :math:`\tilde\phi` has a dominant Fourier coefficient
    :math:`c_{\mathrm{dom}} = 0.7` at a random parity :math:`s^*` and
    three secondary coefficients :math:`c_{\mathrm{sec}} = 0.1` each,
    giving :math:`\sum |c_i| = 1`.  This models the realistic case
    where a clear signal coexists with structured Fourier noise.

    Parameters
    ----------
    n : int
        Number of input bits.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    phi : list[float]
        Label-bias function.
    target_s : int
        The dominant parity index :math:`s^*`.
    parseval_weight : float
        :math:`\sum c_i^2`, used as :math:`a^2 = b^2`.
    """
    c_dom = 0.7
    n_secondary = 3
    c_sec = 0.1

    all_indices = rng.choice(2**n - 1, size=1 + n_secondary, replace=False) + 1
    target_s = int(all_indices[0])
    secondary = [int(s) for s in all_indices[1:]]

    N = 2**n
    phi = []
    for x in range(N):
        val = c_dom * _chi(target_s, x)
        for s_j in secondary:
            val += c_sec * _chi(s_j, x)
        phi.append((1.0 - val) / 2.0)

    parseval_weight = c_dom**2 + n_secondary * c_sec**2
    return phi, target_s, parseval_weight

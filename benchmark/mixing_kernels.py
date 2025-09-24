"""Self-contained mixing matrix generators for the benchmark suite."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def _circular_distance(length: int) -> Array:
    """Return the matrix with circular distances for indices ``0..length-1``."""
    idx = np.arange(length, dtype=np.int64)
    diff = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(diff, length - diff).astype(np.float64)


def create_mixing_q(
    M: int,
    *,
    mtype: str = "gauss",
    B: int | None = None,
    L: int | None = None,
    sigma: float = 1.0,
    eps: float = 1e-9,
) -> Array:
    """Create a strictly positive row-stochastic mixing matrix.

    Parameters
    ----------
    M:
        Number of layers.
    mtype:
        ``"gauss"`` for a circular Gaussian kernel, ``"block"`` for a Gaussian on
        ``B`` blocks lifted to layers via ``L`` (with ``M = B * L``).
    B, L:
        Additional block parameters used when ``mtype == "block"``.
    sigma:
        Base width of the Gaussian kernel.
    eps:
        Added everywhere before row normalisation to ensure strict positivity.
    """

    if mtype not in {"gauss", "block"}:
        raise ValueError("mtype must be 'gauss' or 'block'")

    if mtype == "gauss":
        dist = _circular_distance(M)
        kernel = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        kernel += eps
        return kernel / kernel.sum(axis=1, keepdims=True)

    if M == 1:
        B = L = 1
    else:
        if B is None and L is None:
            raise ValueError("For mtype='block' provide either B or L.")
        if L is None:
            L = M // B
        if B is None:
            B = M // L
        if B * L != M:
            raise ValueError("Inconsistent block parameters: B * L must equal M.")

    block_dist = _circular_distance(B)
    block_kernel = np.exp(-(block_dist ** 2) / (2.0 * sigma ** 2))
    block_kernel += eps
    block_kernel /= block_kernel.sum(axis=1, keepdims=True)

    kernel = np.kron(block_kernel, np.ones((L, L), dtype=np.float64))
    kernel += eps
    return kernel / kernel.sum(axis=1, keepdims=True)


def _apply_periodic_band(width: int, size: int) -> Array:
    """Return an indicator matrix with ones on band diagonals of given width."""
    mat = np.zeros((size, size), dtype=int)
    for offset in range(width + 1):
        np.fill_diagonal(mat[offset:], 1)
        np.fill_diagonal(mat[:, offset:], 1)

    if width == 0:
        return mat

    for offset in range(1, width + 1):
        for i in range(size):
            mat[i, (i + offset) % size] = 1
            mat[(i + offset) % size, i] = 1
    return mat


def create_mixing_q_step(M: int, width: int, *, eps: float = 1e-8) -> Array:
    """Create a band matrix with periodic boundaries and normalised rows."""
    band = _apply_periodic_band(width, M).astype(np.float64)
    band += eps
    return band / band.sum(axis=1, keepdims=True)


def _signed_offsets(n: int) -> Array:
    offsets = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    half = n // 2
    offsets[offsets > half] -= n
    return offsets.astype(np.float64)


def create_mixing_q_directional(
    M: int,
    *,
    mtype: str = "gauss",
    B: int | None = None,
    L: int | None = None,
    sigma: float = 1.0,
    shift: float = 0.0,
    skew: float = 0.0,
    eps: float = 1e-9,
) -> Array:
    """Create a directed Gaussian-like kernel with optional skew and shift."""

    if mtype not in {"gauss", "block"}:
        raise ValueError("mtype must be 'gauss' or 'block'")

    def _two_sided(rel: Array, base_sigma: float, skewness: float) -> Array:
        sign = np.sign(rel)
        width = base_sigma * (1.0 + skewness * sign)
        width = np.maximum(width, 0.05 * base_sigma)
        return np.exp(-(rel ** 2) / (2.0 * width ** 2))

    if mtype == "gauss":
        delta = _signed_offsets(M)
        rel = (delta - shift) % M
        half = M // 2
        rel[rel > half] -= M
        kernel = _two_sided(rel, sigma, skew) + eps
        return kernel / kernel.sum(axis=1, keepdims=True)

    if M == 1:
        B = L = 1
    else:
        if B is None and L is None:
            raise ValueError("For mtype='block' provide either B or L.")
        if L is None:
            L = M // B
        if B is None:
            B = M // L
        if B * L != M:
            raise ValueError("Inconsistent block parameters: B * L must equal M.")

    delta_blocks = _signed_offsets(B)
    shift_blocks = shift / max(L, 1)
    rel_blocks = (delta_blocks - shift_blocks) % B
    half_blocks = B // 2
    rel_blocks[rel_blocks > half_blocks] -= B

    block_kernel = _two_sided(rel_blocks, sigma, skew) + eps
    block_kernel /= block_kernel.sum(axis=1, keepdims=True)

    kernel = np.kron(block_kernel, np.ones((L, L), dtype=np.float64))
    kernel += eps
    return kernel / kernel.sum(axis=1, keepdims=True)


__all__ = [
    "create_mixing_q",
    "create_mixing_q_step",
    "create_mixing_q_directional",
]

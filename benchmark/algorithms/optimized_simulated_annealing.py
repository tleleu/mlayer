"""Numba-accelerated simulated annealing implementation for sparse Ising models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from numba import njit


# ---------------------------------------------------------------------------
# Random number generator helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def _xorshift32(state: np.uint32) -> np.uint32:
    """Return the next state of a xorshift32 generator."""

    x = state
    x ^= (x << 13) & np.uint32(0xFFFFFFFF)
    x ^= (x >> 17) & np.uint32(0xFFFFFFFF)
    x ^= (x << 5) & np.uint32(0xFFFFFFFF)
    return x & np.uint32(0xFFFFFFFF)


@njit(cache=True)
def _uniform01(state: np.uint32) -> Tuple[np.uint32, float]:
    """Sample a float in ``[0, 1)`` and return the updated RNG state."""

    state = _xorshift32(state)
    return state, float(state) / 4294967296.0


@njit(cache=True)
def _randint(state: np.uint32, upper: int) -> Tuple[np.uint32, int]:
    """Sample an integer in ``[0, upper)`` using ``xorshift32``."""

    state = _xorshift32(state)
    return state, int(state % np.uint32(upper))


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------


@njit(cache=True)
def _compute_local_fields(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, state: np.ndarray
) -> np.ndarray:
    """Return the local fields ``h_i = sum_j J_ij s_j`` for ``state``."""

    n = state.shape[0]
    fields = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for idx in range(row_start, row_end):
            j = indices[idx]
            acc += data[idx] * state[j]
        fields[i] = acc
    return fields


@njit(cache=True)
def _simulated_annealing_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    spins: np.ndarray,
    beta: float,
    sweeps: int,
    rng_states: np.ndarray,
    zero_temp_break: bool,
    beta_is_inf: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SA for all replicas and return updated RNG states and energies."""

    replicas, n = spins.shape
    energies = np.empty(replicas, dtype=np.float64)
    final_spins = np.empty_like(spins)

    for replica in range(replicas):
        state = spins[replica].copy()
        rng = rng_states[replica]

        local_fields = _compute_local_fields(indptr, indices, data, state)
        energy = -0.5 * np.dot(state.astype(np.float64), local_fields)

        for _ in range(sweeps):
            flips_this_sweep = 0
            for _ in range(n):
                rng, idx = _randint(rng, n)
                s_old = state[idx]
                field = local_fields[idx]
                delta = 2.0 * s_old * field

                accept = False
                if delta <= 0.0:
                    accept = True
                elif not beta_is_inf:
                    rng, u = _uniform01(rng)
                    if u < np.exp(-beta * delta):
                        accept = True

                if accept:
                    flips_this_sweep += 1
                    s_new = -s_old
                    diff = float(s_new - s_old)
                    state[idx] = s_new
                    energy += delta

                    row_start = indptr[idx]
                    row_end = indptr[idx + 1]
                    diag = 0.0
                    for pos in range(row_start, row_end):
                        j = indices[pos]
                        weight = data[pos]
                        if j == idx:
                            diag += weight
                        else:
                            local_fields[j] += weight * diff

                    if diag != 0.0:
                        local_fields[idx] += diag * diff

            if beta_is_inf and zero_temp_break and flips_this_sweep == 0:
                break

        final_spins[replica] = state
        energies[replica] = energy
        rng_states[replica] = rng

    return energies, final_spins, rng_states


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class OptimizedSimulatedAnnealingConfig:
    steps: int
    K: int
    beta: float
    zero_temp_terminate: bool = True


def run_optimized_simulated_annealing(
    J: np.ndarray,
    seed: int,
    *,
    config: OptimizedSimulatedAnnealingConfig,
    initial: Optional[np.ndarray] = None,
    steps_override: Optional[int] = None,
) -> np.ndarray:
    """Execute the optimized SA kernel and return spins with shape ``(K, N)``."""

    if config.steps <= 0:
        raise ValueError("Number of steps must be positive")

    if sp.isspmatrix_csr(J):
        matrix = J.copy()
    elif sp.issparse(J):
        matrix = J.tocsr()
    else:
        matrix = sp.csr_matrix(J)

    if not matrix.has_sorted_indices:
        matrix.sort_indices()

    matrix = matrix.astype(np.float64, copy=False)
    n = matrix.shape[0]
    K = config.K

    rng = np.random.default_rng(seed)
    if initial is None:
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(K, n))
    else:
        spins = np.array(initial, dtype=np.int8, copy=True)
        if spins.shape != (K, n):
            raise ValueError(
                f"Initial state has shape {spins.shape}, expected {(K, n)}"
            )

    sweeps = int(steps_override) if steps_override is not None else int(config.steps)
    if sweeps <= 0:
        raise ValueError("steps_override must be positive")

    rng_states = rng.integers(1, 2**32 - 1, size=K, dtype=np.uint32)

    indptr = matrix.indptr.astype(np.int64)
    indices = matrix.indices.astype(np.int64)
    data = matrix.data.astype(np.float64)

    beta = float(config.beta)
    beta_is_inf = np.isinf(beta)

    energies, final_spins, _ = _simulated_annealing_kernel(
        indptr,
        indices,
        data,
        spins.astype(np.int8),
        beta,
        sweeps,
        rng_states,
        bool(config.zero_temp_terminate),
        beta_is_inf,
    )

    return final_spins.astype(np.int8)


__all__ = [
    "OptimizedSimulatedAnnealingConfig",
    "run_optimized_simulated_annealing",
]
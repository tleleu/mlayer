"""Interface to the legacy Neal-based simulated annealing implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

import neal.simulated_annealing as _neal_sa


@dataclass
class LegacySimulatedAnnealingConfig:
    steps: int
    K: int
    beta: float
    code: str = "neal"


def run_legacy_simulated_annealing(
    J: np.ndarray | sp.spmatrix,
    seed: int,
    *,
    config: LegacySimulatedAnnealingConfig,
    initial: Optional[np.ndarray] = None,
    steps_override: Optional[int] = None,
) -> np.ndarray:
    """Execute the Neal-based SA solver and return spins with shape ``(K, N)``."""

    rng = np.random.default_rng(seed)
    N = J.shape[0]
    K = config.K

    if steps_override is not None:
        if steps_override <= 0:
            raise ValueError("steps_override must be positive")
        steps = int(steps_override)
    else:
        steps = int(config.steps)
        if steps <= 0:
            raise ValueError("Number of steps must be positive")

    if initial is None:
        x0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(K, N))
    else:
        x0 = np.array(initial, dtype=np.int8, copy=True)
        if x0.shape != (K, N):
            raise ValueError(f"Initial state has shape {x0.shape}, expected {(K, N)}")

    if sp.issparse(J):
        coo = J.tocoo()
    else:
        coo = sp.coo_matrix(np.asarray(J))

    num_sweeps_per_beta = 1
    beta_schedule = np.full(steps, config.beta, dtype=float)
    ldata = np.zeros(N, dtype=float)
    interrupt_function = None

    samples, _ = _neal_sa.simulated_annealing(
        K,
        ldata,
        coo.row.astype(np.int64, copy=False),
        coo.col.astype(np.int64, copy=False),
        np.asarray(coo.data, dtype=float),
        num_sweeps_per_beta,
        beta_schedule,
        int(rng.integers(0, 2**31, dtype=np.int64)),
        x0,
        interrupt_function,
    )

    return np.asarray(samples, dtype=np.int8)


__all__ = [
    "LegacySimulatedAnnealingConfig",
    "run_legacy_simulated_annealing",
]

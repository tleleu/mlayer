"""Interface to the legacy Neal-based simulated annealing implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_MCMC_NEAL = PROJECT_ROOT / "MCMC_neal"

if str(LEGACY_MCMC_NEAL) not in sys.path:
    sys.path.insert(0, str(LEGACY_MCMC_NEAL))

import SA  # type: ignore  # legacy module


@dataclass
class LegacySimulatedAnnealingConfig:
    steps: int
    K: int
    beta: float
    code: str = "neal"


def run_legacy_simulated_annealing(
    J: np.ndarray,
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

    _, final_spins = SA.run_SA(
        N,
        J,
        steps,
        K,
        config.beta,
        SAcode=config.code,
        x0=x0,
    )

    return np.asarray(final_spins, dtype=np.int8)


__all__ = [
    "LegacySimulatedAnnealingConfig",
    "run_legacy_simulated_annealing",
]

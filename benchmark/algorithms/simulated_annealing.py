"""Simulated annealing wrapper used by the benchmark pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_MCMC_NEAL = PROJECT_ROOT / "MCMC_neal"

if str(LEGACY_MCMC_NEAL) not in sys.path:
    sys.path.insert(0, str(LEGACY_MCMC_NEAL))

import SA  # type: ignore  # legacy module


@dataclass
class SimulatedAnnealingConfig:
    steps: int
    K: int
    beta: float
    code: str = "neal"


class SimulatedAnnealingRunner:
    """Thin wrapper around the legacy ``SA.run_SA`` entry point."""

    def __init__(self, config: SimulatedAnnealingConfig) -> None:
        self.config = config

    def run(
        self,
        J: np.ndarray,
        seed: int,
        initial: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        K = self.config.K
        N = J.shape[0]
        x0 = initial
        if x0 is None:
            x0 = rng.choice([-1, 1], size=(K, N)).astype(np.int8)
        n_steps = steps if steps is not None else self.config.steps
        _, final_spins = SA.run_SA(
            N,
            J,
            n_steps,
            K,
            self.config.beta,
            SAcode=self.config.code,
            x0=x0,
        )
        return final_spins


__all__ = ["SimulatedAnnealingConfig", "SimulatedAnnealingRunner"]

"""Simulated annealing wrappers used by the benchmark pipeline."""
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


try:
    from .optimized_simulated_annealing import (
        OptimizedSimulatedAnnealingConfig,
        run_optimized_simulated_annealing,
    )

    _OPTIMIZED_SA_AVAILABLE = True
except Exception:  # pragma: no cover - numba optional dependency
    OptimizedSimulatedAnnealingConfig = None  # type: ignore[assignment]
    run_optimized_simulated_annealing = None  # type: ignore[assignment]
    _OPTIMIZED_SA_AVAILABLE = False


@dataclass
class SimulatedAnnealingConfig:
    steps: int
    K: int
    beta: float
    code: str = "neal"
    zero_temp_terminate: bool = True


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
        code = self.config.code.lower()

        if code in {"neal", "legacy"}:
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

        if code in {"optimized", "numba", "local"}:
            if not _OPTIMIZED_SA_AVAILABLE:
                raise RuntimeError(
                    "Optimized simulated annealing backend requires numba and scipy"
                )

            opt_config = OptimizedSimulatedAnnealingConfig(
                steps=self.config.steps,
                K=K,
                beta=self.config.beta,
                zero_temp_terminate=self.config.zero_temp_terminate,
            )

            return run_optimized_simulated_annealing(
                J,
                seed,
                config=opt_config,
                initial=x0,
                steps_override=n_steps,
            )

        raise ValueError(f"Unknown simulated annealing backend '{self.config.code}'")


__all__ = ["SimulatedAnnealingConfig", "SimulatedAnnealingRunner"]

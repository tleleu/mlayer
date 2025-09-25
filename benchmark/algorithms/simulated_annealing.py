"""Simulated annealing wrappers used by the benchmark pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

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

try:
    from .legacy_simulated_annealing import (
        LegacySimulatedAnnealingConfig,
        run_legacy_simulated_annealing,
    )

    _LEGACY_SA_AVAILABLE = True
except Exception:  # pragma: no cover - legacy module optional at import time
    LegacySimulatedAnnealingConfig = None  # type: ignore[assignment]
    run_legacy_simulated_annealing = None  # type: ignore[assignment]
    _LEGACY_SA_AVAILABLE = False


@dataclass
class SimulatedAnnealingConfig:
    steps: int
    K: int
    beta: float
    code: str = "neal"
    zero_temp_terminate: bool = True


class SimulatedAnnealingRunner:
    """Thin wrapper around the available simulated annealing backends."""

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
            x0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(K, N))
        else:
            x0 = np.array(x0, dtype=np.int8, copy=True)
            if x0.shape != (K, N):
                raise ValueError(f"Initial state has shape {x0.shape}, expected {(K, N)}")
        n_steps = steps if steps is not None else self.config.steps
        code = self.config.code.lower()

        if code in {"neal", "legacy"}:
            if not _LEGACY_SA_AVAILABLE:
                raise RuntimeError("Legacy simulated annealing backend is unavailable")

            legacy_config = LegacySimulatedAnnealingConfig(
                steps=self.config.steps,
                K=K,
                beta=self.config.beta,
                code=self.config.code,
            )

            return run_legacy_simulated_annealing(
                J,
                seed,
                config=legacy_config,
                initial=x0,
                steps_override=n_steps,
            )

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

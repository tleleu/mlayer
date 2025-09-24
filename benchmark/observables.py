"""Computation of benchmark observables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MCMC_NEAL = PROJECT_ROOT / "MCMC_neal"

if str(LEGACY_MCMC_NEAL) not in sys.path:
    sys.path.insert(0, str(LEGACY_MCMC_NEAL))

import stats  # type: ignore  # legacy helper


@dataclass
class EnergyObservables:
    energy_mean: float
    energy_min: float
    q_average: float


class ObservableEvaluator:
    """Translate spin configurations into scalar observables."""

    def compute(self, spins: np.ndarray, J0_dense: np.ndarray, M: int) -> EnergyObservables:
        energy = stats.calculate_energy_replicas(spins, J0_dense, M)
        e_mean = float(np.mean(energy))
        e_min = float(np.min(energy))

        K = spins.shape[0]
        N0 = J0_dense.shape[0]
        spins_reshaped = spins.reshape(K, M, N0)
        overlap = np.matmul(spins_reshaped, spins_reshaped.transpose(0, 2, 1))
        off_diag_sum = overlap.sum(axis=(1, 2)) - np.trace(overlap, axis1=1, axis2=2)
        qavg_val = float((off_diag_sum / (M * (M - 1)) / N0).mean()) if M > 1 else float("nan")

        return EnergyObservables(e_mean, e_min, qavg_val)


__all__ = ["EnergyObservables", "ObservableEvaluator"]

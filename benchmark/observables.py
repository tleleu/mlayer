"""Computation of benchmark observables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def _as_dense_matrix(J: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Return a dense ``numpy`` array view of ``J``."""

    if sp.issparse(J):
        return np.asarray(J.todense())
    return np.asarray(J)


def _calculate_energy(spins: np.ndarray, J: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Return the energy of each configuration in ``spins``.

    Parameters
    ----------
    spins:
        Array with shape ``(K, N)`` containing ``K`` spin configurations of
        ``N`` spins each.
    J:
        Dense interaction matrix with shape ``(N, N)``.
    """

    J_dense = _as_dense_matrix(J)
    spins_float = spins.astype(float)
    interaction = np.matmul(spins_float, J_dense)
    return -0.5 * np.sum(spins_float * interaction, axis=1)


def _calculate_energy_replicas(
    final_spins: np.ndarray, J: np.ndarray | sp.spmatrix, M: int
) -> np.ndarray:
    """Energy per spin for ``M`` replicas of each configuration.

    Parameters
    ----------
    final_spins:
        Array of shape ``(K, M * N)`` storing ``K`` configurations, each with
        ``M`` replicas of ``N`` spins.
    J:
        Dense interaction matrix with shape ``(N, N)``.
    M:
        Number of replicas encoded in ``final_spins``.
    """

    K = final_spins.shape[0]
    N = J.shape[0]
    spins_reshaped = final_spins.reshape(K, M, N)

    energies = np.empty((K, M))
    for m in range(M):
        energies[:, m] = _calculate_energy(spins_reshaped[:, m, :], J)

    return energies / N


@dataclass
class EnergyObservables:
    energy_mean: float
    energy_min: float
    q_average: float


class ObservableEvaluator:
    """Translate spin configurations into scalar observables."""

    def compute(
        self, spins: np.ndarray, J0: np.ndarray | sp.spmatrix, M: int
    ) -> EnergyObservables:
        energy = _calculate_energy_replicas(spins, J0, M)
        e_mean = float(np.mean(energy))
        e_min = float(np.min(energy))

        K = spins.shape[0]
        N0 = int(J0.shape[0])
        spins_reshaped = spins.reshape(K, M, N0)
        overlap = np.matmul(spins_reshaped, spins_reshaped.transpose(0, 2, 1))
        off_diag_sum = overlap.sum(axis=(1, 2)) - np.trace(overlap, axis1=1, axis2=2)
        qavg_val = float((off_diag_sum / (M * (M - 1)) / N0).mean()) if M > 1 else float("nan")

        return EnergyObservables(e_mean, e_min, qavg_val)


__all__ = ["EnergyObservables", "ObservableEvaluator"]

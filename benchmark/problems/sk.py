"""Sherrington-Kirkpatrick (SK) mean-field instance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .base import IsingProblem, ProblemConfig


@dataclass
class SKProblem(IsingProblem):
    """Simple dense SK model with ±1 couplings."""

    seed: int = 0

    def __init__(self, N: int, seed: int = 0) -> None:
        super().__init__(ProblemConfig(N=N, params={"seed": seed}))
        self.seed = seed

    def build_couplings(self) -> sp.spmatrix:  # type: ignore[override]
        return _create_sk(self.config.N, seed=self.seed)


def _create_sk(N: int, seed: int = 0) -> sp.spmatrix:
    """Generate dense ±1 couplings following the legacy helper."""

    rng = np.random.default_rng(seed)
    couplings = rng.choice([-1.0, 1.0], size=(N, N))
    couplings = np.triu(couplings, k=1)
    couplings = couplings + couplings.T
    return sp.csr_matrix(couplings)


__all__ = ["SKProblem"]

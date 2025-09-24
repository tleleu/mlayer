"""Sherrington-Kirkpatrick (SK) mean-field instance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .base import IsingProblem, ProblemConfig

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MCMC = PROJECT_ROOT / "MCMC"

if str(LEGACY_MCMC) not in sys.path:
    sys.path.insert(0, str(LEGACY_MCMC))


@dataclass
class SKProblem(IsingProblem):
    """Simple dense SK model with Gaussian couplings."""

    seed: int = 0

    def __init__(self, N: int, seed: int = 0) -> None:
        super().__init__(ProblemConfig(N=N, params={"seed": seed}))
        self.seed = seed

    def build_couplings(self) -> sp.spmatrix:  # type: ignore[override]
        rng = np.random.default_rng(self.seed)
        J = rng.normal(loc=0.0, scale=1.0 / np.sqrt(self.config.N), size=(self.config.N, self.config.N))
        np.fill_diagonal(J, 0.0)
        # enforce symmetry
        J = (J + J.T) / 2.0
        return sp.csr_matrix(J)


__all__ = ["SKProblem"]

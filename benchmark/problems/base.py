"""Problem definitions for benchmark Ising instances."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MCMC = PROJECT_ROOT / "MCMC"

if str(LEGACY_MCMC) not in sys.path:
    sys.path.insert(0, str(LEGACY_MCMC))


@dataclass
class ProblemConfig:
    """Generic configuration for an Ising problem."""

    N: int
    params: Dict[str, Any]


class IsingProblem(ABC):
    """Interface implemented by all problem generators."""

    def __init__(self, config: ProblemConfig) -> None:
        self.config = config

    @abstractmethod
    def build_couplings(self) -> sp.spmatrix:
        """Return the sparse interaction matrix ``J``."""

    def build_dense(self) -> np.ndarray:
        """Convenience helper returning a dense ``numpy`` array."""

        J = self.build_couplings()
        return np.array(J.todense()) if sp.issparse(J) else np.array(J)


__all__ = ["ProblemConfig", "IsingProblem"]

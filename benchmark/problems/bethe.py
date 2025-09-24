"""Bethe lattice Ising instance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import IsingProblem, ProblemConfig

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_MCMC = PROJECT_ROOT / "MCMC"

if str(LEGACY_MCMC) not in sys.path:
    sys.path.insert(0, str(LEGACY_MCMC))

from instance import create_Bethe  # type: ignore  # legacy helper


@dataclass
class BetheProblem(IsingProblem):
    """Problem wrapper building Bethe instances using the legacy helper."""

    degree: int

    def __init__(self, N: int, degree: int) -> None:
        super().__init__(ProblemConfig(N=N, params={"degree": degree}))
        self.degree = degree

    def build_couplings(self):  # type: ignore[override]
        return create_Bethe(self.config.N, self.degree + 1)


__all__ = ["BetheProblem"]

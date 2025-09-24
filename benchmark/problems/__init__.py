"""Collection of benchmark problem definitions."""

from .bethe import BetheProblem
from .sk import SKProblem
from .base import ProblemConfig, IsingProblem

__all__ = ["BetheProblem", "SKProblem", "ProblemConfig", "IsingProblem"]

"""Bethe lattice Ising instance."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .base import IsingProblem, ProblemConfig


@dataclass
class BetheProblem(IsingProblem):
    """Problem wrapper building Bethe instances with inline generation."""

    degree: int

    def __init__(self, N: int, degree: int) -> None:
        super().__init__(ProblemConfig(N=N, params={"degree": degree}))
        self.degree = degree

    def build_couplings(self) -> sp.spmatrix:  # type: ignore[override]
        return _create_bethe(self.config.N, self.degree + 1)


def _create_bethe(N: int, degree: int) -> sp.spmatrix:
    """Generate a random regular graph with ±1 couplings."""

    if N * degree % 2 != 0:
        raise ValueError("N * degree must be even for a regular graph")

    graph = nx.random_regular_graph(degree, N)
    couplings = nx.to_scipy_sparse_array(graph, format="lil", dtype=np.int8)

    rows, cols = couplings.nonzero()
    mask = rows < cols
    signs = np.random.choice([-1, 1], size=mask.sum())

    for (i, j), sign in zip(zip(rows[mask], cols[mask]), signs):
        couplings[i, j] = sign
        couplings[j, i] = sign

    return couplings.tocsr()


__all__ = ["BetheProblem"]

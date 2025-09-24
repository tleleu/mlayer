"""M-layer transformation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MCMC = PROJECT_ROOT / "MCMC"
LEGACY_MCMC_NEAL = PROJECT_ROOT / "MCMC_neal"

for path in (LEGACY_MCMC, LEGACY_MCMC_NEAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mlayer2  # type: ignore  # legacy module
import lib  # type: ignore  # legacy module


@dataclass
class MLayerTransformRequest:
    """Configuration for :class:`MLayerTransformer`."""

    J0_dense: np.ndarray
    M: int
    mixing_matrix: np.ndarray
    typeperm: str = "asym"
    backend: str = "mlayer2"


class MLayerTransformer:
    """Wraps the different legacy M-layer implementations behind one API."""

    def transform(self, request: MLayerTransformRequest) -> np.ndarray:
        """Return the dense coupling matrix after applying the M-layer map."""

        backend = request.backend.lower()
        if backend == "mlayer2":
            result = mlayer2.Mlayer(
                request.J0_dense,
                request.M,
                request.mixing_matrix,
                typeperm=request.typeperm,
            )
        elif backend == "lib":
            result = lib.Mlayer(
                request.J0_dense,
                request.M,
                permute=True,
                GoG=True,
                typeperm=request.typeperm,
                C=request.mixing_matrix,
            )
        else:
            raise ValueError(f"Unsupported M-layer backend: {request.backend}")

        if sp.issparse(result):
            return np.array(result.todense())
        return np.array(result)


__all__ = [
    "MLayerTransformer",
    "MLayerTransformRequest",
]

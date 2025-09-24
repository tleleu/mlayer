"""M-layer transformation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .mlayer_core import mlayer

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MLayerConstructor = Callable[[np.ndarray, int, np.ndarray, str], np.ndarray]


@dataclass
class MLayerTransformRequest:
    """Configuration for :class:`MLayerTransformer`."""

    J0_dense: np.ndarray
    M: int
    mixing_matrix: np.ndarray
    typeperm: str = "asym"
    backend: str = "permanental"


class MLayerTransformer:
    """Wrap the M-layer implementation behind a simple API."""

    def transform(self, request: MLayerTransformRequest) -> np.ndarray:
        """Return the dense coupling matrix after applying the M-layer map."""

        backend = request.backend.lower()
        if backend not in {"mlayer2", "permanental"}:
            raise ValueError(f"Unsupported M-layer backend: {request.backend}")

        result = mlayer(
            request.J0_dense,
            request.M,
            request.mixing_matrix,
            typeperm=request.typeperm,
        )

        if sp.issparse(result):
            return np.array(result.todense())
        return np.array(result)


def create_mlayer_constructor(backend: str) -> MLayerConstructor:
    """Return a callable that applies the requested M-layer transformation."""

    backend_name = backend.lower()
    if backend_name not in {"mlayer2", "permanental"}:
        raise ValueError(f"Unsupported M-layer constructor backend: {backend}")

    def _mlayer2_constructor(
        J0_dense: np.ndarray,
        M: int,
        Q: np.ndarray,
        typeperm: str = "asym",
    ) -> np.ndarray:
        result = mlayer(J0_dense, M, Q, typeperm=typeperm)
        if sp.issparse(result):
            return np.array(result.todense())
        return np.array(result)

    return _mlayer2_constructor


__all__ = [
    "MLayerTransformer",
    "MLayerTransformRequest",
    "MLayerConstructor",
    "create_mlayer_constructor",
]

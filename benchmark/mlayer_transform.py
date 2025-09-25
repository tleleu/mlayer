"""M-layer transformation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import scipy.sparse as sp

from .mlayer_alt import mlayer as mlayer_alternative
from .mlayer_core import mlayer as mlayer_permanental


_MLAYER_BACKENDS: Dict[
    str, Callable[[sp.spmatrix | np.ndarray, int, np.ndarray, str], sp.csr_matrix]
] = {
    "permanental": mlayer_permanental,
    "permanental_alt": mlayer_alternative,
    # ``mlayer2`` is kept as an alias for backwards compatibility.
    "mlayer2": mlayer_permanental,
}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MLayerConstructor = Callable[[sp.spmatrix | np.ndarray, int, np.ndarray, str], sp.csr_matrix]


@dataclass
class MLayerTransformRequest:
    """Configuration for :class:`MLayerTransformer`."""

    couplings: sp.spmatrix | np.ndarray
    M: int
    mixing_matrix: np.ndarray
    typeperm: str = "asym"
    backend: str = "permanental"


class MLayerTransformer:
    """Wrap the M-layer implementation behind a simple API."""

    def transform(self, request: MLayerTransformRequest) -> sp.csr_matrix:
        """Return the sparse coupling matrix after applying the M-layer map."""

        backend = request.backend.lower()
        if backend not in _MLAYER_BACKENDS:
            raise ValueError(f"Unsupported M-layer backend: {request.backend}")

        mlayer_impl = _MLAYER_BACKENDS[backend]

        result = mlayer_impl(
            request.couplings,
            request.M,
            request.mixing_matrix,
            typeperm=request.typeperm,
        )
        return result if sp.issparse(result) else sp.csr_matrix(result)


def create_mlayer_constructor(backend: str) -> MLayerConstructor:
    """Return a callable that applies the requested M-layer transformation."""

    backend_name = backend.lower()
    if backend_name not in _MLAYER_BACKENDS:
        raise ValueError(f"Unsupported M-layer constructor backend: {backend}")

    def _mlayer2_constructor(
        couplings: sp.spmatrix | np.ndarray,
        M: int,
        Q: np.ndarray,
        typeperm: str = "asym",
    ) -> sp.csr_matrix:
        result = _MLAYER_BACKENDS[backend_name](couplings, M, Q, typeperm=typeperm)
        return result if sp.issparse(result) else sp.csr_matrix(result)

    return _mlayer2_constructor


__all__ = [
    "MLayerTransformer",
    "MLayerTransformRequest",
    "MLayerConstructor",
    "create_mlayer_constructor",
]

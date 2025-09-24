"""Utilities for constructing mixing matrices used by the M-layer transformation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import numpy as np

from .mixing_kernels import (
    create_mixing_q,
    create_mixing_q_directional,
    create_mixing_q_step,
)


class MixingMatrixBackend(str, Enum):
    """Enumerates the supported ways of generating the mixing matrix."""

    MLAYER_BLOCK = "mlayer_block"
    LIB_BLOCK = "lib_block"
    MLAYER_STEP = "mlayer_step"
    LIB_STEP = "lib_step"
    MLAYER_DIRECTIONAL = "mlayer_directional"


@dataclass
class MixingMatrixRequest:
    """Lightweight configuration bundle for :class:`MixingMatrixFactory`."""

    backend: MixingMatrixBackend
    M: int
    sigma: float
    index: int
    L: int = 2
    mtype: str = "block"
    shift: float = 0.0
    skew: float = 0.0

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "M": self.M,
            "sigma": self.sigma,
            "index": self.index,
            "L": self.L,
            "mtype": self.mtype,
            "shift": self.shift,
            "skew": self.skew,
        }


class MixingMatrixFactory:
    """Factory creating mixing matrices compatible with the legacy pipeline."""

    def create(self, request: MixingMatrixRequest) -> np.ndarray:
        """Return the dense mixing matrix associated with *request*."""

        backend = request.backend
        sigma = request.sigma
        if backend is MixingMatrixBackend.MLAYER_BLOCK:
            return create_mixing_q(
                request.M,
                mtype=request.mtype,
                sigma=sigma * 20.0,
                L=request.L,
            )
        if backend is MixingMatrixBackend.LIB_BLOCK:
            return create_mixing_q(
                request.M,
                mtype=request.mtype,
                sigma=sigma * 20.0,
                L=request.L,
            )
        if backend is MixingMatrixBackend.MLAYER_STEP:
            return create_mixing_q_step(request.M, request.index + 1)
        if backend is MixingMatrixBackend.LIB_STEP:
            return create_mixing_q_step(request.M, request.index + 1)
        if backend is MixingMatrixBackend.MLAYER_DIRECTIONAL:
            return create_mixing_q_directional(
                request.M,
                mtype=request.mtype,
                sigma=sigma * 20.0,
                L=request.L,
                shift=request.shift,
                skew=request.skew,
            )
        raise ValueError(f"Unsupported mixing backend: {backend}")


__all__ = [
    "MixingMatrixBackend",
    "MixingMatrixFactory",
    "MixingMatrixRequest",
]

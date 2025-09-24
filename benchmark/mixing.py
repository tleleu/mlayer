"""Utilities for constructing mixing matrices used by the M-layer transformation."""
from __future__ import annotations

import numpy as np

from .mixing_kernels import (
    create_mixing_q,
    create_mixing_q_directional,
    create_mixing_q_step,
)


class MixingMatrix:
    """Callable helper for constructing mixing matrices.

    Parameters are fixed at construction time and the instance can then be
    reused to generate matrices for different ``M`` or ``sigma`` values.
    """

    def __init__(
        self,
        backend: str,
        *,
        L: int = 2,
        mtype: str = "block",
        shift: float = 0.0,
        skew: float = 0.0,
    ) -> None:
        self.backend = backend
        self.L = L
        self.mtype = mtype
        self.shift = shift
        self.skew = skew

    def __call__(self, M: int, sigma: float, index: int) -> np.ndarray:
        """Return the dense mixing matrix for the configured backend."""

        backend = self.backend
        if backend in {"mlayer_block", "lib_block"}:
            return create_mixing_q(
                M,
                mtype=self.mtype,
                sigma=sigma * 20.0,
                L=self.L,
            )
        if backend in {"mlayer_step", "lib_step"}:
            return create_mixing_q_step(M, index + 1)
        if backend == "mlayer_directional":
            return create_mixing_q_directional(
                M,
                mtype=self.mtype,
                sigma=sigma * 20.0,
                L=self.L,
                shift=self.shift,
                skew=self.skew,
            )
        raise ValueError(f"Unsupported mixing backend: {backend}")


__all__ = ["MixingMatrix"]

"""Alternative M-layer implementation using sequential permutations."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from MCMC_neal.lib import (
    directed_permutation_sequential,
    generate_permutation,
)


def _flatten(layer: int, node: int, size: int) -> int:
    """Map a layer/node pair to a flat index."""
    return layer * size + node


def _sample_permutation(
    mixing: np.ndarray,
    *,
    eliminate_drift: bool,
    kappa: float,
) -> np.ndarray:
    """Draw a permutation for the M-layer wiring."""
    if abs(kappa) > 0.0:
        return directed_permutation_sequential(
            mixing,
            kappa=kappa,
            eliminate_drift=eliminate_drift,
            rng=None,
        )
    return generate_permutation(mixing, eliminate_drift=eliminate_drift, rng=None)


def mlayer(
    couplings: np.ndarray | sp.spmatrix,
    layers: int,
    mixing: np.ndarray,
    *,
    permute: bool = True,
    GoG: bool = False,
    typeperm: str = "asym",
    eliminate_drift: bool = False,
    kappa: float = 0.0,
) -> sp.csr_matrix:
    """Lift ``couplings`` to ``layers`` using the sequential permutation scheme."""

    del GoG  # parameter kept for API compatibility

    if layers == 1:
        return sp.csr_matrix(couplings)

    if sp.issparse(couplings):
        base_sparse = sp.csr_matrix(couplings)
        n = int(base_sparse.shape[0])
        upper = sp.triu(base_sparse, k=0).tocoo()
        entries = zip(upper.row, upper.col, upper.data)
    else:
        base = np.asarray(couplings)
        n = int(base.shape[0])
        upper = np.triu(base)
        rows, cols = np.nonzero(upper)
        entries = ((int(i), int(j), float(upper[i, j])) for i, j in zip(rows, cols))

    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    for i, j, weight in entries:
        weight = float(weight)
        if weight == 0.0:
            continue

        if not permute:
            for layer in range(layers):
                flat_row = _flatten(layer, i, n)
                flat_col = _flatten(layer, j, n)
                row_idx.append(flat_row)
                col_idx.append(flat_col)
                data.append(weight)

                row_idx.append(flat_col)
                col_idx.append(flat_row)
                data.append(weight)
            continue

        perm_forward = _sample_permutation(
            mixing, eliminate_drift=eliminate_drift, kappa=kappa
        )
        perm_backward = (
            perm_forward
            if typeperm == "sym"
            else _sample_permutation(
                mixing, eliminate_drift=eliminate_drift, kappa=kappa
            )
        )

        for layer in range(layers):
            tgt_forward = int(perm_forward[layer])
            tgt_backward = int(perm_backward[layer])

            row_idx.append(_flatten(tgt_forward, i, n))
            col_idx.append(_flatten(layer, j, n))
            data.append(weight)

            row_idx.append(_flatten(tgt_backward, j, n))
            col_idx.append(_flatten(layer, i, n))
            data.append(weight)

    size = n * layers
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(size, size))


__all__ = ["mlayer"]

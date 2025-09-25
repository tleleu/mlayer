"""Clean-room implementation of the permanental M-layer lifting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

Array = np.ndarray


@dataclass
class PermanentalSamplerConfig:
    exact_threshold: int = 14
    balance: bool = True


def _sinkhorn_balance(matrix: Array, *, max_iter: int = 1_000, tol: float = 1e-12) -> Array:
    q = np.array(matrix, dtype=np.float64, copy=True)
    q[q <= 0.0] = np.finfo(float).tiny
    r = np.ones(q.shape[0], dtype=np.float64)
    c = np.ones(q.shape[1], dtype=np.float64)

    for _ in range(max_iter):
        rq = q * c
        row_sums = rq.sum(axis=1)
        row_sums[row_sums == 0.0] = 1.0
        r_new = r / row_sums

        cq = (q.T * r_new).T
        col_sums = cq.sum(axis=0)
        col_sums[col_sums == 0.0] = 1.0
        c_new = c / col_sums

        delta = max(np.max(np.abs(r_new - r)), np.max(np.abs(c_new - c)))
        r, c = r_new, c_new
        if delta < tol:
            break

    return (q.T * r).T * c


def _permanental_sampler_exact(q: Array) -> Array:
    M = int(q.shape[0])
    mask_full = (1 << M) - 1
    dp = np.zeros((M + 1, 1 << M), dtype=np.float64)
    dp[M, 0] = 1.0

    for r in range(M - 1, -1, -1):
        for mask in range(1 << M):
            acc = 0.0
            m = mask
            j = 0
            while m:
                if m & 1:
                    acc += q[r, j] * dp[r + 1, mask & ~(1 << j)]
                m >>= 1
                j += 1
            dp[r, mask] = acc

    perm = np.empty(M, dtype=np.int64)
    mask = mask_full
    for r in range(M):
        total = 0.0
        weights = np.zeros(M, dtype=np.float64)
        m = mask
        j = 0
        while m:
            if m & 1:
                w = q[r, j] * dp[r + 1, mask & ~(1 << j)]
                weights[j] = w
                total += w
            m >>= 1
            j += 1

        if total <= 0.0:
            available = [j for j in range(M) if (mask >> j) & 1]
            chosen = max(available, key=lambda j: q[r, j])
        else:
            u = np.random.rand() * total
            csum = 0.0
            chosen = -1
            for j in range(M):
                if (mask >> j) & 1:
                    csum += weights[j]
                    if u <= csum:
                        chosen = j
                        break
            if chosen == -1:
                for j in range(M):
                    if (mask >> j) & 1:
                        chosen = j
                        break

        perm[r] = chosen
        mask &= ~(1 << chosen)

    return perm


def _permanental_sampler_gumbel_hungarian(q: Array, eps: float = 1e-18) -> Array:
    M = q.shape[0]
    gumbel = -np.log(-np.log(np.random.rand(M, M) + eps) + eps)
    scores = np.log(q + eps) + gumbel
    _, col_ind = linear_sum_assignment(-scores)
    return col_ind.astype(np.int64)


def generate_permutation_permanental(
    q: Array, *, config: PermanentalSamplerConfig | None = None
) -> Array:
    """Sample a permutation with probability proportional to the permanent."""

    if config is None:
        config = PermanentalSamplerConfig()

    work = np.asarray(q, dtype=np.float64)
    if config.balance:
        work = _sinkhorn_balance(work)

    if work.shape[0] <= config.exact_threshold:
        return _permanental_sampler_exact(work)
    return _permanental_sampler_gumbel_hungarian(work)


def _flatten(layer: int, node: int, size: int) -> int:
    return layer * size + node


def mlayer(
    couplings: Array | sp.spmatrix,
    layers: int,
    mixing: Array,
    *,
    typeperm: str = "sym",
    sampler_config: PermanentalSamplerConfig | None = None,
) -> sp.csr_matrix:
    """Lift a base coupling matrix to ``layers`` using permanental wiring."""

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

    sampler_config = sampler_config or PermanentalSamplerConfig()

    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    for i, j, weight in entries:
        weight = float(weight)
        if weight == 0.0:
            continue

        if typeperm == "sym":
            perm = generate_permutation_permanental(mixing, config=sampler_config)
            for layer in range(layers):
                target = int(perm[layer])
                row_idx.append(_flatten(target, i, n))
                col_idx.append(_flatten(layer, j, n))
                data.append(weight)

                row_idx.append(_flatten(layer, j, n))
                col_idx.append(_flatten(target, i, n))
                data.append(weight)
        elif typeperm == "asym":
            perm_forward = generate_permutation_permanental(mixing, config=sampler_config)
            perm_backward = generate_permutation_permanental(mixing, config=sampler_config)
            for layer in range(layers):
                tgt_f = int(perm_forward[layer])
                tgt_b = int(perm_backward[layer])
                row_idx.append(_flatten(tgt_f, i, n))
                col_idx.append(_flatten(layer, j, n))
                data.append(weight)

                row_idx.append(_flatten(tgt_b, j, n))
                col_idx.append(_flatten(layer, i, n))
                data.append(weight)
        else:
            raise ValueError("typeperm must be 'sym' or 'asym'")

    size = n * layers
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(size, size))


__all__ = [
    "PermanentalSamplerConfig",
    "generate_permutation_permanental",
    "mlayer",
]

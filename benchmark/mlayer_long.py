"""M-layer with long-cycle permutations (uniform Q by default)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


# ---------------------------- helpers ----------------------------

def _flatten(layer: int, node: int, size: int) -> int:
    return layer * size + node


def _as_float_pos(q: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    a = np.asarray(q, dtype=np.float64)
    a[a <= 0.0] = eps
    return a


def _cycles_of_perm(pi: np.ndarray) -> list[list[int]]:
    M = int(pi.size)
    seen = np.zeros(M, dtype=bool)
    cycles: list[list[int]] = []
    for s in range(M):
        if seen[s]:
            continue
        c = []
        u = s
        while not seen[u]:
            seen[u] = True
            c.append(u)
            u = int(pi[u])
        cycles.append(c)
    return cycles


def _hungarian_argmax_logQ(logQ: np.ndarray) -> np.ndarray:
    r, c = linear_sum_assignment(-logQ)
    pi = np.empty(logQ.shape[0], dtype=np.int64)
    pi[r] = c
    return pi


def _join_two_cycles_via_swap(pi: np.ndarray, i: int, j: int) -> None:
    pi[i], pi[j] = pi[j], pi[i]


def _cycle_labels(pi: np.ndarray) -> np.ndarray:
    M = pi.size
    lbl = -np.ones(M, dtype=np.int64)
    lab = 0
    for c in _cycles_of_perm(pi):
        for u in c:
            lbl[u] = lab
        lab += 1
    return lbl


def _sample_long_cycle_from_Q(
    Q: np.ndarray,
    *,
    theta: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample permutation ~ (∏ Q[l,pi(l)]) * theta^(#cycles), theta<1 → long cycles."""
    if rng is None:
        rng = np.random.default_rng()

    Qp = _as_float_pos(Q)
    logQ = np.log(Qp)

    # start from max assignment
    pi = _hungarian_argmax_logQ(logQ)

    lab = _cycle_labels(pi)
    ncycles = int(lab.max() + 1)

    while ncycles > 1:
        labs = rng.choice(ncycles, size=2, replace=False)
        I = np.nonzero(lab == labs[0])[0]
        J = np.nonzero(lab == labs[1])[0]
        i = int(rng.choice(I))
        j = int(rng.choice(J))
        ai, aj = int(pi[i]), int(pi[j])

        delta_logQ = (logQ[i, aj] + logQ[j, ai]) - (logQ[i, ai] + logQ[j, aj])
        delta = delta_logQ + np.log(theta)  # cycle count goes down by 1
        if delta >= 0 or rng.random() < np.exp(delta):
            _join_two_cycles_via_swap(pi, i, j)
            lab = _cycle_labels(pi)
            ncycles = int(lab.max() + 1)

    return pi


# ---------------------------- public API ----------------------------

def mlayer(
    couplings: np.ndarray | sp.spmatrix,
    layers: int,
    mixing: np.ndarray | None = None,
    *,
    permute: bool = True,
    GoG: bool = False,
    typeperm: str = "asym",
    theta: float = 0.1,
    rng: np.random.Generator | None = None,
) -> sp.csr_matrix:
    """
    Lift 'couplings' to 'layers' using a permutation that promotes very long loops.

    Parameters
    ----------
    mixing : np.ndarray | None
        Row-stochastic Q. If None, defaults to uniform.
    typeperm : {'sym','asym'}
        Use one long-cycle perm for both directions or two independent ones.
    theta : float in (0,1)
        Cycle penalty via factor theta^{#cycles}. Smaller -> longer cycles.
    """
    del GoG

    if layers == 1:
        return sp.csr_matrix(couplings)

    # make mixing uniform if not provided
    if mixing is None:
        mixing = np.ones((layers, layers), dtype=np.float64) / layers

    # base entries (upper triangle)
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

    if not permute:
        data: list[float] = []
        ri: list[int] = []
        ci: list[int] = []
        for i, j, w in entries:
            if w == 0.0:
                continue
            for layer in range(layers):
                a = _flatten(layer, i, n)
                b = _flatten(layer, j, n)
                ri.extend((a, b))
                ci.extend((b, a))
                data.extend((w, w))
        size = n * layers
        return sp.csr_matrix((data, (ri, ci)), shape=(size, size))

    # draw permutation(s)
    perm_fwd = _sample_long_cycle_from_Q(mixing, theta=theta, rng=rng)
    if typeperm == "sym":
        perm_bwd = perm_fwd
    elif typeperm == "asym":
        perm_bwd = _sample_long_cycle_from_Q(mixing, theta=theta, rng=rng)
    else:
        raise ValueError("typeperm must be 'sym' or 'asym'")

    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    for i, j, w in entries:
        w = float(w)
        if w == 0.0:
            continue
        for layer in range(layers):
            tf = int(perm_fwd[layer])
            tb = int(perm_bwd[layer])

            row_idx.append(_flatten(tf, i, n))
            col_idx.append(_flatten(layer, j, n))
            data.append(w)

            row_idx.append(_flatten(tb, j, n))
            col_idx.append(_flatten(layer, i, n))
            data.append(w)

    size = n * layers
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(size, size))


__all__ = ["mlayer"]

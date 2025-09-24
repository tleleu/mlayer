import numpy as np
from numba import njit

@njit(parallel=False, fastmath=True, cache=True)
def step_dynamic_SP(h, y, Q, n_blocks, deg, logW_neighbor, h_hat_tmp):
    """
    Zero-T 1-RSB dynamic SP step (block mixing Q).
    Neighbor tilt: log w_neigh = y * sum_b E_b, with
      u_b = 0.5*(|J+H_b| - |J-H_b|),  E_b = 0.5*(|J+H_b| + |J-H_b|),  H = Q @ h_src.
    No site reweighting in the field-law update.
    """
    NE = h.shape[0]
    B  = int(n_blocks)
    D  = int(deg)
    d  = D - 1  # excess degree

    # --- 1) Single-neighbor proposals & weights (i.i.d. from the pool)
    for e in range(NE):
        # sample an input field i.i.d. from the current population
        src = np.random.randint(0, NE)

        # sample coupling J (extend as needed if |J| varies)
        J = 1.0 if (np.random.rand() > 0.5) else -1.0

        # H = Q @ h[src,:]
        H = np.zeros(B)
        for b in range(B):
            s = 0.0
            for c in range(B):
                s += Q[b, c] * h[src, c]
            H[b] = s

        # zero-T channel maps per block
        sumE = 0.0
        for b in range(B):
            abs_plus  = np.abs(J + H[b])
            abs_minus = np.abs(J - H[b])
            u_b = 0.5 * (abs_plus - abs_minus)            # half-field
            E_b = 0.5 * (abs_plus + abs_minus)            # channel energy
            h_hat_tmp[e, b] = u_b
            sumE += E_b

        # neighbor log-weight: + y * sum_b E_b
        #logW_neighbor[e] = y * sumE
        logW_neighbor[e] = y * sumE / B

    # --- 2) Neighbor-level resampling (softmax with max shift)
    m = logW_neighbor[0]
    for e in range(1, NE):
        if logW_neighbor[e] > m:
            m = logW_neighbor[e]
    Z = 0.0
    for e in range(NE):
        Z += np.exp(logW_neighbor[e] - m)

    C = np.empty(NE)
    csum = 0.0
    for e in range(NE):
        csum += np.exp(logW_neighbor[e] - m) / Z
        C[e] = csum
    C[NE - 1] = 1.0

    # resample NE single-neighbor half-fields u
    u_resampled = np.empty_like(h_hat_tmp)
    for e in range(NE):
        r = np.random.rand()
        idx = np.searchsorted(C, r)
        for b in range(B):
            u_resampled[e, b] = h_hat_tmp[idx, b]

    # --- 3) Aggregate d = D-1 neighbors to form new cavity fields
    for e in range(NE):
        for b in range(B):
            h[e, b] = 0.0
        for _ in range(d):
            idx = np.random.randint(0, NE)
            for b in range(B):
                h[e, b] += u_resampled[idx, b]

    # keep last resampled single-neighbor pool for diagnostics
    for e in range(NE):
        for b in range(B):
            h_hat_tmp[e, b] = u_resampled[e, b]

    return h, h_hat_tmp



# ---------- driver ----------------------------------------------------

def run_PD(y: float,
           Q: np.ndarray,
           B: int,
           deg: int,
           n_edges: int = 200_000,
           n_iter: int  = 60,
           seed: int    = 1234,
           init_range: tuple = (-2, 3)):
    """
    Population dynamics loop for zero-T 1-RSB cavity with block mixing.

    Notes:
      - With B=1 and Q=[[1.0]] this matches your validated single-block code.
      - 'deg' is the graph degree c; the update aggregates D-1 neighbors.

    Returns:
      h          : (n_edges, B) cavity fields after n_iter iters
      h_hat_last : (n_edges, B) last single-neighbor contributions (diagnostic)
    """
    # defensively row-normalize Q
    Q = Q.astype(np.float64).copy()
    rs = Q.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    Q /= rs

    rng = np.random.default_rng(seed)
    low, high = init_range
    h = rng.integers(low, high, size=(n_edges, B)).astype(np.float64)
    #h0 = rng.integers(low, high, size=n_edges).astype(np.float64)
    #h = np.tile(h0[:, None], (1, B))

    logW_joint = np.empty(n_edges, dtype=np.float64)
    h_hat_tmp  = np.empty_like(h)

    for _ in range(n_iter):
        h, h_hat_tmp = step_dynamic_SP(h, y, Q, B, deg, logW_joint, h_hat_tmp)

    return h, h_hat_tmp

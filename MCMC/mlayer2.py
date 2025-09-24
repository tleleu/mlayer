import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------
# Periodic / block mixing kernels Q  (row-stochastic)
# ---------------------------------------------------------------------
def create_mixing_Q(
        M: int,
        mtype: str = "gauss",
        *,
        B: int | None = None,
        L: int | None = None,
        sigma: float = 1.0,
        eps: float = 1e-9,
) -> np.ndarray:
    """
    Construct an M×M mixing kernel Q (row-stochastic).
    - 'gauss'  : periodic Gaussian in layer space (distance on a ring)
    - 'block'  : periodic Gaussian in block space, repeated to layers

    Notes:
      • eps > 0 ensures strict positivity before normalization.
      • 'gauss' now uses circular distance: d(i,j) = min(|i-j|, M-|i-j|).
    """
    if mtype not in {"gauss", "block"}:
        raise ValueError("mtype must be 'gauss' or 'block'")

    if mtype == "gauss":
        x = np.arange(M, dtype=np.int64)
        # circular distance on a ring of length M
        d = np.abs(x[:, None] - x[None, :])
        dist = np.minimum(d, M - d).astype(np.float64)
        Q = np.exp(-dist**2 / (2.0 * sigma**2))
        Q += eps
        Q /= Q.sum(axis=1, keepdims=True)
        return Q

    # --- block kernel on a ring of B blocks, lifted to layers by L ---
    if M == 1:
        B = L = 1
    else:
        if B is None and L is None:
            raise ValueError("For mtype='block' supply either B or L.")
        if L is None:
            L = M // B
        if B is None:
            B = M // L
        if B * L != M:
            raise ValueError("Inconsistent block parameters:  B * L must equal M.")

    block_idx = np.arange(B, dtype=np.int64)
    dd = np.abs(block_idx[:, None] - block_idx[None, :])
    distB = np.minimum(dd, B - dd).astype(np.float64)

    Q_block = np.exp(-distB**2 / (2.0 * sigma**2))
    Q_block += eps
    Q_block /= Q_block.sum(axis=1, keepdims=True)

    # lift each block entry over L×L sub-blocks
    Q = np.kron(Q_block, np.ones((L, L), dtype=np.float64))
    Q += eps
    Q /= Q.sum(axis=1, keepdims=True)
    return Q

def create_mixing_Q_band(
        M: int,
        m: int,
        h: float,
        *,
        eps: float = 1e-12,
        periodic: bool = True,
) -> np.ndarray:
    """
    Construct an M×M band-diagonal mixing kernel Q (row-stochastic).
   
    Parameters
    ----------
    M : int
        Number of layers.
    m : int
        Band half-width. Each row connects to indices j with |i-j| <= m
        (using circular distance if periodic=True).
    h : float
        Relative height < 1 for off-diagonal entries within the band.
        Diagonal entries are 1. Values outside the band get eps.
    eps : float
        Small positive regularizer to enforce positivity.
    periodic : bool
        If True, distance is circular on a ring of length M.
   
    Returns
    -------
    Q : ndarray, shape (M, M)
        Row-stochastic mixing matrix.
    """
    x = np.arange(M, dtype=np.int64)
    d = np.abs(x[:, None] - x[None, :])
    if periodic:
        dist = np.minimum(d, M - d)
    else:
        dist = d

    # start with diagonal = 1, band entries = h, rest = 0
    Q = np.where(dist == 0, 1.0, 0.0)
    band_mask = (dist > 0) & (dist <= m)
    Q[band_mask] = h

    # enforce strict positivity everywhere
    Q += eps
    # normalize rows
    Q /= Q.sum(axis=1, keepdims=True)
    return Q

def create_mixing_Q_dir(
    M: int,
    mtype: str = "gauss",
    *,
    B: int | None = None,
    L: int | None = None,
    sigma: float = 1.0,
    shift: float = 0.0,   # center (in layer units for gauss; in layers for block too)
    skew: float = 0.0,    # in [-1, 1]; >0 fattens +Δ tail, <0 fattens −Δ tail
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Row-stochastic Q (>0). Modes:
      - 'gauss': directed, skewed Gaussian on the ring of M layers
      - 'block': same on B blocks, then lifted by L (M = B*L)
    Parameters:
      sigma : base width
      shift : preferred offset Δ (mod ring)
      skew  : asymmetric width on ±Δ. Effective width = sigma * (1 + skew*sign(rel)),
              clamped to ≥ 0.05*sigma.
    """
    if mtype not in {"gauss","block"}:
        raise ValueError("mtype must be 'gauss' or 'block'")

    def signed_offsets(n: int) -> np.ndarray:
        x = np.arange(n)
        d = (x[None,:] - x[:,None]) % n         # (b - a) mod n
        half = n // 2
        d[d > half] -= n                        # wrap to [-⌊n/2⌋, ⌈n/2⌉]
        return d.astype(np.float64)

    def two_sided_shifted(rel: np.ndarray, sigma: float, skew: float) -> np.ndarray:
        # width per entry: different on ±rel
        sgn = np.sign(rel)
        width = sigma * (1.0 + skew * sgn)
        width = np.maximum(width, 0.05 * sigma)
        return np.exp(-(rel**2) / (2.0 * width**2))

    if mtype == "gauss":
        Δ = signed_offsets(M)
        # distance to desired shift (wrapped like Δ)
        rel = (Δ - shift) % M
        half = M // 2
        rel[rel > half] -= M
        Q = two_sided_shifted(rel, sigma, skew) + eps
        Q /= Q.sum(axis=1, keepdims=True)
        return Q

    # --- block mode ---
    if M == 1:
        B = L = 1
    else:
        if B is None and L is None:
            raise ValueError("For mtype='block' supply either B or L.")
        if L is None: L = M // B
        if B is None: B = M // L
        if B * L != M:
            raise ValueError("Inconsistent block parameters: B*L must equal M.")

    # block-level directed+skewed kernel; shift given in *layers*, convert to blocks
    ΔB = signed_offsets(B)
    shift_blocks = shift / max(L, 1)
    relB = (ΔB - shift_blocks) % B
    halfB = B // 2
    relB[relB > halfB] -= B

    Q_block = two_sided_shifted(relB, sigma, skew) + eps
    Q_block /= Q_block.sum(axis=1, keepdims=True)

    Q = np.kron(Q_block, np.ones((L, L), dtype=np.float64))
    Q += eps
    Q /= Q.sum(axis=1, keepdims=True)
    return Q

def apply_generalized_diagonals_with_cycle(matrix_size, max_diagonal):
    # Create a square matrix filled with zeros
    C = np.zeros((matrix_size, matrix_size), dtype=int)
    
    # Apply the diagonals dynamically with regular boundary conditions
    for offset in range(0, max_diagonal + 1):
        np.fill_diagonal(C[offset:], 1)        # Diagonal below the main diagonal
        np.fill_diagonal(C[:, offset:], 1)     # Diagonal above the main diagonal
    
    # Apply periodic (cycle) boundary conditions by wrapping around diagonals
    for offset in range(1, max_diagonal + 1):
        for i in range(matrix_size):
            # Wrap around from top-right to bottom-left
            C[i, (i + offset) % matrix_size] = 1
            # Wrap around from bottom-left to top-right
            C[(i + offset) % matrix_size, i] = 1
    
    return C

def create_mixing_Q_step(M,width):
    
    Q = apply_generalized_diagonals_with_cycle(M, width)
    Q = Q + 0.00000001
    Q = Q/np.max(Q)
    
    row_sums = Q.sum(axis=1, keepdims=True)
    Q = np.divide(Q, row_sums, out=np.zeros_like(Q), where=row_sums!=0)

    return Q

# ---------------------------------------------------------------------
# Permanental sampler  P(π) ∝ ∏_α Q[α, π(α)]
#  - exact DP for M <= exact_threshold  (O(M^2 2^M))
#  - Perturb-and-MAP (Gumbel + Hungarian) fallback for large M
#  Row/col scaling (Sinkhorn) preserves the distribution → optional.
# ---------------------------------------------------------------------
def sinkhorn_balance(q, max_iter=1000, tol=1e-12):
    q = np.array(q, dtype=np.float64, copy=True)
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
    Q_bal = (q.T * r).T * c
    return Q_bal

def _permanental_sampler_exact(q: np.ndarray) -> np.ndarray:
    M = q.shape[0]
    full_mask = (1 << M) - 1
    # dp[r, mask] = perm( q[r:, columns in 'mask'] )
    dp = np.zeros((M + 1, 1 << M), dtype=np.float64)
    dp[M, 0] = 1.0

    # bottom-up DP over minors
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

    # sequential sampling
    perm = np.empty(M, dtype=np.int64)
    mask = full_mask
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
            # numeric fallback: pick maximal available q[r,j]
            bestv, bestj = -1.0, -1
            for j in range(M):
                if (mask >> j) & 1:
                    if q[r, j] > bestv:
                        bestv, bestj = q[r, j], j
            chosen = bestj
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

def _permanental_sampler_gumbel_hungarian(q: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    M = q.shape[0]
    # i.i.d. Gumbel(0,1) noise
    G = -np.log(-np.log(np.random.rand(M, M) + eps) + eps)
    S = np.log(q + eps) + G
    # maximize ⇒ minimize negative scores
    row_ind, col_ind = linear_sum_assignment(-S)
    return col_ind

def generate_permutation_permanental(Q: np.ndarray, exact_threshold: int = 14, balance: bool = True) -> np.ndarray:
    """
    Draw a permutation π ~ P(π) ∝ ∏_α Q[α, π(α)], Q>0.
    For M <= exact_threshold use exact DP; otherwise perturb-and-MAP.
    """
    q = np.asarray(Q, dtype=np.float64)
    if balance:
        q = sinkhorn_balance(q)
    M = q.shape[0]
    if M <= exact_threshold:
        return _permanental_sampler_exact(q)
    else:
        return _permanental_sampler_gumbel_hungarian(q)


# ---------------------------------------------------------------------
# Utility: flatten (layer, node) → global index
# ---------------------------------------------------------------------
def flatten_index(a: int, i: int, N: int) -> int:
    return a * N + i


# ---------------------------------------------------------------------
# M-layer lift using permanental sampler (default symmetric wiring)
# ---------------------------------------------------------------------
def Mlayer(J: np.ndarray, M: int, Q: np.ndarray, typeperm: str = 'sym',
           exact_threshold: int = 14, balance: bool = True) -> sp.csr_matrix:
    """
    Build the M-layer lifted coupling matrix using the structured permutation ensemble.
    - For each base edge (i,j), sample a permutation π with P(π) ∝ ∏_α Q[α, π(α)].
    - Symmetric case ('sym'): use the same π for both directions (undirected Ising).
    - Asymmetric ('asym'): draws two independent permutations (not recommended for Ising).

    Returns CSR matrix of shape (N*M, N*M).
    """
    if M == 1:
        return sp.csr_matrix(J)

    N = J.shape[0]
    J_ = np.triu(J)
    rows, cols = np.nonzero(J_)

    irow_M = []
    icol_M = []
    qdata_M = []

    for i, j in zip(rows, cols):
        Jij = J_[i, j]
        if Jij == 0.0:
            continue

        if typeperm == 'sym':
            perm = generate_permutation_permanental(Q, exact_threshold=exact_threshold, balance=balance)
            for a in range(M):
                b = int(perm[a])
                # add both directions for symmetry
                irow_M.append(flatten_index(b, i, N));  icol_M.append(flatten_index(a, j, N));  qdata_M.append(Jij)
                irow_M.append(flatten_index(a, j, N));  icol_M.append(flatten_index(b, i, N));  qdata_M.append(Jij)

        elif typeperm == 'asym':
            # not recommended for Ising; included for completeness
            perm1 = generate_permutation_permanental(Q, exact_threshold=exact_threshold, balance=balance)
            perm2 = generate_permutation_permanental(Q, exact_threshold=exact_threshold, balance=balance)
            
            if False:
                import utils
                per_row, mean_shift, direction, strength, vals, probs = utils.shift_metrics_from_perm(perm1)
                utils.plot_shift_distribution(vals, probs, outpath="energy/shift.pdf")
                
                s1, c1 = utils.best_shift(perm1)            # NOT a list-comp
                print("best_shift:", s1, "coherence:", c1)
            
            for a in range(M):
                b1 = int(perm1[a])
                b2 = int(perm2[a])
                irow_M.append(flatten_index(b1, i, N));  icol_M.append(flatten_index(a, j, N));  qdata_M.append(Jij)
                irow_M.append(flatten_index(b2, j, N));  icol_M.append(flatten_index(a, i, N));  qdata_M.append(Jij)
        else:
            raise ValueError("typeperm must be 'sym' or 'asym'.")

    JM = sp.csr_matrix((qdata_M, (irow_M, icol_M)), shape=(N * M, N * M))
    return JM
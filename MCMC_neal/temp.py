#old way to generate mlayer
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
  
def generate_permutation(C, *, eliminate_drift=False, rng=None):
    """
    Sequential without-replacement sampler.
    If eliminate_drift=True, apply:
      - random row order,
      - random cyclic rotation of rows+cols (for ring kernels),
      - 50% orientation flip via C^T.
    """
    rng = np.random.default_rng(rng)
    M = C.shape[0]

    def core(Cmat, row_order):
        perm = np.empty(M, dtype=int)
        assigned = np.zeros(M, dtype=bool)
        for a in row_order:
            probs = Cmat[a].astype(float).copy()
            probs[assigned] = 0.0
            s = probs.sum()
            if s <= 0.0:
                # uniform over remaining columns
                choices = np.flatnonzero(~assigned)
                b = rng.choice(choices)
            else:
                probs /= s
                b = rng.choice(M, p=probs)
            perm[a] = b
            assigned[b] = True
        return perm

    if not eliminate_drift:
        return core(C, np.arange(M))

    # --- drift elimination ---
    # 1) random orientation flip with prob 1/2
    use_CT = rng.random() < 0.5
    C_eff = C.T if use_CT else C

    # 2) random cyclic rotation (preserves ring-locality, kills global direction)
    k = rng.integers(M) if M > 1 else 0
    C_rot = np.roll(np.roll(C_eff, -k, axis=0), -k, axis=1)

    # 3) random row order
    row_order = rng.permutation(M)

    # sample on rotated, possibly transposed kernel
    perm_rot = core(C_rot, row_order)

    # undo rotation
    # perm_rot maps a' -> b' in rotated indices; map back: a=(a'+k)%M, b=(b'+k)%M
    perm = np.empty(M, dtype=int)
    for a_prime, b_prime in enumerate(perm_rot):
        a = (a_prime + k) % M
        b = (b_prime + k) % M
        perm[a] = b

    # if we sampled on C^T, we built b->a; invert to get a->b
    if use_CT:
        inv = np.empty(M, dtype=int)
        inv[perm] = np.arange(M)
        perm = inv

    return perm

def directed_permutation_sequential(C, *, kappa=0.0, eliminate_drift=False, rng=None):
    """
    C: row-stochastic MxM.
    shift: target Δ (can be float). Positive favors b ≈ a+shift.
    tau: width of the peak around 'shift'. Smaller -> sharper.
    eliminate_drift: if True, random phase and ±shift to zero-out global drift.
    """
    tau=5.0
    
    rng = np.random.default_rng(rng)
    M = C.shape[0]
    a = np.arange(M)

    # Δ(a,b) = (b-a) wrapped into [-M//2, ..., +M//2] with tie kept as +M/2
    Δ = (a[None, :] - a[:, None]) % M
    half = M // 2
    Δ[Δ > half] -= M

    def wrap_rel(d, s):
        # relative distance to desired shift 's', wrapped to same interval
        x = (d - s) % M
        x[x > half] -= M
        return x

    def core(Cmat, s_eff):
        perm = np.empty(M, dtype=int)
        assigned = np.zeros(M, dtype=bool)
        eps = 1e-300
        for ai in range(M):
            rel = wrap_rel(Δ[ai].astype(float), s_eff)
            # bias toward Δ ≈ s_eff
            logits = np.log(Cmat[ai].astype(float) + eps) - 0.5 * (rel / max(tau, 1e-9))**2
            probs = np.exp(logits - logits.max())
            probs[assigned] = 0.0
            s = probs.sum()
            if s <= 0.0:
                cand = np.flatnonzero(~assigned)
                b = rng.choice(cand) if cand.size else int(np.argmax(probs))
            else:
                probs /= s
                b = rng.choice(M, p=probs)
            perm[ai] = b
            assigned[b] = True
        return perm

    if not eliminate_drift:
        return core(C, kappa)

    # phase randomization and sign flip of the kappa
    k = rng.integers(M) if M > 1 else 0
    Crot = np.roll(np.roll(C, -k, axis=0), -k, axis=1)
    s_eff = kappa if (rng.random() < 0.5) else -kappa
    p_rot = core(Crot, s_eff)

    # undo rotation
    perm = np.empty(M, dtype=int)
    for ap, bp in enumerate(p_rot):
        a0 = (ap + k) % M
        b0 = (bp + k) % M
        perm[a0] = b0
    return perm

def Mlayer1(J, M, permute=True, GoG=False, typeperm='asym', C=[], eliminate_drift=False, kappa=0.0):
    
    def generate_permutation_(C):
        if np.abs(kappa)>0:
            return directed_permutation_sequential(C, kappa=kappa, eliminate_drift=eliminate_drift, rng=None)
        else:
            return generate_permutation(C,eliminate_drift=eliminate_drift)
           
    if M==1:
        return sp.csr_matrix(J)

    N = J.shape[0]
    
    Q = np.triu(J)
    
    nzi = np.nonzero(Q)
    
    irow = []
    icol = []
    qdata = []
    for i, j in zip(nzi[0], nzi[1]):
        irow.append(i)
        icol.append(j)
        qdata.append(Q[i, j])
        
    def flatten_index(a, i, N):
        return a * N + i

    irow_M = []
    icol_M = []
    qdata_M = []
    for i, j, q in zip(irow,icol,qdata):
 
        perm1 = generate_permutation_(C)
        perm2 = generate_permutation_(C)
            
        for a in range(M):
            
            b1 = perm1[a]
            b2 = perm2[a]
            
            irow_M.append(flatten_index(b1, i, N))
            icol_M.append(flatten_index(a, j, N))
            qdata_M.append(Q[i, j])
            
            irow_M.append(flatten_index(b2, j, N))
            icol_M.append(flatten_index(a, i, N))
            qdata_M.append(Q[i, j])
  
    JM = sp.csr_matrix((qdata_M, (irow_M, icol_M)), shape=(N*M, N*M))
    
            
    return JM


#new way to generate mlayer

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
def Mlayer2(J: np.ndarray, M: int, Q: np.ndarray, typeperm: str = 'sym',
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

        perm1 = generate_permutation_permanental(Q, exact_threshold=exact_threshold, balance=balance)
        perm2 = generate_permutation_permanental(Q, exact_threshold=exact_threshold, balance=balance)
        for a in range(M):
            b1 = int(perm1[a])
            b2 = int(perm2[a])
            irow_M.append(flatten_index(b1, i, N));  icol_M.append(flatten_index(a, j, N));  qdata_M.append(Jij)
            irow_M.append(flatten_index(b2, j, N));  icol_M.append(flatten_index(a, i, N));  qdata_M.append(Jij)

    JM = sp.csr_matrix((qdata_M, (irow_M, icol_M)), shape=(N * M, N * M))
    return JM
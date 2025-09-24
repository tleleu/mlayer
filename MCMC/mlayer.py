import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import random

def flatten_index(a, i, N):
    return a * N + i

def generate_permutation(C):
    M = C.shape[0]  # Size of the matrix
    permutation = np.zeros(M, dtype=int)  # To store the permutation
    assigned = np.full(M, False)  # To track which elements have been assigned

    for a in range(M):
        # Row C[a, :] gives the probabilities of permuting a to any b
        row_probabilities = C[a, :].copy()
        # Set the probabilities of already assigned elements to 0
        row_probabilities[assigned] = 0
        # Normalize the row so the sum of the probabilities is 1
        row_probabilities /= np.sum(row_probabilities)
        # Sample b from the row based on the given probabilities
        b = np.random.choice(M, p=row_probabilities)
        # Assign the permutation of a -> b
        permutation[a] = b
        # Mark b as assigned
        assigned[b] = True
    return permutation

import numpy as np

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
    Construct an M×M mixing kernel Q.

    Parameters
    ----------
    M      : int
        Total number of layers (replicas).
    mtype  : {'gauss', 'block'}
        'gauss'  -> per‑layer Gaussian (original behaviour).
        'block'  -> Gaussian kernel defined on blocks of size L.
    B      : int, optional
        Number of blocks (only for mtype='block').
    L      : int, optional
        Block size  (only for mtype='block').
        Exactly one of B or L must be given; the other is inferred.
    sigma  : float, optional
        Std‑dev of the Gaussian; interpreted in *layer* units for
        mtype='gauss' and in *block* units for mtype='block'.
    eps    : float, optional
        Small positive value added before normalisation to avoid zeros.

    Returns
    -------
    Q : ndarray, shape (M, M)
        Row‑stochastic mixing kernel.
    """
    if mtype not in {"gauss", "block"}:
        raise ValueError("mtype must be 'gauss' or 'block'")

    # ------------------------------------------------------------------
    # 1)  Per‑layer Gaussian (diagonal‑centred)
    # ------------------------------------------------------------------
    if mtype == "gauss":
        x = np.arange(M)
        Q = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * sigma**2))
        Q += eps                        # ensure strictly positive entries
        Q /= Q.sum(axis=1, keepdims=True)
        return Q

    # ------------------------------------------------------------------
    # 2)  Block‑Gaussian kernel
    # ------------------------------------------------------------------
    # Resolve B and L so that  B * L == M
    if M == 1:              # trivial cover → exactly one block of size 1
        B = L = 1
    else:                   # standard logic
        if B is None and L is None:
            raise ValueError("For mtype='block' supply either B or L.")
        if L is None:
            L = M // B
        if B is None:
            B = M // L
        if B * L != M:
            raise ValueError("Inconsistent block parameters:  B * L must equal M.")
            
    block_idx = np.arange(B)
    dist = np.minimum(np.abs(block_idx[:, None] - block_idx[None, :]),
                      B - np.abs(block_idx[:, None] - block_idx[None, :]))
    Q_block = np.exp(-dist**2 / (2.0 * sigma**2))
    
    Q_block += eps
    Q_block /= Q_block.sum(axis=1, keepdims=True)  # row-stochastic
    
    # Lift to layer space by repeating each block entry over L×L sub‑blocks
    Q = np.kron(Q_block, np.ones((L, L)))  # shape (M, M)
    Q += eps
    Q /= Q.sum(axis=1, keepdims=True)  # final row normalisation

    return Q


def Mlayer(J, M, Q, typeperm='asym'):
    
    if M==1:
        return sp.csr_matrix(J)

    N = J.shape[0]
    J_ = np.triu(J)
    nzi = np.nonzero(J_)
    
    irow = []
    icol = []
    qdata = []
    for i, j in zip(nzi[0], nzi[1]):
        irow.append(i)
        icol.append(j)
        qdata.append(J_[i, j])
           
    irow_M = []
    icol_M = []
    qdata_M = []
    for i, j, q in zip(irow,icol,qdata):
        
        perm1 = generate_permutation(Q)
        perm2 = generate_permutation(Q)
            
        for a in range(M):
              
            b1 = perm1[a]
            b2 = perm2[a]
            
            if typeperm == 'asym':
                
                    irow_M.append(flatten_index(b1, i, N))
                    icol_M.append(flatten_index(a, j, N))
                    qdata_M.append(J_[i, j])
                    irow_M.append(flatten_index(b2, j, N))
                    icol_M.append(flatten_index(a, i, N))
                    qdata_M.append(J_[i, j])
                    
            if typeperm == 'sym':
           
                    irow_M.append(flatten_index(b1, i, N))
                    icol_M.append(flatten_index(a, j, N))
                    qdata_M.append(J_[i, j])
                    irow_M.append(flatten_index(a, j, N))
                    icol_M.append(flatten_index(b1, i, N))
                    qdata_M.append(J_[i, j])
                    
    JM = sp.csr_matrix((qdata_M, (irow_M, icol_M)), shape=(N*M, N*M))
    
    return JM
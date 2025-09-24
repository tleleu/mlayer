# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Zero‑temperature Glauber dynamics on a sparse (CSR) Ising graph.

Design notes
------------
* *Typed memoryviews only* – no NumPy overhead in the inner loop.
* Heavy compute is run **without the GIL** (`with nogil:`) so other Python
  threads aren’t blocked.  If you later upgrade to Cython 3.x, you can wrap the
  `for k in range(K)` with `prange` to get full OpenMP parallelism.
* In‑place Fisher–Yates shuffle avoids costly `np.random.permutation`.
"""
import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, srand
from libc.time   cimport time

# ---------------------------------------------------------------------
# helper : Fisher–Yates shuffle of an int32 buffer  (GIL‑free & noexcept)
# ---------------------------------------------------------------------
cdef inline void _shuffle_int32(int[:] arr) nogil noexcept:
    cdef Py_ssize_t i, swap_pos
    cdef int tmp
    for i in range(arr.shape[0] - 1, 0, -1):
        swap_pos      = rand() % (i + 1)
        tmp           = arr[i]
        arr[i]        = arr[swap_pos]
        arr[swap_pos] = tmp


# ---------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------
cpdef np.ndarray[np.int8_t, ndim=2] run_mcmc_seq_sparse(
        double[:] data,            # CSR data
        int[:]    indices,         # CSR indices
        int[:]    indptr,          # CSR indptr (len == N+1)
        int       N,               # number of spins
        signed char[:, :] sigma,   # (K, N) replicas – updated in place
        double    theta,
        double    beta,            # unused at T = 0 (kept for API symmetry)
        int       seed,
        int       steps):
    """
    Perform `steps` full‑lattice sweeps of zero‑temperature Glauber dynamics.
    """
    # locals
    cdef Py_ssize_t K = sigma.shape[0]
    cdef int[:] order = np.arange(N, dtype=np.int32)
    cdef int step, i, j, k, idx
    cdef int row_start, row_end
    cdef double h_i

    srand(seed if seed >= 0 else <unsigned int>time(NULL))

    for step in range(steps):
        _shuffle_int32(order)                 # cheap; keep GIL

        # heavy compute – run without the GIL
        with nogil:
            for k in range(K):
                for i in range(N):
                    idx       = order[i]
                    h_i       = theta
                    row_start = indptr[idx]
                    row_end   = indptr[idx + 1]
                    for j in range(row_start, row_end):
                        h_i += data[j] * sigma[k, indices[j]]
                    if   h_i > 0.0:
                        sigma[k, idx] =  1
                    elif h_i < 0.0:
                        sigma[k, idx] = -1
                    # else unchanged

    return np.asarray(sigma)

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Simulated annealing MCMC on a sparse (CSR) Ising graph.

- Graph: arbitrary weighted CSR (data, indices, indptr)
- Spins: signed char in {+1, -1}, shape (K, N) for K replicas
- Updates:
    * T == 0:   zero-temperature quench (deterministic sign(h_i))
    * T  > 0:   heat-bath (Glauber) by default, or Metropolis
- Schedule: temps[:] (vector of temperatures). Do `sweeps_per_temp` full sweeps at each T.

Speed notes
-----------
* Typed memoryviews + local C pointers for CSR arrays (cheap inner loop).
* Hot loops run without the GIL; only the in-place Fisher–Yates shuffle holds the GIL briefly.
* RNG: libc rand()/srand() (fast). For strict reproducibility across platforms,
  consider swapping in a fixed RNG (e.g., xorshift) later.

Compile with OpenMP to parallelize over replicas (K) by replacing `for k in range(K)` with
`for k in prange(K, nogil=True)` (see comment near the loop).
"""
import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time   cimport time
from libc.math   cimport exp

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
# numerics helpers
# ---------------------------------------------------------------------
cdef inline double _sigmoid(double x) nogil:
    """Stable logistic 1/(1+exp(-x)) without overflow."""
    cdef double z
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    else:
        z = exp(x)
        return z / (1.0 + z)

cdef inline double _urand(double inv_rand_maxp1) nogil:
    """Uniform in [0,1) using libc rand()."""
    return (<double>rand()) * inv_rand_maxp1


# ---------------------------------------------------------------------
# main routine: simulated annealing over a temperature schedule
# ---------------------------------------------------------------------
cpdef np.ndarray[np.int8_t, ndim=2] run_mcmc_seq_sparse_anneal(
        double[:] data,            # CSR data
        int[:]    indices,         # CSR indices
        int[:]    indptr,          # CSR indptr (len == N+1)
        int       N,               # number of spins
        signed char[:, :] sigma,   # (K, N) replicas – updated in place
        double    theta,           # external field
        double[:] temps,           # temperature schedule (len = nT)
        int       sweeps_per_temp, # full-lattice sweeps per temperature
        int       seed,
        int       update_rule=0    # 0 = heat-bath (Glauber), 1 = Metropolis
    ):
    """
    Perform simulated annealing on a general sparse Ising graph (CSR).

    Parameters
    ----------
    data, indices, indptr : CSR arrays defining couplings J_ij (data) and adjacency.
    N      : number of spins.
    sigma  : (K, N) spins (±1) for K independent replicas; updated in place.
    theta  : external field term added to the local field h_i.
    temps  : vector of temperatures. If any entry is 0.0, that step is a quench.
    sweeps_per_temp : number of full sweeps to run at each temperature.
    seed   : RNG seed (if <0, seeds from time()).
    update_rule : 0 for heat-bath (Glauber), 1 for Metropolis.

    Returns
    -------
    sigma : the same array (updated) returned as a NumPy view for convenience.
    """
    cdef Py_ssize_t K = sigma.shape[0]
    if sigma.shape[1] != N:
        raise ValueError("sigma.shape[1] must equal N")

    # iteration state
    cdef int nT = temps.shape[0]
    cdef int[:] order = np.arange(N, dtype=np.int32)

    # raw C pointers for CSR arrays (slightly faster than MV indexing in hot loop)
    cdef double* data_p   = &data[0]
    cdef int*    indices_p= &indices[0]
    cdef int*    indptr_p = &indptr[0]

    cdef int t, sweep, i, j, k, idx
    cdef int row_start, row_end
    cdef double T, beta, h_i, u, dE, prob, x
    cdef double inv_rand_maxp1 = 1.0 / ((<double>RAND_MAX) + 1.0)

    srand(seed if seed >= 0 else <unsigned int>time(NULL))

    for t in range(nT):
        T = temps[t]

        # ---- Zero temperature: quench (deterministic sign of local field)
        if T <= 0.0:
            for sweep in range(sweeps_per_temp):
                _shuffle_int32(order)  # cheap; can keep GIL or move inside nogil

                # heavy compute – run without the GIL
                with nogil:
                    # (Optional) parallelize replicas with OpenMP:
                    # from cython.parallel cimport prange
                    # for k in prange(K, nogil=True):
                    for k in range(K):
                        for i in range(N):
                            idx       = order[i]
                            h_i       = theta
                            row_start = indptr_p[idx]
                            row_end   = indptr_p[idx + 1]
                            for j in range(row_start, row_end):
                                h_i += data_p[j] * sigma[k, indices_p[j]]

                            if   h_i > 0.0:
                                sigma[k, idx] =  1
                            elif h_i < 0.0:
                                sigma[k, idx] = -1
                            # else: unchanged if h_i == 0
            continue

        # ---- Finite temperature
        beta = 1.0 / T
        for sweep in range(sweeps_per_temp):
            _shuffle_int32(order)

            with nogil:
                if update_rule == 0:
                    # -------- Heat-bath (Glauber): s_i := +1 with p = 1/(1+exp(-2βh_i))
                    for k in range(K):
                        for i in range(N):
                            idx       = order[i]
                            h_i       = theta
                            row_start = indptr_p[idx]
                            row_end   = indptr_p[idx + 1]
                            for j in range(row_start, row_end):
                                h_i += data_p[j] * sigma[k, indices_p[j]]

                            x    = 2.0 * beta * h_i
                            prob = _sigmoid(x)
                            u    = _urand(inv_rand_maxp1)
                            sigma[k, idx] = 1 if u < prob else -1

                else:
                    # -------- Metropolis: flip with prob min(1, exp(-β ΔE)), ΔE = 2 s_i h_i
                    for k in range(K):
                        for i in range(N):
                            idx       = order[i]
                            h_i       = theta
                            row_start = indptr_p[idx]
                            row_end   = indptr_p[idx + 1]
                            for j in range(row_start, row_end):
                                h_i += data_p[j] * sigma[k, indices_p[j]]

                            dE = 2.0 * sigma[k, idx] * h_i
                            if dE > 0.0:
                                prob = exp(-beta * dE)     # safe (argument ≤ 0)
                                u    = _urand(inv_rand_maxp1)
                                if u < prob:
                                    sigma[k, idx] = -sigma[k, idx]
                            elif dE < 0.0:
                                sigma[k, idx] = -sigma[k, idx]
                            # else ΔE == 0 -> unchanged

    return np.asarray(sigma)

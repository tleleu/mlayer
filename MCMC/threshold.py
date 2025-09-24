import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from ising_mcmc_sparse import run_mcmc_seq_sparse
from sa_sparse_ising import run_mcmc_seq_sparse_anneal

from instance import create_Bethe
#from mlayer   import Mlayer, create_mixing_Q
from mlayer2   import Mlayer, create_mixing_Q
import stats   # helper module with calculate_energy_replicas, is_local_minimum, …

import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_single_mcmc(J, J0, N, K, steps, theta, beta, seed):
    sigma = np.random.choice([-1, 1], size=(K, N)).astype(np.int8)
    J_csr = sp.csr_matrix(J, dtype=np.float64)
    return run_mcmc_seq_sparse(
        J_csr.data,
        J_csr.indices.astype(np.int32, copy=False),
        J_csr.indptr.astype(np.int32, copy=False),
        N,
        sigma, theta, beta, seed, steps,
    )


def run_single_mcmc_sa(J, J0, N, K, steps, theta, beta, seed):
    """
    Simulated annealing version of run_single_mcmc with the same signature.

    Parameters are identical to your zero-T wrapper; J0 and beta are kept for
    compatibility and not used here. The annealing schedule is chosen
    automatically from acceptance-probability heuristics and uses geometric
    cooling with one sweep per temperature.

    Returns
    -------
    sigma_final : (K, N) int8 array of spins (+/-1), final configurations.
    """
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    # --- initial spins (K replicas), reproducible with the given seed
    rng = np.random.default_rng(seed if seed is not None else None)
    sigma = rng.choice([-1, 1], size=(K, N)).astype(np.int8)

    # --- convert weights exactly as in the zero-T wrapper
    J_csr = sp.csr_matrix(J, dtype=np.float64)
    ind32 = J_csr.indices.astype(np.int32, copy=False)
    ptr32 = J_csr.indptr.astype(np.int32, copy=False)

    # --- pick ΔE_ref from a random configuration (Metropolis acceptance model)
    #     ΔE_i = 2 s_i h_i, with h = theta + J * s  (use one replica for speed)
    s0 = rng.choice([-1, 1], size=N).astype(np.int8)
    h = theta + (J_csr @ s0.astype(np.float64))
    dE = 2.0 * s0.astype(np.float64) * h
    pos = dE[dE > 0.0]
    if pos.size > 0:
        dE_ref = float(np.median(pos))
    else:
        # Degenerate fallback: use typical absolute row-sum scale
        # ΔE_ref ≈ 2*(|theta| + median_j sum_i |J_ji|)
        row_abs = np.asarray(np.abs(J_csr).sum(axis=1)).ravel()
        dE_ref = 2.0 * (abs(theta) + float(np.median(row_abs)))
        if dE_ref == 0.0:
            dE_ref = 1.0  # final fallback

    # --- choose SA endpoints from target uphill acceptance probabilities
    p_start = 0.80   # ~80% uphill acceptance at start
    p_final = 1e-3   # ~0.1% uphill acceptance at the end
    T_max = -dE_ref / np.log(p_start)
    T_min = -dE_ref / np.log(p_final)
    # guardrails
    T_max = float(max(T_max, 1e-12))
    T_min = float(max(min(T_min, T_max), 1e-12))

    # --- standard SA schedule: geometric (exponential) cooling
    if steps == 1:
        temps = np.array([T_min], dtype=np.float64)
    else:
        temps = np.geomspace(T_max, T_min, steps, dtype=np.float64)

    # --- one full sweep per temperature (classic SA)
    sweeps_per_temp = 1
    update_rule = 1  # 1 = Metropolis (matches acceptance model used to pick T's)

    sigma_final = run_mcmc_seq_sparse_anneal(
        J_csr.data,           # double[:]
        ind32,                # int[:]
        ptr32,                # int[:]
        N,
        sigma,                # signed char[:, :]
        float(theta),
        temps,                # double[:], length == steps
        sweeps_per_temp,
        int(seed) if seed is not None else -1,
        update_rule
    )

    return sigma_final

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
k       = 2

if True: #longer simulation
    N0      = 30
    L = 30
    Ml      = L*np.array([2, 4, 10])             # layer multipliers
    #SigmaL = np.exp(np.linspace(np.log(0.2),np.log(2.9),20))    # Gaussian widths
    SigmaL = np.linspace(0.1,2.0,15)
    reps    = 5
    K       = 30
    steps0  = 200
    typeperm = 'asym'
else: #short
    N0      = 30
    L = 30
    Ml      = L*np.array([2, 4])             # layer multipliers
    #SigmaL = np.exp(np.linspace(np.log(0.2),np.log(2.9),20))    # Gaussian widths
    SigmaL = np.linspace(0.1,2.0,15)
    reps    = 5
    K       = 10
    steps0  = 50
    typeperm = 'asym'
seed0   = 42

beta  = 1_000.0
theta = 0.0

show_gap = False
show_residual = False

run_SA = False

# ---------------------------------------------------------------------------
# Storage:  mean energies for each (σ, M, rep)
# ---------------------------------------------------------------------------
Emean = np.zeros((len(SigmaL), len(Ml), reps))   # average over K replicas
Qavg   = np.zeros_like(Emean)                     # qavg for each cell

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
for r in tqdm(range(reps)):
    J0      = create_Bethe(N0, k + 1)          # fixed disorder for this rep
    e0_r    = np.inf                           # global ground state tracker

    for i_sigma, sigma in enumerate(SigmaL):
        for iM, M in enumerate(Ml):
            # --- Build coupling
            Q = create_mixing_Q(M, mtype="block", sigma=sigma, L=L)
            J = np.array(Mlayer(J0.todense(), M, Q, typeperm=typeperm).todense())

            # --- Run MCMC
            N, steps = N0 * M, steps0 * M
            if run_SA == False:
                spins = run_single_mcmc(J, J0, N, K, steps,
                                        theta, beta, seed0 + r)
            else:
                spins = run_single_mcmc_sa(J, J0, N, K, steps,
                                        theta, beta, seed0 + r)

            # --- Energy statistics
            energy  = stats.calculate_energy_replicas(spins, J0.todense(), M)
            Emean[i_sigma, iM, r] = np.mean(energy)
            e0_r = min(e0_r, energy.min())     # update global ground state

            # --- Overlap qavg
            spins_reshaped = spins.reshape(K, M, N0)         # (K,M,N)
            # dot product along N:  (K,M,N) · (K,N,M) -> (K,M,M)
            overlap = np.matmul(spins_reshaped, spins_reshaped.transpose(0, 2, 1))
            # remove diagonal & average over M(M‑1)
            off_diag_sum = overlap.sum(axis=(1, 2)) - np.trace(overlap, axis1=1, axis2=2)
            if M>1:
                qavg_replica = off_diag_sum / (M * (M - 1)) / N0
                Qavg[i_sigma, iM, r] = qavg_replica.mean()      # average over K
            else:
                Qavg[i_sigma, iM, r] = np.nan

    # convert mean energy → residual energy for this rep
    if show_residual:
        Emean[:, :, r] -= e0_r

# ---------------------------------------------------------------------------
# Aggregate across repetitions
# ---------------------------------------------------------------------------
mean_res  = Emean.mean(axis=2)
ci95_res  = 1.96 * Emean.std(axis=2) / np.sqrt(reps)

mean_qavg = Qavg.mean(axis=2)
ci95_qavg = 1.96 * Qavg.std(axis=2) / np.sqrt(reps)

# ---------------------------------------------------------------------------
# Plot: residual energy  &  qavg  vs σ
# ---------------------------------------------------------------------------

def lambda2_of_sigma(sigma, M_ref=20, L=2):
    """Return the 2nd‑largest eigenvalue of Q for a given σ.
    Uses a small reference M (any M works because λ’s of the
    Kronecker‑expanded kernel equal those of the block kernel)."""
    Q = create_mixing_Q(M_ref, mtype="block", sigma=sigma, L=L)
    eigvals = np.sort(np.real(np.linalg.eigvals(Q)))[::-1]  # descending
    return eigvals[1]

fig = plt.figure(figsize=(12, 4))

# -- 4.1  residual energy ---------------------------------------------------
plt.subplot(1, 2, 1)
for iM, M in enumerate(Ml):
    if show_gap:
        Lambda2L = np.array([lambda2_of_sigma(s, M_ref=M, L=2) for s in SigmaL])
        xval = 1-Lambda2L
    else:
        xval = SigmaL

    plt.errorbar(
        xval,                       # <-- X axis
        mean_res[:, iM],                # residual energy
        yerr=ci95_res[:, iM],
        marker='o',
        label=f'M={M}',
    )
    
if show_gap:
    plt.xscale('log')
    plt.xlabel(r'spectral gap 1-$\lambda_2$')
else:
    plt.xlabel(r'\sigma$')
plt.title('a')
if show_residual:
    plt.ylabel(r'residual energy $\langle e-e_0\rangle$')
    plt.yscale('log')
else:
    plt.ylabel(r'energy $\langle e\rangle$')
    plt.ylim(-1.3,-0.7)
plt.grid(True, ls=':')
plt.legend()

# -- 4.2  average overlap ---------------------------------------------------
plt.subplot(1, 2, 2)
for iM, M in enumerate(Ml):
    if show_gap:
        Lambda2L = np.array([lambda2_of_sigma(s, M_ref=M, L=2) for s in SigmaL])
        xval = 1-Lambda2L
    else:
        xval = SigmaL
        
    plt.errorbar(
        xval,
        mean_qavg[:, iM],
        yerr=ci95_qavg[:, iM],
        marker='s',
        label=f'M={M}',
    )
if show_gap:
    plt.yscale('log')
    plt.xlabel(r'spectral gap 1-$\lambda_2$')
else:
    plt.xlabel(r'$\sigma$')
plt.ylabel(r'inter-layer overlap $\langle q_{\mathrm{avg}}\rangle$')
plt.ylim(0,1.02)
plt.title('b')
plt.grid(True, ls=':')
plt.legend()
fig.suptitle(fr"$N={N0}$, $reps={reps}$, $K={K}$, $steps0={steps0}$, $L={L}$, run_SA={run_SA}")

plt.tight_layout()
plt.savefig("./threshold/threshold.pdf", format="pdf", bbox_inches="tight")
plt.show()


folder_name = "threshold"
os.makedirs(folder_name, exist_ok=True)
filename = f"threshold_N{N0}_reps{reps}_K{K}_steps{steps0}_L{L}.npz"
file_path = os.path.join(folder_name, filename)
np.savez(file_path, SigmaL=SigmaL, mean_res=mean_res, ci95_res=ci95_res,
         mean_qavg=mean_qavg, ci95_qavg=ci95_qavg)

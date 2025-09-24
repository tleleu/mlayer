import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent            # folder of this file
sys.path.insert(0, str((HERE / ".." / "MCMC").resolve()))

import SA

from instance import create_Bethe
import mlayer2
from mlayer2  import Mlayer, create_mixing_Q, create_mixing_Q_band, create_mixing_Q_step
import stats   # helper module with calculate_energy_replicas, is_local_minimum, …

import os
import lib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_single_mcmc(J, J0, N, K, steps, theta, beta, seed):
    sigma = np.random.choice([-1, 1], size=(K, N)).astype(np.int8)
    timeseq, final_spins = SA.run_SA(N,J,steps,K,beta,SAcode='neal',x0=sigma)
    return final_spins

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
k       = 2

if False: #longer simulation (1077.56s/it)
    N0      = 100
    L = 2
    Ml      = L*np.array([10, 20, 50, 100, 200, 400])             # layer multipliers
    SigmaL = np.linspace(0.01,0.10,10)
    reps    = 1
    K       = 50
    steps0  = 2*N0
    typeperm = 'asym'
else: #short
    N0      = 100
    L = 2
    Ml      = L*np.array([10, 20, 50, 100, 200])             # layer multipliers
    SigmaL = np.linspace(0.01,0.10,10)
    reps    = 1
    K       = 50
    steps0  = 2*N0
    typeperm = 'asym'
seed0   = 42

beta  = 1_000.0
theta = 0.0

show_gap = False
show_residual = True

run_SA = False

# ---------------------------------------------------------------------------
# Storage:  mean energies for each (σ, M, rep)
# ---------------------------------------------------------------------------
Emean = np.zeros((len(SigmaL), len(Ml), reps))   # average over K replicas
Qavg   = np.zeros_like(Emean)                     # qavg for each cell

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
#for r in tqdm(range(reps)):
for r in range(reps):
    J0      = create_Bethe(N0, k + 1)          # fixed disorder for this rep
    e0_r    = np.inf                           # global ground state tracker

    for iM, M in tqdm(enumerate(Ml)):
        for i_sigma, sigma in enumerate(SigmaL):
            # --- Build coupling
            
            mlayer_type = 3
            
            #new mlayer
            if mlayer_type == 0:
                Q = create_mixing_Q(M, mtype="block", sigma=sigma * M, L=L)
                J = np.array(mlayer2.Mlayer(J0.todense(), M, Q, typeperm=typeperm).todense())
            elif mlayer_type == 1:
                Q = create_mixing_Q_step(M,i_sigma+1)
                J = np.array(mlayer2.Mlayer(J0.todense(), M, Q, typeperm=typeperm).todense())
                
            #previous mlayer
            elif mlayer_type == 2:
                #J = lib.Mlayer(J0.todense(), M, permute=True, GoG=True, typeperm='asym', width=int(np.floor(sigma * M))).todense()
                Q = lib.create_mixing_Q_step(M,i_sigma+1)
                J = lib.Mlayer(J0.todense(), M, permute=True, GoG=True, typeperm='asym', C=Q).todense()

            elif mlayer_type == 3:
                Q = create_mixing_Q(M, mtype="block", sigma=sigma * M, L=L)
                J = lib.Mlayer(J0.todense(), M, permute=True, GoG=True, typeperm='asym', C=Q).todense()

            if False:
                plt.figure(figsize=(6, 5))
                plt.imshow(np.abs(J)>0, aspect='equal')          # viridis colormap by default
                plt.title(f"Mixing kernel Q  (M={Q.shape[0]})")
                plt.xlabel("Layer index β")
                plt.ylabel("Layer index α")
                plt.colorbar(label=r"$Q_{\alpha\beta}$")
                plt.tight_layout()
                plt.show()

                ERROR

            # --- Run MCMC
            N, steps = N0 * M, steps0 * M
            if run_SA == False:
                spins = run_single_mcmc(J, J0, N, K, steps,
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
# save
# ---------------------------------------------------------------------------


folder_name = "energy"
os.makedirs(folder_name, exist_ok=True)

filename = f"energy_N{N0}_reps{reps}_K{K}_steps{steps0}_L{L}.npz"
file_path = os.path.join(folder_name, filename)
np.savez(file_path, SigmaL=SigmaL, mean_res=mean_res, ci95_res=ci95_res,
         mean_qavg=mean_qavg, ci95_qavg=ci95_qavg)

# ---------------------------------------------------------------------------
# Plot: residual energy  &  qavg  vs σ
# ---------------------------------------------------------------------------

def lambda2_of_sigma(sigma, M_ref=20, L=2):
    """Return the 2nd‑largest eigenvalue of Q for a given σ.
    Uses a small reference M (any M works because λ’s of the
    Kronecker‑expanded kernel equal those of the block kernel)."""
    Q = create_mixing_Q(M_ref, mtype="block", sigma=sigma, L=L)
    #Q = create_mixing_Q_band(M=M_ref, m=L, h=sigma)
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
        
    xval = np.array(list(range(10))) + 1

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
    #plt.xlabel(r'$\sigma / M$')
    plt.xlabel(r'$d$')
plt.title('a')
if show_residual:
    plt.ylabel(r'residual energy $\langle e-e_0\rangle$')
    plt.yscale('log')
else:
    plt.ylabel(r'energy $\langle e\rangle$')
    plt.ylim(-1.3,-0.7)
#plt.xlim(0,np.max(SigmaL))
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
        
    xval = np.array(list(range(10))) + 1
       
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
    #plt.xlabel(r'$\sigma / M$')
    plt.xlabel(r'$d$')
plt.ylabel(r'inter-layer overlap $\langle q_{\mathrm{avg}}\rangle$')
plt.ylim(0,1.02)
#plt.xlim(0,np.max(SigmaL))
plt.title('b')
plt.grid(True, ls=':')
plt.legend()
fig.suptitle(fr"$N={N0}$, $reps={reps}$, $K={K}$, $steps0={steps0}$, $L={L}$, run_SA={run_SA}")

plt.tight_layout()
plt.savefig("energy/energy.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Extra figure: best value of (residual / sigma) vs M  [log–log]
# ---------------------------------------------------------------------------

# argmin over sigma for each M
min_idx = np.nanargmin(mean_res, axis=0)                         # shape (len(Ml),)
min_vals = mean_res[min_idx, np.arange(len(Ml))]                 # min residual per M
min_errs = ci95_res[min_idx, np.arange(len(Ml))]                 # corresponding 95% CI
min_sigmas = SigmaL[min_idx]                                     # sigma achieving the min


plt.figure(figsize=(5, 4))
plt.errorbar(Ml, np.clip(min_vals, np.finfo(float).tiny, None),  # avoid log(0)
             yerr=min_errs, marker='o')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('M')
plt.ylabel(r'$\min_{\sigma}\ \langle e-e_0\rangle$')
plt.title('Minimum residual vs M')
plt.grid(True, which='both', ls=':')

# annotate σ* per point (optional)
for x, y, s in zip(Ml, min_vals, min_sigmas):
    plt.annotate(fr'σ*={s:.2g}', (x, y), textcoords='offset points',
                 xytext=(5, 5), fontsize=8)

# prepare positive, finite data
tiny = np.finfo(float).tiny
x = Ml.astype(float)
y = np.clip(min_vals, tiny, None)
yerr = np.clip(min_errs, tiny, None)
mask = np.isfinite(x) & np.isfinite(y) & (y > 0)

# weights in log-space: var[log y] ≈ (yerr/y)^2
w = 1.0 / np.clip((yerr[mask] / y[mask])**2, 1e-12, None)

# linear fit in log space
logx = np.log(x[mask])
logy = np.log(y[mask])
p, loga = np.polyfit(logx, logy, deg=1, w=w)   # slope p, intercept loga
a = np.exp(loga)

# fitted curve over M range
M_fit = np.geomspace(x[mask].min(), x[mask].max(), 256)
y_fit = a * M_fit**p

# overlay
plt.plot(M_fit, y_fit, linestyle=':', color='red', label=fr'fit slope={p:.3f}')
plt.legend()


plt.tight_layout()
plt.savefig(os.path.join(folder_name, "min_residual_vs_M.pdf"),
            format="pdf", bbox_inches="tight")
plt.show()

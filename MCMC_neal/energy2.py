# -*- coding: utf-8 -*-
"""
Parallel σ-sweep version with a flag to switch to sequential.
Keeps outputs and plotting consistent with the original script.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from pathlib import Path
import sys
import os
import numpy as np
import scipy.sparse as sp  # kept for compatibility with upstream helpers
import matplotlib.pyplot as plt
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# project paths
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((HERE / ".." / "MCMC").resolve()))

import SA
from instance import create_Bethe
import mlayer2
from mlayer2 import Mlayer, create_mixing_Q, create_mixing_Q_band, create_mixing_Q_step, create_mixing_Q_dir
import stats
import lib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_single_mcmc(J, N, K, steps, beta, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.choice([-1, 1], size=(K, N)).astype(np.int8)
    _, final_spins = SA.run_SA(N, J, steps, K, beta, SAcode="neal", x0=x0)
    return final_spins

def _build_J(J0_dense, M, sigma, i_sigma, mlayer_type, typeperm, L, shift=0.0, skew=0.0):
    if mlayer_type == 0:
        #Q = create_mixing_Q(M, mtype="block", sigma=sigma * M, L=L)
        Q = create_mixing_Q(M, mtype="block", sigma=sigma * 20, L=L)
        J = np.array(mlayer2.Mlayer(J0_dense, M, Q, typeperm=typeperm).todense())
    elif mlayer_type == 1:
        #Q = create_mixing_Q(M, mtype="block", sigma=sigma * M, L=L)
        Q = create_mixing_Q(M, mtype="block", sigma=sigma * 20, L=L)
        J = lib.Mlayer(J0_dense, M, permute=True, GoG=True, typeperm="asym", C=Q).todense()
    elif mlayer_type == 2:
        Q = create_mixing_Q_step(M, i_sigma + 1)
        J = np.array(mlayer2.Mlayer(J0_dense, M, Q, typeperm=typeperm).todense())
    elif mlayer_type == 3:
        Q = lib.create_mixing_Q_step(M, i_sigma + 1)
        J = lib.Mlayer(J0_dense, M, permute=True, GoG=True, typeperm="asym", C=Q).todense()
    elif mlayer_type == 4:
        Q = create_mixing_Q_dir(M, mtype="block", sigma=sigma * 20, L=L, shift=shift, skew=skew)
        J = np.array(mlayer2.Mlayer(J0_dense, M, Q, typeperm=typeperm).todense())
    else:
        raise ValueError("unknown mlayer_type")
    return J

def _eval_sigma_task(args):
    (
        i_sigma, sigma, shift, skew, iM, M, J0_dense, N0, steps0, K, beta,
        L, mlayer_type, typeperm, seed_base
    ) = args

    J = _build_J(J0_dense, M, sigma, i_sigma, mlayer_type, typeperm, L, shift=shift, skew=skew)

    N, steps = N0 * M, steps0 * M
    spins = run_single_mcmc(J, N, K, steps, beta, seed_base + i_sigma)

    energy = stats.calculate_energy_replicas(spins, J0_dense, M)
    e_mean = float(np.mean(energy))
    e_min = float(np.min(energy))

    spins_reshaped = spins.reshape(K, M, N0)
    overlap = np.matmul(spins_reshaped, spins_reshaped.transpose(0, 2, 1))
    off_diag_sum = overlap.sum(axis=(1, 2)) - np.trace(overlap, axis1=1, axis2=2)
    qavg_val = float((off_diag_sum / (M * (M - 1)) / N0).mean()) if M > 1 else np.nan

    return i_sigma, e_mean, qavg_val, e_min

def lambda2_of_sigma(sigma, M_ref=20, L=2):
    Q = create_mixing_Q(M_ref, mtype="block", sigma=sigma, L=L)
    eigvals = np.sort(np.real(np.linalg.eigvals(Q)))[::-1]
    return eigvals[1]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
k = 2

# choose short/long preset
USE_LONG = False
if USE_LONG:
    N0 = 100
    L = 2
    Ml = L * np.array([10, 20, 50, 100, 200, 400])
    SigmaL = np.linspace(0.01, 0.10, 10)
    reps = 1
    K = 50
    steps0 = 2 * N0
    typeperm = "asym"
else:
    N0 = 100
    L = 2
    Ml = L * np.array([10, 20, 50, 100])
    SigmaL = np.linspace(0.01, 0.10, 10)
    reps = 1
    K = 50
    steps0 = 2 * N0
    typeperm = "asym"

seed0 = 42
beta = 1_000.0

show_gap = False
show_residual = True
run_SA = False  # retained for compatibility with earlier code paths

# mixing kernel impl
MLAYER_TYPE = 4  # 0,1,2,3 as in original
shift = 0.0
skew = 0.2

# parallel controls
PARALLEL_SIGMA = True        # False -> sequential σ loop
N_WORKERS = None             # None -> os.cpu_count()
START_METHOD = "spawn"       # "spawn" is portable; "fork" is fine on Linux

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # storage
    Emean = np.zeros((len(SigmaL), len(Ml), reps))
    Qavg = np.zeros_like(Emean)

    for r in range(reps):
        J0 = create_Bethe(N0, k + 1)
        J0_dense = J0.todense()
        e0_r = np.inf

        for iM, M in tqdm(enumerate(Ml), total=len(Ml), desc="M sweep"):
            seed_base = int(seed0 + 10_000 * r + 1_000 * iM)

            if PARALLEL_SIGMA:
                ctx = mp.get_context(START_METHOD)
                n_workers = N_WORKERS or min(len(SigmaL), os.cpu_count() or 1)

                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                    futures = [
                        ex.submit(
                            _eval_sigma_task,
                            (
                                i_sigma, float(sigma), float(shift), float(skew), iM, int(M), J0_dense, N0, steps0, K,
                                beta, L, MLAYER_TYPE, typeperm, seed_base
                            ),
                        )
                        for i_sigma, sigma in enumerate(SigmaL)
                    ]
                    for f in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        leave=False,
                        desc=f"M={M}"
                    ):
                        i_sigma, e_mean, qavg_val, e_min = f.result()
                        Emean[i_sigma, iM, r] = e_mean
                        Qavg[i_sigma, iM, r] = qavg_val
                        if e_min < e0_r:
                            e0_r = e_min
            else:
                for i_sigma, sigma in enumerate(SigmaL):
                    i_s, e_mean, qavg_val, e_min = _eval_sigma_task(
                        (
                            i_sigma, float(sigma), iM, int(M), J0_dense, N0, steps0, K,
                            beta, L, MLAYER_TYPE, typeperm, seed_base
                        )
                    )
                    Emean[i_s, iM, r] = e_mean
                    Qavg[i_s, iM, r] = qavg_val
                    if e_min < e0_r:
                        e0_r = e_min

        if show_residual:
            Emean[:, :, r] -= e0_r

    # aggregate
    mean_res = Emean.mean(axis=2)
    ci95_res = 1.96 * Emean.std(axis=2) / np.sqrt(reps)
    mean_qavg = Qavg.mean(axis=2)
    ci95_qavg = 1.96 * Qavg.std(axis=2) / np.sqrt(reps)

    # save
    folder_name = "energy"
    os.makedirs(folder_name, exist_ok=True)
    filename = f"energy_N{N0}_reps{reps}_K{K}_steps{steps0}_L{L}.npz"
    file_path = os.path.join(folder_name, filename)
    np.savez(
        file_path,
        SigmaL=SigmaL,
        mean_res=mean_res,
        ci95_res=ci95_res,
        mean_qavg=mean_qavg,
        ci95_qavg=ci95_qavg,
    )

    # plots
    fig = plt.figure(figsize=(12, 4))

    # 4.1 residual energy
    plt.subplot(1, 2, 1)
    for iM, M in enumerate(Ml):
        if show_gap:
            Lambda2L = np.array([lambda2_of_sigma(s, M_ref=M, L=2) for s in SigmaL])
            xval = 1 - Lambda2L
        else:
            xval = SigmaL
        # preserve original override
        xval = np.arange(len(SigmaL)) + 1
        plt.errorbar(
            xval,
            mean_res[:, iM],
            yerr=ci95_res[:, iM],
            marker="o",
            label=f"M={M}",
        )
    if show_gap:
        plt.xscale("log")
        plt.xlabel(r'spectral gap 1-$\lambda_2$')
    else:
        plt.xlabel(r'$d$')
    plt.title('a')
    if show_residual:
        plt.ylabel(r'residual energy $\langle e-e_0\rangle$')
        plt.yscale('log')
    else:
        plt.ylabel(r'energy $\langle e\rangle$')
        plt.ylim(-1.3, -0.7)
    plt.grid(True, ls=':')
    plt.legend()

    # 4.2 average overlap
    plt.subplot(1, 2, 2)
    for iM, M in enumerate(Ml):
        if show_gap:
            Lambda2L = np.array([lambda2_of_sigma(s, M_ref=M, L=2) for s in SigmaL])
            xval = 1 - Lambda2L
        else:
            xval = SigmaL
        # preserve original override
        xval = np.arange(len(SigmaL)) + 1
        plt.errorbar(
            xval,
            mean_qavg[:, iM],
            yerr=ci95_qavg[:, iM],
            marker='s',
            label=f"M={M}",
        )
    if show_gap:
        plt.yscale('log')
        plt.xlabel(r'spectral gap 1-$\lambda_2$')
    else:
        plt.xlabel(r'$d$')
    plt.ylabel(r'inter-layer overlap $\langle q_{\mathrm{avg}}\rangle$')
    plt.ylim(0, 1.02)
    plt.title('b')
    plt.grid(True, ls=':')
    plt.legend()
    fig.suptitle(fr"$N={N0}$, $reps={reps}$, $K={K}$, $steps0={steps0}$, $L={L}$, run_SA={run_SA}")

    plt.tight_layout()
    plt.savefig("energy/energy.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # extra figure: min residual vs M
    min_idx = np.nanargmin(mean_res, axis=0)
    min_vals = mean_res[min_idx, np.arange(len(Ml))]
    min_errs = ci95_res[min_idx, np.arange(len(Ml))]
    min_sigmas = SigmaL[min_idx]

    plt.figure(figsize=(5, 4))
    plt.errorbar(Ml, np.clip(min_vals, np.finfo(float).tiny, None),
                 yerr=min_errs, marker='o')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('M')
    plt.ylabel(r'$\min_{\sigma}\ \langle e-e_0\rangle$')
    plt.title('Minimum residual vs M')
    plt.grid(True, which='both', ls=':')

    for x, y, s in zip(Ml, min_vals, min_sigmas):
        plt.annotate(fr'σ*={s:.2g}', (x, y), textcoords='offset points',
                     xytext=(5, 5), fontsize=8)

    tiny = np.finfo(float).tiny
    x = Ml.astype(float)
    y = np.clip(min_vals, tiny, None)
    yerr = np.clip(min_errs, tiny, None)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)

    w = 1.0 / np.clip((yerr[mask] / y[mask])**2, 1e-12, None)
    logx = np.log(x[mask])
    logy = np.log(y[mask])
    p, loga = np.polyfit(logx, logy, deg=1, w=w)
    a = np.exp(loga)

    M_fit = np.geomspace(x[mask].min(), x[mask].max(), 256)
    y_fit = a * M_fit**p
    plt.plot(M_fit, y_fit, linestyle=':', color='red', label=fr'fit slope={p:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("energy", "min_residual_vs_M.pdf"),
                format="pdf", bbox_inches="tight")
    plt.show()

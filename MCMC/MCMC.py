import numpy as np
import scipy.sparse as sp
from ising_mcmc_sparse import run_mcmc_seq_sparse
from sa_sparse_ising import run_mcmc_seq_sparse_anneal

import stats
import time
from tqdm import tqdm

def run_single_mcmc(J, J0, N, K, steps, theta, beta, seed):
    sigma = np.random.choice([-1, 1], size=(K, N)).astype(np.int8)
    J_csr = sp.csr_matrix(J, dtype=np.float64)

    sigma_final = run_mcmc_seq_sparse(
        J_csr.data,
        J_csr.indices.astype(np.int32, copy=False),
        J_csr.indptr.astype(np.int32, copy=False),
        N,
        sigma,
        theta,
        beta,
        seed,
        steps
    )

    return sigma_final


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


def run_mcmc_experiment(J0_list, aJ_list, N0, Ml, reps, theta, beta, m0, K, steps0, seed, run_SA):
    results = {}
    e0_r_list = []

    for r in tqdm(range(reps)):
        np.random.seed(seed + r)  # Slight change per rep
        results[r] = {}
        all_energies_r = []

        J0 = J0_list[r]
        aJ = aJ_list[r]

        for iM, M in enumerate(Ml):
            J = aJ[iM]
            N = N0 * M
            steps = steps0 * M

            t0 = time.time()
            if run_SA == False:
                sigma_final = run_single_mcmc(J, J0, N, K, steps, theta, beta, seed + r)
            else:
                sigma_final = run_single_mcmc_sa(J, J0, N, K, steps, theta, beta, seed + r)
            t1 = time.time()

            m_final = np.mean(sigma_final, axis=0)
            energy = stats.calculate_energy_replicas(sigma_final, J0.todense(), M)
            menergy = np.mean(energy)

            h = (J @ sigma_final.T + theta)
            E = -np.sum(sigma_final.T * h, axis=0) / N
            menergy_the = np.mean(E)

            ismin = stats.is_local_minimum(sigma_final, J)
            idx = np.where(ismin)[0]
            sigma_final_ = sigma_final[idx] / sigma_final[idx, 0][:, np.newaxis]
            unique_configs = np.unique(sigma_final_, axis=0)
            num_unique = unique_configs.shape[0]

            spins_reshaped = sigma_final.reshape(K, M, N0)
            nunique = np.array([np.unique(spins_reshaped[k_, :, :], axis=0).shape[0] for k_ in range(K)])
            p_all_same = np.mean(nunique == 1)

            results[r][M] = {
                'm_final': m_final,
                'energy': energy,
                'menergy': menergy,
                'menergy_the': menergy_the,
                'p_all_same': p_all_same,
                'is_local_min': np.mean(ismin),
                'num_unique_LM': num_unique,
                'compute_time': t1 - t0,
            }

            all_energies_r.extend(np.ravel(energy).tolist())

        e0_r = np.min(all_energies_r)
        e0_r_list.append(e0_r)

    return results, e0_r_list


import numpy as np
import matplotlib.pyplot as plt
from instance import create_Bethe
from mlayer import Mlayer, create_mixing_Q

if __name__ == "__main__":
        
    # --- Parameters
    N0 = 30
    k = 2
    #Ml = [1, 2, 4, 10, 50, 100]
    Ml = [1, 2, 4, 10, 20]
    reps = 10
    K = 50
    steps0 = 100
    seed = 42
    
    sigma = 0.1
    
    #not used
    beta = 1000
    m0 = 0.0
    theta = 0.0
    
    run_SA = True
    
    # --- Generate J0 and multilayer J
    J0_list, aJ_list = [], []
    for rep in range(reps):
        J0 = create_Bethe(N0, k + 1)
        J0_list.append(J0)
    
        Q_rep, aJ_rep = [], []
        for M in Ml:                       # loop over layer multipliers
            #Q = create_mixing_Q(M, mtype="gauss")
            Q = create_mixing_Q(M, mtype="block", L=2, sigma=sigma)
            Q_rep.append(Q)
            J_M = np.array(
                Mlayer(J0.todense(), M, Q, typeperm="asym").todense(),
                dtype=float,
            )
            aJ_rep.append(J_M)
            
        aJ_list.append(aJ_rep)
    
    # --- Run MCMC experiment
    mcmc_results, e0_r_list = run_mcmc_experiment(
        J0_list, aJ_list, N0=N0, Ml=Ml, reps=reps,
        theta=theta, beta=beta, m0=m0, K=K, steps0=steps0, seed=seed, run_SA=run_SA
    )
    
    
    
    
    # Initialize containers for plotting per-repetition curves
    de_per_r = np.zeros((reps, len(Ml)))
    energy_per_r = np.zeros((reps, len(Ml)))
    p0_per_r = np.zeros((reps, len(Ml)))
    
    for r in range(reps):
        e0_r = e0_r_list[r]
        for iM, M in enumerate(Ml):
            res = mcmc_results[r][M]
            en_array = np.array(res['energy']).flatten()
            energy_per_r[r, iM] = np.mean(en_array)
            de_per_r[r, iM] = np.mean(en_array - e0_r)
            p0_per_r[r, iM] = np.mean(en_array == e0_r)
    
    # --- Compute stats
    mean_energy = np.mean(energy_per_r, axis=0)
    std_energy = np.std(energy_per_r, axis=0)
    ci_energy = 1.96 * std_energy / np.sqrt(reps)
    
    mean_p0 = np.mean(p0_per_r, axis=0)
    std_p0 = np.std(p0_per_r, axis=0)
    ci_p0 = 1.96 * std_p0 / np.sqrt(reps)

    mean_de = np.mean(de_per_r, axis=0)
    std_de = np.std(de_per_r, axis=0)
    ci_de = 1.96 * std_de / np.sqrt(reps)

    # --- Plot
    # plt.figure(figsize=(14, 5))
    
    # # Energy vs M per rep
    # plt.subplot(1, 2, 1)
    # for r in range(reps):
    #     plt.plot(Ml, energy_per_r[r], '-o', alpha=0.4, label=f'rep {r}')
    # plt.errorbar(Ml, mean_energy, yerr=ci_energy, fmt='o-', color='black', linewidth=2, label='Mean ± CI')
    # plt.xlabel("M")
    # plt.ylabel("Energy")
    # plt.title("Energy vs M per repetition")
    # plt.ylim(-1.3,-0.9)
    # plt.xlim(0,100)
    # plt.grid(True)
    # plt.legend()
    
    # # p0 vs M per rep
    # plt.subplot(1, 2, 2)
    # for r in range(reps):
    #     plt.plot(Ml, p0_per_r[r], '--s', alpha=0.4, label=f'rep {r}')
    # plt.errorbar(Ml, mean_p0, yerr=ci_p0, fmt='s-', color='black', linewidth=2, label='Mean ± CI')
    # plt.xlabel("M")
    # plt.ylabel("P(E = e₀ᵣ)")
    # plt.title("Ground state recovery vs M")
    # plt.ylim(0,1.1)
    # plt.xlim(0,100)
    # plt.grid(True)
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
        
     
    # de vs M per rep
    plt.subplot(1, 1, 1)
    for r in range(reps):
        plt.plot(Ml, de_per_r[r], '--s', alpha=0.4, label=f'rep {r}')
    plt.errorbar(Ml, mean_de, yerr=ci_de, fmt='s-', color='black', linewidth=2, label='Mean ± CI')
    plt.xlabel("M")
    plt.ylabel("Residual energy de")
    plt.title("Residual energy vs M")
    plt.grid(True)
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    
    plt.show()
        
# ---------------------------------------------------------------------
#  Structured‑M‑layer zero‑T population dynamics (RS / 1‑RSB, B ≥ 1)
# We expect:
#   Bethe k-3,  RS:                 -1.2777 
#               1RSB (factorized):  -1.2723
#               1RSB (full):        −1.2717
#               Brute force:        -1.2720
# ---------------------------------------------------------------------
import numpy as np
from numba import njit, prange
from tqdm import tqdm
import os, time, pandas as pd
import matplotlib.pyplot as plt
import misc
#from RDE import step_dynamic_SP, run_PD
#from RDE_alt import step_dynamic_SP, run_PD      # reweight blocks independently
#from RDE_alt2 import step_dynamic_SP, run_PD      # makes more sense? P^e({h_b^e}_b)
#from RDE_alt2speed import step_dynamic_SP, run_PD 
#from RDE_fixed import step_dynamic_SP, run_PD     #not correct fields B=1 (to fix)
#from RDE_fixed2 import step_dynamic_SP, run_PD   #not correct fields B=1
#from RDE_fixed3 import step_dynamic_SP, run_PD    #correct field B=1!
#from RDE_fixed4 import step_dynamic_SP, run_PD    # testing new
#from RDE_fixed5 import step_dynamic_SP, run_PD    # WORKING
#from RDE_fixed6 import step_dynamic_SP, run_PD    # accelerated
#from RDE_fixed7 import step_dynamic_SP, run_PD    # NEW: correct half edge reweighting
#from RDE_fixed9 import step_dynamic_SP, run_PD    # After fix of formalism
from RDE_fixed10 import step_dynamic_SP, run_PD    # fixed again

import G90

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MCMC')))
from mlayer import create_mixing_Q

from multiprocessing import Pool, cpu_count

def worker(args):
    iy, y, n_pop, n_iter, Q, B, deg = args
    # Run simulation
    #h_pop, hat_pop = run_PD(y, n_pop, n_iter, Q, B, deg)
    h_pop, hat_pop = run_PD(y, Q, B, deg, n_pop, n_iter)
    # Compute evaluations
    eval_result = phi_RSB(y, h_pop, deg, Q)
    return iy, eval_result


@njit
def channel_zeroT_uE(h_vec, J, Q):
    """Zero-T channel: given block field h (shape: B,), coupling J, and mixing Q (B,B),
    return u_vec (half-fields, shape B) and E_vec (channel energies, shape B)."""
    B = h_vec.shape[0]
    u = np.zeros(B)
    E = np.zeros(B)
    # H = Q @ h
    H = np.zeros(B)
    for b in range(B):
        s = 0.0
        for c in range(B):
            s += Q[b, c] * h_vec[c]
        H[b] = s
    for b in range(B):
        ap = np.abs(J + H[b])
        am = np.abs(J - H[b])
        u[b] = 0.5 * (ap - am)
        E[b] = 0.5 * (ap + am)
    return u, E

@njit
def link_energy_zeroT(h_i, h_j, J, Q):
    """ΔE^(2) for one link (block version).
       Return e_link = - max_{σ_i,σ_j, b,b' with Q[b,b']>0} [ h_i[b] σ_i + h_j[b'] σ_j + J σ_i σ_j ]."""
    B = h_i.shape[0]
    best = -1.0e300  # we maximize 'val'
    for b in range(B):
        for bp in range(B):
            if Q[b, bp] <= 0.0:
                continue
            hi = h_i[b]
            hj = h_j[bp]
            # enumerate σ_i, σ_j ∈ {±1}
            # four possibilities; inline max to keep Numba happy
            v1 =  hi + hj + J     # σ_i=+1, σ_j=+1
            v2 =  hi - hj - J     # σ_i=+1, σ_j=-1
            v3 = -hi + hj - J     # σ_i=-1, σ_j=+1
            v4 = -hi - hj + J     # σ_i=-1, σ_j=-1
            # take the maximum
            if v1 > best: best = v1
            if v2 > best: best = v2
            if v3 > best: best = v3
            if v4 > best: best = v4
    return -best  # ΔE^(2)

@njit
def log_mean_exp(logw):
    """Numerically stable log-mean-exp over 1D array."""
    m = logw[0]
    n = logw.shape[0]
    for i in range(1, n):
        if logw[i] > m:
            m = logw[i]
    s = 0.0
    for i in range(n):
        s += np.exp(logw[i] - m)
    return m + np.log(s / n)

@njit
def phi_RSB(y, h_pop, deg, Q):
    """
    Zero-T 1-RSB potential Φ(y) for a D=deg regular graph (block mixing Q).
    Uses the site/link log-moment functionals; no half-edge term.
    """
    n  = h_pop.shape[0]
    B  = h_pop.shape[1]
    c  = deg  # graph degree

    # Monte-Carlo batch size: use n samples by default
    Nmc = n

    # --- site term: ΔE^(1) = - sum_k sum_b E_b - sum_b | sum_k u_b |
    logw_site = np.zeros(Nmc)
    for k in range(Nmc):
        # draw c i.i.d. neighbors
        sumE = 0.0
        h_out = np.zeros(B)
        for _ in range(c):
            idx = np.random.randint(0, n)
            J   = 1.0 if (np.random.rand() > 0.5) else -1.0
            u_vec, E_vec = channel_zeroT_uE(h_pop[idx], J, Q)
            # accumulate
            for b in range(B):
                sumE   += E_vec[b]
                h_out[b] += u_vec[b]
        # ΔE^(1)
        sabs = 0.0
        for b in range(B):
            sabs += np.abs(h_out[b])
        e_site = - (sumE + sabs)
        #logw_site[k] = -y * e_site
        #logw_site[k] = -y * e_site / B
        logw_site[k] = -y * e_site / B #TESTING (!)

    # --- link term: ΔE^(2) for a single link
    logw_link = np.zeros(Nmc)
    for k in range(Nmc):
        i  = np.random.randint(0, n)
        j  = np.random.randint(0, n)
        J  = 1.0 if (np.random.rand() > 0.5) else -1.0
        e_link = link_energy_zeroT(h_pop[i], h_pop[j], J, Q)
        logw_link[k] = -y * e_link

    # assemble Φ(y)
    phi = -(1.0 / y) * log_mean_exp(logw_site) + (c / (2.0 * y)) * log_mean_exp(logw_link)
    return phi



# ----------------------------------------------------------------------
#  demo / benchmark
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- graph parameters ----------
    k      = 2                 # c‑regular with c = k+1
    deg    = k + 1
    
    # ---------- population‑dynamics ----------
    if True: #standard (13h)
        n_pop  = 80_000    # > 300_000
        n_iter = 300# 100         # > 30            
        Ny     = np.linspace(0.01, 0.75, 15)
        rep    = 2000
        #B_list = [1,2,4,6,8,10]
        B_list = [1,4,10]
        #B_list = [2]
        
    else: #short
        n_pop  = 30_000    # > 300_000
        n_iter = 50# 100         # > 30            
        Ny     = np.linspace(0.01, 1.0, 15)
        rep    = 500
        B_list = [1,4,10]
        
    L = 2
    
    # ---------- choose the block sizes to compare ----------
    colors = plt.cm.viridis(np.linspace(0, 1, len(B_list)))  # 1 colour per B
    
    # ---------- storage ----------
    mean_E_all, err_E_all = [], []
    mean_S_all, err_S_all = [], []
    
    t0 = time.time()
    for B in B_list:
        
        # ----- build Gaussian kernel Q for this B -----
        #σ = 1.55 #0.1: B=10 ressemble B=1
        σ = 1.0
        Q   = create_mixing_Q(B, mtype="block", sigma=σ, L=L)
        #Q = np.ones((B,B))/B
        if B>1:
            λ2  = np.sort(np.real(np.linalg.eigvals(Q)))[-2]
            print(f"B={B:2d}  lambda2 {λ2}:, sigma {σ}")
        else:
            print(f"B={B:2d}")
    
        # ----- containers for this B -----
        Evals = np.zeros((len(Ny), rep))
     
        tasks = [(iy, Ny[iy], n_pop, n_iter, Q, B, deg) for iy in range(len(Ny)) for _ in range(rep)]

        # Run parallel computations
        with Pool(processes=min(len(tasks), cpu_count())) as pool:
            results = pool.map(worker, tasks)
    
        # Collect results
        k = 0
        for iy, eval_result in results:
            r = k % rep  # Determine repetition index
            iy = k // rep  # Determine y index
            Evals[iy, r] = eval_result
            k += 1
        
        mean_E = Evals.mean(axis=1); err_E = Evals.std(axis=1)*1.96/np.sqrt(rep)
    
        mean_E_all.append(mean_E); err_E_all.append(err_E)
        
        if True:
            misc.save_plot_data(B, Ny, mean_E, err_E, mean_E, err_E, out_dir="res")
    
    print(f"total wall time: {time.time()-t0:.1f}s")
    
    # --- measure the collapse between blocks ---
    #if B>1:
    #    nbdiff = np.sum(h_pop[:,0]-h_pop[:,1])
    #    print(f"difference between blocks: {nbdiff}")
    
    # ---------- save data ----------
    if False:
        for c, B, mE, eE, mS, eS in zip(colors, B_list,
                                        mean_E_all, err_E_all,
                                        mean_S_all, err_S_all):
            misc.save_plot_data(B, Ny, mE, eE, mS, eS, out_dir="res")
            
            
    folder_name = "complexity_B2"
    os.makedirs(folder_name, exist_ok=True)
    
    # ---------- Quick plots (3 panels: phi, e, Sigma) ----------
    fig = plt.figure(figsize=(14, 6))
    
    # Grid layout: left column: phi & epsilon (stacked), right column: Sigma
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.45)
    ax_phi = fig.add_subplot(gs[0, 0])
    ax_e   = fig.add_subplot(gs[1, 0])
    ax_S   = fig.add_subplot(gs[:, 1])  # right panel spans both rows
    
    fontsize = 14
    markers = ['o', 's', '^', 'v', 'D']  # for different B
    colors  = plt.cm.viridis(np.linspace(0, 1, len(B_list)))
    
    # ---------- M-layer cavity theory ----------
    phi_the = np.array([G90.phi_G90(y) for y in Ny])
    mu_phi  = phi_the * Ny
    epsilon_the = np.gradient(mu_phi, Ny)
    Sigma_the   = Ny**2 * np.gradient(phi_the, Ny)
    
    # Theory lines
    ax_phi.plot(Ny, phi_the, 'k-', lw=2, label='[MP03]')
    ax_e.plot(Ny, epsilon_the, 'k-', lw=2, label='[MP03]')
    ax_S.plot(epsilon_the, Sigma_the, 'k-', lw=2, label='[MP03]')
    
    # ---------- Simulation data ----------
    data_dict = {}
    for i, (B, c) in enumerate(zip(B_list, colors)):
        marker = markers[i % len(markers)]
        label = f"B={B}"
    
        NyB, mE, eE, mS, eS = misc.load_plot_data(B, directory="res")
        data_dict[B] = (NyB, mE)
    
        # Top-left: phi
        ax_phi.errorbar(NyB, mE, eE, marker=marker, linestyle='', color=c, markersize=6, label=label)
    
        # Bottom-left: epsilon
        epsilon = np.gradient(NyB * mE, NyB)
        ax_e.plot(NyB, epsilon, marker=marker, linestyle='', color=c, markersize=6)
    
        # Right: Sigma
        Sigma = NyB**2 * np.gradient(mE, NyB)
        ax_S.plot(epsilon, Sigma, marker=marker, linestyle='', color=c, markersize=6)
    
    # ---------- Connect B points with line (same color as marker) ----------
    selected_Bs = [1, 4]  # example
    for B in selected_Bs:
        NyB, mE = data_dict[B]
        epsilon = np.gradient(NyB * mE, NyB)
        Sigma = NyB**2 * np.gradient(mE, NyB)
        
        # Top-left: phi line
        ax_phi.plot(NyB, mE, '-', lw=1.5, color=colors[B_list.index(B)])
        # Bottom-left: epsilon line
        ax_e.plot(NyB, epsilon, '-', lw=1.5, color=colors[B_list.index(B)])
        # Right: Sigma line
        ax_S.plot(epsilon, Sigma, '-', lw=1.5, color=colors[B_list.index(B)])
    
    # ---------- Reference lines ----------
    ref_lines = dict(eRSB=-1.2717, eRSBfac=-1.2723, ebf=-1.2719, ebf2=-1.2720)
    for ax in [ax_phi, ax_e]:
        ax.axhline(ref_lines['eRSB'], linestyle='--', color='b', alpha=0.5, label='[RSB]')
        ax.axhline(ref_lines['eRSBfac'], linestyle='-', color='k', alpha=0.5, label='[RSB-fac]')
        ax.axhline(ref_lines['ebf'], linestyle=':', color='m', alpha=0.5, label='[BF]')
        ax.axhline(ref_lines['ebf2'], linestyle='-.', color='r', alpha=0.5, label='[BF2]')
    
    for ax in [ax_S]:
        ax.axvline(ref_lines['eRSB'], linestyle='--', color='b', alpha=0.5)
        ax.axvline(ref_lines['eRSBfac'], linestyle='-', color='k', alpha=0.5)
        ax.axvline(ref_lines['ebf'], linestyle=':', color='m', alpha=0.5)
        ax.axvline(ref_lines['ebf2'], linestyle='-.', color='r', alpha=0.5)
    
    # ---------- Axes limits ----------
    ax_phi.set_xlim(np.min(Ny), np.max(Ny))
    ax_phi.set_ylim(-1.278, -1.27)
    ax_e.set_xlim(np.min(Ny), np.max(Ny))
    ax_e.set_ylim(-1.278, -1.27)
    ax_S.set_xlim(-1.278, -1.27)
    ax_S.set_ylim(0.0, 0.0008)
    
    # ---------- Labels & titles ----------
    ax_phi.set_xlabel("Parisi parameter y", fontsize=fontsize)
    ax_phi.set_ylabel("Free energy φ", fontsize=fontsize)
    ax_phi.set_title("a", fontsize=fontsize)
    ax_e.set_xlabel("Parisi parameter y", fontsize=fontsize)
    ax_e.set_ylabel("Energy-per-site e", fontsize=fontsize)
    ax_e.set_title("b", fontsize=fontsize)
    ax_S.set_xlabel("Energy-per-site e", fontsize=fontsize)
    ax_S.set_ylabel("complexity Σ", fontsize=fontsize)
    ax_S.set_title("c", fontsize=fontsize)
    
    for ax in [ax_phi, ax_e, ax_S]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    # ---------- Legends ----------
    for ax in [ax_phi, ax_e, ax_S]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict()
        for h, l in zip(handles, labels):
            if l not in by_label:
                by_label[l] = h
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best', ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.savefig("complexity_B2/complexity_B2temp.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # ---------- Save all data ----------
    
    for i, (B, mE, eE) in enumerate(zip(B_list, mean_E_all, err_E_all)):
        
        filename = f"complexity_B{B}_n{n_pop}_iter{n_iter}_rep{rep}.npz"
        file_path = os.path.join(folder_name, filename)
        
        np.savez(file_path,
                 B=B,
                 Ny=Ny,
                 mean_E=mE,
                 err_E=eE,
                 # Optionally precompute epsilon & Sigma
                 epsilon=np.gradient(Ny * mE, Ny),
                 Sigma=Ny**2 * np.gradient(mE, Ny),
                 # Save theory only once for the first B
                 phi_the=phi_the if i == 0 else None,
                 epsilon_the=epsilon_the if i == 0 else None,
                 Sigma_the=Sigma_the if i == 0 else None,
                 B_list=np.array(B_list))

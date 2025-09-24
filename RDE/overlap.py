# ---------------------------------------------------------------------
#  Structured‑M‑layer zero‑T population dynamics (RS / 1‑RSB, B ≥ 1)
# ---------------------------------------------------------------------
import numpy as np
from numba import njit, prange
from tqdm import tqdm
import os, time, pandas as pd
import matplotlib.pyplot as plt
import misc
#from RDE_fixed5 import step_dynamic_SP, run_PD    # WORKING
#from RDE_fixed6 import step_dynamic_SP, run_PD    # accelerated
#from RDE_fixed7 import step_dynamic_SP, run_PD    # new
from RDE_fixed10 import step_dynamic_SP, run_PD    # fixed again

from complexity_B2 import phi_RSB, channel_zeroT_uE, link_energy_zeroT, log_mean_exp

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MCMC')))
from mlayer import create_mixing_Q

import G90
import os

import contraction

# ---------- kernel ----------------------------------------------------
def get_gaussian_Q(B, sigma):
    idx = np.arange(B, dtype=np.float64)
    bb, cc = np.meshgrid(idx, idx)
    g = np.exp(-(bb - cc)**2 / (2 * sigma**2)).astype(np.float64)
    return g / g.sum(axis=1, keepdims=True)

# ---------- observable ------------------------------------------------

@njit
def spin_overlap(hat_pop, deg, B, n_nodes):
    """
    Monte-Carlo estimator of  q_spin = <σ_ib σ_ic>_{b<c}.
      hat_pop : (n_edges, B)  array of propagated fields
      deg     : degree  (k+1 in your notation)
      B       : number of blocks
      n_nodes : how many synthetic "nodes" you draw
    """
    n_edges = hat_pop.shape[0]
    corr, cnt = 0.0, 0

    for _ in range(n_nodes):
        # ----- build one synthetic node ---------------------
        field_sum = np.zeros(B, dtype=np.float64)
        for _ in range(deg):                        # draw deg in-messages
            idx = np.random.randint(0, n_edges)
            field_sum += hat_pop[idx]

        # ----- convert to spin sign  (+1 / –1, random tie-break) ------
        spin = np.empty(B, dtype=np.float64)
        for b in range(B):
            #if field_sum[b] == 0.:                  # tie-break
            #    spin[b] = 1. if np.random.rand() < 0.5 else -1.
            #else:
            #    spin[b] = np.sign(field_sum[b])
            spin[b] = np.sign(field_sum[b])

        # ----- accumulate pairwise products ----------------
        #for b in range(B):
        #    for c in range(b+1, B):
        #        corr += spin[b] * spin[c]
        #        cnt  += 1
        
        # accumulate pairwise "XNOR" contributions
        for b in range(B):
            sb = spin[b]
            for c in range(b+1, B):
                if sb == spin[c]:
                    corr += 1.0
                elif sb == -spin[c]: 
                    corr += -1.0
                else:
                    corr += 0.0
                cnt += 1

    return corr / cnt      # average over blocks & synthetic nodes

# ----------------------------------------------------------------------
#  demo:  q_block  versus  spectral gap   for several B
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- PD / graph parameters ----------
    k        = 2                  # excess degree  (c = k+1)
    deg      = k + 1
    n_pop    = 30_000
    n_iter   = 50
    reps     = 1                  # repeats per σ (increase for error bars)
    y_fix    = 0.01                # chosen Parisi parameter  (y = 0.4174 for GS)
    
    show_gap = False

    # ---------- σ grid (same for every B) -----------------------------
    #SigmaL = np.exp(np.linspace(np.log(0.1), np.log(0.95), 20))
    SigmaL = np.exp(np.linspace(np.log(0.2), np.log(2.0), 20))

    # ---------- block sizes to compare --------------------------------
    L      = 30                  
    B_list = L*np.array([2,5,10])  
    cmap   = plt.cm.viridis(np.linspace(0, 1, len(B_list)))

    # ---------- main loop ---------------------------------------------
    results = {}                           #  key = B  → (gaps, q_vals)

    t0 = time.time()
    for B in B_list:                       # -------------- COMPUTE ----------
        gaps, q_vals = [], []
        p_vals, pth_vals = [], []
        e_vals = []
        
        rho0_mean_vals, rho0_q_vals = [], []
        alpha_mean_vals, alpha_q_vals = [], []
        lam2_vals = []
    
        for sigma in tqdm(SigmaL, desc=f"σ‑sweep (B={B})", leave=False):
            #Q   = create_mixing_Q(B, mtype="block", sigma=sigma, L=1, eps=1e-3)
            Q   = create_mixing_Q(B, mtype="block", sigma=sigma, L=L)
    
            # spectral gap 1‑λ₂(Q)
            if B>1:
                λ2  = np.sort(np.real(np.linalg.eigvals(Q)))[-2]
            else:
                λ2 = 0
            gaps.append(1.0 - λ2)
    
            # population dynamics
            h_pop, hat_pop = run_PD(y_fix, Q, B, deg, n_pop, n_iter)
            
            eval_result = phi_RSB(y_fix, h_pop, deg, Q)
            e_vals.append(eval_result)
            
            q_vals.append(spin_overlap(hat_pop, deg=deg, B=B,
                                       n_nodes=min(n_pop, 5000)))
            
            #p0 = np.mean(h_pop==0)
            #p1 = np.mean(h_pop==1)
            #p2 = np.mean(h_pop==2)
            bins = np.linspace(-2.5, 2.5, 6)
            values, bin_edges = np.histogram(h_pop, bins=bins)
            normalized_values = values / np.sum(values)
            p0 = normalized_values[2]; p1 = normalized_values[3]; p2 = normalized_values[4]; 
            p_vals.append([p0,p1,p2])
            
            p0_, p1_, p2_, q = G90.compute_values(y_fix)
            pth_vals.append([p0_,p1_,p2_])
            
            # --- zero‑T contraction metric for this σ ---
            # λ2 we already computed above as 'λ2'
            lam2_vals.append(λ2)
            
            # activity α from half‑fields: unsaturated channels are |u_b| < |J| (±J → |J|=1)
            Jabs = 1.0
            tol = 1e-12
            mask = (np.abs(hat_pop) < (Jabs - tol))     # shape (n_edges, B)
            
            alpha_mean = mask.mean()
            alpha_q    = np.quantile(mask.mean(axis=1), 0.995)   # conservative 99.5% quantile over edges
            
            rho0_mean  = (deg - 1) * alpha_mean * λ2
            rho0_q     = (deg - 1) * alpha_q    * λ2
            
            alpha_mean_vals.append(alpha_mean)
            alpha_q_vals.append(alpha_q)
            rho0_mean_vals.append(rho0_mean)
            rho0_q_vals.append(rho0_q)
            
        # store as NumPy arrays for later plotting
        results[B] = (np.asarray(gaps), np.asarray(q_vals), np.asarray(e_vals), np.asarray(p_vals), np.asarray(pth_vals))
    
        # post process to find the treshold
        rho0_mean_vals = np.asarray(rho0_mean_vals)
        rho0_q_vals    = np.asarray(rho0_q_vals)
        alpha_mean_vals = np.asarray(alpha_mean_vals)
        alpha_q_vals    = np.asarray(alpha_q_vals)
        lam2_vals       = np.asarray(lam2_vals)
        
        sigma_star_mean = contraction.first_crossing_sigma(SigmaL, rho0_mean_vals, target=1.0)
        sigma_star_q    = contraction.first_crossing_sigma(SigmaL, rho0_q_vals,    target=1.0)
        
        print(f"[B={B}]  σ* (mean-α) = {sigma_star_mean},   σ* (q=99.5% α) = {sigma_star_q}")
        
        # Keep your original 'results' tuple unchanged, and store contraction arrays separately:
        if "thr" not in locals():
            thr = {}
        thr[B] = {
            "sigma_star_mean": sigma_star_mean,
            "sigma_star_q": sigma_star_q,
            "rho0_mean": rho0_mean_vals,
            "rho0_q": rho0_q_vals,
            "alpha_mean": alpha_mean_vals,
            "alpha_q": alpha_q_vals,
            "lambda2": lam2_vals,
        }
    
    print(f"total compute wall‑time: {time.time()-t0:.1f} s")
        
    # --------- plot ---------------------------------------------------
    fig, (ax_e, ax_b) = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)

    for colour, B in zip(cmap, B_list):
        gaps, q_vals, e_vals, p_vals, pth_vals = results[B]
    
        # plot data
        if show_gap:
            ax_b.plot(gaps, q_vals, "o-", color=colour, label=f"B={B}")
            ax_e.plot(gaps, e_vals, "o-", color=colour, label=f"B={B}")
        else:
            ax_b.plot(SigmaL, q_vals, "o-", color=colour, label=f"B={B}")
            ax_e.plot(SigmaL, e_vals, "o-", color=colour, label=f"B={B}")
            eRS = -1.2777
            eRSBfac = -1.2723
            ax_e.plot(SigmaL, np.ones(len(SigmaL))*eRSBfac, ':r', alpha=0.5)
            ax_e.plot(SigmaL, np.ones(len(SigmaL))*eRS, ':m', alpha=0.5)
    
        # add vertical dotted line at threshold σ*
        if 'thr' in locals() and B in thr:
            s_star = thr[B]["sigma_star_mean"]   # or "sigma_star_mean"
            if s_star is not None:
                if show_gap:
                    # convert σ* to gap = 1-λ2(Q(σ*))
                    g_star = 1.0 - thr[B]["lambda2"][np.argmin(np.abs(SigmaL-s_star))]
                    ax_b.axvline(g_star, ls=":", color=colour, alpha=0.8)
                    ax_e.axvline(g_star, ls=":", color=colour, alpha=0.8)
                else:
                    ax_b.axvline(s_star, ls=":", color=colour, alpha=0.8)
                    ax_e.axvline(s_star, ls=":", color=colour, alpha=0.8)
    
    # cosmetics
    if show_gap:
        ax_e.set_xscale("log")
        ax_e.set_xlabel(r"spectral gap  $(1-\lambda_2(Q))$")
        ax_b.set_xscale("log")
        ax_b.set_xlabel(r"spectral gap  $(1-\lambda_2(Q))$")
    else:
        ax_e.set_xlabel(r"$\sigma$")
        ax_b.set_xlabel(r"$\sigma$")
    ax_e.set_ylabel(r"energy  $e$")
    ax_e.set_ylim(-1.3, -0.7)
    ax_e.grid(alpha=.3)
    ax_e.set_title(r"(a)  energy")
    ax_b.set_ylabel(r"block–block overlap  $q_{\mathrm{block}}$")
    ax_b.set_ylim(0, 1.02)
    ax_b.grid(alpha=.3)
    ax_b.set_title(r"(b)  overlap  $q$")
    ax_b.legend(title=r"number of blocks $B$")
    
    fig.suptitle(fr"$y={y_fix}$   ($n_\mathrm{{pop}}={n_pop}$, {n_iter} sweeps)")

    
    
    # ----- cosmetics: right panel (b) -----
    
    if show_gap:
        ax_b.set_xscale("log")
        ax_b.set_xlabel(r"spectral gap  $(1-\lambda_2(Q))$")
    else:
        ax_b.set_xlabel(r"$\sigma$")
    ax_b.set_ylabel(r"block–block overlap  $q_{\mathrm{block}}$")
    ax_b.set_ylim(0, 1.02)
    ax_b.grid(alpha=.3)
    ax_b.set_title(fr"(b)  overlap  $q$")
    
    # global/overall title
    fig.suptitle(fr"$y={y_fix}$   ($n_\mathrm{{pop}}={n_pop}$, {n_iter} sweeps)")
    
    plt.savefig("./overlap/overlap_energy.pdf")
    plt.show()

    
    # --------- plot ---------------------------------------------------
    for colour, B in zip(cmap, B_list):
        gaps, q_vals, e_vals, p_vals, pth_vals = results[B]          # retrieve
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
    
        for k in range(3):
            if show_gap:
                cp = ax.plot(gaps, p_vals[:,k], label=f'p{k}')
                ax.plot(gaps, pth_vals[:,k], '--', color=cp[0].get_color())
            else:
                cp = ax.plot(SigmaL, p_vals[:,k], label=f'p{k}')
                ax.plot(SigmaL, pth_vals[:,k], '--', color=cp[0].get_color())
        if show_gap:
            ax.set_xscale("log")
            ax.set_xlabel(r"spectral gap  $(1-\lambda_2(Q))$")
        else:
            ax.set_xlabel(r"$\sigma$")
            ax.set_ylabel(r"cavity field distribution $p_k$")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3)
        ax.set_title(fr"$B={B}$, $y={y_fix}$   ($n_\mathrm{{pop}}={n_pop}$, {n_iter} sweeps)")
        ax.legend()
        fig.tight_layout()
        plt.savefig("distribution.pdf")
        plt.show()
    
    
    folder_name = "overlap"
    os.makedirs(folder_name, exist_ok=True)
    filename = f"overlap_y{y_fix}_steps{n_iter}_n{n_pop}_L{L}.npz"
    file_path = os.path.join(folder_name, filename)
    np.savez(file_path, SigmaL=SigmaL, results=results, thr=thr)

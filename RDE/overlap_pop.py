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
#from RDE_fixed7 import step_dynamic_SP, run_PD    # NEW: correct half edge reweighting
#from RDE_fixed8 import step_dynamic_SP, run_PD    # non-singular mixing
#from RDE_fixed9 import step_dynamic_SP, run_PD    # After fix of formalism
from RDE_fixed10 import step_dynamic_SP, run_PD    # fixed again

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MCMC')))
from mlayer import create_mixing_Q

import G90

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
    #n_pop    = 100_000
    #n_popL   = [10_000,30_000,50_000,100_000,200_000,400_000]
    n_popL   = [10_000,30_000,50_000]
    n_iter   = 50
    reps     = 5                  # repeats per σ (increase for error bars)
    y_fix    = 0.05                # chosen Parisi parameter  (y = 0.4174 for GS)

    show_gap = False

    # ---------- σ grid (same for every B) -----------------------------
    #SigmaL = np.exp(np.linspace(np.log(0.1), np.log(0.95), 20))
    #SigmaL = np.exp(np.linspace(np.log(0.2), np.log(0.9), 20))
    sigma = 1.55
    
    # ---------- block sizes to compare --------------------------------
    #B_list = [2,4,8,10,12]                      
    B_list = [4,10]      
    L      = 2                 
    cmap   = plt.cm.viridis(np.linspace(0, 1, len(B_list)))

    # ---------- main loop ---------------------------------------------
    results = {}                           #  key = B  → (gaps, q_vals)

    t0 = time.time()
    for B in B_list:                       # -------------- COMPUTE ----------
        gaps, q_vals = [], []
        p_vals, pth_vals = [], []
        
        #for sigma in tqdm(SigmaL, desc=f"σ‑sweep (B={B})", leave=False):
        for n_pop in tqdm(n_popL, desc=f"n_pop‑sweep (B={B})", leave=False):
            #Q   = create_mixing_Q(B, mtype="block", sigma=sigma, L=1, eps=1e-3)
            Q   = create_mixing_Q(B*L, mtype="block", sigma=sigma, L=L)
            #Q = np.eye(B,B)
            
            # spectral gap 1‑λ₂(Q)
            if B>1:
                λ2  = np.sort(np.real(np.linalg.eigvals(Q)))[-2]
            else:
                λ2 = 1
            gaps.append(1.0 - λ2)
    
            gaps_, q_vals_ = [], []
            p_vals_, pth_vals_ = [], []
            
            for r in range(reps):
                # population dynamics
                h_pop, hat_pop = run_PD(y_fix, Q, B, deg, n_pop, n_iter)
                
                if B>1:
                    q_vals_.append(spin_overlap(hat_pop, deg=deg, B=B,
                                               n_nodes=min(n_pop, 5000)))
                else:
                    q_vals_.append(1.0)
                
                #p0 = np.mean(h_pop==0)
                #p1 = np.mean(h_pop==1)
                #p2 = np.mean(h_pop==2)
                bins = np.linspace(-2.5, 2.5, 6)
                values, bin_edges = np.histogram(h_pop, bins=bins)
                normalized_values = values / np.sum(values)
                p0 = normalized_values[2]; p1 = normalized_values[3]; p2 = normalized_values[4]
                p_vals_.append([p0,p1,p2])
                
                p0_, p1_, p2_, q = G90.compute_values(y_fix)
                pth_vals_.append([p0_,p1_,p2_])
                
            
            q_vals.append(q_vals_)
            pth_vals.append(pth_vals_)
            p_vals.append(p_vals_)
            
    
        # store as NumPy arrays for later plotting
        results[B] = (np.asarray(gaps), np.asarray(q_vals), np.asarray(p_vals), np.asarray(pth_vals))
    
    print(f"total compute wall‑time: {time.time()-t0:.1f} s")
        
    # --------- plot ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.5))

    for colour, B in zip(cmap, B_list):
        gaps, q_vals, p_vals, pth_vals = results[B]          # retrieve
        ax.errorbar(n_popL, np.mean(q_vals,1), np.std(q_vals,1)*1.96/np.sqrt(reps), marker="o", color=colour, label=f"B={B}")
    
    #ax.set_xscale("log")
    #ax.set_xlabel(r"spectral gap  $(1-\lambda_2(Q))$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"block–block overlap  $q_{\mathrm{block}}$")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.set_title(fr"$y={y_fix}$   ($n_\mathrm{{pop}}={n_pop}$, {n_iter} sweeps)")
    ax.legend(title=r"number of blocks $B$")
    fig.tight_layout()
    plt.savefig("overlap.pdf")
    plt.show()
    
    
    # --------- plot ---------------------------------------------------
    for colour, B in zip(cmap, B_list):
        gaps, q_vals, p_vals, pth_vals = results[B]          # retrieve
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
    
        for k in range(3):
            cp = ax.errorbar(n_popL, np.mean(p_vals[:,:,k],1), np.std(p_vals[:,:,k],1)*1.96/np.sqrt(reps), label=f'p{k}')
            ax.errorbar(n_popL, np.mean(pth_vals[:,:,k],1), np.std(pth_vals[:,:,k],1)*1.96/np.sqrt(reps), linestyle='--', color=cp[0].get_color())
        #ax.set_xscale("log")
        #ax.set_xlabel(r"spectral gap  $(1-\lambda_2(Q))$")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"cavity field distribution $p_k$")
        ax.set_ylim(0, 0.4)
        ax.grid(alpha=.3)
        ax.set_title(fr"$B={B}$, $y={y_fix}$   ($n_\mathrm{{pop}}={n_pop}$, {n_iter} sweeps)")
        ax.legend()
        fig.tight_layout()
        plt.savefig("distribution.pdf")
        plt.show()
    

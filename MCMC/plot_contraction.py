import numpy as np
import matplotlib.pyplot as plt

# ---------------- Load data ----------------
# Theory data
#theory_file = "./../RDE/overlap/overlap_y0.11_steps500_n30000_L2.npz"  # 
#theory_file = "./../RDE/overlap/overlap_y0.01_steps50_n30000_L30.npz"  # 
#theory_file = "./../RDE/overlap/overlap_y0.11_steps50_n30000_L30.npz"  # 
#theory_file = "./../RDE/overlap/overlap_y0.21_steps50_n30000_L30.npz"  # 
theory_file = "./../RDE/overlap/overlap_y0.31_steps50_n30000_L30.npz"  # 
theory_data = np.load(theory_file, allow_pickle=True)
SigmaL_th = theory_data['SigmaL']
results = theory_data['results'].item()  # results is a dict
thr = theory_data['thr'].item()  # results is a dict

name = theory_file.split("/")[-1]             # get basename
parts = name.replace(".npz","").split("_")    # split by underscores
L_part = [p for p in parts if p.startswith("L")][0]
L = int(L_part[1:])

# MCMC data
#mcmc_file = "./../MCMC/threshold/threshold_N30_reps5_K30_steps50_L2.npz"  # 
#mcmc_file = "./../MCMC/threshold/threshold_N100_reps5_K30_steps200_L10.npz"  # SA
#mcmc_file = "./../MCMC/threshold/threshold_N30_reps5_K30_steps200_L10.npz"  # <-- replace with your file
mcmc_file = "./../MCMC/threshold/threshold_N30_reps5_K30_steps200_L30.npz"  # <-- replace with your file
mcmc_data = np.load(mcmc_file, allow_pickle=True)
SigmaL_mc = mcmc_data['SigmaL']
mean_res = mcmc_data['mean_res']  # shape (len(SigmaL), len(Ml))
ci95_res = mcmc_data['ci95_res']
mean_qavg = mcmc_data['mean_qavg']  # shape (len(SigmaL), len(Ml))
ci95_qavg = mcmc_data['ci95_qavg']
Ml = mean_res.shape[1]  # number of MCMC curves

from matplotlib.lines import Line2D

# --- figure ---
fig, ax = plt.subplots(figsize=(7, 4))

# --- style helpers ---
cmap    = plt.get_cmap('tab10')
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '>']  # extend if needed

# Gather displayed B values (Bl = B/L) in consistent order with 'results'
B_list = list(results.keys())
Bl = [int(B // L) for B in B_list]  # displayed block count per group

# --- plot THEORY curves (solid) + threshold σ* (dotted vertical) ---
for i, B in enumerate(B_list):
    col = cmap(i % 10)
    mk  = markers[i % len(markers)]
    gaps, q_vals, e_vals, p_vals, pth_vals = results[B]

    # Theory: solid line with marker
    ax.plot(SigmaL_th, q_vals,
            linestyle='-',
            marker=mk,
            color=col,
            label=f'Theory B={int(B//L)}')

    # Threshold vertical line (if available)
    if 'thr' in locals() and B in thr:
        # pick which threshold to show ("sigma_star_q" or "sigma_star_mean")
        sigma_star = thr[B].get("sigma_star_mean", None)
        if sigma_star is not None:
            ax.axvline(sigma_star, ls=':', color=col, alpha=0.85)

# --- plot MCMC curves (dashed) ---
# We assume there is one MCMC series per displayed B (same ordering).
for iM, B_disp in enumerate(Bl[:Ml]):  # guard if Ml < len(Bl)
    col = cmap(iM % 10)
    mk  = markers[iM % len(markers)]
    ax.errorbar(SigmaL_mc,
                mean_qavg[:, iM],
                yerr=ci95_qavg[:, iM],
                linestyle='--',
                marker=mk,
                color=col,
                capsize=3,
                label=f'MCMC B={B_disp}')

# --- axes, labels, limits ---
ax.set_xlabel(r'Standard deviation of mixing kernel $\sigma$')
ax.set_ylabel(r'Block-to-block overlap $\langle q\rangle$')
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)

# --- build combined legend (markers for B, line styles for source, dotted for σ*) ---
legend_elements = []

# Marker/color entries for each displayed B (no lines)
# (Use the same colors/markers as plotted.)
for i, B_disp in enumerate(Bl):
    col = cmap(i % 10)
    mk  = markers[i % len(markers)]
    legend_elements.append(Line2D([0], [0], color=col, marker=mk, lw=0, label=f'B={B_disp}'))

# Source styles (theory vs MCMC)
legend_elements.append(Line2D([0], [0], color='k', ls='-', label='Theory'))
legend_elements.append(Line2D([0], [0], color='k', ls='--', label='MCMC'))

# Contraction threshold style
legend_elements.append(Line2D([0], [0], color='k', ls=':', label=r'$\sigma^\star$ threshold'))

ax.legend(handles=legend_elements, ncol=2)

plt.tight_layout()
plt.savefig("./threshold/superimposed_overlap.pdf")
plt.show()




if False:
        
    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(7,5))
    
    # Plot theory curves
    cmap = plt.get_cmap('tab10')  # color map
    B_list = list(results.keys())
    for i, B in enumerate(B_list):
        gaps, q_vals, e_vals, p_vals, pth_vals = results[B]
        ax.plot(SigmaL_th, e_vals, 'o-', color=cmap(i), label=f'Theory B={B}')
    
    # Plot MCMC curves with error bars
    for iM in range(Ml):
        ax.errorbar(
            SigmaL_mc,
            mean_res[:, iM],
            yerr=ci95_res[:, iM],
            fmt='s--', capsize=3, label=f'MCMC M={iM+1}'
        )
    
    # Axes and labels
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'energy $\langle e\rangle$ / residual energy $\langle e-e_0\rangle$')
    ax.set_ylim(-1.3, -0.7)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Superimposed theory (left panel) and MCMC (residual energy)")
    
    plt.tight_layout()
    plt.savefig("./threshold/superimposed_energy.pdf")
    plt.show()

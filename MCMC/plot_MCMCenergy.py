import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# user inputs
# --------------------------
N_list = [30,50,100]                 # edit as needed
DATA_DIR = "energy"                      # where the .npz files live
OUTFILE  = os.path.join(DATA_DIR, "plot_MCMCenergy.pdf")

# If your .npz did NOT save Ml, set this to the exact M values used when generating.
# Example for L=2 and [2,4,10,20,50,100]:  ML_OVERRIDE = 2*np.array([2,4,10,20,50,100])
ML_OVERRIDE = None                       # e.g. np.array([4,8,20,40,100,200])

# --------------------------
# helpers
# --------------------------
def load_latest_npz_for_N(N, data_dir):
    pattern = os.path.join(data_dir, f"energy_N{N}_*.npz")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files match {pattern}")
    # pick most recent by mtime
    path = max(matches, key=os.path.getmtime)
    return np.load(path, allow_pickle=False)

def min_over_sigma(mean_res, ci95_res):
    """
    For each column M, pick sigma that minimizes mean_res[:, M].
    Returns:
      min_vals, min_errs, min_idx
    """
    idx = np.nanargmin(mean_res, axis=0)                                   # (nM,)
    cols = np.arange(mean_res.shape[1])
    mins  = mean_res[idx, cols]
    errs  = ci95_res[idx, cols]
    return mins, errs, idx

def powerlaw_fit_loglog(Mvals, y, yerr):
    """
    Weighted linear fit in log space:
        log y = p * log M + log a
    weights w = 1 / var[log y] ≈ (y / yerr)^2
    Returns p, a and mask used.
    """
    tiny = np.finfo(float).tiny
    x = Mvals.astype(float)
    y = np.clip(y, tiny, None)
    yerr = np.clip(yerr, tiny, None)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)

    if mask.sum() < 2:
        return np.nan, np.nan, mask

    w = 1.0 / np.clip((yerr[mask] / y[mask])**2, 1e-12, None)
    logx = np.log(x[mask])
    logy = np.log(y[mask])
    p, loga = np.polyfit(logx, logy, deg=1, w=w)
    a = np.exp(loga)
    return p, a, mask

# --------------------------
# main
# --------------------------
plt.figure(figsize=(5.2, 4.2))

legend_entries = []

for N in N_list:
    data = load_latest_npz_for_N(N, DATA_DIR)

    # required arrays in .npz produced by your generator
    SigmaL   = data["SigmaL"]           # shape (nS,)
    mean_res = data["mean_res"]         # shape (nS, nM)
    ci95_res = data["ci95_res"]         # shape (nS, nM)

    # try to get Ml from file; else use override; else fall back to 1..nM
    nM = mean_res.shape[1]
    if "Ml" in data.files:
        Ml = data["Ml"].astype(float)
    elif ML_OVERRIDE is not None:
        Ml = np.asarray(ML_OVERRIDE, dtype=float)
        if Ml.shape[0] != nM:
            raise ValueError(f"ML_OVERRIDE has length {Ml.shape[0]} but data has {nM} M points")
    else:
        Ml = np.arange(1, nM + 1, dtype=float)  # fallback labels if Ml was not saved

    # compute minima over sigma for each M
    min_vals, min_errs, min_idx = min_over_sigma(mean_res, ci95_res)
    min_sigmas = SigmaL[min_idx]

    # plot errorbars
    plt.errorbar(Ml, np.clip(min_vals, np.finfo(float).tiny, None),
                 yerr=min_errs, marker='o', linestyle='-', capsize=2, label=f"N={N}")

    # fit and overlay a dotted line
    p, a, mask = powerlaw_fit_loglog(Ml, min_vals, min_errs)
    if np.isfinite(p):
        M_fit = np.geomspace(Ml[mask].min(), Ml[mask].max(), 256)
        y_fit = a * M_fit**p
        plt.plot(M_fit, y_fit, linestyle=':', label=f"fit N={N}, slope={p:.3f}")

    # optional: annotate sigma* at each point (comment out if cluttered)
    for xM, yv, s in zip(Ml, min_vals, min_sigmas):
        plt.annotate(fr'σ*={s:.2g}', (xM, yv), textcoords='offset points',
                     xytext=(5, 5), fontsize=7)

plt.xscale('log'); plt.yscale('log')
plt.xlabel('M')
plt.ylabel(r'$\min_{\sigma}\ \langle e-e_0\rangle$')
plt.title('Minimum residual vs M (multiple N)')
plt.grid(True, which='both', ls=':')
plt.legend(fontsize=8, ncol=1)
plt.tight_layout()
os.makedirs(DATA_DIR, exist_ok=True)
plt.savefig(OUTFILE, format="pdf", bbox_inches="tight")
plt.show()

print(f"Saved: {OUTFILE}")

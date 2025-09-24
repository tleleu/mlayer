# ---------------------------------------------------------------
#  Predicted retention P_retain vs. isotropy μ from zero-T PD
#   - single Gaussian mixing width σ=1.5, L=10
#   - B in {2,5,10} * L => {20, 50, 100}
#   - μ = 1 - (deg-1) * α_mean, where α_mean averages unsaturation per edge
#   - σ_eff^2 ≈ ((deg-1) * C_max * λ2(Q))^2 with C_max = 0.995-quantile activity
#   - P_retain = 1 - exp( - (B-1) * μ / σ_eff^2 ), with cR^2 set to 1
#   - Plots: (a) cloud + binned dotted curve; (b) linearity check
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, time

# -- your provided PD & mixing-Q builders -----------------------
from RDE_fixed10 import run_PD
from mlayer import create_mixing_Q

# ------------------ utilities ----------------------------------

def lambda2_of_Q(Q: np.ndarray) -> float:
    """Second largest eigenvalue (by real part) for symmetric row-stochastic Q."""
    w = np.linalg.eigvals(Q)
    w = np.sort(np.real(w))
    return w[-2] if w.size >= 2 else 0.0

def mu_and_activity_from_hat(hat_pop: np.ndarray,
                             deg: int,
                             Jabs: float = 1.0,
                             tol: float = 1e-12) -> tuple[float, float, float]:
    """
    From half-fields hat_pop (n_edges, B):
      - Per (edge,block) unsaturation: |hat| < |J|
      - Per-edge fraction of unsaturated blocks: frac_unsat_e
      - α_mean  = mean_e frac_unsat_e   (used for μ)
      - α_q995  = 0.995-quantile over edges of frac_unsat_e (proxy for C_max)
      - μ = 1 - (deg-1) * α_mean, clamped to [1e-8, 1.0]
    Returns: (mu, alpha_mean, alpha_q995)
    """
    mask = (np.abs(hat_pop) < (abs(Jabs) - tol))  # boolean matrix (n_edges, B)
    # per-edge fraction of unsaturated channels
    frac_unsat = mask.mean(axis=1)                # shape (n_edges,)

    alpha_mean = float(frac_unsat.mean())
    q = 0.995
    alpha_q995 = float(np.quantile(frac_unsat, q))

    mu = 1.0 - (deg - 1) * alpha_mean
    return mu, alpha_mean, alpha_q995

def sigma_eff2_from_activity(alpha_max: float, lam2: float, deg: int,
                             eps: float = 1e-12) -> float:
    """
    σ_eff^2 ≈ ((deg-1) * C_max * λ2(Q))^2; unknown prefactor absorbed by cR^2.
    """
    base = (deg - 1) * alpha_max * lam2
    return float(max(base * base, eps))

def run_trials_for_B(B: int, Q: np.ndarray, y: float, deg: int,
                     n_edges: int, n_iter: int,
                     n_trials: int, seed0: int,
                     verbose: bool = True):
    """
    For fixed (B, Q), collect samples of (mu, sigma_eff2, Predicted P_retain).
    """
    lam2 = lambda2_of_Q(Q)
    rng  = np.random.default_rng(seed0)
    mu_list, P_list, xlin_list, ylin_list = [], [], [], []

    for t in tqdm(range(n_trials), desc=f"B={B}: trials", disable=not verbose):
        h, hat = run_PD(y, Q, B, deg, n_edges=n_edges, n_iter=n_iter, seed=int(rng.integers(1<<31)))

        mu, alpha_mean, alpha_q995 = mu_and_activity_from_hat(hat, deg=deg)
        sigma_eff2 = sigma_eff2_from_activity(alpha_max=alpha_mean, lam2=lam2, deg=deg)
        #sigma_eff2 = sigma_eff2_from_activity(alpha_max=alpha_q995, lam2=lam2, deg=deg)

        # Retention with cR^2 = 1
        exponent = (B - 1) * mu / sigma_eff2
        # numerical safety
        exponent = float(np.clip(exponent, 0.0, 200.0))
        P_ret = 1.0 - np.exp(-exponent)
        #print(P_ret)

        # store
        mu_list.append(mu)
        P_list.append(P_ret)
        # linearity check: -log(1-P) vs (B-1)*mu/sigma_eff^2
        xlin_list.append(exponent)
        ylin_list.append(-np.log(max(1.0 - P_ret, 1e-300)))

    return (np.array(mu_list), np.array(P_list),
            np.array(xlin_list), np.array(ylin_list), lam2)

def bin_xy(x: np.ndarray, y: np.ndarray, bins: np.ndarray):
    """
    Bin x into 'bins' and compute mean(y) per bin. Returns (centers, mean_y, counts).
    """
    idx = np.digitize(x, bins, right=False) - 1
    idx = np.clip(idx, 0, len(bins)-2)
    nb  = len(bins) - 1
    sums = np.zeros(nb); counts = np.zeros(nb, dtype=int)
    for i, yi in zip(idx, y):
        sums[i]   += yi
        counts[i] += 1
    mean_y = np.divide(sums, np.maximum(counts, 1), out=np.zeros_like(sums), where=counts>0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, mean_y, counts

def jitter(values: np.ndarray, scale: float, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return values + rng.normal(0.0, scale, size=values.shape)

# ---------------- driver --------------------

if __name__ == "__main__":
    # PD / graph parameters
    k_excess   = 2
    deg        = k_excess + 1
    n_edges    = 30_000
    n_iter     = 50
    y_fix      = 0.21 #ref 0.41

    # mixing-Q
    L          = 30
    #L          = 2
    #sigma      = 0.5 #ref 0.5
    #sigma_list = [0.425, 0.47]
    sigma_list = [0.5, 0.5]
    #B_list     = [2*L, 5*L, 10*L]     # {20, 50, 100}
    B_list     = [2*L, 5*L]     # {20, 50, 100}

    # sampling
    trials_per_B = 1000
    seed0        = 123456

    # bins (shared across B)
    mu_bins = np.linspace(0.0, 0.05, 50)   # 15 bins

    # collect data
    results = {}
    t0 = time.time()
    for idx, B in enumerate(B_list):
        sigma = sigma_list[idx]  
        Q = create_mixing_Q(B, mtype="block", sigma=sigma, L=L)
        mu, P, xlin, ylin, lam2 = run_trials_for_B(
            B, Q, y_fix, deg, n_edges, n_iter, n_trials=trials_per_B, seed0=seed0 + 97*B
        )
        c_mu, c_P, c_cnt = bin_xy(mu, P, mu_bins)
        results[B] = dict(mu=mu, P=P, xlin=xlin, ylin=ylin,
                          mu_bins=mu_bins, mu_centers=c_mu, P_mean=c_P, counts=c_cnt, lam2=lam2)
        
        # save raw
        np.savez(f"isotropy/retain_vs_mu_data_B{B}_sigma{sigma:.3f}.npz",
                 params=dict(deg=deg, y_fix=y_fix, L=L, sigma=sigma,
                             trials_per_B=trials_per_B, n_edges=n_edges, n_iter=n_iter),
                 B_list=np.array(B_list),
                 results=results)
        print(f"Saved retain_vs_mu/retain_vs_mu_data_B{B}_sigma{sigma:.3f}.npz")

    print(f"Total wall-time: {time.time()-t0:.1f} s")

    # --------------- plots --------------------

    os.makedirs("isotropy", exist_ok=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(B_list)))
    rng  = np.random.default_rng(0)

    # (A) P_retain vs mu : cloud + dotted binned curve
    plt.figure(figsize=(8.0, 5.0))
    for color, B, sigma in zip(cmap, B_list, sigma_list):
        mu   = results[B]["mu"]
        P    = results[B]["P"]
        c_mu = results[B]["mu_centers"]
        c_P  = results[B]["P_mean"]

        # cloud (μ, P) with slight vertical jitter for readability
        #P_cloud = np.clip(jitter(P, 0.01, rng), 0.0, 1.0)
        P_cloud = P
        plt.scatter(mu, P_cloud, s=12, alpha=0.28, color=color, edgecolor='none')

        # dotted bin curve
        plt.plot(c_mu, c_P, ":", color=color, lw=2.3, label=f"B={B}, σ={sigma:.3f}")

    plt.xlabel(r"isotropy proxy $\mu = 1-(\mathrm{deg}-1)\,\alpha_{\mathrm{mean}}$")
    plt.ylabel(r"predicted retention $P_{\mathrm{retain}}$")
    plt.title(fr"$L={L}$, $y={y_fix}$, deg={deg}  (c$R^2$ set to 1)")
    plt.xlim(mu_bins[0], mu_bins[-1]); plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig("isotropy/Pretain_vs_mu_cloud_dotted.pdf")
    plt.show()

    # (B) Linearity check: -log(1-P) vs (B-1)*mu/sigma_eff^2
    plt.figure(figsize=(7.2, 4.5))
    for color, B in zip(cmap, B_list):
        xlin = results[B]["xlin"]
        ylin = results[B]["ylin"]
        # binned to reduce noise
        xbins = np.linspace(0.0, np.percentile(xlin, 99.5), 20)
        xc, yc, _ = bin_xy(xlin, ylin, xbins)
        plt.plot(xc, yc, "o-", color=color, label=f"B={B}", alpha=0.8)
    # reference y=x
    xmax = plt.gca().get_xlim()[1]
    plt.plot([0, xmax], [0, xmax], "k--", alpha=0.5, label="y=x")
    plt.xlabel(r"$x=\frac{(B-1)\,\mu}{\sigma_{\mathrm{eff}}^2}$")
    plt.ylabel(r"$-\log(1-P_{\mathrm{retain}})$")
    plt.title("Linearity check (c$R^2=1$): expect near y=x")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig("isotropy/linearity_check.pdf")
    plt.show()

# ---------------------------------------------------------------
#  Empirical P_retain(μ) from zero-T dynamic SP (population dynamics)
#   - No closed-form surrogate used in estimation of P_retain
#   - Capture/hold defined via block-orthogonal energy ratio r_perp(t)
#   - μ = 1 - (deg-1) * α_mean measured at capture time
#   - Optional collapse check vs (B-1) μ / σ_eff^2
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, time

from RDE_fixed10 import step_dynamic_SP          # your stepper (returns updated h and hat)
from mlayer import create_mixing_Q               # your mixing builder

# ----------------- helpers --------------------

def lambda2_of_Q(Q: np.ndarray) -> float:
    w = np.linalg.eigvals(Q)
    w = np.sort(np.real(w))
    return w[-2] if w.size >= 2 else 0.0

def block_orth_ratio(hat: np.ndarray, eps: float = 1e-12) -> float:
    """
    hat: (n_edges, B) array of single-neighbor half-fields at this step.
    r_perp = mean_e var_b(hat) / mean_e mean_b(hat^2)
    dimensionless; small r_perp ⇒ block alignment (orthogonal modes quenched).
    """
    # per-edge mean over blocks
    mean_b = hat.mean(axis=1, keepdims=True)
    var_b  = ((hat - mean_b)**2).mean(axis=1)      # (n_edges,)
    num    = float(var_b.mean())
    den    = float((hat*hat).mean())
    return num / (den + eps)

def alpha_stats_from_hat(hat: np.ndarray, Jabs: float = 1.0, tol: float = 1e-12,
                         q_quantile: float = 0.995):
    """
    Unsaturation indicators at zero T: |hat| < |J|.
    Returns:
      alpha_mean : mean over edges, blocks
      alpha_q    : q-quantile over edges of per-edge unsaturation fraction (conservative)
    """
    mask = (np.abs(hat) < (abs(Jabs) - tol))
    per_edge_frac = mask.mean(axis=1)
    alpha_mean = float(per_edge_frac.mean())
    alpha_q    = float(np.quantile(per_edge_frac, q_quantile))
    return alpha_mean, alpha_q

def sigma_eff2_from_alpha(alpha_max: float, lam2: float, deg: int, eps: float = 1e-12) -> float:
    """
    Scalar bound used for the collapse check (operator norm of covariance):
      σ_eff^2 ≈ ((deg-1) * alpha_max * λ2(Q))^2
    """
    base = (deg - 1) * alpha_max * lam2
    return float(max(base*base, eps))

def run_one_trial(y: float, Q: np.ndarray, B: int, deg: int,
                  n_edges: int, n_steps: int,
                  tau_cap: float, tau_hold: float, T_hold: int,
                  seed: int):
    """
    One PD trajectory with capture/hold detection.
    Returns:
      success (0/1), mu_at_cap, sigma_eff2_at_cap (for collapse plot), lam2(Q).
    """
    rng = np.random.default_rng(seed)

    # init cavity fields
    init_low, init_high = -2, 3
    h = rng.integers(init_low, init_high, size=(n_edges, B)).astype(np.float64)
    hat = np.empty_like(h)    # will be filled by stepper
    logW = np.empty(n_edges, dtype=np.float64)

    # normalize Q defensively
    Q = Q.astype(np.float64).copy()
    rs = Q.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    Q /= rs

    # time loop
    t_cap = None
    hat_at_cap = None
    r_series = []

    for t in range(n_steps):
        # one dynamic SP step
        h, hat = step_dynamic_SP(h, y, Q, B, deg, logW, hat)

        # block-orth ratio
        r = block_orth_ratio(hat)
        r_series.append(r)
      
        # first time we are "captured"
        if t_cap is None and r <= tau_cap:
            t_cap = t
            hat_at_cap = hat.copy()

        # if captured, check hold window
        if t_cap is not None:
            if t >= t_cap + T_hold:
                # decide success only if all intermediate r <= tau_hold
                window_ok = all(r_ <= tau_hold for r_ in r_series[t_cap : t_cap + T_hold + 1])
                success = 1 if window_ok else 0
                # compute μ and σ_eff^2 at capture
                #alpha_mean, alpha_q = alpha_stats_from_hat(hat_at_cap)
                alpha_mean, alpha_q = alpha_stats_from_hat(hat)
                mu = 1.0 - (deg - 1) * alpha_mean
                #mu = max(mu, 0.0)  # clamp
                lam2 = lambda2_of_Q(Q)
                sigma_eff2 = sigma_eff2_from_alpha(alpha_max=alpha_mean, lam2=lam2, deg=deg)
                #sigma_eff2 = sigma_eff2_from_alpha(alpha_max=alpha_q, lam2=lam2, deg=deg)
                return success, mu, sigma_eff2, lam2

    # no capture within n_steps
    if t_cap is not None:
        alpha_mean, alpha_q = alpha_stats_from_hat(hat_at_cap)
        mu = 1.0 - (deg - 1) * alpha_mean
        lam2 = lambda2_of_Q(Q)
        sigma_eff2 = sigma_eff2_from_alpha(alpha_max=alpha_mean, lam2=lam2, deg=deg)
        return 0, mu, sigma_eff2, lam2
    
    if t_cap is None:
        hat_eval = hat
        alpha_mean, alpha_q = alpha_stats_from_hat(hat_eval)
        mu = max(0.0, 1.0 - (deg-1)*alpha_mean)
        sigma_eff2 = sigma_eff2_from_alpha(alpha_max=alpha_q, lam2=lam2, deg=deg)
        return 0, mu, sigma_eff2, lam2    
            
def bin_xy(x: np.ndarray, y: np.ndarray, bins: np.ndarray):
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
    # graph / PD params
    k_excess   = 2
    deg        = k_excess + 1
    y_fix      = 0.21
    n_edges    = 30_000
    n_steps    = 50

    # capture/hold thresholds (tune mildly if needed)
    tau_cap  = 1e-2     # declare 'captured' when r_perp <= 1e-2
    tau_hold = 2e-2     # must remain below this tighter threshold
    T_hold   = 10       # consecutive steps required to stay inside

    # mixing Q: single sigma, fixed L
    #L          = 5
    L          = 2
    #sigma      = 0.5
    #sigma      = 0.425 #B=2L
    #sigma      = 0.47 #B=5L
    sigma_list = [0.425, 0.47]
    #B_list     = [2*L, 5*L, 10*L]     # {20, 50, 100}
    #B_list     = [2*L, 5*L]     # {20, 50, 100}
    B_list     = [2*L, 5*L]

    # trials
    trials_per_B = 1000
    seed0        = 12345

    # μ-bins (shared)
    mu_bins = np.linspace(-0.07, 0.03, 40)

    # collect results
    results = {}
    t0 = time.time()
    for idx, B in enumerate(B_list):
        sigma = sigma_list[idx]
        Q = create_mixing_Q(B, mtype="block", sigma=sigma, L=L)
        lam2 = lambda2_of_Q(Q)

        mu_list, succ_list, xlin_list, ylin_list = [], [], [], []

        for t in tqdm(range(trials_per_B), desc=f"B={B}: trials", leave=False):
            success, mu, sigma_eff2, _ = run_one_trial(
                y=y_fix, Q=Q, B=B, deg=deg, n_edges=n_edges, n_steps=n_steps,
                tau_cap=tau_cap, tau_hold=tau_hold, T_hold=T_hold,
                seed=seed0 + 7919*B + 17*t
            )
            #print(mu,success)
            mu_list.append(mu)
            succ_list.append(success)
            # collapse check coordinates
            xlin = (B - 1) * mu / max(sigma_eff2, 1e-12)
            ylin = -np.log(max(1.0 - success, 1e-12))   # = 0 if success=0; = +∞ if success=1; use cap
            xlin_list.append(xlin)
            ylin_list.append(min(ylin, 10.0))           # cap for plotting

        mu_arr   = np.array(mu_list)
        succ_arr = np.array(succ_list, dtype=float)
        xlin_arr = np.array(xlin_list)
        ylin_arr = np.array(ylin_list)

        c_mu, c_P, c_cnt = bin_xy(mu_arr, succ_arr, mu_bins)

        results[B] = dict(
            mu=mu_arr, P=succ_arr,
            mu_bins=mu_bins, mu_centers=c_mu, P_mean=c_P, counts=c_cnt,
            xlin=xlin_arr, ylin=ylin_arr, lam2=lam2, sigma=sigma
        )
        
        #save
        out_dir = "empirical_retain"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, f"empirical_retain_B{B}_sigma{sigma:.3f}.npz"),
            params=dict(deg=deg, y_fix=y_fix, L=L, B=B, sigma=sigma,
                        trials_per_B=trials_per_B, n_edges=n_edges, n_steps=n_steps,
                        tau_cap=tau_cap, tau_hold=tau_hold, T_hold=T_hold),
            lam2=lam2,
            mu=mu_arr, P=succ_arr, xlin=xlin_arr, ylin=ylin_arr,
            mu_bins=mu_bins, mu_centers=c_mu, P_mean=c_P, counts=c_cnt
        )

    print(f"Total wall-time: {time.time()-t0:.1f} s")

    # ---------- plots ----------
    os.makedirs("isotropy_empirical", exist_ok=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(B_list)))
    rng  = np.random.default_rng(0)

    # (A) Empirical P_retain vs μ : cloud + binned curve
    plt.figure(figsize=(8.0, 5.0))
    for color, B in zip(cmap, B_list):
        sigma = results[B]["sigma"]
        mu   = results[B]["mu"]
        P    = results[B]["P"]
        c_mu = results[B]["mu_centers"]
        c_P  = results[B]["P_mean"]

        # cloud (jitter P slightly for visibility)
        #P_cloud = np.clip(jitter(P, 0.03, rng), 0.0, 1.0)
        plt.scatter(mu, P, s=16, alpha=0.30, color=color, edgecolor='none')

        # dotted bin curve
        plt.plot(c_mu, c_P, ":", color=color, lw=2.3, label=f"B={B}, σ={sigma:.3f}")

    plt.xlabel(r"isotropy proxy  $\mu = \max\{0,\,1-(\mathrm{deg}-1)\,\alpha_{\mathrm{mean}}\}$ at capture")
    plt.ylabel(r"empirical retention $P_{\mathrm{retain}}$")
    plt.title(
        fr"$L={L}$, $y={y_fix}$, deg={deg} "
        fr"(capture: $r_\perp \leq {tau_cap}$, hold {T_hold} steps)"
    )
    plt.legend(title="(B, σ)")
    plt.xlim(mu_bins[0], mu_bins[-1]); plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("isotropy_empirical/Pretain_vs_mu.pdf"); plt.show()

    # (B) Collapse check:  -log(1-P)  vs  (B-1) μ / σ_eff^2
    plt.figure(figsize=(7.2, 4.6))
    for color, B in zip(cmap, B_list):
        sigma = results[B]["sigma"]
        x = results[B]["xlin"]
        y = results[B]["ylin"]
        # bin for readability
        xbins = np.linspace(0.0, np.percentile(x, 99.0), 20)
        xc, yc, _ = bin_xy(x, y, xbins)
        plt.plot(xc, yc, "o-", color=color, alpha=0.9, label=f"B={B}, σ={sigma:.3f}")

    # reference y=x
    xmax = plt.gca().get_xlim()[1]
    plt.plot([0, xmax], [0, xmax], "k--", alpha=0.5, label="y=x (ideal)")
    plt.xlabel(r"$x=(B-1)\mu/\sigma_{\rm eff}^2$  (measured at capture)")
    plt.ylabel(r"$-\log(1-P_{\rm retain})$")
    plt.legend(title="(B, σ)")
    plt.title("Collapse check (expect near-linear trend)")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("isotropy_empirical/collapse_check.pdf"); plt.show()

# ---------------------------------------------------------------
#  Load & compare: theory vs empirical P_retain(Î¼)
#    - scans retain_vs_mu/*.npz  and  empirical_retain/*.npz
#    - matches by (B, sigma)
#    - plots per-(B,Ï) panel: empirical cloud + dotted (emp) + solid (theory)
#    - plots superimposed summary across (B,Ï)
# ---------------------------------------------------------------
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ----------------------- helpers -------------------------------

def parse_B_sigma_from_name(fname):
    """
    Try to parse B and sigma from filename with several patterns.
    Returns (B:int or None, sigma:float or None)
    """
    base = os.path.basename(fname)
    patts = [
        r"_B(?P<B>\d+)_sigma(?P<sig>\d+\.\d+)",   # ..._B50_sigma0.470.npz
        r"_B(?P<B>\d+)\.npz",                     # ..._B50.npz    (sigma may be in npz)
    ]
    for p in patts:
        m = re.search(p, base)
        if m:
            B = int(m.group("B"))
            sig = float(m.group("sig")) if "sig" in m.groupdict() else None
            return B, sig
    return None, None

def safe_get(d, keys, default=None):
    for k in keys:
        if k in d: 
            return d[k]
    return default

def load_theory_npz(path):
    """
    Load one theory npz file.
    Returns a dict:
      {'B':..., 'sigma':..., 'mu':..., 'P':..., 'mu_centers':..., 'P_mean':..., 'source':path}
    """
    data = np.load(path, allow_pickle=True)
    B_file, sig_file = parse_B_sigma_from_name(path)

    # try to read params
    params = data.get('params', None)
    B = B_file
    sigma = sig_file
    if params is not None:
        p = params.item() if isinstance(params, np.ndarray) else params
        B = p.get('B', B)
        sigma = p.get('sigma', sigma)

    # results may be saved flat or inside dict
    results = data.get('results', None)

    if results is None:
        # flat style (per-file only)
        mu = data['mu']
        P  = data['P']
        mu_centers = safe_get(data, ['mu_centers', 'mu_centres', 'mu_center'])
        P_mean     = safe_get(data, ['P_mean', 'Pmean'])
        return {'B': int(B), 'sigma': float(sigma) if sigma is not None else None,
                'mu': mu, 'P': P,
                'mu_centers': mu_centers, 'P_mean': P_mean,
                'source': path, 'kind': 'theory'}
    else:
        # dict-of-B style; extract the B in this filename if possible
        res = results.item() if isinstance(results, np.ndarray) else results
        if B is None:
            # if no B in filename, try single key
            if len(res) == 1:
                B = int(list(res.keys())[0])
            else:
                raise ValueError(f"Cannot infer B from {path}")
        entry = res[int(B)]
        mu = entry['mu']
        P  = entry['P']
        mu_centers = entry['mu_centers']
        P_mean     = entry['P_mean']
        return {'B': int(B), 'sigma': float(sigma) if sigma is not None else None,
                'mu': mu, 'P': P,
                'mu_centers': mu_centers, 'P_mean': P_mean,
                'source': path, 'kind': 'theory'}

def load_empirical_npz(path):
    """
    Load one empirical npz file.
    Returns a dict:
      {'B':..., 'sigma':..., 'mu':..., 'P':..., 'mu_centers':..., 'P_mean':..., 'source':path}
    """
    data = np.load(path, allow_pickle=True)
    B_file, sig_file = parse_B_sigma_from_name(path)

    # params may exist; sometimes sigma not in name
    params = data.get('params', None)
    B = B_file
    sigma = sig_file
    if params is not None:
        p = params.item() if isinstance(params, np.ndarray) else params
        B = p.get('B', B)
        sigma = p.get('sigma', sigma)

    results = data.get('results', None)

    if results is None:
        # flat style (per-file only)
        mu = data['mu']
        P  = data['P']          # empirical successes 0/1 (already averaged in bins below)
        mu_centers = safe_get(data, ['mu_centers', 'mu_centres'])
        P_mean     = safe_get(data, ['P_mean', 'Pmean'])
        return {'B': int(B), 'sigma': float(sigma) if sigma is not None else None,
                'mu': mu, 'P': P,
                'mu_centers': mu_centers, 'P_mean': P_mean,
                'source': path, 'kind': 'empirical'}
    else:
        # dict-of-B style
        res = results.item() if isinstance(results, np.ndarray) else results
        if B is None:
            if len(res) == 1:
                B = int(list(res.keys())[0])
            else:
                raise ValueError(f"Cannot infer B from {path}")
        entry = res[int(B)]
        mu = entry['mu']
        P  = entry['P']         # empirical successes 0/1
        mu_centers = entry['mu_centers']
        P_mean     = entry['P_mean']
        return {'B': int(B), 'sigma': float(sigma) if sigma is not None else None,
                'mu': mu, 'P': P,
                'mu_centers': mu_centers, 'P_mean': P_mean,
                'source': path, 'kind': 'empirical'}

def collect_by_B_sigma(folder, kind):
    """
    Scan folder for npz files and load them by kind ('theory' or 'empirical').
    Returns dict keyed by (B, sigma) -> entry
    """
    out = {}
    files = glob(os.path.join(folder, "*.npz"))
    for path in files:
        try:
            entry = load_theory_npz(path) if kind == 'theory' else load_empirical_npz(path)
            B = entry['B']; sigma = entry['sigma']
            if B is None or sigma is None:
                # try to recover sigma from filename-less store (skip if missing)
                continue
            out[(B, float(sigma))] = entry
        except Exception as e:
            print(f"[WARN] skip {path}: {e}")
    return out

def ensure_common_bins(mu_centers_theo, P_mean_theo, mu_centers_emp, P_mean_emp):
    """
    If the theory and empirical binnings differ slightly, linearly interpolate
    theory to empirical centers (or vice-versa).
    """
    if mu_centers_theo is None or mu_centers_emp is None:
        return mu_centers_emp, P_mean_theo, P_mean_emp
    if np.allclose(mu_centers_theo, mu_centers_emp):
        return mu_centers_emp, P_mean_theo, P_mean_emp
    # interpolate theory onto empirical centers
    x = mu_centers_theo
    y = P_mean_theo
    xi = mu_centers_emp
    yi = np.interp(xi, x, y, left=y[0], right=y[-1])
    return xi, yi, P_mean_emp

# ----------------------- main ----------------------------------

if __name__ == "__main__":
    # where your npz files live
    theory_dir     = "isotropy"
    empirical_dir  = "isotropy_empirical"
    
    # 1) how to form the CANONICAL key from each dataset's native key/value
    #    return None to drop an item
    def key_fn_theory(native_key, rec):
        B, sigma = native_key            # e.g. native is (B, sigma)
        return (int(B), round(float(sigma), 3))
    
    def key_fn_emp(native_key, rec):
        B, sigma = native_key
        return (int(B), round(float(sigma), 3))
    
    # 2) optional explicit sets (in canonical space)
    #    leave as None to use all available, or give lists like [(4,0.250), (8,0.125)]
    CANONICAL_KEYS_FOR_PANELS = None         # used for per-(B,σ) panels; requires both sides present
    CANONICAL_KEYS_EMP_ONLY   = [(4, 0.425), (10, 0.470)]         # used for superimposed dotted curves
    CANONICAL_KEYS_THEO_ONLY  = [(60, 0.5), (150, 0.5)]         # used for superimposed solid curves
    # -----------------------------------------------------------------------------
    
    # load
    theo_raw = collect_by_B_sigma(theory_dir, kind="theory")
    emp_raw  = collect_by_B_sigma(empirical_dir,  kind="empirical")
    
    def build_index(d, key_fn):
        """Map native -> canonical. Return: idx[canonical]=rec, rev[canonical]=native."""
        idx, rev = {}, {}
        for nk, rec in d.items():
            ck = key_fn(nk, rec)
            if ck is None:
                continue
            idx[ck] = rec
            rev[ck] = nk
        return idx, rev
    
    theo, theo_rev = build_index(theo_raw, key_fn_theory)
    emp,  emp_rev  = build_index(emp_raw,  key_fn_emp)
    
    # (1) Superimposed summary across (B,σ)
    # independent key choices per side
    emp_keys_plot  = (sorted(emp.keys())  if CANONICAL_KEYS_EMP_ONLY  is None
                      else [k for k in CANONICAL_KEYS_EMP_ONLY  if k in emp])
    theo_keys_plot = (sorted(theo.keys()) if CANONICAL_KEYS_THEO_ONLY is None
                      else [k for k in CANONICAL_KEYS_THEO_ONLY if k in theo])
    
    # make a deterministic color per canonical key across both sets
    all_for_palette = sorted(set(emp_keys_plot) | set(theo_keys_plot))
    palette = dict(zip(all_for_palette, plt.cm.tab20(np.linspace(0, 1, len(all_for_palette)))))
    
    # plot
    
    mu0_th  = 0.0
    mu0_emp = -0.045
    
    def plot_segments(ax, x, y, linestyle, label=None, mask=None, marker="o", **kwargs):
        x = np.asarray(x); y = np.asarray(y)
        base = np.isfinite(x) & np.isfinite(y)
        if mask is None:
            mask = base & (y > 0)            # default: link only y>0
        else:
            mask = base & mask
        if not np.any(mask):
            return
        idx = np.where(mask)[0]
        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        first = True
        for g in groups:
            if g.size >= 2:
                ax.plot(x[g], y[g], linestyle, label=(label if first else None), **kwargs)
            #else:
            #    ax.plot(x[g], y[g], linestyle="None", marker=marker,
            #            label=(label if first else None), **kwargs)
            first = False
            
    os.makedirs("plot_isotropy", exist_ok=True)
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 5.2))
    
    # --- Left: empirical only (P vs mu) ------------------------------------------
    for key in emp_keys_plot:
        ep = emp.get(key)
        if ep is None:
            continue
        muC_emp, PM_emp = ep['mu_centers'], ep['P_mean']
        if muC_emp is None or PM_emp is None:
            continue
        B, sigma = key
        # link only P>0
        plot_segments(
            axL, muC_emp, PM_emp, ":", 
            label=fr"emp: B={B}, $\sigma={sigma:.3f}$",
            mask=(np.asarray(PM_emp) > 0),
            color=palette[key], lw=2.3
        )
    axL.set_xlabel(r"isotropy proxy $\mu$")
    axL.set_ylabel(r"$P_{\rm retain}$")
    axL.set_title("Empirical")
    axL.grid(alpha=0.3)
    axL.legend(ncols=2, fontsize=9)
    
    # --- Middle: theory only (P vs mu) -------------------------------------------
    for key in theo_keys_plot:
        th = theo.get(key)
        if th is None:
            continue
        muC_th, PM_th = th['mu_centers'], th['P_mean']
        if muC_th is None or PM_th is None:
            continue
        B, sigma = key
        # link only P>0
        plot_segments(
            axM, muC_th, PM_th, "-", 
            label=fr"theo: B={B}, $\sigma={sigma:.3f}$",
            mask=(np.asarray(PM_th) > 0),
            color=palette[key], lw=2.0, alpha=0.9
        )
    axM.set_xlabel(r"isotropy proxy $\mu$")
    axM.set_ylabel(r"$P_{\rm retain}$")
    axM.set_title("Theory")
    axM.grid(alpha=0.3)
    axM.legend(ncols=2, fontsize=9)
    
    # --- Right: overlap in log(1 - P) vs (mu - mu0) ------------------------------
    for key in emp_keys_plot:
        ep = emp.get(key)
        if ep is None:
            continue
        muC_emp, PM_emp = ep['mu_centers'], ep['P_mean']
        if muC_emp is None or PM_emp is None:
            continue
        x = np.asarray(muC_emp) - mu0_emp
        y = 1.0 - np.asarray(PM_emp)
        # exclude P==0 (y==1) and P==1 (y==0)
        mask = (y > 0) & (y < 1)
        B, sigma = key
        plot_segments(axR, x, y, ":", label=fr"emp: B={B}, $\sigma={sigma:.3f}$",
                      mask=mask, color=palette[key], lw=2.0)
    
    for key in theo_keys_plot:
        th = theo.get(key)
        if th is None:
            continue
        muC_th, PM_th = th['mu_centers'], th['P_mean']
        if muC_th is None or PM_th is None:
            continue
        x = np.asarray(muC_th) - mu0_th
        y = 1.0 - np.asarray(PM_th)
        mask = (y > 0) & (y < 1)
        B, sigma = key
        plot_segments(axR, x, y, "-", label=fr"theo: B={B}, $\sigma={sigma:.3f}$",
                      mask=mask, color=palette[key], lw=2.0, alpha=0.9)
    
    axR.set_xlabel(r"$\mu - \mu_0$")
    axR.set_ylabel(r"$1 - P_{\rm retain}$")
    axR.set_yscale("log")
    axR.set_title(r"Overlap: log-scale vs $\mu-\mu_0$")
    axR.grid(alpha=0.3, which="both")
    axR.legend(ncols=2, fontsize=9)
    
    os.makedirs("plot_isotropy", exist_ok=True)
    plt.tight_layout(); plt.savefig("plot_isotropy/retain_theory_empirical.pdf"); plt.show()

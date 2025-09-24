import os
import numpy as np
import matplotlib.pyplot as plt

# ===================== config =====================
folder_name = "complexity_B2"  # change if needed
outfile = os.path.join(folder_name, f"{os.path.basename(folder_name)}.pdf")

# ===================== utils ======================
def coerce_array(x):
    """Return a 1-D float ndarray or None if not valid."""
    try:
        arr = np.array(x, dtype=float).squeeze()
        if arr.ndim == 1 and arr.size > 1:
            return arr
        return None
    except Exception:
        return None

# ===================== load =======================
if not os.path.isdir(folder_name):
    raise FileNotFoundError(f"Folder not found: {folder_name}")

file_list = sorted(f for f in os.listdir(folder_name) if f.endswith(".npz"))
if not file_list:
    raise FileNotFoundError(f"No .npz files in {folder_name}")

data_all = {}
phi_the = epsilon_the = Sigma_the = None

for fname in file_list:
    path = os.path.join(folder_name, fname)
    z = np.load(path, allow_pickle=True)

    # robust B cast (handles scalars stored oddly)
    B = int(np.array(z["B"]).item())
    Ny = np.array(z["Ny"], dtype=float)
    mE = np.array(z["mean_E"], dtype=float)
    eE = np.array(z["err_E"], dtype=float)
    eps = np.array(z["epsilon"], dtype=float)
    Sig = np.array(z["Sigma"], dtype=float)

    data_all[B] = dict(Ny=Ny, mean_E=mE, err_E=eE, epsilon=eps, Sigma=Sig)

    # take first valid theory if present
    if phi_the is None:
        if "phi_the" in z.files:
            cand_phi = coerce_array(z["phi_the"])
            cand_eps = coerce_array(z["epsilon_the"]) if "epsilon_the" in z.files else None
            cand_sig = coerce_array(z["Sigma_the"])   if "Sigma_the"   in z.files else None
            if cand_phi is not None:
                phi_the, epsilon_the, Sigma_the = cand_phi, cand_eps, cand_sig

B_list = np.array(sorted(data_all.keys()))
# canonical Ny for theory checks
Ny0 = data_all[B_list[0]]["Ny"]

# ===================== theory =====================
need_recompute = (
    phi_the is None or
    phi_the.size != Ny0.size or
    (epsilon_the is not None and epsilon_the.size != Ny0.size) or
    (Sigma_the   is not None and Sigma_the.size   != Ny0.size)
)

if need_recompute:
    try:
        import G90  # optional
        phi_the = np.array([G90.phi_G90(y) for y in Ny0], dtype=float)
        epsilon_the = np.gradient(Ny0 * phi_the, Ny0, edge_order=2)
        Sigma_the   = Ny0**2 * np.gradient(phi_the, Ny0, edge_order=2)
    except Exception:
        phi_the = epsilon_the = Sigma_the = None  # proceed without theory

# ===================== figure =====================
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1],
                      wspace=0.25, hspace=0.45)
ax_phi = fig.add_subplot(gs[0, 0])
ax_e   = fig.add_subplot(gs[1, 0])
ax_S   = fig.add_subplot(gs[:, 1])

fontsize = 14
markers = ['o', 's', '^', 'v', 'D', 'P', '*']
colors  = plt.cm.viridis(np.linspace(0, 1, len(B_list)))

# ===================== plot theory =====================
if phi_the is not None and phi_the.size == Ny0.size:
    ax_phi.plot(Ny0, phi_the, 'k-', lw=2, label='[MP03]')
if epsilon_the is not None and epsilon_the.size == Ny0.size:
    ax_e.plot(Ny0, epsilon_the, 'k-', lw=2, label='[MP03]')
if (epsilon_the is not None and Sigma_the is not None and
        epsilon_the.size == Sigma_the.size):
    ax_S.plot(epsilon_the, Sigma_the, 'k-', lw=2, label='[MP03]')

# ===================== plot sims ======================
data_dict = {}
for i, (B, c) in enumerate(zip(B_list, colors)):
    m = markers[i % len(markers)]
    rec = data_all[B]
    NyB = rec["Ny"]; mE = rec["mean_E"]; eE = rec["err_E"]

    data_dict[B] = (NyB, mE)

    # φ(y)
    ax_phi.errorbar(NyB, mE, yerr=eE, marker=m, linestyle='', color=c,
                    markersize=6, label=f"B={B}")

    # ε(y) and Σ(e)
    epsB = np.gradient(NyB * mE, NyB, edge_order=2)
    sigB = NyB**2 * np.gradient(mE, NyB, edge_order=2)
    ax_e.plot(NyB, epsB, marker=m, linestyle='', color=c, markersize=6)
    ax_S.plot(epsB, sigB, marker=m, linestyle='', color=c, markersize=6)

# connect points
for B in B_list:
    NyB, mE = data_dict[B]
    epsB = np.gradient(NyB * mE, NyB, edge_order=2)
    sigB = NyB**2 * np.gradient(mE, NyB, edge_order=2)
    idx = np.where(B_list == B)[0][0]
    ax_phi.plot(NyB, mE, '-', lw=1.5, color=colors[idx])
    ax_e.plot(NyB, epsB, '-', lw=1.5, color=colors[idx])
    ax_S.plot(epsB, sigB, '-', lw=1.5, color=colors[idx])

# ===================== refs (energies only) ===========
ref_e = dict(eRSB=-1.2717, eRSBfac=-1.2723, ebf=-1.2719, ebf2=-1.2720)
for ax in [ax_e]:
    ax.axhline(ref_e['eRSB'],    linestyle='--', color='b', alpha=0.5, label='[RSB]')
    ax.axhline(ref_e['eRSBfac'], linestyle='-',  color='k', alpha=0.5, label='[RSB-fac]')
    ax.axhline(ref_e['ebf'],     linestyle=':',  color='m', alpha=0.5, label='[BF]')
    ax.axhline(ref_e['ebf2'],    linestyle='-.', color='r', alpha=0.5, label='[BF2]')
for ax in [ax_S]:
    for v, ls, col in [(ref_e['eRSB'],'--','b'), (ref_e['eRSBfac'],'-','k'),
                       (ref_e['ebf'],':','m'), (ref_e['ebf2'],'-.','r')]:
        ax.axvline(v, linestyle=ls, color=col, alpha=0.5)

# ---------- Axes limits ---------- 
ax_phi.set_xlim(np.min(Ny), np.max(Ny)) 
ax_phi.set_ylim(-1.278, -1.27) 
ax_e.set_xlim(np.min(Ny), np.max(Ny)) 
ax_e.set_ylim(-1.278, -1.27) 
ax_S.set_xlim(-1.278, -1.27) 
ax_S.set_ylim(0.0, 0.0008)

# ===================== labels ========================
ax_phi.set_xlabel("Parisi parameter y", fontsize=fontsize)
ax_phi.set_ylabel("Free energy φ", fontsize=fontsize)
ax_phi.set_title("a", fontsize=fontsize)

ax_e.set_xlabel("Parisi parameter y", fontsize=fontsize)
ax_e.set_ylabel("Energy-per-site e", fontsize=fontsize)
ax_e.set_title("b", fontsize=fontsize)

ax_S.set_xlabel("Energy-per-site e", fontsize=fontsize)
ax_S.set_ylabel("Complexity Σ", fontsize=fontsize)
ax_S.set_title("c", fontsize=fontsize)

for ax in [ax_phi, ax_e, ax_S]:
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

# ===================== legends =======================
for ax in [ax_phi, ax_e, ax_S]:
    h, l = ax.get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(h, l):
        if label not in by_label:
            by_label[label] = handle
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best', ncol=2, frameon=False)

plt.tight_layout()
plt.savefig(outfile, format="pdf", bbox_inches="tight")
plt.show()
print(f"Saved: {outfile}")

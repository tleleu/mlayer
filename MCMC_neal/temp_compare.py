import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ---------- helpers ----------
def signed_ring_diff_vec(perm):
    M = len(perm)
    d = (perm - np.arange(M)) % M
    d[d > M//2] -= M
    return d  # ints in [-floor(M/2), ..., ceil(M/2)]

def perm_from_block(block_MxM):
    """block_MxM: sparse MxM with exactly one nonzero per column."""
    coo = block_MxM.tocoo()
    perm = np.full(block_MxM.shape[1], -1, dtype=int)
    for r, c in zip(coo.row, coo.col):
        if perm[c] != -1:
            # multiple hits in a column: pick the largest magnitude (or first)
            pass
        perm[c] = r
    if np.any(perm < 0):
        missing = np.where(perm < 0)[0]
        raise ValueError(f"column(s) with no assignment: {missing.tolist()}")
    return perm

def extract_perms(JM, J, M):
    """Return forward/backward permutations for each base edge (i<j)."""
    N = J.shape[0]
    rows, cols = np.nonzero(np.triu(J))
    perms_fwd, perms_bwd = [], []
    for i, j in zip(rows, cols):
        r_i = np.arange(M)*N + i
        c_j = np.arange(M)*N + j
        r_j = np.arange(M)*N + j
        c_i = np.arange(M)*N + i
        # forward block (i -> j)
        Bij = JM[r_i, :][:, c_j]
        # backward block (j -> i)
        Bji = JM[r_j, :][:, c_i]
        perms_fwd.append(perm_from_block(Bij))
        perms_bwd.append(perm_from_block(Bji))
    return perms_fwd, perms_bwd

def best_shift(perm):
    """Return (argmax shift s, coherence fraction) where shift aligns pi(a)=a+s mod M."""
    M = len(perm)
    hits = np.zeros(M, dtype=int)
    a = np.arange(M)
    for s in range(M):
        hits[s] = np.sum(perm == (a + s) % M)
    s_star = int(np.argmax(hits))
    return s_star, hits.max() / M

def pair_alignment(pi, pj):
    """Max fraction over circular shifts s of {a: pi(a)=pj(a)+s mod M}."""
    M = len(pi)
    hits = np.zeros(M, dtype=int)
    for s in range(M):
        hits[s] = np.sum(pi == (pj + s) % M)
    return hits.max() / M

def aggregate_shift_hist(perms):
    """Aggregate Δ=(pi(a)-a) over all edges; return (vals, probs, mean_shift, direction, strength)."""
    if not perms:
        return np.array([0]), np.array([1.0]), 0.0, 0.0, 0.0
    M = len(perms[0])
    deltas = np.concatenate([signed_ring_diff_vec(p) for p in perms])
    # histogram support
    vals = np.arange(-M//2, M//2 + 1)
    counts = np.array([(deltas == v).sum() for v in vals], dtype=float)
    probs = counts / counts.sum()
    mean_shift = float(deltas.mean())
    theta = 2*np.pi * (np.mod(deltas, M)) / M
    Z = np.exp(1j*theta).mean()
    direction, strength = float(np.angle(Z)), float(np.abs(Z))
    return vals, probs, mean_shift, direction, strength

def avg_block(perms):
    """Average MxM block (expected permutation matrix) over edges."""
    if not perms:
        return np.zeros((1, 1))
    M = len(perms[0])
    A = np.zeros((M, M), dtype=float)
    for p in perms:
        A[p, np.arange(M)] += 1.0
    A /= len(perms)
    return A

def plot_hist(vals, probs, title, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.bar(vals, probs, width=1.0)
    plt.xlabel("signed shift Δ")
    plt.ylabel("probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_heatmap(A, title, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.imshow(A, origin='lower', aspect='auto')
    plt.colorbar(label='avg weight')
    plt.xlabel("column a")
    plt.ylabel("row b")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
def directed_ring_Q(M, shift=0.0, tau=4.0, eps=1e-12):
    """
    Asymmetric ring kernel that prefers b ≈ a + shift (mod M).
    tau controls width; smaller -> sharper band around the shifted diagonal.
    """
    a = np.arange(M)
    # signed Δ = (b - a) in [-⌊M/2⌋, …, ⌈M/2⌉], tie kept as +M/2
    Δ = (a[None,:] - a[:,None]) % M
    half = M//2
    Δ[Δ > half] -= M

    # bias toward Δ ≈ shift
    rel = (Δ - shift) % M
    rel[rel > half] -= M
    Q = np.exp(-(rel.astype(float)**2) / (2.0 * tau**2)) + eps   # >0
    Q /= Q.sum(axis=1, keepdims=True)                            # row-stochastic
    return Q

# ---------- main comparator ----------
def compare_mlayers(J, M, C, Q, outdir="energy/mlayer_compare",
                    build1=None, build2=None,
                    typeperm='asym', exact_threshold=14, balance=True,
                    pair_sample_cap=5000, eliminate_drift=False, kappa=0.0):
    """
    Build Mlayer1 and Mlayer2, extract permutations, and compare structures.
    Expects:
      - build1(J,M, C=C) -> JM1
      - build2(J,M, Q=Q, typeperm=..., exact_threshold=..., balance=...) -> JM2
    """
    assert build1 is not None and build2 is not None

    # Build lifts
    JM1 = build1(J, M, C=C, eliminate_drift=eliminate_drift, kappa=kappa)
    JM2 = build2(J, M, Q=Q, typeperm=typeperm, exact_threshold=exact_threshold, balance=balance)

    # Extract per-edge permutations (forward direction; backward similar if needed)
    p1_fwd, p1_bwd = extract_perms(JM1, J, M)
    p2_fwd, p2_bwd = extract_perms(JM2, J, M)

    # Shift histograms
    v1, pr1, ms1, dir1, str1 = aggregate_shift_hist(p1_fwd)
    v2, pr2, ms2, dir2, str2 = aggregate_shift_hist(p2_fwd)

    # Coherence per edge
    coh1 = np.array([best_shift(p)[1] for p in p1_fwd])
    coh2 = np.array([best_shift(p)[1] for p in p2_fwd])

    # Pairwise alignment within each method (sample pairs if many)
    K1, K2 = len(p1_fwd), len(p2_fwd)
    def mean_pair_align(perms):
        import itertools
        idx = list(range(len(perms)))
        pairs = list(itertools.combinations(idx, 2))
        if len(pairs) > pair_sample_cap:
            sel = np.random.default_rng(0).choice(len(pairs), pair_sample_cap, replace=False)
            pairs = [pairs[t] for t in sel]
        return float(np.mean([pair_alignment(perms[i], perms[j]) for i, j in pairs])) if pairs else 1.0

    mpa1 = mean_pair_align(p1_fwd)
    mpa2 = mean_pair_align(p2_fwd)

    # Cross-method alignment (same edge)
    cross_align = float(np.mean([pair_alignment(a, b) for a, b in zip(p1_fwd, p2_fwd)])) if K1 and K2 else 1.0

    # Average blocks
    A1 = avg_block(p1_fwd)
    A2 = avg_block(p2_fwd)
    frob = float(np.linalg.norm(A1 - A2, 'fro'))

    # Print summary
    print("=== Mlayer comparison (forward direction) ===")
    print(f"#edges: {K1}")
    print("--- Shift histogram (Δ) ---")
    nz1 = [(int(v), float(p)) for v, p in zip(v1, pr1) if p > 0]
    nz2 = [(int(v), float(p)) for v, p in zip(v2, pr2) if p > 0]
    print("Mlayer1 nonzero bins:", nz1)
    print("Mlayer2 nonzero bins:", nz2)
    print(f"Mlayer1: mean_shift={ms1:.6f}, direction={dir1:.6f} rad, strength={str1:.6f}")
    print(f"Mlayer2: mean_shift={ms2:.6f}, direction={dir2:.6f} rad, strength={str2:.6f}")
    print("--- Coherence & alignment ---")
    print(f"Mlayer1: mean coherence={coh1.mean():.4f} (±{coh1.std():.4f}), mean pair alignment={mpa1:.4f}")
    print(f"Mlayer2: mean coherence={coh2.mean():.4f} (±{coh2.std():.4f}), mean pair alignment={mpa2:.4f}")
    print(f"Cross-method alignment (same edges): {cross_align:.4f}")
    print("--- Average block ---")
    print(f"||A1 - A2||_F = {frob:.6f}")

    # Plots
    plot_hist(v1, pr1, "Shift distribution Δ (Mlayer1)", os.path.join(outdir, "shift_mlayer1.pdf"))
    plot_hist(v2, pr2, "Shift distribution Δ (Mlayer2)", os.path.join(outdir, "shift_mlayer2.pdf"))
    plot_heatmap(A1, "Average block A (Mlayer1)", os.path.join(outdir, "Aavg_mlayer1.pdf"))
    plot_heatmap(A2, "Average block A (Mlayer2)", os.path.join(outdir, "Aavg_mlayer2.pdf"))

    # Return raw artifacts if needed
    return {
        "perms": {"mlayer1_fwd": p1_fwd, "mlayer2_fwd": p2_fwd,
                  "mlayer1_bwd": p1_bwd, "mlayer2_bwd": p2_bwd},
        "hist": {"vals": v1, "probs1": pr1, "probs2": pr2},
        "stats": {
            "ms1": ms1, "dir1": dir1, "str1": str1,
            "ms2": ms2, "dir2": dir2, "str2": str2,
            "coh1_mean": float(coh1.mean()), "coh1_std": float(coh1.std()),
            "coh2_mean": float(coh2.mean()), "coh2_std": float(coh2.std()),
            "pair_align1": mpa1, "pair_align2": mpa2, "cross_align": cross_align,
            "frob_A_diff": frob
        },
        "Aavg": {"mlayer1": A1, "mlayer2": A2},
        "outdir": outdir
    }

if __name__ == "__main__":

    import temp
    from instance import create_Bethe
    from mlayer2 import create_mixing_Q
  
    N       = 100
    M       = 100
    
    sigma   = 1.0
    L       = 2
    
    J = create_Bethe(N, 2 + 1)
    J = J.todense()
    #Q = create_mixing_Q(M, mtype="block", sigma=sigma * 20, L=L)
    Q = directed_ring_Q(M, shift=+2.5, tau=5.0) 
    
    # ---------- usage example ----------
    # JM1 vs JM2:
    res = compare_mlayers(J, M, Q, Q, build1=temp.Mlayer1, build2=temp.Mlayer2,
                       typeperm='asym', exact_threshold=14, balance=True, eliminate_drift=True, kappa=0.00)


import os, numpy as np
import matplotlib.pyplot as plt

def signed_ring_diff(b, a, M):
    d = (b - a) % M
    if d > M//2:
        d -= M
    return d

def shift_metrics_from_perm(pi):
    """
    pi: 1D array of length M with values 0..M-1 (a permutation)
    Returns:
      per_row_shift: Δ_a = wrap(pi[a]-a) in [-⌊M/2⌋, ⌈M/2⌉)
      mean_shift: average Δ_a
      direction: arg of order parameter in radians
      strength: |order parameter| in [0,1]
      hist_vals, hist_probs: histogram of Δ
    """
    pi = np.asarray(pi, dtype=int)
    M = pi.size
    per_row_shift = np.array([signed_ring_diff(pi[a], a, M) for a in range(M)], dtype=int)
    mean_shift = per_row_shift.mean()

    # circular order parameter
    theta = 2*np.pi * ((pi - np.arange(M)) % M) / M
    Z = np.exp(1j*theta).mean()
    direction, strength = np.angle(Z), np.abs(Z)

    # histogram over signed shifts
    shifts = np.arange(-M//2, M//2 + 1)
    counts = np.array([(per_row_shift == d).sum() for d in shifts], dtype=float)
    probs = counts / counts.sum()

    return per_row_shift, mean_shift, direction, strength, shifts, probs

def plot_shift_distribution(vals, probs, outpath="energy/shift.pdf"):
    vals = np.asarray(vals)
    probs = np.asarray(probs, dtype=float)
    assert vals.shape == probs.shape
    M = len(vals)

    # Basic stats
    nz = np.flatnonzero(probs > 0)
    mean_shift = float(np.sum(vals * probs))
    theta = 2*np.pi * (np.mod(vals, M)) / M
    Z = np.sum(probs * np.exp(1j*theta))
    direction, strength = float(np.angle(Z)), float(np.abs(Z))

    print("Nonzero bins (shift -> prob):")
    for i in nz:
        print(f"{int(vals[i]):>4} -> {probs[i]:.3f}")
    print(f"mean_shift = {mean_shift:.6f}")
    print(f"direction = {direction:.6f} rad, strength = {strength:.6f}")

    # Plot
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.bar(vals, probs, width=1.0)
    plt.xlabel("signed shift Δ")
    plt.ylabel("probability")
    plt.title("Shift distribution")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
def best_shift(pi):
    M = len(pi)
    # count matches for each shift s: pi(a) == a+s (mod M)
    hits = np.zeros(M, dtype=int)
    a = np.arange(M)
    for s in range(M):
        hits[s] = np.sum((pi == (a + s) % M))
    return int(np.argmax(hits)), hits.max()/M  # (argmax shift, fraction)

def pair_alignment(pi, pj):
    M = len(pi)
    hits = np.zeros(M, dtype=int)
    for s in range(M):
        hits[s] = np.sum(pi == (pj + s) % M)
    return hits.max()/M
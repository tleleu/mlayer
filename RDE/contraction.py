import numpy as np
from typing import Sequence, Tuple, Dict, Optional

def lambda2_of_Q(Q: np.ndarray) -> float:
    """
    Second largest eigenvalue (by real part) for a symmetric row-stochastic Q.
    """
    w = np.linalg.eigvals(Q)
    w = np.sort(np.real(w))
    return w[-2] if w.size >= 2 else 0.0

def alpha_from_h_frozen_backbone(h_pop: np.ndarray,
                                 Q: np.ndarray,
                                 J: float = 1.0,
                                 tol: float = 1e-12,
                                 mode: str = "quantile",
                                 q: float = 0.995) -> float:
    """
    Zero-T activity α = P(|(Q h)_b| < |J|) with a fixed backbone h_pop.
    mode="mean" gives average activity; mode="quantile" uses a high quantile
    over edges (conservative, often matches observed onset better).
    """
    # H = Q h  (h_pop: edges×B; Q: B×B)
    H = h_pop @ Q.T
    mask = (np.abs(H) < (abs(J) - tol))  # unsaturated channels
    if mode == "mean":
        return float(mask.mean())
    per_edge = mask.mean(axis=1)         # fraction active per edge
    return float(np.quantile(per_edge, q))

def rho0_from_Q_and_h(Q: np.ndarray,
                      h_pop: np.ndarray,
                      deg: int,
                      J: float = 1.0,
                      alpha_mode: str = "quantile",
                      alpha_q: float = 0.995) -> Dict[str, float]:
    """
    Compute zero-T contraction metric ρ0 = (deg-1)*α*λ2(Q) for a single Q and h_pop.
    """
    lam2 = lambda2_of_Q(Q)
    alpha = alpha_from_h_frozen_backbone(h_pop, Q, J=J, mode=alpha_mode, q=alpha_q)
    rho0  = (deg - 1) * alpha * lam2
    return {"rho0": rho0, "alpha": alpha, "lambda2": lam2}

def find_sigma_star_zeroT_from_Qgrid(h_pop: np.ndarray,
                                     sigmas: Sequence[float],
                                     Q_list: Sequence[np.ndarray],
                                     deg: int,
                                     J: float = 1.0,
                                     alpha_mode: str = "quantile",
                                     alpha_q: float = 0.995
                                     ) -> Tuple[Optional[float], Dict[str, np.ndarray]]:
    """
    Frozen-backbone zero-T threshold from a precomputed grid of (σ, Q(σ)):

    - h_pop: (n_edges, B) backbone cavity fields at your chosen y (fixed).
    - sigmas: sequence of σ values (not necessarily uniform, but strictly increasing recommended).
    - Q_list: sequence of Q matrices, same length as sigmas, where Q_list[i] = Q(sigmas[i]).
    - deg: graph degree d.
    - Returns (sigma_star, info). If no crossing of ρ0=1 is found, sigma_star=None.

    The function finds the first index i with ρ0(σ_i) ≥ 1 and ρ0(σ_{i+1}) ≤ 1 (or vice versa),
    and linearly interpolates between σ_i and σ_{i+1}.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    assert len(sigmas) == len(Q_list), "sigmas and Q_list must have the same length"

    # Compute arrays of α(σ), λ2(σ), ρ0(σ)
    alphas = np.empty(len(sigmas), dtype=float)
    lam2s  = np.empty(len(sigmas), dtype=float)
    rho0s  = np.empty(len(sigmas), dtype=float)

    for i, (s, Qs) in enumerate(zip(sigmas, Q_list)):
        lam2s[i]  = lambda2_of_Q(Qs)
        alphas[i] = alpha_from_h_frozen_backbone(h_pop, Qs, J=J,
                                                 mode=alpha_mode, q=alpha_q)
        rho0s[i]  = (deg - 1) * alphas[i] * lam2s[i]

    # Find first crossing of 1
    # We look for consecutive points where (ρ0-1) changes sign.
    diff = rho0s - 1.0
    sign = np.sign(diff)

    # indices where sign change occurs between i and i+1
    crossings = np.where(sign[:-1] * sign[1:] < 0)[0]

    sigma_star: Optional[float] = None
    if crossings.size > 0:
        i = int(crossings[0])  # first crossing
        s0, s1 = sigmas[i], sigmas[i+1]
        r0, r1 = rho0s[i], rho0s[i+1]
        # linear interpolation for ρ0(s)=1
        if r1 != r0:
            sigma_star = s0 + (1.0 - r0) * (s1 - s0) / (r1 - r0)
        else:
            sigma_star = 0.5 * (s0 + s1)
    else:
        # If no strict sign change, consider edge cases:
        # If entire curve is above 1, threshold lies to the right of last σ
        # If entire curve is below 1, threshold lies to the left of first σ
        sigma_star = None

    info = {
        "sigmas": sigmas,
        "alpha":  alphas,
        "lambda2": lam2s,
        "rho0":   rho0s,
        "crossings_idx": crossings
    }
    return sigma_star, info

def first_crossing_sigma(sigmas, rho_vals, target=1.0):
    """
    Given arrays sigmas (monotone) and rho_vals(σ), return the first σ
    where rho crosses 'target' by linear interpolation. Returns None if no crossing.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    r = np.asarray(rho_vals, dtype=float) - target
    sgn = np.sign(r)
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]   # sign change between i and i+1
    if idx.size == 0:
        return None
    i = int(idx[0])
    s0, s1 = sigmas[i], sigmas[i+1]
    r0, r1 = r[i], r[i+1]
    if r1 != r0:
        return s0 + (-r0) * (s1 - s0) / (r1 - r0)
    else:
        return 0.5*(s0 + s1)

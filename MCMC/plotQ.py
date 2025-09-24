import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# If you already have Q, comment out this example.
# This just builds a demo block‑Gaussian kernel.
from mlayer2 import create_mixing_Q, create_mixing_Q_band, apply_generalized_diagonals_with_cycle, create_mixing_Q_step        # replace with actual import

from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent            # folder of this file
sys.path.insert(0, str((HERE / ".." / "MCMC_neal").resolve()))
import lib

M, B, sigma = 100, 10, 1.1                     # total layers, blocks, std‑dev

if True:
    Q = create_mixing_Q(M, mtype="block", B=B, sigma=sigma)

if False:
    Q = create_mixing_Q_band(M, 3, 0.9)
    
if False:
    Q = create_mixing_Q_step(M, 2)
    
if False:
    Q = lib.create_mixing_Q_step(M, 2)

if False:
    Q = apply_generalized_diagonals_with_cycle(M, 2)
    Q = Q + 0.00000001
    Q = Q/np.max(Q)

# -------------------------------------------------

plt.figure(figsize=(6, 5))
plt.imshow(Q, aspect='equal')          # viridis colormap by default
plt.title(f"Mixing kernel Q  (M={Q.shape[0]})")
plt.xlabel("Layer index β")
plt.ylabel("Layer index α")
plt.colorbar(label=r"$Q_{\alpha\beta}$")
plt.tight_layout()
plt.show()

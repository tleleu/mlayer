import numpy as np
import os
import pandas as pd

# ---------- saving ----------------------------------------------------- #

def save_plot_data(B, Ny, mean_E, err_E, mean_S, err_S, out_dir="res"):
    # Build a tidy two‑column DataFrame for each curve …
    df1 = pd.DataFrame({"y": Ny, "mean_E": mean_E, "err_E": err_E})
    df2 = pd.DataFrame({"y": Ny, "mean_S": mean_S, "err_S": err_S})
    df3 = pd.DataFrame({"U": mean_E, "mean_S": mean_S, "err_S": err_S})

    # … then concatenate them side‑by‑side using a multi‑level column index.
    tidy = pd.concat(
        {"|E_DCM| vs y": df1, "Σ vs y": df2, "Σ vs U": df3},
        axis=1
    )

    # Make sure the output folder exists.
    os.makedirs(out_dir, exist_ok=True)

    # Build the filename; adapt the format spec as you like.
    filename = f"data_B{B}.csv"
    filepath = os.path.join(out_dir, filename)

    # Write to disk.
    tidy.to_csv(filepath, index=False)
    print(f"Saved plot data → {filepath}")
    
def load_plot_data(B, directory="res"):
    path = os.path.join(directory, f"data_B{B}.csv")

    # Two‑level header: [curve label, field name]
    df = pd.read_csv(path, header=[0, 1])

    # Extract the columns in the same order you plotted them
    Ny     = df["|E_DCM| vs y"]["y"].to_numpy()
    mean_E = df["|E_DCM| vs y"]["mean_E"].to_numpy()
    err_E  = df["|E_DCM| vs y"]["err_E"].to_numpy()

    mean_S = df["Σ vs y"]["mean_S"].to_numpy()
    err_S  = df["Σ vs y"]["err_S"].to_numpy()

    return Ny, mean_E, err_E, mean_S, err_S
    
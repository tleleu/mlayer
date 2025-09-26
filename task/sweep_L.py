"""Sweep over different layer counts and plot residual energy trends."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from benchmark.runner import BenchmarkConfig, BenchmarkRunner


def sweep_layer_counts(
    *,
    L_values: Iterable[int] = (1, 2, 3, 4, 5),
    output_root: Path | None = None,
) -> None:
    """Run the benchmark for a range of ``L`` values and plot the results."""

    output_root = (
        output_root
        if output_root is not None
        else Path(__file__).resolve().parent / "results" / "sweep_L"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    best_residuals: list[float] = []
    best_errors: list[float] = []
    best_sigmas: list[float] = []
    recorded_L: list[int] = []

    for L in L_values:
        config = BenchmarkConfig(
            N0=100,
            L=int(L),
            Ml=(100,),
            parallel=False,
        )

        runner = BenchmarkRunner(config)
        output_dir = output_root / f"L_{L}"
        result = runner.run(output_dir=output_dir, save=False, plot=True)

        mean_residuals = result.mean_res[:, 0]
        ci95_residuals = result.ci95_res[:, 0]
        sigma_grid = result.sigmal

        best_index = int(np.nanargmin(mean_residuals))
        best_residuals.append(float(mean_residuals[best_index]))
        best_errors.append(float(ci95_residuals[best_index]))
        best_sigmas.append(float(sigma_grid[best_index]))
        recorded_L.append(int(L))

        plt.close("all")

    if recorded_L:
        plt.figure(figsize=(6, 4))
        y_values = np.clip(best_residuals, np.finfo(float).tiny, None)
        y_errors = np.clip(best_errors, 0.0, None)

        plt.errorbar(recorded_L, y_values, yerr=y_errors, marker="o")
        plt.yscale("log")
        plt.xlabel("L")
        plt.ylabel(r"best residual energy $\min_{\sigma} \langle e - e_0 \rangle$")
        plt.title("Minimum residual energy vs L")
        plt.grid(True, which="both", ls=":")

        for L, y, sigma in zip(recorded_L, y_values, best_sigmas):
            plt.annotate(
                fr"σ*={sigma:.2g}",
                (L, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            output_root / "best_residual_vs_L.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()


def main() -> None:
    sweep_layer_counts()


if __name__ == "__main__":
    main()

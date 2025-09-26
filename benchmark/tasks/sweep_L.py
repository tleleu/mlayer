"""Sweep over different layer counts and plot residual energy trends."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runner import BenchmarkConfig, BenchmarkRunner


DEFAULT_L_VALUES: Sequence[int] = (1, 2, 3, 4, 5)


def sweep_layer_counts(
    *,
    L_values: Iterable[int] = DEFAULT_L_VALUES,
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

        summary = np.column_stack(
            [
                np.asarray(recorded_L, dtype=int),
                np.asarray(best_sigmas, dtype=float),
                np.asarray(best_residuals, dtype=float),
                np.asarray(best_errors, dtype=float),
            ]
        )
        np.savetxt(
            output_root / "best_residual_vs_L.csv",
            summary,
            delimiter=",",
            header="L,sigma_best,residual_best,ci95_residual",
            comments="",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the benchmark sweep for several layer counts and collect "
            "residual statistics."
        )
    )
    parser.add_argument(
        "--L",
        "--layers",
        dest="L_values",
        metavar="L",
        type=int,
        nargs="+",
        help=(
            "Layer counts to evaluate. Defaults to the preset sequence "
            f"{list(DEFAULT_L_VALUES)}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where results should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    L_values = args.L_values or DEFAULT_L_VALUES
    sweep_layer_counts(L_values=L_values, output_root=args.output_root)


if __name__ == "__main__":
    main()

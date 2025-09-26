"""Run shift and skew benchmarks and plot residual energy trends."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from benchmark.runner import BenchmarkConfig, BenchmarkRunner


def _create_config(
    backend: str,
    shift: float,
    skew: float,
    M_values: Sequence[int] = (100, 200, 300),
) -> BenchmarkConfig:
    """Create a benchmark configuration with the desired backend and offsets."""

    sigma_values = tuple(float(x) for x in np.linspace(0.01, 0.1, 10))
    return BenchmarkConfig(
        N0=100,
        problem="bethe",
        L=2,
        Ml=tuple(int(M) for M in M_values),
        sigmal=sigma_values,
        reps=1,
        K=50,
        steps0=200,
        beta=1_000.0,
        typeperm="asym",
        mlayer_backend=backend,
        mixing_backend="mlayer_directional",
        shift=float(shift),
        skew=float(skew),
        seed0=42,
        parallel=False,
        parallel_workers=None,
        sa_backend="neal",
        sa_zero_temp_terminate=True,
    )


def _run_benchmark(config: BenchmarkConfig, output_dir: Path) -> Dict[str, np.ndarray]:
    """Execute the benchmark and persist the per-run results."""

    runner = BenchmarkRunner(config)
    cfg = runner.config
    result = runner.run(save=False, plot=False)

    Ml = np.asarray(result.Ml, dtype=int)
    sigmal = np.asarray(result.sigmal, dtype=float)
    mean_res = result.mean_res
    ci95_res = result.ci95_res
    mean_qavg = result.mean_qavg
    ci95_qavg = result.ci95_qavg
    residual_energy = result.residual_energy

    output_dir.mkdir(parents=True, exist_ok=True)
    Ml_label = "-".join(str(int(x)) for x in Ml)
    filename = output_dir / (
        "energy_"
        f"backend-{cfg.mlayer_backend}_M{Ml_label}_shift{float(cfg.shift):.3f}_skew{float(cfg.skew):.3f}.npz"
    )

    np.savez(
        filename,
        Ml=Ml,
        sigmal=sigmal,
        mean_res=mean_res,
        ci95_res=ci95_res,
        mean_qavg=mean_qavg,
        ci95_qavg=ci95_qavg,
        residual_energy=residual_energy,
        mlayer_backend=np.array(cfg.mlayer_backend),
        shift=float(cfg.shift),
        skew=float(cfg.skew),
        reps=cfg.reps,
    )

    return {
        "Ml": Ml,
        "sigmal": sigmal,
        "mean_res": mean_res,
        "ci95_res": ci95_res,
        "mean_qavg": mean_qavg,
        "ci95_qavg": ci95_qavg,
        "residual_energy": residual_energy,
        "output_file": filename,
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--M-values",
        dest="M_values",
        metavar="M",
        type=int,
        nargs="+",
        default=(100, 200, 300),
        help=(
            "Sequence of permanental sample counts to evaluate. "
            "Provide one or more integers."
        ),
    )
    parser.add_argument(
        "--reference-M",
        dest="reference_M",
        type=int,
        default=None,
        help=(
            "Sample count to use for the permanental_alt reference. "
            "Defaults to the largest value in --M-values."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.M_values:
        raise ValueError("At least one M value must be provided.")

    M_values = tuple(int(M) for M in args.M_values)
    M_values_tuple = tuple(M_values)
    reference_M = int(args.reference_M) if args.reference_M is not None else max(M_values)

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results" / "shift_skew"

    # Run the permanental_alt baseline.
    permanental_alt_config = _create_config(
        "permanental_alt",
        shift=0.0,
        skew=0.0,
        M_values=(reference_M,),
    )
    permanental_alt_result = _run_benchmark(permanental_alt_config, results_dir)

    # Containers for the permanental runs.
    permanental_cache: Dict[Tuple[Tuple[int, ...], float, float], Dict[str, np.ndarray]] = {}

    def run_permanental(shift: float, skew: float) -> Dict[str, np.ndarray]:
        key = (M_values_tuple, float(shift), float(skew))
        if key not in permanental_cache:
            config = _create_config(
                "permanental",
                shift=shift,
                skew=skew,
                M_values=M_values_tuple,
            )
            permanental_cache[key] = _run_benchmark(config, results_dir)
            print(
                "Completed permanental run: "
                f"M={M_values}, shift={shift:.3f}, skew={skew:.3f}, "
                f"residual={permanental_cache[key]['residual_energy']:.6f}"
            )
        return permanental_cache[key]

    skew_points = np.linspace(0.0, 0.9, 10)
    skew_mean_res = None
    skew_residuals = np.empty((len(skew_points), len(M_values)))
    for i_skew, skew in enumerate(skew_points):
        result = run_permanental(shift=0.0, skew=float(skew))
        if skew_mean_res is None:
            skew_mean_res = np.empty((len(skew_points), len(result["sigmal"]), len(M_values)))
        skew_mean_res[i_skew] = result["mean_res"]
        skew_residuals[i_skew] = result["mean_res"].min(axis=0)

    shift_points = np.linspace(0.0, 3.0, 10)
    shift_mean_res = None
    shift_residuals = np.empty((len(shift_points), len(M_values)))
    for i_shift, shift in enumerate(shift_points):
        result = run_permanental(shift=float(shift), skew=0.0)
        if shift_mean_res is None:
            shift_mean_res = np.empty((len(shift_points), len(result["sigmal"]), len(M_values)))
        shift_mean_res[i_shift] = result["mean_res"]
        shift_residuals[i_shift] = result["mean_res"].min(axis=0)

    sigma_values = run_permanental(shift=0.0, skew=0.0)["sigmal"]

    np.savez(
        results_dir / "permanental_shift_curve.npz",
        shifts=shift_points,
        residual_energy=shift_residuals,
        mean_residual=shift_mean_res,
        backend=np.array("permanental"),
        M_values=np.asarray(M_values, dtype=int),
        reference_backend=np.array("permanental_alt"),
        reference_residual=permanental_alt_result["residual_energy"],
        reference_M=np.array(reference_M, dtype=int),
    )

    np.savez(
        results_dir / "permanental_skew_curve.npz",
        skews=skew_points,
        residual_energy=skew_residuals,
        mean_residual=skew_mean_res,
        backend=np.array("permanental"),
        M_values=np.asarray(M_values, dtype=int),
        reference_backend=np.array("permanental_alt"),
        reference_residual=permanental_alt_result["residual_energy"],
        reference_M=np.array(reference_M, dtype=int),
    )

    plt.style.use("ggplot")

    fig_shift, ax_shift = plt.subplots(figsize=(6, 4))
    for iM, M in enumerate(M_values):
        ax_shift.plot(
            shift_points,
            shift_residuals[:, iM],
            marker="o",
            label=f"permanental (M={M})",
        )
    ax_shift.axhline(
        y=permanental_alt_result["residual_energy"],
        color="C1",
        linestyle="--",
        label=f"permanental_alt (M={reference_M})",
    )
    ax_shift.set_xlabel("Shift")
    ax_shift.set_ylabel("Residual energy")
    ax_shift.set_title("Residual energy vs. shift")
    ax_shift.legend()
    fig_shift.tight_layout()
    fig_shift.savefig(results_dir / "residual_energy_vs_shift.png", dpi=300)
    plt.close(fig_shift)

    fig_skew, ax_skew = plt.subplots(figsize=(6, 4))
    for iM, M in enumerate(M_values):
        ax_skew.plot(
            skew_points,
            skew_residuals[:, iM],
            marker="o",
            label=f"permanental (M={M})",
        )
    ax_skew.axhline(
        y=permanental_alt_result["residual_energy"],
        color="C1",
        linestyle="--",
        label=f"permanental_alt (M={reference_M})",
    )
    ax_skew.set_xlabel("Skew")
    ax_skew.set_ylabel("Residual energy")
    ax_skew.set_title("Residual energy vs. skew")
    ax_skew.legend()
    fig_skew.tight_layout()
    fig_skew.savefig(results_dir / "residual_energy_vs_skew.png", dpi=300)
    plt.close(fig_skew)

    if shift_mean_res is None or skew_mean_res is None:
        raise RuntimeError("Expected shift and skew mean residual arrays to be populated")

    sigma_values = np.asarray(sigma_values, dtype=float)

    for iM, M in enumerate(M_values):
        fig_sigma_shift, ax_sigma_shift = plt.subplots(figsize=(6, 4))
        for i_shift, shift in enumerate(shift_points):
            ax_sigma_shift.plot(
                sigma_values,
                shift_mean_res[i_shift, :, iM],
                marker="o",
                label=f"shift={shift:.2f}",
            )
        ax_sigma_shift.set_xlabel(r"$\sigma$")
        ax_sigma_shift.set_ylabel("Residual energy")
        ax_sigma_shift.set_title(f"Residual vs. $\\sigma$ for shifts (M={M})")
        ax_sigma_shift.legend(ncol=2, fontsize="small")
        fig_sigma_shift.tight_layout()
        fig_sigma_shift.savefig(
            results_dir / f"residual_vs_sigma_shift_M{int(M)}.png",
            dpi=300,
        )
        plt.close(fig_sigma_shift)

        fig_sigma_skew, ax_sigma_skew = plt.subplots(figsize=(6, 4))
        for i_skew, skew in enumerate(skew_points):
            ax_sigma_skew.plot(
                sigma_values,
                skew_mean_res[i_skew, :, iM],
                marker="o",
                label=f"skew={skew:.2f}",
            )
        ax_sigma_skew.set_xlabel(r"$\sigma$")
        ax_sigma_skew.set_ylabel("Residual energy")
        ax_sigma_skew.set_title(f"Residual vs. $\\sigma$ for skews (M={M})")
        ax_sigma_skew.legend(ncol=2, fontsize="small")
        fig_sigma_skew.tight_layout()
        fig_sigma_skew.savefig(
            results_dir / f"residual_vs_sigma_skew_M{int(M)}.png",
            dpi=300,
        )
        plt.close(fig_sigma_skew)


if __name__ == "__main__":
    main()

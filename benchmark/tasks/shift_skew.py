"""Run shift and skew benchmarks and plot residual energy trends."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from benchmark.runner import BenchmarkConfig, BenchmarkRunner


def _create_config(backend: str, shift: float, skew: float, M: int) -> BenchmarkConfig:
    """Create a benchmark configuration with the desired backend and offsets."""

    sigma_values = tuple(float(x) for x in np.linspace(0.01, 0.1, 10))
    return BenchmarkConfig(
        N0=100,
        problem="bethe",
        L=2,
        Ml=(100, 150, 200),
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
        parallel=True,
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
        "mlayer_backend": np.array(cfg.mlayer_backend),
    }


def _min_residual_by_M(result: Dict[str, np.ndarray]) -> Dict[int, float]:
    """Return the minimum residual energy for each permanental sample count."""

    Ml = np.asarray(result["Ml"], dtype=int)
    mean_res = np.asarray(result["mean_res"], dtype=float)
    min_residuals = mean_res.min(axis=0)
    return {int(M): float(min_residuals[i]) for i, M in enumerate(Ml)}


def _plot_residual_vs_sigma(
    result: Dict[str, np.ndarray],
    backend: str,
    output_dir: Path,
    plotted_keys: Set[Tuple[str, float, float]],
    *,
    shift: float,
    skew: float,
) -> None:
    """Create a residual energy vs. sigma plot for the given configuration."""

    key = (backend, float(shift), float(skew))
    if key in plotted_keys:
        return
    plotted_keys.add(key)

    sigmal = np.asarray(result["sigmal"], dtype=float)
    mean_res = np.asarray(result["mean_res"], dtype=float)
    Ml = np.asarray(result["Ml"], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 4))
    for iM, M in enumerate(Ml):
        ax.plot(
            sigmal,
            mean_res[:, iM],
            marker="o",
            label=f"{backend} (M={int(M)})",
        )

    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Residual energy")
    ax.set_title(
        "Residual energy vs. sigma\n"
        f"shift={float(shift):.3f}, skew={float(skew):.3f}"
    )
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()

    filename = (
        f"residual_vs_sigma_{backend}_shift{float(shift):.3f}_skew{float(skew):.3f}.pdf"
    )
    fig.savefig(output_dir / filename)
    plt.close(fig)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--M-values",
        dest="M_values",
        metavar="M",
        type=int,
        nargs="+",
        default=(100, 150, 200),
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
    reference_M = int(args.reference_M) if args.reference_M is not None else max(M_values)

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results" / "shift_skew"

    # Run the permanental_alt baseline.
    plt.style.use("ggplot")

    permanental_alt_config = _create_config("permanental_alt", shift=0.0, skew=0.0, M=reference_M)
    permanental_alt_result = _run_benchmark(permanental_alt_config, results_dir)

    permanental_alt_min_residuals = _min_residual_by_M(permanental_alt_result)
    if reference_M not in permanental_alt_min_residuals:
        available = ", ".join(str(M) for M in sorted(permanental_alt_min_residuals))
        raise ValueError(
            f"Reference M={reference_M} is not available in the benchmark results. "
            f"Available M values: {available}"
        )
    permanental_alt_reference_residual = permanental_alt_min_residuals[reference_M]

    # Containers for the permanental runs.
    permanental_cache: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}
    sigma_plot_dir = results_dir / "sigma_curves"
    sigma_plot_dir.mkdir(parents=True, exist_ok=True)
    plotted_sigma_curves: Set[Tuple[str, float, float]] = set()

    def run_permanental(shift: float, skew: float) -> Dict[str, np.ndarray]:
        key = (float(shift), float(skew))
        if key not in permanental_cache:
            config = _create_config("permanental", shift=shift, skew=skew, M=reference_M)
            permanental_cache[key] = _run_benchmark(config, results_dir)
            min_residuals = _min_residual_by_M(permanental_cache[key])
            summary = ", ".join(
                f"M={M}: {residual:.6f}" for M, residual in sorted(min_residuals.items())
            )
            print(
                f"Completed permanental run: shift={shift:.3f}, skew={skew:.3f}, "
                f"residuals=({summary})"
            )
        return permanental_cache[key]

    backend_alt_value = permanental_alt_result["mlayer_backend"]
    backend_alt = (
        str(backend_alt_value.item())
        if isinstance(backend_alt_value, np.ndarray)
        else str(backend_alt_value)
    )
    _plot_residual_vs_sigma(
        permanental_alt_result,
        backend_alt,
        sigma_plot_dir,
        plotted_sigma_curves,
        shift=float(permanental_alt_config.shift),
        skew=float(permanental_alt_config.skew),
    )

    skew_points = np.linspace(0.0, 0.9, 10)
    skew_residuals = np.empty((len(M_values), len(skew_points)))
    for j, skew in enumerate(skew_points):
        result = run_permanental(shift=0.0, skew=float(skew))
        _plot_residual_vs_sigma(
            result,
            "permanental",
            sigma_plot_dir,
            plotted_sigma_curves,
            shift=0.0,
            skew=float(skew),
        )
        min_residuals = _min_residual_by_M(result)
        for iM, M in enumerate(M_values):
            try:
                skew_residuals[iM, j] = min_residuals[int(M)]
            except KeyError as exc:
                available = ", ".join(str(val) for val in sorted(min_residuals))
                raise ValueError(
                    f"Requested M={M} is not available for skew={float(skew):.3f}. "
                    f"Available M values: {available}"
                ) from exc

    shift_points = np.linspace(0.0, 3.0, 10)
    shift_residuals = np.empty((len(M_values), len(shift_points)))
    for j, shift in enumerate(shift_points):
        result = run_permanental(shift=float(shift), skew=0.0)
        _plot_residual_vs_sigma(
            result,
            "permanental",
            sigma_plot_dir,
            plotted_sigma_curves,
            shift=float(shift),
            skew=0.0,
        )
        min_residuals = _min_residual_by_M(result)
        for iM, M in enumerate(M_values):
            try:
                shift_residuals[iM, j] = min_residuals[int(M)]
            except KeyError as exc:
                available = ", ".join(str(val) for val in sorted(min_residuals))
                raise ValueError(
                    f"Requested M={M} is not available for shift={float(shift):.3f}. "
                    f"Available M values: {available}"
                ) from exc

    np.savez(
        results_dir / "permanental_shift_curve.npz",
        shifts=shift_points,
        residual_energy=shift_residuals,
        backend=np.array("permanental"),
        M_values=np.asarray(M_values, dtype=int),
        reference_backend=np.array("permanental_alt"),
        reference_residual=permanental_alt_reference_residual,
        reference_M=np.array(reference_M, dtype=int),
    )

    np.savez(
        results_dir / "permanental_skew_curve.npz",
        skews=skew_points,
        residual_energy=skew_residuals,
        backend=np.array("permanental"),
        M_values=np.asarray(M_values, dtype=int),
        reference_backend=np.array("permanental_alt"),
        reference_residual=permanental_alt_reference_residual,
        reference_M=np.array(reference_M, dtype=int),
    )

    fig_shift, ax_shift = plt.subplots(figsize=(6, 4))
    for iM, M in enumerate(M_values):
        ax_shift.plot(
            shift_points,
            shift_residuals[iM],
            marker="o",
            label=f"permanental (M={M})",
        )
    ax_shift.axhline(
        y=permanental_alt_reference_residual,
        color="C1",
        linestyle="--",
        label=f"permanental_alt (M={reference_M})",
    )
    ax_shift.set_xlabel("Shift")
    ax_shift.set_ylabel("Residual energy")
    ax_shift.set_title("Residual energy vs. shift")
    ax_shift.set_yscale('log')
    ax_shift.legend()
    fig_shift.tight_layout()
    fig_shift.savefig(results_dir / "residual_energy_vs_shift.pdf")
    plt.close(fig_shift)

    fig_skew, ax_skew = plt.subplots(figsize=(6, 4))
    for iM, M in enumerate(M_values):
        ax_skew.plot(
            skew_points,
            skew_residuals[iM],
            marker="o",
            label=f"permanental (M={M})",
        )
    ax_skew.axhline(
        y=permanental_alt_reference_residual,
        color="C1",
        linestyle="--",
        label=f"permanental_alt (M={reference_M})",
    )
    ax_skew.set_xlabel("Skew")
    ax_skew.set_ylabel("Residual energy")
    ax_skew.set_title("Residual energy vs. skew")
    ax_skew.set_yscale('log')
    ax_skew.legend()
    fig_skew.tight_layout()
    fig_skew.savefig(results_dir / "residual_energy_vs_skew.pdf")
    plt.close(fig_skew)


if __name__ == "__main__":
    main()

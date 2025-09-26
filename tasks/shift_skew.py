"""Run shift and skew benchmarks and plot residual energy trends."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmark.runner import BenchmarkConfig, BenchmarkRunner


def _create_config(backend: str, shift: float, skew: float) -> BenchmarkConfig:
    """Create a benchmark configuration with the desired backend and offsets."""

    sigma_values = tuple(float(x) for x in np.linspace(0.01, 0.1, 10))
    return BenchmarkConfig(
        N0=100,
        problem="bethe",
        L=2,
        Ml=(400,),
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
    filename = output_dir / (
        "energy_"
        f"backend-{cfg.mlayer_backend}_shift{float(cfg.shift):.3f}_skew{float(cfg.skew):.3f}.npz"
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


def main() -> None:
    results_dir = Path(__file__).resolve().parent.parent / "results" / "shift_skew"

    # Run the permanental_alt baseline.
    permanental_alt_config = _create_config("permanental_alt", shift=0.0, skew=0.0)
    permanental_alt_result = _run_benchmark(permanental_alt_config, results_dir)

    # Containers for the permanental runs.
    permanental_cache: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}

    def run_permanental(shift: float, skew: float) -> Dict[str, np.ndarray]:
        key = (float(shift), float(skew))
        if key not in permanental_cache:
            config = _create_config("permanental", shift=shift, skew=skew)
            permanental_cache[key] = _run_benchmark(config, results_dir)
            print(
                f"Completed permanental run: shift={shift:.3f}, skew={skew:.3f}, "
                f"residual={permanental_cache[key]['residual_energy']:.6f}"
            )
        return permanental_cache[key]

    skew_points = np.linspace(0.0, 0.9, 10)
    skew_residuals = np.array(
        [run_permanental(shift=0.0, skew=float(skew))["residual_energy"] for skew in skew_points]
    )

    shift_points = np.linspace(0.0, 3.0, 10)
    shift_residuals = np.array(
        [run_permanental(shift=float(shift), skew=0.0)["residual_energy"] for shift in shift_points]
    )

    np.savez(
        results_dir / "permanental_shift_curve.npz",
        shifts=shift_points,
        residual_energy=shift_residuals,
        backend=np.array("permanental"),
        reference_backend=np.array("permanental_alt"),
        reference_residual=permanental_alt_result["residual_energy"],
    )

    np.savez(
        results_dir / "permanental_skew_curve.npz",
        skews=skew_points,
        residual_energy=skew_residuals,
        backend=np.array("permanental"),
        reference_backend=np.array("permanental_alt"),
        reference_residual=permanental_alt_result["residual_energy"],
    )

    plt.style.use("ggplot")

    fig_shift, ax_shift = plt.subplots(figsize=(6, 4))
    ax_shift.plot(shift_points, shift_residuals, marker="o", label="permanental")
    ax_shift.axhline(
        y=permanental_alt_result["residual_energy"],
        color="C1",
        linestyle="--",
        label="permanental_alt",
    )
    ax_shift.set_xlabel("Shift")
    ax_shift.set_ylabel("Residual energy")
    ax_shift.set_title("Residual energy vs. shift")
    ax_shift.legend()
    fig_shift.tight_layout()
    fig_shift.savefig(results_dir / "residual_energy_vs_shift.png", dpi=300)
    plt.close(fig_shift)

    fig_skew, ax_skew = plt.subplots(figsize=(6, 4))
    ax_skew.plot(skew_points, skew_residuals, marker="o", label="permanental")
    ax_skew.axhline(
        y=permanental_alt_result["residual_energy"],
        color="C1",
        linestyle="--",
        label="permanental_alt",
    )
    ax_skew.set_xlabel("Skew")
    ax_skew.set_ylabel("Residual energy")
    ax_skew.set_title("Residual energy vs. skew")
    ax_skew.legend()
    fig_skew.tight_layout()
    fig_skew.savefig(results_dir / "residual_energy_vs_skew.png", dpi=300)
    plt.close(fig_skew)


if __name__ == "__main__":
    main()

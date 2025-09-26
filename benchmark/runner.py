"""Reference benchmark driver using the modular components."""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, cast

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

if __package__ in {None, ""}:
    #sys.path.append(str(Path(__file__).resolve().parent.parent))
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(repo_root))
    sys.path.append(str(repo_root / "MCMC_neal"))                    # temp

from benchmark.mixing import MixingMatrix
from benchmark.mlayer_transform import MLayerConstructor, create_mlayer_constructor
from benchmark.observables import ObservableEvaluator
from benchmark.plotting import BenchmarkPlotter, PlotConfig
from benchmark.problems import BetheProblem, IsingProblem
from benchmark.algorithms.simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingRunner,
)


@dataclass
class BenchmarkConfig:
    N0: int = 100
    problem: str = "bethe"
    L: int = 2
    Ml: Sequence[int] = (20, 40, 100)
    sigmal: Sequence[float] = tuple(np.linspace(0.01, 0.1, 10))
    reps: int = 1
    K: int = 50
    steps0: int = 200
    beta: float = 1_000.0
    typeperm: str = "asym"
    mlayer_backend: str = "permanental"
    #mlayer_backend: str = "permanental_alt"                 # induces drift 
    mixing_backend: str = "mlayer_directional"
    shift: float = 0.0
    skew: float = 0.7
    seed0: int = 42
    parallel: bool = True
    parallel_workers: Optional[int] = None
    sa_backend: str = "neal"
    #sa_backend: str = "optimized"
    sa_zero_temp_terminate: bool = True


@dataclass
class BenchmarkResult:
    """Container for raw and aggregated benchmark statistics."""

    Ml: np.ndarray
    sigmal: np.ndarray
    energy: np.ndarray
    qavg: np.ndarray
    mean_res: np.ndarray
    ci95_res: np.ndarray
    mean_qavg: np.ndarray
    ci95_qavg: np.ndarray
    residual_energy: float
    output_file: Optional[Path] = None


@dataclass
class BenchmarkDefinition:
    """Holds the constructors required to assemble a benchmark run."""

    problem_constructor: Callable[["BenchmarkConfig"], IsingProblem]
    mixing_constructor: Callable[["BenchmarkConfig"], MixingMatrix]
    mlayer_constructor: Callable[["BenchmarkConfig"], MLayerConstructor]
    observable_constructor: Callable[["BenchmarkConfig"], ObservableEvaluator]
    solver_constructor: Callable[["BenchmarkConfig"], SimulatedAnnealingRunner]


BENCHMARK_DEFINITIONS: Dict[str, BenchmarkDefinition] = {
    "bethe": BenchmarkDefinition(
        problem_constructor=lambda config: BetheProblem(config.N0, degree=2),
        mixing_constructor=lambda config: MixingMatrix(
            config.mixing_backend,
            L=config.L,
            shift=config.shift,
            skew=config.skew,
        ),
        mlayer_constructor=lambda config: create_mlayer_constructor(
            config.mlayer_backend
        ),
        observable_constructor=lambda _: ObservableEvaluator(),
        solver_constructor=lambda config: SimulatedAnnealingRunner(
            SimulatedAnnealingConfig(
                steps=config.steps0,
                K=config.K,
                beta=config.beta,
                code=config.sa_backend,
                zero_temp_terminate=config.sa_zero_temp_terminate,
            )
        ),
    )
}


_WORKER_CONTEXT: dict[str, object] = {}


def _init_parallel_worker(config: BenchmarkConfig) -> None:
    """Initialise heavy benchmark components inside a worker process."""

    global _WORKER_CONTEXT

    mixing_matrix = MixingMatrix(
        config.mixing_backend,
        L=config.L,
        shift=config.shift,
        skew=config.skew,
    )
    mlayer_constructor = create_mlayer_constructor(config.mlayer_backend)
    solver_config = SimulatedAnnealingConfig(
        steps=config.steps0,
        K=config.K,
        beta=config.beta,
        code=config.sa_backend,
        zero_temp_terminate=config.sa_zero_temp_terminate,
    )
    solver = SimulatedAnnealingRunner(solver_config)
    observable = ObservableEvaluator()

    _WORKER_CONTEXT = {
        "config": config,
        "mixing_matrix": mixing_matrix,
        "mlayer_constructor": mlayer_constructor,
        "solver": solver,
        "observable": observable,
    }


def _evaluate_sigma_parallel(
    args: tuple[
        np.ndarray | sp.spmatrix,
        int,
        int,
        int,
        float,
    ]
) -> tuple[int, float, float, float]:
    """Evaluate a single ``sigma`` value inside a worker process."""

    if not _WORKER_CONTEXT:
        raise RuntimeError("Parallel worker context has not been initialised")

    J0, M, seed_base, i_sigma, sigma = args

    config = cast(BenchmarkConfig, _WORKER_CONTEXT["config"])
    mixing_matrix = cast(MixingMatrix, _WORKER_CONTEXT["mixing_matrix"])
    mlayer_constructor = cast(MLayerConstructor, _WORKER_CONTEXT["mlayer_constructor"])
    solver = cast(SimulatedAnnealingRunner, _WORKER_CONTEXT["solver"])
    observable = cast(ObservableEvaluator, _WORKER_CONTEXT["observable"])

    Q = mixing_matrix(int(M), float(sigma), int(i_sigma))
    J = mlayer_constructor(J0, int(M), Q, typeperm=config.typeperm)
    spins = solver.run(
        J,
        seed=int(seed_base + int(i_sigma)),
        steps=int(config.steps0 * int(M)),
    )
    observables = observable.compute(spins, J0, int(M))

    return (
        int(i_sigma),
        float(observables.energy_mean),
        float(observables.q_average),
        float(observables.energy_min),
    )


class BenchmarkRunner:

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        try:
            self.definition = BENCHMARK_DEFINITIONS[config.problem.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown benchmark problem '{config.problem}'") from exc

        self.problem_constructor = self.definition.problem_constructor
        self.mixing_matrix = self.definition.mixing_constructor(config)
        self.mlayer_transform = self.definition.mlayer_constructor(config)
        self.observable = self.definition.observable_constructor(config)
        self.solver = self.definition.solver_constructor(config)

    def run(
        self,
        output_dir: Path | None = None,
        *,
        save: bool = True,
        plot: bool = True,
        filename: str | None = None,
    ) -> BenchmarkResult:
        """Execute the benchmark and optionally persist or plot the results."""

        cfg = self.config
        Ml = np.asarray(cfg.Ml, dtype=int)
        sigmal = np.asarray(cfg.sigmal, dtype=float)

        Emean = np.zeros((len(sigmal), len(Ml), cfg.reps))
        Qavg = np.zeros_like(Emean)

        sa = self.solver

        executor: ProcessPoolExecutor | None = None
        if cfg.parallel:
            executor = ProcessPoolExecutor(
                max_workers=cfg.parallel_workers,
                initializer=_init_parallel_worker,
                initargs=(cfg,),
            )

        try:
            for r in range(cfg.reps):
                problem = self.problem_constructor(cfg)
                J0 = problem.build_couplings()
                e0_r = np.inf

                for iM, M in enumerate(tqdm(Ml, desc="M sweep")):
                    seed_base = int(cfg.seed0 + 10_000 * r + 1_000 * iM)

                    def evaluate_sigma(
                        item: tuple[int, float]
                    ) -> tuple[int, float, float, float]:
                        i_sigma, sigma = item
                        Q = self.mixing_matrix(int(M), float(sigma), i_sigma)

                        J = self.mlayer_transform(J0, int(M), Q, cfg.typeperm)

                        spins = sa.run(
                            J,
                            seed=seed_base + i_sigma,
                            steps=cfg.steps0 * int(M),
                        )
                        observables = self.observable.compute(spins, J0, int(M))

                        return (
                            i_sigma,
                            float(observables.energy_mean),
                            float(observables.q_average),
                            float(observables.energy_min),
                        )

                    sigma_iter = list(enumerate(sigmal))
                    if executor is not None:
                        task_iter = (
                            (J0, int(M), seed_base, int(i_sigma), float(sigma))
                            for i_sigma, sigma in sigma_iter
                        )
                        results = list(
                            executor.map(_evaluate_sigma_parallel, task_iter)
                        )
                    else:
                        results = [evaluate_sigma(item) for item in sigma_iter]

                    for i_sigma, energy_mean, q_average, energy_min in results:
                        Emean[i_sigma, iM, r] = energy_mean
                        Qavg[i_sigma, iM, r] = q_average
                        if energy_min < e0_r:
                            e0_r = energy_min

                Emean[:, :, r] -= e0_r
        finally:
            if executor is not None:
                executor.shutdown()

        mean_res = Emean.mean(axis=2)
        ci95_res = 1.96 * Emean.std(axis=2) / np.sqrt(cfg.reps)
        mean_qavg = Qavg.mean(axis=2)
        ci95_qavg = 1.96 * Qavg.std(axis=2) / np.sqrt(cfg.reps)
        residual_energy = float(mean_res.min())

        result = BenchmarkResult(
            Ml=Ml,
            sigmal=sigmal,
            energy=Emean,
            qavg=Qavg,
            mean_res=mean_res,
            ci95_res=ci95_res,
            mean_qavg=mean_qavg,
            ci95_qavg=ci95_qavg,
            residual_energy=residual_energy,
        )

        target_dir: Path | None = None
        if save or plot:
            target_dir = (
                output_dir
                if output_dir is not None
                else Path(__file__).resolve().parent / "results" / "runner"
            )
            target_dir.mkdir(parents=True, exist_ok=True)

        if save:
            assert target_dir is not None
            default_filename = (
                "energy_"
                f"N{cfg.N0}_reps{cfg.reps}_K{cfg.K}_steps{cfg.steps0}_L{cfg.L}_"
                f"backend-{cfg.sa_backend}_shift{cfg.shift}_skew{cfg.skew}.npz"
            )
            output_name = filename or default_filename
            output_path = target_dir / output_name
            np.savez(
                output_path,
                sigmal=sigmal,
                mean_res=mean_res,
                ci95_res=ci95_res,
                mean_qavg=mean_qavg,
                ci95_qavg=ci95_qavg,
                sa_backend=cfg.sa_backend,
                shift=cfg.shift,
                skew=cfg.skew,
            )
            result.output_file = output_path

        if plot:
            assert target_dir is not None
            plotter = BenchmarkPlotter(PlotConfig(show_residual=True))
            plot_title = (
                fr"$N={cfg.N0}$, $reps={cfg.reps}$, $K={cfg.K}$, "
                fr"$steps0={cfg.steps0}$, $L={cfg.L}$"
            )
            plotter.plot_energy_summary(
                sigmal,
                Ml,
                mean_res,
                ci95_res,
                mean_qavg,
                ci95_qavg,
                title=plot_title,
                output_dir=target_dir,
            )
            plotter.plot_min_residual(
                Ml,
                mean_res,
                ci95_res,
                sigmal,
                output_dir=target_dir,
            )

        return result


__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkDefinition",
    "BenchmarkResult",
]


def main() -> None:
    """Run the benchmark with the default configuration."""

    runner = BenchmarkRunner(BenchmarkConfig())
    runner.run()


if __name__ == "__main__":
    main()

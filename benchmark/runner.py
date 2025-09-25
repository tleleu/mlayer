"""Reference benchmark driver using the modular components."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
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
    sigmal: Sequence[float] = tuple(np.linspace(0.01, 0.3, 10))
    reps: int = 1
    K: int = 50
    steps0: int = 200
    beta: float = 1_000.0
    typeperm: str = "asym"
    #mlayer_backend: str = "permanental"
    mlayer_backend: str = "permanental_alt"
    mixing_backend: str = "mlayer_directional"
    shift: float = 0.0
    skew: float = 0.0
    seed0: int = 42
    parallel: bool = False


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
            )
        ),
    )
}


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

    def run(self) -> None:
        cfg = self.config
        Ml = np.asarray(cfg.Ml, dtype=int)
        sigmal = np.asarray(cfg.sigmal, dtype=float)

        Emean = np.zeros((len(sigmal), len(Ml), cfg.reps))
        Qavg = np.zeros_like(Emean)

        sa = self.solver

        for r in range(cfg.reps):
            problem = self.problem_constructor(cfg)
            J0_dense = problem.build_dense()
            e0_r = np.inf

            for iM, M in enumerate(tqdm(Ml, desc="M sweep")):
                seed_base = int(cfg.seed0 + 10_000 * r + 1_000 * iM)

                for i_sigma, sigma in enumerate(sigmal):
                    Q = self.mixing_matrix(int(M), float(sigma), i_sigma)

                    J = self.mlayer_transform(J0_dense, int(M), Q, cfg.typeperm)

                    spins = sa.run(
                        J,
                        seed=seed_base + i_sigma,
                        steps=cfg.steps0 * int(M),
                    )
                    observables = self.observable.compute(spins, J0_dense, int(M))

                    Emean[i_sigma, iM, r] = observables.energy_mean
                    Qavg[i_sigma, iM, r] = observables.q_average
                    if observables.energy_min < e0_r:
                        e0_r = observables.energy_min

            Emean[:, :, r] -= e0_r

        mean_res = Emean.mean(axis=2)
        ci95_res = 1.96 * Emean.std(axis=2) / np.sqrt(cfg.reps)
        mean_qavg = Qavg.mean(axis=2)
        ci95_qavg = 1.96 * Qavg.std(axis=2) / np.sqrt(cfg.reps)

        folder = Path("results/energy")
        folder.mkdir(parents=True, exist_ok=True)
        filename = folder / f"energy_N{cfg.N0}_reps{cfg.reps}_K{cfg.K}_steps{cfg.steps0}_L{cfg.L}.npz"
        np.savez(
            filename,
            sigmal=sigmal,
            mean_res=mean_res,
            ci95_res=ci95_res,
            mean_qavg=mean_qavg,
            ci95_qavg=ci95_qavg,
        )

        plotter = BenchmarkPlotter(PlotConfig(show_residual=True))
        plot_title = (
            fr"$N={cfg.N0}$, $reps={cfg.reps}$, $K={cfg.K}$, "
            fr"$steps0={cfg.steps0}$, $L={cfg.L}$"
        )
        plotter.plot_energy_summary(sigmal, Ml, mean_res, ci95_res, mean_qavg, ci95_qavg, title=plot_title)
        plotter.plot_min_residual(Ml, mean_res, ci95_res, sigmal)


__all__ = ["BenchmarkRunner", "BenchmarkConfig", "BenchmarkDefinition"]


def main() -> None:
    """Run the benchmark with the default configuration."""

    runner = BenchmarkRunner(BenchmarkConfig())
    runner.run()


if __name__ == "__main__":
    main()

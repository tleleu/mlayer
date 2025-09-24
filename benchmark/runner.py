"""Reference benchmark driver using the modular components."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from tqdm import tqdm

from .mixing import MixingMatrixBackend, MixingMatrixFactory, MixingMatrixRequest
from .mlayer_transform import MLayerTransformRequest, MLayerTransformer
from .observables import ObservableEvaluator
from .plotting import BenchmarkPlotter, PlotConfig
from .problems.bethe import BetheProblem
from .algorithms.simulated_annealing import SimulatedAnnealingConfig, SimulatedAnnealingRunner


@dataclass
class BenchmarkConfig:
    N0: int = 100
    L: int = 2
    Ml: Sequence[int] = (20, 40, 100)
    SigmaL: Sequence[float] = tuple(np.linspace(0.01, 0.10, 10))
    reps: int = 1
    K: int = 50
    steps0: int = 200
    beta: float = 1_000.0
    typeperm: str = "asym"
    mlayer_backend: str = "mlayer2"
    mixing_backend: MixingMatrixBackend = MixingMatrixBackend.MLAYER_DIRECTIONAL
    shift: float = 0.0
    skew: float = 0.2
    seed0: int = 42
    parallel: bool = False


class BenchmarkRunner:
    """Coordinates the pieces required to reproduce ``energy2.py``."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.mixing_factory = MixingMatrixFactory()
        self.transformer = MLayerTransformer()
        self.observable = ObservableEvaluator()

    def run(self) -> None:
        cfg = self.config
        Ml = np.asarray(cfg.Ml, dtype=int)
        SigmaL = np.asarray(cfg.SigmaL, dtype=float)

        Emean = np.zeros((len(SigmaL), len(Ml), cfg.reps))
        Qavg = np.zeros_like(Emean)

        sa = SimulatedAnnealingRunner(
            SimulatedAnnealingConfig(
                steps=cfg.steps0,
                K=cfg.K,
                beta=cfg.beta,
            )
        )

        for r in range(cfg.reps):
            problem = BetheProblem(cfg.N0, degree=cfg.L - 1)
            J0_dense = problem.build_dense()
            e0_r = np.inf

            for iM, M in enumerate(tqdm(Ml, desc="M sweep")):
                seed_base = int(cfg.seed0 + 10_000 * r + 1_000 * iM)

                for i_sigma, sigma in enumerate(SigmaL):
                    mixing_request = MixingMatrixRequest(
                        backend=cfg.mixing_backend,
                        M=int(M),
                        sigma=float(sigma),
                        index=i_sigma,
                        L=cfg.L,
                        shift=cfg.shift,
                        skew=cfg.skew,
                    )
                    Q = self.mixing_factory.create(mixing_request)

                    transform_request = MLayerTransformRequest(
                        J0_dense=J0_dense,
                        M=int(M),
                        mixing_matrix=Q,
                        typeperm=cfg.typeperm,
                        backend=cfg.mlayer_backend,
                    )
                    J = self.transformer.transform(transform_request)

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

        folder = Path("energy")
        folder.mkdir(exist_ok=True)
        filename = folder / f"energy_N{cfg.N0}_reps{cfg.reps}_K{cfg.K}_steps{cfg.steps0}_L{cfg.L}.npz"
        np.savez(
            filename,
            SigmaL=SigmaL,
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
        plotter.plot_energy_summary(SigmaL, Ml, mean_res, ci95_res, mean_qavg, ci95_qavg, title=plot_title)
        plotter.plot_min_residual(Ml, mean_res, ci95_res, SigmaL)


__all__ = ["BenchmarkRunner", "BenchmarkConfig"]

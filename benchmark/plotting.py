"""Plotting helpers for the modular benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotConfig:
    show_gap: bool = False
    show_residual: bool = True


class BenchmarkPlotter:
    """Reproduces the figures from the legacy ``energy2.py`` script."""

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()

    def plot_energy_summary(
        self,
        sigma_grid: Sequence[float],
        M_values: Sequence[int],
        mean_res: np.ndarray,
        ci95_res: np.ndarray,
        mean_qavg: np.ndarray,
        ci95_qavg: np.ndarray,
        title: str = "",
        output_dir: str | Path = "energy",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(12, 4))

        # Residual energy panel
        plt.subplot(1, 2, 1)
        for iM, M in enumerate(M_values):
            xval = self._x_axis(sigma_grid, M)
            plt.errorbar(
                xval,
                mean_res[:, iM],
                yerr=ci95_res[:, iM],
                marker="o",
                label=f"M={M}",
            )
        plt.title("a")
        if self.config.show_gap:
            plt.xscale("log")
            plt.xlabel(r'spectral gap 1-$\lambda_2$')
        else:
            plt.xlabel(r'$d$')
        if self.config.show_residual:
            plt.ylabel(r'residual energy $\langle e-e_0\rangle$')
            plt.yscale('log')
        else:
            plt.ylabel(r'energy $\langle e\rangle$')
            plt.ylim(-1.3, -0.7)
        plt.grid(True, ls=':')
        plt.legend()

        # Overlap panel
        plt.subplot(1, 2, 2)
        for iM, M in enumerate(M_values):
            xval = self._x_axis(sigma_grid, M)
            plt.errorbar(
                xval,
                mean_qavg[:, iM],
                yerr=ci95_qavg[:, iM],
                marker='s',
                label=f"M={M}",
            )
        if self.config.show_gap:
            plt.yscale('log')
            plt.xlabel(r'spectral gap 1-$\lambda_2$')
        else:
            plt.xlabel(r'$d$')
        plt.ylabel(r'inter-layer overlap $\langle q_{\mathrm{avg}}\rangle$')
        plt.ylim(0, 1.02)
        plt.title('b')
        plt.grid(True, ls=':')
        plt.legend()
        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.savefig(output_dir / "energy.pdf", format="pdf", bbox_inches="tight")

    def plot_min_residual(
        self,
        M_values: Sequence[int],
        mean_res: np.ndarray,
        ci95_res: np.ndarray,
        sigma_grid: Sequence[float],
        output_dir: str | Path = "energy",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        min_idx = np.nanargmin(mean_res, axis=0)
        min_vals = mean_res[min_idx, np.arange(len(M_values))]
        min_errs = ci95_res[min_idx, np.arange(len(M_values))]
        min_sigmas = np.array(sigma_grid)[min_idx]

        plt.figure(figsize=(5, 4))
        plt.errorbar(M_values, np.clip(min_vals, np.finfo(float).tiny, None), yerr=min_errs, marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('M')
        plt.ylabel(r'$min_{\sigma} \langle e-e_0\rangle$')
        plt.title('Minimum residual vs M')
        plt.grid(True, which='both', ls=':')

        for x, y, s in zip(M_values, min_vals, min_sigmas):
            plt.annotate(fr'σ*={s:.2g}', (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)

        tiny = np.finfo(float).tiny
        x = np.asarray(M_values, dtype=float)
        y = np.clip(min_vals, tiny, None)
        yerr = np.clip(min_errs, tiny, None)
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)

        if np.any(mask):
            w = 1.0 / np.clip((yerr[mask] / y[mask]) ** 2, 1e-12, None)
            logx = np.log(x[mask])
            logy = np.log(y[mask])
            p, loga = np.polyfit(logx, logy, deg=1, w=w)
            a = np.exp(loga)

            M_fit = np.geomspace(x[mask].min(), x[mask].max(), 256)
            y_fit = a * M_fit ** p
            plt.plot(M_fit, y_fit, linestyle=':', color='red', label=fr'fit slope={p:.3f}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "min_residual_vs_M.pdf", format="pdf", bbox_inches="tight")

    def _x_axis(self, sigma_grid: Sequence[float], M: int) -> np.ndarray:
        xval = np.arange(len(sigma_grid)) + 1
        return np.asarray(xval, dtype=float)


__all__ = ["BenchmarkPlotter", "PlotConfig"]

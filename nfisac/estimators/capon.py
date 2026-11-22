"""B2: near-field Capon / MVDR on a (u, w) grid with diagonal loading."""
from __future__ import annotations

import time

import numpy as np

from nfisac.config import ArrayCfg
from nfisac.estimators.base import EstResult, sample_covariance, targets_from_uw
from nfisac.grids import PolarGrid
from nfisac.peaks import pick_peaks_2d
from nfisac.spectrum import GridSteering, capon_spectrum


class Capon2D:
    """Capon spectrum P = 1 / (a^H (R + delta I)^{-1} a); K largest local maxima."""

    name = "B2"

    def __init__(
        self,
        cfg: ArrayCfg,
        grid: PolarGrid,
        loading: float = 1e-3,
        keep_spectrum: bool = False,
    ) -> None:
        self.cfg = cfg
        self.grid = grid
        self.loading = loading
        self.keep_spectrum = keep_spectrum
        self._gs = GridSteering(cfg, grid)

    def fit(self, Y: np.ndarray, K: int | None = None, X: np.ndarray | None = None) -> EstResult:
        if K is None:
            raise ValueError("Capon2D requires K")
        t0 = time.perf_counter()
        R = sample_covariance(Y)
        P = capon_spectrum(self.cfg, R, self._gs, loading=self.loading)
        idx = pick_peaks_2d(P, K)
        u = self.grid.u[idx[:, 0]]
        w = self.grid.w[idx[:, 1]]
        extra: dict = {"K": K}
        if self.keep_spectrum:
            extra["spectrum"] = P
        return EstResult(
            targets=targets_from_uw(u, w), runtime_s=time.perf_counter() - t0, extra=extra
        )


__all__ = ["Capon2D"]

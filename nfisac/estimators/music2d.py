"""B1: 2D-MUSIC on a fine (u, w) grid, and B1-r: the same + local continuous refinement.

B1-r refines each peak separately by minimising the MUSIC null spectrum
a^H U_n U_n^H a with Nelder-Mead in cell-normalised (u, w).  It removes the grid floor
while keeping the MUSIC cost, so that the comparison with P1 (joint DML) isolates the
effect of the cost function rather than of the grid (KE_HOACH_4_TUAN.md §2.1).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from nfisac.config import ArrayCfg
from nfisac.estimators.base import EstResult, sample_covariance, targets_from_uw
from nfisac.geometry import steering, uw_to_rt
from nfisac.grids import PolarGrid
from nfisac.mdl import mdl_estimate_K
from nfisac.peaks import pick_peaks_2d
from nfisac.spectrum import GridSteering, music_null, music_spectrum, signal_subspace


def refine_music_peak(
    cfg: ArrayCfg,
    Us: np.ndarray,
    u0: float,
    w0: float,
    cell_u: float,
    cell_w: float,
    max_shift_cells: float = 1.0,
    mode: str = "exact",
) -> tuple[float, float]:
    """Continuous minimisation of the MUSIC null spectrum around (u0, w0).

    Optimisation variable x = (u / cell_u, w / cell_w).  If the optimum moves further than
    ``max_shift_cells`` (Chebyshev) from the start, or leaves the valid domain, the start
    point is returned unchanged.
    """
    x0 = np.array([u0 / cell_u, w0 / cell_w])

    def f(x: np.ndarray) -> float:
        u, w = x[0] * cell_u, x[1] * cell_w
        if not (-1.0 < u < 1.0) or w <= 0.0:
            return 1e6
        r, th = uw_to_rt(u, w)
        a = steering(cfg, float(r), float(th), mode=mode)[:, None]
        return float(music_null(Us, a)[0])

    step = 0.05
    simplex = np.array([x0, x0 + [step, 0.0], x0 + [0.0, step]])
    res = minimize(
        f,
        x0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-12, "maxiter": 400},
    )
    x = res.x
    if np.max(np.abs(x - x0)) > max_shift_cells or not np.isfinite(res.fun):
        return u0, w0
    return float(x[0] * cell_u), float(x[1] * cell_w)


class Music2D:
    """2D-MUSIC estimator (B1) with optional local refinement (B1-r).

    Parameters
    ----------
    cfg : ArrayCfg
    grid : PolarGrid
        Search grid (DECISIONS #4: oversampling 8 per cell on both axes).
    refine : bool
        If True, acts as B1-r.
    loading : float
        Diagonal loading factor for the sample covariance (relative to tr(R)/N).
    mode : str
        Steering model assumed by the estimator.
    keep_spectrum : bool
        Store the spectrum in ``EstResult.extra["spectrum"]`` (memory heavy in MC runs).
    """

    def __init__(
        self,
        cfg: ArrayCfg,
        grid: PolarGrid,
        refine: bool = False,
        loading: float = 0.0,
        mode: str = "exact",
        keep_spectrum: bool = False,
        min_sep: int = 1,
    ) -> None:
        self.cfg = cfg
        self.grid = grid
        self.refine = refine
        self.loading = loading
        self.mode = mode
        self.keep_spectrum = keep_spectrum
        self.min_sep = min_sep
        self.name = "B1-r" if refine else "B1"
        self._gs = GridSteering(cfg, grid, mode=mode)

    def fit(self, Y: np.ndarray, K: int | None = None, X: np.ndarray | None = None) -> EstResult:
        t0 = time.perf_counter()
        R = sample_covariance(Y, self.loading)
        evals_all = np.linalg.eigvalsh(R)[::-1]
        if K is None:
            K = max(1, mdl_estimate_K(evals_all, Y.shape[1]))
        Us, evals = signal_subspace(R, K)
        P = music_spectrum(self.cfg, Us, self._gs)
        idx = pick_peaks_2d(P, K, min_sep=self.min_sep)
        u = self.grid.u[idx[:, 0]]
        w = self.grid.w[idx[:, 1]]
        on_grid = targets_from_uw(u, w)
        if self.refine:
            uw = [
                refine_music_peak(
                    self.cfg, Us, ui, wi, self.grid.cell_u, self.grid.cell_w, mode=self.mode
                )
                for ui, wi in zip(u, w, strict=True)
            ]
            u = np.array([p[0] for p in uw])
            w = np.array([p[1] for p in uw])
        runtime = time.perf_counter() - t0
        # "on_grid": the B1 (un-refined) estimates, so one B1-r call also yields B1.
        extra: dict = {"K": K, "evals": evals, "grid_size": self.grid.size, "on_grid": on_grid}
        if self.keep_spectrum:
            extra["spectrum"] = P
        return EstResult(targets=targets_from_uw(u, w), runtime_s=runtime, extra=extra)


__all__ = ["Music2D", "refine_music_peak"]

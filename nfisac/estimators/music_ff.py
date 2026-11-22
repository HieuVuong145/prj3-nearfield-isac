"""B0: far-field 1D MUSIC (angle only).

Uses the planar-wave steering vector, so it cannot estimate range; returned targets have
r = NaN.  Serves to show that co-directional targets cannot be separated without the
near-field (range-dependent) phase.
"""
from __future__ import annotations

import time

import numpy as np

from nfisac.config import ArrayCfg
from nfisac.estimators.base import EstResult, sample_covariance, targets_from_uw
from nfisac.geometry import steering
from nfisac.mdl import mdl_estimate_K
from nfisac.peaks import pick_peaks_1d
from nfisac.spectrum import music_null, signal_subspace


class MusicFarField:
    """Far-field MUSIC over u = cos(theta) in [u_min, u_max] with step (2/N)/ov_u."""

    name = "B0"

    def __init__(
        self,
        cfg: ArrayCfg,
        theta_min: float,
        theta_max: float,
        ov_u: float = 8.0,
        loading: float = 0.0,
        keep_spectrum: bool = False,
    ) -> None:
        self.cfg = cfg
        n = int(np.ceil((np.cos(theta_min) - np.cos(theta_max)) / (2.0 / cfg.N / ov_u))) + 1
        self.u = np.linspace(np.cos(theta_max), np.cos(theta_min), n)
        self._A = steering(cfg, np.full(n, 1e9), np.arccos(self.u), mode="far")
        self.loading = loading
        self.keep_spectrum = keep_spectrum

    def fit(self, Y: np.ndarray, K: int | None = None, X: np.ndarray | None = None) -> EstResult:
        t0 = time.perf_counter()
        R = sample_covariance(Y, self.loading)
        if K is None:
            K = max(1, mdl_estimate_K(np.linalg.eigvalsh(R), Y.shape[1]))
        Us, evals = signal_subspace(R, K)
        P = 1.0 / np.maximum(music_null(Us, self._A), 1e-12)
        idx = pick_peaks_1d(P, K)
        u = self.u[idx]
        extra: dict = {"K": K, "evals": evals}
        if self.keep_spectrum:
            extra["spectrum"] = P
        return EstResult(
            targets=targets_from_uw(u, np.full(u.size, np.nan)),
            runtime_s=time.perf_counter() - t0,
            extra=extra,
        )


__all__ = ["MusicFarField"]

"""Polar-domain search grids in normalized coordinates (u = cos(theta), w = 1/r).

Design (see KE_HOACH_4_TUAN.md §2.1 and DECISIONS.md #4):

* Angle "cell" (Rayleigh):   cell_u = 2 / N.
* Range "cell":               cell_w(theta) = the w-offset at which the exact-steering
  coherence |a(w0,theta)^H a(w0+cell_w,theta)| / N drops to ``mu_cell`` (default 0.5,
  same convention as the Cui-Dai polar codebook).  Measured numerically by bisection.
  Under the Fresnel approximation cell_w(theta) ~ cell_w(90 deg) / sin^2(theta).
* A grid with oversampling ``ov`` uses steps cell/ov.  Because the w-cell is smallest at
  broadside, a rectangular (u, w) grid with step cell_w(90 deg)/ov_w is at least ``ov_w``
  times oversampled everywhere in the region.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from nfisac.config import ArrayCfg
from nfisac.geometry import steering, uw_to_rt


def cell_u(cfg: ArrayCfg) -> float:
    """Angular resolution cell in u = cos(theta): 2 / N."""
    return 2.0 / cfg.N


@lru_cache(maxsize=256)
def _cell_w_cached(cfg: ArrayCfg, theta: float, mu: float, r_ref: float) -> float:
    a0 = steering(cfg, r_ref, theta)
    w0 = 1.0 / r_ref

    def coh(dw: float) -> float:
        return abs(np.vdot(a0, steering(cfg, 1.0 / (w0 + dw), theta))) / cfg.N

    # Bracket: coherence decreases monotonically from 1 over the main lobe.
    lo, hi = 0.0, 1e-4
    while coh(hi) > mu:
        lo, hi = hi, hi * 2.0
        if hi > 1e3:
            raise RuntimeError("cell_w bracketing failed (range not observable at this theta?)")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if coh(mid) > mu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cell_w(
    cfg: ArrayCfg, theta: float = np.pi / 2, mu: float = 0.5, r_ref: float = 10.0
) -> float:
    """Range resolution cell in w = 1/r [1/m], measured with the exact steering vector.

    Parameters
    ----------
    theta : float
        Angle in radians (default broadside).
    mu : float
        Coherence level defining the cell edge (default 0.5).
    r_ref : float
        Reference range (m) at which the cell is measured; the cell is nearly
        independent of r_ref inside the Fresnel region.
    """
    return _cell_w_cached(cfg, float(theta), float(mu), float(r_ref))


@dataclass(frozen=True)
class PolarGrid:
    """Rectangular grid in (u, w).

    Attributes
    ----------
    u : np.ndarray, shape (Nu,)      cos(theta) samples (ascending)
    w : np.ndarray, shape (Nw,)      1/r samples in 1/m (ascending)
    cell_u, cell_w : float           resolution cells used to build the grid
    """

    u: np.ndarray
    w: np.ndarray
    cell_u: float
    cell_w: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.u.size, self.w.size)

    @property
    def size(self) -> int:
        return self.u.size * self.w.size

    @property
    def step_u(self) -> float:
        return float(self.u[1] - self.u[0]) if self.u.size > 1 else self.cell_u

    @property
    def step_w(self) -> float:
        return float(self.w[1] - self.w[0]) if self.w.size > 1 else self.cell_w

    def flat_uw(self) -> tuple[np.ndarray, np.ndarray]:
        """Flattened (u, w) of all points, C-order over shape (Nu, Nw)."""
        U, W = np.meshgrid(self.u, self.w, indexing="ij")
        return U.ravel(), W.ravel()

    def flat_rt(self) -> tuple[np.ndarray, np.ndarray]:
        """Flattened (r [m], theta [rad]) of all points, C-order over shape (Nu, Nw)."""
        u, w = self.flat_uw()
        return uw_to_rt(u, w)


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    n = max(1, int(np.ceil((hi - lo) / step - 1e-9)) + 1)
    if n == 1:
        return np.array([0.5 * (lo + hi)])
    return np.linspace(lo, hi, n)


def polar_grid(
    cfg: ArrayCfg,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    ov_u: float = 8.0,
    ov_w: float = 8.0,
    mu_cell: float = 0.5,
) -> PolarGrid:
    """Build a rectangular (u, w) grid covering r in [r_min, r_max], theta in [theta_min, theta_max].

    Steps are cell_u/ov_u and cell_w(90 deg)/ov_w (angles in radians, ranges in meters).
    """
    if not (0 < r_min < r_max):
        raise ValueError("need 0 < r_min < r_max")
    if not (0 <= theta_min < theta_max <= np.pi):
        raise ValueError("need 0 <= theta_min < theta_max <= pi")
    cu = cell_u(cfg)
    cw = cell_w(cfg, np.pi / 2, mu_cell)
    u = _axis(np.cos(theta_max), np.cos(theta_min), cu / ov_u)
    w = _axis(1.0 / r_max, 1.0 / r_min, cw / ov_w)
    return PolarGrid(u=u, w=w, cell_u=cu, cell_w=cw)


def roi_grid(
    cfg: ArrayCfg,
    u0: float,
    w0: float,
    half_u_cells: float,
    half_w_cells: float,
    ov_u: float = 8.0,
    ov_w: float = 8.0,
    mu_cell: float = 0.5,
    r_min: float = 1.0,
    r_max: float = 1e4,
) -> PolarGrid:
    """Rectangular (u, w) window centred at (u0, w0) spanning +-half cells on each axis.

    The window is clipped to u in [-1, 1] and w in [1/r_max, 1/r_min].
    """
    cu = cell_u(cfg)
    cw = cell_w(cfg, np.pi / 2, mu_cell)
    u_lo, u_hi = max(-1.0, u0 - half_u_cells * cu), min(1.0, u0 + half_u_cells * cu)
    w_lo = max(1.0 / r_max, w0 - half_w_cells * cw)
    w_hi = min(1.0 / r_min, w0 + half_w_cells * cw)
    return PolarGrid(u=_axis(u_lo, u_hi, cu / ov_u), w=_axis(w_lo, w_hi, cw / ov_w), cell_u=cu, cell_w=cw)


def chunk_iter(n: int, chunk: int) -> Iterator[slice]:
    """Yield consecutive slices covering range(n) in blocks of at most ``chunk``."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    for start in range(0, n, chunk):
        yield slice(start, min(n, start + chunk))


__all__ = ["PolarGrid", "cell_u", "cell_w", "chunk_iter", "polar_grid", "roi_grid"]

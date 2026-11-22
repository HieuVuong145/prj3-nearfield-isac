"""Chunked spatial pseudo-spectra (MUSIC, Capon) over polar grids.

All spectra are evaluated block-wise so that the (N x G) grid steering matrix never has
to be materialised in full (N = 513 with ~10^6 grid points would need > 8 GB).
Steering blocks can optionally be cached across Monte-Carlo trials when they fit in
``cache_limit`` complex elements.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from nfisac.config import ArrayCfg
from nfisac.geometry import steering
from nfisac.grids import PolarGrid, chunk_iter

DEFAULT_CHUNK = 4096
DEFAULT_CACHE_LIMIT = 8_000_000  # complex128 elements (~128 MB)


class GridSteering:
    """Lazily evaluated (and optionally cached) steering vectors of a :class:`PolarGrid`.

    Parameters
    ----------
    cfg : ArrayCfg
    grid : PolarGrid
    mode : {"exact", "fresnel", "far"}
        Steering model used by the *estimator* (may differ from the data model).
    chunk : int
        Number of grid points per block.
    cache_limit : int
        Cache blocks if N * G <= cache_limit; 0 disables caching.
    """

    def __init__(
        self,
        cfg: ArrayCfg,
        grid: PolarGrid,
        mode: str = "exact",
        chunk: int = DEFAULT_CHUNK,
        cache_limit: int = DEFAULT_CACHE_LIMIT,
    ) -> None:
        self.cfg = cfg
        self.grid = grid
        self.mode = mode
        self.chunk = chunk
        self.r, self.theta = grid.flat_rt()
        self._cache: list[np.ndarray] | None = None
        self._use_cache = cfg.N * grid.size <= cache_limit

    def blocks(self) -> Iterator[tuple[slice, np.ndarray]]:
        """Iterate over (slice, A_block) with A_block of shape (N, block_size)."""
        if self._use_cache and self._cache is not None:
            return zip(chunk_iter(self.grid.size, self.chunk), self._cache, strict=True)
        return self._generate()

    def _generate(self) -> Iterator[tuple[slice, np.ndarray]]:
        store = [] if self._use_cache else None
        for sl in chunk_iter(self.grid.size, self.chunk):
            A = steering(self.cfg, self.r[sl], self.theta[sl], mode=self.mode)
            if store is not None:
                store.append(A)
            yield sl, A
        if store is not None:
            self._cache = store

    def evaluate(self, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Apply ``fn(A_block) -> (block_size,)`` to all blocks; returns array of shape grid.shape."""
        out = np.empty(self.grid.size, dtype=np.float64)
        for sl, A in self.blocks():
            out[sl] = fn(A)
        return out.reshape(self.grid.shape)


def signal_subspace(R: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a Hermitian covariance; return (U_s (N, K), eigenvalues descending (N,))."""
    evals, evecs = np.linalg.eigh(R)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    return evecs[:, :K], evals


def music_null(Us: np.ndarray, A: np.ndarray) -> np.ndarray:
    """MUSIC null spectrum a^H U_n U_n^H a = ||a||^2 - ||U_s^H a||^2 for each column of A."""
    G = Us.conj().T @ A
    return np.real(np.sum(np.abs(A) ** 2, axis=0) - np.sum(np.abs(G) ** 2, axis=0))


def music_spectrum(
    cfg: ArrayCfg, Us: np.ndarray, grid: PolarGrid | GridSteering, floor: float = 1e-12
) -> np.ndarray:
    """2D MUSIC pseudo-spectrum P = 1 / (N - ||U_s^H a||^2), shape grid.shape.

    Parameters
    ----------
    Us : np.ndarray
        Signal subspace of shape (N, K).
    grid : PolarGrid or GridSteering
        Pass a GridSteering to reuse cached steering blocks across calls.
    """
    gs = grid if isinstance(grid, GridSteering) else GridSteering(cfg, grid, cache_limit=0)
    return gs.evaluate(lambda A: 1.0 / np.maximum(music_null(Us, A), floor))


def capon_spectrum(
    cfg: ArrayCfg, R: np.ndarray, grid: PolarGrid | GridSteering, loading: float = 1e-3
) -> np.ndarray:
    """Capon/MVDR spectrum P = 1 / (a^H (R + delta I)^{-1} a), delta = loading * tr(R)/N."""
    N = R.shape[0]
    delta = loading * float(np.real(np.trace(R))) / N
    Ri = np.linalg.inv(R + delta * np.eye(N))
    gs = grid if isinstance(grid, GridSteering) else GridSteering(cfg, grid, cache_limit=0)
    return gs.evaluate(lambda A: 1.0 / np.real(np.sum(A.conj() * (Ri @ A), axis=0)))


__all__ = [
    "GridSteering",
    "capon_spectrum",
    "music_null",
    "music_spectrum",
    "signal_subspace",
]

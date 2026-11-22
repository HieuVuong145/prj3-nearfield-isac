"""Local-maximum peak picking on 1D / 2D pseudo-spectra."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def _local_max_mask(P: np.ndarray, size: int, edges: bool) -> np.ndarray:
    Pm = maximum_filter(P, size=size, mode="constant", cval=-np.inf)
    mask = (P >= Pm) & np.isfinite(P)
    if not edges:
        if P.ndim == 1:
            mask[[0, -1]] = False
        else:
            mask[[0, -1], :] = False
            mask[:, [0, -1]] = False
    return mask


def pick_peaks_2d(
    P: np.ndarray,
    K: int,
    min_sep: int = 1,
    edges: bool = True,
    rel_floor: float = 0.0,
) -> np.ndarray:
    """Return indices of the K largest local maxima of a 2D spectrum.

    Parameters
    ----------
    P : np.ndarray, shape (Nu, Nw)
        Pseudo-spectrum (linear scale, larger = more likely).
    K : int
        Maximum number of peaks to return.
    min_sep : int
        Neighbourhood half-width (grid points) of the local-maximum test and of the greedy
        non-maximum suppression between accepted peaks (Chebyshev distance > min_sep).
    edges : bool
        If False, maxima on the grid boundary are discarded (useful for local windows,
        where a boundary maximum usually means the true peak is outside).
    rel_floor : float
        Discard peaks with P < rel_floor * max(P).

    Returns
    -------
    np.ndarray, shape (K', 2), int
        (iu, iw) indices sorted by decreasing P; K' <= K.
    """
    mask = _local_max_mask(P, 2 * min_sep + 1, edges)
    if rel_floor > 0:
        mask &= P >= rel_floor * np.nanmax(P)
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=int)
    order = np.argsort(P[idx[:, 0], idx[:, 1]])[::-1]
    idx = idx[order]
    chosen: list[np.ndarray] = []
    for p in idx:
        if all(np.max(np.abs(p - q)) > min_sep for q in chosen):
            chosen.append(p)
            if len(chosen) == K:
                break
    return np.array(chosen, dtype=int).reshape(-1, 2)


def pick_peaks_1d(P: np.ndarray, K: int, min_sep: int = 1, edges: bool = True) -> np.ndarray:
    """1D analogue of :func:`pick_peaks_2d`; returns up to K indices sorted by decreasing P."""
    mask = _local_max_mask(P, 2 * min_sep + 1, edges)
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(P[idx])[::-1]]
    chosen: list[int] = []
    for i in idx:
        if all(abs(i - j) > min_sep for j in chosen):
            chosen.append(int(i))
            if len(chosen) == K:
                break
    return np.array(chosen, dtype=int)


__all__ = ["pick_peaks_1d", "pick_peaks_2d"]

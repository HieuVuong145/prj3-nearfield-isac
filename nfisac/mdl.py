"""Model-order selection: MDL estimate of the number of targets (Wax & Kailath 1985)."""
from __future__ import annotations

import numpy as np


def mdl_scores(evals: np.ndarray, T: int, k_max: int | None = None) -> np.ndarray:
    """MDL(k) for k = 0..k_max from the eigenvalues of the sample covariance.

    MDL(k) = -T (N-k) log(g_k / a_k) + 0.5 k (2N - k) log T,
    with g_k, a_k the geometric / arithmetic means of the N-k smallest eigenvalues.

    Parameters
    ----------
    evals : np.ndarray, shape (N,)
        Eigenvalues (any order; sorted internally in decreasing order).
    T : int
        Number of snapshots.
    k_max : int or None
        Largest model order tested (default N - 1, capped at T - 1).
    """
    lam = np.sort(np.real(evals))[::-1]
    lam = np.maximum(lam, 1e-300)
    N = lam.size
    if k_max is None:
        k_max = N - 1
    k_max = min(k_max, N - 1)
    scores = np.empty(k_max + 1)
    log_lam = np.log(lam)
    for k in range(k_max + 1):
        tail = lam[k:]
        m = N - k
        log_g = np.mean(log_lam[k:])
        log_a = np.log(np.mean(tail))
        scores[k] = -T * m * (log_g - log_a) + 0.5 * k * (2 * N - k) * np.log(T)
    return scores


def mdl_estimate_K(evals: np.ndarray, T: int, k_max: int | None = None) -> int:
    """Return K_hat = argmin_k MDL(k)."""
    return int(np.argmin(mdl_scores(evals, T, k_max)))


__all__ = ["mdl_estimate_K", "mdl_scores"]

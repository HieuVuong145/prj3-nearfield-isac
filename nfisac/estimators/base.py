"""Base protocol and result container for parameter estimators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from nfisac.config import Target


@dataclass
class EstResult:
    """Result returned by an estimator.

    Attributes
    ----------
    targets : list[Target]
        Estimated targets with (r, theta, beta).
    runtime_s : float
        Execution time in seconds.
    extra : dict[str, Any]
        Additional diagnostics, spectra, intermediate outputs.
    """

    targets: list[Target]
    runtime_s: float
    extra: dict[str, Any] = field(default_factory=dict)


class Estimator(Protocol):
    """Protocol for target location estimators."""

    name: str

    def fit(self, Y: np.ndarray, K: int | None = None, X: np.ndarray | None = None) -> EstResult:
        """Estimate parameters from received signal array Y.

        Parameters
        ----------
        Y : np.ndarray
            Received signal matrix of shape (N, T).
        K : int or None, optional
            Number of targets. If None, the algorithm should estimate K (e.g. via MDL).
        X : np.ndarray or None, optional
            Known transmit waveform (N, T).  Ignored by blind estimators (B0/B1/B2);
            required by known-waveform estimators (P1).

        Returns
        -------
        EstResult
        """
        ...


def targets_from_uw(u: np.ndarray, w: np.ndarray) -> list[Target]:
    """Build Target list from normalized coordinates (beta unknown -> NaN)."""
    u = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    w = np.asarray(w, dtype=np.float64)
    with np.errstate(divide="ignore"):
        r = np.where(np.isfinite(w), 1.0 / w, np.nan)
    return [
        Target(r=float(ri), theta=float(np.arccos(ui)), beta=complex(np.nan, 0.0))
        for ui, ri in zip(u, r, strict=True)
    ]


def sample_covariance(Y: np.ndarray, loading: float = 0.0) -> np.ndarray:
    """R_hat = Y Y^H / T (+ loading * tr(R)/N * I).  Warns if T < N and loading == 0."""
    N, T = Y.shape
    R = (Y @ Y.conj().T) / T
    if T < N and loading == 0.0:
        import warnings

        warnings.warn(f"T={T} < N={N}: sample covariance is rank deficient", stacklevel=2)
    if loading > 0.0:
        R = R + loading * float(np.real(np.trace(R))) / N * np.eye(N)
    return R

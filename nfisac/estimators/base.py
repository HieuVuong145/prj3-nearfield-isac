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

    def fit(self, Y: np.ndarray, K: int | None = None) -> EstResult:
        """Estimate parameters from received signal array Y.

        Parameters
        ----------
        Y : np.ndarray
            Received signal matrix of shape (N, T).
        K : int or None, optional
            Number of targets. If None, the algorithm should estimate K (e.g. via MDL).

        Returns
        -------
        EstResult
        """
        ...

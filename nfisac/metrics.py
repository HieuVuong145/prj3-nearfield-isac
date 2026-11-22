"""Evaluation metrics: Hungarian matching, resolution test, P_res with Wilson CI, RMSE.

All matching and resolution decisions are made in normalized cell units
(u / cell_u, w / cell_w), cf. DECISIONS.md #3:

* matching: Hungarian assignment minimising the Euclidean distance in cell units;
* resolved (one trial): exactly K estimates, all finite, and for every target k
  |u_hat - u_k| < sep_u,k / 2 and |w_hat - w_k| < sep_w,k / 2, where sep_*,k is the
  separation (along that axis) from target k to its nearest neighbour (nearest in
  cell-normalised Euclidean distance).  If that separation is (numerically) zero, the
  fixed threshold ``zero_sep_cells`` (0.5 cell) is used for the axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from nfisac.config import Target

_BIG = 1e12


def targets_uw(targets: list[Target]) -> np.ndarray:
    """Array of shape (K, 2) with columns (u, w) = (cos theta, 1/r); NaN propagates."""
    if not targets:
        return np.zeros((0, 2))
    r = np.array([t.r for t in targets], dtype=np.float64)
    th = np.array([t.theta for t in targets], dtype=np.float64)
    return np.column_stack([np.cos(th), 1.0 / r])


def match(
    est: list[Target], true: list[Target], cell_u: float, cell_w: float
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching in cell units.

    Returns
    -------
    (i_true, i_est) : index arrays of matched pairs (length min(K_est, K_true)).
    NaN coordinates (e.g. a far-field estimator without range) get a large cost on that axis.
    """
    if not est or not true:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    scale = np.array([cell_u, cell_w])
    pt = targets_uw(true) / scale
    pe = targets_uw(est) / scale
    diff = pt[:, None, :] - pe[None, :, :]
    diff = np.where(np.isfinite(diff), diff, 0.0)  # ignore missing axes for the assignment
    cost = np.sqrt(np.sum(diff**2, axis=-1))
    it, ie = linear_sum_assignment(cost)
    return it, ie


def _nn_separations(pt: np.ndarray) -> np.ndarray:
    """Per-target |separation| along (u, w) to the nearest neighbour; shape (K, 2)."""
    K = pt.shape[0]
    if K < 2:
        return np.zeros((K, 2))
    d = np.linalg.norm(pt[:, None, :] - pt[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    return np.abs(pt - pt[nn])


def resolution_thresholds(
    true: list[Target], cell_u: float, cell_w: float, zero_sep_cells: float = 0.5
) -> np.ndarray:
    """Per-target, per-axis error thresholds in cell units, shape (K, 2)."""
    pt = targets_uw(true) / np.array([cell_u, cell_w])
    sep = _nn_separations(pt)
    thr = sep / 2.0
    thr[sep < 1e-9] = zero_sep_cells
    if pt.shape[0] == 1:
        thr[:] = zero_sep_cells
    return thr


def is_resolved(
    est: list[Target],
    true: list[Target],
    cell_u: float,
    cell_w: float,
    axes: tuple[str, ...] = ("u", "w"),
    zero_sep_cells: float = 0.5,
) -> bool:
    """Resolution test of a single trial (see module docstring).

    ``axes=("u",)`` evaluates angle-only resolution (for the far-field baseline B0).
    """
    K = len(true)
    if len(est) != K:
        return False
    it, ie = match(est, true, cell_u, cell_w)
    scale = np.array([cell_u, cell_w])
    err = np.abs(targets_uw(true)[it] - targets_uw(est)[ie]) / scale
    thr = resolution_thresholds(true, cell_u, cell_w, zero_sep_cells)[it]
    cols = [i for i, a in enumerate(("u", "w")) if a in axes]
    e = err[:, cols]
    return bool(np.all(np.isfinite(e)) and np.all(e < thr[:, cols]))


def wilson_ci(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(0.5 + conf / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class PRes:
    """Probability of resolution with confidence interval."""

    p: float
    lo: float
    hi: float
    n: int


def p_res(resolved: np.ndarray | list[bool], conf: float = 0.95) -> PRes:
    """Empirical P_res over trials, with Wilson interval."""
    arr = np.asarray(resolved, dtype=bool)
    s, n = int(arr.sum()), int(arr.size)
    lo, hi = wilson_ci(s, n, conf)
    return PRes(p=s / n if n else float("nan"), lo=lo, hi=hi, n=n)


def matched_errors(
    est: list[Target], true: list[Target], cell_u: float, cell_w: float
) -> np.ndarray | None:
    """Signed errors per true target after matching, shape (K, 4): (dr [m], dtheta [rad], du, dw).

    Returns None if the number of estimates differs from K (such trials are excluded
    from RMSE and reported via the rejection rate instead).
    """
    if len(est) != len(true):
        return None
    it, ie = match(est, true, cell_u, cell_w)
    out = np.full((len(true), 4), np.nan)
    for a, b in zip(it, ie, strict=True):
        e, t = est[b], true[a]
        out[a] = [
            e.r - t.r,
            e.theta - t.theta,
            np.cos(e.theta) - np.cos(t.theta),
            1.0 / e.r - 1.0 / t.r,
        ]
    return out


def rmse(errors: np.ndarray, axis: int = 0) -> np.ndarray:
    """Root-mean-square over ``axis`` ignoring NaN."""
    return np.sqrt(np.nanmean(np.asarray(errors) ** 2, axis=axis))


def position_error(est: Target, true: Target) -> float:
    """Euclidean position error in meters (x = r cos theta, y = r sin theta)."""
    pe = np.array([est.r * np.cos(est.theta), est.r * np.sin(est.theta)])
    pt = np.array([true.r * np.cos(true.theta), true.r * np.sin(true.theta)])
    return float(np.linalg.norm(pe - pt))


__all__ = [
    "PRes",
    "is_resolved",
    "match",
    "matched_errors",
    "p_res",
    "position_error",
    "resolution_thresholds",
    "rmse",
    "targets_uw",
    "wilson_ci",
]

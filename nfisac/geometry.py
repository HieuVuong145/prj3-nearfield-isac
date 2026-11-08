"""Array geometry, steering vectors, analytical derivatives, and spatial coherence."""
from __future__ import annotations

import numpy as np

from nfisac.config import ArrayCfg


def steering(
    cfg: ArrayCfg,
    r: float | np.ndarray,
    theta: float | np.ndarray,
    mode: str = "exact",
    amplitude: bool = False,
) -> np.ndarray:
    """Compute near-field / far-field steering vector(s).

    Parameters
    ----------
    cfg : ArrayCfg
        Antenna array configuration.
    r : float or np.ndarray
        Distance from array center in meters. Shape () or (K,).
    theta : float or np.ndarray
        Azimuth angle in radians (theta = pi / 2 is broadside). Shape () or (K,).
    mode : {"exact", "fresnel", "far"}, default="exact"
        Steering vector model:
        - "exact": Spherical wave model using exact distances r_n.
        - "fresnel": Second-order Taylor expansion (Fresnel approximation).
        - "far": Planar wave model (far-field, depends only on theta).
    amplitude : bool, default=False
        If True, applies distance-dependent path variation (r / r_n) across elements.

    Returns
    -------
    np.ndarray
        Steering vector of shape (N,) if r and theta are scalar, or (N, K) if array-like.
        Dtype is complex128.
    """
    is_scalar = np.isscalar(r) and np.isscalar(theta)
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))

    if r_arr.shape != theta_arr.shape:
        raise ValueError(
            f"Shape mismatch: r has shape {r_arr.shape} but theta has shape {theta_arr.shape}"
        )

    # Broadcast shapes: n_d is (N, 1), r and angles are (1, K)
    n_d = (cfg.n * cfg.d)[:, np.newaxis]
    r_row = r_arr[np.newaxis, :]
    cos_theta = np.cos(theta_arr)[np.newaxis, :]
    sin_theta = np.sin(theta_arr)[np.newaxis, :]
    lam = cfg.lam

    if mode == "exact":
        # r_n = sqrt(r^2 + (nd)^2 - 2*r*nd*cos(theta))
        rn = np.sqrt(r_row**2 + n_d**2 - 2.0 * r_row * n_d * cos_theta)
        phase = -(2.0 * np.pi / lam) * (rn - r_row)
        a = np.exp(1j * phase)
        if amplitude:
            a = a * (r_row / rn)
    elif mode == "fresnel":
        # r_n - r ≈ -nd*cos(theta) + (nd*sin(theta))^2 / (2r)
        # a_n ≈ exp(j * 2pi/lambda * (nd*cos(theta) - (nd*sin(theta))^2 / (2r)))
        phase = (2.0 * np.pi / lam) * (n_d * cos_theta - (n_d * sin_theta) ** 2 / (2.0 * r_row))
        a = np.exp(1j * phase)
        if amplitude:
            # First order expansion of r / r_n ≈ 1 + nd*cos(theta)/r
            a = a * (1.0 + (n_d * cos_theta) / r_row)
    elif mode == "far":
        # a_n ≈ exp(j * 2pi/lambda * nd*cos(theta))
        phase = (2.0 * np.pi / lam) * (n_d * cos_theta)
        a = np.exp(1j * phase)
    else:
        raise ValueError(f"Unknown steering vector mode: {mode!r}. Must be 'exact', 'fresnel', or 'far'.")

    return a[:, 0] if is_scalar else a


def steering_grad(
    cfg: ArrayCfg,
    r: float | np.ndarray,
    theta: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact steering vector and its analytical partial derivatives.

    Parameters
    ----------
    cfg : ArrayCfg
        Antenna array configuration.
    r : float or np.ndarray
        Distance from array center in meters. Shape () or (K,).
    theta : float or np.ndarray
        Azimuth angle in radians. Shape () or (K,).

    Returns
    -------
    tuple of three np.ndarray
        (a, da_dr, da_dtheta)
        Each has shape (N,) if scalar inputs, or (N, K) if array inputs.
        All dtypes are complex128.
    """
    is_scalar = np.isscalar(r) and np.isscalar(theta)
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))

    if r_arr.shape != theta_arr.shape:
        raise ValueError(
            f"Shape mismatch: r has shape {r_arr.shape} but theta has shape {theta_arr.shape}"
        )

    n_d = (cfg.n * cfg.d)[:, np.newaxis]
    r_row = r_arr[np.newaxis, :]
    cos_theta = np.cos(theta_arr)[np.newaxis, :]
    sin_theta = np.sin(theta_arr)[np.newaxis, :]
    lam = cfg.lam

    rn = np.sqrt(r_row**2 + n_d**2 - 2.0 * r_row * n_d * cos_theta)
    phase = -(2.0 * np.pi / lam) * (rn - r_row)
    a = np.exp(1j * phase)

    # ∂a_n/∂r = -j*(2pi/lambda) * ((r - nd*cos(theta))/rn - 1) * a_n
    da_dr = -1j * (2.0 * np.pi / lam) * ((r_row - n_d * cos_theta) / rn - 1.0) * a

    # ∂a_n/∂theta = -j*(2pi/lambda) * (r * nd * sin(theta) / rn) * a_n
    da_dtheta = -1j * (2.0 * np.pi / lam) * (r_row * n_d * sin_theta / rn) * a

    if is_scalar:
        return a[:, 0], da_dr[:, 0], da_dtheta[:, 0]
    return a, da_dr, da_dtheta


def coherence(a1: np.ndarray, a2: np.ndarray) -> float | np.ndarray:
    """Compute spatial coherence |a1^H @ a2| / N between steering vectors.

    Parameters
    ----------
    a1 : np.ndarray
        Steering vector(s) of shape (N,) or (N, K1).
    a2 : np.ndarray
        Steering vector(s) of shape (N,) or (N, K2).

    Returns
    -------
    float or np.ndarray
        Scalar in [0, 1] if both inputs are 1D, or matrix of shape (K1, K2).
    """
    n_antennas = a1.shape[0]
    if a1.ndim == 1 and a2.ndim == 1:
        return float(np.abs(np.vdot(a1, a2)) / n_antennas)

    a1_2d = np.atleast_2d(a1)
    a2_2d = np.atleast_2d(a2)
    if a1.ndim == 1:
        a1_2d = a1[:, np.newaxis]
    if a2.ndim == 1:
        a2_2d = a2[:, np.newaxis]

    coh = np.abs(a1_2d.conj().T @ a2_2d) / n_antennas
    if a1.ndim == 1 and a2.ndim == 1:
        return float(coh[0, 0])
    return coh


def check_fresnel_validity(cfg: ArrayCfg, r: float, theta: float) -> float:
    """Calculate the maximum absolute phase difference between exact and Fresnel steering vectors.

    Parameters
    ----------
    cfg : ArrayCfg
        Antenna array configuration.
    r : float
        Distance in meters.
    theta : float
        Azimuth angle in radians.

    Returns
    -------
    float
        Maximum phase error across all array elements in radians.
    """
    a_exact = steering(cfg, r, theta, mode="exact")
    a_fresnel = steering(cfg, r, theta, mode="fresnel")
    phase_diff = np.abs(np.angle(a_exact * a_fresnel.conj()))
    return float(np.max(phase_diff))


__all__ = [
    "ArrayCfg",
    "check_fresnel_validity",
    "coherence",
    "steering",
    "steering_grad",
]

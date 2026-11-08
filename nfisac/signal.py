"""Waveform generation, monostatic echo simulation, and noise utilities."""
from __future__ import annotations

import numpy as np

from nfisac.config import ArrayCfg, Target
from nfisac.geometry import steering


def noise_var_from_snr(snr_db: float, P: float = 1.0, beta_mag: float = 1.0) -> float:
    """Calculate noise variance per antenna sigma^2 from target SNR.

    According to the project unified convention:
    SNR = |beta|^2 * P / sigma^2  =>  sigma^2 = |beta|^2 * P * 10^(-SNR_dB / 10).

    Parameters
    ----------
    snr_db : float
        Signal-to-noise ratio in decibels (dB).
    P : float, default=1.0
        Total transmit power.
    beta_mag : float, default=1.0
        Target reflection coefficient magnitude |beta|.

    Returns
    -------
    float
        Noise variance per antenna sigma^2.
    """
    return float((beta_mag**2) * P * (10.0 ** (-snr_db / 10.0)))


def make_waveform(
    cfg: ArrayCfg,
    T: int,
    P: float = 1.0,
    kind: str = "iso",
    focus: tuple[float, float] | None = None,
    rho: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate transmitted waveform matrix X of shape (N, T).

    Parameters
    ----------
    cfg : ArrayCfg
        Antenna array configuration.
    T : int
        Number of temporal snapshots.
    P : float, default=1.0
        Total transmit power: Tr(R_x) -> P.
    kind : {"iso", "beam", "mix"}, default="iso"
        Waveform covariance structure:
        - "iso": Isotropic full-rank waveform, R_x -> (P / N) * I.
        - "beam": Rank-1 directional beam focused at focus=(r0, theta0).
        - "mix": Linear combination R_x = (1 - rho) * R_beam + rho * R_iso.
    focus : tuple[float, float] or None, default=None
        Focal coordinates (r0, theta0) in (meters, radians) for "beam" or "mix".
        If None, defaults to broadside mid-range (15.0, pi / 2).
    rho : float, default=0.2
        Mixing fraction in [0, 1] for "mix" waveform:
        weight rho is allocated to isotropic, and (1 - rho) to focused beam.
    rng : np.random.Generator or None, default=None
        Random number generator. If None, default generator is used.

    Returns
    -------
    np.ndarray
        Transmitted signal matrix X of shape (N, T), dtype complex128.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = cfg.N

    if kind == "iso":
        # Each element i.i.d. CN(0, P / N)
        std = np.sqrt(P / (2.0 * N))
        X = std * (rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T)))
        return X

    if kind == "beam":
        r0, theta0 = (15.0, np.pi / 2.0) if focus is None else focus
        # w is unit-norm conjugate steering vector
        a_foc = steering(cfg, r0, theta0, mode="exact")
        w = a_foc.conj() / np.sqrt(N)
        # s_tx is single data symbol sequence of length T with unit variance
        s_tx = (1.0 / np.sqrt(2.0)) * (
            rng.standard_normal(T) + 1j * rng.standard_normal(T)
        )
        X = np.sqrt(P) * (w[:, np.newaxis] * s_tx[np.newaxis, :])
        return X

    if kind == "mix":
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        r0, theta0 = (15.0, np.pi / 2.0) if focus is None else focus
        a_foc = steering(cfg, r0, theta0, mode="exact")
        w = a_foc.conj() / np.sqrt(N)
        s_tx = (1.0 / np.sqrt(2.0)) * (
            rng.standard_normal(T) + 1j * rng.standard_normal(T)
        )
        X_beam = np.sqrt(P) * (w[:, np.newaxis] * s_tx[np.newaxis, :])

        std_iso = np.sqrt(P / (2.0 * N))
        X_iso = std_iso * (rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T)))

        X = np.sqrt(1.0 - rho) * X_beam + np.sqrt(rho) * X_iso
        return X

    raise ValueError(f"Unknown waveform kind: {kind!r}. Must be 'iso', 'beam', or 'mix'.")


def simulate_echo(
    cfg: ArrayCfg,
    targets: list[Target],
    X: np.ndarray,
    sigma2: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate monostatic echo received signal Y of shape (N, T).

    Model:
    Y = sum_k beta_k * a_k * a_k^T * X + Z
    where Z ~ CN(0, sigma^2 * I).

    Parameters
    ----------
    cfg : ArrayCfg
        Antenna array configuration.
    targets : list[Target]
        List of targets with (r, theta, beta).
    X : np.ndarray
        Transmitted signal matrix of shape (N, T).
    sigma2 : float
        Noise variance per antenna.
    rng : np.random.Generator or None, default=None
        Random number generator. If None, default generator is used.

    Returns
    -------
    np.ndarray
        Received echo signal matrix Y of shape (N, T), dtype complex128.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, T = X.shape

    # Target echo component S
    S = np.zeros((N, T), dtype=np.complex128)
    for target in targets:
        a_k = steering(cfg, target.r, target.theta, mode="exact")
        # a_k^T @ X is (1, T), outer product with a_k gives (N, T)
        s_k = np.outer(a_k, a_k @ X)
        S += target.beta * s_k

    # Complex Gaussian noise Z ~ CN(0, sigma2 * I)
    if sigma2 > 0:
        std_z = np.sqrt(sigma2 / 2.0)
        Z = std_z * (rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T)))
    else:
        Z = np.zeros((N, T), dtype=np.complex128)

    return S + Z


__all__ = [
    "make_waveform",
    "noise_var_from_snr",
    "simulate_echo",
]

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
    users: list[tuple[float, float]] | None = None,
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
        - "orth": Deterministically orthogonal constant-modulus rows (requires T >= N):
          X = sqrt(P/N) * F[rows, :T] * diag(q), F = T-point DFT, q random QPSK,
          so X X^H / T = (P/N) I exactly.
        - "streams": U communication beams (one per entry of ``users``) with i.i.d.
          symbols plus an isotropic sensing probe of power fraction ``rho``:
          R_x = (1 - rho) * (P/U) * sum_u w_u w_u^H + rho * (P/N) I.  rank = U if rho = 0.
    focus : tuple[float, float] or None, default=None
        Focal coordinates (r0, theta0) in (meters, radians) for "beam" or "mix".
        If None, defaults to broadside mid-range (15.0, pi / 2).
    rho : float, default=0.2
        Mixing fraction in [0, 1] for "mix" waveform:
        weight rho is allocated to isotropic, and (1 - rho) to focused beam.
    rng : np.random.Generator or None, default=None
        Random number generator. If None, default generator is used.
    users : list of (r, theta) or None
        Focal points of the communication beams for ``kind="streams"``.

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

    if kind == "orth":
        if T < N:
            raise ValueError(f"kind='orth' requires T >= N (got T={T}, N={N})")
        rows = rng.permutation(T)[:N]
        t = np.arange(T)
        F = np.exp(2j * np.pi * np.outer(rows, t) / T)
        q = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, size=T)))
        return np.sqrt(P / N) * F * q[np.newaxis, :]

    if kind == "streams":
        if not users:
            raise ValueError("kind='streams' requires a non-empty list of users=(r, theta)")
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        U = len(users)
        r_u = np.array([u[0] for u in users], dtype=np.float64)
        th_u = np.array([u[1] for u in users], dtype=np.float64)
        W = steering(cfg, r_u, th_u, mode="exact").conj() / np.sqrt(N)  # (N, U), unit columns
        C = (rng.standard_normal((U, T)) + 1j * rng.standard_normal((U, T))) / np.sqrt(2.0)
        X_com = np.sqrt(P / U) * (W @ C)
        std_iso = np.sqrt(P / (2.0 * N))
        X_iso = std_iso * (rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T)))
        return np.sqrt(1.0 - rho) * X_com + np.sqrt(rho) * X_iso

    raise ValueError(
        f"Unknown waveform kind: {kind!r}. Must be 'iso', 'beam', 'mix', 'orth' or 'streams'."
    )


def sample_cov(X: np.ndarray) -> np.ndarray:
    """Sample covariance X X^H / T of an (N, T) matrix (transmit R_x or receive R_y)."""
    return (X @ X.conj().T) / X.shape[1]


def coupling_matrix(N: int, c1: complex = 0.2, c2: complex = 0.05) -> np.ndarray:
    """Banded symmetric Toeplitz mutual-coupling matrix C = toeplitz([1, c1, c2, 0, ...])."""
    from scipy.linalg import toeplitz

    col = np.zeros(N, dtype=np.complex128)
    col[0] = 1.0
    if N > 1:
        col[1] = c1
    if N > 2:
        col[2] = c2
    return toeplitz(col)


def simulate_echo(
    cfg: ArrayCfg,
    targets: list[Target],
    X: np.ndarray,
    sigma2: float,
    rng: np.random.Generator | None = None,
    mode: str = "exact",
    amplitude: bool = False,
    coupling: np.ndarray | None = None,
    pos_err_std: float = 0.0,
) -> np.ndarray:
    """Simulate monostatic echo received signal Y of shape (N, T).

    Model:
    Y = sum_k beta_k * a_k * a_k^T * X + Z
    where Z ~ CN(0, sigma^2 * I).

    Model-mismatch hooks (all off by default; used by robustness experiments only):
    ``mode`` ("exact" | "fresnel") and ``amplitude`` select the data-generating steering
    model; ``coupling`` (N, N) replaces a_k by C a_k; ``pos_err_std`` (meters) perturbs the
    element x-positions by i.i.d. N(0, pos_err_std^2), drawn once per call from ``rng``
    (before the noise).

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

    positions = None
    if pos_err_std > 0.0:
        positions = cfg.n * cfg.d + pos_err_std * rng.standard_normal(N)
        if mode != "exact":
            raise ValueError("pos_err_std > 0 is only supported with mode='exact'")

    # Target echo component S
    S = np.zeros((N, T), dtype=np.complex128)
    for target in targets:
        a_k = steering(
            cfg, target.r, target.theta, mode=mode, amplitude=amplitude, positions=positions
        )
        if coupling is not None:
            a_k = coupling @ a_k
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
    "coupling_matrix",
    "sample_cov",
    "make_waveform",
    "noise_var_from_snr",
    "simulate_echo",
]

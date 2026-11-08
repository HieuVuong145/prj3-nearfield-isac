"""Fisher Information Matrix (FIM), Cramér-Rao Bound (CRB), and Position Error Bound (PEB)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfisac.config import ArrayCfg, Target
from nfisac.geometry import steering, steering_grad


@dataclass(frozen=True)
class CrbResult:
    """Container for multi-target Cramér-Rao Bound results.

    Attributes
    ----------
    J : np.ndarray
        Full Fisher Information Matrix of shape (4K, 4K).
    C : np.ndarray
        Full CRB covariance matrix of shape (4K, 4K).
    C_rt : list[np.ndarray]
        List of 2x2 covariance sub-matrices for (r_k, theta_k) of each target.
    peb : list[float]
        Position Error Bound (PEB) in meters for each target: sqrt(Tr(Phi * C_k * Phi^T)).
    crb_r : list[float]
        Variance lower bound for range r_k (m^2).
    crb_theta : list[float]
        Variance lower bound for angle theta_k (rad^2).
    sqrt_crb_r : list[float]
        Standard deviation lower bound for range r_k (m).
    sqrt_crb_theta : list[float]
        Standard deviation lower bound for angle theta_k (rad).
    """

    J: np.ndarray
    C: np.ndarray
    C_rt: list[np.ndarray]
    peb: list[float]
    crb_r: list[float]
    crb_theta: list[float]
    sqrt_crb_r: list[float]
    sqrt_crb_theta: list[float]


def peb(C_k: np.ndarray, r: float, theta: float) -> float:
    """Calculate Position Error Bound (PEB) from (r, theta) 2x2 covariance matrix.

    PEB^2 = trace(Phi * C_k * Phi^T)
    where Phi is the Jacobian of (r, theta) -> (x, y):
    Phi = [[cos(theta), -r*sin(theta)], [sin(theta), r*cos(theta)]].

    Parameters
    ----------
    C_k : np.ndarray
        2x2 covariance sub-matrix corresponding to parameters [r_k, theta_k].
    r : float
        Target distance in meters.
    theta : float
        Target azimuth angle in radians.

    Returns
    -------
    float
        Position error bound in meters (standard deviation).
    """
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    Phi = np.array(
        [[cos_th, -r * sin_th], [sin_th, r * cos_th]],
        dtype=np.float64,
    )
    pos_cov = Phi @ C_k @ Phi.T
    val = float(np.trace(pos_cov))
    return float(np.sqrt(max(0.0, val)))


def fim(
    cfg: ArrayCfg,
    targets: list[Target],
    Rx: np.ndarray,
    T: int,
    sigma2: float,
) -> np.ndarray:
    """Calculate the Fisher Information Matrix (FIM) for K targets.

    Parameter vector ordering:
    gamma = [r_1..r_K, theta_1..theta_K, Re(beta_1)..Re(beta_K), Im(beta_1)..Im(beta_K)]
    Total parameters = 4 * K.

    M(gamma) = sum_k beta_k * a_k * a_k^T
    B_dot_i = dM / d(gamma_i)
    J_ij = (2 * T / sigma^2) * Re(trace(B_dot_i^H @ B_dot_j @ Rx))

    Parameters
    ----------
    cfg : ArrayCfg
        Antenna array configuration.
    targets : list[Target]
        List of targets with (r, theta, beta).
    Rx : np.ndarray
        Transmitted signal covariance matrix of shape (N, N).
    T : int
        Number of temporal snapshots.
    sigma2 : float
        Noise variance per antenna.

    Returns
    -------
    np.ndarray
        Symmetric Fisher Information Matrix of shape (4K, 4K), dtype float64.
    """
    K = len(targets)
    if K == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # 1. Compute analytical partial derivatives dM / dgamma_i
    # Pre-allocate derivative matrices
    B_dots: list[np.ndarray] = []

    # Part 1: dM / dr_k  for k in 0..K-1
    for target in targets:
        a_k, da_dr, _ = steering_grad(cfg, target.r, target.theta)
        # dM/dr = beta_k * (da_dr @ a^T + a @ da_dr^T)
        B_r = target.beta * (np.outer(da_dr, a_k) + np.outer(a_k, da_dr))
        B_dots.append(B_r)

    # Part 2: dM / dtheta_k  for k in 0..K-1
    for target in targets:
        a_k, _, da_dth = steering_grad(cfg, target.r, target.theta)
        # dM/dtheta = beta_k * (da_dth @ a^T + a @ da_dth^T)
        B_th = target.beta * (np.outer(da_dth, a_k) + np.outer(a_k, da_dth))
        B_dots.append(B_th)

    # Part 3: dM / dRe(beta_k) = a_k @ a_k^T
    for target in targets:
        a_k = steering(cfg, target.r, target.theta, mode="exact")
        B_re = np.outer(a_k, a_k)
        B_dots.append(B_re)

    # Part 4: dM / dIm(beta_k) = 1j * a_k @ a_k^T
    for target in targets:
        a_k = steering(cfg, target.r, target.theta, mode="exact")
        B_im = 1j * np.outer(a_k, a_k)
        B_dots.append(B_im)

    P_tot = 4 * K
    J = np.zeros((P_tot, P_tot), dtype=np.float64)

    # Pre-multiply B_j by Rx for efficiency: (B_j @ Rx)
    B_Rx = [B @ Rx for B in B_dots]
    scale = (2.0 * T) / sigma2

    for i in range(P_tot):
        for j in range(i, P_tot):
            # trace(B_i^H @ B_j @ Rx) == np.vdot(B_dots[i], B_Rx[j])
            val = scale * float(np.vdot(B_dots[i], B_Rx[j]).real)
            J[i, j] = val
            J[j, i] = val

    return J


def crb(
    cfg: ArrayCfg,
    targets: list[Target],
    Rx: np.ndarray,
    T: int,
    sigma2: float,
) -> CrbResult:
    """Calculate the Cramér-Rao Bound (CRB) and PEB for K targets.

    Parameters
    ----------
    cfg : ArrayCfg
        Antenna array configuration.
    targets : list[Target]
        List of targets with (r, theta, beta).
    Rx : np.ndarray
        Transmitted signal covariance matrix of shape (N, N).
    T : int
        Number of snapshots.
    sigma2 : float
        Noise variance per antenna.

    Returns
    -------
    CrbResult
        Dataclass containing full FIM, full covariance, 2x2 blocks, and PEB.
    """
    K = len(targets)
    if K == 0:
        return CrbResult(
            J=np.zeros((0, 0)),
            C=np.zeros((0, 0)),
            C_rt=[],
            peb=[],
            crb_r=[],
            crb_theta=[],
            sqrt_crb_r=[],
            sqrt_crb_theta=[],
        )

    J = fim(cfg, targets, Rx, T, sigma2)
    # Invert FIM using pseudo-inverse with hermitian=True for numerical stability
    C = np.linalg.pinv(J, hermitian=True)

    C_rt_list: list[np.ndarray] = []
    peb_list: list[float] = []
    crb_r_list: list[float] = []
    crb_th_list: list[float] = []
    sqrt_r_list: list[float] = []
    sqrt_th_list: list[float] = []

    for k in range(K):
        # r_k index is k, theta_k index is K + k
        idx_r = k
        idx_th = K + k

        C_k = np.array(
            [[C[idx_r, idx_r], C[idx_r, idx_th]], [C[idx_th, idx_r], C[idx_th, idx_th]]],
            dtype=np.float64,
        )
        C_rt_list.append(C_k)

        var_r = float(max(0.0, C_k[0, 0]))
        var_th = float(max(0.0, C_k[1, 1]))

        crb_r_list.append(var_r)
        crb_th_list.append(var_th)
        sqrt_r_list.append(float(np.sqrt(var_r)))
        sqrt_th_list.append(float(np.sqrt(var_th)))

        peb_val = peb(C_k, targets[k].r, targets[k].theta)
        peb_list.append(peb_val)

    return CrbResult(
        J=J,
        C=C,
        C_rt=C_rt_list,
        peb=peb_list,
        crb_r=crb_r_list,
        crb_theta=crb_th_list,
        sqrt_crb_r=sqrt_r_list,
        sqrt_crb_theta=sqrt_th_list,
    )


__all__ = [
    "CrbResult",
    "crb",
    "fim",
    "peb",
]

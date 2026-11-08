"""Tests for T3: crb module (FIM, CRB, PEB)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nfisac.config import ArrayCfg, Target
from nfisac.crb import crb, fim, peb
from nfisac.geometry import steering


@pytest.fixture
def default_cfg() -> ArrayCfg:
    return ArrayCfg(N=129, fc=28e9, d_over_lambda=0.5)


def test_fim_analytical_vs_finite_difference():
    """Criterion 1: FIM matches FIM computed from central finite differences of M(gamma).

    Relative error < 1e-4.
    """
    cfg = ArrayCfg(N=65)
    T = 256
    P = 1.0
    Rx = (P / cfg.N) * np.eye(cfg.N, dtype=np.complex128)
    sigma2 = 0.01

    targets = [
        Target(r=12.0, theta=math.radians(65), beta=1.0 + 0.5j),
        Target(r=25.0, theta=math.radians(85), beta=0.8 - 0.3j),
    ]
    K = len(targets)
    J_analytical = fim(cfg, targets, Rx, T, sigma2)

    # Numerical differentiation of M(gamma)
    def compute_M(params: np.ndarray) -> np.ndarray:
        M = np.zeros((cfg.N, cfg.N), dtype=np.complex128)
        for k in range(K):
            r_k = params[k]
            th_k = params[K + k]
            re_b = params[2 * K + k]
            im_b = params[3 * K + k]
            beta_k = re_b + 1j * im_b
            a_k = steering(cfg, r_k, th_k)
            M += beta_k * np.outer(a_k, a_k)
        return M

    params0 = np.array(
        [t.r for t in targets]
        + [t.theta for t in targets]
        + [t.beta.real for t in targets]
        + [t.beta.imag for t in targets]
    )

    P_tot = 4 * K
    B_dots_fd: list[np.ndarray] = []
    for i in range(P_tot):
        p_plus = params0.copy()
        p_minus = params0.copy()
        # Appropriate step size for each parameter type
        step = 1e-4 if i < K else (1e-5 if i < 2 * K else 1e-6)
        p_plus[i] += step
        p_minus[i] -= step
        dM = (compute_M(p_plus) - compute_M(p_minus)) / (2.0 * step)
        B_dots_fd.append(dM)

    scale = (2.0 * T) / sigma2
    J_fd = np.zeros((P_tot, P_tot), dtype=np.float64)
    B_Rx_fd = [B @ Rx for B in B_dots_fd]
    for i in range(P_tot):
        for j in range(i, P_tot):
            val = scale * float(np.vdot(B_dots_fd[i], B_Rx_fd[j]).real)
            J_fd[i, j] = val
            J_fd[j, i] = val

    rel_err = np.linalg.norm(J_analytical - J_fd) / np.linalg.norm(J_analytical)
    assert rel_err < 1e-4, f"FIM finite difference rel error {rel_err:.2e} >= 1e-4"


def test_fim_symmetry_and_positive_definiteness(default_cfg: ArrayCfg):
    """Criterion 2: J is symmetric and positive definite (K=1, 2) in default configuration."""
    T = 256
    P = 1.0
    Rx = (P / default_cfg.N) * np.eye(default_cfg.N, dtype=np.complex128)
    sigma2 = 0.01

    test_cases = [
        [Target(r=15.0, theta=math.radians(60), beta=1.0)],
        [
            Target(r=10.0, theta=math.radians(60), beta=1.0),
            Target(r=25.0, theta=math.radians(80), beta=1.0),
        ],
    ]

    for targets in test_cases:
        J = fim(default_cfg, targets, Rx, T, sigma2)
        # Symmetry
        np.testing.assert_allclose(J, J.T, atol=1e-12)
        # Positive definite: all eigenvalues strictly positive
        eigs = np.linalg.eigvalsh(J)
        assert np.all(eigs > 0), f"FIM not positive definite, min eigenvalue: {eigs[0]}"


def test_crb_scaling_with_snr_and_snapshots(default_cfg: ArrayCfg):
    """Criterion 3: CRB decreases by 10 dB when SNR increases by 10 dB (±0.1 dB),

    and decreases by 3 dB when T doubles.
    """
    targets = [Target(r=15.0, theta=math.radians(60), beta=1.0)]
    P = 1.0
    Rx = (P / default_cfg.N) * np.eye(default_cfg.N, dtype=np.complex128)

    # 1. Scaling with SNR
    sigma2_10db = 0.1
    sigma2_20db = 0.01
    res_10db = crb(default_cfg, targets, Rx, T=256, sigma2=sigma2_10db)
    res_20db = crb(default_cfg, targets, Rx, T=256, sigma2=sigma2_20db)

    # 10 * log10(CRB_10db / CRB_20db) should be 10.0 dB
    ratio_r = 10.0 * math.log10(res_10db.crb_r[0] / res_20db.crb_r[0])
    ratio_th = 10.0 * math.log10(res_10db.crb_theta[0] / res_20db.crb_theta[0])
    assert abs(ratio_r - 10.0) < 0.1
    assert abs(ratio_th - 10.0) < 0.1

    # 2. Scaling with T (doubling T reduces CRB by ~3.01 dB)
    res_T256 = crb(default_cfg, targets, Rx, T=256, sigma2=0.01)
    res_T512 = crb(default_cfg, targets, Rx, T=512, sigma2=0.01)

    ratio_T_r = 10.0 * math.log10(res_T256.crb_r[0] / res_T512.crb_r[0])
    ratio_T_th = 10.0 * math.log10(res_T256.crb_theta[0] / res_T512.crb_theta[0])
    expected_reduction_db = 10.0 * math.log10(2.0)  # ~3.0103 dB
    assert abs(ratio_T_r - expected_reduction_db) < 0.1
    assert abs(ratio_T_th - expected_reduction_db) < 0.1


def test_k1_crb_distance_dependence(default_cfg: ArrayCfg):
    """Criterion 4: K=1: CRB(r) increases as r increases from 5 to 40 m; CRB(theta) varies little."""
    T = 256
    P = 1.0
    Rx = (P / default_cfg.N) * np.eye(default_cfg.N, dtype=np.complex128)
    sigma2 = 0.01

    r_values = [5.0, 10.0, 20.0, 30.0, 40.0]
    crb_r_list = []
    crb_th_list = []

    for r in r_values:
        t = [Target(r=r, theta=math.radians(90), beta=1.0)]
        res = crb(default_cfg, t, Rx, T, sigma2)
        crb_r_list.append(res.crb_r[0])
        crb_th_list.append(res.crb_theta[0])

    # CRB(r) must strictly increase with r
    for i in range(len(crb_r_list) - 1):
        assert crb_r_list[i] < crb_r_list[i + 1]

    # CRB(theta) varies minimally (< 5% relative spread)
    th_spread = (max(crb_th_list) - min(crb_th_list)) / min(crb_th_list)
    assert th_spread < 0.05


def test_k2_crb_monotonicity_with_decreasing_dr(default_cfg: ArrayCfg):
    """Criterion 5: K=2, same theta, delta_r decreasing: CRB increases monotonically."""
    T = 256
    P = 1.0
    Rx = (P / default_cfg.N) * np.eye(default_cfg.N, dtype=np.complex128)
    sigma2 = 0.01

    dr_values = [5.0, 3.0, 2.0, 1.0, 0.5]
    crb_r_list = []

    for dr in dr_values:
        targets = [
            Target(r=15.0, theta=math.radians(90), beta=1.0),
            Target(r=15.0 + dr, theta=math.radians(90), beta=1.0),
        ]
        res = crb(default_cfg, targets, Rx, T, sigma2)
        crb_r_list.append(res.crb_r[0])

    # As delta_r gets smaller, CRB must monotonically increase
    for i in range(len(crb_r_list) - 1):
        assert crb_r_list[i] < crb_r_list[i + 1], (
            f"CRB did not increase as dr decreased: {crb_r_list[i]} >= {crb_r_list[i+1]}"
        )


def test_peb_coordinate_transformation(default_cfg: ArrayCfg):
    """Criterion 6: peb matches coordinate change (checked against Jacobian Phi)."""
    r = 15.0
    theta = math.radians(60)
    C_k = np.array([[0.04, 0.002], [0.002, 0.0005]])

    # Hand calculation using Jacobian Phi = [[cos, -r*sin], [sin, r*cos]]
    Phi = np.array(
        [
            [math.cos(theta), -r * math.sin(theta)],
            [math.sin(theta), r * math.cos(theta)],
        ]
    )
    expected_peb = math.sqrt(float(np.trace(Phi @ C_k @ Phi.T)))

    calculated_peb = peb(C_k, r, theta)
    assert abs(calculated_peb - expected_peb) < 1e-12


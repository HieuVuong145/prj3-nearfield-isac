"""Tests for T1: geometry module (steering vectors, analytical derivatives, coherence)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nfisac.config import ArrayCfg
from nfisac.geometry import (
    check_fresnel_validity,
    coherence,
    steering,
    steering_grad,
)


@pytest.fixture
def default_cfg() -> ArrayCfg:
    return ArrayCfg(N=129, fc=28e9, d_over_lambda=0.5)


def test_rayleigh_distance(default_cfg: ArrayCfg):
    """Criterion 4: With N=129, f=28 GHz, d=lam/2: rayleigh ≈ 87.8 ± 0.5 m."""
    assert abs(default_cfg.rayleigh - 87.8) < 0.5


def test_steering_norm_and_unit_magnitude(default_cfg: ArrayCfg):
    """Criterion 1: |a_n| = 1 and ||a||^2 = N for all n."""
    for mode in ["exact", "fresnel", "far"]:
        a = steering(default_cfg, r=15.0, theta=math.radians(60), mode=mode)
        assert a.shape == (default_cfg.N,)
        np.testing.assert_allclose(np.abs(a), 1.0, rtol=1e-12)
        norm_sq = np.sum(np.abs(a) ** 2)
        np.testing.assert_allclose(norm_sq, default_cfg.N, rtol=1e-12)


def test_far_field_asymptotic_equivalence(default_cfg: ArrayCfg):
    """Criterion 2: With r = 1e6 m: steering(exact) matches steering(far) (error < 1e-4)."""
    r_far = 1e6
    theta = math.radians(60)
    a_exact = steering(default_cfg, r=r_far, theta=theta, mode="exact")
    a_far = steering(default_cfg, r=r_far, theta=theta, mode="far")

    max_err = np.max(np.abs(a_exact - a_far))
    assert max_err < 1e-4


def test_steering_grad_central_finite_diff(default_cfg: ArrayCfg):
    """Criterion 3: steering_grad matches central finite difference (rel error < 1e-5) at >= 20 random points."""
    rng = np.random.default_rng(2026)
    step_r = 1e-4
    step_theta = 1e-5

    for _ in range(25):
        r = float(rng.uniform(5.0, 50.0))
        theta = float(rng.uniform(math.radians(35.0), math.radians(145.0)))

        a, da_dr, da_dtheta = steering_grad(default_cfg, r, theta)

        # Central FD for r
        a_r_plus = steering(default_cfg, r + step_r, theta)
        a_r_minus = steering(default_cfg, r - step_r, theta)
        da_dr_fd = (a_r_plus - a_r_minus) / (2.0 * step_r)
        rel_err_r = np.linalg.norm(da_dr - da_dr_fd) / np.linalg.norm(da_dr)
        assert rel_err_r < 1e-5, f"Rel err in da_dr too large: {rel_err_r} at r={r}, theta={theta}"

        # Central FD for theta
        a_th_plus = steering(default_cfg, r, theta + step_theta)
        a_th_minus = steering(default_cfg, r, theta - step_theta)
        da_dtheta_fd = (a_th_plus - a_th_minus) / (2.0 * step_theta)
        rel_err_theta = np.linalg.norm(da_dtheta - da_dtheta_fd) / np.linalg.norm(da_dtheta)
        assert rel_err_theta < 1e-5, (
            f"Rel err in da_dtheta too large: {rel_err_theta} at r={r}, theta={theta}"
        )


def test_fresnel_validity_and_monotonic_growth(default_cfg: ArrayCfg):
    """Criterion 5: Fresnel matches exact within 5% phase error when r >= 10 m, theta = 90 deg;

    error strictly increases as r decreases.
    """
    theta_broadside = math.pi / 2.0
    phase_err_10m = check_fresnel_validity(default_cfg, r=10.0, theta=theta_broadside)
    # 5% phase error (0.05 rad or 5% of pi)
    assert phase_err_10m < 0.05

    # Check monotonic increase as r decreases from 20m down to 1m
    r_values = [20.0, 15.0, 10.0, 5.0, 2.0, 1.0]
    errors = [check_fresnel_validity(default_cfg, r, theta_broadside) for r in r_values]
    for i in range(len(errors) - 1):
        assert errors[i] < errors[i + 1], f"Error did not increase from {r_values[i]}m to {r_values[i+1]}m"


def test_steering_vector_symmetry(default_cfg: ArrayCfg):
    """Criterion 6: a(r, theta) with n -> -n corresponds to theta -> pi - theta."""
    r = 15.0
    for theta_deg in [30.0, 45.0, 60.0, 75.0]:
        theta = math.radians(theta_deg)
        a_forward = steering(default_cfg, r, theta)
        a_reverse_theta = steering(default_cfg, r, math.pi - theta)

        # n -> -n reverses the array along element axis
        np.testing.assert_allclose(a_forward[::-1], a_reverse_theta, rtol=1e-11, atol=1e-11)


def test_vectorization_multi_target(default_cfg: ArrayCfg):
    """Verify batch vectorization for shape (N, K)."""
    r_arr = np.array([10.0, 20.0, 30.0])
    theta_arr = np.array([math.radians(45), math.radians(90), math.radians(120)])

    A = steering(default_cfg, r_arr, theta_arr)
    assert A.shape == (default_cfg.N, 3)

    for k in range(3):
        a_k = steering(default_cfg, r_arr[k], theta_arr[k])
        np.testing.assert_allclose(A[:, k], a_k, rtol=1e-14)

    A_grad, da_dr, da_dth = steering_grad(default_cfg, r_arr, theta_arr)
    assert A_grad.shape == (default_cfg.N, 3)
    assert da_dr.shape == (default_cfg.N, 3)
    assert da_dth.shape == (default_cfg.N, 3)


def test_coherence_properties(default_cfg: ArrayCfg):
    """Verify coherence properties."""
    a1 = steering(default_cfg, 10.0, math.radians(90))
    a2 = steering(default_cfg, 10.0, math.radians(90))
    assert abs(coherence(a1, a2) - 1.0) < 1e-14

    a3 = steering(default_cfg, 10.0, math.radians(45))
    coh_13 = coherence(a1, a3)
    assert 0.0 <= coh_13 < 1.0

    # Matrix coherence
    A = np.column_stack([a1, a3])
    C = coherence(A, A)
    assert C.shape == (2, 2)
    assert abs(C[0, 0] - 1.0) < 1e-14
    assert abs(C[1, 1] - 1.0) < 1e-14
    assert abs(C[0, 1] - coh_13) < 1e-14


def test_amplitude_variation(default_cfg: ArrayCfg):
    """Verify amplitude=True modulates elements by r / r_n."""
    r = 5.0
    theta = math.radians(60)
    a_amp = steering(default_cfg, r, theta, amplitude=True)
    assert a_amp.shape == (default_cfg.N,)
    # Center element (n=0, index 64) has r_0 = r, so |a_center| = 1
    assert abs(abs(a_amp[default_cfg.n_half]) - 1.0) < 1e-14
    # End elements have r_n != r, so |a_n| != 1
    assert abs(abs(a_amp[0]) - 1.0) > 1e-4


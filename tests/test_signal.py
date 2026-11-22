"""Tests for T2: signal module (waveform generation, echo simulation, SNR)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nfisac.config import ArrayCfg, Target
from nfisac.signal import make_waveform, noise_var_from_snr, simulate_echo


@pytest.fixture
def default_cfg() -> ArrayCfg:
    return ArrayCfg(N=129, fc=28e9, d_over_lambda=0.5)


def test_noise_var_from_snr():
    """Verify SNR to noise variance conversion."""
    # SNR = 10 dB, P = 1.0, |beta| = 1.0 => sigma2 = 0.1
    assert abs(noise_var_from_snr(10.0) - 0.1) < 1e-12
    # SNR = 20 dB, P = 1.0, |beta| = 1.0 => sigma2 = 0.01
    assert abs(noise_var_from_snr(20.0) - 0.01) < 1e-12
    # SNR = 0 dB, P = 2.0, |beta| = 2.0 => sigma2 = 4 * 2 = 8.0
    assert abs(noise_var_from_snr(0.0, P=2.0, beta_mag=2.0) - 8.0) < 1e-12


def test_iso_covariance_convergence():
    """Criterion 1: With iso: X X^H / T converges to (P/N)*I as T is large.

    Relative Frobenius error < 5% with T = 20000 (tested on N=33).
    """
    cfg = ArrayCfg(N=33)
    P = 1.0
    T = 20000
    rng = np.random.default_rng(42)

    X = make_waveform(cfg, T=T, P=P, kind="iso", rng=rng)
    assert X.shape == (cfg.N, T)

    Rx = (X @ X.conj().T) / T
    Rx_expected = (P / cfg.N) * np.eye(cfg.N)

    rel_fro = np.linalg.norm(Rx - Rx_expected, "fro") / np.linalg.norm(Rx_expected, "fro")
    assert rel_fro < 0.05, f"Relative Frobenius error too large: {rel_fro:.4f}"


def test_beam_rank_one(default_cfg: ArrayCfg):
    """Criterion 2: With beam: algebraic rank of X X^H is strictly 1."""
    rng = np.random.default_rng(42)
    T = 4000
    X = make_waveform(default_cfg, T=T, P=1.0, kind="beam", rng=rng)
    assert X.shape == (default_cfg.N, T)

    Rx = (X @ X.conj().T) / T
    rank = np.linalg.matrix_rank(Rx)
    assert rank == 1

    # Total transmit power should match P
    assert abs(np.trace(Rx).real - 1.0) < 0.05


def test_mix_waveform_properties(default_cfg: ArrayCfg):
    """Verify mix waveform interpolates between beam and isotropic."""
    rng = np.random.default_rng(42)
    T = 5000
    P = 1.0
    rho = 0.3
    X = make_waveform(default_cfg, T=T, P=P, kind="mix", rho=rho, rng=rng)
    Rx = (X @ X.conj().T) / T

    # Tr(Rx) should be approximately P
    assert abs(np.trace(Rx).real - P) < 0.05


def test_signal_power_per_antenna(default_cfg: ArrayCfg):
    """Criterion 3: Average signal power per antenna ≈ |beta|^2 * P (error < 5%)."""
    rng = np.random.default_rng(123)
    target = Target(r=15.0, theta=math.radians(60), beta=1.5 * np.exp(1j * 0.7))
    P = 2.0
    T = 15000

    X = make_waveform(default_cfg, T=T, P=P, kind="iso", rng=rng)
    # Simulate echo with no noise
    Y = simulate_echo(default_cfg, [target], X, sigma2=0.0, rng=rng)

    power_per_antenna = np.mean(np.abs(Y) ** 2, axis=1)  # shape (N,)
    avg_power = float(np.mean(power_per_antenna))

    expected_power = (abs(target.beta) ** 2) * P
    rel_err = abs(avg_power - expected_power) / expected_power
    assert rel_err < 0.05, f"Signal power rel error {rel_err:.4f} > 5%"


def test_noise_power_per_antenna(default_cfg: ArrayCfg):
    """Criterion 4: Average noise power per antenna ≈ sigma^2 (error < 5%)."""
    rng = np.random.default_rng(456)
    sigma2 = 0.05
    T = 15000
    X = np.zeros((default_cfg.N, T), dtype=np.complex128)

    # Simulate echo with 0 targets -> only noise
    Y = simulate_echo(default_cfg, [], X, sigma2=sigma2, rng=rng)

    measured_noise_var = float(np.mean(np.abs(Y) ** 2))
    rel_err = abs(measured_noise_var - sigma2) / sigma2
    assert rel_err < 0.05, f"Noise power rel error {rel_err:.4f} > 5%"


def test_two_targets_eigenvalues_iso_vs_beam(default_cfg: ArrayCfg):
    """Criterion 5: With 2 targets and iso: eigenvalue of R_Y has exactly 2 eigenvalues

    significantly larger than sigma^2 at SNR 20 dB.
    With beam: only 1 eigenvalue significantly larger.
    """
    rng = np.random.default_rng(789)
    targets = [
        Target(r=10.0, theta=math.radians(60), beta=1.0),
        Target(r=25.0, theta=math.radians(90), beta=1.0),
    ]
    T = 512
    P = 1.0
    snr_db = 20.0
    sigma2 = noise_var_from_snr(snr_db, P=P)

    # 1. ISO waveform
    X_iso = make_waveform(default_cfg, T=T, P=P, kind="iso", rng=rng)
    Y_iso = simulate_echo(default_cfg, targets, X_iso, sigma2=sigma2, rng=rng)
    Ry_iso = (Y_iso @ Y_iso.conj().T) / T
    eig_iso = np.sort(np.linalg.eigvalsh(Ry_iso))[::-1]

    # Two largest eigenvalues must be much larger than sigma2 (e.g. > 10 * sigma2)
    assert eig_iso[0] > 10.0 * sigma2
    assert eig_iso[1] > 10.0 * sigma2
    # Third eigenvalue must be close to noise floor (< 3 * sigma2)
    assert eig_iso[2] < 3.0 * sigma2

    # 2. BEAM waveform (focused between the two)
    focus = (17.5, math.radians(75))
    X_beam = make_waveform(default_cfg, T=T, P=P, kind="beam", focus=focus, rng=rng)
    Y_beam = simulate_echo(default_cfg, targets, X_beam, sigma2=sigma2, rng=rng)
    Ry_beam = (Y_beam @ Y_beam.conj().T) / T
    eig_beam = np.sort(np.linalg.eigvalsh(Ry_beam))[::-1]

    # Only 1 eigenvalue much larger than sigma2
    assert eig_beam[0] > 10.0 * sigma2
    # Second eigenvalue is down at the noise floor
    assert eig_beam[1] < 3.0 * sigma2


def test_orth_waveform_exact_covariance(default_cfg: ArrayCfg):
    """kind='orth': X X^H / T = (P/N) I exactly; constant modulus; requires T >= N."""
    rng = np.random.default_rng(0)
    X = make_waveform(default_cfg, 256, P=2.0, kind="orth", rng=rng)
    np.testing.assert_allclose(X @ X.conj().T / 256, 2.0 / default_cfg.N * np.eye(default_cfg.N), atol=1e-12)
    np.testing.assert_allclose(np.abs(X), np.sqrt(2.0 / default_cfg.N))
    with pytest.raises(ValueError):
        make_waveform(default_cfg, 100, kind="orth", rng=rng)


def test_streams_waveform_rank(default_cfg: ArrayCfg):
    """kind='streams' with rho=0 has rank U; rho>0 makes it full rank."""
    from nfisac.signal import sample_cov

    rng = np.random.default_rng(1)
    users = [(10.0, 1.2), (20.0, 1.7)]
    Rx = sample_cov(make_waveform(default_cfg, 1000, kind="streams", users=users, rho=0.0, rng=rng))
    ev = np.linalg.eigvalsh(Rx)[::-1]
    assert ev[1] > 1e-3 and ev[2] < 1e-10 * ev[0]
    Rx = sample_cov(make_waveform(default_cfg, 1000, kind="streams", users=users, rho=0.3, rng=rng))
    assert np.linalg.matrix_rank(Rx) == default_cfg.N
    assert abs(np.real(np.trace(Rx)) - 1.0) < 0.05


def test_echo_mismatch_hooks_default_equivalence(default_cfg: ArrayCfg):
    """Identity coupling and zero position error reproduce the nominal echo."""
    from nfisac.signal import coupling_matrix

    tg = [Target(r=12.0, theta=1.3, beta=0.5 + 0.2j)]
    X = make_waveform(default_cfg, 64, rng=np.random.default_rng(2))
    Y0 = simulate_echo(default_cfg, tg, X, 0.0)
    Y1 = simulate_echo(default_cfg, tg, X, 0.0, coupling=coupling_matrix(default_cfg.N, 0.0, 0.0))
    Y2 = simulate_echo(default_cfg, tg, X, 0.0, rng=np.random.default_rng(3), pos_err_std=0.0)
    np.testing.assert_allclose(Y1, Y0, atol=1e-12)
    np.testing.assert_allclose(Y2, Y0, atol=1e-12)
    Y3 = simulate_echo(default_cfg, tg, X, 0.0, rng=np.random.default_rng(3), pos_err_std=1e-3)
    assert np.linalg.norm(Y3 - Y0) > 1e-3 * np.linalg.norm(Y0)

"""Tests for T5/T6: B0, B1, B1-r, B2, MDL."""
from __future__ import annotations

import numpy as np
import pytest

from nfisac.config import ArrayCfg, Target
from nfisac.crb import crb
from nfisac.estimators import Capon2D, Music2D, MusicFarField
from nfisac.grids import polar_grid, roi_grid
from nfisac.mdl import mdl_estimate_K
from nfisac.metrics import is_resolved, matched_errors, rmse
from nfisac.signal import make_waveform, noise_var_from_snr, sample_cov, simulate_echo

D2R = np.pi / 180


@pytest.fixture(scope="module")
def cfg() -> ArrayCfg:
    return ArrayCfg(N=129)


@pytest.fixture(scope="module")
def full_grid(cfg):
    return polar_grid(cfg, 3.0, 60.0, 30 * D2R, 150 * D2R)


def _echo(cfg, targets, snr_db, rng, T=256, kind="orth"):
    X = make_waveform(cfg, T, kind=kind, rng=rng)
    return simulate_echo(cfg, targets, X, noise_var_from_snr(snr_db), rng), X


def test_b1_single_target_within_one_cell(cfg, full_grid):
    """1 target (15 m, 60 deg), SNR 20 dB: B1 error <= 1 grid step on both axes."""
    rng = np.random.default_rng(1)
    tg = [Target(15.0, 60 * D2R)]
    Y, _ = _echo(cfg, tg, 20.0, rng)
    est = Music2D(cfg, full_grid).fit(Y, 1)
    e = matched_errors(est.targets, tg, full_grid.cell_u, full_grid.cell_w)[0]
    assert abs(e[2]) <= full_grid.step_u and abs(e[3]) <= full_grid.step_w


def test_b1r_efficiency_vs_crb(cfg):
    """B1-r RMSE close to the known-X CRB at SNR 0 dB.

    Blind MUSIC only uses the receive-side steering, so it is expected to be about
    sqrt(2) above the known-waveform CRB (observed ~1.45); the bound here is 1.8.
    """
    rng = np.random.default_rng(2)
    tg = [Target(15.0, 60 * D2R)]
    g = roi_grid(cfg, np.cos(tg[0].theta), 1 / 15, 20, 3)
    est = Music2D(cfg, g, refine=True)
    errs = []
    for _ in range(60):
        Y, _ = _echo(cfg, tg, 0.0, rng)
        errs.append(matched_errors(est.fit(Y, 1).targets, tg, g.cell_u, g.cell_w)[0])
    rm = rmse(np.array(errs))
    c = crb(cfg, tg, np.eye(cfg.N) / cfg.N, 256, noise_var_from_snr(0.0))
    assert rm[0] / c.sqrt_crb_r[0] < 1.8
    assert rm[1] / c.sqrt_crb_theta[0] < 1.8


def test_b1_two_far_targets_resolved(cfg, full_grid):
    """2 targets Delta theta = 20 deg: resolved in every trial at SNR 20 dB."""
    rng = np.random.default_rng(3)
    tg = [Target(15.0, 80 * D2R), Target(15.0, 100 * D2R, np.exp(0.7j))]
    est = Music2D(cfg, full_grid)
    for _ in range(20):
        Y, _ = _echo(cfg, tg, 20.0, rng)
        assert is_resolved(est.fit(Y, 2).targets, tg, full_grid.cell_u, full_grid.cell_w)


def test_reproduce_n65_reference_case():
    """N = 65 over D = 0.5 m (d ~ 0.73 lambda), target (20 m, 45 deg).

    The region must exclude the grating-lobe direction u - lambda/d (theta ~ 131 deg);
    over the full [30, 150] deg region MUSIC can lock onto the grating lobe.
    """
    lam = 299_792_458.0 / 28e9
    cfg65 = ArrayCfg(N=65, d_over_lambda=0.5 / 64 / lam)
    rng = np.random.default_rng(4)
    tg = [Target(20.0, 45 * D2R)]
    Y, _ = _echo(cfg65, tg, 20.0, rng)
    g = polar_grid(cfg65, 5.0, 40.0, 20 * D2R, 90 * D2R)
    t = Music2D(cfg65, g, refine=True).fit(Y, 1).targets[0]
    assert abs(t.r - 20.0) <= 0.2 and abs(t.theta - 45 * D2R) <= 1 * D2R


@pytest.mark.parametrize("K", [1, 2, 3])
def test_mdl_well_separated(cfg, K):
    rng = np.random.default_rng(10 + K)
    pos = [(10.0, 60.0), (20.0, 90.0), (8.0, 120.0)][:K]
    tg = [Target(r, th * D2R, np.exp(1j * rng.uniform(0, 2 * np.pi))) for r, th in pos]
    ok = 0
    for _ in range(20):
        Y, _ = _echo(cfg, tg, 20.0, rng)
        ok += mdl_estimate_K(np.linalg.eigvalsh(sample_cov(Y)), Y.shape[1]) == K
    assert ok / 20 >= 0.95


def test_b0_fails_b1_succeeds_same_angle(cfg, full_grid):
    """Co-directional targets at 10 m and 25 m: B0 cannot resolve, B1 can (near-field evidence)."""
    rng = np.random.default_rng(5)
    tg = [Target(10.0, 90 * D2R), Target(25.0, 90 * D2R, np.exp(1j))]
    b0 = MusicFarField(cfg, 30 * D2R, 150 * D2R)
    b1 = Music2D(cfg, full_grid)
    r0, r1 = [], []
    for _ in range(20):
        Y, _ = _echo(cfg, tg, 20.0, rng)
        r0.append(is_resolved(b0.fit(Y, 2).targets, tg, full_grid.cell_u, full_grid.cell_w, axes=("u",)))
        r1.append(is_resolved(b1.fit(Y, 2).targets, tg, full_grid.cell_u, full_grid.cell_w))
    assert np.mean(r0) < 0.1
    assert np.mean(r1) >= 0.9


def test_capon_single_target(cfg):
    rng = np.random.default_rng(6)
    tg = [Target(12.0, 70 * D2R)]
    g = roi_grid(cfg, np.cos(tg[0].theta), 1 / 12, 10, 3)
    Y, _ = _echo(cfg, tg, 10.0, rng)
    e = matched_errors(Capon2D(cfg, g).fit(Y, 1).targets, tg, g.cell_u, g.cell_w)[0]
    assert abs(e[2]) <= g.step_u and abs(e[3]) <= g.step_w

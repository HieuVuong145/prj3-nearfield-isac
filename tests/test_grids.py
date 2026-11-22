"""Tests for T4: grids, spectrum, peaks."""
from __future__ import annotations

import numpy as np
import pytest

from nfisac.config import ArrayCfg
from nfisac.geometry import steering, uw_to_rt
from nfisac.grids import cell_u, cell_w, chunk_iter, polar_grid, roi_grid
from nfisac.peaks import pick_peaks_1d, pick_peaks_2d
from nfisac.spectrum import GridSteering, capon_spectrum, music_null, music_spectrum

D2R = np.pi / 180


@pytest.fixture
def cfg() -> ArrayCfg:
    return ArrayCfg(N=129)


def _coh(cfg, u1, w1, u2, w2):
    r1, t1 = uw_to_rt(u1, w1)
    r2, t2 = uw_to_rt(u2, w2)
    return abs(np.vdot(steering(cfg, float(r1), float(t1)), steering(cfg, float(r2), float(t2)))) / cfg.N


def test_cell_w_definition(cfg):
    """Coherence at exactly one w-cell equals mu_cell."""
    cw = cell_w(cfg, np.pi / 2, mu=0.5, r_ref=10.0)
    assert abs(_coh(cfg, 0.0, 0.1, 0.0, 0.1 + cw) - 0.5) < 1e-3
    # Rayleigh-scale sanity: same order as lambda / D^2 (spec 1.7, c in [1, 8])
    assert 1.0 < cw / (cfg.lam / cfg.D**2) < 8.0


def test_cell_w_scales_with_sin2(cfg):
    """Fresnel prediction: cell_w(theta) ~ cell_w(90 deg) / sin^2(theta)."""
    c90 = cell_w(cfg, np.pi / 2)
    for th in (60, 45, 30):
        ratio = cell_w(cfg, th * D2R) / c90
        assert abs(ratio * np.sin(th * D2R) ** 2 - 1.0) < 0.03


def test_fine_grid_adjacent_coherence(cfg):
    """Fine grid (ov = 8): adjacent points along both axes have coherence >= 0.95."""
    g = polar_grid(cfg, 3.0, 60.0, 30 * D2R, 150 * D2R, ov_u=8, ov_w=8)
    assert abs(g.step_u - cell_u(cfg) / 8) / g.step_u < 0.05
    assert g.step_w <= g.cell_w / 8 * 1.001
    for th in (30, 60, 90, 120):
        u = np.cos(th * D2R)
        for w in (1 / 50, 1 / 10, 1 / 4):
            assert _coh(cfg, u, w, u, w + g.step_w) >= 0.95
            assert _coh(cfg, u, w, u + g.step_u, w) >= 0.95


def test_range_cells_decrease_towards_endfire(cfg):
    """Fewer resolution cells across [r_min, r_max] near endfire than at broadside."""
    span = 1 / 3 - 1 / 60
    n90 = span / cell_w(cfg, 90 * D2R)
    n30 = span / cell_w(cfg, 30 * D2R)
    assert n30 < n90


def test_roi_grid_clipping(cfg):
    g = roi_grid(cfg, 0.99, 0.01, 10, 10, r_min=3.0, r_max=200.0)
    assert g.u.max() <= 1.0 and g.w.min() >= 1 / 200.0 - 1e-15


def test_chunk_iter_covers():
    sls = list(chunk_iter(10, 3))
    assert [s.start for s in sls] == [0, 3, 6, 9]
    assert sls[-1].stop == 10


def test_chunked_spectrum_matches_direct(cfg):
    """Chunked MUSIC / Capon spectra equal the one-shot computation to 1e-10."""
    rng = np.random.default_rng(0)
    g = roi_grid(cfg, 0.3, 0.1, 3, 2, ov_u=4, ov_w=4)
    Us, _ = np.linalg.qr(rng.standard_normal((cfg.N, 2)) + 1j * rng.standard_normal((cfg.N, 2)))
    r, th = g.flat_rt()
    A = steering(cfg, r, th)
    direct = (1.0 / music_null(Us, A)).reshape(g.shape)
    for chunk in (7, 64, 10_000):
        gs = GridSteering(cfg, g, chunk=chunk)
        P1 = music_spectrum(cfg, Us, gs)
        P2 = music_spectrum(cfg, Us, gs)  # second call uses the cache
        np.testing.assert_allclose(P1, direct, rtol=1e-10)
        np.testing.assert_array_equal(P1, P2)
    B = rng.standard_normal((cfg.N, 300)) + 1j * rng.standard_normal((cfg.N, 300))
    R = B @ B.conj().T / 300
    Ri = np.linalg.inv(R + 1e-3 * np.real(np.trace(R)) / cfg.N * np.eye(cfg.N))
    direct_c = (1.0 / np.real(np.sum(A.conj() * (Ri @ A), axis=0))).reshape(g.shape)
    np.testing.assert_allclose(capon_spectrum(cfg, R, GridSteering(cfg, g, chunk=11)), direct_c, rtol=1e-10)


def test_pick_peaks_2d_gaussians():
    """Finds the K main Gaussian peaks and ignores weaker side peaks."""
    x, y = np.meshgrid(np.arange(100), np.arange(80), indexing="ij")
    centres = [(20, 15, 1.0), (60, 50, 0.8), (40, 70, 0.6)]
    P = np.zeros_like(x, dtype=float)
    for cx, cy, a in centres:
        P += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 20.0)
    P += 0.1 * np.exp(-((x - 85) ** 2 + (y - 10) ** 2) / 5.0)  # sidelobe-like weak peak
    idx = pick_peaks_2d(P, 3)
    assert {tuple(p) for p in idx} == {(c[0], c[1]) for c in centres}
    assert tuple(idx[0]) == (20, 15)  # sorted by height
    assert len(pick_peaks_2d(P, 10, rel_floor=0.3)) == 3


def test_pick_peaks_edges_and_1d():
    P = np.linspace(0, 1, 50)[:, None] * np.ones((1, 5))  # monotone ramp: max on edge
    assert len(pick_peaks_2d(P, 1, edges=False)) == 0
    assert len(pick_peaks_2d(P, 1, edges=True)) == 1
    p1 = np.exp(-((np.arange(200) - 50) ** 2) / 10) + 0.5 * np.exp(-((np.arange(200) - 150) ** 2) / 10)
    assert list(pick_peaks_1d(p1, 2)) == [50, 150]

"""Tests for metrics (matching, resolution, P_res, RMSE) and the Monte-Carlo runner."""
from __future__ import annotations

import numpy as np

from nfisac.config import Target
from nfisac.mc import run_mc
from nfisac.metrics import (
    is_resolved,
    match,
    matched_errors,
    p_res,
    resolution_thresholds,
    rmse,
    wilson_ci,
)

D2R = np.pi / 180
CU, CW = 2 / 129, 0.1089


def _t(r, th_deg):
    return Target(r, th_deg * D2R)


def test_match_permutation():
    true = [_t(10, 80), _t(20, 100)]
    est = [_t(20.1, 100.1), _t(10.1, 79.9)]
    it, ie = match(est, true, CU, CW)
    assert dict(zip(it, ie, strict=True)) == {0: 1, 1: 0}


def test_is_resolved_hand_cases():
    true = [_t(10, 90), _t(10, 92)]  # same r, delta u ~ 0.035 = 2.25 cells
    good = [_t(10.0, 90.2), _t(10.0, 91.8)]
    assert is_resolved(good, true, CU, CW)
    assert not is_resolved(good[:1], true, CU, CW)  # missing peak
    assert not is_resolved([_t(10, 91.0), _t(10, 91.9)], true, CU, CW)  # one error > half sep
    far_w = [_t(5.0, 90.2), _t(10.0, 91.8)]  # w error 0.1 = 0.92 cell > 0.5 fixed thr
    assert not is_resolved(far_w, true, CU, CW)
    nan_r = [Target(np.nan, 90.2 * D2R), Target(np.nan, 91.8 * D2R)]
    assert not is_resolved(nan_r, true, CU, CW)
    assert is_resolved(nan_r, true, CU, CW, axes=("u",))


def test_thresholds_zero_separation():
    thr = resolution_thresholds([_t(10, 90), _t(12, 90)], CU, CW)
    np.testing.assert_allclose(thr[:, 0], 0.5)  # zero u-separation -> fixed 0.5 cell
    np.testing.assert_allclose(thr[:, 1], abs(1 / 10 - 1 / 12) / CW / 2)


def test_wilson_and_p_res():
    lo, hi = wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and hi > 0.1
    pr = p_res([True] * 9 + [False])
    assert pr.p == 0.9 and pr.lo < 0.9 < pr.hi and pr.n == 10


def test_matched_errors_and_rmse():
    true = [_t(10, 90)]
    assert matched_errors([], true, CU, CW) is None
    e = matched_errors([_t(10.5, 91)], true, CU, CW)
    assert abs(e[0, 0] - 0.5) < 1e-12 and abs(e[0, 1] - D2R) < 1e-12
    np.testing.assert_allclose(rmse(np.array([[3.0], [4.0]])), [np.sqrt(12.5)])


def _toy_trial(rng: np.random.Generator, scale: float) -> dict:
    return {"x": scale * rng.standard_normal(3), "y": float(rng.uniform())}


def test_mc_bit_exact_and_cache(tmp_path):
    a = run_mc(_toy_trial, {"scale": 2.0}, n_trials=12, seed=7, n_jobs=1)
    b = run_mc(_toy_trial, {"scale": 2.0}, n_trials=12, seed=7, n_jobs=2)
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["y"], b["y"])
    assert a["x"].shape == (12, 3)
    c = run_mc(_toy_trial, {"scale": 2.0}, n_trials=12, seed=7, cache_dir=tmp_path, tag="toy")
    files = list(tmp_path.glob("toy_*.npz"))
    assert len(files) == 1
    d = run_mc(_toy_trial, {"scale": 2.0}, n_trials=12, seed=7, cache_dir=tmp_path, tag="toy")
    np.testing.assert_array_equal(c["x"], d["x"])
    e = run_mc(_toy_trial, {"scale": 2.0}, n_trials=12, seed=8, n_jobs=1)
    assert not np.array_equal(a["x"], e["x"])

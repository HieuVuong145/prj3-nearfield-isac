"""Tests for T0: repo initialisation, config loading, import."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# ── T0 acceptance criteria ──────────────────────────────────────────────────

def test_import_nfisac():
    """python -c 'import nfisac' must succeed."""
    import nfisac  # noqa: F401


def test_import_config():
    """config module must expose ArrayCfg and load_config."""
    from nfisac.config import ArrayCfg, load_config  # noqa: F401


def test_arraycfg_defaults():
    """ArrayCfg default properties are consistent with spec §1.1."""
    from nfisac.config import ArrayCfg

    cfg = ArrayCfg(N=129)
    assert cfg.N == 129
    assert cfg.fc == 28e9
    assert cfg.d_over_lambda == 0.5
    assert cfg.c == 299_792_458.0

    # derived properties
    lam = cfg.c / cfg.fc
    assert abs(cfg.lam - lam) < 1e-15
    assert abs(cfg.d - lam * 0.5) < 1e-15

    D_expected = (129 - 1) * cfg.d
    assert abs(cfg.D - D_expected) < 1e-12

    rayleigh_expected = 2 * D_expected**2 / lam
    assert abs(cfg.rayleigh - rayleigh_expected) < 1e-6

    # n indices: array from -(N//2) to +(N//2)
    n = cfg.n
    assert isinstance(n, np.ndarray)
    assert len(n) == 129
    assert n[0] == -64
    assert n[-1] == 64
    assert n[64] == 0  # center

    # n_half
    assert cfg.n_half == 64


def test_arraycfg_frozen():
    """ArrayCfg must be immutable (frozen dataclass)."""
    from nfisac.config import ArrayCfg

    cfg = ArrayCfg(N=129)
    with pytest.raises(AttributeError):
        cfg.N = 65  # type: ignore[misc]


def test_arraycfg_odd_n():
    """ArrayCfg must only accept odd N."""
    from nfisac.config import ArrayCfg

    with pytest.raises((ValueError, AssertionError)):
        ArrayCfg(N=128)


def test_load_default_config():
    """load_config() loads configs/default.yaml and returns a ProjectCfg."""
    from nfisac.config import load_config

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    assert cfg_path.exists(), f"default.yaml not found at {cfg_path}"

    pcfg = load_config(cfg_path)

    # Check array section
    assert pcfg.array.N == 129
    assert pcfg.array.fc == 28e9

    # Check signal section
    assert pcfg.signal.T == 256
    assert pcfg.signal.P == 1.0
    assert pcfg.signal.waveform == "iso"
    assert pcfg.signal.snr_db == 10

    # Check targets_region
    assert pcfg.targets_region.r_min == 3.0
    assert pcfg.targets_region.r_max == 60.0
    assert pcfg.targets_region.theta_deg == [30, 150]

    # Check grid section
    assert pcfg.grid.music_mu == 0.95
    assert pcfg.grid.coarse_mu == 0.7
    assert pcfg.grid.cs_mu == 0.5
    assert pcfg.grid.u_oversample == 2

    # Check mc section
    assert pcfg.mc.n_trials == 200
    assert pcfg.mc.seed == 0
    assert pcfg.mc.n_jobs == -1


def test_target_dataclass():
    """Target dataclass stores r, theta (rad), beta (complex)."""
    from nfisac.config import Target

    t = Target(r=15.0, theta=math.radians(60), beta=1.0 + 0.5j)
    assert t.r == 15.0
    assert abs(t.theta - math.radians(60)) < 1e-15
    assert t.beta == 1.0 + 0.5j


def test_all_placeholder_modules_importable():
    """Every module listed in the repo structure must be importable."""
    modules = [
        "nfisac.config",
        "nfisac.geometry",
        "nfisac.signal",
        "nfisac.crb",
        "nfisac.grids",
        "nfisac.spectrum",
        "nfisac.peaks",
        "nfisac.mdl",
        "nfisac.estimators",
        "nfisac.estimators.base",
        "nfisac.estimators.music_ff",
        "nfisac.estimators.music2d",
        "nfisac.estimators.capon",
        "nfisac.estimators.music_hier",
        "nfisac.estimators.refine_dml",
        "nfisac.estimators.cs",
        "nfisac.estimators.ap_dml",
        "nfisac.metrics",
        "nfisac.mc",
    ]
    import importlib

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


def test_estimator_protocol_and_estresult():
    """Estimator Protocol and EstResult must be defined in estimators.base."""
    from nfisac.estimators.base import Estimator, EstResult  # noqa: F401

    # EstResult must hold targets, runtime_s, extra
    res = EstResult(targets=[], runtime_s=0.01, extra={})
    assert res.targets == []
    assert res.runtime_s == 0.01
    assert res.extra == {}


"""Shared trial functions for the experiments (module-level so they can be pickled by mc.run_mc).

Estimators are built once per worker process through ``functools.lru_cache`` so that the
grid steering cache is reused across trials.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from nfisac.config import ArrayCfg, Target
from nfisac.estimators import Music2D, MusicFarField
from nfisac.grids import polar_grid, roi_grid
from nfisac.metrics import is_resolved, matched_errors
from nfisac.signal import make_waveform, noise_var_from_snr, simulate_echo

D2R = np.pi / 180
RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = RESULTS / "figs"
CACHE = RESULTS / "cache"


def targets_from_spec(spec: list[tuple[float, float, float]]) -> list[Target]:
    """[(r [m], theta [deg], beta phase [rad]), ...] -> Target list with |beta| = 1."""
    return [Target(r, th * D2R, complex(np.exp(1j * ph))) for r, th, ph in spec]


@lru_cache(maxsize=16)
def build_music(N: int, grid_spec: tuple, refine: bool = True) -> Music2D:
    """grid_spec = ("full", r_min, r_max, th_min_deg, th_max_deg)
    or ("roi", u0, w0, half_u_cells, half_w_cells)."""
    cfg = ArrayCfg(N)
    if grid_spec[0] == "full":
        _, r0, r1, t0, t1 = grid_spec
        g = polar_grid(cfg, r0, r1, t0 * D2R, t1 * D2R)
    elif grid_spec[0] == "roi":
        _, u0, w0, hu, hw = grid_spec
        g = roi_grid(cfg, u0, w0, hu, hw, r_min=3.0, r_max=60.0)
    elif grid_spec[0] == "band":  # +-hu angle cells around u0, full w range [1/r1, 1/r0]
        _, u0, hu, r0, r1 = grid_spec
        full = polar_grid(cfg, r0, r1, 30 * D2R, 150 * D2R)
        g = roi_grid(cfg, u0, 0.5 * (1 / r0 + 1 / r1), hu, 0.5 * (1 / r0 - 1 / r1) / full.cell_w,
                     r_min=r0, r_max=r1)
    else:
        raise ValueError(grid_spec)
    return Music2D(cfg, g, refine=refine)


@lru_cache(maxsize=4)
def build_b0(N: int) -> MusicFarField:
    return MusicFarField(ArrayCfg(N), 30 * D2R, 150 * D2R)


def trial_music(
    rng: np.random.Generator,
    N: int,
    T: int,
    snr_db: float,
    targets: list,
    grid_spec: tuple,
    waveform: str = "orth",
    with_b0: bool = False,
    random_phase: bool = True,
) -> dict:
    """One trial: simulate echo, run B1-r (which also yields B1), optionally B0.

    With ``random_phase`` (default, spec §1.3) the reflection phases are drawn i.i.d.
    uniform per trial; the phases in ``targets`` are then ignored.  This matters: the
    smallest signal eigenvalue and the CRB of close pairs depend on arg(b1 b2* c*^2).

    Returns errors (K, 4) for B1 and B1-r [dr, dtheta, du, dw] (NaN if count != K),
    resolved flags and runtimes.
    """
    cfg = ArrayCfg(N)
    if random_phase:
        targets = [(r, th, float(rng.uniform(0, 2 * np.pi))) for r, th, _ in targets]
    tg = targets_from_spec(targets)
    K = len(tg)
    X = make_waveform(cfg, T, kind=waveform, rng=rng)
    Y = simulate_echo(cfg, tg, X, noise_var_from_snr(snr_db), rng)
    est = build_music(N, tuple(grid_spec), True)
    cu, cw = est.grid.cell_u, est.grid.cell_w
    res = est.fit(Y, K)
    out = {}
    for name, tl in (("B1", res.extra["on_grid"]), ("B1-r", res.targets)):
        e = matched_errors(tl, tg, cu, cw)
        out[f"err_{name}"] = np.full((K, 4), np.nan) if e is None else e
        out[f"res_{name}"] = is_resolved(tl, tg, cu, cw)
    out["time_B1-r"] = res.runtime_s
    if with_b0:
        r0 = build_b0(N).fit(Y, K)
        out["res_B0"] = is_resolved(r0.targets, tg, cu, cw, axes=("u",))
        out["res_B0_uw"] = is_resolved(r0.targets, tg, cu, cw)
    return out

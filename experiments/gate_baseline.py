"""Baseline gate (KE_HOACH_4_TUAN.md §5, việc #15).

1. Near-field evidence at N = 257: co-directional targets at 10 m and 25 m (theta = 90 deg):
   B0 (far-field) must fail, B1 must succeed.
2. Empirical probe of P_res(B1, B1-r) vs (separation in cells, SNR) for same-angle and
   same-range pairs, to fix the sweep ranges of F4/F5 (compare with SRL_calibration).

Run:  python -m experiments.gate_baseline [--trials 40]
Output: results/gate_baseline.json and a printed table.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.common import CACHE, D2R, RESULTS, trial_music
from nfisac.config import ArrayCfg
from nfisac.grids import cell_u, cell_w
from nfisac.mc import run_mc
from nfisac.metrics import p_res

N, T = 257, 512
PROBE_SNRS = (-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0)


def pair_same_angle(r0: float, th_deg: float, d_cells: float, cw: float) -> list:
    dw = d_cells * cw
    return [(1 / (1 / r0 - dw / 2), th_deg, 0.0), (1 / (1 / r0 + dw / 2), th_deg, np.pi / 3)]


def pair_same_range(r0: float, th_deg: float, d_cells: float, cu: float) -> list:
    u0 = np.cos(th_deg * D2R)
    return [(r0, np.rad2deg(np.arccos(u0 - d_cells * cu / 2)), 0.0),
            (r0, np.rad2deg(np.arccos(u0 + d_cells * cu / 2)), np.pi / 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = ArrayCfg(N)
    cu, cw = cell_u(cfg), cell_w(cfg)
    out: dict = {"N": N, "T": T, "cell_u": cu, "cell_w": cw}

    # 1. gate: B0 vs B1, 10 m & 25 m same angle, SNR -20 dB (integrated ~ +31 dB)
    tg = [(10.0, 90.0, 0.0), (25.0, 90.0, 1.0)]
    gate = {}
    for snr in (-30.0, -20.0):
        r = run_mc(trial_music, dict(N=N, T=T, snr_db=snr, targets=tg,
                                     grid_spec=("band", 0.0, 12, 3.0, 60.0), with_b0=True),
                   args.trials, seed=1, n_jobs=args.jobs, cache_dir=CACHE, tag="gate")
        gate[snr] = {k: p_res(r[f"res_{k}"]).__dict__ for k in ("B0", "B1", "B1-r")}
        print(f"gate SNR {snr:+.0f} dB:", {k: round(v["p"], 3) for k, v in gate[snr].items()})
    out["gate"] = gate

    # 2. probe
    probe = []
    for kind in ("range", "angle"):
        for snr in PROBE_SNRS:
            for dc in (0.1, 0.2, 0.3, 0.5, 1.0):
                if kind == "range":
                    tg = pair_same_angle(10.0, 90.0, dc, cw)
                    spec = ("roi", 0.0, 0.1, 8, dc / 2 + 3)
                else:
                    tg = pair_same_range(10.0, 90.0, dc, cu)
                    spec = ("roi", 0.0, 0.1, dc / 2 + 6, 3)
                r = run_mc(trial_music, dict(N=N, T=T, snr_db=snr, targets=tg, grid_spec=spec),
                           args.trials, seed=2, n_jobs=args.jobs, cache_dir=CACHE, tag="probe")
                row = {"kind": kind, "snr": snr, "d_cells": dc,
                       "B1": p_res(r["res_B1"]).p, "B1-r": p_res(r["res_B1-r"]).p}
                probe.append(row)
    out["probe"] = probe
    print("\nP_res probe (B1 / B1-r), N=257, T=512, r0=10 m, theta=90 deg")
    for kind in ("range", "angle"):
        print(f"[{kind}]  SNR\\d  " + "  ".join(f"{d:>9}" for d in (0.1, 0.2, 0.3, 0.5, 1.0)))
        for snr in PROBE_SNRS:
            cells = [f"{p['B1']:.2f}/{p['B1-r']:.2f}" for p in probe if p["kind"] == kind and p["snr"] == snr]
            print(f"         {snr:+.0f}   " + "  ".join(f"{c:>9}" for c in cells))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gate_baseline.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

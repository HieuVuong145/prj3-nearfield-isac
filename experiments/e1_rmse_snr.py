"""F2 - RMSE vs SNR for one target, compared with sqrt(CRB) (known X) and the grid floor.

Draft: B1 and B1-r only (P1, B3, SBL are added by tasks #24, #29).
Search region: +-20 angle cells around the true u, full w range [1/60, 1/3] (so that the
global-search threshold effect is visible), fine grid ov = 8.

Run:  python -m experiments.e1_rmse_snr [--trials 100]
Output: results/figs/F2_rmse_snr.png, results/F2_rmse_snr.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from experiments.common import (
    CACHE,
    D2R,
    FIGS,
    RESULTS,
    build_music,
    targets_from_spec,
    trial_music,
)
from experiments.plotstyle import algo_line, ref_line, setup
from nfisac.config import ArrayCfg
from nfisac.crb import crb
from nfisac.mc import run_mc
from nfisac.metrics import rmse
from nfisac.signal import noise_var_from_snr

N, T = 257, 512
R0, TH0 = 15.0, 60.0
SNRS = np.arange(-50.0, 1.0, 5.0)


def main() -> None:
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    cfg = ArrayCfg(N)
    spec = [(R0, TH0, 0.0)]
    tg = targets_from_spec(spec)
    # Offset the band centre by 0.37 grid steps so that the target is NOT on a grid point.
    u_c = float(np.cos(TH0 * D2R) + 0.37 * (2.0 / N) / 8)
    grid_spec = ("band", u_c, 20, 3.0, 60.0)
    g = build_music(N, grid_spec, True).grid

    rows = []
    rm = {"B1": [], "B1-r": []}
    crb_r, crb_t = [], []
    for snr in SNRS:
        r = run_mc(trial_music, dict(N=N, T=T, snr_db=float(snr), targets=spec, grid_spec=grid_spec),
                   args.trials, seed=10, n_jobs=args.jobs, cache_dir=CACHE, tag="F2")
        for name in rm:
            e = r[f"err_{name}"][:, 0, :]  # (trials, 4)
            rm[name].append(rmse(e))
        c = crb(cfg, tg, np.eye(N) / N, T, noise_var_from_snr(float(snr)))
        crb_r.append(c.sqrt_crb_r[0])
        crb_t.append(np.rad2deg(c.sqrt_crb_theta[0]))
        rows.append([snr, *[rm[n][-1][0] for n in rm], crb_r[-1],
                     *[np.rad2deg(rm[n][-1][1]) for n in rm], crb_t[-1]])
        print(f"SNR {snr:+5.0f} dB | RMSE r: B1 {rm['B1'][-1][0]:.4f}  B1-r {rm['B1-r'][-1][0]:.4f}"
              f"  sqrtCRB {crb_r[-1]:.4f} m | ratio B1-r/CRB {rm['B1-r'][-1][0] / crb_r[-1]:.2f}")

    # grid floor (uniform quantisation): step/sqrt(12), mapped to r and theta
    floor_r = R0**2 * g.step_w / np.sqrt(12)
    floor_t = np.rad2deg(g.step_u / np.sin(TH0 * D2R) / np.sqrt(12))

    setup()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, cr, fl, yl in ((axs[0], 0, crb_r, floor_r, "RMSE(r) (m)"),
                                (axs[1], 1, crb_t, floor_t, "RMSE(θ) (độ)")):
        for name in ("B1", "B1-r"):
            y = np.array([v[col] for v in rm[name]])
            algo_line(ax, SNRS, y if col == 0 else np.rad2deg(y), name)
        ref_line(ax, SNRS, cr, "√CRB (X đã biết)", "--")
        ax.axhline(fl, color="#52514e", ls=":", lw=1.5, label="Sàn lưới bước/√12")
        ax.set_yscale("log")
        ax.set_xlabel("SNR mỗi phần tử, mỗi snapshot (dB)")
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    fig.suptitle(f"1 mục tiêu ({R0:.0f} m, {TH0:.0f}°), N = {N}, T = {T}, X trực giao, "
                 f"{args.trials} lần MC  [bản nháp]", fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "F2_rmse_snr.png")
    with open(RESULTS / "F2_rmse_snr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "rmse_r_B1", "rmse_r_B1r", "sqrt_crb_r", "rmse_th_B1_deg",
                    "rmse_th_B1r_deg", "sqrt_crb_th_deg"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

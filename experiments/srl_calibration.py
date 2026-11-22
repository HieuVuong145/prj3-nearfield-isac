"""Statistical resolution limit (SRL) from the multi-target CRB -> sweep ranges for F4/F5.

Criterion (Smith 2005; El Korso et al. 2012): a separation delta is resolvable when
|delta| >= sqrt(CRB_delta).  Since CRB ∝ sigma^2, the minimum SNR is closed-form:

    SNR_min(delta) = 20 log10( sqrt(CRB_delta)|_{0 dB} / |delta| )   [dB]

The bound depends on the relative reflection phase; we report the median over
N_PHASE uniformly spaced phases (the experiments draw phases uniformly at random).

Run:  python -m experiments.srl_calibration
Output: results/figs/SRL_calibration.png, results/srl_calibration.csv
"""
from __future__ import annotations

import csv

import numpy as np

from experiments.common import D2R, FIGS, RESULTS
from experiments.plotstyle import INK2, setup
from nfisac.config import ArrayCfg, Target
from nfisac.crb import crb, crb_separation
from nfisac.grids import cell_u, cell_w

DELTA_CELLS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
N_PHASE = 16
PHASES = np.arange(N_PHASE) * 2 * np.pi / N_PHASE
N_COLORS = {129: "#1baf7a", 257: "#2a78d6", 513: "#6250d6"}


def snr_min_range(N: int, T: int, r0: float, theta_deg: float, d_cells: np.ndarray) -> np.ndarray:
    """Same angle; targets at w0 -/+ delta/2 (delta in broadside w-cells)."""
    cfg = ArrayCfg(N)
    cw = cell_w(cfg)
    out = []
    for dc in d_cells:
        dw = dc * cw
        w1, w2 = 1 / r0 - dw / 2, 1 / r0 + dw / 2
        v = []
        for ph in PHASES:
            tg = [Target(1 / w1, theta_deg * D2R, 1.0), Target(1 / w2, theta_deg * D2R, np.exp(1j * ph))]
            res = crb(cfg, tg, np.eye(N) / N, T, 1.0)
            v.append(20 * np.log10(crb_separation(res, tg, coord="w") / dw))
        out.append(np.median(v))
    return np.array(out)


def snr_min_angle(N: int, T: int, r0: float, theta_deg: float, d_cells: np.ndarray) -> np.ndarray:
    """Same range; targets at u0 -/+ delta/2 (delta in u-cells 2/N)."""
    cfg = ArrayCfg(N)
    cu = cell_u(cfg)
    u0 = np.cos(theta_deg * D2R)
    out = []
    for dc in d_cells:
        du = dc * cu
        v = []
        for ph in PHASES:
            tg = [Target(r0, np.arccos(u0 - du / 2), 1.0), Target(r0, np.arccos(u0 + du / 2), np.exp(1j * ph))]
            res = crb(cfg, tg, np.eye(N) / N, T, 1.0)
            v.append(20 * np.log10(crb_separation(res, tg, coord="u") / du))
        out.append(np.median(v))
    return np.array(out)


def main() -> None:
    import matplotlib.pyplot as plt

    setup()
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for N in (129, 257, 513):
        T = 2 * N - 2 if N > 256 else 256
        T = max(T, N)
        for th, ls in ((90, "-"), (60, "--")):
            s = snr_min_range(N, T, 10.0, th, DELTA_CELLS)
            axs[0].plot(DELTA_CELLS, s, ls, color=N_COLORS[N], marker="o", label=f"N={N}, θ={th}°")
            rows += [("range", N, T, th, 10.0, d, v) for d, v in zip(DELTA_CELLS, s, strict=True)]
        s = snr_min_angle(N, T, 10.0, 90, DELTA_CELLS)
        axs[1].plot(DELTA_CELLS, s, color=N_COLORS[N], marker="o", label=f"N={N}")
        rows += [("angle", N, T, 90, 10.0, d, v) for d, v in zip(DELTA_CELLS, s, strict=True)]

    for ax, xl, t in (
        (axs[0], "Độ tách Δw (đơn vị ô cell_w(90°))", "(a) Cùng góc, r₀ = 10 m"),
        (axs[1], "Độ tách Δu (đơn vị ô 2/N)", "(b) Cùng khoảng cách r = 10 m, θ = 90°"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_title(t, fontsize=10)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("SNR tối thiểu theo SRL (dB)")
    fig.suptitle("Giới hạn phân giải thống kê |δ| ≥ √CRB_δ (X trực giao, T ≈ 2N, |β₁|=|β₂|, trung vị theo pha)", fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "SRL_calibration.png")

    with open(RESULTS / "srl_calibration.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["axis", "N", "T", "theta_deg", "r0", "delta_cells", "snr_min_db"])
        wr.writerows(rows)
    print("axis   N    T  theta  " + "  ".join(f"{d:>5}" for d in DELTA_CELLS))
    for axis in ("range", "angle"):
        for N in (129, 257, 513):
            for th in (90, 60):
                v = [r[6] for r in rows if r[0] == axis and r[1] == N and r[3] == th]
                if v:
                    T = [r[2] for r in rows if r[0] == axis and r[1] == N][0]
                    print(f"{axis:6} {N:3} {T:4} {th:5}  " + "  ".join(f"{x:5.1f}" for x in v))


if __name__ == "__main__":
    main()

"""F1 - sanity: MUSIC spectrum of one target (N=129) and the N=65 reference case.

Run:  python -m experiments.e0_sanity
Output: results/figs/F1_sanity.png, printed peak locations.
"""
from __future__ import annotations

import numpy as np

from experiments.common import D2R, FIGS
from experiments.plotstyle import INK, SEQ, setup
from nfisac.config import ArrayCfg, Target
from nfisac.estimators import Music2D
from nfisac.grids import polar_grid
from nfisac.signal import make_waveform, noise_var_from_snr, simulate_echo


def panel(ax, cfg, grid, tg, snr_db, rng, title):
    import matplotlib.pyplot as plt

    X = make_waveform(cfg, 256, kind="orth", rng=rng)
    Y = simulate_echo(cfg, tg, X, noise_var_from_snr(snr_db), rng)
    res = Music2D(cfg, grid, refine=True, keep_spectrum=True).fit(Y, len(tg))
    P = res.extra["spectrum"]
    PdB = 10 * np.log10(P / P.max())
    th = np.rad2deg(np.arccos(grid.u))
    r = 1 / grid.w
    m = ax.pcolormesh(th, r, PdB.T, cmap=SEQ, vmin=-50, vmax=0, shading="auto")
    for t in tg:
        ax.plot(np.rad2deg(t.theta), t.r, "o", mfc="none", mec=INK, ms=11, mew=1.5)
    for t in res.targets:
        ax.plot(np.rad2deg(t.theta), t.r, "+", color="#eb6834", ms=11, mew=2)
    ax.set_xlabel("θ (độ, tính từ trục mảng)")
    ax.set_ylabel("r (m)")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    plt.colorbar(m, ax=ax, label="Phổ MUSIC chuẩn hoá (dB)")
    return res


def main() -> None:
    import matplotlib.pyplot as plt

    setup()
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    cfg = ArrayCfg(129)
    tg = [Target(15.0, 60 * D2R)]
    g = polar_grid(cfg, 5.0, 40.0, 56 * D2R, 64 * D2R)
    r1 = panel(axs[0], cfg, g, tg, 0.0, rng, "(a) N = 129, d = λ/2, mục tiêu (15 m, 60°), SNR 0 dB")

    lam = 299_792_458.0 / 28e9
    cfg65 = ArrayCfg(65, d_over_lambda=0.5 / 64 / lam)
    tg65 = [Target(20.0, 45 * D2R)]
    g65 = polar_grid(cfg65, 5.0, 40.0, 40 * D2R, 50 * D2R)
    r2 = panel(axs[1], cfg65, g65, tg65, 20.0, rng, "(b) Tái tạo: N = 65, D = 0.5 m (d ≈ 0.73λ), (20 m, 45°)")

    fig.text(0.01, 0.01, "○ vị trí thật   + ước lượng B1-r", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "F1_sanity.png")
    for name, res in (("N=129", r1), ("N=65", r2)):
        t = res.targets[0]
        print(f"{name}: r_hat = {t.r:.3f} m, theta_hat = {np.rad2deg(t.theta):.3f} deg")


if __name__ == "__main__":
    main()

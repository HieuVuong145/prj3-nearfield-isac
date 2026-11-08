"""Configuration dataclasses and YAML loading utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ArrayCfg:
    """ULA antenna array configuration.

    Parameters
    ----------
    N : int
        Number of antenna elements (must be odd).
    fc : float, default=28.0e9
        Carrier frequency in Hz.
    d_over_lambda : float, default=0.5
        Inter-element spacing normalized by wavelength (d / lambda).
    c : float, default=299_792_458.0
        Speed of light in m/s.
    """

    N: int
    fc: float = 28.0e9
    d_over_lambda: float = 0.5
    c: float = 299_792_458.0

    def __post_init__(self) -> None:
        if self.N % 2 == 0:
            raise ValueError(f"Number of antennas N must be odd, got {self.N}")
        object.__setattr__(self, "N", int(self.N))
        object.__setattr__(self, "fc", float(self.fc))
        object.__setattr__(self, "d_over_lambda", float(self.d_over_lambda))
        object.__setattr__(self, "c", float(self.c))

    @property
    def lam(self) -> float:
        """Carrier wavelength lambda in meters."""
        return self.c / self.fc

    @property
    def d(self) -> float:
        """Antenna spacing in meters."""
        return self.lam * self.d_over_lambda

    @property
    def D(self) -> float:
        """Total array aperture in meters: (N - 1) * d."""
        return (self.N - 1) * self.d

    @property
    def rayleigh(self) -> float:
        """Rayleigh distance (Fraunhofer distance) in meters: 2 * D^2 / lambda."""
        return 2.0 * self.D**2 / self.lam

    @property
    def n(self) -> np.ndarray:
        """Antenna element indices centered at 0: np.arange(-(N // 2), N // 2 + 1)."""
        return np.arange(-(self.N // 2), self.N // 2 + 1)

    @property
    def n_half(self) -> int:
        """Half-array index limit: N // 2."""
        return self.N // 2


@dataclass
class Target:
    """Target parameter container.

    Parameters
    ----------
    r : float
        Distance from array center in meters.
    theta : float
        Azimuth angle in radians (theta = pi / 2 is broadside).
    beta : complex, default=1.0+0.0j
        Complex reflection coefficient / channel path gain.
    """

    r: float
    theta: float
    beta: complex = 1.0 + 0.0j


@dataclass(frozen=True)
class SignalCfg:
    """Transmitted signal configuration.

    Parameters
    ----------
    T : int, default=256
        Number of snapshots.
    P : float, default=1.0
        Total transmit power.
    waveform : str, default="iso"
        Waveform type: "iso" (isotropic, full rank), "beam" (rank-1 directional), "mix".
    snr_db : float, default=10.0
        Signal-to-noise ratio in dB.
    """

    T: int = 256
    P: float = 1.0
    waveform: str = "iso"
    snr_db: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "T", int(self.T))
        object.__setattr__(self, "P", float(self.P))
        object.__setattr__(self, "snr_db", float(self.snr_db))


@dataclass(frozen=True)
class TargetsRegionCfg:
    """Region of interest for target localization.

    Parameters
    ----------
    r_min : float, default=3.0
        Minimum distance in meters.
    r_max : float, default=60.0
        Maximum distance in meters.
    theta_deg: list[float] = field(default_factory=lambda: [30.0, 150.0])
        Azimuth angle interval [theta_min, theta_max] in degrees.
    """

    r_min: float = 3.0
    r_max: float = 60.0
    theta_deg: list[float] = field(default_factory=lambda: [30.0, 150.0])

    def __post_init__(self) -> None:
        object.__setattr__(self, "r_min", float(self.r_min))
        object.__setattr__(self, "r_max", float(self.r_max))
        object.__setattr__(self, "theta_deg", [float(x) for x in self.theta_deg])


@dataclass(frozen=True)
class GridCfg:
    """Spatial search grid parameters.

    Parameters
    ----------
    music_mu : float, default=0.95
        Coherence between adjacent grid points in fine 2D-MUSIC grid.
    coarse_mu : float, default=0.7
        Coherence in coarse grid (hierarchical MUSIC).
    cs_mu : float, default=0.5
        Coherence bound for compressive sensing dictionary.
    u_oversample : int, default=2
        Oversampling factor for angle grid u = cos(theta).
    """

    music_mu: float = 0.95
    coarse_mu: float = 0.7
    cs_mu: float = 0.5
    u_oversample: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "music_mu", float(self.music_mu))
        object.__setattr__(self, "coarse_mu", float(self.coarse_mu))
        object.__setattr__(self, "cs_mu", float(self.cs_mu))
        object.__setattr__(self, "u_oversample", int(self.u_oversample))


@dataclass(frozen=True)
class McCfg:
    """Monte-Carlo simulation configuration.

    Parameters
    ----------
    n_trials : int, default=200
        Number of Monte-Carlo trials.
    seed : int, default=0
        Base random seed.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all available cores).
    """

    n_trials: int = 200
    seed: int = 0
    n_jobs: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_trials", int(self.n_trials))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "n_jobs", int(self.n_jobs))


@dataclass(frozen=True)
class ProjectCfg:
    """Master project configuration combining all sub-configurations."""

    array: ArrayCfg
    signal: SignalCfg
    targets_region: TargetsRegionCfg
    grid: GridCfg
    mc: McCfg


def load_config(path: str | Path) -> ProjectCfg:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    ProjectCfg
        Structured project configuration.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    array_cfg = ArrayCfg(**data.get("array", {}))
    signal_cfg = SignalCfg(**data.get("signal", {}))
    targets_region_cfg = TargetsRegionCfg(**data.get("targets_region", {}))
    grid_cfg = GridCfg(**data.get("grid", {}))
    mc_cfg = McCfg(**data.get("mc", {}))

    return ProjectCfg(
        array=array_cfg,
        signal=signal_cfg,
        targets_region=targets_region_cfg,
        grid=grid_cfg,
        mc=mc_cfg,
    )

"""Monte-Carlo runner: per-trial seeds, process parallelism, .npz cache keyed by config hash.

Reproducibility contract: trial i always receives ``np.random.default_rng(child_i)`` with
``child_i = SeedSequence(seed).spawn(n_trials)[i]``, so results are bit-identical for any
``n_jobs`` and across reruns.

The trial function must be a module-level (picklable) callable

    trial_fn(rng: np.random.Generator, **params) -> dict[str, float | np.ndarray]

returning arrays of a fixed shape per key.  Results are stacked along a new first axis.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def config_hash(obj: Any) -> str:
    """Stable short hash of a JSON-serialisable configuration."""
    s = json.dumps(obj, sort_keys=True, default=repr)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _run_one(args: tuple[Callable[..., dict], np.random.SeedSequence, dict]) -> dict:
    fn, ss, params = args
    return fn(np.random.default_rng(ss), **params)


def _stack(results: list[dict]) -> dict[str, np.ndarray]:
    keys = results[0].keys()
    return {k: np.stack([np.asarray(r[k]) for r in results]) for k in keys}


def run_mc(
    trial_fn: Callable[..., dict],
    params: dict,
    n_trials: int,
    seed: int = 0,
    n_jobs: int = 1,
    cache_dir: str | Path | None = None,
    tag: str = "",
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Run ``n_trials`` independent trials of ``trial_fn(rng, **params)``.

    Parameters
    ----------
    params : dict
        Keyword arguments (must be picklable; JSON-serialisable part is used for the hash).
    n_jobs : int
        1 = serial; -1 = all cores.  Worker BLAS threads are pinned to 1.
    cache_dir : path or None
        If given, results are stored as ``<cache_dir>/<tag>_<hash>.npz`` and reloaded.
    """
    key = config_hash(
        {"fn": f"{trial_fn.__module__}.{trial_fn.__qualname__}", "params": params,
         "n": n_trials, "seed": seed}
    )
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{tag + '_' if tag else ''}{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                return {k: f[k] for k in f.files}

    children = np.random.SeedSequence(seed).spawn(n_trials)
    jobs = [(trial_fn, ss, params) for ss in children]
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if n_jobs == 1:
        it = map(_run_one, jobs)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=n_trials, desc=tag or "mc")
        results = list(it)
    else:
        saved = {v: os.environ.get(v) for v in _THREAD_VARS}
        for v in _THREAD_VARS:
            os.environ[v] = "1"
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                it = ex.map(_run_one, jobs, chunksize=max(1, n_trials // (4 * n_jobs)))
                if progress:
                    from tqdm import tqdm

                    it = tqdm(it, total=n_trials, desc=tag or "mc")
                results = list(it)
        finally:
            for v, val in saved.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val

    out = _stack(results)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
    return out


__all__ = ["config_hash", "run_mc"]

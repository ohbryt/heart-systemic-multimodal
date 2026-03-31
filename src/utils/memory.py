"""
RAM-aware utility functions for the Heart-Systemic Multi-modal Pipeline.

Provides helpers for:
    - Querying available system RAM.
    - Downsampling AnnData objects to prevent OOM errors.
    - Disk-backed caching of expensive load operations.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# RAM utilities
# ---------------------------------------------------------------------------


def check_available_ram(unit: str = "GB") -> float:
    """Return the amount of currently available system RAM.

    Uses ``psutil`` when available; falls back to parsing ``/proc/meminfo``
    on Linux, or returns ``float('inf')`` on unsupported platforms.

    Args:
        unit: Output unit — one of ``"B"``, ``"KB"``, ``"MB"``, ``"GB"``.
            Defaults to ``"GB"``.

    Returns:
        Available RAM as a float in the requested unit.

    Example::

        ram_gb = check_available_ram()
        if ram_gb < 8:
            logger.warning("Low RAM: %.1f GB available", ram_gb)
    """
    divisors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if unit not in divisors:
        raise ValueError(f"Unknown unit '{unit}'. Choose from {list(divisors)}")
    div = divisors[unit]

    available_bytes: float | None = None

    # Try psutil first (most portable).
    try:
        import psutil  # type: ignore[import]
        available_bytes = float(psutil.virtual_memory().available)
    except ImportError:
        pass

    # Linux fallback: /proc/meminfo
    if available_bytes is None:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                with meminfo.open() as fh:
                    for line in fh:
                        if line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            available_bytes = float(kb * 1024)
                            break
            except (OSError, ValueError) as exc:
                logger.debug("Could not parse /proc/meminfo: %s", exc)

    if available_bytes is None:
        logger.debug("Cannot determine available RAM on this platform; returning inf.")
        return float("inf")

    return available_bytes / div


def estimate_adata_ram(adata: Any, unit: str = "GB") -> float:
    """Estimate in-memory footprint of an AnnData object.

    Sums the nbytes of ``.X``, ``.obs``, ``.var``, and ``obsm`` layers.

    Args:
        adata: An :class:`anndata.AnnData` instance.
        unit: Output unit.  Defaults to ``"GB"``.

    Returns:
        Estimated RAM usage as a float.
    """
    divisors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if unit not in divisors:
        raise ValueError(f"Unknown unit '{unit}'.")

    total = 0
    try:
        import scipy.sparse as sp

        X = adata.X
        if sp.issparse(X):
            total += X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        elif hasattr(X, "nbytes"):
            total += X.nbytes

        if hasattr(adata, "obs") and adata.obs is not None:
            total += adata.obs.memory_usage(deep=True).sum()
        if hasattr(adata, "var") and adata.var is not None:
            total += adata.var.memory_usage(deep=True).sum()
        for key, arr in (adata.obsm or {}).items():
            if hasattr(arr, "nbytes"):
                total += arr.nbytes
    except Exception as exc:  # noqa: BLE001
        logger.debug("RAM estimation failed: %s", exc)

    return total / divisors[unit]


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def downsample_adata(
    adata: Any,
    max_cells: int,
    random_seed: int = 42,
    stratify_key: str | None = None,
) -> Any:
    """Randomly subsample an AnnData to at most *max_cells* observations.

    If ``adata.n_obs <= max_cells``, the original object is returned unchanged.
    Optionally performs stratified sampling to preserve cell-type proportions.

    Args:
        adata: An :class:`anndata.AnnData` instance.
        max_cells: Maximum number of cells to retain.
        random_seed: Seed for reproducibility.
        stratify_key: ``.obs`` column to use for stratified sampling.
            If ``None``, simple random sampling is used.

    Returns:
        A view (or copy) of the input AnnData with at most *max_cells* rows.

    Example::

        adata_small = downsample_adata(adata, max_cells=50_000, stratify_key="cell_type")
    """
    if adata.n_obs <= max_cells:
        logger.debug(
            "downsample_adata: n_obs=%d <= max_cells=%d, skipping.", adata.n_obs, max_cells
        )
        return adata

    rng = np.random.default_rng(random_seed)
    logger.info(
        "Downsampling AnnData from %d to %d cells (seed=%d).",
        adata.n_obs,
        max_cells,
        random_seed,
    )

    if stratify_key is not None and stratify_key in adata.obs.columns:
        groups = adata.obs[stratify_key].values
        unique_groups, counts = np.unique(groups, return_counts=True)
        fractions = counts / counts.sum()
        per_group = np.maximum(1, (fractions * max_cells).astype(int))
        # Clamp to actual group sizes.
        per_group = np.minimum(per_group, counts)

        selected_indices: list[int] = []
        for group, n_select in zip(unique_groups, per_group):
            idx = np.where(groups == group)[0]
            chosen = rng.choice(idx, size=int(n_select), replace=False)
            selected_indices.extend(chosen.tolist())

        selected_indices = sorted(selected_indices)
        logger.debug(
            "Stratified sampling: %d groups, retained %d cells.",
            len(unique_groups),
            len(selected_indices),
        )
        return adata[selected_indices].copy()
    else:
        selected = rng.choice(adata.n_obs, size=max_cells, replace=False)
        selected = np.sort(selected)
        return adata[selected].copy()


# ---------------------------------------------------------------------------
# Disk-backed caching
# ---------------------------------------------------------------------------


def _cache_key(path: Path, loader_fn: Callable) -> str:
    """Generate a cache key based on file path, mtime, and loader identity.

    Args:
        path: Path to the source data file.
        loader_fn: The loader callable — its qualified name is included in the key.

    Returns:
        A hex digest string suitable for use as a filename stem.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0

    raw = f"{path.resolve()}|{mtime}|{loader_fn.__qualname__}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cached_load(
    path: str | Path,
    loader_fn: Callable[..., T],
    cache_dir: str | Path | None = None,
    force_reload: bool = False,
    *,
    loader_kwargs: dict[str, Any] | None = None,
) -> T:
    """Load data from *path* using *loader_fn*, caching the result to disk.

    On first call the loader is invoked and the result is pickled to
    *cache_dir*.  Subsequent calls with the same path+mtime return the cached
    object instantly, skipping the loader.

    Args:
        path: Path to the source data file.
        loader_fn: Callable that accepts *path* (and optional **loader_kwargs**)
            and returns the loaded data object.
        cache_dir: Directory to store ``.pkl`` cache files.  If ``None``,
            caching is disabled and the loader is always called directly.
        force_reload: If ``True``, ignore existing cache and reload from source.
        loader_kwargs: Additional keyword arguments forwarded to *loader_fn*.

    Returns:
        The loaded (or cached) data object.

    Example::

        import scanpy as sc
        adata = cached_load(
            "data/raw/sample.h5ad",
            sc.read_h5ad,
            cache_dir="data/cache",
        )
    """
    path = Path(path)
    loader_kwargs = loader_kwargs or {}

    if cache_dir is None:
        logger.debug("cached_load: caching disabled, loading directly from %s", path)
        return loader_fn(path, **loader_kwargs)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(path, loader_fn)
    cache_file = cache_dir / f"{path.stem}_{key}.pkl"

    if cache_file.exists() and not force_reload:
        logger.info("Loading from cache: %s", cache_file)
        t0 = time.perf_counter()
        try:
            with cache_file.open("rb") as fh:
                obj = pickle.load(fh)
            elapsed = time.perf_counter() - t0
            logger.debug("Cache hit: loaded in %.2fs", elapsed)
            return obj  # type: ignore[return-value]
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            logger.warning("Cache file corrupt (%s); reloading from source.", exc)

    logger.info("Cache miss — loading from source: %s", path)
    t0 = time.perf_counter()
    obj = loader_fn(path, **loader_kwargs)
    elapsed = time.perf_counter() - t0
    logger.debug("Source load completed in %.2fs", elapsed)

    # Persist to cache.
    try:
        with cache_file.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Cached result to: %s", cache_file)
    except (OSError, pickle.PicklingError) as exc:
        logger.warning("Could not write cache file: %s", exc)

    return obj  # type: ignore[return-value]


def clear_cache(cache_dir: str | Path, pattern: str = "*.pkl") -> int:
    """Remove all cache files matching *pattern* from *cache_dir*.

    Args:
        cache_dir: Directory containing cache files.
        pattern: Glob pattern for cache files.  Defaults to ``"*.pkl"``.

    Returns:
        Number of files deleted.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        logger.debug("clear_cache: directory does not exist: %s", cache_dir)
        return 0

    deleted = 0
    for f in cache_dir.glob(pattern):
        try:
            f.unlink()
            deleted += 1
        except OSError as exc:
            logger.warning("Could not delete cache file %s: %s", f, exc)

    logger.info("Cleared %d cache file(s) from %s", deleted, cache_dir)
    return deleted


def ram_guard(
    adata: Any,
    max_cells: int | None,
    config: dict | None = None,
    stratify_key: str | None = None,
) -> Any:
    """Convenience wrapper: downsample if *max_cells* is set and log RAM state.

    Combines :func:`check_available_ram`, :func:`estimate_adata_ram`, and
    :func:`downsample_adata` into a single call used at data-loading boundaries.

    Args:
        adata: Loaded AnnData object.
        max_cells: Cap on observations.  ``None`` means no downsampling.
        config: Optional config dict; reads ``preprocessing.random_seed`` if present.
        stratify_key: Passed to :func:`downsample_adata`.

    Returns:
        Possibly downsampled AnnData.
    """
    avail = check_available_ram()
    estimated = estimate_adata_ram(adata)
    logger.info(
        "RAM: %.1f GB available | AnnData estimate: %.1f GB (n_obs=%d, n_vars=%d)",
        avail,
        estimated,
        adata.n_obs,
        adata.n_vars,
    )

    if avail < float("inf") and estimated > avail * 0.75:
        logger.warning(
            "AnnData (%.1f GB) may exceed 75%% of available RAM (%.1f GB). "
            "Consider reducing downsample_to in config.",
            estimated,
            avail,
        )

    if max_cells is not None and max_cells > 0:
        seed = 42
        if config:
            seed = config.get("preprocessing", {}).get("random_seed", 42)
        adata = downsample_adata(adata, max_cells=max_cells, random_seed=seed,
                                 stratify_key=stratify_key)

    return adata

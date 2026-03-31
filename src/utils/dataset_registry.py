"""
Dataset registry for the Heart-Systemic Multi-modal Pipeline.

Loads ``config/datasets.yaml`` and provides filtered views of the curated
dataset catalogue.  All dataset entries follow the schema defined in that file:
id, type, category, disease, modality, enabled, priority, url, notes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to the bundled datasets catalogue, relative to this file's location.
_DEFAULT_DATASETS_PATH = (
    Path(__file__).parent.parent.parent / "config" / "datasets.yaml"
)


class DatasetRegistry:
    """Load and query the curated dataset catalogue.

    The catalogue is a YAML file containing a top-level ``datasets`` list.
    Each entry is a dict with at minimum the keys ``id``, ``category``,
    ``modality``, ``disease``, and ``enabled``.

    Args:
        datasets_path: Path to the datasets YAML file.  Defaults to the
            bundled ``config/datasets.yaml``.

    Raises:
        FileNotFoundError: If *datasets_path* does not exist.
        ValueError: If the YAML does not contain a ``datasets`` list.

    Example::

        registry = DatasetRegistry()
        cardiac_sc = registry.filter_by(category="heart_sc", enabled=True)
        gse = registry.get_by_id("GSE183852")
    """

    def __init__(self, datasets_path: str | Path | None = None) -> None:
        self._path = Path(datasets_path or _DEFAULT_DATASETS_PATH).resolve()
        self._datasets: list[dict[str, Any]] = self._load()
        logger.info(
            "DatasetRegistry loaded %d datasets from %s",
            len(self._datasets),
            self._path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        """Parse the YAML file and return the flat list of dataset dicts."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"Datasets catalogue not found at: {self._path}. "
                "Ensure config/datasets.yaml is present."
            )
        with self._path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict) or "datasets" not in raw:
            raise ValueError(
                f"Expected a YAML mapping with a top-level 'datasets' key "
                f"in {self._path}. Got: {type(raw)}"
            )

        entries = raw["datasets"]
        if not isinstance(entries, list):
            raise ValueError(
                f"'datasets' key in {self._path} must be a YAML sequence."
            )

        # Warn about entries missing the required 'id' field.
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                logger.warning("Dataset entry %d is not a mapping — skipped.", i)
                continue
            if "id" not in entry:
                logger.warning(
                    "Dataset entry %d has no 'id' field — it will be "
                    "accessible only via index-based iteration.",
                    i,
                )

        return [e for e in entries if isinstance(e, dict)]

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def get_by_id(self, dataset_id: str) -> dict[str, Any] | None:
        """Return the first dataset entry matching *dataset_id*.

        Comparison is case-insensitive.

        Args:
            dataset_id: The ``id`` value to search for (e.g. ``"GSE183852"``).

        Returns:
            The dataset dict, or ``None`` if no match is found.
        """
        target = dataset_id.lower()
        for entry in self._datasets:
            if str(entry.get("id", "")).lower() == target:
                return entry
        logger.debug("Dataset id '%s' not found in registry.", dataset_id)
        return None

    def filter_by(
        self,
        *,
        category: str | None = None,
        disease: str | None = None,
        modality: str | None = None,
        enabled: bool | None = None,
        dataset_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all datasets matching every supplied filter criterion.

        All arguments are optional; passing no arguments returns all entries.
        String comparisons are case-insensitive substring matches.

        Args:
            category: Match entries whose ``category`` contains this string
                (e.g. ``"heart_sc"``, ``"plasma_ev"``).
            disease: Match entries whose ``disease`` contains this string
                (e.g. ``"heart_failure"``).
            modality: Match entries whose ``modality`` contains this string
                (e.g. ``"snRNA-seq"``).
            enabled: If ``True``, return only enabled datasets; if ``False``,
                only disabled ones; if ``None``, ignore the flag.
            dataset_type: Match entries whose ``type`` field contains this
                string (e.g. ``"geo"``, ``"pride"``).

        Returns:
            List of matching dataset dicts, sorted by ``priority`` ascending.
        """
        results: list[dict[str, Any]] = []

        for entry in self._datasets:
            if category is not None:
                if category.lower() not in str(entry.get("category", "")).lower():
                    continue
            if disease is not None:
                if disease.lower() not in str(entry.get("disease", "")).lower():
                    continue
            if modality is not None:
                if modality.lower() not in str(entry.get("modality", "")).lower():
                    continue
            if enabled is not None:
                if bool(entry.get("enabled", False)) != enabled:
                    continue
            if dataset_type is not None:
                if dataset_type.lower() not in str(entry.get("type", "")).lower():
                    continue
            results.append(entry)

        # Sort by priority (lower number = higher priority); default 99.
        results.sort(key=lambda e: int(e.get("priority", 99)))
        logger.debug(
            "filter_by(category=%r, disease=%r, modality=%r, enabled=%r, "
            "type=%r) → %d results",
            category,
            disease,
            modality,
            enabled,
            dataset_type,
            len(results),
        )
        return results

    def list_enabled(self) -> list[dict[str, Any]]:
        """Return all datasets with ``enabled: true``, sorted by priority.

        Returns:
            List of enabled dataset dicts.
        """
        return self.filter_by(enabled=True)

    def list_categories(self) -> list[str]:
        """Return a sorted list of unique category values in the registry.

        Returns:
            Sorted list of category strings, e.g.
            ``["heart_sc", "heart_spatial", "plasma_ev", "responder",
            "secretome"]``.
        """
        seen: set[str] = set()
        for entry in self._datasets:
            cat = entry.get("category")
            if cat:
                seen.add(str(cat))
        return sorted(seen)

    def list_modalities(self) -> list[str]:
        """Return a sorted list of unique modality values in the registry.

        Returns:
            Sorted list of modality strings.
        """
        seen: set[str] = set()
        for entry in self._datasets:
            mod = entry.get("modality")
            if mod:
                seen.add(str(mod))
        return sorted(seen)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with counts by category and enabled status.

        Returns:
            Dict with keys ``total``, ``enabled``, ``disabled``,
            ``by_category`` (mapping category → count).
        """
        by_cat: dict[str, int] = {}
        enabled_count = 0
        for entry in self._datasets:
            cat = str(entry.get("category", "unknown"))
            by_cat[cat] = by_cat.get(cat, 0) + 1
            if entry.get("enabled", False):
                enabled_count += 1

        return {
            "total": len(self._datasets),
            "enabled": enabled_count,
            "disabled": len(self._datasets) - enabled_count,
            "by_category": dict(sorted(by_cat.items())),
        }

    def __len__(self) -> int:
        return len(self._datasets)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"DatasetRegistry(total={s['total']}, enabled={s['enabled']}, "
            f"categories={list(s['by_category'].keys())})"
        )

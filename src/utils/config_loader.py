"""
Configuration loader for the Heart-Systemic Multi-modal Pipeline.

Handles loading the default YAML config, merging with optional user
overrides, and validating that required paths exist or can be created.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to the bundled default config, relative to this file's package root.
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a deep copy of *base*.

    Scalar values in *override* replace those in *base*; nested dicts are
    merged recursively so that partial overrides do not wipe sibling keys.

    Args:
        base: The baseline configuration dictionary.
        override: User-supplied overrides to layer on top.

    Returns:
        A new dict with all base keys, updated by override values.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_paths(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Expand relative paths inside the ``paths`` block to absolute paths.

    Any path value that is not already absolute is resolved relative to
    *project_root*.

    Args:
        config: Parsed configuration dictionary (mutated in-place).
        project_root: Absolute path to the project root directory.

    Returns:
        The same dict with all ``paths`` values converted to absolute strings.
    """
    paths_block = config.get("paths", {})
    for key, raw in paths_block.items():
        if raw is None:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = project_root / p
        paths_block[key] = str(p)
    config["paths"] = paths_block
    return config


def _ensure_directories(config: dict[str, Any]) -> None:
    """Create output directories referenced in ``paths`` if they do not exist.

    Only creates directories for keys that represent writable output locations
    (results, cache, logs, figures, tables, processed).  Input data directories
    are *not* auto-created so that missing data is surfaced as an error later.

    Args:
        config: Fully resolved configuration dictionary.
    """
    output_keys = {
        "results_dir",
        "cache_dir",
        "logs_dir",
        "figures_dir",
        "tables_dir",
        "processed_dir",
    }
    paths_block = config.get("paths", {})
    for key in output_keys:
        raw = paths_block.get(key)
        if raw:
            p = Path(raw)
            p.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory: %s", p)


def _validate_config(config: dict[str, Any]) -> list[str]:
    """Run basic validation checks on a resolved config dict.

    Checks:
    - Required top-level sections are present.
    - ``paths.data_dir`` exists on disk (warning, not error).
    - At least one dataset is enabled.

    Args:
        config: Fully resolved configuration dictionary.

    Returns:
        A list of warning/error strings.  Empty list means validation passed.
    """
    issues: list[str] = []

    required_sections = ["paths", "datasets", "preprocessing", "scoring", "analysis"]
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required config section: '{section}'")

    # Warn (not error) if raw data dir is absent — user may not have downloaded yet.
    data_dir = config.get("paths", {}).get("data_dir")
    if data_dir and not Path(data_dir).exists():
        issues.append(
            f"WARNING: data_dir '{data_dir}' does not exist. "
            "Run 'hsm download' to fetch datasets."
        )

    # Check at least one dataset enabled.
    datasets = config.get("datasets", {})
    enabled = [k for k, v in datasets.items() if isinstance(v, dict) and v.get("enabled", False)]
    if not enabled:
        issues.append("WARNING: No datasets are enabled in the config.")

    # Validate scoring gene sets are non-empty lists.
    for program, spec in config.get("scoring", {}).items():
        genes = spec.get("genes", []) if isinstance(spec, dict) else []
        if not genes:
            issues.append(f"WARNING: Scoring program '{program}' has no genes defined.")

    return issues


class ConfigLoader:
    """Load, merge, validate, and expose pipeline configuration.

    Usage::

        loader = ConfigLoader(user_config_path="my_config.yaml")
        config = loader.config
        data_dir = loader.get_path("data_dir")

    Args:
        user_config_path: Optional path to a user YAML file.  Values here
            override the defaults without replacing unspecified keys.
        project_root: Root directory for resolving relative paths.  Defaults
            to the current working directory.
    """

    def __init__(
        self,
        user_config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self._project_root = Path(project_root or Path.cwd()).resolve()
        self._config = self._load_and_merge(user_config_path)
        self._post_process()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_and_merge(self, user_path: str | Path | None) -> dict[str, Any]:
        """Load default config then layer user overrides on top."""
        if not _DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Default config not found at: {_DEFAULT_CONFIG_PATH}. "
                "Ensure the package is installed correctly."
            )

        with _DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        logger.debug("Loaded default config from %s", _DEFAULT_CONFIG_PATH)

        if user_path is not None:
            user_path = Path(user_path).resolve()
            if not user_path.exists():
                raise FileNotFoundError(f"User config file not found: {user_path}")
            with user_path.open("r", encoding="utf-8") as fh:
                overrides = yaml.safe_load(fh) or {}
            logger.info("Merging user config from %s", user_path)
            base = _deep_merge(base, overrides)

        return base

    def _post_process(self) -> None:
        """Resolve paths, create output dirs, and run validation."""
        self._config = _resolve_paths(self._config, self._project_root)
        _ensure_directories(self._config)

        issues = _validate_config(self._config)
        for issue in issues:
            if issue.startswith("WARNING"):
                logger.warning(issue)
            else:
                logger.error(issue)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """The fully merged and resolved configuration dictionary."""
        return self._config

    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieve a nested config value using dot-path keys.

        Args:
            *keys: Sequence of nested dict keys, e.g. ``get("preprocessing", "n_top_genes")``.
            default: Fallback value if the key path is absent.

        Returns:
            The config value or *default*.
        """
        node: Any = self._config
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is default:
                return default
        return node

    def get_path(self, key: str) -> Path:
        """Return a resolved ``Path`` for a key in the ``paths`` block.

        Args:
            key: A key within the ``paths`` config section.

        Returns:
            Resolved absolute ``Path``.

        Raises:
            KeyError: If the key is not present in the paths block.
        """
        raw = self._config.get("paths", {}).get(key)
        if raw is None:
            raise KeyError(f"Path key '{key}' not found in config paths block.")
        return Path(raw)

    def enabled_datasets(self) -> dict[str, dict[str, Any]]:
        """Return only the dataset entries where ``enabled: true``.

        Returns:
            Dict mapping dataset name to its configuration sub-dict.
        """
        return {
            name: spec
            for name, spec in self._config.get("datasets", {}).items()
            if isinstance(spec, dict) and spec.get("enabled", False)
        }

    def scoring_gene_sets(self) -> dict[str, list[str]]:
        """Return a dict mapping program name to list of gene symbols.

        Returns:
            E.g. ``{"fibrosis": ["COL1A1", ...], "ECM": [...], ...}``
        """
        result: dict[str, list[str]] = {}
        for program, spec in self._config.get("scoring", {}).items():
            if isinstance(spec, dict):
                result[program] = spec.get("genes", [])
        return result

    def dump(self) -> str:
        """Serialize the current (merged) config back to a YAML string.

        Returns:
            YAML-formatted string representation of the config.
        """
        return yaml.dump(self._config, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def load_config(
    user_config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> ConfigLoader:
    """Load pipeline configuration, merging defaults with optional user overrides.

    This is the primary entry-point for all pipeline modules.

    Args:
        user_config_path: Path to a user YAML override file.  If ``None``,
            only the bundled defaults are used.
        project_root: Root directory for resolving relative paths.  Defaults
            to ``os.getcwd()``.

    Returns:
        A fully initialized :class:`ConfigLoader` instance.

    Example::

        cfg = load_config("my_overrides.yaml")
        data_dir = cfg.get_path("data_dir")
        datasets = cfg.enabled_datasets()
    """
    return ConfigLoader(user_config_path=user_config_path, project_root=project_root)

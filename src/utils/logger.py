"""
Rich-based logging setup for the Heart-Systemic Multi-modal Pipeline.

Provides:
    - Console handler with Rich markup, level-based coloring, and timestamps.
    - File handler writing plain-text logs to results/logs/<timestamp>.log.
    - A module-level ``get_logger()`` factory used throughout the codebase.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOG_LEVEL = logging.INFO
_LOG_FORMAT_FILE = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FILENAME_TEMPLATE = "%Y%m%d_%H%M%S_pipeline.log"

# Sentinel to avoid re-configuring the root logger multiple times.
_LOGGING_CONFIGURED: bool = False

# Rich console instance (stderr so it doesn't pollute piped stdout output).
_console = Console(stderr=True, highlight=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path | None = None,
    log_filename: str | None = None,
    force: bool = False,
) -> Path | None:
    """Configure the root logger with a Rich console handler and optional file handler.

    This function is idempotent: calling it a second time is a no-op unless
    *force=True*.  It should be called once at the start of each CLI command.

    Args:
        level: Logging level as a string (``"DEBUG"``, ``"INFO"``, etc.) or
            ``logging`` integer constant.
        log_dir: Directory in which to create the timestamped log file.
            If ``None``, file logging is disabled.
        log_filename: Override the auto-generated log filename.  Useful for
            deterministic test log paths.
        force: If ``True``, reconfigure even if already set up.

    Returns:
        The absolute ``Path`` to the log file, or ``None`` if file logging
        is disabled.
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force:
        return None

    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), _DEFAULT_LOG_LEVEL)
    else:
        numeric_level = level

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate output on re-configuration.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # ── Rich console handler ──────────────────────────────────────────────
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
        level=numeric_level,
    )
    # RichHandler formats the message itself; we just pass the bare record.
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root_logger.addHandler(rich_handler)

    # ── File handler ──────────────────────────────────────────────────────
    log_file_path: Path | None = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = log_filename or datetime.now().strftime(_LOG_FILENAME_TEMPLATE)
        log_file_path = log_dir / filename

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT_FILE, datefmt=_DATE_FORMAT))
        root_logger.addHandler(file_handler)

    # Silence overly chatty third-party loggers at WARNING unless DEBUG is on.
    if numeric_level > logging.DEBUG:
        for noisy in ("urllib3", "fsspec", "anndata", "numba", "h5py", "tiledb"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True

    logger = logging.getLogger(__name__)
    if log_file_path:
        logger.debug("File logging active: %s", log_file_path)

    return log_file_path


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring the root logger is at least minimally configured.

    If :func:`setup_logging` has not been called yet, a fallback configuration
    with INFO-level console output is applied automatically.

    Args:
        name: Logger name — typically pass ``__name__`` from the calling module.

    Returns:
        A :class:`logging.Logger` instance bound to *name*.

    Example::

        logger = get_logger(__name__)
        logger.info("Processing %d cells", n)
    """
    if not _LOGGING_CONFIGURED:
        # Minimal fallback so modules can log before the CLI sets up logging.
        setup_logging(level="INFO", log_dir=None, force=False)

    return logging.getLogger(name)


def log_section(title: str, logger: Optional[logging.Logger] = None) -> None:
    """Emit a visually prominent section header to the log.

    Uses Rich markup for the console handler and plain text for file logs.

    Args:
        title: Section heading text.
        logger: Logger to use.  If ``None``, uses the root logger.
    """
    _log = logger or logging.getLogger()
    separator = "=" * 60
    _log.info("")
    _log.info(separator)
    _log.info("  %s", title.upper())
    _log.info(separator)
    _log.info("")


def log_config_summary(config: dict, logger: Optional[logging.Logger] = None) -> None:
    """Log a concise summary of the active configuration.

    Args:
        config: The resolved configuration dictionary.
        logger: Logger to use.  If ``None``, uses the root logger.
    """
    _log = logger or logging.getLogger()
    _log.info("Active configuration summary:")
    paths = config.get("paths", {})
    for key, val in paths.items():
        _log.info("  paths.%-20s = %s", key, val)

    datasets = config.get("datasets", {})
    enabled = [k for k, v in datasets.items() if isinstance(v, dict) and v.get("enabled")]
    _log.info("  Enabled datasets (%d): %s", len(enabled), ", ".join(enabled))

    prep = config.get("preprocessing", {})
    _log.info(
        "  n_top_genes=%s  min_cells=%s  downsample_to=%s",
        prep.get("n_top_genes", "?"),
        prep.get("min_cells", "?"),
        prep.get("downsample_to", "None"),
    )

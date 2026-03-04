from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, csv_path: str | Path) -> None:
    ensure_parent(csv_path)
    df.to_csv(csv_path, index=False)


def write_json(payload: Dict[str, Any], path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def now_utc_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()

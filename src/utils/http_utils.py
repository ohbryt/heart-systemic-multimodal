from __future__ import annotations

import time
from typing import Any, Dict

import requests


def get_json_with_retry(
    url: str,
    params: Dict[str, Any] | None = None,
    timeout: int = 30,
    max_retries: int = 2,
    backoff_seconds: float = 0.5,
) -> Dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"transient_status_{resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries - 1:
                break
            time.sleep(backoff_seconds * (2**attempt))
    raise RuntimeError(f"Failed request after retries: {url}") from last_exc

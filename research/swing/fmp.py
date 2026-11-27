"""Minimal client for the Financial Modeling Prep API.

The API key is never stored in this repository. It is read from the environment variable
FMP_API_KEY, or from the file named by FMP_KEY_FILE, and sent as a request header so it
cannot leak through a logged URL.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

BASE = "https://financialmodelingprep.com/stable"
_session = requests.Session()


def _key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key and os.environ.get("FMP_KEY_FILE"):
        key = Path(os.environ["FMP_KEY_FILE"]).read_text().strip()
    if not key:
        raise RuntimeError("Set FMP_API_KEY or FMP_KEY_FILE to use the FMP API.")
    return key


def get(endpoint: str, retries: int = 5, **params):
    """Return parsed JSON, or None when the API has nothing for this request."""
    for attempt in range(retries):
        try:
            r = _session.get(f"{BASE}/{endpoint}", params=params, headers={"apikey": _key()}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:                      # rate limit: wait and retry
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (402, 403, 404):
            return {"_status": r.status_code, "_text": r.text[:200]}
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        try:
            return r.json()
        except ValueError:
            return {"_status": r.status_code, "_text": r.text[:200]}
    return {"_status": "failed"}

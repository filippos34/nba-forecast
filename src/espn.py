"""
src/espn.py — ESPN public API client with on-disk cache, retries and throttle
==============================================================================
All ESPN access goes through `get_json`. Raw responses are cached as gzipped
JSON under data/raw/espn/<kind>/<key>.json.gz so re-runs never refetch data
that can no longer change (completed dates, final box scores).

The scoreboard endpoint only accepts single dates (`dates=YYYYMMDD`); ranges
return HTTP 400 since at least 2026-09-28.
"""

from __future__ import annotations

import gzip
import json
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "espn"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

MIN_INTERVAL = 1.0      # seconds between live requests (~1 req/s)
MAX_RETRIES = 5
BACKOFF_BASE = 2.0      # seconds; doubles each retry, plus jitter

_last_request = 0.0
_session = requests.Session()
# Keep the default python-requests User-Agent: ESPN's CDN returns 403 to
# non-browser strings that start with "Mozilla/5.0" (checked 2026-09-28).


class ESPNError(RuntimeError):
    pass


def cache_path(kind: str, key: str) -> Path:
    return RAW_DIR / kind / f"{key}.json.gz"


def read_cache(kind: str, key: str):
    p = cache_path(kind, key)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_cache(kind: str, key: str, payload: dict) -> Path:
    p = cache_path(kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(p)  # atomic, so an interrupted run never leaves a half file
    return p


def _throttle(min_interval: float):
    global _last_request
    wait = _last_request + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def fetch(path: str, params: dict | None = None, min_interval: float = MIN_INTERVAL,
          timeout: float = 30) -> dict:
    """GET ESPN_BASE/path with throttle and exponential backoff on 429/5xx/network errors."""
    url = f"{ESPN_BASE}/{path.lstrip('/')}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        _throttle(min_interval)
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
            else:  # 4xx other than 429 will not fix itself
                raise ESPNError(f"{url} {params} → HTTP {r.status_code}: {r.text[:200]}")
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            last_err = repr(e)
        time.sleep(BACKOFF_BASE * 2 ** attempt + random.uniform(0, 1))
    raise ESPNError(f"{url} {params} failed after {MAX_RETRIES} attempts: {last_err}")


def get_json(kind: str, key: str, path: str, params: dict | None = None,
             cacheable=lambda payload: True, refresh: bool = False, **kw) -> dict:
    """Return the cached payload if present, else fetch; cache it when `cacheable(payload)`."""
    if not refresh:
        cached = read_cache(kind, key)
        if cached is not None:
            return cached
    payload = fetch(path, params, **kw)
    payload.setdefault("_fetched_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if cacheable(payload):
        write_cache(kind, key, payload)
    return payload


# ── Endpoint helpers ─────────────────────────────────────────────────────────

def _all_final(payload: dict) -> bool:
    return all(ev["competitions"][0]["status"]["type"].get("completed", False)
               for ev in payload.get("events", []))


def scoreboard(d: date, refresh: bool = False) -> dict:
    """One ESPN (US-Eastern) calendar date. Cached only once every game on it is final
    and the date is at least 2 days in the past, so schedule changes are never frozen."""
    key = d.strftime("%Y%m%d")
    settled = (date.today() - d).days >= 2
    return get_json("scoreboard", key, "scoreboard", {"dates": key, "limit": 100},
                    cacheable=lambda p: settled and _all_final(p), refresh=refresh,
                    min_interval=0.3)


def summary(game_id: str | int, refresh: bool = False) -> dict:
    """Game summary (box score). Cached only when the game is final."""
    def final(p):
        comp = p.get("header", {}).get("competitions", [{}])[0]
        return comp.get("status", {}).get("type", {}).get("completed", False)
    return get_json("summary", str(game_id), "summary", {"event": str(game_id)},
                    cacheable=final, refresh=refresh)


def injuries_raw() -> dict:
    """Live league-wide injuries feed (never cached here; snapshots live in injury_snapshots)."""
    return fetch("injuries", min_interval=0.3)

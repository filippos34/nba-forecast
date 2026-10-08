"""Loads config.yaml from the repo root. Missing keys fall back to DEFAULTS."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

DEFAULTS = {
    "model": {"injury_adjust": False, "early_season_games": 15},
    "availability": {},
    "spread": {},
    "ops": {"host": "laptop", "keep_awake": True, "disabled_jobs": []},
    "site": {"deploy": False, "remote": "", "branch": "gh-pages", "url": "", "base": "/", "title_odds": False,
             "early_label_games": 10, "consensus_exclude": [], "live_hours_utc": [14, 4]},
}
try:                                   # parked market/staking settings: private module, absent in the public repo
    from private_defaults import DEFAULTS as _PRIVATE
    DEFAULTS.update(_PRIVATE)
except ImportError:
    pass


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


@lru_cache(maxsize=None)
def load(path: Path = CONFIG_PATH) -> dict:
    over = yaml.safe_load(path.read_text()) if path.exists() else {}
    return _merge(DEFAULTS, over or {})


def get(dotted: str):
    node = load()
    for part in dotted.split("."):
        node = node[part]
    return node

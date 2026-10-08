"""
src/polymarket_live.py — Polymarket midpoints for the site's games (public; runs on GitHub Actions)
====================================================================================================
For every game in model.json: Gamma event `nba-<away>-<home>-<ET date>` → moneyline market → CLOB best
bid / ask for both sides (resolve-bot walls ≤ 0.01 / ≥ 0.99 ignored). P(home) = mean(mid_home, 1 − mid_away);
spread = home ask − home bid. Read-only public endpoints; no keys.

    python3 src/polymarket_live.py --model data/live/model.json --out data/live/polymarket.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

GAMMA, CLOB = "https://gamma-api.polymarket.com", "https://clob.polymarket.com"
SLUG_CODE = {"NY": "nyk", "GS": "gsw", "SA": "sas", "NO": "nop", "UTAH": "uta", "WSH": "was"}
NICK = {"Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA", "Bulls": "CHI", "Cavaliers": "CLE",
        "Mavericks": "DAL", "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GS", "Rockets": "HOU", "Pacers": "IND",
        "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM", "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN",
        "Pelicans": "NO", "Knicks": "NY", "Thunder": "OKC", "Magic": "ORL", "76ers": "PHI", "Suns": "PHX",
        "Trail Blazers": "POR", "Kings": "SAC", "Spurs": "SA", "Raptors": "TOR", "Jazz": "UTAH", "Wizards": "WSH"}


class SchemaChanged(RuntimeError):
    pass


def _req(method: str, url: str, **kw):
    """GET/POST with retries on 429 / 5xx / network errors; anything else fails loudly."""
    for attempt in range(4):
        try:
            r = requests.request(method, url, timeout=30, **kw)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except requests.ConnectionError:
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{url}: no 200 after retries")


def team(name: str) -> str | None:
    name = (name or "").strip()
    return NICK.get(name) or next((v for k, v in NICK.items() if name.endswith(" " + k)), None)


def slug(away: str, home: str, date_et: str) -> str:
    code = lambda t: SLUG_CODE.get(t, t.lower())                     # noqa: E731
    return f"nba-{code(away)}-{code(home)}-{date_et}"


def top(book: dict) -> tuple[float | None, float | None]:
    if "bids" not in book or "asks" not in book:
        raise SchemaChanged(f"CLOB book layout changed: {sorted(book)[:8]}")
    bids = [float(b["price"]) for b in book["bids"] if float(b["price"]) > 0.01]
    asks = [float(a["price"]) for a in book["asks"] if float(a["price"]) < 0.99]
    return (max(bids) if bids else None), (min(asks) if asks else None)


def fetch(model: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out, missing = {}, []
    for g in model.get("games", []):
        sl = slug(g["away"], g["home"], model["date"])
        evs = _req("GET", f"{GAMMA}/events", params={"slug": sl})
        if not evs:
            missing.append(sl)
            continue
        mk = next((m for m in evs[0].get("markets", []) if m.get("sportsMarketType") == "moneyline"), None)
        if mk is None:
            missing.append(evs[0].get("slug"))
            continue
        if "clobTokenIds" not in mk or "outcomes" not in mk:
            raise SchemaChanged("Gamma market layout changed")
        side = {team(o): t for o, t in zip(json.loads(mk["outcomes"]), json.loads(mk["clobTokenIds"]))}
        if set(side) != {g["home"], g["away"]}:
            raise SchemaChanged(f"moneyline outcomes {list(side)} ≠ {g['away']}@{g['home']}")
        books = {b["asset_id"]: b for b in _req("POST", f"{CLOB}/books",
                                                  json=[{"token_id": side[g["home"]]}, {"token_id": side[g["away"]]}])}
        hb, ha = top(books.get(side[g["home"]], {"bids": [], "asks": []}))
        ab, aa = top(books.get(side[g["away"]], {"bids": [], "asks": []}))
        if None in (hb, ha, ab, aa):
            continue
        mid_h, mid_a = (hb + ha) / 2, (ab + aa) / 2
        out[str(g["game_id"])] = {"p_home": round((mid_h + 1 - mid_a) / 2, 4), "bid": hb, "ask": ha,
                                  "spread_pp": round(100 * (ha - hb), 1), "as_of": now}
    return {"generated_at": now, "source": "Polymarket CLOB midpoint", "games": out, "not_listed": missing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = fetch(json.loads(a.model.read_text()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"polymarket: {len(res['games'])} games priced, {len(res['not_listed'])} not listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

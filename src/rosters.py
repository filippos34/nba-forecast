"""
src/rosters.py — Current rosters, offseason roster changes, player-value table
===============================================================================
1. Snapshot all 30 ESPN rosters (raw JSON kept under data/raw/espn/rosters/<date>/)
   → data/rosters_2026_27.csv
2. Pull the ESPN transactions feed (calendar 2026, free text) → data/transactions_2026.csv
3. Compare each player's last 2025-26 team (data/player_games.parquet, RS + playoffs)
   with his current team → data/roster_changes_2026.csv
   (player, old team, new team, how: trade / FA / draft / waived / unknown, date)
4. Re-key the player-value table to 2026-27 teams → data/player_values_2026_27.csv
   Interim values: the April hand-typed PLAYER_RAPTOR numbers where a player has one,
   rookies at replacement level (0.0), everyone else blank. The box-score impact
   rating in Phase 2 replaces this. predict.py does not use it while
   config.yaml model.injury_adjust is false.

Usage: python3 src/rosters.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPLACEMENT_LEVEL = 0.0


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).strip()


# ── 1. rosters ──────────────────────────────────────────────────────────────

def fetch_rosters(snapshot_day: str | None = None) -> pd.DataFrame:
    snapshot_day = snapshot_day or datetime.now(timezone.utc).strftime("%Y%m%d")
    teams = espn.fetch("teams", min_interval=1.0)
    teams = teams["sports"][0]["leagues"][0]["teams"]
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for t in teams:
        tm = t["team"]
        key = f"{snapshot_day}/{tm['abbreviation']}"
        payload = espn.read_cache("rosters", key)
        if payload is None:
            payload = espn.fetch(f"teams/{tm['id']}/roster", min_interval=1.0)
            payload["_fetched_at"] = fetched_at
            espn.write_cache("rosters", key, payload)
        for a in payload.get("athletes", []):
            rows.append({
                "fetched_at": payload.get("_fetched_at", fetched_at),
                "team": tm["abbreviation"],
                "athlete_id": int(a["id"]),
                "player_name": a.get("displayName", ""),
                "position": (a.get("position") or {}).get("abbreviation", ""),
                "experience": (a.get("experience") or {}).get("years"),
                "age": a.get("age"),
                "status": (a.get("status") or {}).get("type", ""),
            })
    return pd.DataFrame(rows)


# ── 2. transactions ─────────────────────────────────────────────────────────

def fetch_transactions() -> pd.DataFrame:
    rows, page, pages = [], 1, 1
    while page <= pages:
        p = espn.fetch("transactions", {"page": page}, min_interval=1.0)
        pages = p.get("pageCount", 1)
        for t in p.get("transactions", []):
            rows.append({"date": t.get("date", "")[:10],
                         "team": (t.get("team") or {}).get("abbreviation", ""),
                         "description": t.get("description", "")})
        page += 1
    return pd.DataFrame(rows)


VERBS = r"(?:Signed|Re-signed|Acquired|Waived|Traded|Claimed|Converted|Exercised|Released|Declined|Assigned|Recalled|Sent|Named|Agreed|Hired|Fired|Announced|Placed|Activated|Suspended|Bought)"


def clauses(desc: str) -> list[str]:
    """ESPN bundles several moves per entry ("Signed X. Acquired Y. Waived Z.");
    split at a sentence end followed by a transaction verb."""
    return [c.strip() for c in re.split(rf"(?<=\.)\s+(?={VERBS}\b)", desc) if c.strip()]


def classify(desc: str) -> str:
    d = desc.lower()
    if d.startswith("acquired") or "in exchange for" in d or " trade" in d:
        return "trade"
    if "waived" in d or "released" in d or "bought out" in d:
        return "waived"
    if "draft" in d or "rookie scale" in d:
        return "draft"
    if "re-signed" in d:
        return "re-signed"
    if "signed" in d or "claimed" in d:
        return "FA"
    return "other"


# ── 3. roster changes ───────────────────────────────────────────────────────

def last_teams(player_games: pd.DataFrame, season: str = "2025-26") -> pd.DataFrame:
    pg = player_games[player_games["season"] == season].sort_values(["date_local", "game_id"])
    last = pg.groupby("athlete_id").tail(1)
    played = pg[~pg["did_not_play"]].groupby("athlete_id").agg(
        games=("game_id", "nunique"), minutes=("min", "sum"))
    out = last[["athlete_id", "player_name", "team", "date_local"]].rename(
        columns={"team": "old_team", "date_local": "last_game"})
    return out.merge(played, on="athlete_id", how="left").fillna({"games": 0, "minutes": 0})


def match_transaction(name: str, new_team: str | None, old_team: str | None,
                      tx: pd.DataFrame) -> tuple[str, str, str]:
    """Latest transaction mentioning the player (by last name + first initial), preferring
    his new team's entry. Returns (how, date, description)."""
    parts = name.split()
    if len(parts) < 2:
        return "unknown", "", ""
    last = re.escape(parts[-1] if parts[-1].lower() not in ("jr.", "jr", "ii", "iii", "iv", "sr.")
                     else parts[-2])
    first = re.escape(parts[0])
    rows = [{"date": t.date, "team": t.team, "description": c}
            for t in tx.itertuples() for c in clauses(t.description)
            if re.search(rf"\b{last}\b", c, re.I) and re.search(first, c, re.I)]
    hits = pd.DataFrame(rows, columns=["date", "team", "description"])
    if new_team and old_team:
        # moved: the new team's acquiring clause says how; a waiver by the old team
        # followed by a signing elsewhere is a free-agent move
        pref = hits[hits["team"] == new_team]
        if not len(pref):
            waived = hits[(hits["team"] == old_team) & hits["description"].map(classify).eq("waived")]
            if len(waived):
                h = waived.sort_values("date").iloc[-1]
                return "FA", h["date"], h["description"] + " (then signed elsewhere)"
        hits = pref if len(pref) else hits
    elif new_team:
        pref = hits[hits["team"] == new_team]
        hits = pref if len(pref) else hits
    elif old_team:
        pref = hits[hits["team"] == old_team]
        hits = pref if len(pref) else hits
    if not len(hits):
        return "unknown", "", ""
    h = hits.sort_values("date").iloc[-1]
    return classify(h["description"]), h["date"], h["description"]


def roster_changes(rosters: pd.DataFrame, lt: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    cur = rosters[["athlete_id", "player_name", "team", "experience"]].rename(columns={"team": "new_team"})
    m = cur.merge(lt[["athlete_id", "old_team", "games", "minutes"]], on="athlete_id", how="outer")
    names = pd.concat([cur.set_index("athlete_id")["player_name"],
                       lt.set_index("athlete_id")["player_name"]])
    m["player_name"] = m["athlete_id"].map(names[~names.index.duplicated()])
    changed = m[m["old_team"] != m["new_team"]].copy()
    out = []
    for r in changed.itertuples():
        new_team = r.new_team if isinstance(r.new_team, str) else None
        old_team = r.old_team if isinstance(r.old_team, str) else None
        how, when, desc = match_transaction(r.player_name, new_team, old_team, tx)
        if old_team is None and (r.experience == 0 or pd.isna(r.experience)) and how in ("unknown", "FA"):
            how = "draft" if how == "unknown" else how   # rookie (drafted or undrafted signing)
        if new_team is None and how == "unknown":
            how = "unsigned/left"
        out.append({"athlete_id": int(r.athlete_id), "player": r.player_name,
                    "old_team": old_team or "", "new_team": new_team or "",
                    "how": how, "date": when, "transaction": desc,
                    "games_2025_26": int(r.games) if pd.notna(r.games) else 0,
                    "minutes_2025_26": float(r.minutes) if pd.notna(r.minutes) else 0.0})
    return pd.DataFrame(out).sort_values(["how", "new_team", "player"]).reset_index(drop=True)


# ── 4. player values ────────────────────────────────────────────────────────

def player_values(rosters: pd.DataFrame, lt: pd.DataFrame) -> pd.DataFrame:
    from archive.player_raptor_apr2026 import PLAYER_RAPTOR
    raptor = {norm_name(n): v for (n, _), v in PLAYER_RAPTOR.items()}
    pv = rosters[["athlete_id", "player_name", "team", "experience", "position"]].merge(
        lt[["athlete_id", "games", "minutes"]], on="athlete_id", how="left")
    pv["mpg_2025_26"] = (pv["minutes"] / pv["games"]).round(1)
    pv["value"] = pv["player_name"].map(lambda n: raptor.get(norm_name(n)))
    pv["source"] = pv["value"].notna().map({True: "raptor_table_apr2026", False: ""})
    rookie = pv["value"].isna() & (pv["experience"] == 0)
    pv.loc[rookie, ["value", "source"]] = [REPLACEMENT_LEVEL, "replacement_rookie"]
    return pv.drop(columns=["minutes"]).rename(columns={"games": "games_2025_26"})


def main() -> int:
    rosters = fetch_rosters()
    path = DATA / "rosters_2026_27.csv"
    if path.exists():   # keep every snapshot (point-in-time membership); one per fetch
        old = pd.read_csv(path)
        old = old[old["fetched_at"] != rosters["fetched_at"].iloc[0]]
        pd.concat([old, rosters], ignore_index=True).to_csv(path, index=False)
    else:
        rosters.to_csv(path, index=False)
    print(f"rosters: {len(rosters)} players, {rosters['team'].nunique()} teams")
    tx = fetch_transactions()
    tx.to_csv(DATA / "transactions_2026.csv", index=False)
    print(f"transactions: {len(tx)} ({tx['date'].min()} → {tx['date'].max()})")
    lt = last_teams(pd.read_parquet(DATA / "player_games.parquet"))
    rc = roster_changes(rosters, lt, tx)
    rc.to_csv(DATA / "roster_changes_2026.csv", index=False)
    print(f"roster changes: {len(rc)}  {rc['how'].value_counts().to_dict()}")
    pv = player_values(rosters, lt)
    pv.to_csv(DATA / "player_values_2026_27.csv", index=False)
    print(f"player values: {len(pv)} rows, {pv['source'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

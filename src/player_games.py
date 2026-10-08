"""
src/player_games.py — Per-player box scores from ESPN game summaries
=====================================================================
For every completed regular-season, play-in and playoff game in data/games_all.csv:
  1. fetch /summary?event=ID (~1 req/s, resumable: cached game IDs are skipped;
     raw JSON kept gzipped under data/raw/espn/summary/)
  2. parse each player's line → data/player_games.parquet

Columns: game_id, date_local, season, season_type, team, opponent, is_home,
athlete_id, player_name, position, starter, did_not_play, dnp_reason, ejected,
min, pts, fgm, fga, fg3m, fg3a, ftm, fta, reb, oreb, dreb, ast, tov, stl, blk,
pf, plus_minus, periods, box_incomplete

`dnp_reason` is ESPN's post-game text (e.g. "RIGHT FOOT", "COACH'S DECISION").
It says who did not play, NOT what was known before tip — never use it as a
pregame input (see ground rule 1).

Usage:
    python3 src/player_games.py              # fetch missing + rebuild parquet
    python3 src/player_games.py --parse-only
    python3 src/player_games.py --limit 50   # fetch at most 50 new games
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "player_games.parquet"

STAT_MAP = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast", "TO": "tov",
            "STL": "stl", "BLK": "blk", "OREB": "oreb", "DREB": "dreb", "PF": "pf",
            "+/-": "plus_minus"}
SPLIT_MAP = {"FG": ("fgm", "fga"), "3PT": ("fg3m", "fg3a"), "FT": ("ftm", "fta")}
NUM_COLS = ["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "oreb",
            "dreb", "ast", "tov", "stl", "blk", "pf", "plus_minus"]


def _num(x):
    if x in (None, "", "--"):
        return None
    try:
        return float(str(x).replace("+", ""))
    except ValueError:
        return None


def parse_summary(payload: dict, meta: dict) -> list[dict]:
    """meta: game_id, date_local, season, season_type (from games_all)."""
    comp = payload["header"]["competitions"][0]
    side = {c["team"]["abbreviation"]: c["homeAway"] for c in comp["competitors"]}
    periods = max((len(c.get("linescores", [])) for c in comp["competitors"]), default=0)
    teams = list(side)
    rows = []
    for block in payload.get("boxscore", {}).get("players", []):
        team = block["team"]["abbreviation"]
        opp = next((t for t in teams if t != team), None)
        if not block.get("statistics"):
            continue
        st = block["statistics"][0]
        names = st.get("names", [])
        for a in st.get("athletes", []):
            ath = a.get("athlete", {})
            row = {**meta, "team": team, "opponent": opp, "is_home": side.get(team) == "home",
                   "athlete_id": int(ath["id"]) if ath.get("id") else None,
                   "player_name": ath.get("displayName", ""),
                   "position": (ath.get("position") or {}).get("abbreviation", ""),
                   "starter": bool(a.get("starter", False)),
                   "did_not_play": bool(a.get("didNotPlay", False)) or not a.get("stats"),
                   "dnp_reason": a.get("reason", "") if (a.get("didNotPlay") or not a.get("stats")) else "",
                   "ejected": bool(a.get("ejected", False)),
                   "periods": periods}
            for c in NUM_COLS:
                row[c] = None
            for name, val in zip(names, a.get("stats", [])):
                if name in STAT_MAP:
                    row[STAT_MAP[name]] = _num(val)
                elif name in SPLIT_MAP and "-" in str(val):
                    m, att = str(val).split("-", 1)
                    row[SPLIT_MAP[name][0]], row[SPLIT_MAP[name][1]] = _num(m), _num(att)
            if row["did_not_play"]:
                row["min"] = 0.0
            rows.append(row)
    return rows


def target_games() -> pd.DataFrame:
    g = pd.read_csv(DATA / "games_all.csv", parse_dates=["date_local"])
    g = g[g["completed"] & g["season"].isin(
        ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])]
    return g[["game_id", "date_local", "season", "season_type"]].sort_values("date_local")


def _have() -> set[int]:
    return set(pd.read_parquet(OUT, columns=["game_id"])["game_id"]) if OUT.exists() else set()


def fetch_missing(games: pd.DataFrame, limit: int | None = None) -> int:
    """Fetch summaries for games that are neither in player_games.parquet nor cached (a fresh
    host without the raw cache does not re-download history)."""
    have = _have()
    todo = [gid for gid in games["game_id"] if gid not in have and espn.read_cache("summary", str(gid)) is None]
    if limit:
        todo = todo[:limit]
    print(f"{len(games)} games, {len(games) - len(todo)} cached, fetching {len(todo)} "
          f"(~{len(todo) / 60:.0f} min at 1 req/s)", flush=True)
    t0, failed = time.time(), []
    for i, gid in enumerate(todo, 1):
        try:
            espn.summary(gid)
        except espn.ESPNError as e:
            failed.append(gid)
            print(f"  FAILED {gid}: {e}", flush=True)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    if failed:
        print(f"{len(failed)} failed (re-run to retry): {failed[:20]}")
    return len(todo) - len(failed)


def flag_incomplete(df: pd.DataFrame, tol: float = 5.0) -> pd.DataFrame:
    """box_incomplete = the team's player minutes miss the expected 240 (+25 per OT)
    by more than `tol`, or player points do not sum to the team score — ESPN left out
    player lines (10 CHI 2025-26 games). Keep these rows for team Elo; exclude them from player-rating training."""
    played = df[~df["did_not_play"]]
    tm = played.groupby(["game_id", "team"]).agg(mins=("min", "sum"), pts=("pts", "sum"),
                                                 periods=("periods", "max"))
    expected = 240 + 25 * (tm["periods"].clip(lower=4) - 4)
    ga = pd.read_csv(DATA / "games_all.csv", usecols=["game_id", "home_team", "away_team",
                                                      "home_pts", "away_pts"])
    score = pd.concat([ga.set_index(["game_id", "home_team"])["home_pts"],
                       ga.set_index(["game_id", "away_team"])["away_pts"]])
    score.index.names = ["game_id", "team"]
    pts_off = tm["pts"] != score.reindex(tm.index)
    bad = set(tm.index[((tm["mins"] - expected).abs() > tol) | pts_off])
    df["box_incomplete"] = [(g, t) in bad for g, t in zip(df["game_id"], df["team"])]
    return df


def build_parquet(games: pd.DataFrame) -> pd.DataFrame:
    """Parse every cached summary; games without a cached summary keep their existing rows."""
    existing = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    rows, missing, kept = [], [], []
    for g in games.itertuples():
        payload = espn.read_cache("summary", str(g.game_id))
        if payload is None:
            if len(existing) and (existing["game_id"] == g.game_id).any():
                kept.append(g.game_id)
            else:
                missing.append(g.game_id)
            continue
        rows += parse_summary(payload, {"game_id": g.game_id, "date_local": g.date_local,
                                        "season": g.season, "season_type": g.season_type})
    parsed = pd.DataFrame(rows)
    if kept:
        old = existing[existing["game_id"].isin(kept)].drop(columns=["box_incomplete"], errors="ignore")
        parsed = pd.concat([old, parsed], ignore_index=True) if len(parsed) else old
    df = flag_incomplete(parsed)
    df.to_parquet(OUT, index=False)
    print(f"{len(df)} player-game rows from {df['game_id'].nunique()} games → {OUT.relative_to(ROOT)}"
          f"; {len(missing)} games not yet fetched")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    games = target_games()
    if not args.parse_only:
        fetch_missing(games, args.limit)
    build_parquet(games)
    return 0


if __name__ == "__main__":
    sys.exit(main())

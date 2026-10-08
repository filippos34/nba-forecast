"""
src/injury_status.py — Pregame status per (game, team, player) at the decision time
=====================================================================================
For each game: decision time = tip − 60 min (injury_reports.DECISION_MINUTES). Use the
latest official report with issued_at ≤ decision time in which the team has submitted
(not "NOT YET SUBMITTED"). Report rows are matched to ESPN game_ids by (ET date,
away@home) and to ESPN athlete_ids by normalized name within the team and season.

Output: data/injury_status_t60.parquet
  game_id, team, athlete_id, player_report, status, reason, issued_at, file, minutes_before_tip
Players of the team not on that report are "Not listed" (added by the consumer).

Also: p_plays_table() — P(plays | status) estimated on given seasons.
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from injury_reports import DECISION_MINUTES, OUT as REPORTS_PARQUET  # noqa: E402
from rosters import norm_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "injury_status_t60.parquet"

NBA_TO_ESPN = {"NYK": "NY", "GSW": "GS", "SAS": "SA", "NOP": "NO", "UTA": "UTAH", "WAS": "WSH"}
TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NO": "New Orleans Pelicans",
    "NY": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SA": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTAH": "Utah Jazz",
    "WSH": "Washington Wizards",
}
_TEAM_KEY = {re.sub(r"[^a-z0-9]", "", v.lower()): k for k, v in TEAM_NAMES.items()}
_TEAM_KEY["losangelesclippers"] = "LAC"
STATUS_ORDER = ["Out", "Doubtful", "Questionable", "Probable", "Available", "Not listed"]


def team_abbr(name) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    return _TEAM_KEY.get(re.sub(r"[^a-z0-9]", "", name.lower()))


def report_name(s: str) -> str:
    """'Porter Jr., Kevin' → 'kevin porter'; handles the space-less 2025-26 layout."""
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)   # 'PippenJr.' → 'Pippen Jr.'
    return norm_name(s)


def _squash(s: str) -> str:
    return s.replace(" ", "")


def build(reports: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None,
          out_path: Path | None = OUT) -> pd.DataFrame:
    """as_of (UTC): live use — a report counts only if issued ≤ min(tip − 60 min, as_of)."""
    rep = pd.read_parquet(REPORTS_PARQUET) if reports is None else reports
    rep = rep.copy()
    rep["issued_at"] = pd.to_datetime(rep["issued_at"], utc=True)
    rep["team_abbr"] = rep["team"].map(team_abbr)
    mu = rep["matchup"].fillna("").str.extract(r"^([A-Z]{2,3})@([A-Z]{2,3})$")
    rep["away"] = mu[0].replace(NBA_TO_ESPN)
    rep["home"] = mu[1].replace(NBA_TO_ESPN)
    rep["gdate"] = pd.to_datetime(rep["game_date"], format="%m/%d/%Y", errors="coerce")

    ga = pd.read_csv(DATA / "games_all.csv", parse_dates=["date_local"])
    ga = ga[ga["season_type"].isin(["regular", "playoff", "playin"])]
    ga["tip"] = pd.to_datetime(ga["tip_utc"], utc=True)
    key = ga.set_index(["date_local", "away_team", "home_team"])[["game_id", "tip", "season"]]
    rep = rep.join(key, on=["gdate", "away", "home"])
    unmatched_games = rep["game_id"].isna().mean()
    rep = rep.dropna(subset=["game_id"])
    wrong_team = ~rep["team_abbr"].isin([None]) & rep["team_abbr"].notna() & \
        (rep["team_abbr"] != rep["home"]) & (rep["team_abbr"] != rep["away"])
    if wrong_team.mean() > 0.001:
        raise ValueError(f"{wrong_team.mean():.2%} report rows name a team outside their matchup — parser bug")
    rep = rep[~wrong_team]
    rep["game_id"] = rep["game_id"].astype(int)
    rep["cutoff"] = rep["tip"] - pd.Timedelta(minutes=DECISION_MINUTES)
    if as_of is not None:
        rep["cutoff"] = rep["cutoff"].clip(upper=pd.Timestamp(as_of))
    rep = rep[rep["issued_at"] <= rep["cutoff"]]

    # latest submitted report per (game, team)
    sub = rep[rep["status"] != "NOT YET SUBMITTED"]
    last = sub.groupby(["game_id", "team_abbr"])["issued_at"].max().rename("chosen")
    sub = sub.join(last, on=["game_id", "team_abbr"])
    sel = sub[sub["issued_at"] == sub["chosen"]].copy()

    # athlete matching within team-season (any box-score row, played or DNP)
    pg = pd.read_parquet(DATA / "player_games.parquet", columns=["athlete_id", "player_name", "team", "season"])
    rosters = DATA / "rosters_2026_27.csv"     # names for a season without box scores yet
    if rosters.exists():
        r = pd.read_csv(rosters, usecols=["athlete_id", "player_name", "team"]).assign(season="2026-27")
        pg = pd.concat([pg, r], ignore_index=True)
    pg["key"] = pg["player_name"].map(norm_name).map(_squash)
    roster = pg.drop_duplicates(["season", "team", "key"])
    lookup = {(s, t, k): a for s, t, k, a in zip(roster["season"], roster["team"], roster["key"], roster["athlete_id"])}
    by_ts = roster.groupby(["season", "team"])["key"].apply(list).to_dict()
    by_s = pg.drop_duplicates(["season", "key"]).set_index(["season", "key"])["athlete_id"].to_dict()

    ids = []
    for s, t, name in zip(sel["season"], sel["team_abbr"], sel["player"]):
        k = _squash(report_name(name if isinstance(name, str) else ""))
        a = lookup.get((s, t, k))
        if a is None:
            close = difflib.get_close_matches(k, by_ts.get((s, t), []), n=1, cutoff=0.85)
            a = lookup.get((s, t, close[0])) if close else by_s.get((s, k))
        ids.append(a)
    sel["athlete_id"] = ids
    out = sel.rename(columns={"player": "player_report"})[
        ["game_id", "team_abbr", "athlete_id", "player_report", "status", "reason", "issued_at", "file", "tip"]]
    out = out.rename(columns={"team_abbr": "team"})
    out["minutes_before_tip"] = (out["tip"] - out["issued_at"]).dt.total_seconds() / 60
    out = out.drop(columns="tip").drop_duplicates(["game_id", "team", "athlete_id", "player_report"])
    if out_path is not None:
        out.to_parquet(out_path, index=False)
    print(f"{len(out)} status rows for {out['game_id'].nunique()} games; "
          f"unmatched report games {unmatched_games:.1%}; unmatched players "
          f"{out['athlete_id'].isna().mean():.1%} (mostly two-way / G League players with no NBA minutes)")
    return out


def status_for(cand: pd.DataFrame, status: pd.DataFrame) -> pd.Series:
    """Status per candidate row (game_id, team, athlete_id); 'Not listed' if absent,
    NaN if no report covers that team-game at the decision time."""
    s = status.dropna(subset=["athlete_id"]).astype({"athlete_id": int})
    s = s.drop_duplicates(["game_id", "team", "athlete_id"], keep="first")
    m = cand[["game_id", "team", "athlete_id"]].merge(s[["game_id", "team", "athlete_id", "status"]],
                                                      how="left", on=["game_id", "team", "athlete_id"])
    covered = set(zip(status["game_id"], status["team"]))
    has = np.array([(g, t) in covered for g, t in zip(m["game_id"], m["team"])])
    st = m["status"].where(m["status"].notna(), np.where(has, "Not listed", None))
    return pd.Series(st.to_numpy(), index=cand.index)


def p_plays_table(cand: pd.DataFrame, status: pd.Series, seasons_mask: np.ndarray,
                  min_proj: float = 0.0) -> pd.DataFrame:
    """P(plays | status) among candidates with projected minutes > min_proj."""
    d = pd.DataFrame({"status": status, "played": cand["played"].astype(float),
                      "proj": cand["proj_min"]})[seasons_mask & (cand["proj_min"] > min_proj).to_numpy()]
    d = d.dropna(subset=["status"])
    t = d.groupby("status")["played"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "p_plays"})
    t["se"] = np.sqrt(t["p_plays"] * (1 - t["p_plays"]) / t["n"])
    return t.reindex([s for s in STATUS_ORDER if s in t.index])


if __name__ == "__main__":
    build()

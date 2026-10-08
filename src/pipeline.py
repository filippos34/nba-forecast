"""
NBA Data Pipeline — ESPN public scoreboard API
==============================================
Fetches every game one US-Eastern date at a time (ESPN's scoreboard stopped
accepting date ranges in 2026) through src/espn.py, which caches settled
dates under data/raw/espn/scoreboard/ and retries with backoff.

Outputs (data/):
  games_all.csv          every RS / play-in / playoff event, incl. scheduled games
  games_raw.csv          completed regular-season games (model input)
  playoff_games_raw.csv  completed playoff games
  playin_games_raw.csv   completed play-in games
  schedule_2026_27.csv   2026-27 regular-season schedule, tip times UTC + local, B2B flags
  team_ratings.csv, season_summary.csv

Date columns:
  date        legacy: the UTC calendar date of tip-off (late West-coast games land
              on the next day). Kept because the production model was built on it.
  date_local  the US-Eastern date ESPN lists the game under (the real game day).
  tip_utc     tip-off timestamp, UTC.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espn  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# The 30 real NBA franchises using ESPN abbreviations.
# ESPN uses NY (not NYK), GS (not GSW), SA (not SAS), NO (not NOP), UTAH, WSH
NBA_TEAMS = {
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO",  "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SA",  "SAC", "TOR", "UTAH","WSH",
}

# Arena time zone per team (for local tip times)
TEAM_TZ = {
    **{t: "America/New_York" for t in ("ATL", "BKN", "BOS", "CHA", "CLE", "DET", "IND",
                                        "MIA", "NY", "ORL", "PHI", "TOR", "WSH")},
    **{t: "America/Chicago" for t in ("CHI", "DAL", "HOU", "MEM", "MIL", "MIN", "NO",
                                       "OKC", "SA")},
    "DEN": "America/Denver", "UTAH": "America/Denver", "PHX": "America/Phoenix",
    **{t: "America/Los_Angeles" for t in ("GS", "LAC", "LAL", "POR", "SAC")},
}

# ESPN uses the END year as the season id (2025-26 → 2026). Each window covers
# preseason through the Finals; the season type on each event decides what it is.
SEASONS = {
    "2021-22": {"espn_year": 2022, "start": date(2021, 10, 1), "end": date(2022, 6, 30)},
    "2022-23": {"espn_year": 2023, "start": date(2022, 10, 1), "end": date(2023, 6, 30)},
    "2023-24": {"espn_year": 2024, "start": date(2023, 10, 1), "end": date(2024, 6, 30)},
    "2024-25": {"espn_year": 2025, "start": date(2024, 10, 1), "end": date(2025, 6, 30)},
    "2025-26": {"espn_year": 2026, "start": date(2025, 10, 1), "end": date(2026, 6, 30)},
}
SCHEDULE_SEASON = ("2026-27", {"espn_year": 2027, "start": date(2026, 10, 15),
                               "end": date(2027, 4, 30)})

SEASON_TYPES = {2: "regular", 3: "playoff", 5: "playin"}
MODEL_SEASONS = list(SEASONS)   # seasons the rating walk trains on


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_event(event: dict, query_date: date, season_label: str, espn_year: int):
    """One ESPN event → row dict, or None for preseason / other-year / non-NBA events."""
    season = event.get("season", {})
    stype = season.get("type")
    if stype not in SEASON_TYPES or season.get("year") != espn_year:
        return None
    comp = event["competitions"][0]
    status = comp.get("status", {}).get("type", {})
    completed = bool(status.get("completed", False))

    teams = {}
    for c in comp["competitors"]:
        score = c.get("score")
        try:
            pts = int(score) if completed else None
        except (TypeError, ValueError):
            return None  # "final" without a score — treat as bad data, skip
        teams[c["homeAway"]] = (c["team"]["abbreviation"], pts)
    if set(teams) != {"home", "away"}:
        return None
    (ht, hp), (at, ap) = teams["home"], teams["away"]
    if ht not in NBA_TEAMS or at not in NBA_TEAMS:  # All-Star, exhibition vs non-NBA
        return None

    tip = pd.Timestamp(event["date"])  # tz-aware UTC
    return {
        "game_id":      int(event["id"]),
        "date":         tip.tz_localize(None).normalize() if tip.tzinfo is None
                        else tip.tz_convert("UTC").tz_localize(None).normalize(),
        "date_local":   pd.Timestamp(query_date),
        "tip_utc":      tip.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":       season_label,
        "season_type":  SEASON_TYPES[stype],
        "home_team":    ht,
        "away_team":    at,
        "home_pts":     hp,
        "away_pts":     ap,
        "completed":    completed,
        "status":       status.get("name", ""),
        "is_neutral": bool(comp.get("neutralSite", False)),
        "comp_type":    (comp.get("type") or {}).get("abbreviation", ""),
    }


def fetch_season(label: str, cfg: dict, verbose: bool = True) -> list[dict]:
    rows = []
    days = list(daterange(cfg["start"], cfg["end"]))
    for i, d in enumerate(days, 1):
        payload = espn.scoreboard(d)
        for ev in payload.get("events", []):
            row = parse_event(ev, d, label, cfg["espn_year"])
            if row:
                rows.append(row)
        if verbose and (i % 30 == 0 or i == len(days)):
            print(f"    {label}: {i}/{len(days)} dates, {len(rows)} games")
    return rows


def build_games_all(seasons: dict, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for label, cfg in seasons.items():
        rows.extend(fetch_season(label, cfg, verbose))
    df = pd.DataFrame(rows)
    # A postponed game can appear under two dates; keep the latest listing.
    df = df.sort_values(["date_local", "game_id"]).drop_duplicates("game_id", keep="last")
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


GAMES_RAW_COLS = ["game_id", "date", "season", "home_team", "away_team",
                  "home_pts", "away_pts"]
EXTRA_COLS = ["date_local", "tip_utc", "is_neutral", "comp_type"]


def completed_games(games_all: pd.DataFrame, season_type: str) -> pd.DataFrame:
    df = games_all[(games_all["season_type"] == season_type) & games_all["completed"]
                   & games_all["season"].isin(MODEL_SEASONS)].copy()
    df[["home_pts", "away_pts"]] = df[["home_pts", "away_pts"]].astype(int)
    return df[GAMES_RAW_COLS + EXTRA_COLS].reset_index(drop=True)


def build_schedule(games_all: pd.DataFrame, label: str) -> pd.DataFrame:
    """Regular-season schedule with UTC + arena-local tip times and local-date B2B flags."""
    s = games_all[(games_all["season"] == label) &
                  (games_all["season_type"] == "regular")].copy()
    tips = pd.to_datetime(s["tip_utc"], utc=True)
    s["tip_local"] = [t.tz_convert(TEAM_TZ[h]).strftime("%Y-%m-%dT%H:%M:%S%z")
                      for t, h in zip(tips, s["home_team"])]
    s["tip_et"] = tips.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    s = s.sort_values(["date_local", "tip_utc", "game_id"])
    played = set(zip(s["home_team"], s["date_local"])) | set(zip(s["away_team"], s["date_local"]))
    prev = s["date_local"] - pd.Timedelta(days=1)
    s["home_b2b"] = [(t, d) in played for t, d in zip(s["home_team"], prev)]
    s["away_b2b"] = [(t, d) in played for t, d in zip(s["away_team"], prev)]
    cols = ["game_id", "date_local", "tip_utc", "tip_et", "tip_local", "season",
            "home_team", "away_team", "is_neutral", "comp_type", "status",
            "home_b2b", "away_b2b"]
    return s[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Back-to-back flags
# ---------------------------------------------------------------------------

def add_back_to_back_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    game_dates: set = set()
    for _, r in df.iterrows():
        game_dates.add((r["home_team"], r["date"].date()))
        game_dates.add((r["away_team"], r["date"].date()))

    def b2b(team, gdate):
        prev = (pd.Timestamp(gdate) - pd.Timedelta(days=1)).date()
        return (team, prev) in game_dates

    df["home_b2b"] = df.apply(lambda r: b2b(r["home_team"], r["date"]), axis=1)
    df["away_b2b"] = df.apply(lambda r: b2b(r["away_team"], r["date"]), axis=1)
    return df


# ---------------------------------------------------------------------------
# Rolling team ratings — ZERO LOOKAHEAD BIAS
# ---------------------------------------------------------------------------

def build_team_ratings(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every game, record each team's season-to-date rating BEFORE
    that game is played, then update the cumulative totals.

    Ratings:
      rolling_ppg   = cumulative points scored / games played
      rolling_papg  = cumulative points allowed / games played
      rolling_net   = rolling_ppg - rolling_papg
    """
    records = []

    for season in games_df["season"].unique():
        sg = games_df[games_df["season"] == season].sort_values("date")

        # team -> {sum_pts, sum_opp_pts, n}
        cumul: dict = {}

        def pre_game_rating(team):
            s = cumul.get(team)
            if not s or s["n"] == 0:
                return None, None, None
            ppg  = s["sum_pts"]     / s["n"]
            papg = s["sum_opp_pts"] / s["n"]
            return ppg, papg, ppg - papg

        def update(team, pts, opp_pts):
            if team not in cumul:
                cumul[team] = {"sum_pts": 0, "sum_opp_pts": 0, "n": 0}
            cumul[team]["sum_pts"]     += pts
            cumul[team]["sum_opp_pts"] += opp_pts
            cumul[team]["n"]           += 1

        for _, g in sg.iterrows():
            ht, at = g["home_team"], g["away_team"]
            hp, ap = int(g["home_pts"]), int(g["away_pts"])

            # Record PRE-GAME ratings for both teams
            for team, opp, pts, opp_pts, is_home in [
                (ht, at, hp, ap, True),
                (at, ht, ap, hp, False),
            ]:
                n = cumul.get(team, {}).get("n", 0)
                ppg, papg, net = pre_game_rating(team)
                records.append({
                    "season":        season,
                    "date":          g["date"],
                    "game_id":       g["game_id"],
                    "team":          team,
                    "opponent":      opp,
                    "is_home":       is_home,
                    "pts_scored":    pts,
                    "pts_allowed":   opp_pts,
                    "games_played":  n,
                    "rolling_ppg":   round(ppg,  3) if ppg  is not None else None,
                    "rolling_papg":  round(papg, 3) if papg is not None else None,
                    "rolling_net":   round(net,  3) if net  is not None else None,
                })

            # Update AFTER recording (preserves no-lookahead guarantee)
            update(ht, hp, ap)
            update(at, ap, hp)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Season summary
# ---------------------------------------------------------------------------

def build_season_summary(games_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in games_df["season"].unique():
        sg = games_df[games_df["season"] == season]
        cumul: dict = {}

        def upd(team, pts, opp_pts):
            if team not in cumul:
                cumul[team] = {"sum_pts": 0, "sum_opp_pts": 0, "n": 0}
            cumul[team]["sum_pts"]     += pts
            cumul[team]["sum_opp_pts"] += opp_pts
            cumul[team]["n"]           += 1

        for _, g in sg.iterrows():
            upd(g["home_team"], g["home_pts"], g["away_pts"])
            upd(g["away_team"], g["away_pts"], g["home_pts"])

        for team, s in cumul.items():
            if s["n"] == 0:
                continue
            ppg  = s["sum_pts"]     / s["n"]
            papg = s["sum_opp_pts"] / s["n"]
            rows.append({
                "season": season,
                "team":   team,
                "games":  s["n"],
                "ppg":    round(ppg,  2),
                "papg":   round(papg, 2),
                "net":    round(ppg - papg, 2),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["season", "net"], ascending=[True, False]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Display + sanity check
# ---------------------------------------------------------------------------

def print_summary_table(summary_df: pd.DataFrame):
    for season in summary_df["season"].unique():
        s = summary_df[summary_df["season"] == season]
        print(f"\n{'='*55}")
        print(f"  {season} — Teams ranked by Net PPG")
        print(f"{'='*55}")
        print(f"  {'Rk':<4} {'Team':<5} {'PPG':>7} {'PAPG':>7} {'Net':>7} {'G':>4}")
        print(f"  {'-'*45}")
        for rank, (_, row) in enumerate(s.iterrows(), 1):
            print(
                f"  {rank:<4} {row['team']:<5} "
                f"{row['ppg']:>7.1f} "
                f"{row['papg']:>7.1f} "
                f"{row['net']:>+7.1f} "
                f"{int(row['games']):>4}"
            )


def sanity_check(summary_df: pd.DataFrame):
    """
    Spot-check: known strong teams should appear near the top.
    2023-24: BOS was clearly the best regular-season team
    2024-25: OKC, CLE, BOS expected near top
    """
    # ESPN abbreviations: NY = Knicks, GS = Warriors, NO = Pelicans, SA = Spurs
    expected = {
        "2021-22": {"GS", "MEM", "PHX", "MIL", "BOS"},
        "2022-23": {"MIL", "BOS", "DEN", "OKC", "CLE"},
        "2023-24": {"BOS", "OKC", "MIN", "DEN", "NY"},
        "2024-25": {"OKC", "CLE", "BOS", "MEM", "NY"},
    }
    print(f"\n{'='*55}")
    print("  SANITY CHECK")
    print(f"{'='*55}")
    all_ok = True
    for season, good_teams in expected.items():
        if season not in summary_df["season"].values:
            continue
        top5 = set(summary_df[summary_df["season"] == season].head(5)["team"])
        overlap = top5 & good_teams
        ok = len(overlap) >= 2
        if not ok:
            all_ok = False
        tag = "PASS" if ok else "WARN"
        print(f"  {season}: top-5 = {sorted(top5)}")
        print(f"           known good = {sorted(good_teams)}")
        print(f"           overlap = {sorted(overlap)}  [{tag}]")
    print(f"\n  Overall: {'PASS' if all_ok else 'WARN — worth investigating'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("NBA Data Pipeline (ESPN, one date per request)")
    print("=" * 55)

    print("\n[1/7] Fetching all seasons (cached dates are read from disk)...")
    games_all = build_games_all({**SEASONS, SCHEDULE_SEASON[0]: SCHEDULE_SEASON[1]})
    games_all.to_csv(DATA_DIR / "games_all.csv", index=False)
    print(f"       {len(games_all)} events → games_all.csv")

    print("\n[2/7] Regular season → games_raw.csv")
    games_df = completed_games(games_all, "regular")
    games_df = add_back_to_back_flag(games_df)
    games_df = games_df[GAMES_RAW_COLS + ["home_b2b", "away_b2b"] + EXTRA_COLS]
    games_df.to_csv(DATA_DIR / "games_raw.csv", index=False)
    for s_ in sorted(games_df["season"].unique()):
        print(f"       {s_}: {(games_df['season'] == s_).sum()} games")

    print("\n[3/7] Playoffs → playoff_games_raw.csv, play-in → playin_games_raw.csv")
    for stype, fname in (("playoff", "playoff_games_raw.csv"), ("playin", "playin_games_raw.csv")):
        df = completed_games(games_all, stype)
        df.to_csv(DATA_DIR / fname, index=False)
        print(f"       {fname}: {len(df)} games {df['season'].value_counts().sort_index().to_dict()}")

    print(f"\n[4/7] {SCHEDULE_SEASON[0]} schedule → schedule_2026_27.csv")
    sched = build_schedule(games_all, SCHEDULE_SEASON[0])
    sched.to_csv(DATA_DIR / "schedule_2026_27.csv", index=False)
    print(f"       {len(sched)} games, first {sched['date_local'].min().date()}, "
          f"last {sched['date_local'].max().date()}")

    print("\n[5/7] Rolling team ratings (no lookahead) → team_ratings.csv")
    build_team_ratings(games_df).to_csv(DATA_DIR / "team_ratings.csv", index=False)

    print("\n[6/7] Season summary → season_summary.csv")
    summary_df = build_season_summary(games_df)
    summary_df.to_csv(DATA_DIR / "season_summary.csv", index=False)

    print("\n[7/7] Sanity check")
    sanity_check(summary_df)
    print("\nNext: python3 src/build_ratings.py")


if __name__ == "__main__":
    main()

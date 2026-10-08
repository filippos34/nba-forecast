"""
src/live.py — Context for upcoming games, computed exactly as in the backtest walk
===================================================================================
game_context(et_date) → one row per game on that US-Eastern date:
  game_id, tip (UTC), home_team, away_team, is_neutral,
  home_rest / away_rest   arena-local rest days (build_ratings: rest_basis "local"),
                          counting every earlier game (completed or scheduled) of the team
  home_games / away_games games the team has played this regular season before this tip
  early_season            either team has played fewer than model.early_season_games
predict_day(et_date) → model forecasts for every game that day (latest official injury report).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import build_ratings as br
import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _all_games() -> pd.DataFrame:
    g = pd.read_csv(DATA / "games_all.csv")
    g = g[g["season_type"].isin(["regular", "playoff", "playin"])].copy()
    g["tip"] = pd.to_datetime(g["tip_utc"], utc=True)
    g["date"] = g["tip"].dt.tz_convert(None).dt.normalize()
    return g.sort_values(["tip", "game_id"]).reset_index(drop=True)


def game_context(et_date) -> pd.DataFrame:
    et_date = pd.Timestamp(et_date).strftime("%Y-%m-%d")
    g = _all_games()
    g["date_arena"], _ = br.arena_local(g["tip_utc"], g["home_team"])
    h, a = br._team_rest(g, "date_arena")
    g["home_rest"], g["away_rest"] = h, a
    rs = g[g["season_type"] == "regular"]
    today = g[g["date_local"] == et_date].copy()
    if today.empty:
        return today
    n_early = int(config.get("model.early_season_games"))

    def played(team, tip, season):
        m = rs[(rs["season"] == season) & (rs["tip"] < tip) & rs["completed"]]
        return int(((m["home_team"] == team) | (m["away_team"] == team)).sum())

    today["home_games"] = [played(t, tip, s) for t, tip, s in zip(today["home_team"], today["tip"], today["season"])]
    today["away_games"] = [played(t, tip, s) for t, tip, s in zip(today["away_team"], today["tip"], today["season"])]
    today["early_season"] = (today[["home_games", "away_games"]].min(axis=1) < n_early) & \
        (today["season_type"] == "regular")
    cols = ["game_id", "tip", "tip_utc", "home_team", "away_team", "is_neutral", "season", "season_type",
            "home_rest", "away_rest", "home_games", "away_games", "early_season"]
    return today[cols].reset_index(drop=True)


def predict_day(et_date, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Predictions for every game on the ET date with the latest official report ≤ as_of."""
    import availability_model as am
    import predict
    ctx = game_context(et_date)
    if ctx.empty:
        return ctx, None
    try:
        status = am.live_status_rows(pd.Timestamp(et_date).date(), as_of)
    except Exception as e:
        print(f"  official report unavailable: {e}")
        status = None
    rows = []
    for g in ctx.itertuples():
        pred = predict.predict_game(g.home_team, g.away_team, pd.Timestamp(et_date).date(),
                                    home_rest_days=g.home_rest, away_rest_days=g.away_rest,
                                    tip=g.tip, status_rows=status, game_id=g.game_id, neutral=bool(g.is_neutral))
        rows.append({**g._asdict(), "pred": pred, "p_home": pred["final_prob"]})
    return pd.DataFrame(rows), status

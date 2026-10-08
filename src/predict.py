"""
src/predict.py — Live single-game prediction (same walk and parameters as the backtest)
========================================================================================
P(home win) = logistic((home Elo + home court + rest + availability) − (away Elo + availability))

  Elo            data/elo_ratings_current.csv (build_ratings.py, production walk)
  home court     +HC_fav when the home side is stronger, +HC_dog otherwise, 0 at neutral sites
  rest           back-to-back penalty on arena-local rest days (build_ratings.PRODUCTION)
  availability   src/availability_model.py (Gate 2), when config.yaml model.injury_adjust

Every prediction also carries:
  prob_no_avail     same ratings without the live availability adjustment
  prob_baseline_v2  the baseline_v2 model (its own ratings, no layer) — daily tracking of
                    whether the layer helps this season

The hand-typed PLAYER_RAPTOR table is archived in src/archive/player_raptor_apr2026.py.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

import build_ratings as br
import config

DATA_DIR = Path(__file__).parent.parent / "data"

INITIAL_RATING = br.INITIAL_RATING
POINTS_TO_ELO = br.POINTS_TO_ELO
HC_FAV = br.PRODUCTION.HC_fav
HC_DOG = br.PRODUCTION.HC_dog


def load_ratings(path: Path = DATA_DIR / "elo_ratings_current.csv") -> dict:
    """{team: elo} for the next game (production walk)."""
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["elo"]))


def load_baseline_ratings() -> dict | None:
    p = DATA_DIR / "elo_ratings_baseline_v2.csv"
    return load_ratings(p) if p.exists() else None


def _elo_prob(home_adj: float, away_adj: float) -> float:
    return br.elo_win_prob(home_adj, away_adj)


def _rest_elo(home_rest: int, away_rest: int, p: br.EloParams = br.PRODUCTION) -> float:
    def pen(d):
        return p.rest_b2b if d == 0 else (p.rest_1day if d == 1 else 0.0)
    return pen(min(max(home_rest, 0), 7)) - pen(min(max(away_rest, 0), 7))


def _base_prob(h_elo, a_elo, home_rest, away_rest, neutral, h_adj=0.0, a_adj=0.0,
               p: br.EloParams = br.PRODUCTION) -> tuple[float, float]:
    """Same arithmetic as build_ratings.run_walk: split home court decided on the
    ratings before the availability adjustment."""
    hc = p.HC_fav if h_elo >= a_elo else p.HC_dog
    if neutral and p.neutral_no_hc:
        hc = 0.0
    diff = (h_elo + h_adj + hc + _rest_elo(home_rest, away_rest, p)) - (a_elo + a_adj)
    return _elo_prob(diff, 0.0), hc


def injury_elo_delta(team: str, injuries: list) -> float:
    """Removed at Gate 2 (hand-typed table archived). Kept so old callers (the separate
    playoff model) run unchanged: always 0. Use predict_game(status_rows=...) instead."""
    return 0.0


def predict_game(
    home_team: str,
    away_team: str,
    game_date,
    home_rest_days: int = 3,
    away_rest_days: int = 3,
    home_injuries: list = None,          # deprecated (old ESPN-feed format), ignored
    away_injuries: list = None,
    ratings: dict = None,
    tip: pd.Timestamp | None = None,
    status_rows: pd.DataFrame | None = None,
    game_id: int | None = None,
    neutral: bool = False,
    verbose: bool = False,
) -> dict:
    """status_rows: injury_status rows (game_id, team, athlete_id, status, issued_at) from the
    latest official report before the decision time; None = no report yet (everyone
    counted as Not listed). tip: UTC tip time (defaults to 23:00Z on game_date)."""
    if home_injuries or away_injuries:
        warnings.warn("home_injuries/away_injuries are ignored; pass status_rows (official report)")
    if ratings is None:
        ratings = load_ratings()
    if isinstance(game_date, str):
        game_date = datetime.strptime(game_date, "%Y-%m-%d").date()
    if tip is None:
        tip = pd.Timestamp(datetime.combine(game_date, datetime.min.time()), tz="UTC") + pd.Timedelta(hours=23)
    h_elo = ratings.get(home_team, INITIAL_RATING)
    a_elo = ratings.get(away_team, INITIAL_RATING)

    none = {"elo": 0.0, "points": 0.0, "report_issued_at": None, "detail": None}
    h_av = a_av = none
    if config.get("model.injury_adjust"):
        import availability_model as am

        def team_rows(team):
            if status_rows is None or game_id is None:
                return None
            r = status_rows[(status_rows["game_id"] == game_id) & (status_rows["team"] == team)]
            return r if len(r) else None
        h_av = am.live_team_adjustment(home_team, tip, team_rows(home_team))
        a_av = am.live_team_adjustment(away_team, tip, team_rows(away_team))

    final_prob, hc = _base_prob(h_elo, a_elo, home_rest_days, away_rest_days, neutral, h_av["elo"], a_av["elo"])
    prob_no_avail, _ = _base_prob(h_elo, a_elo, home_rest_days, away_rest_days, neutral)
    base = load_baseline_ratings()
    prob_v2 = (_base_prob(base.get(home_team, INITIAL_RATING), base.get(away_team, INITIAL_RATING),
                          home_rest_days, away_rest_days, neutral)[0] if base else None)

    result = {
        "home_team": home_team, "away_team": away_team, "date": str(game_date),
        "home_elo_base": round(h_elo, 1), "away_elo_base": round(a_elo, 1),
        "home_inj_delta": round(h_av["elo"], 1), "away_inj_delta": round(a_av["elo"], 1),
        "home_court_bonus": hc, "rest_adj_elo": round(_rest_elo(home_rest_days, away_rest_days), 1),
        "raw_prob": round(prob_no_avail, 4),
        "prob_no_avail": round(prob_no_avail, 4),
        "prob_baseline_v2": round(prob_v2, 4) if prob_v2 is not None else None,
        "final_prob": round(final_prob, 4),
        "home_report_issued_at": h_av["report_issued_at"], "away_report_issued_at": a_av["report_issued_at"],
        "home_avail_detail": h_av["detail"], "away_avail_detail": a_av["detail"],
    }
    if verbose:
        v2 = "n/a" if prob_v2 is None else f"{prob_v2:.1%}"
        print(f"{away_team} @ {home_team} {game_date}: Elo {h_elo:.0f} vs {a_elo:.0f}, HC {hc:+.0f}, "
              f"rest {result['rest_adj_elo']:+.0f}, availability {h_av['elo']:+.0f}/{a_av['elo']:+.0f} Elo "
              f"→ P(home) {final_prob:.1%} (no layer {prob_no_avail:.1%}, baseline_v2 {v2})")
    return result

"""
src/build_ratings.py — Canonical Elo build (the production model)
==================================================================
Walk-forward Elo over every completed game (regular season, play-in, playoffs)
in tip order. This is the only code that produces ratings: the backtest and the
live pipeline use exactly this walk with exactly the same parameters.

Parameters (EloParams):
  K, HC_fav / HC_dog       split home court (+50 when home is favoured, +35 otherwise)
  carryover                fraction kept toward 1500 at each new season
  rest_b2b, rest_1day      Elo added to a team on 0 / 1 days rest
  rest_basis               "utc"   — legacy: days between UTC dates of tip-off
                           "local" — days between arena-local game dates (2.0a)
  neutral_no_hc            no home court at neutral sites (2.0b)
  decay, decay_anchor      K × decay^(anchor − season) for seasons before the anchor.
                           Depends only on each game's own season, so appending
                           new games never changes an earlier prediction (2.0c)
  tz_east, tz_west         Elo per hour of time-zone change since the team's previous
                           game, travelling east / west (retest)
  K_playoff_rs, HC_playoff play-in and playoff games update the ratings with this K
                           (0 = they do not: the April behaviour)
LEGACY reproduces the April 2026 production file byte for byte.

A per-game Elo adjustment (e.g. player availability, 2.2) can be passed to
run_walk(); it shifts the pre-game expectation, and the update uses that same
adjusted expectation.

Outputs (data/ unless --out-dir):
  elo_predictions.csv      one row per RS game: games columns + rest days,
                           actual_home_win, home_win_prob, final_prob
  elo_ratings_current.csv  team, elo = rating for the next game (after a completed
                           season: final walk ratings + offseason carryover), end_rs
  elo_ratings_history.csv  season, stage (end_rs | end_season | preseason_next), team, elo

Usage:
  python3 src/build_ratings.py [--in-season]
  python3 src/build_ratings.py --games g.csv --playoffs p.csv --out-dir /tmp/x --legacy
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

INITIAL_RATING = 1500.0
POINTS_TO_ELO = 28.0


@dataclass(frozen=True)
class EloParams:
    K: float = 10.0
    HC_fav: float = 50.0
    HC_dog: float = 35.0
    carryover: float = 0.75
    rest_b2b: float = -56.0
    rest_1day: float = 0.0
    rest_basis: str = "utc"
    neutral_no_hc: bool = False
    decay: float = 0.60
    decay_anchor: str = "2025-26"
    tz_east: float = 0.0
    tz_west: float = 0.0
    K_playoff_rs: float = 0.0
    HC_playoff: float = 50.0
    var_season: float = 0.0      # 2.4 Kalman-style: team rating variance (steady state = 1) jumps
    var_decay: float = 0.1       #   by var_season at each season start, relaxes toward 1 at rate
                                 #   var_decay per game; the update uses K × variance


@dataclass(frozen=True)
class PlayoffParams:
    """Separate playoff game/series model (predict_playoff.py), not the rating walk."""
    K: float = 3.0
    HC: float = 50.0


LEGACY = EloParams()
# baseline_v2 = step 2.0 fixes: rest on arena-local dates with the B2B penalty refit
# on 2021-22→2024-25 (−60), no home court at neutral sites, fixed decay anchor.
# Changed only through reports/experiments.csv (experiments/NN_*.py).
BASELINE_V2 = EloParams(rest_basis="local", neutral_no_hc=True, rest_b2b=-60.0)
PRODUCTION = BASELINE_V2


def load_playoff_params(path: Path = DATA / "playoff_params.json") -> PlayoffParams:
    if path.exists():
        p = json.loads(path.read_text())
        return PlayoffParams(K=float(p["K"]), HC=float(p["HC"]))
    return PlayoffParams()


# ── Elo math ────────────────────────────────────────────────────────────────

def elo_win_prob(home_elo: float, away_elo: float) -> float:
    """P(home wins) given home-adjusted ratings."""
    return 1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo) / 400.0))


def mov_mult(abs_margin: float, winner_elo_adv: float) -> float:
    """FiveThirtyEight margin-of-victory multiplier (margin floored at 1)."""
    return np.log(max(1.0, abs_margin) + 1.0) * 2.2 / (winner_elo_adv * 0.001 + 2.2)


def season_year(season: str) -> int:
    return int(str(season)[:4])


# ── Inputs ──────────────────────────────────────────────────────────────────

def walk_order(df: pd.DataFrame) -> list[str]:
    """Sort key for the walk: real tip order when tip_utc is known. Legacy inputs
    (no tip_utc, e.g. the April 2026 file) fall back to (date, game_id)."""
    return ["date", "tip_utc", "game_id"] if "tip_utc" in df.columns else ["date", "game_id"]


def _team_rest(g: pd.DataFrame, day_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Days since each team's previous game on `day_col` (first game → 7), capped 0..7.
    `g` must already be in walk order."""
    last: dict = {}
    h_out, a_out = np.zeros(len(g), dtype=int), np.zeros(len(g), dtype=int)
    days = pd.to_datetime(g[day_col]).to_numpy()
    for i, (d, ht, at) in enumerate(zip(days, g["home_team"], g["away_team"])):
        for team, out in ((ht, h_out), (at, a_out)):
            prev = last.get(team)
            out[i] = 7 if prev is None else max(0, min(int((d - prev) / np.timedelta64(1, "D")) - 1, 7))
        last[ht] = d
        last[at] = d
    return h_out, a_out


def compute_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper (UTC `date` basis): game_id, team, side, rest_days."""
    g = games.sort_values(walk_order(games)).reset_index(drop=True)
    h, a = _team_rest(g, "date")
    return pd.concat([pd.DataFrame({"game_id": g["game_id"], "team": g["home_team"], "side": "home", "rest_days": h}),
                      pd.DataFrame({"game_id": g["game_id"], "team": g["away_team"], "side": "away", "rest_days": a})])


def arena_local(tips: pd.Series, home: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Arena-local calendar date and UTC offset (hours) of each tip-off, from the
    home team's time zone."""
    from pipeline import TEAM_TZ
    dates, offsets = [], []
    for t, h in zip(pd.to_datetime(tips, utc=True), home):
        lt = t.tz_convert(ZoneInfo(TEAM_TZ.get(h, "America/New_York")))
        dates.append(pd.Timestamp(lt.date()))
        offsets.append(lt.utcoffset().total_seconds() / 3600)
    return pd.Series(dates, index=tips.index), np.array(offsets)


def check_walk_order(df: pd.DataFrame) -> int:
    """With tip_utc known, refuse to run if any team's games are not strictly in tip
    order (lookahead). Legacy inputs only warn about same-UTC-date pairs.
    History: the April (date, game_id) order walked DAL@MEM 2026-03-12 (rescheduled,
    higher ID) after the next day's MEM@DET and CLE@DAL."""
    long = pd.concat([
        df[["date", "game_id"]].assign(t=df["home_team"], pos=df.index),
        df[["date", "game_id"]].assign(t=df["away_team"], pos=df.index),
    ])
    same_date = int(long.duplicated(["date", "t"]).sum())
    if "tip_utc" in df.columns and df["tip_utc"].notna().all():
        long["tip"] = pd.to_datetime(pd.concat([df["tip_utc"], df["tip_utc"]]).to_numpy(), utc=True)
        bad = (long.sort_values("pos").groupby("t")["tip"].diff() <= pd.Timedelta(0)).sum()
        if bad:
            raise ValueError(f"{bad} games would be walked before an earlier-tipping game "
                             "of the same team (lookahead) — fix the ordering")
    elif same_date:
        warnings.warn(f"{same_date} team-dates with two games and no tip_utc to verify order")
    return same_date


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    """Sort into walk order, attach rest days (UTC and local basis), time-zone changes
    and the outcome. Refuses data that would allow lookahead."""
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(walk_order(df)).reset_index(drop=True)
    if not df["game_id"].is_unique:
        raise ValueError("duplicate game_id in games input")
    check_walk_order(df)
    if "season_type" not in df:
        df["season_type"] = "regular"
    if "is_neutral" not in df:
        df["is_neutral"] = False
    df["is_neutral"] = df["is_neutral"].fillna(False).astype(bool)
    df["home_rest_days"], df["away_rest_days"] = _team_rest(df, "date")
    if "tip_utc" in df.columns:
        df["date_arena"], offsets = arena_local(df["tip_utc"], df["home_team"])
        df["home_rest_local"], df["away_rest_local"] = _team_rest(df, "date_arena")
        last: dict = {}
        h_tz, a_tz = np.zeros(len(df)), np.zeros(len(df))
        for i, (off, ht, at, st) in enumerate(zip(offsets, df["home_team"], df["away_team"], df["season"])):
            for team, out in ((ht, h_tz), (at, a_tz)):
                prev = last.get(team)
                out[i] = 0.0 if prev is None or prev[1] != st else off - prev[0]
            last[ht] = (off, st)
            last[at] = (off, st)
        df["home_tz_change"], df["away_tz_change"] = h_tz, a_tz   # + = travelled east
    else:
        df["home_rest_local"], df["away_rest_local"] = df["home_rest_days"], df["away_rest_days"]
        df["home_tz_change"] = df["away_tz_change"] = 0.0
    df["actual_home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    return df


# ── Walk ────────────────────────────────────────────────────────────────────

@dataclass
class WalkResult:
    probs: np.ndarray                      # pre-game P(home win), aligned with prepared games
    diffs: np.ndarray                      # pre-game home − away Elo incl. home court / rest / adj
    end_rs: dict[str, dict[str, float]]    # season → team → Elo after its last regular-season game
    end_season: dict[str, dict[str, float]]  # season → team → Elo after its last game (incl. playoffs)
    final: dict[str, float]                # team → Elo after the last game


def _rest_elo(rest: np.ndarray, p: EloParams) -> np.ndarray:
    return np.where(rest == 0, p.rest_b2b, np.where(rest == 1, p.rest_1day, 0.0))


def _tz_elo(change: np.ndarray, p: EloParams) -> np.ndarray:
    return np.where(change > 0, p.tz_east * change, np.where(change < 0, p.tz_west * -change, 0.0))


def run_walk(games: pd.DataFrame, params: EloParams = PRODUCTION,
             elo_adj: np.ndarray | None = None,
             season_shift: dict | None = None) -> WalkResult:
    """`games` must come from prepare_games(). Each prediction is recorded before the
    ratings are updated with that game's result. `elo_adj` (n × 2, home/away Elo)
    shifts the pre-game expectation, and the update uses that adjusted expectation.
    `season_shift` {(season, team): Elo} is added to a team's rating when that season
    starts (after the carryover) — the preseason roster prior (2.3)."""
    p = params
    seasons = games["season"].to_numpy()
    anchor = season_year(p.decay_anchor)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}

    hi_arr = games["home_team"].map(tidx).to_numpy()
    ai_arr = games["away_team"].map(tidx).to_numpy()
    post = games["season_type"].isin(["playoff", "playin"]).to_numpy()
    neutral = games["is_neutral"].to_numpy() & p.neutral_no_hc
    if p.rest_basis == "local":
        h_rest, a_rest = games["home_rest_local"].to_numpy(), games["away_rest_local"].to_numpy()
    else:
        h_rest, a_rest = games["home_rest_days"].to_numpy(), games["away_rest_days"].to_numpy()
    # rest and travel apply to regular-season games only (the April model)
    situ = np.where(post, 0.0, _rest_elo(np.clip(h_rest, 0, 7), p) - _rest_elo(np.clip(a_rest, 0, 7), p)
                    + _tz_elo(games["home_tz_change"].to_numpy(), p)
                    - _tz_elo(games["away_tz_change"].to_numpy(), p))
    hw_arr = games["actual_home_win"].to_numpy().astype(float)
    margin = (games["home_pts"] - games["away_pts"]).abs().to_numpy().astype(float)
    # float32 decay weights, as in the Session 7 build, so results reproduce exactly
    decay_w = np.array([p.decay ** max(0, anchor - season_year(s)) for s in seasons], dtype=np.float32)
    adj = np.zeros((len(games), 2)) if elo_adj is None else np.asarray(elo_adj, dtype=float)

    ratings = np.full(len(teams), INITIAL_RATING, dtype=np.float64)
    var = np.ones(len(teams))
    probs = np.zeros(len(games))
    diffs = np.zeros(len(games))
    end_rs: dict[str, dict[str, float]] = {}
    end_season: dict[str, dict[str, float]] = {}
    cur = seasons[0] if len(seasons) else None

    for i in range(len(games)):
        if seasons[i] != cur:
            end_season[cur] = dict(zip(teams, ratings.copy()))
            end_rs.setdefault(cur, end_season[cur])
            ratings = p.carryover * ratings + (1.0 - p.carryover) * INITIAL_RATING
            var = var + p.var_season
            cur = seasons[i]
            if season_shift:
                for t, j in tidx.items():
                    ratings[j] += season_shift.get((cur, t), 0.0)
        if post[i] and cur not in end_rs:
            end_rs[cur] = dict(zip(teams, ratings.copy()))
        hi, ai = hi_arr[i], ai_arr[i]
        h_elo, a_elo = ratings[hi] + adj[i, 0], ratings[ai] + adj[i, 1]
        if post[i]:
            hc, k = p.HC_playoff, p.K_playoff_rs
        else:
            hc = p.HC_fav if h_elo - adj[i, 0] >= a_elo - adj[i, 1] else p.HC_dog
            k = p.K
        if neutral[i]:
            hc = 0.0
        h_adj = h_elo + hc + situ[i]
        pr = 1.0 / (1.0 + 10.0 ** (-(h_adj - a_elo) / 400.0))
        probs[i] = pr
        diffs[i] = h_adj - a_elo
        if k == 0:
            continue
        hw = hw_arr[i]
        w_adv = h_adj - a_elo if hw else a_elo - h_adj
        delta = k * float(decay_w[i]) * mov_mult(margin[i], w_adv) * (hw - pr)
        if p.var_season:
            ratings[hi] += delta * var[hi]
            ratings[ai] -= delta * var[ai]
            var[hi] = var[hi] * (1 - p.var_decay) + p.var_decay
            var[ai] = var[ai] * (1 - p.var_decay) + p.var_decay
        else:
            ratings[hi] += delta
            ratings[ai] -= delta

    if cur is not None:
        end_season[cur] = dict(zip(teams, ratings.copy()))
        end_rs.setdefault(cur, end_season[cur])
    return WalkResult(probs=probs, diffs=diffs, end_rs=end_rs, end_season=end_season,
                      final=dict(zip(teams, ratings)))


def run_playoff_walk(playoffs: pd.DataFrame, start: dict[str, float],
                     pp: PlayoffParams) -> tuple[np.ndarray, dict[str, float]]:
    """Separate playoff model: one season's playoffs from its end-of-RS ratings."""
    r = dict(start)
    pg = playoffs.assign(date=pd.to_datetime(playoffs["date"])).sort_values(walk_order(playoffs))
    probs = []
    for ht, at, hp, ap in zip(pg["home_team"], pg["away_team"], pg["home_pts"], pg["away_pts"]):
        h_adj = r.get(ht, INITIAL_RATING) + pp.HC
        a = r.get(at, INITIAL_RATING)
        pr = elo_win_prob(h_adj, a)
        probs.append(pr)
        hw = 1.0 if hp > ap else 0.0
        w_adv = h_adj - a if hw else a - h_adj
        delta = pp.K * mov_mult(abs(hp - ap), w_adv) * (hw - pr)
        r[ht] = r.get(ht, INITIAL_RATING) + delta
        r[at] = a - delta
    return np.array(probs), r


def apply_carryover(ratings: dict[str, float], params: EloParams = PRODUCTION) -> dict[str, float]:
    return {t: params.carryover * e + (1 - params.carryover) * INITIAL_RATING
            for t, e in ratings.items()}


# ── Build ───────────────────────────────────────────────────────────────────

def load_games(games_path: Path = DATA / "games_raw.csv",
               post_paths: tuple[Path, ...] = (DATA / "playin_games_raw.csv",
                                               DATA / "playoff_games_raw.csv")) -> pd.DataFrame:
    """Regular season + play-in + playoffs, tagged with season_type."""
    frames = [pd.read_csv(games_path).assign(season_type="regular")]
    for path, st in zip(post_paths, ("playin", "playoff")):
        if Path(path).exists():
            frames.append(pd.read_csv(path).assign(season_type=st))
    return pd.concat(frames, ignore_index=True)


def availability_on(params: EloParams) -> bool:
    """The player-availability layer applies to the production walk when enabled."""
    import config
    return params == PRODUCTION and bool(config.get("model.injury_adjust"))


def build(games: pd.DataFrame, params: EloParams = PRODUCTION, season_complete: bool = True,
          availability: bool | None = None):
    """Returns (RS predictions, current ratings, history). With the availability layer the
    walk uses the official-report adjustments (Gate 2); predictions also carry
    prob_baseline_v2 (same params, no layer) so the layer can be tracked."""
    prepared = prepare_games(games)
    use_av = availability_on(params) if availability is None else availability
    adj = None
    if use_av:
        import availability_model
        adj = availability_model.historical_adjustments(prepared)
        prepared["home_avail_elo"], prepared["away_avail_elo"] = adj[:, 0], adj[:, 1]
    walk = run_walk(prepared, params, adj)
    prepared["home_win_prob"] = walk.probs
    prepared["final_prob"] = walk.probs   # no post-hoc layers (model_audit reads this)
    if use_av:
        prepared["prob_baseline_v2"] = run_walk(prepared, params).probs
    rs = prepared[prepared["season_type"] == "regular"]
    keep = [c for c in games.columns if c != "season_type"] + \
           ["home_rest_days", "away_rest_days", "actual_home_win", "home_win_prob", "final_prob"]
    if use_av:
        keep += ["home_avail_elo", "away_avail_elo", "prob_baseline_v2"]
    if params.rest_basis == "local":
        keep[keep.index("away_rest_days") + 1:keep.index("away_rest_days") + 1] = ["home_rest_local", "away_rest_local"]
    preds = rs[keep].reset_index(drop=True)

    hist = []
    for stage, table in (("end_rs", walk.end_rs), ("end_season", walk.end_season)):
        for season, r in table.items():
            hist += [{"season": season, "stage": stage, "team": t, "elo": e} for t, e in r.items()]
    last = prepared["season"].iloc[-1]
    cur = pd.DataFrame({"team": list(walk.final), "end_rs": [walk.end_rs[last][t] for t in walk.final],
                        "end_season": list(walk.final.values())})
    if season_complete:
        cur["elo"] = params.carryover * cur["end_season"] + (1 - params.carryover) * INITIAL_RATING
        hist += [{"season": last, "stage": "preseason_next", "team": t, "elo": e}
                 for t, e in zip(cur["team"], cur["elo"])]
    else:
        cur["elo"] = cur["end_season"]
    hist = pd.DataFrame(hist)
    hist["elo"] = hist["elo"].round(2)
    cur = cur[["team", "elo", "end_rs", "end_season"]].round(1).sort_values("elo", ascending=False)
    return preds, cur.reset_index(drop=True), hist


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Canonical Elo build")
    ap.add_argument("--games", type=Path, default=DATA / "games_raw.csv")
    ap.add_argument("--playoffs", type=Path, default=DATA / "playoff_games_raw.csv")
    ap.add_argument("--playin", type=Path, default=DATA / "playin_games_raw.csv")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--legacy", action="store_true", help="April 2026 parameters (reproduction)")
    ap.add_argument("--in-season", action="store_true",
                    help="latest season still running: current = walk ratings, no carryover")
    args = ap.parse_args(argv)

    params = LEGACY if args.legacy else PRODUCTION
    games = load_games(args.games, (args.playin, args.playoffs))
    preds, cur, hist = build(games, params, season_complete=not args.in_season)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out_dir / "elo_predictions.csv", index=False)
    cur.to_csv(args.out_dir / "elo_ratings_current.csv", index=False)
    hist.to_csv(args.out_dir / "elo_ratings_history.csv", index=False)
    if availability_on(params):
        # comparison model without the layer, for daily with/without tracking
        _, cur_v2, _ = build(games, params, season_complete=not args.in_season, availability=False)
        cur_v2.to_csv(args.out_dir / "elo_ratings_baseline_v2.csv", index=False)

    print(f"params: {asdict(params)}  availability layer: {availability_on(params)}")
    for season, g in preds.groupby("season"):
        b = ((g["home_win_prob"] - g["actual_home_win"]) ** 2).mean()
        acc = ((g["home_win_prob"] > 0.5) == g["actual_home_win"]).mean()
        extra = (f"  (baseline_v2 {((g['prob_baseline_v2'] - g['actual_home_win']) ** 2).mean():.4f})"
                 if "prob_baseline_v2" in g else "")
        print(f"  {season}: {len(g):5d} games  Brier {b:.4f}  acc {acc:.1%}{extra}")
    print(f"wrote {args.out_dir}/elo_predictions.csv, elo_ratings_current.csv, elo_ratings_history.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

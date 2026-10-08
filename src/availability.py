"""
src/availability.py — Player availability layer (Phase 2.2)
============================================================
Team Elo adjustment for who is expected to play, relative to the lineup the team's
Elo already reflects:

    adj_team(g) = λ × 28 × Σ_i (E_i(g) − B_i(g)) / 48 × (r_i(g) − r_repl)

  r_i(g)   player value, points of team margin per 48 minutes, from games before g only:
           box-score value (coefficients β fit on the training seasons by regressing team
           margin on team box totals), exponentially weighted over appearances and shrunk
           toward replacement level by minutes played. No history → replacement level.
  E_i(g)   expected minutes in g = P(plays) × projected minutes when playing
           (mean of his last 10 appearances).
  B_i(g)   baseline minutes = his mean minutes over the team's previous 10 games
           (0 when he did not play) — the lineup Elo has been seeing.
  P(plays) version "actual":  1 if he played in g, else 0 (ceiling; uses the box score)
           version "pregame": P(plays | status) from the latest official injury report
           issued ≤ tip − 60 min, estimated on the training seasons; players not on the
           report get P(plays | not listed).

Candidates for a team-game: every player with a box-score row (played or DNP) for that
team in its previous 20 games. Box scores flagged box_incomplete are excluded from
rating training (their rows still count for presence).

The adjustment enters the Elo walk through build_ratings.run_walk(elo_adj=...).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BOX_FEATURES = ["fgm", "fga", "fg3m", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]
POINTS_TO_ELO = 28.0


@dataclass(frozen=True)
class AvailParams:
    value: str = "blend"            # "gmsc" | "pm" | "blend" | "box" (see player_values)
    halflife_games: float = 20.0    # EW half-life of a player's value, in appearances
    shrink_minutes: float = 250.0   # prior weight (minutes) at replacement level
    window_minutes: int = 82        # appearances counted for the shrinkage weight
    proj_games: int = 10            # appearances for projected minutes when playing
    baseline_games: int = 0         # team games for baseline minutes; 0 = no baseline:
                                    # adjustment = full expected-lineup strength (chosen
                                    # on training seasons: −0.0089 vs −0.0032 for 82)
    presence_games: int = 82        # team games for the candidate set
    ridge: float = 1.0


# ── Data ────────────────────────────────────────────────────────────────────

def load_player_games() -> pd.DataFrame:
    pg = pd.read_parquet(DATA / "player_games.parquet")
    ga = pd.read_csv(DATA / "games_all.csv", usecols=["game_id", "tip_utc", "home_team", "away_team",
                                                      "home_pts", "away_pts"])
    pg = pg.merge(ga, on="game_id", how="left")
    pg["tip"] = pd.to_datetime(pg["tip_utc"], utc=True)
    pg["played"] = (~pg["did_not_play"]) & (pg["min"].fillna(0) > 0)
    pg["min"] = pg["min"].fillna(0.0)
    for c in BOX_FEATURES:
        pg[c] = pg[c].fillna(0.0)
    return pg.sort_values(["tip", "game_id", "team", "athlete_id"]).reset_index(drop=True)


# ── Box-score value model ───────────────────────────────────────────────────

def fit_box_model(pg: pd.DataFrame, seasons: list[str], ridge: float = 1.0) -> tuple[np.ndarray, float]:
    """Regress team-game margin on team box totals (training seasons, complete boxes)."""
    d = pg[pg["season"].isin(seasons) & pg["played"] & ~pg["box_incomplete"]]
    tm = d.groupby(["game_id", "team"])[BOX_FEATURES].sum()
    g = pg.drop_duplicates("game_id").set_index("game_id")
    idx = tm.index.to_frame(index=False)
    home = idx["team"].to_numpy() == g.loc[idx["game_id"], "home_team"].to_numpy()
    hp = g.loc[idx["game_id"], "home_pts"].to_numpy()
    ap = g.loc[idx["game_id"], "away_pts"].to_numpy()
    y = np.where(home, hp - ap, ap - hp).astype(float)
    X = tm.to_numpy(float)
    mu = X.mean(axis=0)
    Xc = X - mu
    beta = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ (y - y.mean()))
    intercept = float(y.mean() - mu @ beta)
    return beta, intercept


def game_score(pg: pd.DataFrame) -> np.ndarray:
    """Hollinger Game Score."""
    return (pg["pts"].fillna(0) + 0.4 * pg["fgm"] - 0.7 * pg["fga"] - 0.4 * (pg["fta"] - pg["ftm"])
            + 0.7 * pg["oreb"] + 0.3 * pg["dreb"] + pg["stl"] + 0.7 * pg["ast"] + 0.7 * pg["blk"]
            - 0.4 * pg["pf"] - pg["tov"]).to_numpy(float)


def relative_pm(pg: pd.DataFrame) -> np.ndarray:
    """On-court plus-minus minus the team's full-game margin prorated to his minutes."""
    home = pg["team"] == pg["home_team"]
    margin = np.where(home, pg["home_pts"] - pg["away_pts"], pg["away_pts"] - pg["home_pts"])
    return (pg["plus_minus"].fillna(0).to_numpy(float) - margin * pg["min"].to_numpy() / 48.0)


def player_values(pg: pd.DataFrame, beta: np.ndarray, intercept: float, kind: str = "gmsc",
                  train_seasons: list[str] | None = None) -> pd.Series:
    """Per player-game value. "box": fitted box model (team margin points; rebound-heavy),
    "gmsc": Game Score, "pm": relative on-court plus-minus, "blend": standardized mean of
    gmsc and pm. The scale is absorbed by λ, so only the ranking and spread matter."""
    if kind == "box":
        v = pg[BOX_FEATURES].to_numpy(float) @ beta + intercept * pg["min"].to_numpy() / 240.0
    elif kind == "gmsc":
        v = game_score(pg)
    elif kind == "pm":
        v = relative_pm(pg)
    elif kind == "blend":
        g, m = game_score(pg), relative_pm(pg)
        played = pg["played"].to_numpy()
        if train_seasons is not None:   # scale constants from training seasons only
            played = played & pg["season"].isin(train_seasons).to_numpy()
        per48_g = g[played].sum() / pg.loc[played, "min"].sum()
        # put pm on the Game Score per-minute scale before averaging
        sg = np.std(g[played] / np.maximum(pg.loc[played, "min"], 1))
        sm = np.std(m[played] / np.maximum(pg.loc[played, "min"], 1))
        v = 0.5 * g + 0.5 * (m * sg / sm + per48_g * pg["min"].to_numpy())
    else:
        raise ValueError(kind)
    return pd.Series(np.where(pg["played"], v, 0.0), index=pg.index)


def replacement_level(pg: pd.DataFrame, value: pd.Series, seasons: list[str]) -> float:
    """Minute-weighted per-48 value of deep-bench players (season mpg < 12) in training seasons."""
    d = pg[pg["season"].isin(seasons) & pg["played"] & ~pg["box_incomplete"]].assign(v=value)
    ps = d.groupby(["season", "athlete_id"]).agg(v=("v", "sum"), m=("min", "sum"), n=("game_id", "size"))
    bench = ps[(ps["m"] / ps["n"]) < 12]
    return float(bench["v"].sum() / bench["m"].sum() * 48)


# ── Time-aware player state ─────────────────────────────────────────────────

def player_state(pg: pd.DataFrame, value: pd.Series, r_repl: float, ap: AvailParams) -> pd.DataFrame:
    """State of each player AFTER each of his appearances (tip-ordered): shrunk per-48
    rating and projected minutes when playing. Looked up as-of strictly before a game."""
    d = pg[pg["played"]].assign(v=value[pg["played"]]).copy()
    d.loc[d["box_incomplete"], ["v"]] = np.nan         # excluded from rating training
    d = d.sort_values(["athlete_id", "tip"])
    g = d.groupby("athlete_id", sort=False)
    alpha = 1 - 0.5 ** (1 / ap.halflife_games)
    ok_min = d["min"].where(d["v"].notna())
    ew_v = g["v"].transform(lambda s: s.ewm(alpha=alpha, ignore_na=True).mean())
    ew_m = ok_min.groupby(d["athlete_id"]).transform(lambda s: s.ewm(alpha=alpha, ignore_na=True).mean())
    raw = ew_v / ew_m * 48
    M = ok_min.fillna(0).groupby(d["athlete_id"]).transform(
        lambda s: s.rolling(ap.window_minutes, min_periods=1).sum())
    rating = (M * raw.fillna(r_repl) + ap.shrink_minutes * r_repl) / (M + ap.shrink_minutes)
    proj = g["min"].transform(lambda s: s.rolling(ap.proj_games, min_periods=1).mean())
    return pd.DataFrame({"athlete_id": d["athlete_id"].to_numpy(), "tip": d["tip"].to_numpy(),
                         "rating": rating.to_numpy(), "proj_min": proj.to_numpy()})


def candidates(pg: pd.DataFrame, ap: AvailParams) -> pd.DataFrame:
    """One row per (team-game, player with a box row for that team in its previous
    `presence_games` games): baseline minutes over the previous `baseline_games`, and
    `games_since_seen` = team games since his last box row (played or DNP) for it."""
    tg = pg.drop_duplicates(["game_id", "team"])[["game_id", "team", "tip"]].sort_values("tip")
    mins = pg.groupby(["game_id", "team", "athlete_id"])["min"].sum()
    rows = []
    for team, games in tg.groupby("team", sort=False):
        hist: deque = deque(maxlen=ap.presence_games)     # (game_id, {player: minutes})
        seen: dict = {}                                    # player → team-game index last seen
        for k, (gid, tip) in enumerate(zip(games["game_id"], games["tip"])):
            if hist:
                recent = list(hist)[-ap.baseline_games:] if ap.baseline_games else []
                players = set().union(*(m.keys() for _, m in hist))
                for pid in players:
                    base = sum(m.get(pid, 0.0) for _, m in recent) / len(recent) if recent else 0.0
                    rows.append((gid, team, tip, pid, base, k - seen[pid]))
            try:
                cur = mins.loc[(gid, team)].to_dict()
            except KeyError:
                cur = {}
            for pid in cur:
                seen[pid] = k
            hist.append((gid, cur))
    return pd.DataFrame(rows, columns=["game_id", "team", "tip", "athlete_id", "baseline_min",
                                       "games_since_seen"])


def attach_state(cand: pd.DataFrame, state: pd.DataFrame, pg: pd.DataFrame, r_repl: float) -> pd.DataFrame:
    """As-of joins (strictly before tip): rating, projected minutes, and the team of the
    player's latest box row (so traded players stop counting for the old team). Plus
    whether he actually played in this game (version 'actual' / P(plays) estimation)."""
    c = cand.sort_values("tip")
    s = state.sort_values("tip")
    c = pd.merge_asof(c, s, on="tip", by="athlete_id", allow_exact_matches=False, direction="backward")
    last = pg[["athlete_id", "tip", "team"]].rename(columns={"team": "last_team"}).sort_values("tip")
    c = pd.merge_asof(c, last, on="tip", by="athlete_id", allow_exact_matches=False, direction="backward")
    c["rating"] = c["rating"].fillna(r_repl)
    c["proj_min"] = c["proj_min"].fillna(0.0)
    played = pg[pg["played"]].groupby(["game_id", "team", "athlete_id"]).size()
    key = pd.MultiIndex.from_frame(c[["game_id", "team", "athlete_id"]])
    c["played"] = played.reindex(key).notna().to_numpy()
    c["on_team"] = (c["last_team"] == c["team"]).to_numpy()
    return c.reset_index(drop=True)


def active_mask(c: pd.DataFrame, known_today: np.ndarray, recent_games: int = 5) -> np.ndarray:
    """Candidates that count: still on the team, and seen in one of its last
    `recent_games` box scores or known to be around today (listed on the report for
    version 'pregame'; actually played for version 'actual')."""
    return c["on_team"].to_numpy() & ((c["games_since_seen"].to_numpy() <= recent_games) | known_today)


def roster_candidates(pg: pd.DataFrame, listed: pd.DataFrame | None = None,
                      grace_days: float = 7.0) -> pd.DataFrame:
    """Roster-aware candidates (what a live roster knows, before tip): a player belongs to
    the team of his NEXT box-score row (played or DNP) strictly after the tip, in the
    same season — trades and signings are public before the player's next game. With no
    later row that season, he stays with his last team for `grace_days`. Players listed
    on a team's injury report for the game (`listed`: game_id, team, athlete_id) are
    added. Returns game_id, team, tip, athlete_id, baseline_min (0), games_since_seen (0)."""
    pg = pg.assign(tip_n=pg["tip"].dt.tz_convert(None))            # naive UTC for numpy
    rows = pg[["athlete_id", "tip_n", "team", "season"]].drop_duplicates(["athlete_id", "tip_n"]) \
        .sort_values(["athlete_id", "tip_n"]).rename(columns={"tip_n": "tip"})
    season_start = pg.groupby("season")["tip_n"].min()
    iv = []   # (team, start, end, athlete)
    for aid, r in rows.groupby("athlete_id", sort=False):
        tips, teams, seas = r["tip"].to_numpy(), r["team"].to_numpy(), r["season"].to_numpy()
        for k in range(len(tips)):
            first_of_season = k == 0 or seas[k - 1] != seas[k]
            if first_of_season:   # from season start to his first row, he is with his first team
                iv.append((teams[k], season_start[seas[k]].to_datetime64() - np.timedelta64(1, "D"), tips[k], aid))
            if k + 1 < len(tips) and seas[k + 1] == seas[k]:
                iv.append((teams[k + 1], tips[k], tips[k + 1], aid))
            else:
                iv.append((teams[k], tips[k], tips[k] + np.timedelta64(int(grace_days * 86400), "s"), aid))
    ivs = pd.DataFrame(iv, columns=["team", "start", "end", "athlete_id"])
    tg = pg.drop_duplicates(["game_id", "team"])[["game_id", "team", "tip", "tip_n"]]
    out = []
    for team, games_ in tg.groupby("team", sort=False):
        t_iv = ivs[ivs["team"] == team]
        starts = t_iv["start"].to_numpy("datetime64[ns]")
        ends = t_iv["end"].to_numpy("datetime64[ns]")
        aids = t_iv["athlete_id"].to_numpy()
        for gid, tip in zip(games_["game_id"], games_["tip_n"].to_numpy("datetime64[ns]")):
            m = (starts <= tip) & (tip < ends)   # [start, end): at a row's own tip the next row decides
            for a in np.unique(aids[m]):
                out.append((gid, team, a))
    cand = pd.DataFrame(out, columns=["game_id", "team", "athlete_id"])
    if listed is not None and len(listed):
        lst = listed.dropna(subset=["athlete_id"]).astype({"athlete_id": int})[["game_id", "team", "athlete_id"]]
        cand = pd.concat([cand, lst[lst["game_id"].isin(tg["game_id"])]]).drop_duplicates()
    cand = cand.merge(tg.drop(columns="tip_n"), on=["game_id", "team"])
    cand["baseline_min"] = 0.0
    cand["games_since_seen"] = 0
    return cand


def recency_mask(cand: pd.DataFrame, pg: pd.DataFrame, listed_now: np.ndarray,
                 days: float = 14, early_games: int = 5) -> np.ndarray:
    """Roster mode filter (exp 03e, fixed a priori): the player had a box row (any team)
    in the last `days` days, or is on the report, or it is one of the team's first
    `early_games` games of the season (when nobody has recent rows)."""
    c = cand[["tip", "athlete_id"]].reset_index().sort_values("tip")
    rows = pg[["athlete_id", "tip"]].drop_duplicates()
    rows = rows.assign(prev_tip=rows["tip"]).sort_values("tip")
    c = pd.merge_asof(c, rows, on="tip", by="athlete_id", allow_exact_matches=False,
                      direction="backward").set_index("index").sort_index()
    recent = ((c["tip"] - c["prev_tip"]).dt.total_seconds() <= days * 86400).to_numpy()
    tg = pg.drop_duplicates(["game_id", "team"])[["game_id", "team", "tip", "season"]].sort_values("tip")
    tg["k"] = tg.groupby(["team", "season"]).cumcount()
    k = cand[["game_id", "team"]].merge(tg[["game_id", "team", "k"]], how="left")["k"].to_numpy()
    return recent | listed_now | (k < early_games)


# ── Adjustment ──────────────────────────────────────────────────────────────

def team_adjustment(c: pd.DataFrame, p_plays: np.ndarray, r_repl: float,
                    mask: np.ndarray | None = None) -> pd.Series:
    """Σ (E − B)/48 × (r − r_repl) per (game_id, team), in points, over `mask` rows."""
    contrib = (p_plays * c["proj_min"].to_numpy() - c["baseline_min"].to_numpy()) / 48.0 \
        * (c["rating"].to_numpy() - r_repl)
    if mask is not None:
        contrib = np.where(mask, contrib, 0.0)
    return pd.Series(contrib, index=c.index).groupby([c["game_id"], c["team"]]).sum()


def elo_adjustments(games: pd.DataFrame, team_pts: pd.Series, lam: float) -> np.ndarray:
    """(n × 2) Elo adjustments aligned with a prepared games frame; teams/games without
    box data (e.g. future games) get 0."""
    h = team_pts.reindex(pd.MultiIndex.from_arrays([games["game_id"], games["home_team"]])).fillna(0).to_numpy()
    a = team_pts.reindex(pd.MultiIndex.from_arrays([games["game_id"], games["away_team"]])).fillna(0).to_numpy()
    return lam * POINTS_TO_ELO * np.column_stack([h, a])


@dataclass
class Layer:
    cand: pd.DataFrame        # candidates with state + played
    r_repl: float
    beta: np.ndarray
    intercept: float


def build_layer(train_seasons: list[str], ap: AvailParams = AvailParams(),
                pg: pd.DataFrame | None = None) -> Layer:
    pg = load_player_games() if pg is None else pg
    beta, intercept = fit_box_model(pg, train_seasons, ap.ridge)
    value = player_values(pg, beta, intercept, ap.value, train_seasons)
    r_repl = replacement_level(pg, value, train_seasons)
    state = player_state(pg, value, r_repl, ap)
    cand = attach_state(candidates(pg, ap), state, pg, r_repl)
    return Layer(cand=cand, r_repl=r_repl, beta=beta, intercept=intercept)

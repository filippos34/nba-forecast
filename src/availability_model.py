"""
src/availability_model.py — Production player-availability layer (adopted at Gate 2)
======================================================================================
The frozen model tested in experiments/03c_availability_strict.py. Everything was fit on
2021-22 → 2023-24 only: player-value kind (blend of Game Score and relative plus-minus),
half-life 160 appearances, shrink 500 min to replacement level, no baseline (adjustment =
expected-lineup strength), P(plays | status) (data/model/p_plays_status.csv), λ = 0.8.
Settings live in config.yaml `availability:`.

    Elo adjustment for a team in a game = λ × 28 × Σ_i P_i × proj_min_i / 48 × (r_i − r_repl)

Membership (`availability.membership`):
  last_row  (adopted, = the backtest): players whose latest box-score row is for the
            team, seen in one of its last 5 box scores or listed on its report.
  roster    (experiment 03e, pending decision): team of the player's next box row / the
            live ESPN roster.

Used by build_ratings (historical walk, official reports at T−60) and predict.py (live,
latest official report before the decision time).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import availability as av
import config
import injury_status as ist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
P_TABLE = DATA / "model" / "p_plays_status.csv"


def settings() -> dict:
    return config.get("availability")


def avail_params() -> av.AvailParams:
    s = settings()
    return av.AvailParams(value=s["value"], halflife_games=float(s["halflife_games"]),
                          shrink_minutes=float(s["shrink_minutes"]),
                          baseline_games=int(s["baseline_games"]),
                          presence_games=int(s["presence_games"]))


@dataclass
class Fitted:
    pg: pd.DataFrame
    state: pd.DataFrame
    r_repl: float
    p_table: pd.Series
    lam: float


@lru_cache(maxsize=1)
def fitted() -> Fitted:
    s = settings()
    pg = av.load_player_games()
    fit = list(s["fit_seasons"])
    ap = avail_params()
    value = av.player_values(pg, None, 0.0, ap.value, fit)
    r_repl = av.replacement_level(pg, value, fit)
    state = av.player_state(pg, value, r_repl, ap)
    p_table = pd.read_csv(P_TABLE, index_col=0)["p_plays"]
    return Fitted(pg=pg, state=state, r_repl=r_repl, p_table=p_table, lam=float(s["lambda"]))


def _p_plays(status: pd.Series, table: pd.Series) -> np.ndarray:
    p = status.map(table).to_numpy(float)
    return np.where(np.isnan(p), table["Not listed"], p)


def historical_adjustments(prepared_games: pd.DataFrame, status_rows: pd.DataFrame | None = None) -> np.ndarray:
    """(n × 2) Elo adjustments for a prepared games frame, from the official reports at T−60."""
    f = fitted()
    status_rows = pd.read_parquet(ist.OUT) if status_rows is None else status_rows
    ap = avail_params()
    if settings()["membership"] == "roster":
        listed = status_rows[status_rows["status"] != "Not listed"]
        cand = av.attach_state(av.roster_candidates(f.pg, listed), f.state, f.pg, f.r_repl)
        st = ist.status_for(cand, status_rows)
        mask = av.recency_mask(cand, f.pg, st.notna().to_numpy() & (st != "Not listed").to_numpy())
    else:
        cand = av.attach_state(av.candidates(f.pg, ap), f.state, f.pg, f.r_repl)
        st = ist.status_for(cand, status_rows)
        mask = av.active_mask(cand, st.notna().to_numpy() & (st != "Not listed").to_numpy())
    pts = av.team_adjustment(cand, _p_plays(st, f.p_table), f.r_repl, mask)
    return av.elo_adjustments(prepared_games, pts, f.lam)


def current_roster(team: str, as_of: pd.Timestamp) -> pd.Series | None:
    """athlete_ids on `team` in the latest ESPN roster snapshot fetched before `as_of`."""
    path = DATA / "rosters_2026_27.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path)
    r["fetched_at"] = pd.to_datetime(r["fetched_at"], utc=True)
    r = r[r["fetched_at"] <= as_of]
    if r.empty:
        return None
    r = r[r["fetched_at"] == r["fetched_at"].max()]
    return r.loc[r["team"] == team, "athlete_id"].astype(int)


def live_candidates(team: str, tip: pd.Timestamp) -> pd.DataFrame:
    """Players counted for `team` in a game at `tip` (UTC), with rating / projected minutes
    as of tip.
      last_row: latest box row for this team, seen in one of its last 5 box scores (03c).
      roster:   on the latest ESPN roster snapshot; seen (any team) in the last 14 days,
                or one of the team's first 5 games of the season (03e)."""
    f = fitted()
    pg = f.pg[f.pg["tip"] < tip]
    if settings()["membership"] == "roster":
        members = current_roster(team, tip)
        if members is None:
            raise RuntimeError("membership=roster needs data/rosters_2026_27.csv (python3 src/rosters.py)")
        last_seen = pg.groupby("athlete_id")["tip"].max()
        n_team_games = pg[(pg["team"] == team) & (pg["season"] == pg["season"].max())]["game_id"].nunique() \
            if len(pg) and (tip - pg["tip"].max()) < pd.Timedelta(days=90) else 0
        recent = (tip - members.map(last_seen)) <= pd.Timedelta(days=14)
        seen = set(members[recent.fillna(False).to_numpy() | (n_team_games < 5)])
    else:
        last = pg.sort_values("tip").groupby("athlete_id").tail(1)
        members = last[last["team"] == team]["athlete_id"]
        recent_games = pg[pg["team"] == team].drop_duplicates("game_id").sort_values("tip")["game_id"].tail(5)
        seen = set(pg[pg["game_id"].isin(recent_games) & (pg["team"] == team)]["athlete_id"])
    st = f.state[f.state["tip"] < tip].sort_values("tip").groupby("athlete_id").tail(1).set_index("athlete_id")
    c = pd.DataFrame({"athlete_id": members.to_numpy()})
    c["seen_recently"] = c["athlete_id"].isin(seen)
    c["rating"] = c["athlete_id"].map(st["rating"]).fillna(f.r_repl)
    c["proj_min"] = c["athlete_id"].map(st["proj_min"]).fillna(0.0)
    names = f.pg.drop_duplicates("athlete_id", keep="last").set_index("athlete_id")["player_name"]
    c["player_name"] = c["athlete_id"].map(names)
    return c


def live_team_adjustment(team: str, tip: pd.Timestamp, team_status: pd.DataFrame | None) -> dict:
    """Elo adjustment for one team in an upcoming game. `team_status`: rows of
    injury_status for this game+team (athlete_id, status, issued_at); None = no report
    covering the team yet (everyone counted as Not listed). Returns the adjustment and the
    player-level detail used (for reports)."""
    f = fitted()
    c = live_candidates(team, tip)
    status = pd.Series("Not listed", index=c.index)
    issued = None
    if team_status is not None and len(team_status):
        s = team_status.dropna(subset=["athlete_id"]).astype({"athlete_id": int})
        s = s.drop_duplicates("athlete_id").set_index("athlete_id")["status"]
        status = c["athlete_id"].map(s).fillna("Not listed")
        issued = pd.Timestamp(team_status["issued_at"].max())
        extra = set(s.index) - set(c["athlete_id"])
        if extra:   # listed players who are not members yet (e.g. new signing): counted, at their state
            add = pd.DataFrame({"athlete_id": sorted(extra)})
            stt = f.state[f.state["tip"] < tip].sort_values("tip").groupby("athlete_id").tail(1).set_index("athlete_id")
            add["seen_recently"] = False
            add["rating"] = add["athlete_id"].map(stt["rating"]).fillna(f.r_repl)
            add["proj_min"] = add["athlete_id"].map(stt["proj_min"]).fillna(0.0)
            c = pd.concat([c, add], ignore_index=True)
            status = pd.concat([status, add["athlete_id"].map(s)], ignore_index=True)
    listed = (status != "Not listed").to_numpy()
    counted = c["seen_recently"].to_numpy() | listed
    p = _p_plays(status, f.p_table)
    pre = settings().get("p_plays_pre_report")
    if issued is None and pre is not None:
        # No official report covers this team yet: the expected P(plays) before the report (fit 2021-24,
        # experiments/12), so a morning forecast = the expected value of the validated T−60 forecast.
        p = np.full(len(c), float(pre))
        status = pd.Series("No report yet", index=c.index)
    contrib = np.where(counted, p * c["proj_min"].to_numpy() / 48.0 * (c["rating"].to_numpy() - f.r_repl), 0.0)
    detail = c.assign(status=status.to_numpy(), p_plays=p, counted=counted, contrib_pts=contrib)
    return {"elo": f.lam * 28.0 * float(contrib.sum()), "points": float(contrib.sum()),
            "report_issued_at": issued, "detail": detail}


def write_p_table(table: pd.DataFrame):
    P_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(P_TABLE)


def live_status_rows(game_day, as_of: pd.Timestamp | None = None, n_files: int = 4) -> pd.DataFrame | None:
    """Status rows for games on `game_day` (US-Eastern date) from the latest official
    reports issued ≤ as_of (default now): downloads the day's labels up to now (cached),
    parses the last `n_files` available, and matches them like the backtest (T−60 or as_of,
    whichever is earlier)."""
    import injury_reports as ir
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    labels = [(lab, t) for lab, t in ir.labels_for(game_day) if pd.Timestamp(t) <= as_of]
    missing = ir._missing()
    paths = []
    for lab, _ in reversed(labels):
        path = ir.fetch_pdf(game_day, lab, missing)
        if path:
            paths.append(path)
        if len(paths) >= n_files:
            break
    if not paths:
        return None
    frames = []
    for path in paths:
        issued, rows = ir.parse_pdf(path)
        if issued is None or not rows:
            continue
        df = pd.DataFrame(rows)
        df["issued_at"] = issued.astimezone(pd.Timestamp.now(tz="UTC").tz)
        df["file"] = str(path.relative_to(ir.RAW))
        frames.append(df)
    if not frames:
        return None
    return ist.build(pd.concat(frames, ignore_index=True), as_of=as_of, out_path=None)


def roster_mismatches(as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Players whose team in the latest ESPN roster snapshot differs from the team of their
    last box score — offseason / deadline moves, or errors in the roster feed. Listed in the
    daily report. Columns: athlete_id, player_name, roster_team, last_box_team, last_box_date."""
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    path = DATA / "rosters_2026_27.csv"
    if not path.exists():
        return pd.DataFrame()
    r = pd.read_csv(path)
    r["fetched_at"] = pd.to_datetime(r["fetched_at"], utc=True)
    r = r[r["fetched_at"] <= as_of]
    r = r[r["fetched_at"] == r["fetched_at"].max()][["athlete_id", "player_name", "team"]]
    pg = pd.read_parquet(DATA / "player_games.parquet", columns=["athlete_id", "team", "date_local", "game_id"])
    last = pg.sort_values(["date_local", "game_id"]).groupby("athlete_id").tail(1)
    m = r.merge(last, on="athlete_id", how="inner", suffixes=("_roster", "_box"))
    out = m[m["team_roster"] != m["team_box"]].rename(columns={
        "team_roster": "roster_team", "team_box": "last_box_team", "date_local": "last_box_date"})
    return out[["athlete_id", "player_name", "roster_team", "last_box_team", "last_box_date"]] \
        .sort_values(["roster_team", "player_name"]).reset_index(drop=True)

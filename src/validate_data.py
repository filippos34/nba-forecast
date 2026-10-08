"""
src/validate_data.py — Data-quality checks → reports/data_quality.md
=====================================================================
Every check reports PASS / WARN / FAIL with numbers. Exit code 1 on any FAIL.
Also writes reports/data_manifest.json (rows + sha256 per file) and compares it
with the previous run's manifest.

Usage: python3 src/validate_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ratings import walk_order  # noqa: E402
from pipeline import NBA_TEAMS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MANIFEST = REPORTS / "data_manifest.json"
MANIFEST_FILES = ["games_all.csv", "games_raw.csv", "playoff_games_raw.csv", "playin_games_raw.csv",
                  "schedule_2026_27.csv", "player_games.parquet", "rosters_2026_27.csv",
                  "roster_changes_2026.csv", "transactions_2026.csv", "injury_snapshots.csv",
                  "elo_predictions.csv", "elo_ratings_current.csv"]
RS_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []   # section, check, status, detail
        self.details: list[str] = []

    def add(self, section, check, ok, detail="", warn=False):
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.rows.append((section, check, status, detail))
        return ok

    @property
    def failed(self):
        return any(r[2] == "FAIL" for r in self.rows)


def _load(name, **kw):
    p = DATA / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, **kw)


# ── Games ───────────────────────────────────────────────────────────────────

def check_games(rep: Report, games: pd.DataFrame, label: str, expect_per_team: int | None):
    sec = f"Games: {label}"
    rep.add(sec, "no duplicate game_id", games["game_id"].is_unique,
            f"{games['game_id'].duplicated().sum()} dupes")
    miss = games[["home_pts", "away_pts"]].isna().any(axis=1).sum()
    rep.add(sec, "no missing scores", miss == 0, f"{miss} missing")
    bad = games[~games["home_pts"].between(50, 200) | ~games["away_pts"].between(50, 200)]
    rep.add(sec, "scores in 50–200", bad.empty, f"{len(bad)} outside" +
            (f": {bad[['game_id', 'home_team', 'away_team', 'home_pts', 'away_pts']].head(3).to_dict('records')}"
             if len(bad) else ""))
    ties = (games["home_pts"] == games["away_pts"]).sum()
    rep.add(sec, "no ties", ties == 0, f"{ties} ties")
    same = (games["home_team"] == games["away_team"]).sum()
    rep.add(sec, "home ≠ away", same == 0, f"{same} rows")
    teams = set(games["home_team"]) | set(games["away_team"])
    rep.add(sec, "teams ⊆ canonical 30", teams <= NBA_TEAMS, f"unknown: {sorted(teams - NBA_TEAMS)}")
    if expect_per_team:
        long = pd.concat([games[["season", "home_team"]].rename(columns={"home_team": "t"}),
                          games[["season", "away_team"]].rename(columns={"away_team": "t"})])
        counts = long.groupby(["season", "t"]).size()
        off = counts[counts != expect_per_team]
        detail = ", ".join(f"{s} {t}={n}" for (s, t), n in off.items())
        per_season = games.groupby("season").size().to_dict()
        rep.add(sec, f"{expect_per_team} games per team per season", off.empty,
                f"per season {per_season}; off: {detail or 'none'}", warn=True)
    if "is_neutral" in games:
        ns = games[games["is_neutral"].astype(bool)]
        rep.add(sec, "neutral-site games listed", ns.empty,
                f"{len(ns)} neutral-site games (home court applied by the model): " +
                ", ".join(f"{r.date_local if 'date_local' in ns else r.date} {r.away_team}@{r.home_team} ({r.comp_type})"
                          for r in ns.head(8).itertuples()), warn=True)


def check_dates(rep: Report, ga: pd.DataFrame, now: pd.Timestamp):
    sec = "Dates & time zones"
    tips = pd.to_datetime(ga["tip_utc"], utc=True)
    et_date = tips.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    local = pd.to_datetime(ga["date_local"])
    mism = (et_date != local)
    rep.add(sec, "date_local = ET date of tip", mism.sum() == 0,
            f"{mism.sum()} mismatches (postponed/rescheduled listings)" +
            (f": {ga.loc[mism, ['game_id', 'date_local', 'tip_utc']].head(5).to_dict('records')}" if mism.any() else ""),
            warn=True)
    utc_date = tips.dt.normalize().dt.tz_localize(None)
    legacy = pd.to_datetime(ga["date"])
    rep.add(sec, "legacy `date` = UTC date of tip", (legacy == utc_date).all(),
            f"{(legacy != utc_date).sum()} mismatches")
    shifted = (legacy != local) & ga["season_type"].eq("regular") & ga["completed"]
    rep.add(sec, "legacy UTC `date` differs from local game day", False,
            f"{shifted.sum()} of {int((ga['season_type'].eq('regular') & ga['completed']).sum())} completed RS games "
            f"({shifted.mean() / max(ga['season_type'].eq('regular').mean(), 1e-9):.0%}) sit on the next UTC day; "
            "rest days / B2B in the production model are computed on this date", warn=True)
    fut = ga[ga["completed"] & (tips > now)]
    rep.add(sec, "no future games marked final", fut.empty, f"{len(fut)} rows")
    stale = ga[~ga["completed"] & (tips < now - pd.Timedelta(days=2))]
    by_status = stale["status"].value_counts().to_dict()
    rep.add(sec, f"past unfinished games are postponed/cancelled {by_status}",
            set(by_status) <= {"STATUS_POSTPONED", "STATUS_CANCELED"},
            f"{len(stale)} rows; these IDs never reached final (the rescheduled games carry new IDs)"
            , warn=True)
    rep.add(sec, "no past games left unfinished", stale.empty,
            f"{len(stale)} rows (postponed/cancelled?)" +
            (f": {stale[['game_id', 'date_local', 'home_team', 'away_team', 'status']].head(5).to_dict('records')}"
             if len(stale) else ""), warn=True)


def check_walk_order(rep: Report, games_raw: pd.DataFrame):
    sec = "Rating-walk order (lookahead)"

    def violations(order):
        g = games_raw.sort_values(order).reset_index(drop=True)
        long = pd.concat([g.assign(t=g["home_team"]), g.assign(t=g["away_team"])])
        long["pos"] = long.index
        long = long.sort_values("pos")
        long["tip"] = pd.to_datetime(long["tip_utc"], utc=True)
        return long, int((long.groupby("t")["tip"].diff() <= pd.Timedelta(0)).sum())

    long, legacy = violations(["date", "game_id"])
    _, current = violations(walk_order(games_raw))
    same = long.duplicated(["date", "t"]).sum()
    rep.add(sec, "team walked in tip order (build_ratings order)", current == 0,
            f"{same} team-dates with two games on one UTC date; {current} out of tip order now "
            f"(the April (date, game_id) order had {legacy}: DAL@MEM 2026-03-12)")
    g = games_raw.sort_values(["date", "game_id"]).reset_index(drop=True)
    rest_local = _rest_days(g, "date_local")
    rest_utc = _rest_days(g, "date")
    diff = (rest_local != rest_utc)
    b2b_l, b2b_u = (rest_local == 0).sum(), (rest_utc == 0).sum()
    rep.add(sec, "rest days: UTC date vs local date", diff.sum() == 0,
            f"{diff.sum()} of {len(diff)} team-games get different rest days; "
            f"B2B team-games: {b2b_u} on UTC dates vs {b2b_l} on local dates", warn=True)


def _rest_days(g: pd.DataFrame, col: str) -> pd.Series:
    long = pd.concat([g[[col, "game_id"]].assign(t=g["home_team"], side="h"),
                      g[[col, "game_id"]].assign(t=g["away_team"], side="a")])
    long[col] = pd.to_datetime(long[col])
    long = long.sort_values([col, "game_id"])
    prev = long.groupby("t")[col].shift(1)
    rest = ((long[col] - prev).dt.days - 1).clip(0, 7).fillna(7).astype(int)
    return rest.set_axis(long["game_id"].astype(str) + long["side"]).sort_index()


# ── Players ─────────────────────────────────────────────────────────────────

def check_players(rep: Report, pg: pd.DataFrame, ga: pd.DataFrame):
    sec = "Player box scores"
    done = ga[ga["completed"] & ga["season"].isin(RS_SEASONS)]
    have = pg["game_id"].nunique()
    rep.add(sec, "every completed game has a box score", have == len(done),
            f"{have} of {len(done)} games", warn=True)
    played = pg[~pg["did_not_play"]]
    tm = played.groupby(["game_id", "team"]).agg(mins=("min", "sum"), pts=("pts", "sum"),
                                                 periods=("periods", "max")).reset_index()
    tm["expected"] = 240 + 25 * (tm["periods"].clip(lower=4) - 4)
    tm["dev"] = tm["mins"] - tm["expected"]
    out = tm[tm["dev"].abs() > 5]
    rep.add(sec, "minutes per team-game ≈ 240 (+25/OT), ±5", out.empty,
            f"{len(out)} of {len(tm)} team-games off by >5 (ESPN minutes are rounded); "
            f"median dev {tm['dev'].median():+.0f}" +
            (f"; worst: {out.reindex(out['dev'].abs().sort_values(ascending=False).index)[['game_id', 'team', 'mins', 'expected']].head(5).to_dict('records')}"
             if len(out) else ""), warn=True)
    long = pd.concat([ga.assign(team=ga["home_team"], score=ga["home_pts"]),
                      ga.assign(team=ga["away_team"], score=ga["away_pts"])])[["game_id", "team", "score"]]
    chk = tm.merge(long, on=["game_id", "team"], how="left")
    bad = chk[chk["pts"] != chk["score"]]
    rep.add(sec, "player points sum to team score", bad.empty,
            f"{len(bad)} of {len(chk)} team-games differ" +
            (f": {bad[['game_id', 'team', 'pts', 'score']].head(5).to_dict('records')}" if len(bad) else ""),
            warn=len(bad) < 10)
    teams = set(pg["team"])
    rep.add(sec, "teams ⊆ canonical 30", teams <= NBA_TEAMS, f"unknown: {sorted(teams - NBA_TEAMS)}")
    in_game = pg.merge(ga[["game_id", "home_team", "away_team"]], on="game_id", how="left")
    wrong = in_game[(in_game["team"] != in_game["home_team"]) & (in_game["team"] != in_game["away_team"])]
    rep.add(sec, "player's team is one of the two teams in the game", wrong.empty, f"{len(wrong)} rows")
    two = pg.groupby(["athlete_id", "date_local"])["team"].nunique()
    rep.add(sec, "no player on two teams on one date", (two <= 1).all(), f"{(two > 1).sum()} player-dates")
    dup = pg.duplicated(["game_id", "athlete_id"]).sum()
    rep.add(sec, "one row per player per game", dup == 0, f"{dup} duplicates")
    # roster history: team changes inside a season should be rare (trades / signings)
    s = pg.sort_values(["season", "date_local"])
    moves = s.groupby(["season", "athlete_id"])["team"].apply(lambda x: (x != x.shift()).sum() - 1)
    per_season = moves[moves > 0].groupby(level=0).size().to_dict()
    back_forth = moves[moves >= 3]
    rep.add(sec, "mid-season team changes plausible (≤2 per player-season)", back_forth.empty,
            f"players who changed team mid-season: {per_season}; ≥3 changes: {len(back_forth)}"
            + (f" {back_forth.head(5).to_dict()}" if len(back_forth) else ""), warn=True)


def check_rosters(rep: Report, rosters: pd.DataFrame | None, pg: pd.DataFrame | None,
                  snaps: pd.DataFrame | None):
    sec = "Rosters 2026-27"
    if rosters is None:
        rep.add(sec, "rosters_2026_27.csv present", False, "missing", warn=True)
        return
    rep.add(sec, "30 teams", rosters["team"].nunique() == 30, f"{rosters['team'].nunique()} teams")
    rep.add(sec, "teams ⊆ canonical 30", set(rosters["team"]) <= NBA_TEAMS,
            f"unknown: {sorted(set(rosters['team']) - NBA_TEAMS)}")
    dup = rosters["athlete_id"].duplicated().sum()
    rep.add(sec, "player on one roster only", dup == 0, f"{dup} duplicates")
    sizes = rosters.groupby("team").size()
    rep.add(sec, "roster sizes 13–24 (camp: 21 standard + 3 two-way)", sizes.between(13, 24).all(),
            f"min {sizes.min()} ({sizes.idxmin()}), max {sizes.max()} ({sizes.idxmax()})", warn=True)
    if snaps is not None and len(snaps):
        last = snaps[snaps["fetched_at"] == snaps["fetched_at"].max()]
        m = last.merge(rosters[["athlete_id", "team"]].astype({"athlete_id": str}),
                       left_on=last["athlete_id"].astype(str), right_on="athlete_id",
                       how="left", suffixes=("_inj", "_roster"))
        mism = m[m["team_roster"].notna() & (m["team_inj"] != m["team_roster"])]
        notfound = m["team_roster"].isna().sum()
        rep.add(sec, "injury snapshot team = roster team", mism.empty,
                f"{len(mism)} mismatches, {notfound} injured players not on a roster", warn=True)


# ── Manifest ────────────────────────────────────────────────────────────────

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(rep: Report) -> dict:
    cur = {}
    for name in MANIFEST_FILES:
        p = DATA / name
        if p.exists():
            df = _load(name)
            cur[name] = {"rows": int(len(df)), "sha256": sha256(p)}
    prev = json.loads(MANIFEST.read_text())["files"] if MANIFEST.exists() else {}
    lines = []
    for name, v in cur.items():
        pv = prev.get(name)
        if pv is None:
            lines.append(f"| `{name}` | new | {v['rows']:,} | — |")
        else:
            changed = "changed" if pv["sha256"] != v["sha256"] else "same"
            lines.append(f"| `{name}` | {changed} | {v['rows']:,} | {v['rows'] - pv['rows']:+,} |")
    rep.details.append("## Row counts and checksums vs previous run\n\n"
                       "| file | vs previous | rows | Δ rows |\n|---|---|---|---|\n" + "\n".join(lines))
    return cur


def write_report(rep: Report, now: datetime):
    REPORTS.mkdir(exist_ok=True)
    counts = pd.Series([r[2] for r in rep.rows]).value_counts().to_dict()
    lines = [f"# Data quality report", "",
             f"Generated {now.isoformat(timespec='seconds')} by `src/validate_data.py`. "
             f"**{counts.get('PASS', 0)} pass · {counts.get('WARN', 0)} warn · {counts.get('FAIL', 0)} fail**",
             "", "| Section | Check | Status | Detail |", "|---|---|---|---|"]
    for sec, chk, st, det in rep.rows:
        det = det.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {sec} | {chk} | {'✅' if st == 'PASS' else '⚠️' if st == 'WARN' else '❌'} {st} | {det} |")
    lines += [""] + rep.details
    (REPORTS / "data_quality.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    now = datetime.now(timezone.utc)
    rep = Report()
    ga = _load("games_all.csv")
    gr = _load("games_raw.csv")
    po = _load("playoff_games_raw.csv")
    pi = _load("playin_games_raw.csv")
    check_games(rep, gr, "regular season (games_raw.csv)", 82)
    check_games(rep, po, "playoffs", None)
    check_games(rep, pi, "play-in", None)
    sched = _load("schedule_2026_27.csv")
    if sched is not None:
        long = pd.concat([sched["home_team"], sched["away_team"]]).value_counts()
        rep.add("Schedule 2026-27", "82 games per team", (long == 82).all() and len(long) == 30,
                f"{len(sched)} games; per-team min {long.min()} max {long.max()}", warn=True)
        rep.add("Schedule 2026-27", "no game marked final", not sched["status"].eq("STATUS_FINAL").any(), "")
    check_dates(rep, ga, pd.Timestamp(now))
    check_walk_order(rep, gr)
    pg = _load("player_games.parquet")
    if pg is not None:
        check_players(rep, pg, ga)
    check_rosters(rep, _load("rosters_2026_27.csv"), pg, _load("injury_snapshots.csv"))
    cur = manifest(rep)
    write_report(rep, now)
    MANIFEST.write_text(json.dumps({"generated": now.isoformat(timespec="seconds"), "files": cur}, indent=1))
    for sec, chk, st, det in rep.rows:
        if st != "PASS":
            print(f"{st:4}  {sec}: {chk} — {det[:160]}")
    print(f"→ reports/data_quality.md ({'FAIL' if rep.failed else 'ok'})")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())

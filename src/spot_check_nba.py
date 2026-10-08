"""
src/spot_check_nba.py — Cross-check ESPN box scores against stats.nba.com
==========================================================================
Samples N random completed games from data/player_games.parquet and compares
final scores and per-player minutes / points / rebounds / assists with the
official NBA stats API (the endpoints nba_api wraps: leaguegamelog +
boxscoretraditionalv3). nba_api's own HTTP client times out against
stats.nba.com from here, so requests go through curl_cffi with a browser TLS
fingerprint. Used for validation only — nothing is built on this source.

Output: reports/spot_check_nba_api.md

Usage: python3 src/spot_check_nba.py [--n 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests as cr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rosters import norm_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATS = "https://stats.nba.com/stats"
HEADERS = {"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
           "Accept": "application/json, text/plain, */*",
           "x-nba-stats-origin": "stats", "x-nba-stats-token": "true"}
NBA_TO_ESPN = {"NYK": "NY", "GSW": "GS", "SAS": "SA", "NOP": "NO", "UTA": "UTAH", "WAS": "WSH"}
SEASON_TYPES = {"regular": "Regular Season", "playoff": "Playoffs", "playin": "PlayIn"}


def _get(endpoint: str, params: dict) -> dict:
    for attempt in range(4):
        try:
            r = cr.get(f"{STATS}/{endpoint}", params=params, headers=HEADERS,
                       impersonate="chrome", timeout=40)
            if r.status_code == 200:
                time.sleep(1.5)
                return r.json()
        except Exception:
            pass
        time.sleep(3 * 2 ** attempt)
    raise RuntimeError(f"stats.nba.com {endpoint} {params} failed")


def game_index(season: str, season_type: str) -> pd.DataFrame:
    js = _get("leaguegamelog", {"Counter": 0, "Direction": "DESC", "LeagueID": "00",
                                "PlayerOrTeam": "T", "Season": season,
                                "SeasonType": SEASON_TYPES[season_type], "Sorter": "DATE"})
    rs = js["resultSets"][0]
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    df["team"] = df["TEAM_ABBREVIATION"].replace(NBA_TO_ESPN)
    home = df[df["MATCHUP"].str.contains(" vs. ")]
    away = df[df["MATCHUP"].str.contains(" @ ")]
    g = home.merge(away[["GAME_ID", "team", "PTS"]], on="GAME_ID", suffixes=("_h", "_a"))
    return pd.DataFrame({"nba_game_id": g["GAME_ID"], "date_local": pd.to_datetime(g["GAME_DATE"]),
                         "home_team": g["team_h"], "away_team": g["team_a"],
                         "home_pts": g["PTS_h"], "away_pts": g["PTS_a"]})


def nba_box(nba_game_id: str) -> pd.DataFrame:
    js = _get("boxscoretraditionalv3", {"GameID": nba_game_id, "StartPeriod": 0, "EndPeriod": 0,
                                        "StartRange": 0, "EndRange": 0, "RangeType": 0})
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = js["boxScoreTraditional"][side]
        for p in t["players"]:
            s = p["statistics"]
            mins = s.get("minutes") or ""
            m = 0.0
            if ":" in mins:
                mm, ss = mins.split(":")
                m = int(mm) + int(ss) / 60
            rows.append({"team": NBA_TO_ESPN.get(t["teamTricode"], t["teamTricode"]),
                         "name": norm_name(f"{p['firstName']} {p['familyName']}"),
                         "min_nba": m, "pts_nba": s["points"],
                         "reb_nba": s["reboundsTotal"], "ast_nba": s["assists"]})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    pg = pd.read_parquet(DATA / "player_games.parquet")
    ga = pd.read_csv(DATA / "games_all.csv", parse_dates=["date_local"])
    games = ga[ga["game_id"].isin(pg["game_id"].unique())]
    sample = games.sample(args.n, random_state=args.seed)

    idx = {}
    for (season, stype) in sorted(set(zip(sample["season"], sample["season_type"]))):
        idx[(season, stype)] = game_index(season, stype)

    results, player_rows = [], []
    for g in sample.itertuples():
        ix = idx[(g.season, g.season_type)]
        m = ix[(ix["date_local"] == g.date_local) & (ix["home_team"] == g.home_team) &
               (ix["away_team"] == g.away_team)]
        if m.empty:
            results.append({"game_id": g.game_id, "matched": False})
            continue
        m = m.iloc[0]
        score_ok = (m["home_pts"] == g.home_pts) and (m["away_pts"] == g.away_pts)
        nb = nba_box(m["nba_game_id"])
        eb = pg[(pg["game_id"] == g.game_id) & ~pg["did_not_play"]].copy()
        eb["name"] = eb["player_name"].map(norm_name)
        j = eb.merge(nb, on=["team", "name"], how="outer", indicator=True)
        both = j[j["_merge"] == "both"]
        nb_played = nb[nb["min_nba"] > 0]
        player_rows.append(both.assign(game_id=g.game_id))
        results.append({
            "game_id": g.game_id, "nba_game_id": m["nba_game_id"], "date": g.date_local.date(),
            "matchup": f"{g.away_team}@{g.home_team}", "season_type": g.season_type, "matched": True,
            "score_ok": score_ok, "espn_players": len(eb), "nba_players": len(nb_played),
            "name_unmatched": int((j["_merge"] == "left_only").sum()
                                  + ((j["_merge"] == "right_only") & (j["min_nba"] > 0)).sum()),
            "pts_exact": float((both["pts"] == both["pts_nba"]).mean()),
            "min_within_1": float(((both["min"] - both["min_nba"]).abs() <= 1).mean()),
            "reb_exact": float((both["reb"] == both["reb_nba"]).mean()),
            "ast_exact": float((both["ast"] == both["ast_nba"]).mean()),
        })
    res = pd.DataFrame(results)
    pr = pd.concat(player_rows) if player_rows else pd.DataFrame()
    ok = res[res["matched"]]
    lines = ["# ESPN vs stats.nba.com spot check", "",
             f"{args.n} random completed games (seed {args.seed}) from `data/player_games.parquet`, "
             "compared with stats.nba.com (`leaguegamelog`, `boxscoretraditionalv3`).", "",
             f"- Games matched to an NBA game ID: **{len(ok)}/{len(res)}**",
             f"- Final score identical: **{int(ok['score_ok'].sum())}/{len(ok)}**",
             f"- Player-lines compared: **{len(pr)}**; names unmatched (played): {int(ok['name_unmatched'].sum())}",
             f"- Points identical: **{(pr['pts'] == pr['pts_nba']).mean():.1%}**",
             f"- Minutes within 1: **{((pr['min'] - pr['min_nba']).abs() <= 1).mean():.1%}** "
             "(ESPN rounds minutes to whole numbers)",
             f"- Rebounds identical: **{(pr['reb'] == pr['reb_nba']).mean():.1%}**, "
             f"assists identical: **{(pr['ast'] == pr['ast_nba']).mean():.1%}**", "",
             "## Per game", "", ok.drop(columns=["matched"]).to_markdown(index=False, floatfmt=".3f")]
    miss = res[~res["matched"]]
    if len(miss):
        lines += ["", f"Unmatched ESPN games: {miss['game_id'].tolist()}"]
    bad = pr[(pr["pts"] != pr["pts_nba"]) | ((pr["min"] - pr["min_nba"]).abs() > 1)]
    if len(bad):
        lines += ["", "## Player lines that differ", "",
                  bad[["game_id", "team", "player_name", "min", "min_nba", "pts", "pts_nba"]]
                  .head(40).to_markdown(index=False, floatfmt=".1f")]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "spot_check_nba_api.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

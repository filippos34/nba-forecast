"""2.3 Preseason roster prior: at each season start, rating += γ × 28 × ΔS where
S = 5 × minutes-weighted mean (player value − replacement) of the roster, weights =
each player's previous-season minutes (any team), values = player ratings as of the end
of the previous season (availability.player_state, no lookahead).
  old roster: players with a box row for the team in its last previous-season RS game
  new roster: players with a box row (played or DNP) in its first game of the season —
              the active+inactive list, known before tip.
γ fit on training seasons. Reported on each team's first 20 games of 2025-26 as well."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from common import br, ev, games, rs_frame, rs_probs, train_brier
import availability as av

BASE = br.BASELINE_V2
GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def roster_shift(train_seasons, pg):
    ap = av.AvailParams()
    value = av.player_values(pg, None, 0.0, ap.value, train_seasons)
    r_repl = av.replacement_level(pg, value, train_seasons)
    state = av.player_state(pg, value, r_repl, ap).sort_values("tip")
    rs = pg[pg["season_type"] == "regular"]
    mins = rs[rs["played"]].groupby(["season", "athlete_id"])["min"].sum()
    seasons = sorted(rs["season"].unique())
    shift = {}
    for prev, cur in zip(seasons, seasons[1:]):
        p_rs, c_rs = rs[rs["season"] == prev], rs[rs["season"] == cur]
        cutoff = c_rs["tip"].min()
        rating = state[state["tip"] < cutoff].groupby("athlete_id")["rating"].last()
        w = mins.loc[prev] if prev in mins.index.get_level_values(0) else pd.Series(dtype=float)

        def strength(ids):
            ids = list(ids)
            ww = w.reindex(ids).fillna(0.0).to_numpy()
            rr = rating.reindex(ids).fillna(r_repl).to_numpy() - r_repl
            return 5 * float(ww @ rr / ww.sum()) if ww.sum() > 0 else 0.0

        for team in c_rs["team"].unique():
            last_gid = p_rs[p_rs["team"] == team].sort_values("tip")["game_id"].iloc[-1]
            first_gid = c_rs[c_rs["team"] == team].sort_values("tip")["game_id"].iloc[0]
            old = p_rs[(p_rs["game_id"] == last_gid) & (p_rs["team"] == team)]["athlete_id"]
            new = c_rs[(c_rs["game_id"] == first_gid) & (c_rs["team"] == team)]["athlete_id"]
            shift[(cur, team)] = strength(new) - strength(old)
    return shift


def probs(shift_pts, gamma):
    sh = {k: gamma * 28 * v for k, v in shift_pts.items()}
    return br.run_walk(games(), BASE, season_shift=sh).probs[(games()["season_type"] == "regular").to_numpy()]


def main():
    pg = av.load_player_games()
    out = {}
    for fold in ("test", "fold2"):
        sp = roster_shift(ev.TRAIN[fold], pg)
        res = [(train_brier(probs(sp, gm), ev.TRAIN[fold]), gm) for gm in GRID]
        out[fold] = (min(res)[1], sp)
        print(f"{fold}: γ = {min(res)[1]}  train Brier by γ: " + ", ".join(f"{g}:{b:.5f}" for b, g in res))
    rs = rs_frame()
    gamma, sp = out["test"]
    g2, sp2 = out["fold2"]
    c = ev.compare(rs, rs_probs(BASE), probs(sp, gamma), rs_probs(BASE), probs(sp2, g2))
    print(ev.format_compare(c))
    # first 20 games per team, 2025-26
    t = rs[rs["season"] == "2025-26"].copy()
    t["hn"] = t.groupby("home_team").cumcount(); t["an"] = t.groupby("away_team").cumcount()
    first = np.zeros(len(rs), dtype=bool)
    first[t.index.to_numpy()] = ((t["hn"] < 20) | (t["an"] < 20)).to_numpy()  # home team's first 20 home or away team's first 20 away games
    y = rs["actual_home_win"].to_numpy()
    b = ev.paired_bootstrap(rs_probs(BASE)[first], probs(sp, gamma)[first], y[first])
    print(f"first ~20 games per team 2025-26 (n={first.sum()}): Δ {b['delta']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
    top = sorted(sp.items(), key=lambda kv: -abs(kv[1]) if kv[0][0] == "2025-26" else 0)[:6]
    print("largest 2025-26 roster shifts (pts):", [(k[1], round(v, 2)) for k, v in top])
    ev.log_experiment("06", "2.3 preseason roster prior", "experiment", "baseline_v2",
                      f"γ={gamma} × 28 × Δ roster strength", {"gamma": gamma}, c,
                      notes=f"fold2 γ={g2}; first-20-games Δ {b['delta']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()

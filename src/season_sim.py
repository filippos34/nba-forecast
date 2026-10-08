"""
src/season_sim.py — 2026-27 season Monte Carlo (title odds for the website)
===========================================================================
Team strength = exactly the daily forecasts' v3 inputs: production Elo (data/elo_ratings_current.csv)
+ availability_model.live_team_adjustment (roster membership; no injury report yet → everyone
"Not listed"), + a per-simulation offset ~ N(0, σ) on the Elo part for preseason uncertainty.
Per-game probability = predict._base_prob (home court split decided on Elo before availability). σ = 65 Elo: the pooled SD of
(end-of-season Elo − preseason Elo) over 2022-26 (it grows to ~100 in the latest, full-K seasons —
a modelling choice, documented on the site).

Regular season: every scheduled game (data/schedule_2026_27.csv, 1,200 known; the 30 NBA Cup-dependent
games are not yet scheduled) with split home court, 0 at neutral sites and the B2B penalty from the
schedule's local-date flags — the production walk's arithmetic, ratings held fixed within a season.
Postseason: conference seeding by wins (random tiebreak), play-in (7v8, 9v10, loser 7/8 vs winner 9/10),
best-of-7 series, 2-2-1-1-1, playoff home court (data/playoff_params.json HC), higher seed at home.

    python3 src/season_sim.py [--n 50000]  → data/title_odds_2026_27.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_ratings as br  # noqa: E402

DATA = ROOT / "data"
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NY", "ORL", "PHI", "TOR", "WSH"}
SIGMA_TEAM = 65.0


def p_home(diff):
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def team_inputs(as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per team: `elo` (data/elo_ratings_current.csv, what predict.py reads) and `avail` (the same
    availability_model.live_team_adjustment the daily forecasts use; no injury report yet)."""
    import availability_model as am
    elo = pd.read_csv(DATA / "elo_ratings_current.csv").set_index("team")["elo"]
    sched = pd.read_csv(DATA / "schedule_2026_27.csv")
    tip = pd.Timestamp(sched["tip_utc"].min()) if as_of is None else pd.Timestamp(as_of)
    avail = pd.Series({t: am.live_team_adjustment(t, tip, None)["elo"] for t in elo.index})
    return pd.DataFrame({"elo": elo, "avail": avail})


def team_strength(as_of: pd.Timestamp | None = None) -> pd.Series:
    """v3 strength = Elo + availability (Elo units)."""
    t = team_inputs(as_of)
    return t["elo"] + t["avail"]


def game_prob(h_elo, a_elo, h_av, a_av, neutral, rest, p=None):
    """Vectorised predict._base_prob: split home court decided on Elo BEFORE the availability
    adjustment (as in the validated walk), 0 at neutral sites; rest = home − away rest Elo."""
    p = br.PRODUCTION if p is None else p
    hc = np.where(h_elo >= a_elo, p.HC_fav, p.HC_dog) * ~np.asarray(neutral, bool)
    return p_home(h_elo + h_av + hc + rest - (a_elo + a_av))


def simulate(n: int = 50_000, seed: int = 0, inputs: pd.DataFrame | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = br.PRODUCTION
    hc_po = br.load_playoff_params().HC
    inputs = team_inputs() if inputs is None else inputs
    teams = sorted(inputs.index)
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    E = inputs["elo"].reindex(teams).to_numpy(float)[None, :] + rng.normal(0, SIGMA_TEAM, (n, T))  # Elo + uncertainty
    AV = inputs["avail"].reindex(teams).to_numpy(float)
    R = E + AV[None, :]                                                        # v3 strength, n × T
    base = (inputs["elo"] + inputs["avail"]).reindex(teams).to_numpy(float)

    s = pd.read_csv(DATA / "schedule_2026_27.csv")
    h = s["home_team"].map(idx).to_numpy()
    a = s["away_team"].map(idx).to_numpy()
    neutral = s["is_neutral"].astype(bool).to_numpy()
    rest = (np.where(s["home_b2b"], p.rest_b2b, 0.0) - np.where(s["away_b2b"], p.rest_b2b, 0.0))
    wins = np.zeros((n, T), dtype=np.int32)
    for k in range(0, len(s), 100):                                          # chunk games to bound memory
        hh, aa = h[k:k + 100], a[k:k + 100]
        pr = game_prob(E[:, hh], E[:, aa], AV[hh], AV[aa], neutral[k:k + 100], rest[k:k + 100], p)
        win = rng.random(pr.shape) < pr
        np.add.at(wins, (slice(None), hh), win)
        np.add.at(wins, (slice(None), aa), ~win)

    def series(x, y, best_of=7):
        """x has home court (higher seed). x, y: arrays of team indices per sim → winner indices."""
        need = best_of // 2 + 1
        home_games = [1, 1, 0, 0, 1, 0, 1]                                   # 2-2-1-1-1 from x's view
        rx, ry = R[np.arange(n), x], R[np.arange(n), y]
        wx = np.zeros(n, int)
        wy = np.zeros(n, int)
        for g in range(best_of):
            live = (wx < need) & (wy < need)
            d = np.where(home_games[g], rx + hc_po - ry, rx - (ry + hc_po))
            xw = rng.random(n) < p_home(d)
            wx += live & xw
            wy += live & ~xw
        return np.where(wx >= need, x, y)

    def single(x, y):
        rx, ry = R[np.arange(n), x], R[np.arange(n), y]
        return np.where(rng.random(n) < p_home(rx + hc_po - ry), x, y)

    conf_win, finals = {}, {}
    for conf in ("East", "West"):
        members = np.array([idx[t] for t in teams if (t in EAST) == (conf == "East")])
        w = wins[:, members] + rng.random((n, len(members))) * 0.1               # random tiebreak
        order = members[np.argsort(-w, axis=1)]                                    # seeds 1..15 per sim
        s7, s8, s9, s10 = order[:, 6], order[:, 7], order[:, 8], order[:, 9]
        w78 = single(s7, s8)
        l78 = np.where(w78 == s7, s8, s7)
        w910 = single(s9, s10)
        seed8 = single(l78, w910)
        seeds = np.column_stack([order[:, :6], w78, seed8])                        # 1..8
        r1 = [series(seeds[:, i], seeds[:, 7 - i]) for i in range(4)]            # 1v8, 2v7, 3v6, 4v5

        def better(x, y):
            # higher seed = position in `seeds` row
            pos = np.argmax(seeds[:, :, None] == np.stack([x, y], axis=1)[:, None, :], axis=1)
            return np.where(pos[:, 0] <= pos[:, 1], x, y), np.where(pos[:, 0] <= pos[:, 1], y, x)
        a_, b_ = better(r1[0], r1[3])
        sf1 = series(a_, b_)
        a_, b_ = better(r1[1], r1[2])
        sf2 = series(a_, b_)
        a_, b_ = better(sf1, sf2)
        conf_champ = series(a_, b_)
        conf_win[conf] = conf_champ
        finals[conf] = wins[np.arange(n), conf_champ]
    e, w_ = conf_win["East"], conf_win["West"]
    e_home = finals["East"] + rng.random(n) * 0.1 >= finals["West"]
    champ = np.where(e_home, series(e, w_), series(w_, e))

    out = pd.DataFrame({"team": teams, "strength": base.round(1),
                        "elo": inputs["elo"].reindex(teams).round(1).to_numpy(),
                        "avail": inputs["avail"].reindex(teams).round(1).to_numpy(),
                        "mean_wins": wins.mean(axis=0).round(1),
                        "p_east_or_west": [np.mean((conf_win["East"] == i) | (conf_win["West"] == i)) for i in range(T)],
                        "p_champion": [np.mean(champ == i) for i in range(T)]})
    out["conference"] = out["team"].map(lambda t: "East" if t in EAST else "West")
    return out.sort_values("p_champion", ascending=False).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    a = ap.parse_args(argv)
    t = simulate(a.n)
    t.to_csv(DATA / "title_odds_2026_27.csv", index=False)
    print(t.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

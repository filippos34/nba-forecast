"""2.2(b) roster-aware membership (for live/backtest consistency). Same frozen layer as 03c
(blend, half-life 160, shrink 500, no baseline, P(plays|status) from 2021-22→2023-24);
only candidate membership changes: team of the player's next box row in the same season
(+ players on the team's report). λ refit on 2021-22→2023-24 only."""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from common import ROOT, br, ev, games, rs_frame, rs_probs, train_brier
import availability as av
import injury_status as ist

FIT = ["2021-22", "2022-23", "2023-24"]
BASE = br.BASELINE_V2
AP = av.AvailParams(value="blend", halflife_games=160.0, shrink_minutes=500.0, baseline_games=0)
LAMS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]


RECENCY = True          # fixed a priori: 14 days / first 5 team games


def main():
    pg = av.load_player_games()
    status_rows = pd.read_parquet(ist.OUT)
    value = av.player_values(pg, None, 0.0, AP.value, FIT)
    r_repl = av.replacement_level(pg, value, FIT)
    state = av.player_state(pg, value, r_repl, AP)
    listed = status_rows[status_rows["status"] != "Not listed"]
    cand = av.attach_state(av.roster_candidates(pg, listed), state, pg, r_repl)
    st = ist.status_for(cand, status_rows)
    listed_now = st.notna().to_numpy() & (st != "Not listed").to_numpy()
    mask = av.recency_mask(cand, pg, listed_now) if RECENCY else np.ones(len(cand), bool)
    season_of = games().set_index("game_id")["season"]
    fit = cand["game_id"].map(season_of).isin(FIT).to_numpy()
    table = ist.p_plays_table(cand[mask], st[mask], fit[mask])
    p = st.map(table["p_plays"]).to_numpy(float)
    p = np.where(np.isnan(p), table.loc["Not listed", "p_plays"], p)
    pts = av.team_adjustment(cand, p, r_repl, mask)
    g = games()
    b0 = train_brier(rs_probs(BASE), FIT)
    fits = [(train_brier(rs_probs(BASE, av.elo_adjustments(g, pts, lam)), FIT), lam) for lam in LAMS]
    lam = min(fits)[1]
    print("λ fit (2021-24):", ", ".join(f"{l}:{b - b0:+.5f}" for b, l in fits))
    print("P(plays | status), roster-aware, 2021-24:\n", table.round(3).to_string())
    rs = rs_frame(); y = rs["actual_home_win"].to_numpy()
    pb, pn = rs_probs(BASE), rs_probs(BASE, av.elo_adjustments(g, pts, lam))
    res = {}
    for name, m in {"2024-25": (rs["season"] == "2024-25").to_numpy(),
                    "2025-26": (rs["season"] == "2025-26").to_numpy(),
                    "2025-26 thru Apr 7": ev.subsets(rs)["test_apr7"],
                    "pooled 2024-26": rs["season"].isin(["2024-25", "2025-26"]).to_numpy()}.items():
        bt = ev.paired_bootstrap(pb[m], pn[m], y[m]); res[name] = bt
        print(f"{name:<20} Brier {ev.metrics(pb[m], y[m])['brier']:.4f} → {ev.metrics(pn[m], y[m])['brier']:.4f}  "
              f"Δ {bt['delta']:+.4f} [{bt['ci_lo']:+.4f}, {bt['ci_hi']:+.4f}]")
    # first 10 games of each team, 2025-26 — where membership matters most
    t = rs[rs["season"] == "2025-26"]
    early = np.zeros(len(rs), bool)
    early[t.index[(t.groupby("home_team").cumcount() < 5) | (t.groupby("away_team").cumcount() < 5)]] = True
    bt = ev.paired_bootstrap(pb[early], pn[early], y[early])
    print(f"2025-26 early-season games (n={early.sum()}): Δ {bt['delta']:+.4f} [{bt['ci_lo']:+.4f}, {bt['ci_hi']:+.4f}]")
    c = ev.compare(rs, pb, pn, pb, pn)
    tag = "03e" if RECENCY else "03d"
    ev.log_experiment(tag, "2.2(b) availability — strict, roster-aware membership"
                      + (" + 14-day recency" if RECENCY else ""), "experiment", "baseline_v2",
                      f"03c layer + next-row roster membership{' + recency' if RECENCY else ''}, λ={lam}",
                      {"avail": asdict(AP), "lambda": lam, "membership": "next box row, same season; +listed"}, c,
                      notes=f"pooled 2024-26 Δ {res['pooled 2024-26']['delta']:+.4f} "
                            f"[{res['pooled 2024-26']['ci_lo']:+.4f}, {res['pooled 2024-26']['ci_hi']:+.4f}]")
    return lam, table, res


if __name__ == "__main__":
    lam, table, res = main()
    if RECENCY:   # production (Gate 2 decision: membership = roster)
        import availability_model as am
        am.write_p_table(table)
        pooled = res["pooled 2024-26"]
        print(f"wrote data/model/p_plays_status.csv; λ={lam}; pooled Δ {pooled['delta']:+.4f} "
              f"[{pooled['ci_lo']:+.4f}, {pooled['ci_hi']:+.4f}]")

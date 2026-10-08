"""
2.2 Player availability layer.
  03a  version (a): who actually played (box score) — CEILING only, never decides acceptance
  03b  version (b): official injury report at T−60, P(plays | status) estimated on the
       training seasons — the result that counts (rule 4)
Configuration chosen on the 2021-22→2024-25 training seasons only (explore_*.py):
player value = blend of Game Score and relative plus-minus, half-life 20 appearances,
shrink 250 min to replacement level, adjustment = full expected-lineup strength
(no baseline), λ fit on the training seasons of each fold.

Usage: python3 03_availability.py [a|b|both]
"""
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

from common import br, ev, games, rs_frame, rs_probs, train_brier, ROOT
import availability as av
import injury_status as ist

AP = av.AvailParams()
LAMS = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
BASE = br.BASELINE_V2


def fit_lambda(pts, seasons):
    g = games()
    res = [(train_brier(rs_probs(BASE, av.elo_adjustments(g, pts, lam)), seasons), lam) for lam in LAMS]
    return min(res)[1], pd.DataFrame(res, columns=["train_brier", "lam"])


def season_mask(cand, seasons):
    season_of = games().set_index("game_id")["season"]
    return cand["game_id"].map(season_of).isin(seasons).to_numpy()


def version_a(fold):
    L = av.build_layer(ev.TRAIN[fold], AP, pg=PG)
    c = L.cand
    played = c["played"].to_numpy()
    pts = av.team_adjustment(c, played.astype(float), L.r_repl, av.active_mask(c, played))
    lam, _ = fit_lambda(pts, ev.TRAIN[fold])
    return pts, lam


def version_b(fold, status_rows):
    L = av.build_layer(ev.TRAIN[fold], AP, pg=PG)
    c = L.cand
    st = ist.status_for(c, status_rows)
    listed = st.notna().to_numpy() & (st != "Not listed").to_numpy()
    mask = av.active_mask(c, listed)
    train = season_mask(c, ev.TRAIN[fold])
    table = ist.p_plays_table(c[mask], st[mask], train[mask])
    p = st.map(table["p_plays"]).to_numpy(dtype=float)
    p_not_listed = table.loc["Not listed", "p_plays"]
    p = np.where(np.isnan(p), p_not_listed, p)          # no report covering the team-game
    pts = av.team_adjustment(c, p, L.r_repl, mask)
    lam, _ = fit_lambda(pts, ev.TRAIN[fold])
    coverage = st.notna()[mask & season_mask(c, [ev.TEST_SEASON[fold]])].mean()
    return pts, lam, table, coverage


def run(which):
    rs = rs_frame()
    g = games()
    out = {}
    if which in ("a", "both"):
        pts, lam = version_a("test")
        pts2, lam2 = version_a("fold2")
        c = ev.compare(rs, rs_probs(BASE), rs_probs(BASE, av.elo_adjustments(g, pts, lam)),
                       rs_probs(BASE), rs_probs(BASE, av.elo_adjustments(g, pts2, lam2)))
        print(f"\n== 03a availability, ACTUAL DNPs (ceiling)  λ={lam} (fold2 λ={lam2})\n{ev.format_compare(c)}")
        ev.log_experiment("03a", "2.2(a) player availability — actual DNPs (ceiling)", "ceiling",
                          "baseline_v2", f"expected-lineup strength, λ={lam}",
                          {"avail": asdict(AP), "lambda": lam}, c, decision="CEILING (not counted)",
                          notes=f"fold2 λ={lam2}")
        out["a"] = c
    if which in ("b", "both"):
        status_rows = pd.read_parquet(ist.OUT)
        pts, lam, table, cov = version_b("test", status_rows)
        pts2, lam2, table2, cov2 = version_b("fold2", status_rows)
        c = ev.compare(rs, rs_probs(BASE), rs_probs(BASE, av.elo_adjustments(g, pts, lam)),
                       rs_probs(BASE), rs_probs(BASE, av.elo_adjustments(g, pts2, lam2)))
        print(f"\n== 03b availability, PREGAME reports at T−60  λ={lam} (fold2 λ={lam2}); "
              f"report coverage on test {cov:.1%}\n{ev.format_compare(c)}")
        print("\nP(plays | status), training seasons 2021-22→2024-25:\n", table.round(3).to_string())
        table.to_csv(ROOT / "reports" / "p_plays_status.csv")
        ev.log_experiment("03b", "2.2(b) player availability — pregame reports T−60", "experiment",
                          "baseline_v2", f"expected-lineup strength with P(plays|status), λ={lam}",
                          {"avail": asdict(AP), "lambda": lam, "p_plays": table["p_plays"].round(4).to_dict()},
                          c, notes=f"fold2 λ={lam2}; report coverage test {cov:.1%} fold2 {cov2:.1%}")
        out["b"] = c
    return out


if __name__ == "__main__":
    PG = av.load_player_games()
    run(sys.argv[1] if len(sys.argv) > 1 else "both")

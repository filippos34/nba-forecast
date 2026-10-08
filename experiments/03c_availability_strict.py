"""
2.2(b) strict re-test (Gate 2 condition): NOTHING in the availability layer may be tuned
on 2024-25 or 2025-26. Everything — hyperparameter choice (value kind, half-life,
shrinkage, baseline), blend scaling, replacement level, P(plays | status), λ — is fit on
2021-22 → 2023-24 only, selected on version (b) (pregame reports). The frozen model is then
evaluated once on 2024-25, on 2025-26, and pooled.
(The shared Elo baseline_v2 params are identical in both arms.)
"""
import itertools
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from common import ROOT, br, ev, games, rs_frame, rs_probs, train_brier
import availability as av
import injury_status as ist

FIT = ["2021-22", "2022-23", "2023-24"]
BASE = br.BASELINE_V2
LAMS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0]
GRID = dict(value=["gmsc", "pm", "blend"], halflife_games=[20.0, 40.0, 80.0, 160.0],
            shrink_minutes=[250.0, 500.0, 1000.0, 2000.0], baseline_games=[0, 82])


def season_mask(cand, seasons):
    season_of = games().set_index("game_id")["season"]
    return cand["game_id"].map(season_of).isin(seasons).to_numpy()


def layer_b(ap, pg, status_rows):
    """Version (b) team adjustments with everything fit on FIT seasons."""
    L = av.build_layer(FIT, ap, pg=pg)
    c = L.cand
    st = ist.status_for(c, status_rows)
    listed = st.notna().to_numpy() & (st != "Not listed").to_numpy()
    mask = av.active_mask(c, listed)
    fit = season_mask(c, FIT)
    table = ist.p_plays_table(c[mask], st[mask], fit[mask])
    p = st.map(table["p_plays"]).to_numpy(dtype=float)
    p = np.where(np.isnan(p), table.loc["Not listed", "p_plays"], p)
    return av.team_adjustment(c, p, L.r_repl, mask), table, L


def best_lambda(pts):
    g = games()
    return min((train_brier(rs_probs(BASE, av.elo_adjustments(g, pts, lam)), FIT), lam) for lam in LAMS)


def main():
    pg = av.load_player_games()
    status_rows = pd.read_parquet(ist.OUT)
    b0 = train_brier(rs_probs(BASE), FIT)
    results = []
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, combo))
        ap = av.AvailParams(**kw, presence_games=82)
        pts, _, _ = layer_b(ap, pg, status_rows)
        b, lam = best_lambda(pts)
        results.append({**kw, "lam": lam, "fit_brier_delta": b - b0})
        print(f"{kw}  λ {lam}  Δ(2021-24) {b - b0:+.5f}", flush=True)
    res = pd.DataFrame(results).sort_values("fit_brier_delta")
    res.to_csv(ROOT / "reports" / "availability_strict_search.csv", index=False)
    best = res.iloc[0]
    ap = av.AvailParams(value=best["value"], halflife_games=float(best["halflife_games"]),
                        shrink_minutes=float(best["shrink_minutes"]),
                        baseline_games=int(best["baseline_games"]), presence_games=82)
    lam = float(best["lam"])
    pts, table, L = layer_b(ap, pg, status_rows)
    print("\nselected on 2021-24:", asdict(ap), "λ", lam)
    print("P(plays | status), 2021-22→2023-24:\n", table.round(3).to_string())

    rs = rs_frame()
    y = rs["actual_home_win"].to_numpy()
    pb = rs_probs(BASE)
    pn = rs_probs(BASE, av.elo_adjustments(games(), pts, lam))
    out = {}
    for name, m in {"2024-25": (rs["season"] == "2024-25").to_numpy(),
                    "2025-26": (rs["season"] == "2025-26").to_numpy(),
                    "2025-26 thru Apr 7": ev.subsets(rs)["test_apr7"],
                    "pooled 2024-26": rs["season"].isin(["2024-25", "2025-26"]).to_numpy()}.items():
        bt = ev.paired_bootstrap(pb[m], pn[m], y[m])
        mb, mn = ev.metrics(pb[m], y[m]), ev.metrics(pn[m], y[m])
        out[name] = {**bt, "brier_base": mb["brier"], "brier_new": mn["brier"],
                     "ll_base": mb["logloss"], "ll_new": mn["logloss"], "n": mn["n"]}
        print(f"{name:<20} n={mn['n']:5d}  Brier {mb['brier']:.4f} → {mn['brier']:.4f}  "
              f"Δ {bt['delta']:+.4f} [{bt['ci_lo']:+.4f}, {bt['ci_hi']:+.4f}]  "
              f"LL {mb['logloss']:.4f} → {mn['logloss']:.4f}")
    # log in the standard format: test = 2025-26, fold2 = 2024-25 (same frozen model)
    c = ev.compare(rs, pb, pn, pb, pn)
    ev.log_experiment("03c", "2.2(b) availability — STRICT: all tuning on 2021-22→2023-24 only", "experiment",
                      "baseline_v2", f"{best['value']}, hl {best['halflife_games']:g}, shrink "
                      f"{best['shrink_minutes']:g}, baseline {int(best['baseline_games'])}, λ={lam}",
                      {"avail": asdict(ap), "lambda": lam, "p_plays": table["p_plays"].round(4).to_dict()}, c,
                      notes=f"pooled 2024-26 Δ {out['pooled 2024-26']['delta']:+.4f} "
                            f"[{out['pooled 2024-26']['ci_lo']:+.4f}, {out['pooled 2024-26']['ci_hi']:+.4f}]")
    table.to_csv(ROOT / "reports" / "p_plays_status_strict.csv")
    (ROOT / "reports" / "availability_strict.json").write_text(
        json.dumps({"avail": asdict(ap), "lambda": lam, "fit_seasons": FIT, "results": out}, indent=1))


if __name__ == "__main__":
    main()

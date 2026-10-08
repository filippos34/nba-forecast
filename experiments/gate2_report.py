"""Gate 2 materials: reports/experiments.md (from experiments.csv), calibration plot of
baseline_v2 vs the 2.2(b) candidate, P(plays | status) tables (overall and by minutes tier)."""
import numpy as np
import pandas as pd

from common import ROOT, br, ev, games, rs_probs
import availability as av
import calibration_plot as cp
import injury_status as ist
import importlib

m03 = importlib.import_module("03_availability")


def experiments_md():
    e = pd.read_csv(ROOT / "reports" / "experiments.csv")
    f = lambda v: f"{v:+.4f}"
    rows = ["| id | experiment | kind | test Brier (base → new) | Δ test [95% CI] | Δ Apr-7 [95% CI] | "
            "Δ 2024-25 [95% CI] | log loss (base → new) | acc (base → new) | auto verdict | decision |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in e.itertuples():
        rows.append(
            f"| {r.id} | {r.name} | {r.kind} | {r.brier_base:.4f} → {r.brier_new:.4f} | "
            f"{f(r.delta)} [{f(r.ci_lo)}, {f(r.ci_hi)}] | {f(r.delta_apr7)} [{f(r.ci_lo_apr7)}, {f(r.ci_hi_apr7)}] | "
            f"{f(r.delta_fold2)} [{f(r.ci_lo_fold2)}, {f(r.ci_hi_fold2)}] | {r.logloss_base:.4f} → {r.logloss_new:.4f} | "
            f"{r.acc_base:.1%} → {r.acc_new:.1%} | {r.auto_verdict} | {r.decision} |")
    (ROOT / "reports" / "experiments.md").write_text(
        "# Phase 2 experiments\n\nTest = 2025-26 regular season (1,231 games); Apr-7 = the first 1,178 "
        "(without the last 53 tanking games); 2024-25 = second fold with parameters refit on "
        "2021-22→2023-24. Δ = new − base Brier (negative = better), paired bootstrap 2,000 resamples. "
        "Rule 4: adopt only if the test CI excludes zero or Δ ≤ −0.0085.\n\n" + "\n".join(rows) + "\n")


def p_plays_tables():
    m03.PG = av.load_player_games()
    fit = ["2021-22", "2022-23", "2023-24"]      # Gate 2 condition: nothing from 2024-25 / 2025-26
    L = av.build_layer(fit, av.AvailParams(), pg=m03.PG)
    c = L.cand
    st = ist.status_for(c, pd.read_parquet(ist.OUT))
    listed = st.notna().to_numpy() & (st != "Not listed").to_numpy()
    mask = av.active_mask(c, listed)
    train = m03.season_mask(c, fit)
    tier = pd.cut(c["proj_min"], [-1, 10, 20, 30, 60], labels=["<10", "10–20", "20–30", "30+"])
    d = pd.DataFrame({"status": st, "tier": tier, "played": c["played"].astype(float)})[mask & train]
    d = d.dropna(subset=["status"])
    t = d.pivot_table(index="status", columns="tier", values="played", aggfunc="mean", observed=True)
    n = d.pivot_table(index="status", columns="tier", values="played", aggfunc="size", observed=True)
    order = [s for s in ist.STATUS_ORDER if s in t.index]
    t, n = t.reindex(order), n.reindex(order)
    t.round(3).to_csv(ROOT / "reports" / "p_plays_status_by_minutes.csv")
    n.to_csv(ROOT / "reports" / "p_plays_status_by_minutes_n.csv")
    return t, n


def calibration():
    """Production model (baseline_v2 walk + availability layer) vs baseline_v2, 2025-26."""
    pr = pd.read_csv(ROOT / "data" / "elo_predictions.csv")
    rs = __import__("common").rs_frame()
    pr = pr.set_index("game_id").loc[rs["game_id"]]
    return cp.plot({"baseline_v2": pr["prob_baseline_v2"].to_numpy(),
                    "production v3 (+ availability)": pr["home_win_prob"].to_numpy()})


if __name__ == "__main__":
    experiments_md()
    t, _ = p_plays_tables()
    print(t.round(3).to_string())
    calibration()
    print("wrote reports/experiments.md, calibration_2025_26.png/.csv, p_plays_status_by_minutes*.csv")

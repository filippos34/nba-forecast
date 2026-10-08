"""2.1 Margin model: P(win) = Φ(μ/σ) with μ = slope × Elo diff / 28, fit on training
seasons. Structural upgrade (fair spreads / alt lines); accept if at least equal on Brier."""
from dataclasses import asdict

import numpy as np

from common import br, ev, games, rs_mask, rs_frame
import margin_model as mm

BASE = br.BASELINE_V2
g = games()
walk = br.run_walk(g, BASE)
rs = rs_frame()
diff = walk.diffs[rs_mask()]
p_base = walk.probs[rs_mask()]
margin = (rs["home_pts"] - rs["away_pts"]).to_numpy()


def fit_on(seasons):
    m = rs["season"].isin(seasons).to_numpy()
    return mm.fit(diff[m], margin[m])


M = fit_on(ev.TRAIN["test"])
M2 = fit_on(ev.TRAIN["fold2"])
print(f"fit 2021-22→2024-25: slope {M.slope:.3f}, sigma {M.sigma:.2f} pts "
      f"(1 pt of margin = {28 / M.slope:.1f} Elo); fold2 fit: slope {M2.slope:.3f}, sigma {M2.sigma:.2f}")
c = ev.compare(rs, p_base, M.p_win(diff), p_base, M2.p_win(diff))
print(ev.format_compare(c))
t = rs["season"].eq("2025-26").to_numpy()
res = margin[t] - M.mu(diff[t])
print(f"2025-26 margin residual sd {res.std():.2f} (model σ {M.sigma:.2f}); "
      f"cover calibration: P(|z|<1) = {np.mean(np.abs(res / M.sigma) < 1):.3f} (normal 0.683)")
decision = ("ADOPT" if c["test"]["boot"]["delta"] <= 0 else
            "REJECT for win prob (Δ>0, CI incl. 0); ADOPT for spreads only")
ev.log_experiment("04", "2.1 margin model Φ(μ/σ)", "structural", "baseline_v2",
                  f"slope {M.slope:.3f}, σ {M.sigma:.2f}", {"slope": M.slope, "sigma": M.sigma}, c,
                  decision=decision, notes="structural: accept if Brier at least equal; "
                  f"fold2 slope {M2.slope:.3f} σ {M2.sigma:.2f}")
print("decision (structural: at least equal):", decision)

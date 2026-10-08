"""2.3 finished on the adopted model (baseline_v2 walk + availability layer, Gate 2):
  09a playoff K: play-in/playoff games update RS ratings with K_playoff_rs (grid incl. 0)
  09b preseason roster prior: rating += γ·28·Δ roster strength at season start (grid incl. 0)
Fit on the seasons before each test season; test 2025-26 (+ Apr-7 subset), fold 2024-25."""
from dataclasses import asdict, replace
import importlib

import numpy as np

from common import br, ev, games, rs_mask, rs_frame, train_brier
import availability_model as am
import availability as av

g = games()
ADJ = am.historical_adjustments(g)
BASE = br.PRODUCTION
RS = rs_mask()


def probs(params, shift=None):
    return br.run_walk(g, params, ADJ, season_shift=shift).probs[RS]


def fit_field(field, grid, seasons):
    res = [(train_brier(probs(replace(BASE, **{field: v})), seasons), v) for v in grid]
    return min(res)[1], res


rs = rs_frame()
# 09a
k, res = fit_field("K_playoff_rs", [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0], ev.TRAIN["test"])
k2, _ = fit_field("K_playoff_rs", [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0], ev.TRAIN["fold2"])
print("09a K_playoff_rs fit:", k, "(fold2", k2, ")  train:", ", ".join(f"{v:g}:{b:.5f}" for b, v in res))
c = ev.compare(rs, probs(BASE), probs(replace(BASE, K_playoff_rs=k)), probs(BASE), probs(replace(BASE, K_playoff_rs=k2)))
print(ev.format_compare(c))
ev.log_experiment("09a", "2.3 playoff K on the adopted model (grid incl. 0)", "experiment", "v3 (baseline_v2 + availability)",
                  f"K_playoff_rs={k:g}", {"K_playoff_rs": k}, c,
                  decision=("KEEP K=0" if k == 0 else None), notes=f"fold2 fit {k2:g}")

# 09b
m06 = importlib.import_module("06_preseason_prior")
pg = av.load_player_games()
out = {}
for fold in ("test", "fold2"):
    sp = m06.roster_shift(ev.TRAIN[fold], pg)
    res = [(train_brier(probs(BASE, {kk: gm * 28 * v for kk, v in sp.items()}), ev.TRAIN[fold]), gm)
           for gm in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]]
    out[fold] = (min(res)[1], sp)
    print(f"09b γ fit {fold}: {min(res)[1]}  train:", ", ".join(f"{gm}:{b:.5f}" for b, gm in res))
gm, sp = out["test"]; gm2, sp2 = out["fold2"]
c = ev.compare(rs, probs(BASE), probs(BASE, {kk: gm * 28 * v for kk, v in sp.items()}),
               probs(BASE), probs(BASE, {kk: gm2 * 28 * v for kk, v in sp2.items()}))
print(ev.format_compare(c))
ev.log_experiment("09b", "2.3 preseason roster prior on the adopted model (grid incl. 0)", "experiment",
                  "v3 (baseline_v2 + availability)", f"γ={gm}", {"gamma": gm}, c,
                  decision=("KEEP γ=0" if gm == 0 else None), notes=f"fold2 γ={gm2}")

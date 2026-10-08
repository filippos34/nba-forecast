"""Shared helpers for experiments/NN_*.py: one prepared game table, walk runners,
train-season parameter fits."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_ratings as br  # noqa: E402
import evaluation as ev  # noqa: E402

_GAMES = None


def games() -> pd.DataFrame:
    """All games (RS + play-in + playoffs), prepared, in walk order."""
    global _GAMES
    if _GAMES is None:
        _GAMES = br.prepare_games(br.load_games())
    return _GAMES


def rs_mask(g: pd.DataFrame | None = None) -> np.ndarray:
    g = games() if g is None else g
    return (g["season_type"] == "regular").to_numpy()


def rs_frame() -> pd.DataFrame:
    return games()[rs_mask()].reset_index(drop=True)


def rs_probs(params: br.EloParams, elo_adj: np.ndarray | None = None) -> np.ndarray:
    """RS-game probabilities (aligned with rs_frame()) for a full walk with `params`."""
    return br.run_walk(games(), params, elo_adj).probs[rs_mask()]


def train_brier(p: np.ndarray, seasons: list[str]) -> float:
    rs = rs_frame()
    m = rs["season"].isin(seasons).to_numpy()
    y = rs["actual_home_win"].to_numpy()
    return float(np.mean((p[m] - y[m]) ** 2))


def fit(base: br.EloParams, field: str, grid, seasons: list[str], elo_adj=None,
        verbose: bool = True) -> tuple[br.EloParams, pd.DataFrame]:
    """Grid-search one parameter on the TRAINING seasons' Brier."""
    rows = []
    for v in grid:
        p = replace(base, **{field: v})
        rows.append({field: v, "train_brier": train_brier(rs_probs(p, elo_adj), seasons)})
    t = pd.DataFrame(rows)
    best = t.loc[t["train_brier"].idxmin(), field]
    best = best.item() if hasattr(best, "item") else best
    if verbose:
        print(f"  fit {field} on {seasons[0]}→{seasons[-1]}: best {best} "
              f"(train Brier {t['train_brier'].min():.5f})")
    return replace(base, **{field: best}), t


def evaluate(base_params: br.EloParams, new_params: br.EloParams, base_adj=None, new_adj=None,
             new_params_fold2: br.EloParams | None = None, base_params_fold2=None,
             new_adj_fold2=None) -> dict:
    rs = rs_frame()
    pb, pn = rs_probs(base_params, base_adj), rs_probs(new_params, new_adj)
    f2b = rs_probs(base_params_fold2, base_adj) if base_params_fold2 else None
    f2n = rs_probs(new_params_fold2, new_adj_fold2 if new_adj_fold2 is not None else new_adj) \
        if new_params_fold2 else None
    if f2n is not None and f2b is None:
        f2b = pb
    return ev.compare(rs, pb, pn, f2b, f2n)

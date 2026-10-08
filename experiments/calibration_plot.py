"""Calibration (reliability) plot on the 2025-26 test season: baseline_v2 vs the final
Phase 2 model. Writes reports/calibration_2025_26.png and .csv (the table view)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, ev, rs_frame

SERIES = [("#2a78d6", "o"), ("#eb6834", "s")]   # categorical slots 1–2 (validated default palette)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot(models: dict[str, np.ndarray], out_stem: str = "calibration_2025_26"):
    rs = rs_frame()
    m = (rs["season"] == "2025-26").to_numpy()
    y = rs["actual_home_win"].to_numpy()[m]
    tables = []
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1, label="perfect calibration")
    # direct labels: series 1 below-right of its 0.2–0.3 bin, series 2 above-left of its 0.3–0.4 bin
    offsets = [(2, (10, -16), "left"), (3, (-10, 12), "right")]
    for (name, p), (color, marker), (bin_i, off, ha) in zip(models.items(), SERIES, offsets):
        t = ev.calibration_table(p[m], y)
        t.insert(0, "model", name)
        tables.append(t)
        k = t["n"] >= 10
        brier = ev.metrics(p[m], y)["brier"]
        ax.plot(t.loc[k, "mean_pred"], t.loc[k, "actual"], color=color, lw=2, marker=marker,
                ms=8, mec=SURFACE, mew=2, zorder=3, label=f"{name}  (Brier {brier:.4f})")
        anchor = t.iloc[bin_i]
        ax.annotate(name, (anchor["mean_pred"], anchor["actual"]), xytext=off, ha=ha,
                    textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home win)", color=INK2)
    ax.set_ylabel("Actual home-win rate", color=INK2)
    ax.set_title("Calibration, 2025-26 regular season (10 bins, bins with n ≥ 10)",
                 color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"{out_stem}.png", facecolor=SURFACE)
    pd.concat(tables).to_csv(ROOT / "reports" / f"{out_stem}.csv", index=False, float_format="%.4f")
    return pd.concat(tables)

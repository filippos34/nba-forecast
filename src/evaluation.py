"""
src/evaluation.py — Walk-forward evaluation, paired bootstrap, experiment log
==============================================================================
Ground rule 4: a change is accepted only if the paired bootstrap (≥2,000
resamples) of per-game Brier differences vs the baseline has a 95% CI that
excludes zero (improvement), or the improvement is ≥ 0.0085.

Test sets (regular season only):
  test        2025-26, all games                       (primary fold)
  test_apr7   2025-26 through 2026-04-06 local date     (the 1,178 April games:
                                                          excludes the last 53 tanking games)
  fold2       2024-25, all games                       (second fold)
Parameters are fit on seasons before the test season: 2021-22 → 2024-25 for
`test`, 2021-22 → 2023-24 for `fold2`.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CSV = ROOT / "reports" / "experiments.csv"
MDE = 0.0085
N_BOOT = 2000
APR7_CUTOFF = pd.Timestamp("2026-04-06")   # last local game day in the April 2026 file
TRAIN = {"test": ["2021-22", "2022-23", "2023-24", "2024-25"],
         "fold2": ["2021-22", "2022-23", "2023-24"]}
TEST_SEASON = {"test": "2025-26", "fold2": "2024-25"}


def metrics(p, y) -> dict:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return {"n": int(len(p)),
            "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "acc": float(np.mean((p > 0.5) == (y == 1)))}


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                     "actual": float(y[m].mean()) if m.any() else np.nan})
    t = pd.DataFrame(rows)
    t["diff"] = t["actual"] - t["mean_pred"]
    return t


def paired_bootstrap(p_base, p_new, y, n: int = N_BOOT, seed: int = 0) -> dict:
    """Bootstrap the mean per-game Brier difference (new − base); negative = better."""
    y = np.asarray(y, float)
    d = (np.asarray(p_new) - y) ** 2 - (np.asarray(p_base) - y) ** 2
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def decide(boot: dict) -> str:
    """ADOPT if the CI excludes zero on the improving side or Δ ≤ −MDE."""
    if boot["ci_hi"] < 0 or boot["delta"] <= -MDE:
        return "ADOPT"
    return "REJECT"


def subsets(preds: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over an RS predictions frame (needs season, date_local)."""
    local = pd.to_datetime(preds["date_local"])
    return {"test": (preds["season"] == "2025-26").to_numpy(),
            "test_apr7": ((preds["season"] == "2025-26") & (local <= APR7_CUTOFF)).to_numpy(),
            "fold2": (preds["season"] == "2024-25").to_numpy()}


def compare(preds: pd.DataFrame, p_base: np.ndarray, p_new: np.ndarray,
            fold2_base: np.ndarray | None = None, fold2_new: np.ndarray | None = None) -> dict:
    """Full comparison block. fold2_* are predictions from params refit on the fold-2
    training seasons; if absent the test-fold predictions are reused for 2024-25."""
    y = preds["actual_home_win"].to_numpy()
    out = {}
    for name, m in subsets(preds).items():
        pb, pn = (fold2_base, fold2_new) if name == "fold2" and fold2_base is not None else (p_base, p_new)
        b, nw = metrics(pb[m], y[m]), metrics(pn[m], y[m])
        boot = paired_bootstrap(pb[m], pn[m], y[m])
        out[name] = {"base": b, "new": nw, "boot": boot}
    out["decision"] = decide(out["test"]["boot"])
    return out


def format_compare(c: dict) -> str:
    lines = [f"{'subset':<10} {'n':>5} {'base Brier':>10} {'new Brier':>10} {'Δ':>8} "
             f"{'95% CI':>20} {'base LL':>8} {'new LL':>8} {'base acc':>8} {'new acc':>8}"]
    for k in ("test", "test_apr7", "fold2"):
        b, n, bt = c[k]["base"], c[k]["new"], c[k]["boot"]
        lines.append(f"{k:<10} {n['n']:>5} {b['brier']:>10.4f} {n['brier']:>10.4f} {bt['delta']:>+8.4f} "
                     f"[{bt['ci_lo']:+.4f}, {bt['ci_hi']:+.4f}] {b['logloss']:>8.4f} {n['logloss']:>8.4f} "
                     f"{b['acc']:>8.1%} {n['acc']:>8.1%}")
    lines.append(f"decision (rule 4 on test): {c['decision']}")
    return "\n".join(lines)


FIELDS = ["timestamp", "id", "name", "kind", "baseline", "change", "params",
          "n_test", "brier_base", "brier_new", "delta", "ci_lo", "ci_hi",
          "logloss_base", "logloss_new", "acc_base", "acc_new",
          "n_apr7", "brier_apr7_base", "brier_apr7_new", "delta_apr7", "ci_lo_apr7", "ci_hi_apr7",
          "brier_fold2_base", "brier_fold2_new", "delta_fold2", "ci_lo_fold2", "ci_hi_fold2",
          "auto_verdict", "decision", "decision_reason", "notes"]


def log_experiment(exp_id: str, name: str, kind: str, baseline: str, change: str,
                   params: dict, c: dict, decision: str | None = None, notes: str = "",
                   decision_reason: str | None = None):
    """Append (or replace, by id) one row in reports/experiments.csv. `auto_verdict` = rule 4 on the
    test fold; `decision` = the final call (default: follows the verdict) and `decision_reason` = why,
    stated whenever the decision differs from the verdict."""
    t, a, f = c["test"], c["test_apr7"], c["fold2"]
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "id": exp_id,
           "name": name, "kind": kind, "baseline": baseline, "change": change,
           "params": json.dumps(params, default=str),
           "n_test": t["new"]["n"], "brier_base": t["base"]["brier"], "brier_new": t["new"]["brier"],
           "delta": t["boot"]["delta"], "ci_lo": t["boot"]["ci_lo"], "ci_hi": t["boot"]["ci_hi"],
           "logloss_base": t["base"]["logloss"], "logloss_new": t["new"]["logloss"],
           "acc_base": t["base"]["acc"], "acc_new": t["new"]["acc"],
           "n_apr7": a["new"]["n"], "brier_apr7_base": a["base"]["brier"], "brier_apr7_new": a["new"]["brier"],
           "delta_apr7": a["boot"]["delta"], "ci_lo_apr7": a["boot"]["ci_lo"], "ci_hi_apr7": a["boot"]["ci_hi"],
           "brier_fold2_base": f["base"]["brier"], "brier_fold2_new": f["new"]["brier"],
           "delta_fold2": f["boot"]["delta"], "ci_lo_fold2": f["boot"]["ci_lo"], "ci_hi_fold2": f["boot"]["ci_hi"],
           "auto_verdict": c["decision"],
           "decision": decision or {"ADOPT": "Adopted", "REJECT": "Rejected"}.get(c["decision"], c["decision"]),
           "decision_reason": decision_reason or ("" if decision else "Rule 4 on the test fold."), "notes": notes}
    EXPERIMENTS_CSV.parent.mkdir(exist_ok=True)
    rows = []
    if EXPERIMENTS_CSV.exists():
        with EXPERIMENTS_CSV.open() as fh:
            rows = [r for r in csv.DictReader(fh) if r["id"] != exp_id]
    rows.append(row)
    with EXPERIMENTS_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

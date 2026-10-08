"""
src/prediction_log.py — Append-only log of every live prediction (with and without the
availability layer), so the layer's value can be tracked through the season.

data/prediction_log.csv columns:
  logged_at (UTC), game_id, tip_utc, home_team, away_team,
  prob_model        production model (baseline_v2 walk + availability layer)
  prob_no_avail     same ratings, no live availability adjustment
  prob_baseline_v2  baseline_v2 model (own ratings, no layer)
  home_avail_elo, away_avail_elo, home_report_issued_at, away_report_issued_at, model_version
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "prediction_log.csv"
FIELDS = ["logged_at", "game_id", "tip_utc", "home_team", "away_team", "prob_model", "prob_no_avail",
          "prob_baseline_v2", "home_avail_elo", "away_avail_elo", "home_report_issued_at",
          "away_report_issued_at", "model_version"]
MODEL_VERSION = "v3: baseline_v2 + availability (roster membership, exp 03e)"


def log_prediction(pred: dict, game_id=None, tip_utc=None, path: Path = LOG) -> dict:
    row = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "game_id": game_id,
           "tip_utc": tip_utc, "home_team": pred["home_team"], "away_team": pred["away_team"],
           "prob_model": pred["final_prob"], "prob_no_avail": pred["prob_no_avail"],
           "prob_baseline_v2": pred["prob_baseline_v2"], "home_avail_elo": pred["home_inj_delta"],
           "away_avail_elo": pred["away_inj_delta"],
           "home_report_issued_at": pred.get("home_report_issued_at") or "",
           "away_report_issued_at": pred.get("away_report_issued_at") or "",
           "model_version": MODEL_VERSION}
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row

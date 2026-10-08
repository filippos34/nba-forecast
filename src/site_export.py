"""
src/site_export.py — JSON for the static website (site/public/data/)
=====================================================================
Publishes ONLY: model outputs, Polymarket prices (public prediction market), aggregate statistics.
Never individual sportsbook prices: an allow-list keeps Polymarket rows only, and public_guard refuses
any export that names a bookmaker. The site is about forecasting.

  meta.json          generated_at, model version, data timestamps
  today.json         next game day: model win prob, Polymarket prob, model spread, injuries used
  season.json        2026-27 games so far + running Brier / log loss: model vs Polymarket vs baseline Elo
  calibration.json   10-bin calibration (2025-26 backtest; season-to-date once games exist)
  experiments.json   every accepted / rejected change with its CI + caught mistakes
  title_odds.json    season Monte Carlo vs Polymarket title market
  backtest.json      2025-26 out-of-sample summary
python3 src/site_export.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "site" / "public" / "data"
ET = "America/New_York"


def public_snaps(snaps: pd.DataFrame) -> pd.DataFrame:
    """Only Polymarket rows ever leave this module (allow-list, so a new bookmaker can't slip through)."""
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=getattr(snaps, "columns", []))
    out = snaps[snaps["venue"] == "polymarket"].copy()
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce")
    return out


def _write(name: str, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str, allow_nan=False))


def _clean(x):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def polymarket_ml(snaps: pd.DataFrame, game_id: int, home: str, away: str, before=None) -> dict | None:
    """De-vigged Polymarket home-win probability from the two executable asks (power method)."""
    from odds import devig                 # market code: private repo only (snaps is empty without it)
    s = snaps[(snaps["venue"] == "polymarket") & (snaps["game_id"] == game_id) & (snaps["market"] == "ml")]
    if before is not None:
        s = s[s["fetched_at"] <= before]
    s = s.sort_values("fetched_at").drop_duplicates("selection", keep="last").set_index("selection")
    if home not in s.index or away not in s.index:
        return None
    q = np.array([s.loc[home, "implied_prob"], s.loc[away, "implied_prob"]])
    p = devig.power(q)[0] if q.sum() > 1 else q[0] / q.sum()
    return {"p_home": round(float(p), 4), "at": s["fetched_at"].max().isoformat()}


def today(snaps) -> dict:
    from live import predict_day
    import margin_model as mm
    import config
    g = pd.read_csv(ROOT / "data" / "games_all.csv")
    g = g[g["season_type"].isin(["regular", "playoff", "playin"]) & ~g["completed"]]
    now_et = pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d")
    days = sorted(d for d in g["date_local"].unique() if d >= now_et)
    if not days:
        return {"date": None, "games": []}
    d = days[0]
    day, _ = predict_day(d)
    ds = mm.DerivedSpread(sigma=float(config.get("spread")["sigma"]))
    games = []
    for r in day.sort_values("tip").itertuples():
        pred = r.pred
        inj = []
        for team, det in ((r.home_team, pred["home_avail_detail"]), (r.away_team, pred["away_avail_detail"])):
            if det is not None and len(det):
                for x in det[det["status"] != "Not listed"].sort_values("proj_min", ascending=False).itertuples():
                    if x.proj_min >= 1 or x.status in ("Out", "Doubtful"):
                        inj.append({"team": team, "player": x.player_name, "status": x.status,
                                    "p_plays": round(float(x.p_plays), 2), "proj_min": round(float(x.proj_min), 1)})
        rep = [t for t in (pred.get("home_report_issued_at"), pred.get("away_report_issued_at")) if t is not None]
        games.append({"game_id": int(r.game_id), "tip_utc": r.tip_utc, "home": r.home_team, "away": r.away_team,
                      "neutral": bool(r.is_neutral), "model_p_home": round(float(r.p_home), 4),
                      "model_spread_home": round(float(ds.fair_spread(r.p_home)), 1),
                      "polymarket": polymarket_ml(snaps, int(r.game_id), r.home_team, r.away_team) if len(snaps) else None,
                      "injury_report_at": max(rep).isoformat() if rep else None, "injuries": inj,
                      "early_season": bool(r.early_season)})
    return {"date": d, "games": games}


def season(snaps) -> dict:
    """2026-27 completed games: model (as logged before tip), baseline Elo, Polymarket closing prob."""
    log = ROOT / "data" / "prediction_log.csv"
    g = pd.read_csv(ROOT / "data" / "games_all.csv")
    g = g[(g["season"] == "2026-27") & g["completed"] & g["season_type"].eq("regular")]
    rows = []
    if log.exists() and len(g):
        pl = pd.read_csv(log)
        pl["logged_at"] = pd.to_datetime(pl["logged_at"], utc=True)
        for r in g.itertuples():
            tip = pd.Timestamp(r.tip_utc)
            x = pl[(pl["game_id"] == r.game_id) & (pl["logged_at"] < tip)].sort_values("logged_at")
            if x.empty:
                continue
            last = x.iloc[-1]
            pm = polymarket_ml(snaps, int(r.game_id), r.home_team, r.away_team, before=tip) if len(snaps) else None
            rows.append({"game_id": int(r.game_id), "date": r.date_local, "home": r.home_team, "away": r.away_team,
                         "home_win": int(r.home_pts > r.away_pts), "model": float(last["prob_model"]),
                         "baseline_elo": _clean(float(last["prob_baseline_v2"])),
                         "polymarket": None if pm is None else pm["p_home"]})
    df = pd.DataFrame(rows)
    running = []
    if len(df):
        df = df.sort_values("date")
        for k in ("model", "baseline_elo", "polymarket"):
            m = df[k].notna()
            y, p = df.loc[m, "home_win"].to_numpy(float), np.clip(df.loc[m, k].to_numpy(float), 1e-6, 1 - 1e-6)
            df.loc[m, f"brier_{k}"] = np.cumsum((p - y) ** 2) / np.arange(1, m.sum() + 1)
            df.loc[m, f"ll_{k}"] = np.cumsum(-(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.arange(1, m.sum() + 1)
        running = json.loads(df.to_json(orient="records"))
    return {"games": running, "n": len(df),
            "note": "Each forecast is the last one logged before tip-off; Polymarket = de-vigged executable "
                    "prices at the last snapshot before tip."}


def calibration() -> dict:
    import evaluation as ev
    p = pd.read_csv(ROOT / "data" / "elo_predictions.csv")
    s = p[p["season"] == "2025-26"]
    y = s["actual_home_win"].to_numpy()
    out = {}
    for name, col in (("model", "home_win_prob"), ("baseline_elo", "prob_baseline_v2")):
        t = ev.calibration_table(s[col].to_numpy(), y)
        out[name] = [{k: _clean(v) for k, v in r.items()} for r in t.to_dict("records")]
    return {"season": "2025-26 (out of sample)", "n": int(len(s)), "tables": out}


def backtest() -> dict:
    import evaluation as ev
    p = pd.read_csv(ROOT / "data" / "elo_predictions.csv")
    out = []
    for season_ in ("2024-25", "2025-26"):
        s = p[p["season"] == season_]
        y = s["actual_home_win"].to_numpy()
        for name, col in (("Model (v3)", "home_win_prob"), ("Baseline Elo", "prob_baseline_v2")):
            m = ev.metrics(s[col].to_numpy(), y)
            out.append({"season": season_, "model": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
    mb = ROOT / "reports" / "market_benchmark.json"
    agg = json.loads(mb.read_text()) if mb.exists() else {"brier": {}}
    return {"rows": out, "market_aggregate": agg}


CAUGHT = [
    {"title": "Look-ahead in the rating walk (2026-03-12)",
     "what": "A rescheduled game (DAL@MEM) with a later game id was processed after the next day's games, so its "
             "prediction saw future results.", "fix": "Walk in tip-off order; a guard refuses to run otherwise. "
             "144 predictions changed by ≤ 5.5 pp."},
    {"title": "Rest days computed on UTC dates",
     "what": "Late tip-offs fall on the next UTC day: 23% of team-games had wrong rest counts and back-to-backs were "
             "over-counted by ~39%.", "fix": "Rest days on the arena's local date; penalty refit."},
    {"title": "Tuning leak into the second test season",
     "what": "Availability-layer settings had been chosen with 2024-25 data, which was also a test season.",
     "fix": "Everything refit on 2021-24 only, then tested once: pooled Δ Brier −0.0044 [−0.0075, −0.0012]."},
    {"title": "Market comparison leak (withdrawn result)",
     "what": "Comparing the model, which uses injury reports from an hour before tip, with the opening market line "
             "(set days earlier) produced an impossible '+21 pp' result — post-open news leaked into the comparison.",
     "fix": "Withdrawn. Only information-matched comparisons are reported: against the closing line the model adds "
            "nothing measurable (blend weight 0.0, 95% CI [0, 0.09])."},
]


def experiments() -> dict:
    e = pd.read_csv(ROOT / "reports" / "experiments.csv")
    keep = ["id", "name", "kind", "baseline", "change", "n_test", "brier_base", "brier_new", "delta", "ci_lo", "ci_hi",
            "delta_apr7", "delta_fold2", "ci_lo_fold2", "ci_hi_fold2", "logloss_base", "logloss_new",
            "auto_verdict", "decision", "decision_reason", "notes"]
    rows = [{k: _clean(v) for k, v in r.items()} for r in e[keep].to_dict("records")]
    return {"rule": "A change is adopted only if the paired bootstrap 95% CI of the Brier difference excludes zero "
                    "(or the gain is ≥ 0.0085). Test season 2025-26; second fold 2024-25.",
            "experiments": rows, "caught": CAUGHT}


def title_odds(snaps) -> dict:
    sim = pd.read_csv(ROOT / "data" / "title_odds_2026_27.csv")
    poly = {}
    if len(snaps):
        f = snaps[(snaps["venue"] == "polymarket") & (snaps["market"] == "future") &
                  (snaps["event_ref"] == "NBA Champion 2027")].sort_values("fetched_at").drop_duplicates("selection", keep="last")
        if len(f):
            q = f.set_index("selection")["implied_prob"]
            poly = (q / q.sum()).round(4).to_dict()
            poly_at = f["fetched_at"].max().isoformat()
    sim["avail_rel"] = sim["avail"] - sim["avail"].mean()
    sim["rank_elo"] = sim["elo"].rank(ascending=False, method="min").astype(int)
    sim["rank_strength"] = sim["strength"].rank(ascending=False, method="min").astype(int)
    rows = [{"team": r.team, "conference": r.conference, "model": round(r.p_champion, 4),
             "model_conf": round(r.p_east_or_west, 4), "mean_wins": r.mean_wins, "polymarket": poly.get(r.team),
             "elo": r.elo, "avail_rel": round(r.avail_rel, 1), "strength": r.strength,
             "rank_elo": r.rank_elo, "rank_strength": r.rank_strength}
            for r in sim.itertuples()]
    return {"rows": rows, "polymarket_at": poly_at if poly else None,
            "note": "Model: 50,000 simulated seasons (schedule, home court, rest, playoffs) with preseason uncertainty, "
                    "using exactly the team strengths behind the daily game forecasts. "
                    "Polymarket: best asks of the champion market at the snapshot, normalized to sum to 1."}


def check_public(out: Path = OUT) -> None:
    """Refuse to publish if any exported file names a bookmaker."""
    import public_guard
    bad = sorted(f.name for f in out.glob("*.json") if public_guard.bookmaker_tokens(f.read_text()))
    if bad:
        raise RuntimeError(f"bookmaker data in site export: {bad}")


def load_snaps() -> pd.DataFrame:
    try:
        from odds import schema            # market snapshots: private repo only
    except ImportError:
        return pd.DataFrame(columns=["venue", "game_id"])
    return public_snaps(schema.read())    # never export bookmaker rows


def main() -> int:
    import config
    snaps = load_snaps()
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "Elo walk + player-availability layer (v3)", "season": "2026-27"}
    _write("meta.json", meta)
    _write("today.json", today(snaps))
    _write("season.json", season(snaps))
    _write("calibration.json", calibration())
    _write("backtest.json", backtest())
    _write("experiments.json", experiments())
    if config.get("site").get("title_odds"):
        _write("title_odds.json", title_odds(snaps))
    else:
        (OUT / "title_odds.json").unlink(missing_ok=True)     # page hidden → its data is not published either
    check_public()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

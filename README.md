# NBA Forecast Lab

[![tests](https://github.com/filippos34/nba-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/filippos34/nba-forecast/actions/workflows/ci.yml)

**Live site: https://filippos34.github.io/nba-forecast/**

A National Basketball Association win-probability model built **only from game data** — results, schedules, box scores
and official injury reports. Market prices are never inputs; Polymarket is used afterwards, as a benchmark. Every
forecast is frozen before tip-off and scored in public.

## Results (out of sample, walk-forward)

| Season | Model | Games | Brier | Log loss | Accuracy |
|---|---|---|---|---|---|
| 2024-25 | Baseline Elo | 1,231 | 0.2104 | 0.6079 | 67.2% |
| 2024-25 | **Model (Elo + player availability)** | 1,231 | **0.2066** | **0.5992** | 67.3% |
| 2025-26 | Baseline Elo | 1,231 | 0.2053 | 0.5970 | 68.5% |
| 2025-26 | **Model (Elo + player availability)** | 1,231 | **0.2027** | **0.5890** | 68.9% |

- Player availability (expected lineup from the official injury report 60 minutes before tip): pooled 2024-26
  Δ Brier −0.0044, 95% CI [−0.0075, −0.0012] (paired bootstrap, 2,000 resamples), fit on 2021-24 only.
- Honest benchmark: on 974 games of 2025-26 with a sportsbook line, the closing line scored 0.1948 against the model's
  0.2033 on the same games. Blended with the closing line, the model's optimal weight is 0.0 (95% CI [0, 0.09]).
- Seventeen experiments, accepted and rejected, with their confidence intervals: `reports/experiments.csv` and the
  site's Experiments page.

## Architecture

```
 ESPN API ─────────────┐                ┌─► Elo walk (tip-off order, home court, rest, carryover)
 (scores, schedule,    │                │
  box scores, rosters) ├─► timestamped ─┼─► player availability (P(plays | status) × minutes × rating)
 Official injury PDFs ─┤   store        │         │
 (every 15 min)        │   (CSV/Parquet │         ▼
 Polymarket (compare) ─┘    + fetch     │   win probability ─► margin ~ N(σ·Φ⁻¹(p), σ)
                            times)      │         │
                                        │         ▼
                                        └─► forecast log (frozen pre-tip) ─► scoring ─► site JSON ─► static site
 Scheduled jobs (snapshots, morning refresh, health checks) and the market-snapshot code are not part of this repository.
```

| Path | What |
|---|---|
| `src/build_ratings.py` | canonical Elo walk (no look-ahead guard) |
| `src/availability_model.py`, `src/injury_reports.py` | player-availability layer, official injury-report parser |
| `src/evaluation.py` | Brier / log loss / calibration, paired bootstrap rule |
| `src/season_sim.py` | season Monte Carlo → title odds |
| `src/site_export.py`, `site/` | JSON export (model, Polymarket, aggregate stats) and the Astro static site |
| `src/public_guard.py` | publication guard: no bookmaker names, no raw-data paths (runs in CI) |
| `experiments/`, `reports/experiments.csv` | every experiment script and its logged result, accepted or rejected |
| `tests/` | pytest: schemas, no-look-ahead guards, known values |

## No look-ahead

1. Every input carries the time it became known; a forecast refuses inputs from after its own timestamp.
2. Ratings update in tip-off order (guarded by a test).
3. Injury inputs come only from reports stored with their publication time — never from who actually played.
4. Walk-forward validation: fit on earlier seasons, test on later ones.
5. A change is adopted only if the paired-bootstrap 95% CI of the Brier difference excludes zero (or the gain ≥ 0.0085).

## Setup

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                     # must pass
.venv/bin/python src/pipeline.py               # games + schedule from ESPN (cached per date)
.venv/bin/python src/player_games.py           # box scores (resumable, ~1 request/s)
.venv/bin/python src/build_ratings.py          # ratings + predictions
.venv/bin/python src/season_sim.py             # title odds
.venv/bin/python src/site_export.py            # site/public/data/*.json
cd site && npm ci && npx astro dev             # http://localhost:4321
```

## Data

The dataset is not redistributed. It is rebuilt from public sources: ESPN's public API (games, box scores,
rosters, injuries) and the NBA's official injury reports. `pipeline.py`, `player_games.py` and
`injury_reports.py` fetch and cache them locally (gitignored). Without it, the test suite still runs the
model's guards on a synthetic league; the few tests that need the real data are skipped.

Research and education only. Nothing in this repository places orders.

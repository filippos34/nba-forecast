import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Modules in src/ import each other by bare name (e.g. `from elo_model import ...`),
# and src/props uses `from src.props...`, so both roots go on the path.
for p in (ROOT, ROOT / "src", ROOT / "src" / "archive"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def games_raw() -> pd.DataFrame:
    return pd.read_csv(DATA / "games_raw.csv", parse_dates=["date"])


@pytest.fixture(scope="session")
def elo_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA / "elo_predictions.csv", parse_dates=["date"])


# ---- public repo support: the full dataset is private. Tests that need it are marked `needs_data`
# and skip without it; walk-level guards (no look-ahead, tip order, append-only) run on `raw_games`,
# which is the real data when present and a deterministic synthetic league otherwise.
HAVE_DATA = (DATA / "games_raw.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="needs the private dataset (data/)")

TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM",
         "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH"]


def synthetic_games(seasons=("2023-24", "2024-25", "2025-26"), days=80, per_day=6, seed=0) -> pd.DataFrame:
    """Regular-season-shaped games: real team codes, ET dates, UTC tips (some after midnight UTC),
    back-to-backs, a few neutral sites, scores from hidden team strengths. Same columns as games_raw."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows, gid = [], 1
    for s in seasons:
        y = int(s[:4])
        strength = rng.normal(0, 6, len(TEAMS))
        last = {}
        for d in range(days):
            day = pd.Timestamp(f"{y}-10-22") + pd.Timedelta(days=d)
            teams = rng.permutation(len(TEAMS))[: 2 * per_day]
            for k in range(per_day):
                h, a = teams[2 * k], teams[2 * k + 1]
                tip = day + pd.Timedelta(hours=int(rng.choice([19, 20, 22])) + 4)   # ET evening → UTC
                margin = strength[h] - strength[a] + 2.0 + rng.normal(0, 12)
                hp = int(110 + margin / 2); ap = int(110 - margin / 2)
                if hp == ap:
                    hp += 1
                rows.append({"game_id": gid, "date": day.strftime("%Y-%m-%d"), "date_local": day.strftime("%Y-%m-%d"),
                             "tip_utc": tip.strftime("%Y-%m-%dT%H:%M:%SZ"), "season": s,
                             "home_team": TEAMS[h], "away_team": TEAMS[a], "home_pts": hp, "away_pts": ap,
                             "home_b2b": last.get(h) == d - 1, "away_b2b": last.get(a) == d - 1,
                             "is_neutral": bool(rng.random() < 0.01), "season_type": "regular"})
                last[h] = last[a] = d
                gid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw_games() -> pd.DataFrame:
    import build_ratings as br
    return br.load_games() if HAVE_DATA else synthetic_games()

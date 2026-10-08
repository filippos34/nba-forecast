"""Elo math, the canonical build, no-lookahead in the rating walk, and the production prediction path."""
import numpy as np
import pandas as pd
import pytest

import build_ratings as br
import config
import predict
from conftest import DATA, needs_data

APRIL = DATA / "archive" / "elo_predictions_2026-04-07.csv"
GAME_COLS = ["game_id", "date", "season", "home_team", "away_team", "home_pts", "away_pts",
             "home_b2b", "away_b2b"]


@pytest.fixture(scope="module")
def april():
    if not APRIL.exists():
        pytest.skip("needs the private dataset (April 2026 production files)")
    return pd.read_csv(APRIL, parse_dates=["date"])


def test_elo_win_prob_known_values():
    assert br.elo_win_prob(1500, 1500) == pytest.approx(0.5)
    assert br.elo_win_prob(1900, 1500) == pytest.approx(10 / 11)
    assert br.elo_win_prob(1600, 1500) + br.elo_win_prob(1500, 1600) == pytest.approx(1.0)


def test_build_reproduces_april_predictions_exactly(april, tmp_path):
    """The canonical build must regenerate the April production file byte for byte."""
    april[GAME_COLS].to_csv(tmp_path / "games_raw.csv", index=False)
    br.main(["--games", str(tmp_path / "games_raw.csv"), "--playoffs", str(tmp_path / "none.csv"),
             "--playin", str(tmp_path / "none.csv"), "--out-dir", str(tmp_path), "--in-season", "--legacy"])
    assert (tmp_path / "elo_predictions.csv").read_bytes() == APRIL.read_bytes()
    cur = pd.read_csv(tmp_path / "elo_ratings_current.csv").set_index("team")["elo"]
    old = pd.read_csv(DATA / "archive" / "elo_ratings_2026-04-07.csv").set_index("team")["elo"]
    assert (cur - old.reindex(cur.index)).abs().max() < 0.051  # both rounded to 0.1


def test_build_matches_session7_core(april):
    session7 = pytest.importorskip("session7")  # src/archive/session7.py — the build behind the April files
    games = br.prepare_games(april[GAME_COLS])
    walk = br.run_walk(games, br.LEGACY)
    arr, _, n, seasons, _ = session7._prep_arrays(games, decay_rate=0.60)
    probs, _, _ = session7._run_full_predictions(arr, n, seasons, K=10, HC_fav=50, HC_dog=35,
                                                 carryover=0.75)
    np.testing.assert_array_equal(walk.probs, probs)


def test_rating_walk_has_no_lookahead(raw_games):
    """Truncating the future must not change any earlier prediction (production params)."""
    games = br.prepare_games(raw_games)
    full = br.run_walk(games, br.PRODUCTION).probs
    for frac in (0.3, 0.7, 0.95):
        cut = int(len(games) * frac)
        np.testing.assert_array_equal(br.run_walk(games.iloc[:cut], br.PRODUCTION).probs, full[:cut])


def test_appending_a_new_season_never_changes_earlier_predictions(raw_games):
    """2.0c: decay depends only on each game's own season, so adding 2026-27 games
    leaves every 2021-22 → 2025-26 prediction unchanged."""
    raw = raw_games
    base = br.run_walk(br.prepare_games(raw), br.PRODUCTION).probs
    new = raw[raw["season"] == "2025-26"].head(40).copy()
    new["season"] = "2026-27"
    new["game_id"] = new["game_id"] + 10_000_000
    for col in ("date", "date_local"):
        new[col] = (pd.to_datetime(new[col]) + pd.Timedelta(days=365)).dt.strftime("%Y-%m-%d")
    new["tip_utc"] = (pd.to_datetime(new["tip_utc"]) + pd.Timedelta(days=365)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    grown = br.run_walk(br.prepare_games(pd.concat([raw, new], ignore_index=True)), br.PRODUCTION).probs
    np.testing.assert_array_equal(grown[:len(base)], base)
    assert len(grown) == len(base) + 40


def test_walk_is_always_in_tip_order(raw_games):
    """2.0d: whatever order the input arrives in, the walk runs in tip-time order and
    gives identical predictions."""
    raw = raw_games
    games = br.prepare_games(raw)
    tips = pd.to_datetime(games["tip_utc"], utc=True)
    assert tips.is_monotonic_increasing
    shuffled = br.prepare_games(raw.sample(frac=1.0, random_state=7))
    assert list(shuffled["game_id"]) == list(games["game_id"])
    np.testing.assert_array_equal(br.run_walk(shuffled, br.PRODUCTION).probs,
                                  br.run_walk(games, br.PRODUCTION).probs)


def test_prepare_games_rejects_duplicates(april):
    bad = pd.concat([april[GAME_COLS].head(5), april[GAME_COLS].head(1)])
    with pytest.raises(ValueError):
        br.prepare_games(bad)


def test_offseason_carryover_known_value():
    out = br.apply_carryover({"A": 1700.0, "B": 1300.0}, br.LEGACY)
    assert out == {"A": 1650.0, "B": 1350.0}


def test_playoff_walk_is_zero_sum():
    po = pd.DataFrame({"game_id": [1, 2], "date": ["2026-04-20", "2026-04-22"],
                       "home_team": ["A", "B"], "away_team": ["B", "A"],
                       "home_pts": [110, 101], "away_pts": [100, 99]})
    probs, after = br.run_playoff_walk(po, {"A": 1600.0, "B": 1500.0}, br.PlayoffParams(K=3, HC=50))
    assert probs[0] == pytest.approx(br.elo_win_prob(1650, 1500))
    assert sum(after.values()) == pytest.approx(3100.0)


def test_production_brier_april_2025_26(april):
    """The April baseline. CLAUDE.md's old 0.2075 predates the Session 9 rest_1day removal."""
    s = april[april["season"] == "2025-26"]
    brier = ((s["home_win_prob"] - s["actual_home_win"]) ** 2).mean()
    assert brier == pytest.approx(0.2080, abs=5e-5)


# ── predict.py ──────────────────────────────────────────────────────────────

def test_predict_game_schema_and_split_home_court(monkeypatch):
    """Elo arithmetic only (layer off: fictional teams have no players)."""
    real = config.get
    monkeypatch.setattr(config, "get", lambda k: False if k == "model.injury_adjust" else real(k))
    ratings = {"AAA": 1550.0, "BBB": 1500.0}
    fav = predict.predict_game("AAA", "BBB", "2026-10-20", ratings=ratings)
    dog = predict.predict_game("BBB", "AAA", "2026-10-20", ratings=ratings)
    for key in ("home_team", "away_team", "date", "home_court_bonus", "raw_prob", "final_prob"):
        assert key in fav
    assert fav["home_court_bonus"] == predict.HC_FAV == 50.0
    assert dog["home_court_bonus"] == predict.HC_DOG == 35.0
    assert fav["final_prob"] == pytest.approx(br.elo_win_prob(1600, 1500), abs=1e-4)
    assert dog["final_prob"] == pytest.approx(br.elo_win_prob(1535, 1550), abs=1e-4)


def test_old_injury_table_is_retired():
    assert predict.injury_elo_delta("DEN", [{"player_name": "Nikola Jokic", "status": "Out"}]) == 0.0
    assert not hasattr(predict, "PLAYER_RAPTOR")


@needs_data
def test_availability_flag_controls_the_layer(monkeypatch):
    """Flag off → final prob equals the no-layer prob. Flag on (production) → the layer is
    applied and the with/without and baseline_v2 probs are reported for daily tracking."""
    real = config.get
    monkeypatch.setattr(config, "get", lambda k: False if k == "model.injury_adjust" else real(k))
    off = predict.predict_game("DET", "BOS", "2026-10-20")
    assert off["final_prob"] == off["prob_no_avail"] and off["home_avail_detail"] is None
    monkeypatch.setattr(config, "get", real)
    on = predict.predict_game("DET", "BOS", "2026-10-20")
    assert on["home_avail_detail"] is not None and on["prob_baseline_v2"] is not None
    assert on["final_prob"] != on["prob_no_avail"]


def test_live_prob_matches_walk_arithmetic():
    """predict.py uses the production walk's parameters (B2B −60, split HC, neutral 0)."""
    p = br.PRODUCTION
    prob, hc = predict._base_prob(1600, 1500, 0, 3, neutral=False)
    assert hc == p.HC_fav and prob == pytest.approx(br.elo_win_prob(1600 + p.HC_fav + p.rest_b2b, 1500))
    assert predict._base_prob(1600, 1500, 3, 3, neutral=True)[1] == 0.0

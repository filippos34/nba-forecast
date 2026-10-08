"""Phase 2 modules: evaluation, margin model, injury report parsing/status, availability."""
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import availability as av
import build_ratings as br
import evaluation as ev
import injury_reports as ir
import injury_status as ist
import margin_model as mm

FIX = Path(__file__).parent / "fixtures"


# ── evaluation ──────────────────────────────────────────────────────────────

def test_metrics_known_values():
    m = ev.metrics([0.5, 0.5], [1, 0])
    assert m["brier"] == pytest.approx(0.25) and m["logloss"] == pytest.approx(np.log(2))


def test_bootstrap_and_decision_rule():
    y = np.array([1, 0] * 200)
    same = ev.paired_bootstrap(np.full(400, .6), np.full(400, .6), y)
    assert same["delta"] == 0 and ev.decide(same) == "REJECT"
    better = ev.paired_bootstrap(np.full(400, .5), np.where(y == 1, .6, .4), y)
    assert better["ci_hi"] < 0 and ev.decide(better) == "ADOPT"
    assert ev.decide({"delta": -0.009, "ci_lo": -0.02, "ci_hi": 0.001}) == "ADOPT"   # |Δ| ≥ MDE


def test_calibration_table_counts():
    t = ev.calibration_table(np.linspace(0.01, 0.99, 100), np.ones(100))
    assert t["n"].sum() == 100 and len(t) == 10


# ── margin model ────────────────────────────────────────────────────────────

def test_margin_model_fit_and_probs():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 150, 20000)
    margin = 1.7 * diff / 28 + rng.normal(0, 13, 20000)
    m = mm.fit(diff, margin)
    assert m.slope == pytest.approx(1.7, abs=0.05) and m.sigma == pytest.approx(13, abs=0.3)
    assert m.p_win(0) == pytest.approx(0.5)
    assert m.fair_spread(28) == pytest.approx(-m.slope)
    assert m.p_cover(0, 0.0) == pytest.approx(0.5)


# ── injury reports ──────────────────────────────────────────────────────────

def test_labels_and_issue_times():
    old, new = ir.labels_for(date(2025, 12, 21)), ir.labels_for(date(2025, 12, 22))
    assert len(old) == 24 and len(new) == 96
    assert dict(old)["05PM"].strftime("%H:%M") == "17:30"      # hourly files issue at :30
    assert dict(new)["05_15PM"].strftime("%H:%M") == "17:15"
    cutoff = datetime(2023, 1, 15, 18, 0, tzinfo=ir.ET)       # 7pm tip − 60
    assert ir.label_before(date(2023, 1, 15), cutoff) == ["05PM"]


@pytest.mark.parametrize("fname, issued, n_min", [
    ("injury_2021-10-19_04PM.pdf", "2021-10-19 16:30", 40),
    ("injury_2026-01-15_06_30PM.pdf", "2026-01-15 18:30", 150),
])
def test_parse_pdf(fname, issued, n_min):
    if not (FIX / fname).exists():
        pytest.skip("official NBA report PDFs are kept out of the public repo")
    iss, rows = ir.parse_pdf(FIX / fname)
    assert iss.strftime("%Y-%m-%d %H:%M") == issued
    df = pd.DataFrame(rows)
    assert len(df) >= n_min
    assert set(df["status"]) <= set(ir.STATUSES) | {"NOT YET SUBMITTED"}
    assert df["team"].notna().all() and df["matchup"].notna().all()
    for mu, teams in df.groupby("matchup")["team"]:
        codes = {ist.team_abbr(t) for t in teams}
        a, h = mu.split("@")
        assert codes <= {ist.NBA_TO_ESPN.get(a, a), ist.NBA_TO_ESPN.get(h, h)}


def test_report_name_and_team():
    assert ist.report_name("Porter Jr., Kevin") == "kevin porter"
    assert ist.report_name("PippenJr.,Scotty") == "scotty pippen"
    assert ist.team_abbr("LA Clippers") == "LAC" and ist.team_abbr("GoldenStateWarriors") == "GS"


def test_status_for_not_listed_vs_no_report():
    cand = pd.DataFrame({"game_id": [1, 1, 2], "team": ["A", "A", "B"], "athlete_id": [10, 11, 12]})
    status = pd.DataFrame({"game_id": [1], "team": ["A"], "athlete_id": [10.0], "status": ["Out"]})
    st = ist.status_for(cand, status)
    assert list(st.iloc[:2]) == ["Out", "Not listed"] and pd.isna(st.iloc[2])


# ── availability ────────────────────────────────────────────────────────────

def _toy_pg():
    rows = []
    tips = pd.date_range("2025-10-22", periods=6, freq="2D", tz="UTC")
    for k, tip in enumerate(tips):
        for pid, mins, pts in ((1, 36, 30), (2, 30, 10), (3, 12, 4)):
            rows.append({"game_id": k, "team": "A", "athlete_id": pid, "tip": tip, "season": "2025-26",
                         "min": float(mins), "pts": float(pts), "fgm": pts / 2, "fga": pts / 2 + 3,
                         "fg3m": 0.0, "ftm": 0.0, "fta": 0.0, "oreb": 1.0, "dreb": 3.0, "ast": 2.0,
                         "stl": 1.0, "blk": 0.0, "tov": 1.0, "pf": 2.0, "plus_minus": 0.0,
                         "home_team": "A", "away_team": "B", "home_pts": 100, "away_pts": 95,
                         "played": True, "did_not_play": False, "box_incomplete": False})
    return pd.DataFrame(rows)


def test_player_state_is_strictly_before_the_game():
    pg = _toy_pg()
    value = av.player_values(pg, None, 0.0, "gmsc")
    state = av.player_state(pg, value, 0.0, av.AvailParams(shrink_minutes=1.0))
    cand = av.attach_state(av.candidates(pg, av.AvailParams()), state, pg, 0.0)
    before = cand[cand["game_id"] == 5].set_index("athlete_id")["rating"]
    pg2 = pg.copy()
    pg2.loc[(pg2["game_id"] == 5) & (pg2["athlete_id"] == 1), ["pts", "fgm"]] = [80.0, 40.0]
    value2 = av.player_values(pg2, None, 0.0, "gmsc")
    state2 = av.player_state(pg2, value2, 0.0, av.AvailParams(shrink_minutes=1.0))
    cand2 = av.attach_state(av.candidates(pg2, av.AvailParams()), state2, pg2, 0.0)
    after = cand2[cand2["game_id"] == 5].set_index("athlete_id")["rating"]
    pd.testing.assert_series_equal(before, after)       # game 5's own box is not used for game 5


def test_expected_lineup_adjustment_direction():
    c = pd.DataFrame({"game_id": [1, 1], "team": ["A", "A"], "proj_min": [36.0, 20.0],
                      "baseline_min": [0.0, 0.0], "rating": [10.0, 0.0]})
    full = av.team_adjustment(c, np.array([1.0, 1.0]), r_repl=0.0)
    star_out = av.team_adjustment(c, np.array([0.0, 1.0]), r_repl=0.0)
    assert full.iloc[0] == pytest.approx(36 / 48 * 10) and star_out.iloc[0] == 0.0


def test_walk_elo_adjustment_hook(raw_games):
    g = br.prepare_games(raw_games).head(300)
    base = br.run_walk(g, br.PRODUCTION).probs
    zero = br.run_walk(g, br.PRODUCTION, np.zeros((len(g), 2))).probs
    np.testing.assert_array_equal(base, zero)
    adj = np.zeros((len(g), 2)); adj[0, 0] = 100.0
    assert br.run_walk(g, br.PRODUCTION, adj).probs[0] > base[0]


def test_derived_spread_consistent_with_win_prob():
    d = mm.DerivedSpread(sigma=13.675)
    for p in (0.2, 0.5, 0.73):
        assert d.p_cover(p, 0.0) == pytest.approx(p, abs=1e-9)   # spread 0 cover = moneyline
    assert d.fair_spread(0.5) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    z = rng.normal(0, 0.6, 20000)
    margin = 12.0 * z + rng.normal(0, 12.0, 20000)
    assert mm.fit_sigma(mm.norm.cdf(z), margin).sigma == pytest.approx(12.0, rel=0.05)


def test_pre_report_p_plays_only_before_a_report(monkeypatch):
    """No report → every counted player at the fitted pre-report P; same players counted as before.
    With a report → the report's statuses (unchanged behaviour)."""
    from conftest import HAVE_DATA
    if not HAVE_DATA:
        pytest.skip("needs the private dataset (data/)")
    import availability_model as am
    import config
    tip = pd.Timestamp("2026-10-20T23:00:00Z")
    real = config.get
    no_fix = {**real("availability"), "p_plays_pre_report": None}
    monkeypatch.setattr(am, "settings", lambda: no_fix)
    old = am.live_team_adjustment("DEN", tip, None)
    monkeypatch.setattr(am, "settings", lambda: real("availability"))
    new = am.live_team_adjustment("DEN", tip, None)
    pre = float(real("availability")["p_plays_pre_report"])
    assert list(new["detail"]["counted"]) == list(old["detail"]["counted"])
    assert (new["detail"]["p_plays"] == pre).all() and (new["detail"]["status"] == "No report yet").all()
    assert new["elo"] == pytest.approx(old["elo"] * pre / float(am.fitted().p_table["Not listed"]), rel=1e-9)

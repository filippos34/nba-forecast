"""Offline tests: ESPN event parsing, schedule B2B, box-score parsing, roster helpers."""
from datetime import date

import pandas as pd
import pytest

import pipeline
import player_games
import rosters


def _event(gid, tip, stype=2, year=2026, home=("BOS", "110"), away=("NY", "100"),
           completed=True, neutral=False):
    return {"id": str(gid), "date": tip, "season": {"type": stype, "year": year},
            "competitions": [{"status": {"type": {"completed": completed, "name": "STATUS_FINAL"}},
                              "neutralSite": neutral, "type": {"abbreviation": "STD"},
                              "competitors": [
                                  {"homeAway": "home", "team": {"abbreviation": home[0]}, "score": home[1]},
                                  {"homeAway": "away", "team": {"abbreviation": away[0]}, "score": away[1]}]}]}


def test_parse_event_dates_and_types():
    # 10:30pm ET tip on Apr 6 = 02:30Z Apr 7
    row = pipeline.parse_event(_event(1, "2026-04-07T02:30Z"), date(2026, 4, 6), "2025-26", 2026)
    assert row["date"] == pd.Timestamp("2026-04-07")          # legacy UTC date
    assert row["date_local"] == pd.Timestamp("2026-04-06")    # real game day
    assert row["tip_utc"] == "2026-04-07T02:30:00Z"
    assert (row["home_team"], row["home_pts"], row["season_type"]) == ("BOS", 110, "regular")
    assert pipeline.parse_event(_event(2, "2026-04-15T23:30Z", stype=5), date(2026, 4, 15),
                                "2025-26", 2026)["season_type"] == "playin"


def test_parse_event_filters():
    d = date(2026, 1, 1)
    assert pipeline.parse_event(_event(1, "2026-01-01T00:00Z", stype=1), d, "2025-26", 2026) is None  # preseason
    assert pipeline.parse_event(_event(1, "2026-01-01T00:00Z", year=2025), d, "2025-26", 2026) is None
    assert pipeline.parse_event(_event(1, "2026-01-01T00:00Z", home=("LEB", "150")), d, "2025-26", 2026) is None
    sched = pipeline.parse_event(_event(1, "2026-10-21T23:00Z", completed=False, year=2027,
                                        home=("BOS", "0"), away=("NY", "0")), date(2026, 10, 21), "2026-27", 2027)
    assert sched["completed"] is False and sched["home_pts"] is None


def test_schedule_b2b_uses_local_dates():
    ga = pd.DataFrame([
        {"game_id": 1, "season": "2026-27", "season_type": "regular", "date_local": pd.Timestamp("2026-10-21"),
         "tip_utc": "2026-10-22T02:30:00Z", "home_team": "LAL", "away_team": "GS",
         "is_neutral": False, "comp_type": "STD", "status": "STATUS_SCHEDULED"},
        {"game_id": 2, "season": "2026-27", "season_type": "regular", "date_local": pd.Timestamp("2026-10-22"),
         "tip_utc": "2026-10-22T23:00:00Z", "home_team": "BOS", "away_team": "LAL",
         "is_neutral": False, "comp_type": "STD", "status": "STATUS_SCHEDULED"},
    ])
    s = pipeline.build_schedule(ga, "2026-27")
    assert list(s["away_b2b"]) == [False, True] and not s["home_b2b"].any()
    assert s.loc[0, "tip_local"] == "2026-10-21T19:30:00-0700"   # LA local
    assert s.loc[0, "tip_et"] == "2026-10-21T22:30:00-0400"


def _summary():
    names = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
    def ath(i, nm, stats, starter=False, dnp=False, reason=""):
        return {"athlete": {"id": str(i), "displayName": nm, "position": {"abbreviation": "G"}},
                "starter": starter, "didNotPlay": dnp, "reason": reason, "ejected": False, "stats": stats}
    return {"header": {"competitions": [{"competitors": [
                {"homeAway": "home", "team": {"abbreviation": "MEM"}, "linescores": [{}] * 5},
                {"homeAway": "away", "team": {"abbreviation": "CLE"}, "linescores": [{}] * 5}]}]},
            "boxscore": {"players": [
                {"team": {"abbreviation": "MEM"}, "statistics": [{"names": names, "athletes": [
                    ath(1, "A One", ["40", "30", "10-20", "3-8", "7-9", "5", "4", "2", "1", "0", "1", "4", "3", "+5"], True),
                    ath(2, "B Two", [], dnp=True, reason="RIGHT FOOT")]}]},
                {"team": {"abbreviation": "CLE"}, "statistics": [{"names": names, "athletes": [
                    ath(3, "C Three", ["12", "4", "2-3", "0-1", "0-0", "1", "0", "0", "0", "0", "0", "1", "2", "-7"])]}]}]}}


def test_parse_summary_rows():
    rows = player_games.parse_summary(_summary(), {"game_id": 9, "date_local": "2026-04-06",
                                                   "season": "2025-26", "season_type": "regular"})
    df = pd.DataFrame(rows).set_index("athlete_id")
    assert len(df) == 3
    a = df.loc[1]
    assert (a["team"], a["opponent"], a["is_home"], a["starter"]) == ("MEM", "CLE", True, True)
    assert (a["min"], a["pts"], a["fgm"], a["fga"], a["fg3m"], a["ftm"], a["plus_minus"]) == (40, 30, 10, 20, 3, 7, 5)
    assert a["periods"] == 5
    b = df.loc[2]
    assert b["did_not_play"] and b["dnp_reason"] == "RIGHT FOOT" and b["min"] == 0
    assert df.loc[3, "plus_minus"] == -7 and not df.loc[3, "is_home"]


@pytest.mark.parametrize("desc, how", [
    ("Acquired G Rob Dillingham from Chicago Bulls for G Buddy Hield.", "trade"),
    ("Waived G D'Angelo Russell.", "waived"),
    ("Re-signed F Dwight Powell.", "re-signed"),
    ("Signed C Christian Koloko to a 10-day contract.", "FA"),
])
def test_classify_transactions(desc, how):
    assert rosters.classify(desc) == how


def test_match_transaction_prefers_new_team():
    tx = pd.DataFrame([
        {"date": "2026-09-27", "team": "CHA", "description": "Acquired G Rob Dillingham from Chicago Bulls for G Buddy Hield."},
        {"date": "2026-09-27", "team": "CHI", "description": "Acquired G Buddy Hield from Charlotte in exchange for G Rob Dillingham."},
    ])
    how, when, desc = rosters.match_transaction("Buddy Hield", "CHI", "CHA", tx)
    assert (how, when) == ("trade", "2026-09-27") and desc.startswith("Acquired G Buddy Hield")


def test_norm_name():
    assert rosters.norm_name("Luka Dončić") == "luka doncic"
    assert rosters.norm_name("Scotty Pippen Jr.") == "scotty pippen"


def test_match_transaction_uses_the_players_clause():
    tx = pd.DataFrame([
        {"date": "2026-07-26", "team": "PHI",
         "description": "Signed F LeBron James to a contract. Waived F Dalen Terry."},
        {"date": "2026-07-06", "team": "PHI",
         "description": "Signed G Anfernee Simons to a contract. Acquired G Jaylen Brown from the Boston Celtics."},
        {"date": "2026-07-06", "team": "SAC", "description": "Waived G DeMar DeRozan."},
    ])
    assert rosters.match_transaction("LeBron James", "PHI", "LAL", tx)[0] == "FA"
    assert rosters.match_transaction("Jaylen Brown", "PHI", "BOS", tx)[0] == "trade"
    assert rosters.match_transaction("DeMar DeRozan", "DEN", "SAC", tx)[0] == "FA"
    assert rosters.match_transaction("Dalen Terry", None, "PHI", tx)[0] == "waived"

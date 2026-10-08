"""ESPN client cache behaviour and injury snapshot windows/schema (offline)."""
from datetime import date, datetime, timedelta, timezone

import pytest

import espn
import injury_snapshots as snaps


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(espn, "RAW_DIR", tmp_path)
    calls = []

    def fake_fetch(path, params=None, **kw):
        calls.append((path, params))
        return {"events": [{"competitions": [{"status": {"type": {"completed": fake_fetch.final}}}]}],
                "header": {"competitions": [{"status": {"type": {"completed": fake_fetch.final}}}]}}
    fake_fetch.final = True
    monkeypatch.setattr(espn, "fetch", fake_fetch)
    return calls, fake_fetch


def test_cache_roundtrip(tmp_cache):
    espn.write_cache("x", "k", {"a": 1})
    assert espn.read_cache("x", "k") == {"a": 1}
    assert espn.read_cache("x", "missing") is None


def test_completed_past_scoreboard_is_cached(tmp_cache):
    calls, _ = tmp_cache
    d = date.today() - timedelta(days=10)
    espn.scoreboard(d)
    espn.scoreboard(d)
    assert len(calls) == 1  # second call served from cache


def test_recent_or_unfinished_scoreboard_not_cached(tmp_cache):
    calls, fake = tmp_cache
    espn.scoreboard(date.today())            # too recent
    espn.scoreboard(date.today())
    fake.final = False
    old = date.today() - timedelta(days=10)
    espn.scoreboard(old)                     # not final
    espn.scoreboard(old)
    assert len(calls) == 4


def test_unfinished_summary_not_cached(tmp_cache):
    calls, fake = tmp_cache
    fake.final = False
    espn.summary(1)
    espn.summary(1)
    assert len(calls) == 2


def _et(h, m):
    # 2026-09-28 is EDT (UTC-4)
    return datetime(2026, 9, 28, h + 4, m, tzinfo=timezone.utc)


def test_due_window_once_per_window():
    assert snaps.due_window(_et(11, 0), set()) == "morning"
    assert snaps.due_window(_et(11, 0), {("2026-09-28", "morning")}) is None
    assert snaps.due_window(_et(17, 30), set()) == "official"
    assert snaps.due_window(_et(18, 45), set()) == "pregame"
    assert snaps.due_window(_et(14, 0), set()) is None


def test_flatten_schema_and_timestamp():
    payload = {"injuries": [{"id": "1", "injuries": [{
        "status": "Out", "date": "2026-09-21T19:50Z", "shortComment": "torn ACL",
        "details": {"type": "Knee", "location": "Leg", "returnDate": "2027-07-01",
                    "fantasyStatus": {"abbreviation": "OFS"}},
        "athlete": {"displayName": "A Player", "team": {"abbreviation": "ATL"},
                    "position": {"abbreviation": "C"},
                    "links": [{"href": "https://www.espn.com/nba/player/_/id/5105571/a-player"}]},
    }]}]}
    rows = snaps.flatten(payload, "2026-09-28T15:49:16+00:00", "morning")
    assert len(rows) == 1 and set(rows[0]) == set(snaps.FIELDS)
    r = rows[0]
    assert (r["team"], r["athlete_id"], r["status"], r["fantasy_status"]) == ("ATL", "5105571", "Out", "OFS")
    assert datetime.fromisoformat(r["fetched_at"]).tzinfo is not None  # must be tz-aware UTC


def test_missed_windows(tmp_path, monkeypatch):
    import csv
    monkeypatch.setattr(snaps, "SNAP_CSV", tmp_path / "s.csv")
    with (tmp_path / "s.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=snaps.FIELDS); w.writeheader()
        w.writerow({**{k: "" for k in snaps.FIELDS}, "fetched_at": "2026-04-10T15:00:00+00:00", "window": "morning"})
    day = date(2026, 4, 10)   # a game day
    after = datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc)
    assert snaps.missed_windows(day, after) == ["official", "pregame"]
    early = datetime(2026, 4, 10, 16, 5, tzinfo=timezone.utc)   # 12:05 ET: nothing due yet
    assert snaps.missed_windows(day, early) == []


def test_keep_awake_only_in_evening_span(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(snaps, "LOG_DIR", tmp_path)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: started.append(a))
    assert snaps.keep_awake(_et(11, 0)) is False
    assert snaps.keep_awake(_et(17, 0)) is True and len(started) == 1
    assert snaps.keep_awake(_et(18, 0)) is False          # once per ET day

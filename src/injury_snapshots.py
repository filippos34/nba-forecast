"""
src/injury_snapshots.py — Timestamped snapshots of the ESPN injuries feed
==========================================================================
Every snapshot is stored twice:
  data/raw/espn/injuries/<YYYYMMDDTHHMMSSZ>.json.gz   raw payload (replayable)
  data/injury_snapshots.csv                           one row per player per snapshot

`fetched_at` (UTC) is the time the information was known to us; only snapshots
with fetched_at < prediction time may be used for a prediction.

Scheduling: launchd runs this every 15 minutes (see scripts/launchd/). Without
--force a snapshot is only taken inside the US-Eastern windows below, once per
window per day, so missed windows (Mac asleep) are simply skipped — and reported by
missed_windows() / --check in the daily report. Between 16:00 and 19:15 ET the job starts
one `caffeinate -i -t 12000` per day so the Mac stays awake through the evening windows
(the Mac is woken at 23:05 local by `pmset repeat wakeorpoweron`).

Usage:
    python3 src/injury_snapshots.py            # take one if a window is due
    python3 src/injury_snapshots.py --force    # take one now
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SNAP_CSV = DATA_DIR / "injury_snapshots.csv"
ET = ZoneInfo("America/New_York")

# (name, start, end) in US Eastern time
WINDOWS = [
    ("morning",  time(10, 45), time(12, 0)),   # overnight updates settled
    ("official", time(17, 15), time(18, 15)),  # after the league's 5pm ET report
    ("pregame",  time(18, 30), time(19, 15)),  # 15–90 min before typical 7pm ET tips
]

FIELDS = ["fetched_at", "window", "team", "team_id", "athlete_id", "player_name",
          "position", "status", "fantasy_status", "type", "location", "return_date",
          "espn_updated", "short_comment"]


def due_window(now_utc: datetime, taken: set[tuple[str, str]]) -> str | None:
    now_et = now_utc.astimezone(ET)
    for name, start, end in WINDOWS:
        if start <= now_et.time() < end and (now_et.date().isoformat(), name) not in taken:
            return name
    return None


def windows_taken() -> set[tuple[str, str]]:
    if not SNAP_CSV.exists():
        return set()
    out = set()
    with SNAP_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["window"]:
                et_day = datetime.fromisoformat(r["fetched_at"]).astimezone(ET).date().isoformat()
                out.add((et_day, r["window"]))
    return out


def flatten(payload: dict, fetched_at: str, window: str) -> list[dict]:
    rows = []
    for team in payload.get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {}) or {}
            det = inj.get("details", {}) or {}
            ath_id = ""
            for link in ath.get("links", []):
                href = link.get("href", "")
                if "/id/" in href:
                    ath_id = href.split("/id/")[1].split("/")[0]
                    break
            rows.append({
                "fetched_at": fetched_at,
                "window": window,
                "team": (ath.get("team") or {}).get("abbreviation", "") or team.get("abbreviation", ""),
                "team_id": team.get("id", ""),
                "athlete_id": ath_id,
                "player_name": ath.get("displayName", ""),
                "position": (ath.get("position") or {}).get("abbreviation", ""),
                "status": inj.get("status", ""),
                "fantasy_status": (det.get("fantasyStatus") or {}).get("abbreviation", ""),
                "type": det.get("type", ""),
                "location": det.get("location", ""),
                "return_date": det.get("returnDate", ""),
                "espn_updated": inj.get("date", ""),
                "short_comment": (inj.get("shortComment") or "").replace("\n", " "),
            })
    return rows


def take_snapshot(window: str = "") -> tuple[Path, int]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    fetched_at = now.isoformat()
    payload = espn.injuries_raw()
    payload["_fetched_at"] = fetched_at
    raw_path = espn.write_cache("injuries", now.strftime("%Y%m%dT%H%M%SZ"), payload)
    rows = flatten(payload, fetched_at, window)
    new_file = not SNAP_CSV.exists()
    with SNAP_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    return raw_path, len(rows)


AWAKE_FROM, AWAKE_UNTIL = time(16, 0), time(19, 15)   # ET span covering official + pregame
CAFFEINATE_SECONDS = 12000
LOG_DIR = ROOT / "logs"


def keep_awake(now_utc: datetime) -> bool:
    """Inside the evening ET span, start one detached `caffeinate -i -t 12000` per ET day so
    the Mac stays awake through the official and pregame windows. Returns True if started."""
    import subprocess
    try:
        import config
        if not config.get("ops")["keep_awake"]:
            return False
    except Exception:
        pass
    now_et = now_utc.astimezone(ET)
    if not (AWAKE_FROM <= now_et.time() < AWAKE_UNTIL):
        return False
    marker = LOG_DIR / f"caffeinate_{now_et.date().isoformat()}.started"
    if marker.exists():
        return False
    LOG_DIR.mkdir(exist_ok=True)
    subprocess.Popen(["/usr/bin/caffeinate", "-i", "-t", str(CAFFEINATE_SECONDS)],
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    marker.write_text(now_utc.isoformat())
    return True


def missed_windows(et_day, now_utc: datetime | None = None) -> list[str]:
    """Windows of US-Eastern day `et_day` that have passed without a snapshot (only for
    days with NBA games). Empty list = nothing missed."""
    import pandas as pd
    now_utc = now_utc or datetime.now(timezone.utc)
    games = DATA_DIR / "games_all.csv"
    if games.exists():
        g = pd.read_csv(games, usecols=["date_local"])
        if et_day.isoformat() not in set(g["date_local"].astype(str)):
            return []
    taken = windows_taken()
    out = []
    for name, _, end in WINDOWS:
        end_utc = datetime.combine(et_day, end, tzinfo=ET).astimezone(timezone.utc)
        if end_utc <= now_utc and (et_day.isoformat(), name) not in taken:
            out.append(name)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--force", action="store_true", help="snapshot now, outside the windows")
    ap.add_argument("--check", metavar="YYYY-MM-DD", help="list missed windows for an ET day")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    if args.check:
        from datetime import date as _date
        missed = missed_windows(_date.fromisoformat(args.check), now)
        print(f"{args.check}: missed windows: {', '.join(missed) if missed else 'none'}")
        return 1 if missed else 0
    if keep_awake(now):
        print(f"{now.isoformat(timespec='seconds')} caffeinate -i -t {CAFFEINATE_SECONDS} started")
    window = due_window(now, windows_taken())
    if not args.force and window is None:
        return 0
    raw_path, n = take_snapshot(window or "")
    print(f"{now.isoformat(timespec='seconds')} snapshot window={window or 'forced'} "
          f"rows={n} raw={raw_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
src/injury_reports.py — Official NBA injury report archive: download + parse
=============================================================================
Source: https://ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<label>.pdf
  - through 2025-12-21: hourly labels  "05PM"     (issued at HH:30 ET)
  - from    2025-12-22: 15-min labels  "05_15PM"  (issued at HH:MM ET)
The issue time printed in each PDF is the ground truth (`issued_at`); the label
only picks the file. See reports/injury_sources.md.

Download (resumable, ~1 req/s, raw PDFs under data/raw/nba_injury/<date>/<label>.pdf,
misses recorded in data/raw/nba_injury/missing.csv so they are never re-requested):
    python3 src/injury_reports.py download --pass targeted   # files the T-60 backtest needs
    python3 src/injury_reports.py download --pass full       # every label on every game day
    python3 src/injury_reports.py download --date 2026-10-20 # one day
    python3 src/injury_reports.py download --recent 3        # last 3 ET days (daily launchd job)
Parse all downloaded PDFs → data/injury_reports.parquet:
    python3 src/injury_reports.py parse
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "nba_injury"
MISSING = RAW / "missing.csv"
OUT = DATA / "injury_reports.parquet"
BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126 Safari/537.36"}
ET = ZoneInfo("America/New_York")
SWITCH = date(2025, 12, 22)          # first day of 15-minute labels
DECISION_MINUTES = 60                # backtest decision time: T-60 before tip
MIN_INTERVAL = 1.0

_last = 0.0
_session = requests.Session()


# ── Labels ──────────────────────────────────────────────────────────────────

def _ampm(h: int) -> tuple[int, str]:
    return (h % 12 or 12), ("AM" if h < 12 else "PM")


def labels_for(d: date) -> list[tuple[str, datetime]]:
    """All labels of a day with their nominal issue time (ET), in time order."""
    out = []
    if d < SWITCH:
        for h in range(24):
            hh, ap = _ampm(h)
            out.append((f"{hh:02d}{ap}", datetime(d.year, d.month, d.day, h, 30, tzinfo=ET)))
    else:
        for h in range(24):
            for m in (0, 15, 30, 45):
                hh, ap = _ampm(h)
                out.append((f"{hh:02d}_{m:02d}{ap}", datetime(d.year, d.month, d.day, h, m, tzinfo=ET)))
    return out


def label_before(d: date, cutoff: datetime, k: int = 1) -> list[str]:
    """The k latest labels on day d nominally issued at or before `cutoff`."""
    labs = [lab for lab, t in labels_for(d) if t <= cutoff]
    return labs[-k:]


# ── Download ────────────────────────────────────────────────────────────────

def pdf_path(d: date, label: str) -> Path:
    return RAW / d.isoformat() / f"{label}.pdf"


def _missing() -> set[tuple[str, str]]:
    if not MISSING.exists():
        return set()
    with MISSING.open() as f:
        return {(r["date"], r["label"]) for r in csv.DictReader(f)}


def _mark_missing(d: date, label: str, code: int):
    new = not MISSING.exists()
    MISSING.parent.mkdir(parents=True, exist_ok=True)
    with MISSING.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "label", "http"])
        w.writerow([d.isoformat(), label, code])


def fetch_pdf(d: date, label: str, missing: set) -> Path | None:
    """Download one report unless already on disk or known missing. Returns the path or None."""
    global _last
    p = pdf_path(d, label)
    if p.exists():
        return p
    if (d.isoformat(), label) in missing:
        return None
    for attempt in range(4):
        wait = _last + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        try:
            r = _session.get(f"{BASE}{d.isoformat()}_{label}.pdf", headers=UA, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(p)
            return p
        if r.status_code in (403, 404):   # CloudFront answers 403 for absent keys
            _mark_missing(d, label, r.status_code)
            missing.add((d.isoformat(), label))
            return None
        time.sleep(2 ** attempt * 2)
    return None


def game_times() -> pd.DataFrame:
    g = pd.read_csv(DATA / "games_all.csv")
    g = g[g["season_type"].isin(["regular", "playoff", "playin"])]
    g["tip"] = pd.to_datetime(g["tip_utc"], utc=True).dt.tz_convert(ET)
    g["day"] = g["tip"].dt.date
    return g


def targeted_jobs(g: pd.DataFrame, fallback: int = 2) -> list[tuple[date, str]]:
    """For each game: the latest label at/before tip-60 plus `fallback` earlier ones
    (in case a file is missing or was issued late)."""
    jobs = set()
    for day, tip in zip(g["day"], g["tip"]):
        cutoff = tip.to_pydatetime() - timedelta(minutes=DECISION_MINUTES)
        for lab in label_before(day, cutoff, k=1 + fallback):
            jobs.add((day, lab))
    return sorted(jobs)


def full_jobs(g: pd.DataFrame) -> list[tuple[date, str]]:
    return [(d, lab) for d in sorted(set(g["day"])) for lab, _ in labels_for(d)]


def download(jobs: list[tuple[date, str]], log_every: int = 200):
    missing = _missing()
    todo = [(d, lab) for d, lab in jobs if not pdf_path(d, lab).exists()
            and (d.isoformat(), lab) not in missing]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done/known-missing, fetching {len(todo)} "
          f"(~{len(todo) / 3600:.1f} h at 1 req/s)", flush=True)
    got = 0
    t0 = time.time()
    for i, (d, lab) in enumerate(todo, 1):
        if fetch_pdf(d, lab, missing):
            got += 1
        if i % log_every == 0:
            print(f"  {i}/{len(todo)}  got {got}  {time.time() - t0:.0f}s  at {d} {lab}", flush=True)
    print(f"done: {got} new PDFs, {len(todo) - got} missing", flush=True)


# ── Parse ───────────────────────────────────────────────────────────────────

STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
HEADER_RE = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s*(\d{1,2}:\d{2})\s*(AM|PM)")
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}(\(ET\))?$")
MATCHUP_RE = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
PAGE_RE = re.compile(r"\bPage\s*\d+\s*of\s*\d+\b")
NAME_RE = re.compile(r"^[A-Za-z.'\-]+(?:\s?[A-Za-z.'\-]+)*,\s?[A-Za-z.'\-]+")


def parse_pdf(path: Path) -> tuple[datetime | None, list[dict]]:
    """Returns (issued_at ET, rows). Uses word x-positions to assign columns, carrying
    game / team cells down and attaching wrapped reason lines to the nearest row."""
    import pdfplumber

    rows: list[dict] = []
    issued = None
    with pdfplumber.open(path) as pdf:
        cols = None
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1.5)
            text0 = page.extract_text() or ""
            m = HEADER_RE.search(text0.replace("\n", " "))
            if m and issued is None:
                issued = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%m/%d/%y %I:%M %p")
                issued = issued.replace(tzinfo=ET)
            # column x-starts from the header row of this page
            mtop = next((w["top"] for w in words if w["text"] == "Matchup"), None)
            hdr_words = [w for w in words if mtop is not None and abs(w["top"] - mtop) < 3]
            hdr = {}
            for w in sorted(hdr_words, key=lambda w: w["x0"]):   # header line only
                if w["text"] in ("Game", "GameDate", "Time", "GameTime", "Matchup", "Team", "Player",
                                 "PlayerName", "Current", "CurrentStatus", "Reason"):
                    hdr.setdefault(w["text"], w["x0"])
            if "Matchup" in hdr and "Reason" in hdr:
                player_x = hdr.get("Player", hdr.get("PlayerName"))
                status_x = hdr.get("Current", hdr.get("CurrentStatus"))
                game_xs = sorted(x for k, x in hdr.items() if k in ("Game", "GameDate"))
                time_x = hdr.get("Time", hdr.get("GameTime"))
                cols = {"date": game_xs[0] if game_xs else 0, "time": time_x - 20 if time_x else 60,
                        "matchup": hdr["Matchup"], "team": hdr["Team"], "player": player_x,
                        "status": status_x, "reason": hdr["Reason"]}
            if cols is None:
                continue
            order = ["date", "time", "matchup", "team", "player", "status", "reason"]
            bounds = [(c, cols[c]) for c in order]

            def col_of(x):
                name = order[0]
                for c, x0 in bounds:
                    if x >= x0 - 3:
                        name = c
                return name

            lines: dict[float, list] = {}
            for w in words:
                key = round(w["top"] / 3) * 3
                lines.setdefault(key, []).append(w)
            for top in sorted(lines):
                ws = sorted(lines[top], key=lambda w: w["x0"])
                cells: dict[str, list[str]] = {}
                for w in ws:
                    cells.setdefault(col_of(w["x0"]), []).append(w["text"])
                cell = {k: " ".join(v) for k, v in cells.items()}
                joined = " ".join(w["text"] for w in ws)
                if "Injury Report:" in joined or cell.get("reason") == "Reason" \
                        or cell.get("matchup") == "Matchup" or PAGE_RE.search(joined):
                    continue
                rows.append({"top": top, "page": page.page_number, **cell})
    return issued, _assemble(rows)


def _assemble(lines: list[dict]) -> list[dict]:
    out, game_date, game_time, matchup, team = [], None, None, None, None
    pending_reason: list[str] = []
    for ln in lines:
        d = ln.get("date", "")
        for tok in d.split():
            if DATE_RE.match(tok):
                game_date = tok
            elif TIME_RE.match(tok):
                game_time = tok.replace("(ET)", "")
        t = ln.get("time", "").replace("(ET)", "").strip()
        if t and TIME_RE.match(t.split()[0]):
            game_time = t.split()[0]
        mu = ln.get("matchup", "").strip()
        if mu and MATCHUP_RE.match(mu.split()[0]):
            matchup = mu.split()[0]
        tm = ln.get("team", "").strip()
        status_cell = ln.get("status", "").strip()
        player = ln.get("player", "").strip()
        reason = ln.get("reason", "").strip()
        if tm:
            team = tm
        if "NOT YET SUBMITTED" in " ".join(ln.get(k, "") for k in ("team", "player", "status", "reason")).upper():
            out.append({"game_date": game_date, "game_time": game_time, "matchup": matchup,
                        "team": team, "player": None, "status": "NOT YET SUBMITTED", "reason": ""})
            pending_reason = []
            continue
        status = next((s for s in STATUSES if status_cell.startswith(s)), None)
        if status and player:
            full_reason = " ".join(pending_reason + ([reason] if reason else [])).strip()
            out.append({"game_date": game_date, "game_time": game_time, "matchup": matchup,
                        "team": team, "player": player, "status": status, "reason": full_reason})
            pending_reason = []
        elif reason and not player and not status_cell:
            # wrapped reason fragment: before the row (vertically centred cells) or after it
            if out and out[-1]["player"] and _looks_like_continuation(reason):
                out[-1]["reason"] = (out[-1]["reason"] + " " + reason).strip()
            else:
                pending_reason.append(reason)
    return out


def _looks_like_continuation(s: str) -> bool:
    return not re.match(r"^(Injury/Illness|G League|League Suspension|Rest|Personal|Health|"
                        r"Concussion|Not With Team|Suspension|Coach|Return to Competition|"
                        r"GLeague|Injury)", s)


def _parse_one(p: Path):
    try:
        issued, rows = parse_pdf(p)
    except Exception as e:  # corrupt PDF etc.
        return str(p.relative_to(RAW)), None, repr(e)[:80]
    if issued is None:
        return str(p.relative_to(RAW)), None, "no header"
    df = pd.DataFrame(rows)
    if not df.empty:
        df["issued_at"] = issued.astimezone(ZoneInfo("UTC"))
        df["file"] = str(p.relative_to(RAW))
    return str(p.relative_to(RAW)), df, None


def parse_all(workers: int = 8) -> pd.DataFrame:
    from multiprocessing import Pool
    files = sorted(RAW.glob("*/*.pdf"))
    frames, bad = [], []
    with Pool(workers) as pool:
        for i, (name, df, err) in enumerate(pool.imap_unordered(_parse_one, files, chunksize=16), 1):
            if err:
                bad.append((name, err))
            elif df is not None and not df.empty:
                frames.append(df)
            if i % 2000 == 0:
                print(f"  parsed {i}/{len(files)}", flush=True)
    out = pd.concat(frames, ignore_index=True).sort_values(["issued_at", "file"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"{len(out)} rows from {len(files) - len(bad)} PDFs → {OUT.relative_to(ROOT)}; {len(bad)} failed")
    if bad:
        print("  failures:", bad[:10])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--pass", dest="which", choices=["targeted", "full"], default="targeted")
    d.add_argument("--date", type=date.fromisoformat)
    d.add_argument("--recent", type=int, metavar="N",
                   help="every label of the last N US-Eastern days before today (daily job)")
    sub.add_parser("parse")
    args = ap.parse_args(argv)
    if args.cmd == "parse":
        parse_all()
        return 0
    if args.date:
        download([(args.date, lab) for lab, _ in labels_for(args.date)])
        return 0
    if args.recent:
        today_et = datetime.now(ET).date()
        days = [today_et - timedelta(days=k) for k in range(args.recent, 0, -1)]
        download([(d, lab) for d in days for lab, _ in labels_for(d)])
        return 0
    g = game_times()
    g = g[g["season"].isin(["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])]
    download(targeted_jobs(g) if args.which == "targeted" else full_jobs(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())

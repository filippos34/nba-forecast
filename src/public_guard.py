"""
src/public_guard.py — what may never appear in anything published (site JSON, public repo)
==========================================================================================
1. Bookmaker names. Stored as truncated SHA-256 hashes of lowercase word tokens, so this file — which
   is itself public — names no bookmaker. A token matches if its hash is in BOOKMAKER_HASHES.
2. Excluded paths (raw data, market snapshots, the parked staking / alerting code, private notes).

    python -m public_guard <dir>     → exit 1 and list every hit (used by `make publish` and public CI)
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

BOOKMAKER_HASHES = {
    "15203c49255f7aed",
    "19f76a23eb1afa40",
    "1db41fc1e1f3f6f9",
    "2843dabe5df1a46b",
    "28ee793c0f72d1d4",
    "2c0c5201b00d8f2f",
    "2c34b60786ffe623",
    "2c810e4d3f9eedfc",
    "2c9e6b6b92272cde",
    "377fff252c7a975b",
    "37812a20658f8618",
    "408bbfb77e82ad6b",
    "4f300db335a70842",
    "5396538ebe0d9510",
    "5cc22620c7f86752",
    "5e021533773059a4",
    "68bc1c44b93cad6e",
    "698bc431635ad7bc",
    "74953ad30f36d88f",
    "761fb7534028953d",
    "787726ef2162f30a",
    "7e5f7cbbccba57c8",
    "7e8aec273f907a15",
    "7f58e8dd0446b4e8",
    "82f5db6f38c71a54",
    "867b4bf4357a7c0e",
    "8f292ff8768e7a40",
    "900883e45b3aba03",
    "9852707034b729f4",
    "a385abc59f83aef3",
    "a4224a6711e1ec16",
    "b33769b8ce3b1b45",
    "b3f093236eafc1a1",
    "b3f713a4de96a08f",
    "b59c8a33296ffcf2",
    "ba7094570d00ccd6",
    "bbbc2caf036dfe9d",
    "bde05fa521e35b04",
    "c17b3eb1958eb81f",
    "c55c054b058b93b7",
    "c7fe18220357cc6d",
    "c8a1d1068548452c",
    "cb87b7885c9470db",
    "d6d622457af72a64",
    "ea87bd9488f61443",
    "eb170fd056f9aca5",
    "f57b4d6a65346c03",
    "fb5f967a5f38de46",
    "fc87e4aa9aa9d679",
    "fca69c220dca10ea",
}
EXCLUDED_PATHS = [
    r"(^|/)data/", r"(^|/)odds/", r"(^|/)scanner/", r"scrapers?(/|$)", r"(^|/)ops/", r"(^|/)deploy/",
    r"(^|/)archive/", r"(^|/)props/", r"bet", r"stak", r"alert", r"kelly", r"paper_trade", r"historical_lines",
    r"odds_snapshots", r"decision", r"(^|/)clv", r"\.env$", r"\.parquet$", r"\.pdf$", r"CLAUDE\.md$",
    r"super_prompt", r"private_", r"shrink_weight", r"policy_backtest",
]
ALLOWED_DATA = re.compile(r"^site/public/data/[a-z_]+\.json$")       # small derived site JSON
_TOKEN = re.compile(r"[a-z0-9]+")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".ts", ".astro", ".mjs", ".js", ".html",
                 ".css", ".toml", ".ini", ".cfg", ".sh", ".svg", ""}


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def bookmaker_tokens(text: str) -> set[str]:
    """Tokens of `text` that are bookmaker names."""
    return {t for t in set(_TOKEN.findall(text.lower())) if _h(t) in BOOKMAKER_HASHES}


def excluded_path(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if ALLOWED_DATA.match(rel):
        return False
    return any(re.search(p, rel, re.I) for p in EXCLUDED_PATHS)


def scan(root: Path, skip=(".git", "node_modules", "dist", ".astro", "__pycache__", ".venv")) -> list[str]:
    """Every problem under `root`: excluded paths and files mentioning a bookmaker (name redacted)."""
    root = Path(root)
    out = []
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root).as_posix()
        if f.is_dir() or any(part in skip for part in f.relative_to(root).parts):
            continue
        if excluded_path(rel):
            out.append(f"excluded path: {rel}")
        if f.suffix.lower() in TEXT_SUFFIXES:
            try:
                hits = bookmaker_tokens(f.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
            if hits:
                out.append(f"bookmaker name in {rel} ({len(hits)} distinct)")
    return out


def main(argv=None) -> int:
    root = Path((argv or sys.argv[1:] or ["."])[0])
    problems = scan(root)
    for p in problems:
        print(p)
    print(f"public guard: {'FAIL' if problems else 'ok'} ({len(problems)} problems) in {root}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

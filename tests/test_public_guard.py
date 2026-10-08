"""public_guard: bookmaker names are caught (and never written in clear), excluded paths are caught."""
import public_guard as pg

NAME = "stoix" + "iman"           # split so this file itself passes the guard


def test_detects_bookmaker_tokens_any_case():
    assert pg.bookmaker_tokens(f"price from {NAME.upper()} at 1.91") == {NAME}
    assert pg.bookmaker_tokens(f"venue={'bet' + '365'}") == {"bet" + "365"}
    assert pg.bookmaker_tokens("polymarket home 0.55, model 0.52") == set()


def test_guard_file_names_no_bookmaker():
    from pathlib import Path
    assert pg.bookmaker_tokens(Path(pg.__file__).read_text()) == set()


def test_excluded_paths():
    for bad in ["data/games_raw.csv", "src/odds/sheet.py", "src/scanner/scan.py", "src/odds/scrapers/x.py",
                "data/paper_trades.csv", "CLAUDE.md", "src/private_defaults.py", "x/.env", "a/b.parquet"]:
        assert pg.excluded_path(bad), bad
    for ok in ["src/build_ratings.py", "site/public/data/today.json", "tests/test_elo.py", "README.md"]:
        assert not pg.excluded_path(ok), ok


def test_scan_reports_both_kinds(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("x = 1\n")
    assert pg.scan(tmp_path) == []
    (tmp_path / "src" / "note.md").write_text(f"lines from {NAME}\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "games.csv").write_text("a,b\n")
    probs = pg.scan(tmp_path)
    assert any("excluded path: data/games.csv" in p for p in probs)
    assert any("bookmaker name in src/note.md" in p for p in probs)
    assert not any(NAME in p for p in probs)          # report never prints the name

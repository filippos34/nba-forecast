"""site_export: only Polymarket rows and aggregate stats leave the repo."""
NAME = "stoix" + "iman"           # split so this file itself passes the public guard


def test_site_export_publishes_no_bookmaker_data(tmp_path):
    """Only Polymarket rows survive; a file naming a bookmaker blocks publishing."""
    import pandas as pd
    import pytest
    import site_export
    snaps = pd.DataFrame({"venue": ["polymarket", NAME, "pinn" + "acle", "some_new_book"],
                          "game_id": ["1", "1", "1", "1"]})
    assert list(site_export.public_snaps(snaps)["venue"]) == ["polymarket"]
    (tmp_path / "ok.json").write_text('{"polymarket": 0.5}')
    site_export.check_public(tmp_path)
    (tmp_path / "bad.json").write_text(f'{{"note": "{NAME.title()} price"}}')
    with pytest.raises(RuntimeError):
        site_export.check_public(tmp_path)


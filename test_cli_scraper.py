from datetime import date

from cli_scraper import SettlementDB, get_settlement_value, parse_cli_report


def test_parse_cli_report_extracts_date_and_tmax():
    sample = """
    <html><body>
    CLIMATE REPORT FOR FEBRUARY 09 2026
    MAXIMUM TEMPERATURE (F)     36
    </body></html>
    """
    report_date, tmax = parse_cli_report(sample)
    assert report_date == date(2026, 2, 9)
    assert tmax == 36


def test_get_settlement_value_prefers_db(tmp_path):
    db = SettlementDB(path=tmp_path / "settlements.json")
    db.record("KXHIGHNY", date(2026, 2, 9), 37, source="cli")

    out = get_settlement_value(date(2026, 2, 9), series_ticker="KXHIGHNY", db=db)
    assert out is not None
    assert out["tmax_f"] == 37
    assert out["source"] == "cli"
    assert out["confidence"] == "official"


def test_get_settlement_value_uses_ghcn_fallback_when_cli_misses(tmp_path, monkeypatch):
    db = SettlementDB(path=tmp_path / "settlements.json")

    monkeypatch.setattr("cli_scraper.scrape_and_record", lambda series_ticker, db: (date(2026, 2, 8), 35))
    monkeypatch.setattr("cli_scraper._ghcn_tmax_for_series", lambda target_date, series_ticker: 34)

    out = get_settlement_value(date(2026, 2, 9), series_ticker="KXHIGHNY", db=db)
    assert out is not None
    assert out["tmax_f"] == 34
    assert out["source"] == "ghcn_fallback"
    assert out["confidence"] == "provisional"

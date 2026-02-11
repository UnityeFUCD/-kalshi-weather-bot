from datetime import datetime

import config
from bot import WeatherBot
from risk import RiskManager


class _DummyAuth:
    def __init__(self, *args, **kwargs):
        pass


class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass


def test_live_mode_requires_live_trading_gate(monkeypatch):
    monkeypatch.setattr("bot.KalshiAuth", _DummyAuth)
    monkeypatch.setattr("bot.KalshiClient", _DummyClient)
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", "test-key")
    monkeypatch.setattr(config, "LIVE_TRADING", False)

    try:
        WeatherBot(mode="live")
        assert False, "Expected RuntimeError when LIVE_TRADING is false"
    except RuntimeError as exc:
        assert "LIVE_TRADING=true" in str(exc)


def test_risk_manager_uses_market_timezone_for_day_boundary():
    rm = RiskManager()
    expected = datetime.now(config.MARKET_TZ).strftime("%Y-%m-%d")
    assert rm._today_str() == expected

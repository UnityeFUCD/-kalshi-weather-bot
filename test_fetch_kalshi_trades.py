from __future__ import annotations

import fetch_kalshi_trades as mod


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_list_markets_for_series_stops_on_repeated_cursor(monkeypatch):
    payloads = [
        {"markets": [{"ticker": "M1"}], "cursor": "c1"},
        {"markets": [{"ticker": "M2"}], "cursor": "c1"},
    ]

    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        idx = calls["n"]
        calls["n"] += 1
        return _FakeResp(payloads[idx])

    monkeypatch.setattr(mod.requests, "get", fake_get)

    markets = mod.list_markets_for_series("KXHIGHNY", 90)

    assert [m["ticker"] for m in markets] == ["M1", "M2"]
    assert calls["n"] == 2


def test_fetch_trades_for_market_stops_on_repeated_cursor(monkeypatch):
    payloads = [
        {"trades": [{"id": "t1"}], "cursor": "c1"},
        {"trades": [{"id": "t2"}], "cursor": "c1"},
    ]

    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        idx = calls["n"]
        calls["n"] += 1
        return _FakeResp(payloads[idx])

    monkeypatch.setattr(mod.requests, "get", fake_get)

    trades = mod.fetch_trades_for_market("KXHIGHNY-TEST", min_ts=0)

    assert [t["id"] for t in trades] == ["t1", "t2"]
    assert calls["n"] == 2

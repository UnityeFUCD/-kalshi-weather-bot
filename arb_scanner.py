"""
arb_scanner.py -- Detect and log simple weather bucket arbitrage opportunities.

Detection only. No order execution.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from model import compute_fee

logger = logging.getLogger("arb_scanner")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _group_by_event(markets: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for market in markets:
        event_ticker = getattr(market, "event_ticker", None)
        if event_ticker:
            grouped[event_ticker].append(market)
    return grouped


def _base_record(event_ticker: str, arb_type: str, buckets: list[str]) -> dict[str, Any]:
    return {
        "timestamp_utc": _utc_now_iso(),
        "event_ticker": event_ticker,
        "type": arb_type,
        "buckets": buckets,
    }


def scan_bucket_arbitrage(
    series_ticker: str,
    kalshi_client: Any,
    markets: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Detect and log simple bucket arbitrage-like conditions.

    Checks:
    1) Sum YES asks across buckets < 100c  -> buy_all_yes
    2) Sum YES bids across buckets > 100c -> sell_all_yes
    3) Within bucket YES bid + NO bid > 100c -> within_bucket
    4) Within bucket YES bid > YES ask -> stale_price
    """
    if markets is None:
        markets = kalshi_client.get_markets(series_ticker=series_ticker, status="open")
    if not markets:
        return []

    opportunities: list[dict[str, Any]] = []
    grouped = _group_by_event(markets)

    for event_ticker, event_markets in grouped.items():
        active_markets = [m for m in event_markets if m.yes_bid is not None or m.yes_ask is not None]
        if not active_markets:
            continue

        yes_asks = [m.yes_ask for m in active_markets if m.yes_ask is not None]
        yes_bids = [m.yes_bid for m in active_markets if m.yes_bid is not None]

        # Buy-all-YES check
        if len(yes_asks) == len(active_markets):
            sum_yes_asks = int(sum(yes_asks))
            if sum_yes_asks < 100:
                gross_cents = 100 - sum_yes_asks
                fees_dollars = sum(compute_fee(price_cents=ask, count=1, is_maker=False) for ask in yes_asks)
                fees_cents = round(fees_dollars * 100, 2)
                rec = _base_record(event_ticker, "buy_all_yes", [m.ticker for m in active_markets])
                rec.update({
                    "sum_yes_asks": round(sum_yes_asks / 100.0, 4),
                    "guaranteed_profit_cents": gross_cents,
                    "max_contracts": min((m.volume_24h or 0) for m in active_markets) or None,
                    "fees_estimate": fees_cents,
                    "net_profit": round(gross_cents - fees_cents, 2),
                })
                opportunities.append(rec)

        # Sell-all-YES check
        if len(yes_bids) == len(active_markets):
            sum_yes_bids = int(sum(yes_bids))
            if sum_yes_bids > 100:
                gross_cents = sum_yes_bids - 100
                fees_dollars = sum(compute_fee(price_cents=bid, count=1, is_maker=False) for bid in yes_bids)
                fees_cents = round(fees_dollars * 100, 2)
                rec = _base_record(event_ticker, "sell_all_yes", [m.ticker for m in active_markets])
                rec.update({
                    "sum_yes_bids": round(sum_yes_bids / 100.0, 4),
                    "guaranteed_profit_cents": gross_cents,
                    "max_contracts": min((m.volume_24h or 0) for m in active_markets) or None,
                    "fees_estimate": fees_cents,
                    "net_profit": round(gross_cents - fees_cents, 2),
                })
                opportunities.append(rec)

        # Within-bucket checks
        for market in active_markets:
            if market.yes_bid is not None and market.no_bid is not None:
                combo = int(market.yes_bid + market.no_bid)
                if combo > 100:
                    gross_cents = combo - 100
                    fees_dollars = compute_fee(market.yes_bid, 1, is_maker=False) + \
                        compute_fee(market.no_bid, 1, is_maker=False)
                    fees_cents = round(fees_dollars * 100, 2)
                    rec = _base_record(event_ticker, "within_bucket", [market.ticker])
                    rec.update({
                        "yes_bid": market.yes_bid,
                        "no_bid": market.no_bid,
                        "guaranteed_profit_cents": gross_cents,
                        "max_contracts": market.volume_24h or None,
                        "fees_estimate": fees_cents,
                        "net_profit": round(gross_cents - fees_cents, 2),
                    })
                    opportunities.append(rec)

            if market.yes_bid is not None and market.yes_ask is not None and market.yes_bid > market.yes_ask:
                rec = _base_record(event_ticker, "stale_price", [market.ticker])
                rec.update({
                    "yes_bid": market.yes_bid,
                    "yes_ask": market.yes_ask,
                    "crossed_by_cents": market.yes_bid - market.yes_ask,
                    "max_contracts": market.volume_24h or None,
                    "fees_estimate": None,
                    "net_profit": None,
                })
                opportunities.append(rec)

    for record in opportunities:
        _append_jsonl(Path(config.ARB_OPPORTUNITIES_PATH), record)

    if opportunities:
        logger.info("Arb scanner: detected %d opportunity rows for %s", len(opportunities), series_ticker)

    return opportunities


def check_cross_day_dependencies(
    kalshi_client: Any,
    series_ticker: str = config.SERIES_TICKER,
    markets: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Log cross-day pricing observations for adjacent event dates.

    Observation-only; no execution.
    """
    if markets is None:
        markets = kalshi_client.get_markets(series_ticker=series_ticker, status="open")
    if not markets:
        return []

    grouped = _group_by_event(markets)
    event_midpoints: dict[str, float] = {}
    for event_ticker, event_markets in grouped.items():
        mids = []
        for market in event_markets:
            if market.yes_bid is not None and market.yes_ask is not None:
                mids.append((market.yes_bid + market.yes_ask) / 2.0)
            elif market.yes_ask is not None:
                mids.append(float(market.yes_ask))
            elif market.yes_bid is not None:
                mids.append(float(market.yes_bid))
        if mids:
            event_midpoints[event_ticker] = sum(mids) / len(mids)

    records: list[dict[str, Any]] = []
    sorted_events = sorted(event_midpoints.keys())
    for idx in range(len(sorted_events) - 1):
        today_evt = sorted_events[idx]
        next_evt = sorted_events[idx + 1]
        rec = {
            "timestamp_utc": _utc_now_iso(),
            "series_ticker": series_ticker,
            "today_event": today_evt,
            "tomorrow_event": next_evt,
            "avg_midpoint_today_cents": round(event_midpoints[today_evt], 3),
            "avg_midpoint_tomorrow_cents": round(event_midpoints[next_evt], 3),
            "delta_midpoint_cents": round(event_midpoints[next_evt] - event_midpoints[today_evt], 3),
            "type": "cross_day_observation",
        }
        records.append(rec)
        _append_jsonl(Path(config.CROSS_DAY_OBSERVATIONS_PATH), rec)

    return records

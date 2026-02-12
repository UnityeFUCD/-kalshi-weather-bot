"""
bootstrap_datasets.py -- One-shot local dataset bootstrap for backtests.

Creates missing local files used by robustness/backtest scripts:
- data/raw/kalshi/KXHIGHNY_trades.parquet
- data/raw/weather/observations/USW00094728_daily.parquet (via fetch_ghcn_daily.py)

Usage:
  py bootstrap_datasets.py --days 120
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

import config

log = logging.getLogger("bootstrap")
BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_markets(series_ticker: str, days_back: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    min_close_ts = int((now - timedelta(days=days_back)).timestamp())
    max_close_ts = int((now + timedelta(days=2)).timestamp())

    markets: list[dict] = []
    cursor = None
    seen = set()

    for _ in range(250):
        params = {
            "series_ticker": series_ticker,
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASE}/markets", params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        markets.extend(j.get("markets", []))
        nxt = j.get("cursor")
        if not nxt or nxt in seen:
            break
        seen.add(nxt)
        cursor = nxt

    return markets


def fetch_trades_for_ticker(ticker: str, min_ts: int) -> list[dict]:
    rows: list[dict] = []
    cursor = None
    seen = set()

    for _ in range(800):
        params = {"ticker": ticker, "min_ts": min_ts, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASE}/markets/trades", params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        trades = j.get("trades", [])
        rows.extend(trades)
        nxt = j.get("cursor")
        if not nxt or nxt in seen:
            break
        seen.add(nxt)
        cursor = nxt

    return rows


def build_trades_parquet(series_ticker: str, days_back: int) -> Path | None:
    since_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    min_ts = int(since_dt.timestamp())

    markets = fetch_markets(series_ticker, days_back)
    tickers = sorted({m.get("ticker") for m in markets if m.get("ticker") and m.get("volume", 0) > 0})
    log.info("Markets with volume>0: %d", len(tickers))

    all_trades: list[dict] = []
    for i, t in enumerate(tickers, 1):
        try:
            tr = fetch_trades_for_ticker(t, min_ts)
            all_trades.extend(tr)
            if i % 10 == 0:
                log.info("Fetched %d/%d tickers; trades so far=%d", i, len(tickers), len(all_trades))
        except Exception as e:
            log.warning("Ticker %s failed: %s", t, e)

    if not all_trades:
        log.warning("No trades fetched for %s", series_ticker)
        return None

    recs = []
    for t in all_trades:
        created = t.get("created_time") or t.get("createdTime")
        created_dt = pd.to_datetime(created, utc=True, errors="coerce")
        recs.append(
            {
                "trade_id": t.get("id", ""),
                "ticker": t.get("ticker", ""),
                "event_ticker": t.get("event_ticker", ""),
                "market_ticker": t.get("market_ticker", ""),
                "created_time": created,
                "created_dt": created_dt,
                "trade_date": created_dt.date() if not pd.isna(created_dt) else None,
                "yes_price": t.get("yes_price"),
                "no_price": t.get("no_price"),
                "count": t.get("count", 0),
                "taker_side": t.get("taker_side", ""),
            }
        )

    df = pd.DataFrame(recs).drop_duplicates(subset=["trade_id"])
    out_dir = config.PROJECT_ROOT / "data" / "raw" / "kalshi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{series_ticker}_trades.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved %s (%d trades)", out, len(df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXHIGHNY")
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    obs = config.PROJECT_ROOT / "data" / "raw" / "weather" / "observations" / "USW00094728_daily.parquet"
    if not obs.exists():
        raise SystemExit("Missing observations parquet; run fetch_ghcn_daily.py first")

    build_trades_parquet(args.series, args.days)


if __name__ == "__main__":
    main()

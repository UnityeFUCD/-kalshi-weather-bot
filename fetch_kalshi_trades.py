"""
scripts/fetch_kalshi_trades.py -- Pull Historical Kalshi Trade Prints

Uses the official Kalshi API "Get Trades" endpoint to download
historical trade prints for weather markets.

This gives us:
- What prices contracts traded at over time
- Volume patterns (when do people trade?)
- Taker side (who's buying yes vs no?)

Combined with forecast data, this lets us backtest our model.

Usage:
    cd C:\\Users\\Unitye\\Desktop\\kalshi-weather-bot
    py scripts\\fetch_kalshi_trades.py
    py scripts\\fetch_kalshi_trades.py --days 90 --series KXHIGHNY
"""

import sys
import time
import argparse
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_trades_page(auth, base_url, api_path, ticker=None, cursor=None,
                      min_ts=None, max_ts=None, limit=1000):
    """
    Fetch one page of trades from Kalshi API.
    
    Endpoint: GET /trade-api/v2/markets/trades
    Docs: https://docs.kalshi.com/api-reference/market/get-trades
    """
    import requests
    
    path = "%s/markets/trades" % api_path
    params = {"limit": limit}
    
    if ticker:
        params["ticker"] = ticker
    if cursor:
        params["cursor"] = cursor
    if min_ts:
        params["min_ts"] = min_ts
    if max_ts:
        params["max_ts"] = max_ts
    
    url = "%s%s" % (base_url, path)
    headers = auth.headers("GET", path)
    
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all_trades(auth, base_url, api_path, series_ticker, days_back=90):
    """
    Paginate through all trades for a series within the date range.
    
    Returns list of trade dicts.
    """
    since_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    min_ts = int(since_dt.timestamp())   # ✅ integer unix seconds
    max_ts = int(datetime.now(timezone.utc).timestamp())
    
    logger.info("Fetching trades for %s (since %s)...", series_ticker, since_dt.strftime("%Y-%m-%d"))
    
    all_trades = []
    cursor = None
    page = 0
    
    while True:
        page += 1
        try:
            # The API filters by ticker prefix, so we pass the series ticker
            # and it returns trades for all markets in that series
            data = fetch_trades_page(
                auth, base_url, api_path,
                ticker=series_ticker,
                cursor=cursor,
                min_ts=min_ts,
                max_ts=max_ts,
                limit=1000
            )
        except Exception as e:
            logger.error("Page %d failed: %s", page, e)
            # If ticker filter doesn't work as prefix, try without it
            if page == 1 and "ticker" in str(e).lower():
                logger.info("Retrying without ticker filter...")
                data = fetch_trades_page(
                    auth, base_url, api_path,
                    cursor=cursor,
                    min_ts=min_ts,
                    max_ts=max_ts,
                    limit=1000
                )
            else:
                break
        
        trades = data.get("trades", [])
        cursor = data.get("cursor")
        
        if not trades:
            logger.info("  Page %d: no more trades", page)
            break
        
        # Filter to our series (API might return broader results)
        series_trades = [t for t in trades if series_ticker in t.get("ticker", "")]
        all_trades.extend(series_trades)
        
        logger.info("  Page %d: %d trades (%d match %s) | total so far: %d",
                    page, len(trades), len(series_trades), series_ticker, len(all_trades))
        
        if not cursor:
            logger.info("  No more pages (cursor is None)")
            break
        
        # Rate limit: be respectful
        time.sleep(0.3)
    
    logger.info("Total trades fetched: %d", len(all_trades))
    return all_trades


def trades_to_parquet(trades, series_ticker, output_dir):
    """Convert trade list to a clean Parquet file."""
    import pandas as pd
    
    if not trades:
        logger.warning("No trades to save!")
        return None
    
    # Normalize the trade records
    records = []
    for t in trades:
        records.append({
            "trade_id": t.get("id", ""),
            "ticker": t.get("ticker", ""),
            "series_ticker": series_ticker,
            "event_ticker": t.get("event_ticker", ""),
            "yes_price": t.get("yes_price"),        # cents
            "no_price": t.get("no_price"),           # cents
            "count": t.get("count", 0),
            "taker_side": t.get("taker_side", ""),   # "yes" or "no"
            "created_time": t.get("created_time", ""),
        })
    
    df = pd.DataFrame(records)
    
    # Parse timestamps
    df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
    df["trade_date"] = df["created_time"].dt.date
    
    # Add ingestion metadata
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = "%s_trades.parquet" % series_ticker
    filepath = output_dir / filename
    df.to_parquet(filepath, index=False)
    
    size_kb = filepath.stat().st_size / 1024
    logger.info("Saved: %s (%.1f KB, %d trades)", filepath.name, size_kb, len(df))
    
    return filepath


def analyze_trades(parquet_path):
    """Quick analysis of the downloaded trades."""
    import duckdb
    
    logger.info("\nTrade Analysis:")
    con = duckdb.connect()
    
    # Basic stats
    result = con.execute("""
        SELECT 
            count(*) as total_trades,
            sum(count) as total_contracts,
            min(created_time) as first_trade,
            max(created_time) as last_trade,
            count(distinct ticker) as unique_markets,
            count(distinct trade_date) as trading_days
        FROM '%s'
    """ % parquet_path).fetchone()
    
    logger.info("  Total trades:    %d", result[0])
    logger.info("  Total contracts: %d", result[1])
    logger.info("  Date range:      %s to %s", str(result[2])[:10], str(result[3])[:10])
    logger.info("  Unique markets:  %d", result[4])
    logger.info("  Trading days:    %d", result[5])
    
    # Volume by day
    print("\n  Daily volume (last 14 days):")
    print("  %-12s %8s %8s %8s" % ("Date", "Trades", "Contracts", "Avg Price"))
    print("  " + "-" * 40)
    
    rows = con.execute("""
        SELECT 
            trade_date,
            count(*) as trades,
            sum(count) as contracts,
            round(avg(yes_price), 1) as avg_price
        FROM '%s'
        GROUP BY trade_date
        ORDER BY trade_date DESC
        LIMIT 14
    """ % parquet_path).fetchall()
    
    for row in rows:
        print("  %-12s %8d %8d %7.1fc" % (row[0], row[1], row[2], row[3] or 0))
    
    # Taker side distribution
    result = con.execute("""
        SELECT 
            taker_side,
            count(*) as trades,
            sum(count) as contracts
        FROM '%s'
        GROUP BY taker_side
    """ % parquet_path).fetchall()
    
    print("\n  Taker side distribution:")
    for row in result:
        print("    %s: %d trades, %d contracts" % (row[0] or "unknown", row[1], row[2]))
    
    # Price distribution
    result = con.execute("""
        SELECT 
            CASE 
                WHEN yes_price <= 10 THEN '1-10c'
                WHEN yes_price <= 25 THEN '11-25c'
                WHEN yes_price <= 50 THEN '26-50c'
                WHEN yes_price <= 75 THEN '51-75c'
                WHEN yes_price <= 90 THEN '76-90c'
                ELSE '91-99c'
            END as price_band,
            count(*) as trades
        FROM '%s'
        WHERE yes_price IS NOT NULL
        GROUP BY price_band
        ORDER BY price_band
    """ % parquet_path).fetchall()
    
    print("\n  Price distribution:")
    for row in result:
        print("    %-10s %d trades" % (row[0], row[1]))
    
    con.close()
    logger.info("\n  ✅ Trade analysis complete")


def list_markets_for_series(series_ticker: str, days_back: int) -> list[dict]:
    """Return market objects for a series in a close-time window."""
    now = datetime.now(timezone.utc)
    min_close_ts = int((now - timedelta(days=days_back)).timestamp())
    max_close_ts = int((now + timedelta(days=2)).timestamp())

    markets = []
    cursor = None
    seen_cursors = set()
    page = 0
    while True:
        page += 1
        params = {
            "series_ticker": series_ticker,
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        r = requests.get(f"{BASE}/markets", params=params, timeout=30)
        r.raise_for_status()
        j = r.json()

        markets.extend(j.get("markets", []))
        next_cursor = j.get("cursor")

        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            logger.warning("Stopping market pagination on repeated cursor: %s", next_cursor)
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

        if page >= 200:
            logger.warning("Stopping market pagination at safety page cap (%d)", page)
            break

    return markets


def fetch_trades_for_market(market_ticker: str, min_ts: int) -> list[dict]:
    """Fetch all trades for one market ticker since min_ts."""
    trades = []
    cursor = None
    seen_cursors = set()
    page = 0
    while True:
        page += 1
        params = {"ticker": market_ticker, "min_ts": min_ts, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        r = requests.get(f"{BASE}/markets/trades", params=params, timeout=30)
        r.raise_for_status()
        j = r.json()

        trades.extend(j.get("trades", []))
        next_cursor = j.get("cursor")
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            logger.warning("Stopping trade pagination on repeated cursor for %s", market_ticker)
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

        if page >= 500:
            logger.warning("Stopping trade pagination at safety page cap (%d) for %s", page, market_ticker)
            break

    return trades


def main():
    parser = argparse.ArgumentParser(description="Fetch Kalshi trade history")
    parser.add_argument("--series", default="KXHIGHNY", help="Series ticker (default: KXHIGHNY)")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    args = parser.parse_args()
    
    # Load existing auth
    import config
    
    logger.info("=" * 60)
    logger.info("KALSHI TRADE HISTORY DOWNLOAD")
    logger.info("=" * 60)
    logger.info("Series: %s", args.series)
    logger.info("Days back: %d", args.days)
    logger.info("API: %s", config.KALSHI_PROD_URL)
    print()
    
    # Fetch trades using new market-based approach
    series = args.series
    days_back = args.days
    
    since_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    min_ts = int(since_dt.timestamp())
    
    markets = list_markets_for_series(series, days_back)
    
    # optional: skip zero-volume markets to reduce API calls
    market_tickers = [m["ticker"] for m in markets if m.get("volume", 0) > 0 and "ticker" in m]
    
    all_trades = []
    for t in market_tickers:
        all_trades.extend(fetch_trades_for_market(t, min_ts))
    
    print(f"Markets (volume>0): {len(market_tickers)}")
    print(f"Total trades fetched: {len(all_trades)}")
    
    trades = all_trades
    
    if not trades:
        logger.warning("No trades found. This could mean:")
        logger.warning("  - The series ticker doesn't match any trades")
        logger.warning("  - The API's trade history doesn't go back %d days", args.days)
        logger.warning("  - Rate limiting or auth issue")
        logger.warning("Try: py scripts\\fetch_kalshi_trades.py --days 30")
        return
    
    # Save to Parquet
    output_dir = PROJECT_ROOT / "data" / "raw" / "kalshi" / "trades"
    parquet_path = trades_to_parquet(trades, args.series, output_dir)
    
    if parquet_path:
        # Analyze
        analyze_trades(parquet_path)
        
        print("\n" + "=" * 60)
        print("DONE! Trade history saved to:")
        print("  %s" % parquet_path)
        print("\nNext step: py scripts\\build_calibration_dataset.py")
        print("=" * 60)


if __name__ == "__main__":
    main()

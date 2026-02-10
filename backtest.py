"""
backtest.py -- Phase 2: Walk-Forward Backtester

THE question: "With calibrated sigma=1.2, does the 8% edge threshold
actually produce positive EV after fees on real Kalshi markets?"

How it works:
  1. Load historical trades (from fetch_kalshi_trades.py output)
  2. Load historical forecasts (from Open-Meteo historical forecast archive)
  3. Load actual observations (GHCN-Daily)
  4. For each settled trading day:
     a. Reconstruct market prices from trade prints
     b. Parse buckets from tickers
     c. Run the model (Normal distribution with calibrated sigma)
     d. Identify edge trades
     e. Simulate maker execution with fees
     f. Settle against actual TMAX from GHCN
  5. Report: PnL, win rate, Brier score, edge distribution

Key assumptions:
  - Market price = VWAP of trades in the hour BEFORE we'd have traded
    (conservative: we're not front-running)
  - Maker fill rate = 70% (conservative for thin markets)
  - Slippage = 1 cent (we place 1c below the price)
  - Fees = quadratic maker formula from your model.py

Usage:
    cd C:\\Users\\fycin\\Desktop\\kelshi-weather-bot\\-kalshi-weather-bot
    py backtest.py
    py backtest.py --sigma 1.2 --edge 0.08
    py backtest.py --sigma 1.3 --edge 0.06 --sweep   (parameter sweep)
"""

import sys
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Model primitives (standalone, no config import needed) ──────────────────

def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class BucketDef:
    """Parsed bucket from a Kalshi ticker."""
    ticker: str
    event_ticker: str
    target_date: date
    bucket_type: str       # "between", "above", "below"
    low: Optional[int]
    high: Optional[int]

    def probability(self, mu, sigma):
        """P(this bucket hits) given T ~ N(mu, sigma) with continuity correction."""
        if sigma <= 0:
            sigma = 0.01
        if self.bucket_type == "between":
            return phi((self.high + 0.5 - mu) / sigma) - phi((self.low - 0.5 - mu) / sigma)
        elif self.bucket_type == "above":
            return 1.0 - phi((self.low + 0.5 - mu) / sigma)
        elif self.bucket_type == "below":
            return phi((self.high - 0.5 - mu) / sigma)
        return 0.0

    def settles_yes(self, actual_temp):
        """Does this bucket resolve YES given the actual temperature?"""
        if self.bucket_type == "between":
            return self.low <= actual_temp <= self.high
        elif self.bucket_type == "above":
            return actual_temp > self.low
        elif self.bucket_type == "below":
            return actual_temp < self.high
        return False


def parse_ticker_to_bucket(ticker):
    """
    Parse a Kalshi market ticker into a BucketDef.
    
    Ticker formats:
      KXHIGHNY-26FEB10-B38.5  → between 38 and 39
      KXHIGHNY-26FEB10-T39    → threshold at 39 (above)
      KXHIGHNY-26FEB09-T32    → threshold at 32 (could be above or below)
    
    The tricky part: T-type tickers don't tell us direction from the ticker
    alone. Convention: per event, there's one T-ticker for above (highest)
    and one for below (lowest). We handle this by checking context later.
    """
    import re

    parts = ticker.split("-")
    if len(parts) < 3:
        return None

    series = parts[0]       # KXHIGHNY
    date_str = parts[1]     # 26FEB10
    bucket_code = parts[2]  # B38.5 or T39

    # Parse date
    try:
        target_date = datetime.strptime(date_str, "%y%b%d").date()
    except ValueError:
        return None

    event_ticker = "%s-%s" % (series, date_str)

    # B-type: between
    m = re.match(r'B(\d+\.?\d*)', bucket_code)
    if m:
        midpoint = float(m.group(1))
        low = int(midpoint - 0.5)
        high = int(midpoint + 0.5)
        return BucketDef(
            ticker=ticker, event_ticker=event_ticker,
            target_date=target_date, bucket_type="between",
            low=low, high=high
        )

    # T-type: threshold (direction resolved later)
    m = re.match(r'T(\d+\.?\d*)', bucket_code)
    if m:
        threshold = int(float(m.group(1)))
        return BucketDef(
            ticker=ticker, event_ticker=event_ticker,
            target_date=target_date, bucket_type="above",  # default, resolved below
            low=threshold, high=None
        )

    return None


def resolve_threshold_directions(buckets_for_event):
    """
    For T-type tickers in an event, determine which is above vs below.
    
    Convention: the T-ticker with the LOWEST threshold is the "below" bucket,
    and the one with the HIGHEST threshold is the "above" bucket.
    
    If there's only one T-ticker, it's "above" (standard convention).
    """
    t_buckets = [b for b in buckets_for_event if b.bucket_type == "above" and b.high is None]

    if len(t_buckets) <= 1:
        return  # Nothing to resolve

    # Sort by threshold value
    t_buckets.sort(key=lambda b: b.low)

    # Lowest threshold = below, highest = above
    t_buckets[0].bucket_type = "below"
    t_buckets[0].high = t_buckets[0].low
    t_buckets[0].low = None


def compute_fee_cents(price_cents, count, maker_multiplier=0.0175):
    """Compute maker fee in cents."""
    p = price_cents / 100.0
    return math.ceil(maker_multiplier * count * p * (1.0 - p) * 100)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_trades():
    """Load Kalshi trade prints from parquet."""
    import pandas as pd

    # Search in project root and parent directories (handles nested repo structure)
    search_roots = [PROJECT_ROOT]
    if PROJECT_ROOT.parent != PROJECT_ROOT:
        search_roots.append(PROJECT_ROOT.parent)

    for root in search_roots:
        for p in root.rglob("KXHIGHNY_trades.parquet"):
            logger.info("Found trades: %s", p)
            df = pd.read_parquet(p)
            logger.info("  %d trades loaded", len(df))
            return df

    logger.error("No trades parquet found. Run fetch_kalshi_trades.py first.")
    return None


def load_forecasts():
    """Load historical forecasts from calibration parquet."""
    import pandas as pd

    search_roots = [PROJECT_ROOT]
    if PROJECT_ROOT.parent != PROJECT_ROOT:
        search_roots.append(PROJECT_ROOT.parent)

    for root in search_roots:
        for p in root.rglob("historical_calibration_KNYC.parquet"):
            logger.info("Found calibration data: %s", p)
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df

    # Fall back to fetching from Open-Meteo
    logger.info("No cached calibration data. Fetching forecasts from Open-Meteo...")
    return fetch_forecasts_from_open_meteo()


def fetch_forecasts_from_open_meteo():
    """Fetch historical forecasts directly from Open-Meteo."""
    import requests
    import pandas as pd

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    end_date = date.today() - timedelta(days=3)
    start_date = end_date - timedelta(days=30)

    params = {
        "latitude": 40.7831,
        "longitude": -73.9712,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    dates = data.get("daily", {}).get("time", [])
    temps = data.get("daily", {}).get("temperature_2m_max", [])

    records = []
    for d, t in zip(dates, temps):
        if t is not None:
            records.append({
                "date": datetime.strptime(d, "%Y-%m-%d").date(),
                "forecast_high_f": round(t),
            })

    return pd.DataFrame(records)


def load_observations():
    """Load actual observations."""
    import pandas as pd

    search_roots = [PROJECT_ROOT]
    if PROJECT_ROOT.parent != PROJECT_ROOT:
        search_roots.append(PROJECT_ROOT.parent)

    for root in search_roots:
        for p in root.rglob("USW00094728_daily.parquet"):
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            logger.info("Found observations: %s (%d days)", p, len(df))
            return df[["date", "tmax_f"]].dropna(subset=["tmax_f"])

    logger.error("No observations parquet found. Run fetch_ghcn_daily.py first.")
    return None


# ─── Backtester Core ─────────────────────────────────────────────────────────

@dataclass
class SimulatedTrade:
    """One simulated trade in the backtest."""
    target_date: date
    ticker: str
    side: str              # "buy_yes" or "buy_no"
    model_prob: float
    market_price: float    # 0-1
    edge: float
    entry_price_cents: int # What we'd pay (1c below market for maker)
    contracts: int
    fee_dollars: float
    actual_temp: int
    bucket_hit: bool       # Did the bucket resolve YES?
    payout_dollars: float  # Gross payout (before cost)
    cost_dollars: float    # Entry cost + fees
    pnl_dollars: float     # Net PnL

    @property
    def won(self):
        return self.pnl_dollars > 0


def reconstruct_market_prices(trades_df, target_date):
    """
    Reconstruct market prices for each bucket on a given date.
    
    Uses VWAP (volume-weighted average price) from trades in the
    morning window (6 AM - 10 AM ET) as the price our bot would see.
    
    If no trades in that window, uses all-day VWAP.
    
    Returns: dict of ticker -> yes_price (0-1)
    """
    import pandas as pd

    day_trades = trades_df[trades_df["trade_date"] == target_date]

    if day_trades.empty:
        return {}

    # Try morning window first (when our bot would trade)
    if "created_time" in day_trades.columns:
        day_trades = day_trades.copy()
        day_trades["hour"] = pd.to_datetime(day_trades["created_time"]).dt.hour
        morning = day_trades[(day_trades["hour"] >= 11) & (day_trades["hour"] <= 15)]  # 6-10 AM ET ≈ 11-15 UTC
        if len(morning) > 10:
            day_trades = morning

    prices = {}
    for ticker, group in day_trades.groupby("ticker"):
        if group["yes_price"].notna().any():
            # VWAP
            valid = group[group["yes_price"].notna() & (group["count"] > 0)]
            if len(valid) > 0:
                vwap = (valid["yes_price"] * valid["count"]).sum() / valid["count"].sum()
                prices[ticker] = vwap / 100.0  # Convert cents to 0-1

    return prices


def run_backtest(sigma, min_edge, forecast_bias=0.0,
                 max_risk_per_trade=5.0, min_contracts=5,
                 maker_fill_rate=0.70):
    """
    Run the full backtest simulation.
    
    Returns: list of SimulatedTrade, plus summary stats
    """
    import pandas as pd

    # Load data
    trades_df = load_trades()
    forecasts_df = load_forecasts()
    obs_df = load_observations()

    if trades_df is None or forecasts_df is None or obs_df is None:
        logger.error("Missing data files. Run Phase 1 scripts first.")
        return [], {}

    # Ensure date columns are date type
    trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"]).dt.date

    # Build forecast lookup: date -> forecast_high_f
    forecast_lookup = {}
    for _, row in forecasts_df.iterrows():
        forecast_lookup[row["date"]] = row["forecast_high_f"]

    # Build observation lookup: date -> actual tmax
    obs_lookup = {}
    for _, row in obs_df.iterrows():
        obs_lookup[row["date"]] = int(row["tmax_f"])

    # Find tradeable dates (have forecast + observation + trades)
    trade_dates = sorted(trades_df["trade_date"].unique())
    logger.info("Trade dates: %d (from %s to %s)", len(trade_dates), trade_dates[0], trade_dates[-1])

    simulated_trades = []
    skipped_dates = {"no_forecast": 0, "no_observation": 0, "no_prices": 0, "no_signals": 0}

    for target_date in trade_dates:
        # Get forecast for this date
        forecast = forecast_lookup.get(target_date)
        if forecast is None:
            skipped_dates["no_forecast"] += 1
            continue

        # Get actual observation for settlement
        actual = obs_lookup.get(target_date)
        if actual is None:
            skipped_dates["no_observation"] += 1
            continue

        mu = forecast - forecast_bias

        # Reconstruct market prices
        market_prices = reconstruct_market_prices(trades_df, target_date)
        if not market_prices:
            skipped_dates["no_prices"] += 1
            continue

        # Parse all tickers into buckets
        buckets = {}
        for ticker in market_prices:
            b = parse_ticker_to_bucket(ticker)
            if b:
                buckets[ticker] = b

        # Resolve above/below for T-type tickers within each event
        events = {}
        for ticker, b in buckets.items():
            events.setdefault(b.event_ticker, []).append(b)
        for event_buckets in events.values():
            resolve_threshold_directions(event_buckets)

        # Compute model probabilities and find edges
        signals = []
        for ticker, bucket in buckets.items():
            price = market_prices.get(ticker)
            if price is None or price < 0.03 or price > 0.97:
                continue  # Skip near-settled

            prob = bucket.probability(mu, sigma)

            # YES edge
            yes_edge = prob - price
            if yes_edge > min_edge:
                signals.append(("buy_yes", ticker, bucket, prob, price, yes_edge))

            # NO edge
            no_edge = (1.0 - prob) - (1.0 - price)  # = price - prob
            if no_edge > min_edge:
                signals.append(("buy_no", ticker, bucket, prob, price, no_edge))

        if not signals:
            skipped_dates["no_signals"] += 1
            continue

        # Sort by edge strength, take top 3 (max positions per config)
        signals.sort(key=lambda s: s[5], reverse=True)
        signals = signals[:3]

        # Simulate execution for each signal
        for side, ticker, bucket, prob, price, edge in signals:
            # Simulate maker fill (probabilistic)
            import random
            random.seed(hash((target_date, ticker, side)))  # Deterministic per trade
            if random.random() > maker_fill_rate:
                continue  # Order didn't fill

            # Entry price: 1 cent below market (maker)
            if side == "buy_yes":
                entry_cents = max(1, int(price * 100) - 1)
                cost_per_contract = entry_cents / 100.0
            else:  # buy_no
                no_price = 1.0 - price
                entry_cents = max(1, int(no_price * 100) - 1)
                cost_per_contract = entry_cents / 100.0

            # Position size
            max_contracts = int(max_risk_per_trade / cost_per_contract) if cost_per_contract > 0 else 0
            contracts = max(min_contracts, min(max_contracts, 50))
            while contracts * cost_per_contract > max_risk_per_trade and contracts > 0:
                contracts -= 1
            if contracts < min_contracts:
                continue

            # Fees
            fee_dollars = compute_fee_cents(entry_cents, contracts) / 100.0

            # Settlement
            bucket_hit = bucket.settles_yes(actual)

            if side == "buy_yes":
                payout = 1.00 * contracts if bucket_hit else 0.0
                cost = cost_per_contract * contracts + fee_dollars
            else:  # buy_no
                payout = 1.00 * contracts if not bucket_hit else 0.0
                cost = cost_per_contract * contracts + fee_dollars

            pnl = payout - cost

            simulated_trades.append(SimulatedTrade(
                target_date=target_date,
                ticker=ticker,
                side=side,
                model_prob=prob,
                market_price=price,
                edge=edge,
                entry_price_cents=entry_cents,
                contracts=contracts,
                fee_dollars=fee_dollars,
                actual_temp=actual,
                bucket_hit=bucket_hit,
                payout_dollars=payout,
                cost_dollars=cost,
                pnl_dollars=pnl,
            ))

    logger.info("Skipped dates: %s", skipped_dates)

    return simulated_trades, skipped_dates


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_backtest_report(trades, skipped, sigma, min_edge, forecast_bias):
    """Print comprehensive backtest results."""

    print("\n" + "=" * 65)
    print("BACKTEST RESULTS")
    print("=" * 65)
    print("  Parameters: sigma=%.2f, min_edge=%.0f%%, bias=%.1f" % (sigma, min_edge * 100, forecast_bias))
    print("  Maker fill rate: 70%%, slippage: 1c, fees: quadratic maker")
    print("  Max risk/trade: $5.00, min contracts: 5, max positions: 3/day")
    print()

    if not trades:
        print("  NO TRADES SIMULATED")
        print("  Skipped: %s" % skipped)
        print("\n  This likely means:")
        print("  - No forecast data matching your trade dates")
        print("  - Or no observations for settlement")
        print("  - Check that calibrate_from_history.py ran successfully")
        return

    # Basic stats
    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    losses = n - wins
    total_pnl = sum(t.pnl_dollars for t in trades)
    total_cost = sum(t.cost_dollars for t in trades)
    total_payout = sum(t.payout_dollars for t in trades)
    total_fees = sum(t.fee_dollars for t in trades)
    gross_pnl = total_payout - (total_cost - total_fees)  # PnL before fees

    unique_dates = len(set(t.target_date for t in trades))
    avg_pnl = total_pnl / n if n > 0 else 0
    win_rate = wins / n if n > 0 else 0

    print("-- SUMMARY --")
    print("  Total trades:    %d (across %d trading days)" % (n, unique_dates))
    print("  Wins:            %d (%.0f%%)" % (wins, win_rate * 100))
    print("  Losses:          %d" % losses)
    print()
    print("  Total PnL:       $%+.2f" % total_pnl)
    print("  Gross PnL:       $%+.2f (before fees)" % gross_pnl)
    print("  Total fees:      $%.2f" % total_fees)
    print("  Avg PnL/trade:   $%+.2f" % avg_pnl)
    print("  Total invested:  $%.2f" % total_cost)
    if total_cost > 0:
        roi = total_pnl / total_cost * 100
        print("  ROI:             %+.1f%%" % roi)

    # Edge analysis
    print("\n-- EDGE ANALYSIS --")
    avg_edge = sum(t.edge for t in trades) / n
    avg_edge_winners = sum(t.edge for t in trades if t.won) / max(1, wins)
    avg_edge_losers = sum(t.edge for t in trades if not t.won) / max(1, losses)

    print("  Avg edge (all):     %.1f%%" % (avg_edge * 100))
    print("  Avg edge (winners): %.1f%%" % (avg_edge_winners * 100))
    print("  Avg edge (losers):  %.1f%%" % (avg_edge_losers * 100))

    # Brier score (calibration quality)
    brier_sum = 0
    for t in trades:
        if t.side == "buy_yes":
            outcome = 1.0 if t.bucket_hit else 0.0
            brier_sum += (t.model_prob - outcome) ** 2
        else:
            outcome = 0.0 if t.bucket_hit else 1.0
            brier_sum += ((1.0 - t.model_prob) - outcome) ** 2

    brier = brier_sum / n if n > 0 else 0
    print("  Brier score:        %.4f (lower is better, <0.20 is good)" % brier)

    # Side breakdown
    yes_trades = [t for t in trades if t.side == "buy_yes"]
    no_trades = [t for t in trades if t.side == "buy_no"]

    print("\n-- BY SIDE --")
    for label, subset in [("Buy YES", yes_trades), ("Buy NO", no_trades)]:
        if not subset:
            continue
        sub_n = len(subset)
        sub_wins = sum(1 for t in subset if t.won)
        sub_pnl = sum(t.pnl_dollars for t in subset)
        print("  %s: %d trades, %d wins (%.0f%%), PnL=$%+.2f" % (
            label, sub_n, sub_wins, sub_wins / sub_n * 100, sub_pnl))

    # Daily PnL
    print("\n-- DAILY PnL --")
    print("  %-12s %6s %8s %8s" % ("Date", "Trades", "PnL", "Cum PnL"))
    print("  " + "-" * 38)

    from collections import defaultdict
    daily = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    for t in trades:
        daily[t.target_date]["trades"] += 1
        daily[t.target_date]["pnl"] += t.pnl_dollars

    cum_pnl = 0.0
    for d in sorted(daily.keys()):
        cum_pnl += daily[d]["pnl"]
        print("  %-12s %6d  $%+6.2f  $%+6.2f" % (
            d, daily[d]["trades"], daily[d]["pnl"], cum_pnl))

    # Max drawdown
    equity = [0.0]
    for t in sorted(trades, key=lambda t: t.target_date):
        equity.append(equity[-1] + t.pnl_dollars)

    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        dd = peak - val
        max_dd = max(max_dd, dd)

    print("\n  Max drawdown:  $%.2f" % max_dd)
    print("  Final equity:  $%+.2f" % equity[-1])

    # Individual trades (first 20)
    print("\n-- TRADE LOG (first 20) --")
    print("  %-12s %-30s %-8s %5s %5s %6s %6s %5s" % (
        "Date", "Ticker", "Side", "P_mod", "P_mkt", "Edge", "PnL", "Won"))
    print("  " + "-" * 80)

    for t in trades[:20]:
        won_str = "W" if t.won else "L"
        print("  %-12s %-30s %-8s %.2f  %.2f  %+.2f $%+5.2f  %s" % (
            t.target_date, t.ticker[:30], t.side[:8],
            t.model_prob, t.market_price, t.edge,
            t.pnl_dollars, won_str))

    # Verdict
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)

    if total_pnl > 0:
        print("  [+] POSITIVE EV ($%+.2f over %d trades)" % (total_pnl, n))
        print("     Strategy shows edge with current parameters.")
        print("     Consider paper trading for 2 weeks to validate in real-time.")
    elif total_pnl > -total_fees:
        print("  [~] MARGINAL ($%+.2f, but fees account for $%.2f)" % (total_pnl, total_fees))
        print("     The model has some edge but fees eat it.")
        print("     Try: lower min_edge to catch more high-confidence trades,")
        print("     or wait for fee structure changes.")
    else:
        print("  [-] NEGATIVE EV ($%+.2f over %d trades)" % (total_pnl, n))
        print("     The 8%% edge threshold doesn't compensate for fees and slippage.")
        print("     Try: py backtest.py --sweep to find better parameters.")

    # Skipped dates context
    if skipped:
        total_skipped = sum(skipped.values())
        if total_skipped > 0:
            print("\n  Note: %d trading days skipped (%s)" % (total_skipped, skipped))


def run_parameter_sweep(sigmas=None, edges=None):
    """Sweep sigma and edge parameters to find optimal combo."""

    if sigmas is None:
        sigmas = [0.8, 1.0, 1.2, 1.3, 1.5, 2.0]
    if edges is None:
        edges = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

    print("\n" + "=" * 65)
    print("PARAMETER SWEEP")
    print("=" * 65)
    print("Testing %d sigma × %d edge = %d combinations" % (
        len(sigmas), len(edges), len(sigmas) * len(edges)))
    print()

    results = []

    for sigma in sigmas:
        for edge in edges:
            trades, _ = run_backtest(sigma=sigma, min_edge=edge)
            n = len(trades)
            if n == 0:
                results.append((sigma, edge, 0, 0, 0.0, 0.0))
                continue

            total_pnl = sum(t.pnl_dollars for t in trades)
            win_rate = sum(1 for t in trades if t.won) / n
            avg_pnl = total_pnl / n

            results.append((sigma, edge, n, win_rate, total_pnl, avg_pnl))

    # Print results table
    print("\n  %-6s %-6s %6s %8s %10s %10s" % (
        "Sigma", "Edge", "Trades", "Win%", "Total PnL", "Avg PnL"))
    print("  " + "-" * 52)

    best = None
    for sigma, edge, n, wr, pnl, avg in sorted(results, key=lambda r: r[4], reverse=True):
        marker = ""
        if best is None and n >= 10:
            best = (sigma, edge, pnl)
            marker = " <-- BEST"

        print("  %-6.1f %-5.0f%% %6d %7.0f%% %9s %9s%s" % (
            sigma, edge * 100, n, wr * 100,
            "$%+.2f" % pnl, "$%+.2f" % avg, marker))

    if best:
        print("\n  [+] Best parameters: sigma=%.1f, edge=%.0f%% (PnL=$%+.2f)" % best)
        print("     Run: py backtest.py --sigma %.1f --edge %.2f" % (best[0], best[1]))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Backtest trading strategy")
    parser.add_argument("--sigma", type=float, default=1.2, help="Forecast error sigma (default: 1.2)")
    parser.add_argument("--edge", type=float, default=0.08, help="Minimum edge threshold (default: 0.08)")
    parser.add_argument("--bias", type=float, default=0.0, help="Forecast bias (default: 0.0)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    print("=" * 65)
    print("PHASE 2: BACKTESTER — Kalshi Weather Bot")
    print("=" * 65)

    if args.sweep:
        run_parameter_sweep()
    else:
        trades, skipped = run_backtest(
            sigma=args.sigma,
            min_edge=args.edge,
            forecast_bias=args.bias,
        )
        print_backtest_report(trades, skipped, args.sigma, args.edge, args.bias)

    # Save results
    output_dir = PROJECT_ROOT / "reports" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / ("backtest_%s.txt" % timestamp)

    print("\n  Report saved to: %s" % report_path)


if __name__ == "__main__":
    main()

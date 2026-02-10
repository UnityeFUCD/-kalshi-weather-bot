"""
analyze_accuracy.py -- Model vs Market vs Reality

For each settled KXHIGHNY market day, answers:
  1. What did our model predict? (probability per bucket)
  2. What did the Kalshi market price? (VWAP from trades)
  3. What actually happened? (GHCN observed TMAX)
  4. Who was more accurate -- our model or the market?

This is the key question: does our model actually know something
the market doesn't, or are we just getting lucky?

Metrics:
  - Brier score (model vs market) -- lower is better
  - Calibration: when we say 50%, does it hit ~50% of the time?
  - Edge reality: were our "edge" trades actually mispriced?
"""

import sys
import math
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
from collections import defaultdict

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import Bucket, parse_bucket_title, phi
import config


def load_data():
    """Load trades, forecasts, and observations."""
    # Trades
    trades_path = PROJECT_ROOT / "data" / "raw" / "kalshi" / "trades" / "KXHIGHNY_trades.parquet"
    if not trades_path.exists():
        trades_path = PROJECT_ROOT.parent / "data" / "raw" / "kalshi" / "trades" / "KXHIGHNY_trades.parquet"
    trades = pd.read_parquet(trades_path)
    trades["created_time"] = pd.to_datetime(trades["created_time"], utc=True)
    print(f"Loaded {len(trades):,} trades")

    # Forecasts (Open-Meteo historical)
    cal_path = PROJECT_ROOT / "data" / "curated" / "calibration" / "historical_calibration_KNYC.parquet"
    forecasts = None
    if cal_path.exists():
        forecasts = pd.read_parquet(cal_path)
        print(f"Loaded {len(forecasts)} calibration rows")

    # Observations (GHCN)
    obs = pd.read_parquet(config.GHCN_PARQUET_PATH)
    obs["date"] = pd.to_datetime(obs["date"])
    print(f"Loaded {len(obs)} observation days (through {obs['date'].max().date()})")

    return trades, forecasts, obs


def extract_date_from_ticker(ticker):
    """KXHIGHNY-26FEB03-B34.5 -> date(2026, 2, 3)"""
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    date_str = parts[1]  # e.g. "26FEB03"
    try:
        dt = datetime.strptime(date_str, "%y%b%d")
        return dt.date()
    except ValueError:
        return None


def get_vwap_prices(trades_df, target_date):
    """Get VWAP prices for each ticker on a given date.
    Uses all trades on the target date as price estimates."""
    day_trades = trades_df[trades_df["created_time"].dt.date == target_date]
    if day_trades.empty:
        return {}

    prices = {}
    for ticker, group in day_trades.groupby("ticker"):
        if "yes_price" in group.columns:
            vwap = (group["yes_price"] * group["count"]).sum() / group["count"].sum()
            prices[ticker] = vwap / 100.0  # convert cents to dollars
        elif "price" in group.columns:
            vwap = (group["price"] * group["count"]).sum() / group["count"].sum()
            prices[ticker] = vwap / 100.0
    return prices


def get_forecast_for_date(forecasts_df, target_date):
    """Get the NWS forecast high for a target date."""
    if forecasts_df is None:
        return None
    # Calibration dataset uses 'date' and 'forecast_high_f'
    target_str = str(target_date)
    row = forecasts_df[forecasts_df["date"].astype(str) == target_str]
    if row.empty:
        return None
    return float(row.iloc[0]["forecast_high_f"])


def analyze():
    trades, forecasts, obs = load_data()

    # Find all unique market dates from tickers
    all_tickers = trades["ticker"].unique()
    date_tickers = defaultdict(list)
    for t in all_tickers:
        d = extract_date_from_ticker(t)
        if d:
            date_tickers[d].append(t)

    # Only analyze settled dates (where we have GHCN observations)
    obs_dates = set(obs["date"].dt.date)
    settled_dates = sorted(d for d in date_tickers if d in obs_dates)

    print(f"\nSettled market dates with observations: {len(settled_dates)}")
    if not settled_dates:
        print("No settled dates found!")
        return

    # Analysis
    model_brier_scores = []
    market_brier_scores = []
    model_wins = 0
    market_wins = 0
    ties = 0
    edge_trades_correct = 0
    edge_trades_total = 0

    # Calibration bins
    model_cal_bins = defaultdict(lambda: {"count": 0, "hits": 0})
    market_cal_bins = defaultdict(lambda: {"count": 0, "hits": 0})

    sigma = config.SIGMA_1DAY  # 1.2

    print(f"\nUsing sigma={sigma}")
    print("=" * 90)
    print(f"{'Date':<12} {'Actual':>6} {'Fcst':>5} {'Err':>4}  {'Winner Bucket':<28} {'Model':>6} {'Mkt':>6} {'Edge':>6}")
    print("-" * 90)

    for target_date in settled_dates:
        # Get actual temperature
        obs_row = obs[obs["date"].dt.date == target_date]
        if obs_row.empty:
            continue
        actual_temp = int(obs_row.iloc[0]["tmax_f"])

        # Get forecast
        forecast = get_forecast_for_date(forecasts, target_date)
        if forecast is None:
            continue

        mu = forecast - config.FORECAST_BIAS

        # Get market prices (VWAP from day before or same day)
        # Use trades from the day before the target date (when you'd actually trade)
        trade_date = target_date - timedelta(days=1)
        market_prices = get_vwap_prices(trades, trade_date)
        if not market_prices:
            # Try same day morning
            market_prices = get_vwap_prices(trades, target_date)
        if not market_prices:
            continue

        # Parse buckets for this date
        tickers = date_tickers[target_date]
        buckets = []
        for ticker in tickers:
            # No title column in trades -- use ticker as title for parsing
            b = parse_bucket_title(ticker, ticker)
            if b:
                buckets.append(b)

        if not buckets:
            continue

        # Compute model probabilities and find winning bucket
        forecast_err = actual_temp - forecast
        day_model_brier = 0.0
        day_market_brier = 0.0
        winning_bucket = None
        winning_model_p = 0.0
        winning_market_p = 0.0
        n_buckets = 0

        for b in buckets:
            model_p = b.probability(mu, sigma)
            market_p = market_prices.get(b.ticker)
            if market_p is None:
                continue

            outcome = 1.0 if b.settles_yes(actual_temp) else 0.0
            n_buckets += 1

            # Brier scores
            day_model_brier += (model_p - outcome) ** 2
            day_market_brier += (market_p - outcome) ** 2

            # Calibration
            model_bin = round(model_p * 10) / 10  # bin to nearest 10%
            market_bin = round(market_p * 10) / 10
            model_cal_bins[model_bin]["count"] += 1
            model_cal_bins[model_bin]["hits"] += outcome
            market_cal_bins[market_bin]["count"] += 1
            market_cal_bins[market_bin]["hits"] += outcome

            if outcome == 1.0:
                winning_bucket = b
                winning_model_p = model_p
                winning_market_p = market_p

            # Edge trade analysis
            yes_edge = model_p - market_p
            no_edge = market_p - model_p
            if abs(yes_edge) > config.MIN_EDGE:
                edge_trades_total += 1
                if yes_edge > config.MIN_EDGE and outcome == 1.0:
                    edge_trades_correct += 1
                elif no_edge > config.MIN_EDGE and outcome == 0.0:
                    edge_trades_correct += 1

        if n_buckets > 0:
            day_model_brier /= n_buckets
            day_market_brier /= n_buckets
            model_brier_scores.append(day_model_brier)
            market_brier_scores.append(day_market_brier)

            if day_model_brier < day_market_brier:
                model_wins += 1
            elif day_market_brier < day_model_brier:
                market_wins += 1
            else:
                ties += 1

        winner_label = ""
        if winning_bucket:
            if winning_bucket.bucket_type == "between":
                winner_label = f"{winning_bucket.low}-{winning_bucket.high}"
            elif winning_bucket.bucket_type == "above":
                winner_label = f">{winning_bucket.low}"
            else:
                winner_label = f"<{winning_bucket.high}"

        edge_str = ""
        if winning_model_p > 0 and winning_market_p > 0:
            edge = winning_model_p - winning_market_p
            edge_str = f"{edge:+.1%}"

        print(f"{target_date}   {actual_temp:>4}F {forecast:>4.0f}F {actual_temp - forecast:>+3.0f}  "
              f"{winner_label:<28} {winning_model_p:>5.1%} {winning_market_p:>5.1%} {edge_str:>6}")

    # Summary
    print("=" * 90)
    print(f"\n{'ACCURACY COMPARISON':^60}")
    print("=" * 60)

    avg_model_brier = np.mean(model_brier_scores) if model_brier_scores else 0
    avg_market_brier = np.mean(market_brier_scores) if market_brier_scores else 0

    print(f"\n  Settled days analyzed: {len(model_brier_scores)}")
    print(f"\n  Brier Score (lower = better):")
    print(f"    Our model:  {avg_model_brier:.4f}")
    print(f"    Market:     {avg_market_brier:.4f}")
    if avg_model_brier < avg_market_brier:
        pct_better = (avg_market_brier - avg_model_brier) / avg_market_brier * 100
        print(f"    --> Model is {pct_better:.1f}% more accurate than market")
    else:
        pct_worse = (avg_model_brier - avg_market_brier) / avg_market_brier * 100
        print(f"    --> Market is {pct_worse:.1f}% more accurate than model")

    print(f"\n  Day-by-day winner (Brier):")
    print(f"    Model wins:  {model_wins}")
    print(f"    Market wins: {market_wins}")
    print(f"    Ties:        {ties}")

    if edge_trades_total > 0:
        print(f"\n  Edge trades (|edge| > {config.MIN_EDGE:.0%}):")
        print(f"    Total:   {edge_trades_total}")
        print(f"    Correct: {edge_trades_correct} ({edge_trades_correct/edge_trades_total:.0%})")
        print(f"    Wrong:   {edge_trades_total - edge_trades_correct}")
    else:
        print(f"\n  No edge trades found (threshold: {config.MIN_EDGE:.0%})")

    # Calibration table
    print(f"\n  CALIBRATION (predicted prob vs actual hit rate):")
    print(f"  {'Bin':>6} {'Model':>18} {'Market':>18}")
    print(f"  {'':>6} {'n':>5} {'hit%':>6} {'off':>6} {'n':>5} {'hit%':>6} {'off':>6}")
    print(f"  {'-'*54}")

    for b in sorted(set(list(model_cal_bins.keys()) + list(market_cal_bins.keys()))):
        mc = model_cal_bins[b]
        mk = market_cal_bins[b]
        m_hit = mc["hits"] / mc["count"] if mc["count"] > 0 else 0
        k_hit = mk["hits"] / mk["count"] if mk["count"] > 0 else 0
        m_off = m_hit - b if mc["count"] > 0 else 0
        k_off = k_hit - b if mk["count"] > 0 else 0
        print(f"  {b:>5.0%}  {mc['count']:>5} {m_hit:>5.0%} {m_off:>+5.0%}  "
              f"{mk['count']:>5} {k_hit:>5.0%} {k_off:>+5.0%}")

    print(f"\n  (Calibration 'off' = actual hit rate minus predicted prob)")
    print(f"  (Perfect calibration = 0% off in every bin)")


if __name__ == "__main__":
    analyze()

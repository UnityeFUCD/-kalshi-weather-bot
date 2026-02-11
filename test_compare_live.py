"""
test_compare_live.py -- Live Snapshot: Old Model vs New Model on current markets.

Fetches current NWS forecast + ensemble data + METAR observations + Kalshi
market prices, then runs both old and new model side-by-side. Appends
results to reports/model_comparison.jsonl for longitudinal tracking.

Run this every morning during paper trading to build a dataset showing
how often the new model agrees/disagrees with the old one and who's right.

Usage:
    py -X utf8 test_compare_live.py
    py -X utf8 test_compare_live.py --debug
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from kalshi_auth import KalshiAuth
from kalshi_client import KalshiClient
from nws import NWSClient
from model import parse_bucket_title, compute_signals, compute_fee
from confidence import (compute_confidence, compute_boundary_z,
                        extract_bucket_boundaries)
from ensemble import EnsembleForecaster
from delta_tracker import DeltaTracker
from metar_obs import MetarObserver
from market_registry import get_enabled_markets, get_market

logger = logging.getLogger(__name__)

COMPARISON_LOG = PROJECT_ROOT / "reports" / "model_comparison.jsonl"


def _serialize_signal(signal):
    """Compact JSON-safe signal row."""
    return {
        "ticker": signal.bucket.ticker,
        "side": signal.side,
        "edge": round(signal.edge, 6),
        "model_prob": round(signal.model_prob, 6),
        "market_price": round(signal.market_price, 6),
    }


def _best_signal(signal_rows):
    """Pick highest-edge signal from serialized rows."""
    if not signal_rows:
        return None
    best = max(signal_rows, key=lambda row: row.get("edge", 0.0))
    return {
        "bucket": best.get("ticker"),
        "side": best.get("side"),
        "edge": best.get("edge"),
    }


def run_live_comparison():
    """
    Fetch live data and compare old vs new model.
    Returns comparison dict.
    """
    # Initialize clients
    markets_list = get_enabled_markets()
    if not markets_list:
        print("No enabled markets!")
        return None

    mc = markets_list[0]  # KXHIGHNY

    nws = NWSClient(market_config=mc)
    auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
    kalshi = KalshiClient(auth=auth)

    # Phase 4/5 modules
    delta_tracker = DeltaTracker(mc.series_ticker)
    station = config.METAR_STATIONS.get(mc.series_ticker, mc.nws_station)
    metar = MetarObserver(station, nws_client=nws)
    ensemble = EnsembleForecaster(lat=40.7831, lon=-73.9712)

    # Determine target date
    now_et = datetime.now(config.MARKET_TZ)
    hour_et = now_et.hour

    if hour_et >= 8:
        target_date = (now_et + timedelta(days=1)).date()
        sigma_base = mc.sigma_1day
    else:
        target_date = now_et.date()
        if hour_et < 6:
            sigma_base = max(mc.sigma_1day, config.SIGMA_PREDAWN_FLOOR)
        elif hour_et < 14:
            sigma_base = mc.sigma_sameday_am
        else:
            sigma_base = mc.sigma_sameday_pm

    print("=" * 75)
    print("LIVE MODEL COMPARISON")
    print("=" * 75)
    print()
    print("Time: %s ET (hour=%d)" % (now_et.strftime("%Y-%m-%d %H:%M"), hour_et))
    print("Target date: %s" % target_date)
    print("Market: %s (%s)" % (mc.series_ticker, mc.display_name))
    print()

    # ── Step 1: Get NWS forecast ──
    print("[1] Fetching NWS forecast...")
    forecast_temp, period_name = nws.get_high_forecast_for_date(target_date)
    if forecast_temp is None:
        forecast_temp = nws.get_today_high_forecast()

    # Hourly-derived μ
    snapshot, is_revision = delta_tracker.process_hourly_forecast(nws, target_date)
    hourly_mu = delta_tracker.get_hourly_mu(target_date)

    if hourly_mu is not None:
        print("  Period high: %s, Hourly max: %.0fF" % (forecast_temp, hourly_mu))
        mu_new = hourly_mu - mc.forecast_bias
    else:
        print("  Period high: %sF (no hourly data)" % forecast_temp)
        mu_new = forecast_temp - mc.forecast_bias if forecast_temp else None

    if forecast_temp is None and hourly_mu is None:
        print("  FAILED to get forecast -- aborting")
        return None

    mu_old = (forecast_temp or hourly_mu) - mc.forecast_bias
    if mu_new is None:
        mu_new = mu_old

    # ── Step 2: Get METAR obs ──
    print("[2] Fetching METAR observations...")
    obs_temp, obs_source, obs_time = metar.get_current_temp()
    residual_ewma = None

    if obs_temp is not None:
        print("  Current temp: %.1fF from %s" % (obs_temp, obs_source))
        if target_date == now_et.date():
            mu_nudged = metar.compute_mu_nudge(mu_new)
            _, residual_ewma = metar.compute_residual(mu_new)
            if abs(mu_nudged - mu_new) > 0.01:
                print("  Obs nudge: μ %.1f -> %.1f" % (mu_new, mu_nudged))
                mu_new = mu_nudged
    else:
        print("  No METAR observation available")

    # ── Step 3: Get ensemble σ ──
    print("[3] Fetching ensemble forecasts...")
    sigma_ens = None
    n_members = 0
    try:
        sigma_ens, n_members, _ = ensemble.get_ensemble_sigma(target_date)
        if sigma_ens is not None:
            print("  Ensemble σ: %.2fF (%d members)" % (sigma_ens, n_members))
        else:
            print("  Ensemble fetch returned no data")
    except Exception as e:
        print("  Ensemble fetch failed: %s" % e)

    # Compose new sigma: simple max(base, α × ens). No stacking.
    sigma_old = mc.sigma_1day
    sigma_new = sigma_base

    if sigma_ens is not None:
        sigma_new = ensemble.compose_sigma(
            sigma_base=sigma_base,
            sigma_ens=sigma_ens,
        )

    # ── Step 4: Fetch markets ──
    print("[4] Fetching Kalshi markets...")
    all_markets = kalshi.get_markets(series_ticker=mc.series_ticker, status="open")

    if not all_markets:
        print("  No open markets found")
        return None

    # Filter to target date
    date_str = target_date.strftime("%y") + target_date.strftime("%b").upper() + target_date.strftime("%d")
    tradeable = [m for m in all_markets if date_str in m.ticker]

    if not tradeable:
        print("  No markets for %s" % date_str)
        tradeable = all_markets

    print("  Found %d markets for %s" % (len(tradeable), date_str))

    # Parse buckets and prices
    buckets = []
    market_prices = {}

    for m in tradeable:
        bucket = parse_bucket_title(m.ticker, m.title)
        if bucket is None:
            continue
        buckets.append(bucket)

        if m.yes_bid is not None and m.yes_ask is not None:
            price = (m.yes_bid + m.yes_ask) / 200.0
        elif m.yes_ask is not None:
            price = m.yes_ask / 100.0
        elif m.yes_bid is not None:
            price = m.yes_bid / 100.0
        elif m.last_price is not None:
            price = m.last_price / 100.0
        else:
            continue

        market_prices[m.ticker] = price

    if not buckets:
        print("  No parseable buckets")
        return None

    # ── Step 5: Confidence scoring (LOG ONLY, no blocking) ──
    boundaries = extract_bucket_boundaries(buckets)
    boundary_z = compute_boundary_z(mu_new, sigma_new, boundaries)

    forecast_age = delta_tracker.forecast_age_minutes(target_date)
    obs_age = metar.get_obs_age_minutes()

    confidence, gate_scores = compute_confidence(
        forecast_age_min=forecast_age,
        obs_age_min=obs_age,
        residual_ewma=residual_ewma,
        boundary_z=boundary_z,
        hour_et=hour_et,
    )

    # ── Step 6: Run both models ──
    print()
    print("[5] Running models...")

    # Old model signals
    filtered_prices = {t: p for t, p in market_prices.items() if 0.05 <= p <= 0.95}
    old_signals = compute_signals(buckets, filtered_prices, mu_old, sigma_old,
                                  min_edge=config.MIN_EDGE)

    # New model signals: flat 8% edge, no blocking
    new_blocked = False
    new_block_reason = ""
    new_signals = compute_signals(buckets, filtered_prices, mu_new, sigma_new,
                                  min_edge=config.MIN_EDGE)

    # ── Step 7: Print comparison ──
    print()
    print("=" * 75)
    print("COMPARISON RESULTS")
    print("=" * 75)
    print()

    print("%-38s | %-38s" % ("OLD MODEL", "NEW MODEL"))
    print("-" * 38 + " | " + "-" * 38)
    print("%-38s | %-38s" % (
        "μ = %.1fF" % mu_old, "μ = %.1fF" % mu_new))
    print("%-38s | %-38s" % (
        "σ = %.2f (flat)" % sigma_old, "σ = %.2f (composed)" % sigma_new))
    print("%-38s | %-38s" % (
        "MIN_EDGE = %.0f%%" % (config.MIN_EDGE * 100),
        "MIN_EDGE = %.0f%% (flat)" % (config.MIN_EDGE * 100)))
    print("%-38s | %-38s" % (
        "Confidence: N/A",
        "Confidence: %.3f" % confidence))
    print("%-38s | %-38s" % (
        "Boundary z: N/A",
        "Boundary z: %s" % ("%.2f" % boundary_z if boundary_z is not None else "N/A")))
    print("%-38s | %-38s" % (
        "Signals: %d" % len(old_signals),
        "Signals: %d%s" % (len(new_signals),
                           " (BLOCKED: %s)" % new_block_reason if new_blocked else "")))
    print()

    # Market prices table
    print("MARKET STATE:")
    print("  %-40s %6s %8s %8s %8s %8s" % ("Ticker", "Price", "P(old)", "P(new)", "E(old)", "E(new)"))
    print("  " + "-" * 82)

    for b in sorted(buckets, key=lambda b: b.low or 0, reverse=True):
        price = market_prices.get(b.ticker)
        if price is None:
            continue
        p_old = b.probability(mu_old, sigma_old)
        p_new = b.probability(mu_new, sigma_new)
        e_old = p_old - price
        e_new = p_new - price
        print("  %-40s %5.0fc %7.1f%% %7.1f%% %+7.1f%% %+7.1f%%" % (
            b.ticker, price * 100, p_old * 100, p_new * 100, e_old * 100, e_new * 100))

    print()

    # Signal details
    print("OLD MODEL SIGNALS (%d):" % len(old_signals))
    if old_signals:
        for s in old_signals:
            count = _position_size(s.suggested_price)
            risk = s.suggested_price / 100.0 * count
            print("  %s %s: edge=%+.1f%% model=%.1f%% mkt=%.0fc %dx @ %dc risk=$%.2f" % (
                s.side.upper(), s.bucket.ticker.split("-")[-1],
                s.edge * 100, s.model_prob * 100, s.market_price * 100,
                count, s.suggested_price, risk))
    else:
        print("  (none)")

    print()
    print("NEW MODEL SIGNALS (%d):" % len(new_signals))
    if new_blocked:
        print("  BLOCKED: %s" % new_block_reason)
    elif new_signals:
        for s in new_signals:
            count = _position_size(s.suggested_price)
            risk = s.suggested_price / 100.0 * count
            print("  %s %s: edge=%+.1f%% model=%.1f%% mkt=%.0fc %dx @ %dc risk=$%.2f" % (
                s.side.upper(), s.bucket.ticker.split("-")[-1],
                s.edge * 100, s.model_prob * 100, s.market_price * 100,
                count, s.suggested_price, risk))
    else:
        print("  (none -- no edge above %.0f%% threshold)" % (config.MIN_EDGE * 100))

    # Agreement analysis
    print()
    old_tickers = set(s.bucket.ticker for s in old_signals)
    new_tickers = set(s.bucket.ticker for s in new_signals)
    agree = old_tickers & new_tickers
    old_only = old_tickers - new_tickers
    new_only = new_tickers - old_tickers

    if agree:
        print("AGREEMENT: Both models signal %d trade(s)" % len(agree))
    if old_only:
        print("OLD ONLY: %d trade(s) the new model would skip" % len(old_only))
        for t in old_only:
            print("  - %s" % t.split("-")[-1])
    if new_only:
        print("NEW ONLY: %d trade(s) the old model would miss" % len(new_only))
        for t in new_only:
            print("  - %s" % t.split("-")[-1])
    if not agree and not old_only and not new_only:
        print("Both models see no trades.")

    # Gate scores detail
    print()
    print("CONFIDENCE GATE DETAIL:")
    for gate, score in gate_scores.items():
        print("  %-25s %.2f" % (gate, score))

    print()

    # ── Step 8: Log to JSONL ──
    old_signal_rows = [_serialize_signal(s) for s in old_signals]
    new_signal_rows = [_serialize_signal(s) for s in new_signals]

    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_date": str(target_date),
        "hour_et": hour_et,
        "forecast_high": forecast_temp,
        "hourly_mu": hourly_mu,
        "obs_temp": obs_temp,
        "old_model": {
            "mu": mu_old,
            "sigma": sigma_old,
            "min_edge": config.MIN_EDGE,
            "signals": len(old_signals),
            "signal_tickers": [s.bucket.ticker for s in old_signals],
            "signal_details": old_signal_rows,
            "best_edge": _best_signal(old_signal_rows),
        },
        "new_model": {
            "mu": mu_new,
            "sigma": sigma_new,
            "min_edge": config.MIN_EDGE,
            "confidence": confidence,
            "boundary_z": boundary_z,
            "blocked": new_blocked,
            "block_reason": new_block_reason,
            "signals": len(new_signals),
            "signal_tickers": [s.bucket.ticker for s in new_signals],
            "signal_details": new_signal_rows,
            "best_edge": _best_signal(new_signal_rows),
            "ensemble_sigma": sigma_ens,
            "n_members": n_members,
        },
        "gate_scores": gate_scores,
        "agreement": len(agree),
        "old_only": len(old_only),
        "new_only": len(new_only),
    }

    COMPARISON_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(comparison, default=str) + "\n")
    print("Comparison logged to %s" % COMPARISON_LOG)

    return comparison


def _position_size(suggested_price_cents):
    """Compute position size."""
    price_dollars = suggested_price_cents / 100.0
    if price_dollars <= 0:
        return 0
    max_contracts = int(config.MAX_RISK_PER_TRADE / price_dollars)
    count = max(config.MIN_CONTRACTS, min(max_contracts, 50))
    while count * price_dollars > config.MAX_RISK_PER_TRADE and count > 0:
        count -= 1
    return count


# ─── Longitudinal Summary ───────────────────────────────────────────────────

def print_longitudinal_summary():
    """Print summary of all logged comparisons."""
    if not COMPARISON_LOG.exists():
        print("No comparison data yet.")
        return

    records = []
    with open(COMPARISON_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("No comparison records found.")
        return

    print()
    print("=" * 75)
    print("LONGITUDINAL SUMMARY (%d comparisons)" % len(records))
    print("=" * 75)
    print()

    total_agree = sum(r.get("agreement", 0) for r in records)
    total_old_only = sum(r.get("old_only", 0) for r in records)
    total_new_only = sum(r.get("new_only", 0) for r in records)
    total_blocked = sum(1 for r in records if r.get("new_model", {}).get("blocked"))

    print("  Total comparisons: %d" % len(records))
    print("  Agreement:  %d signal(s)" % total_agree)
    print("  Old only:   %d signal(s) (new model would skip)" % total_old_only)
    print("  New only:   %d signal(s) (old model would miss)" % total_new_only)
    print("  Blocked:    %d time(s) (confidence gate blocked new model)" % total_blocked)

    print()
    print("  %-12s %4s %6s %6s %6s %6s %5s" % (
        "Date", "Hour", "σ_old", "σ_new", "Agree", "O-only", "Conf"))
    print("  " + "-" * 55)
    for r in records[-20:]:
        old_m = r.get("old_model", {})
        new_m = r.get("new_model", {})
        print("  %-12s %4d %6.2f %6.2f %6d %6d %5.2f" % (
            r.get("target_date", "?"),
            r.get("hour_et", 0),
            old_m.get("sigma", 0),
            new_m.get("sigma", 0),
            r.get("agreement", 0),
            r.get("old_only", 0),
            new_m.get("confidence", 0),
        ))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Model Comparison")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument("--summary", action="store_true",
                        help="Print longitudinal summary instead of live comparison")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.summary:
        print_longitudinal_summary()
    else:
        run_live_comparison()
        print_longitudinal_summary()


if __name__ == "__main__":
    main()

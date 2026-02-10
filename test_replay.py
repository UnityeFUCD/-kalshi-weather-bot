"""
test_replay.py -- Replay a historical trading day: old model vs new model.

Reconstructs the market state for a given date from paper_trades.jsonl and
runs both the old model (flat σ=1.2, static 8% edge, no confidence gates)
and the new model (ensemble σ, dynamic edge, confidence gates, pre-dawn
blocking) against the same prices.

Usage:
    py -X utf8 test_replay.py --date 2026-02-10
    py -X utf8 test_replay.py --date 2026-02-10 --actual 36
"""

import sys
import json
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from model import Bucket, compute_signals, compute_fee, phi
from confidence import (compute_confidence, compute_dynamic_min_edge,
                        compute_boundary_z, extract_bucket_boundaries,
                        passes_predawn_gates)
from ensemble import EnsembleForecaster
from market_registry import get_market

logger = logging.getLogger(__name__)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_paper_signals(target_date_str):
    """
    Load paper trade signals from paper_trades.jsonl for a given date.
    Returns list of signal records (dict) for that target_date.
    """
    path = config.PAPER_TRADES_PATH
    if not path.exists():
        logger.error("No paper_trades.jsonl found at %s", path)
        return []

    signals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") == "signal" and rec.get("target_date") == target_date_str:
                signals.append(rec)

    return signals


def load_fill_records():
    """Load all fill records from paper_trades.jsonl."""
    path = config.PAPER_TRADES_PATH
    if not path.exists():
        return {}
    fills = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") == "fill":
                fills[rec["trade_id"]] = rec
    return fills


def get_actual_tmax(target_date):
    """Get actual TMAX from GHCN for a date."""
    try:
        import pandas as pd
        ghcn_path = config.GHCN_PARQUET_PATH
        if not ghcn_path.exists():
            return None
        df = pd.read_parquet(ghcn_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        row = df[df["date"] == target_date]
        if not row.empty:
            tmax = row.iloc[0]["tmax_f"]
            if pd.notna(tmax):
                return int(tmax)
    except Exception:
        pass
    return None


# ─── Bucket Reconstruction ──────────────────────────────────────────────────

def reconstruct_buckets_from_signals(signals):
    """
    Reconstruct Bucket objects and market prices from paper_trades.jsonl signals.
    Deduplicates by ticker (takes the latest signal for each).
    """
    # Deduplicate: keep last signal per ticker (latest timestamp)
    by_ticker = {}
    for s in signals:
        by_ticker[s["ticker"]] = s

    buckets = []
    market_prices = {}

    for ticker, sig in by_ticker.items():
        bucket = _parse_bucket_from_ticker(ticker)
        if bucket is None:
            continue
        buckets.append(bucket)
        market_prices[ticker] = sig["market_price"]

    return buckets, market_prices


def _parse_bucket_from_ticker(ticker):
    """Parse a Bucket from ticker format: KXHIGHNY-26FEB10-B34.5 or T39."""
    import re

    parts = ticker.split("-")
    if len(parts) < 3:
        return None

    bucket_code = parts[-1]

    # B-type: between
    m = re.match(r'B(\d+\.?\d*)', bucket_code)
    if m:
        midpoint = float(m.group(1))
        low = int(midpoint - 0.5)
        high = int(midpoint + 0.5)
        return Bucket(ticker=ticker, title="%d to %d" % (low, high),
                      bucket_type="between", low=low, high=high)

    # T-type: threshold (assume above for high T, below for low T)
    m = re.match(r'T(\d+\.?\d*)', bucket_code)
    if m:
        threshold = int(float(m.group(1)))
        # Heuristic: if threshold > 40 for a Feb day, it's "above"
        # More robust: look at market_price -- high price means "below" bucket
        return Bucket(ticker=ticker, title="above %d" % threshold,
                      bucket_type="above", low=threshold, high=None)

    return None


# ─── Settlement Logic ────────────────────────────────────────────────────────

def compute_hypothetical_pnl(signals_list, actual_temp, buckets):
    """
    Compute hypothetical PnL for a list of model signals, given actual temp.
    Returns total_pnl, list of (signal, pnl, won) tuples.
    """
    bucket_lookup = {b.ticker: b for b in buckets}
    results = []
    total_pnl = 0.0

    for sig in signals_list:
        bucket = bucket_lookup.get(sig["ticker"])
        if bucket is None:
            continue

        bucket_hit = bucket.settles_yes(actual_temp)
        contracts = sig["contracts"]
        entry_cents = sig["suggested_price"]
        cost_per_contract = entry_cents / 100.0
        fee = sig.get("fee", compute_fee(entry_cents, contracts, is_maker=True))

        if sig["side"] == "buy_yes":
            payout = 1.00 * contracts if bucket_hit else 0.0
        else:
            payout = 1.00 * contracts if not bucket_hit else 0.0

        cost = cost_per_contract * contracts + fee
        pnl = payout - cost

        results.append({
            "ticker": sig["ticker"],
            "side": sig["side"],
            "model_prob": sig["model_prob"],
            "market_price": sig["market_price"],
            "edge": sig["edge"],
            "contracts": contracts,
            "entry_cents": entry_cents,
            "cost": cost,
            "payout": payout,
            "pnl": pnl,
            "won": pnl > 0,
            "bucket_hit": bucket_hit,
        })
        total_pnl += pnl

    return total_pnl, results


# ─── Model Runners ───────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    mu: float
    sigma: float
    min_edge: float
    confidence: Optional[float]
    boundary_z: Optional[float]
    signals: list          # list of dicts
    blocked: bool
    block_reason: str
    total_pnl: float
    trade_results: list    # list of dicts from compute_hypothetical_pnl


def run_old_model(buckets, market_prices, forecast_high, actual_temp, hour_et):
    """
    Run the OLD model: flat σ=1.2, static 8% edge, no confidence gates.
    """
    mc = get_market("KXHIGHNY")
    sigma = mc.sigma_1day  # 1.2
    mu = forecast_high - mc.forecast_bias

    signals = compute_signals(buckets, market_prices, mu, sigma,
                              min_edge=config.MIN_EDGE)

    # Convert to serializable dicts
    signal_dicts = []
    for s in signals:
        count = _position_size(s.suggested_price)
        fee = compute_fee(s.suggested_price, count, is_maker=True)
        signal_dicts.append({
            "ticker": s.bucket.ticker,
            "side": s.side,
            "model_prob": round(s.model_prob, 4),
            "market_price": round(s.market_price, 4),
            "edge": round(s.edge, 4),
            "suggested_price": s.suggested_price,
            "contracts": count,
            "fee": fee,
        })

    total_pnl, trade_results = compute_hypothetical_pnl(
        signal_dicts, actual_temp, buckets)

    return ModelResult(
        name="OLD (σ=1.2, edge=8%, no gates)",
        mu=mu,
        sigma=sigma,
        min_edge=config.MIN_EDGE,
        confidence=None,
        boundary_z=None,
        signals=signal_dicts,
        blocked=False,
        block_reason="",
        total_pnl=total_pnl,
        trade_results=trade_results,
    )


def run_new_model(buckets, market_prices, forecast_high, actual_temp,
                  hour_et, ensemble_sigma=None):
    """
    Run the NEW V2 model: ensemble σ only, flat 8% edge, NO gates, half-Kelly.

    Manager's directive: better information → smarter trades, not fewer trades.
    Keep ensemble σ (real data from 82 weather models).
    Remove confidence blocks, dynamic edge, boundary stacking.
    Let edge determine position SIZE via Kelly, not whether to trade.
    """
    mc = get_market("KXHIGHNY")
    sigma_base = mc.sigma_1day

    mu = forecast_high - mc.forecast_bias

    # Ensemble σ: simple max(base, α × ens). No stacking.
    sigma = sigma_base
    if ensemble_sigma is not None:
        sigma = max(sigma_base, config.ENSEMBLE_ALPHA * ensemble_sigma)
    else:
        # For historical replay, use typical NYC winter ensemble σ
        sigma = max(sigma_base, config.ENSEMBLE_ALPHA * 1.3)

    # Confidence for LOGGING only (not blocking)
    boundaries = extract_bucket_boundaries(buckets)
    boundary_z = compute_boundary_z(mu, sigma, boundaries)

    if hour_et < 6:
        forecast_age = 360
        obs_age = float("inf")
        residual_ewma = None
    elif hour_et < 10:
        forecast_age = 60
        obs_age = 30
        residual_ewma = 0.5
    else:
        forecast_age = 30
        obs_age = 15
        residual_ewma = 0.3

    confidence, gate_scores = compute_confidence(
        forecast_age_min=forecast_age,
        obs_age_min=obs_age,
        residual_ewma=residual_ewma,
        boundary_z=boundary_z,
        hour_et=hour_et,
    )

    # NO blocking. NO dynamic edge. Flat 8%.
    min_edge = config.MIN_EDGE

    signals = compute_signals(buckets, market_prices, mu, sigma,
                              min_edge=min_edge)

    signal_dicts = []
    for s in signals:
        count = _kelly_position_size(s)
        fee = compute_fee(s.suggested_price, count, is_maker=True)
        signal_dicts.append({
            "ticker": s.bucket.ticker,
            "side": s.side,
            "model_prob": round(s.model_prob, 4),
            "market_price": round(s.market_price, 4),
            "edge": round(s.edge, 4),
            "suggested_price": s.suggested_price,
            "contracts": count,
            "fee": fee,
        })

    total_pnl, trade_results = compute_hypothetical_pnl(
        signal_dicts, actual_temp, buckets)

    return ModelResult(
        name="NEW V2 (ens σ=%.2f, flat 8%%, half-Kelly, no gates)" % sigma,
        mu=mu,
        sigma=sigma,
        min_edge=min_edge,
        confidence=confidence,
        boundary_z=boundary_z,
        signals=signal_dicts,
        blocked=False,
        block_reason="",
        total_pnl=total_pnl,
        trade_results=trade_results,
    )


def _position_size(suggested_price_cents):
    """Compute position size from suggested price (old flat sizing)."""
    price_dollars = suggested_price_cents / 100.0
    if price_dollars <= 0:
        return 0
    max_contracts = int(config.MAX_RISK_PER_TRADE / price_dollars)
    count = max(config.MIN_CONTRACTS, min(max_contracts, 50))
    while count * price_dollars > config.MAX_RISK_PER_TRADE and count > 0:
        count -= 1
    return count


def _kelly_position_size(signal):
    """Half-Kelly position sizing. Edge determines size."""
    price_dollars = signal.suggested_price / 100.0
    if price_dollars <= 0:
        return 0

    if signal.side == "buy_yes":
        payout_complement = 1.0 - signal.market_price
    else:
        payout_complement = signal.market_price

    if payout_complement <= 0:
        return 0

    kelly_f = signal.edge / payout_complement * 0.5  # half-Kelly
    kelly_dollars = min(kelly_f * config.BANKROLL, config.MAX_RISK_PER_TRADE)
    kelly_dollars = max(kelly_dollars, 0)

    max_contracts = int(kelly_dollars / price_dollars) if price_dollars > 0 else 0
    count = max(config.MIN_CONTRACTS, min(max_contracts, 50))
    while count * price_dollars > config.MAX_RISK_PER_TRADE and count > 0:
        count -= 1
    return count


# ─── Report ──────────────────────────────────────────────────────────────────

def print_replay_report(target_date_str, actual_temp, old, new, signals_from_log):
    """Print side-by-side comparison of old vs new model."""
    print()
    print("=" * 75)
    print("REPLAY COMPARISON -- %s" % target_date_str)
    print("=" * 75)
    print()

    print("ACTUAL SETTLEMENT: %dF" % actual_temp)
    print()

    # Header
    print("%-38s | %-38s" % ("OLD MODEL", "NEW MODEL"))
    print("-" * 38 + " | " + "-" * 38)

    # Parameters
    print("%-38s | %-38s" % (
        "σ = %.2f (flat)" % old.sigma,
        "σ = %.2f (ensemble-composed)" % new.sigma))
    print("%-38s | %-38s" % (
        "μ = %.1fF" % old.mu,
        "μ = %.1fF" % new.mu))
    print("%-38s | %-38s" % (
        "MIN_EDGE = %.0f%% (static)" % (old.min_edge * 100),
        "MIN_EDGE = %.1f%% (dynamic)" % (new.min_edge * 100)))
    print("%-38s | %-38s" % (
        "Confidence: N/A",
        "Confidence: %.3f" % new.confidence if new.confidence else "N/A"))
    print("%-38s | %-38s" % (
        "Boundary z: N/A",
        "Boundary z: %.2f" % new.boundary_z if new.boundary_z is not None else "N/A"))
    print("%-38s | %-38s" % (
        "Pre-dawn block: N/A",
        "Blocked: %s" % ("YES (%s)" % new.block_reason if new.blocked else "NO")))
    print()

    # Signals
    print("SIGNALS:")
    max_rows = max(len(old.signals), len(new.signals), 1)

    for i in range(max_rows):
        left = ""
        right = ""
        if i < len(old.signals):
            s = old.signals[i]
            left = "%s %s edge=%+.1f%%" % (
                s["side"][:7].upper(), s["ticker"].split("-")[-1],
                s["edge"] * 100)
        if i < len(new.signals):
            s = new.signals[i]
            right = "%s %s edge=%+.1f%%" % (
                s["side"][:7].upper(), s["ticker"].split("-")[-1],
                s["edge"] * 100)
        print("  %-36s | %-36s" % (left, right))

    if not old.signals and not new.signals:
        print("  %-36s | %-36s" % ("(no signals)", "(no signals)"))

    print()

    # Trade details + PnL
    print("HYPOTHETICAL PnL (if filled at suggested price, settled at %dF):" % actual_temp)
    print()

    print("  OLD MODEL TRADES:")
    if old.trade_results:
        for t in old.trade_results:
            won = "WON" if t["won"] else "LOST"
            print("    %s %s %dx @ %dc: payout=$%.2f cost=$%.2f PnL=$%+.2f [%s]" % (
                t["side"][:7].upper(), t["ticker"].split("-")[-1],
                t["contracts"], t["entry_cents"],
                t["payout"], t["cost"], t["pnl"], won))
        print("    TOTAL: $%+.2f" % old.total_pnl)
    else:
        print("    (no trades)")

    print()
    print("  NEW MODEL TRADES:")
    if new.blocked:
        print("    BLOCKED: %s" % new.block_reason)
        print("    TOTAL: $0.00 (no exposure)")
    elif new.trade_results:
        for t in new.trade_results:
            won = "WON" if t["won"] else "LOST"
            print("    %s %s %dx @ %dc: payout=$%.2f cost=$%.2f PnL=$%+.2f [%s]" % (
                t["side"][:7].upper(), t["ticker"].split("-")[-1],
                t["contracts"], t["entry_cents"],
                t["payout"], t["cost"], t["pnl"], won))
        print("    TOTAL: $%+.2f" % new.total_pnl)
    else:
        print("    (no signals above dynamic threshold)")
        print("    TOTAL: $0.00")

    # Verdict
    print()
    print("=" * 75)
    print("VERDICT")
    print("=" * 75)

    diff = new.total_pnl - old.total_pnl
    if new.blocked or not new.signals:
        if old.total_pnl < 0:
            print("  NEW MODEL AVOIDED the $%.2f loss!" % abs(old.total_pnl))
            print("  Reason: %s" % (new.block_reason or "No signals above dynamic edge threshold"))
            print("  Savings: $%.2f" % abs(old.total_pnl))
        else:
            print("  NEW MODEL would have missed $%.2f profit." % old.total_pnl)
            print("  But this is the cost of safety gates -- fewer trades, higher quality.")
    elif diff > 0:
        print("  NEW MODEL outperformed by $%.2f ($%+.2f vs $%+.2f)" % (
            diff, new.total_pnl, old.total_pnl))
    elif diff < 0:
        print("  OLD MODEL outperformed by $%.2f ($%+.2f vs $%+.2f)" % (
            abs(diff), old.total_pnl, new.total_pnl))
    else:
        print("  Both models had the same outcome: $%+.2f" % old.total_pnl)

    # Probability analysis
    print()
    print("PROBABILITY ANALYSIS (for bucket containing actual=%dF):" % actual_temp)
    for b_label, model in [("OLD", old), ("NEW", new)]:
        # Find relevant bucket probability from signals or compute
        from model import Bucket as BucketClass
        # Reconstruct actual bucket
        actual_bucket_low = actual_temp - (actual_temp % 2)
        actual_bucket_high = actual_bucket_low + 1
        p = phi((actual_bucket_high + 0.5 - model.mu) / model.sigma) - \
            phi((actual_bucket_low - 0.5 - model.mu) / model.sigma)
        print("  %s: P(%d-%d) = %.1f%%  (μ=%.1f, σ=%.2f)" % (
            b_label, actual_bucket_low, actual_bucket_high,
            p * 100, model.mu, model.sigma))

    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay a trading day: old model vs new model")
    parser.add_argument("--date", required=True,
                        help="Target date (YYYY-MM-DD)")
    parser.add_argument("--actual", type=int, default=None,
                        help="Override actual TMAX (F). If not given, reads GHCN.")
    parser.add_argument("--hour", type=int, default=3,
                        help="Simulated ET hour for the trade (default: 3 = 3 AM)")
    parser.add_argument("--ensemble-sigma", type=float, default=None,
                        help="Override ensemble σ (default: estimate from historical)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    target_date_str = args.date
    target_date = date.fromisoformat(target_date_str)
    hour_et = args.hour

    # Load paper trade signals for this date
    signals = load_paper_signals(target_date_str)
    if not signals:
        print("No paper trade signals found for %s in paper_trades.jsonl" % target_date_str)
        print("Searching for any date-matching signals...")
        # Try matching by ticker date portion
        all_signals = []
        path = config.PAPER_TRADES_PATH
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            if rec.get("record_type") == "signal":
                                all_signals.append(rec)
                        except json.JSONDecodeError:
                            pass
        if all_signals:
            dates = sorted(set(s.get("target_date", "?") for s in all_signals))
            print("Available dates: %s" % ", ".join(dates))
        return

    # Reconstruct market state
    buckets, market_prices = reconstruct_buckets_from_signals(signals)

    if not buckets:
        print("Could not reconstruct any buckets from signals")
        return

    # Get actual temperature
    actual_temp = args.actual
    if actual_temp is None:
        actual_temp = get_actual_tmax(target_date)
    if actual_temp is None:
        print("No actual TMAX found for %s. Use --actual to specify." % target_date_str)
        return

    # Get forecast from signal records
    forecast_high = signals[0].get("forecast_high")
    if forecast_high is None:
        forecast_high = int(round(signals[0].get("mu", 35)))

    print("Reconstructed state:")
    print("  Date: %s" % target_date_str)
    print("  Forecast: %dF" % forecast_high)
    print("  Actual: %dF" % actual_temp)
    print("  Buckets: %d" % len(buckets))
    print("  Market prices: %s" % {t.split("-")[-1]: "%.0fc" % (p * 100) for t, p in market_prices.items()})
    print("  Simulated hour: %d ET" % hour_et)
    print("  Signals in log: %d" % len(signals))

    # Run both models
    old_result = run_old_model(buckets, market_prices, forecast_high,
                               actual_temp, hour_et)
    new_result = run_new_model(buckets, market_prices, forecast_high,
                               actual_temp, hour_et,
                               ensemble_sigma=args.ensemble_sigma)

    # Print comparison
    print_replay_report(target_date_str, actual_temp, old_result, new_result, signals)


if __name__ == "__main__":
    main()

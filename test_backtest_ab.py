"""
test_backtest_ab.py -- A/B/C Backtest: Three Model Variants.

Runs the backtest three times:
  A) Old model: σ=1.2, flat 8% edge, no confidence gates
  B) Overfit model (v1): σ=1.43, dynamic edge, confidence gates, boundary boost
  C) NEW model (v2): σ=max(1.2, α×σ_ens), flat 8% edge, no gates, half-Kelly sizing

Uses the same 611K trade history + GHCN observations.
Compares total PnL, win rate, number of trades, and max drawdown.

Usage:
    py -X utf8 test_backtest_ab.py
    py -X utf8 test_backtest_ab.py --sigma-ens 1.3
"""

import sys
import math
import argparse
import logging
import random
import hashlib
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from confidence import (compute_confidence, compute_dynamic_min_edge,
                        compute_boundary_z, extract_bucket_boundaries)

from backtest import (
    phi, BucketDef, parse_ticker_to_bucket, resolve_threshold_directions,
    compute_fee_cents, load_trades, load_forecasts, load_observations,
    reconstruct_market_prices, SimulatedTrade,
)

logger = logging.getLogger(__name__)


# ─── Backtest Runner ─────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    name: str
    sigma: float
    min_edge: float
    forecast_bias: float
    use_confidence_gates: bool    # v1 only: block trading on low confidence
    use_dynamic_edge: bool        # v1 only: time-of-day edge scaling
    boundary_z_boost: bool        # v1 only: inflate σ near bucket edges
    use_kelly: bool               # v2: half-Kelly position sizing
    bankroll: float
    maker_fill_rate: float
    slippage_cents: int           # additional adverse slippage over baseline maker posting
    max_risk_per_trade: float
    min_contracts: int
    variable_sigma: bool = False  # v2: draw per-day σ_ens from simulated distribution
    historical_sigma: bool = False # v2: use real historical ensemble σ from parquet
    sigma_base: float = 1.2      # base σ for variable/historical mode


def _per_day_sigma_ens(target_date, sigma_base, alpha):
    """
    Draw a realistic per-day σ_ens from a deterministic distribution.

    Based on observed ensemble spreads: σ_ens ranges ~0.7-2.5F,
    log-normal with median ~1.1F.  On low-spread days V2 falls back
    to σ_base (same as Old); on high-spread days V2 widens properly.
    """
    # Deterministic seed from date
    seed_val = int(hashlib.sha256(
        ("sigma_ens|" + str(target_date)).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed_val)

    # Log-normal: median=1.1, ~30% of days < 1.09 (V2=Old), ~15% > 2.0
    log_mu = math.log(1.1)
    log_sigma = 0.35
    sigma_ens = rng.lognormvariate(log_mu, log_sigma)

    # Apply compose: max(sigma_base, alpha * sigma_ens)
    return max(sigma_base, alpha * sigma_ens)


_HISTORICAL_SIGMA = None  # loaded on demand

def _load_historical_sigma():
    """Load real per-day σ_ens from ensemble_history.parquet."""
    global _HISTORICAL_SIGMA
    if _HISTORICAL_SIGMA is not None:
        return _HISTORICAL_SIGMA

    import pandas as pd
    path = PROJECT_ROOT / "data" / "curated" / "ensemble_history.parquet"
    if not path.exists():
        logger.warning("No ensemble_history.parquet found -- run fetch_historical_ensemble.py first")
        return {}

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    _HISTORICAL_SIGMA = {}
    for _, row in df.iterrows():
        _HISTORICAL_SIGMA[row["date"]] = row["sigma_v2"]  # pre-composed σ
    logger.info("Loaded historical σ for %d dates", len(_HISTORICAL_SIGMA))
    return _HISTORICAL_SIGMA


def kelly_contracts(edge, market_price, side, price_cents, bankroll, max_risk):
    """
    Half-Kelly position sizing.
    f = edge / payout_complement, then half it.
    """
    price_dollars = price_cents / 100.0
    if price_dollars <= 0:
        return 0

    if side == "buy_yes":
        payout_complement = 1.0 - market_price
    else:
        payout_complement = market_price

    if payout_complement <= 0:
        return 0

    kelly_f = edge / payout_complement * 0.5  # half-Kelly
    kelly_dollars = min(kelly_f * bankroll, max_risk)
    kelly_dollars = max(kelly_dollars, 0)

    contracts = int(kelly_dollars / price_dollars) if price_dollars > 0 else 0
    return contracts


def _stable_uniform(*parts):
    """
    Deterministic pseudo-random in [0,1) from string parts.

    Avoids Python's process-randomized hash() so backtests are reproducible.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed).random()


def run_backtest(cfg, trades_df, forecast_lookup, obs_lookup):
    """Run backtest with given config. Returns (trades, skipped)."""
    import pandas as pd

    trade_dates = sorted(trades_df["trade_date"].unique())
    simulated_trades = []
    skipped = {"no_forecast": 0, "no_observation": 0, "no_prices": 0,
               "no_signals": 0, "confidence_blocked": 0}

    for target_date in trade_dates:
        forecast = forecast_lookup.get(target_date)
        if forecast is None:
            skipped["no_forecast"] += 1
            continue

        actual = obs_lookup.get(target_date)
        if actual is None:
            skipped["no_observation"] += 1
            continue

        mu = forecast - cfg.forecast_bias
        if cfg.historical_sigma:
            hist = _load_historical_sigma()
            sigma = hist.get(target_date, cfg.sigma_base)  # fallback to base if no data
        elif cfg.variable_sigma:
            sigma = _per_day_sigma_ens(target_date, cfg.sigma_base, config.ENSEMBLE_ALPHA)
        else:
            sigma = cfg.sigma

        market_prices = reconstruct_market_prices(trades_df, target_date)
        if not market_prices:
            skipped["no_prices"] += 1
            continue

        buckets = {}
        for ticker in market_prices:
            b = parse_ticker_to_bucket(ticker)
            if b:
                buckets[ticker] = b

        events = {}
        for ticker, b in buckets.items():
            events.setdefault(b.event_ticker, []).append(b)
        for event_buckets in events.values():
            resolve_threshold_directions(event_buckets)

        # V1 model: confidence gates + dynamic edge + boundary boost
        min_edge = cfg.min_edge
        if cfg.use_confidence_gates or cfg.use_dynamic_edge or cfg.boundary_z_boost:
            hour_et = 9
            bucket_list = list(buckets.values())
            boundary_vals = set()
            for b in bucket_list:
                if b.low is not None:
                    boundary_vals.add(b.low)
                    boundary_vals.add(b.low - 0.5)
                if b.high is not None:
                    boundary_vals.add(b.high)
                    boundary_vals.add(b.high + 0.5)
            boundary_z = None
            if boundary_vals and sigma > 0:
                min_dist = min(abs(mu - bv) for bv in boundary_vals)
                boundary_z = min_dist / sigma

            if cfg.boundary_z_boost and boundary_z is not None and boundary_z < 1.5:
                boundary_risk = max(0, 1.5 - boundary_z)
                sigma = max(sigma, sigma + config.ENSEMBLE_GAMMA * boundary_risk)

            confidence, _ = compute_confidence(60, 30, 0.3, boundary_z, hour_et)

            if cfg.use_confidence_gates and confidence < config.MIN_CONFIDENCE_TO_TRADE:
                skipped["confidence_blocked"] += 1
                continue

            if cfg.use_dynamic_edge:
                min_edge = compute_dynamic_min_edge(hour_et, confidence)

        # Find signals
        signals = []
        for ticker, bucket in buckets.items():
            price = market_prices.get(ticker)
            if price is None or price < 0.03 or price > 0.97:
                continue

            prob = bucket.probability(mu, sigma)

            yes_edge = prob - price
            if yes_edge > min_edge:
                signals.append(("buy_yes", ticker, bucket, prob, price, yes_edge))

            no_edge = (1.0 - prob) - (1.0 - price)
            if no_edge > min_edge:
                signals.append(("buy_no", ticker, bucket, prob, price, no_edge))

        if not signals:
            skipped["no_signals"] += 1
            continue

        signals.sort(key=lambda s: s[5], reverse=True)
        signals = signals[:3]

        for side, ticker, bucket, prob, price, edge in signals:
            fill_draw = _stable_uniform(target_date, ticker, side)
            if fill_draw > cfg.maker_fill_rate:
                continue

            if side == "buy_yes":
                base_entry = int(price * 100) - 1
            else:
                no_price = 1.0 - price
                base_entry = int(no_price * 100) - 1

            entry_cents = max(1, min(99, base_entry + max(0, int(cfg.slippage_cents))))

            cost_per_contract = entry_cents / 100.0

            # Position sizing
            if cfg.use_kelly:
                contracts = kelly_contracts(
                    edge, price, side, entry_cents,
                    cfg.bankroll, cfg.max_risk_per_trade)
                contracts = max(cfg.min_contracts, min(contracts, 50))
            else:
                max_c = int(cfg.max_risk_per_trade / cost_per_contract) if cost_per_contract > 0 else 0
                contracts = max(cfg.min_contracts, min(max_c, 50))

            while contracts * cost_per_contract > cfg.max_risk_per_trade and contracts > 0:
                contracts -= 1
            if contracts < cfg.min_contracts:
                continue

            fee_dollars = compute_fee_cents(entry_cents, contracts) / 100.0
            bucket_hit = bucket.settles_yes(actual)

            if side == "buy_yes":
                payout = 1.00 * contracts if bucket_hit else 0.0
            else:
                payout = 1.00 * contracts if not bucket_hit else 0.0

            cost = cost_per_contract * contracts + fee_dollars
            pnl = payout - cost

            simulated_trades.append(SimulatedTrade(
                target_date=target_date, ticker=ticker, side=side,
                model_prob=prob, market_price=price, edge=edge,
                entry_price_cents=entry_cents, contracts=contracts,
                fee_dollars=fee_dollars, actual_temp=actual,
                bucket_hit=bucket_hit, payout_dollars=payout,
                cost_dollars=cost, pnl_dollars=pnl,
            ))

    return simulated_trades, skipped


# ─── Stats + Reporting ───────────────────────────────────────────────────────

def compute_stats(trades):
    if not trades:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "total_fees": 0, "avg_pnl": 0,
                "max_drawdown": 0, "unique_days": 0, "avg_edge": 0,
                "brier": 0, "sharpe": 0}

    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    total_pnl = sum(t.pnl_dollars for t in trades)
    total_fees = sum(t.fee_dollars for t in trades)
    unique_days = len(set(t.target_date for t in trades))
    avg_edge = sum(t.edge for t in trades) / n

    # Max drawdown
    equity = [0.0]
    for t in sorted(trades, key=lambda t: t.target_date):
        equity.append(equity[-1] + t.pnl_dollars)
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        max_dd = max(max_dd, peak - val)

    # Brier
    brier_sum = 0
    for t in trades:
        outcome = (1.0 if t.bucket_hit else 0.0) if t.side == "buy_yes" else (0.0 if t.bucket_hit else 1.0)
        p = t.model_prob if t.side == "buy_yes" else (1.0 - t.model_prob)
        brier_sum += (p - outcome) ** 2
    brier = brier_sum / n

    # Daily PnL for Sharpe
    daily = defaultdict(float)
    for t in trades:
        daily[t.target_date] += t.pnl_dollars
    daily_pnls = list(daily.values())
    if len(daily_pnls) > 1:
        import statistics as st
        sharpe = st.mean(daily_pnls) / st.stdev(daily_pnls) * (252 ** 0.5) if st.stdev(daily_pnls) > 0 else 0
    else:
        sharpe = 0

    return {
        "n": n, "wins": wins, "losses": n - wins,
        "win_rate": wins / n if n > 0 else 0,
        "total_pnl": total_pnl, "total_fees": total_fees,
        "avg_pnl": total_pnl / n if n > 0 else 0,
        "max_drawdown": max_dd, "unique_days": unique_days,
        "avg_edge": avg_edge, "brier": brier, "sharpe": sharpe,
    }


def print_report(results):
    """Print multi-model comparison report."""
    if results:
        cfg0 = results[0][1]
        print("Execution assumptions: fill_rate=%.2f, extra_slippage=%dc" % (
            cfg0.maker_fill_rate, cfg0.slippage_cents))
        print()
    print("=" * 85)
    print("A/B/C BACKTEST COMPARISON")
    print("=" * 85)
    print()

    for name, cfg, _, _ in results:
        extras = []
        if cfg.use_confidence_gates:
            extras.append("confidence gates")
        if cfg.use_dynamic_edge:
            extras.append("dynamic edge")
        if cfg.boundary_z_boost:
            extras.append("boundary boost")
        if cfg.use_kelly:
            extras.append("half-Kelly sizing")
        extra_str = " + ".join(extras) if extras else "none"
        print("  %s: σ=%.2f, edge=%.0f%%, extras=[%s]" % (
            name, cfg.sigma, cfg.min_edge * 100, extra_str))
    print()

    # Table
    header = "%-30s" % "Metric"
    for name, _, _, _ in results:
        header += " | %-18s" % name[:18]
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    rows = [
        ("Total trades", lambda s: "%d" % s["n"]),
        ("Trading days", lambda s: "%d" % s["unique_days"]),
        ("Win rate", lambda s: "%.0f%% (%d/%d)" % (s["win_rate"] * 100, s["wins"], s["n"])),
        ("Total PnL", lambda s: "$%+.2f" % s["total_pnl"]),
        ("Avg PnL/trade", lambda s: "$%+.2f" % s["avg_pnl"]),
        ("Avg edge", lambda s: "%.1f%%" % (s["avg_edge"] * 100)),
        ("Max drawdown", lambda s: "$%.2f" % s["max_drawdown"]),
        ("Brier score", lambda s: "%.4f" % s["brier"]),
        ("Annualized Sharpe", lambda s: "%.2f" % s["sharpe"]),
        ("Total fees", lambda s: "$%.2f" % s["total_fees"]),
    ]

    for label, fmt in rows:
        row = "%-30s" % label
        for _, _, _, stats in results:
            row += " | %-18s" % fmt(stats)
        print(row)

    print("-" * len(header))
    print()

    # Verdict
    best = max(results, key=lambda r: r[3]["total_pnl"])
    best_name = best[0]
    best_pnl = best[3]["total_pnl"]

    print("=" * 85)
    print("VERDICT")
    print("=" * 85)

    for name, _, _, stats in results:
        diff = stats["total_pnl"] - best_pnl
        if name == best_name:
            print("  [WINNER] %s: $%+.2f PnL" % (name, stats["total_pnl"]))
        else:
            print("  %s: $%+.2f PnL (%.0f%% vs winner)" % (
                name, stats["total_pnl"],
                stats["total_pnl"] / best_pnl * 100 if best_pnl != 0 else 0))

    # Compare A vs E (historical σ) specifically
    a_stats = results[0][3]
    e_stats = results[-1][3]  # E is last
    trade_diff = e_stats["n"] - a_stats["n"]
    pnl_diff = e_stats["total_pnl"] - a_stats["total_pnl"]
    dd_diff = e_stats["max_drawdown"] - a_stats["max_drawdown"]

    print()
    print("  Old (A) vs V2-historical (E):")
    if trade_diff != 0:
        print("    Trades: %+d (%+.0f%%)" % (trade_diff, trade_diff / a_stats["n"] * 100))
    if pnl_diff >= 0:
        print("    PnL: +$%.2f improvement" % pnl_diff)
    else:
        print("    PnL: -$%.2f (%.1f%% of total)" % (abs(pnl_diff), abs(pnl_diff) / a_stats["total_pnl"] * 100))
    print("    Drawdown: $%.2f -> $%.2f (%+.0f%%)" % (
        a_stats["max_drawdown"], e_stats["max_drawdown"],
        dd_diff / a_stats["max_drawdown"] * 100 if a_stats["max_drawdown"] > 0 else 0))
    if e_stats["win_rate"] > a_stats["win_rate"]:
        print("    Win rate: IMPROVED %.0f%% -> %.0f%%" % (
            a_stats["win_rate"] * 100, e_stats["win_rate"] * 100))
    elif e_stats["win_rate"] < a_stats["win_rate"]:
        print("    Win rate: DROPPED %.0f%% -> %.0f%%" % (
            a_stats["win_rate"] * 100, e_stats["win_rate"] * 100))

    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A/B/C Backtest")
    parser.add_argument("--sigma-ens", type=float, default=1.3,
                        help="Estimated average ensemble σ for V2 (default: 1.3)")
    parser.add_argument("--fill-rate", type=float, default=0.70)
    parser.add_argument("--slippage", type=int, default=0,
                        help="Additional adverse slippage cents over baseline maker posting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 85)
    print("A/B/C BACKTEST -- Kalshi Weather Bot")
    print("=" * 85)
    print()
    print("Loading data...")

    import pandas as pd

    trades_df = load_trades()
    forecasts_df = load_forecasts()
    obs_df = load_observations()

    if trades_df is None or forecasts_df is None or obs_df is None:
        print("Missing data files.")
        return

    trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"]).dt.date

    forecast_lookup = {}
    for _, row in forecasts_df.iterrows():
        forecast_lookup[row["date"]] = row["forecast_high_f"]

    obs_lookup = {}
    for _, row in obs_df.iterrows():
        obs_lookup[row["date"]] = int(row["tmax_f"])

    print("  Trades: %d" % len(trades_df))
    print("  Forecast days: %d" % len(forecast_lookup))
    print("  Observation days: %d" % len(obs_lookup))
    print()

    # V2 sigma: max(1.2, α × σ_ens)
    sigma_v2 = max(1.2, config.ENSEMBLE_ALPHA * args.sigma_ens)
    print("V2 sigma: max(1.2, %.1f × %.2f) = %.2f" % (
        config.ENSEMBLE_ALPHA, args.sigma_ens, sigma_v2))
    print()

    configs = [
        BacktestConfig(
            name="A: Old (σ=1.2, 8%)",
            sigma=1.2, min_edge=0.08, forecast_bias=0.0,
            use_confidence_gates=False, use_dynamic_edge=False,
            boundary_z_boost=False, use_kelly=False,
            bankroll=50.0, maker_fill_rate=args.fill_rate,
            slippage_cents=args.slippage,
            max_risk_per_trade=5.0, min_contracts=5,
        ),
        BacktestConfig(
            name="B: V1 (gates+dynamic)",
            sigma=1.43, min_edge=0.10, forecast_bias=0.0,
            use_confidence_gates=True, use_dynamic_edge=True,
            boundary_z_boost=True, use_kelly=False,
            bankroll=50.0, maker_fill_rate=args.fill_rate,
            slippage_cents=args.slippage,
            max_risk_per_trade=5.0, min_contracts=5,
        ),
        BacktestConfig(
            name="C: V2-fixed (ens σ+Kelly)",
            sigma=sigma_v2, min_edge=0.08, forecast_bias=0.0,
            use_confidence_gates=False, use_dynamic_edge=False,
            boundary_z_boost=False, use_kelly=True,
            bankroll=50.0, maker_fill_rate=args.fill_rate,
            slippage_cents=args.slippage,
            max_risk_per_trade=5.0, min_contracts=5,
        ),
        BacktestConfig(
            name="D: V2-var (daily σ+Kelly)",
            sigma=0.0, min_edge=0.08, forecast_bias=0.0,
            use_confidence_gates=False, use_dynamic_edge=False,
            boundary_z_boost=False, use_kelly=True,
            bankroll=50.0, maker_fill_rate=args.fill_rate,
            slippage_cents=args.slippage,
            max_risk_per_trade=5.0, min_contracts=5,
            variable_sigma=True, sigma_base=1.2,
        ),
        BacktestConfig(
            name="E: V2-hist (real σ+Kelly)",
            sigma=0.0, min_edge=0.08, forecast_bias=0.0,
            use_confidence_gates=False, use_dynamic_edge=False,
            boundary_z_boost=False, use_kelly=True,
            bankroll=50.0, maker_fill_rate=args.fill_rate,
            slippage_cents=args.slippage,
            max_risk_per_trade=5.0, min_contracts=5,
            historical_sigma=True, sigma_base=1.2,
        ),
    ]

    results = []
    for cfg in configs:
        print("Running %s..." % cfg.name)
        trades, skipped = run_backtest(cfg, trades_df, forecast_lookup, obs_lookup)
        stats = compute_stats(trades)
        results.append((cfg.name, cfg, skipped, stats))
        print("  %d trades, PnL=$%+.2f, win=%.0f%%" % (
            stats["n"], stats["total_pnl"], stats["win_rate"] * 100))

    print_report(results)


if __name__ == "__main__":
    main()

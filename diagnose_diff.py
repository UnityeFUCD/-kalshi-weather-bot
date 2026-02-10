"""
diagnose_diff.py -- Find exactly where Old beats V2 trade-by-trade.

Runs both models, then categorizes every trade into:
  1. BOTH_WIN:  Old and V2 both take the trade and both win
  2. BOTH_LOSE: Old and V2 both take the trade and both lose
  3. OLD_ONLY_WIN:  Old takes it and wins, V2 skips it
  4. OLD_ONLY_LOSE: Old takes it and loses, V2 skips it
  5. V2_ONLY_WIN:  V2 takes it and wins, Old skips it
  6. V2_ONLY_LOSE: V2 takes it and loses, Old skips it
  7. DISAGREE: Both take but different outcome (position sizing difference)
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from test_backtest_ab import BacktestConfig, run_backtest, compute_stats, _stable_uniform
from backtest import load_trades, load_forecasts, load_observations

logging.basicConfig(level=logging.WARNING)


def main():
    import pandas as pd

    trades_df = load_trades()
    forecasts_df = load_forecasts()
    obs_df = load_observations()

    if trades_df is None or forecasts_df is None or obs_df is None:
        print("Missing data.")
        return

    trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"]).dt.date

    forecast_lookup = {}
    for _, row in forecasts_df.iterrows():
        forecast_lookup[row["date"]] = row["forecast_high_f"]

    obs_lookup = {}
    for _, row in obs_df.iterrows():
        obs_lookup[row["date"]] = int(row["tmax_f"])

    sigma_v2 = max(1.2, config.ENSEMBLE_ALPHA * 1.3)

    old_cfg = BacktestConfig(
        name="Old", sigma=1.2, min_edge=0.08, forecast_bias=0.0,
        use_confidence_gates=False, use_dynamic_edge=False,
        boundary_z_boost=False, use_kelly=False,
        bankroll=50.0, maker_fill_rate=0.70, slippage_cents=0,
        max_risk_per_trade=5.0, min_contracts=5,
    )
    v2_cfg = BacktestConfig(
        name="V2-fixed", sigma=sigma_v2, min_edge=0.08, forecast_bias=0.0,
        use_confidence_gates=False, use_dynamic_edge=False,
        boundary_z_boost=False, use_kelly=True,
        bankroll=50.0, maker_fill_rate=0.70, slippage_cents=0,
        max_risk_per_trade=5.0, min_contracts=5,
    )

    old_trades, _ = run_backtest(old_cfg, trades_df, forecast_lookup, obs_lookup)
    v2_trades, _ = run_backtest(v2_cfg, trades_df, forecast_lookup, obs_lookup)

    # Index by (date, ticker, side)
    old_by_key = {}
    for t in old_trades:
        key = (str(t.target_date), t.ticker, t.side)
        old_by_key[key] = t

    v2_by_key = {}
    for t in v2_trades:
        key = (str(t.target_date), t.ticker, t.side)
        v2_by_key[key] = t

    all_keys = set(old_by_key.keys()) | set(v2_by_key.keys())

    categories = defaultdict(list)

    for key in sorted(all_keys):
        old_t = old_by_key.get(key)
        v2_t = v2_by_key.get(key)

        if old_t and v2_t:
            if old_t.won and v2_t.won:
                categories["BOTH_WIN"].append((key, old_t, v2_t))
            elif not old_t.won and not v2_t.won:
                categories["BOTH_LOSE"].append((key, old_t, v2_t))
            else:
                categories["DISAGREE"].append((key, old_t, v2_t))
        elif old_t and not v2_t:
            if old_t.won:
                categories["OLD_ONLY_WIN"].append((key, old_t, None))
            else:
                categories["OLD_ONLY_LOSE"].append((key, old_t, None))
        elif v2_t and not old_t:
            if v2_t.won:
                categories["V2_ONLY_WIN"].append((key, None, v2_t))
            else:
                categories["V2_ONLY_LOSE"].append((key, None, v2_t))

    print("=" * 90)
    print("TRADE-BY-TRADE DIAGNOSIS: OLD vs V2-fixed")
    print("=" * 90)
    print()
    print("Old: %d trades, %d wins, PnL=$%.2f" % (
        len(old_trades), sum(1 for t in old_trades if t.won),
        sum(t.pnl_dollars for t in old_trades)))
    print("V2:  %d trades, %d wins, PnL=$%.2f" % (
        len(v2_trades), sum(1 for t in v2_trades if t.won),
        sum(t.pnl_dollars for t in v2_trades)))
    print()

    for cat in ["BOTH_WIN", "BOTH_LOSE", "OLD_ONLY_WIN", "OLD_ONLY_LOSE",
                "V2_ONLY_WIN", "V2_ONLY_LOSE", "DISAGREE"]:
        items = categories.get(cat, [])
        old_pnl = sum((t[1].pnl_dollars if t[1] else 0) for t in items)
        v2_pnl = sum((t[2].pnl_dollars if t[2] else 0) for t in items)
        print("%-16s  count=%-4d  old_pnl=$%+8.2f  v2_pnl=$%+8.2f  delta=$%+8.2f" % (
            cat, len(items), old_pnl, v2_pnl, v2_pnl - old_pnl))
    print()

    # Deep dive: OLD_ONLY_WIN — trades Old takes and wins but V2 skips
    print("=" * 90)
    print("OLD_ONLY_WIN — trades Old takes and wins, V2 MISSES (V2 loses this PnL)")
    print("=" * 90)
    for key, old_t, _ in categories.get("OLD_ONLY_WIN", []):
        print("  %s | %s %s | edge=%.1f%% prob=%.1f%% mkt=%.0fc | %dx @ %dc | pnl=$%+.2f" % (
            key[0], key[2].upper(), key[1].split("-")[-1],
            old_t.edge * 100, old_t.model_prob * 100, old_t.market_price * 100,
            old_t.contracts, old_t.entry_price_cents, old_t.pnl_dollars))
    print()

    # V2_ONLY_WIN — trades V2 takes and wins but Old skips
    print("=" * 90)
    print("V2_ONLY_WIN — trades V2 takes and wins, Old MISSES")
    print("=" * 90)
    for key, _, v2_t in categories.get("V2_ONLY_WIN", []):
        print("  %s | %s %s | edge=%.1f%% prob=%.1f%% mkt=%.0fc | %dx @ %dc | pnl=$%+.2f" % (
            key[0], key[2].upper(), key[1].split("-")[-1],
            v2_t.edge * 100, v2_t.model_prob * 100, v2_t.market_price * 100,
            v2_t.contracts, v2_t.entry_price_cents, v2_t.pnl_dollars))
    print()

    # OLD_ONLY_LOSE — trades Old takes and loses, V2 wisely skips
    print("=" * 90)
    print("OLD_ONLY_LOSE — trades Old takes and LOSES, V2 wisely skips")
    print("=" * 90)
    for key, old_t, _ in categories.get("OLD_ONLY_LOSE", []):
        print("  %s | %s %s | edge=%.1f%% prob=%.1f%% mkt=%.0fc | %dx @ %dc | pnl=$%+.2f | actual=%d" % (
            key[0], key[2].upper(), key[1].split("-")[-1],
            old_t.edge * 100, old_t.model_prob * 100, old_t.market_price * 100,
            old_t.contracts, old_t.entry_price_cents, old_t.pnl_dollars,
            old_t.actual_temp))
    print()

    # V2_ONLY_LOSE — trades V2 takes and loses that Old avoids
    print("=" * 90)
    print("V2_ONLY_LOSE — trades V2 takes and LOSES, Old wisely skips")
    print("=" * 90)
    for key, _, v2_t in categories.get("V2_ONLY_LOSE", []):
        print("  %s | %s %s | edge=%.1f%% prob=%.1f%% mkt=%.0fc | %dx @ %dc | pnl=$%+.2f | actual=%d" % (
            key[0], key[2].upper(), key[1].split("-")[-1],
            v2_t.edge * 100, v2_t.model_prob * 100, v2_t.market_price * 100,
            v2_t.contracts, v2_t.entry_price_cents, v2_t.pnl_dollars,
            v2_t.actual_temp))
    print()

    # BOTH trades but different PnL (position sizing diff from Kelly)
    print("=" * 90)
    print("BOTH_WIN — PnL difference from Kelly sizing")
    print("=" * 90)
    both_win = categories.get("BOTH_WIN", [])
    old_both_pnl = sum(t[1].pnl_dollars for t in both_win)
    v2_both_pnl = sum(t[2].pnl_dollars for t in both_win)
    print("  Old total: $%.2f across %d trades" % (old_both_pnl, len(both_win)))
    print("  V2 total:  $%.2f across %d trades" % (v2_both_pnl, len(both_win)))
    print("  Delta:     $%+.2f (Kelly sizing effect)" % (v2_both_pnl - old_both_pnl))
    print()

    # Biggest sizing differences
    sizing_diffs = []
    for key, old_t, v2_t in both_win:
        diff = v2_t.pnl_dollars - old_t.pnl_dollars
        sizing_diffs.append((diff, key, old_t, v2_t))
    sizing_diffs.sort(key=lambda x: x[0])

    if sizing_diffs:
        print("  Biggest V2 sizing LOSSES vs Old (Kelly undersized):")
        for diff, key, old_t, v2_t in sizing_diffs[:5]:
            print("    %s %s %s: old=%dx v2=%dx pnl_diff=$%+.2f (edge=%.1f%%)" % (
                key[0], key[2], key[1].split("-")[-1],
                old_t.contracts, v2_t.contracts, diff, old_t.edge * 100))
        print()
        print("  Biggest V2 sizing GAINS vs Old (Kelly oversized):")
        for diff, key, old_t, v2_t in sizing_diffs[-5:]:
            print("    %s %s %s: old=%dx v2=%dx pnl_diff=$%+.2f (edge=%.1f%%)" % (
                key[0], key[2], key[1].split("-")[-1],
                old_t.contracts, v2_t.contracts, diff, old_t.edge * 100))

    print()

    # BOTH_LOSE sizing
    both_lose = categories.get("BOTH_LOSE", [])
    if both_lose:
        old_lose_pnl = sum(t[1].pnl_dollars for t in both_lose)
        v2_lose_pnl = sum(t[2].pnl_dollars for t in both_lose)
        print("=" * 90)
        print("BOTH_LOSE — Kelly sizing effect on losses")
        print("=" * 90)
        print("  Old total: $%.2f across %d trades" % (old_lose_pnl, len(both_lose)))
        print("  V2 total:  $%.2f across %d trades" % (v2_lose_pnl, len(both_lose)))
        print("  Delta:     $%+.2f (positive = V2 lost less)" % (v2_lose_pnl - old_lose_pnl))

    # Summary
    print()
    print("=" * 90)
    print("ROOT CAUSE SUMMARY")
    print("=" * 90)

    old_only_win_pnl = sum(t[1].pnl_dollars for t in categories.get("OLD_ONLY_WIN", []))
    old_only_lose_pnl = sum(t[1].pnl_dollars for t in categories.get("OLD_ONLY_LOSE", []))
    v2_only_win_pnl = sum(t[2].pnl_dollars for t in categories.get("V2_ONLY_WIN", []))
    v2_only_lose_pnl = sum(t[2].pnl_dollars for t in categories.get("V2_ONLY_LOSE", []))

    print()
    print("  SIGNAL SELECTION (different trades taken):")
    print("    Old takes, V2 skips — WINS:  %d trades, $%+.2f" % (
        len(categories.get("OLD_ONLY_WIN", [])), old_only_win_pnl))
    print("    Old takes, V2 skips — LOSES: %d trades, $%+.2f" % (
        len(categories.get("OLD_ONLY_LOSE", [])), old_only_lose_pnl))
    print("    V2 takes, Old skips — WINS:  %d trades, $%+.2f" % (
        len(categories.get("V2_ONLY_WIN", [])), v2_only_win_pnl))
    print("    V2 takes, Old skips — LOSES: %d trades, $%+.2f" % (
        len(categories.get("V2_ONLY_LOSE", [])), v2_only_lose_pnl))
    print("    Net signal selection effect:  $%+.2f" % (
        (v2_only_win_pnl + v2_only_lose_pnl) - (old_only_win_pnl + old_only_lose_pnl)))
    print()
    print("  POSITION SIZING (same trades, different size via Kelly):")
    both_all = categories.get("BOTH_WIN", []) + categories.get("BOTH_LOSE", []) + categories.get("DISAGREE", [])
    sizing_effect = sum((t[2].pnl_dollars if t[2] else 0) - (t[1].pnl_dollars if t[1] else 0) for t in both_all)
    print("    Kelly sizing PnL delta: $%+.2f" % sizing_effect)


if __name__ == "__main__":
    main()

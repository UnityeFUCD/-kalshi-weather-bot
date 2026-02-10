"""
test_robustness_sweep.py -- Stress test old vs new model under execution assumptions.

Varies:
- Maker fill rate
- Additional adverse slippage cents

Purpose:
- Check if new model edge survives more conservative execution conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
from backtest import load_forecasts, load_observations, load_trades
from test_backtest_ab import BacktestConfig, compute_stats, run_backtest, _per_day_sigma_ens


def _build_lookups(forecasts_df, obs_df):
    forecast_lookup = {}
    for _, row in forecasts_df.iterrows():
        forecast_lookup[row["date"]] = row["forecast_high_f"]

    obs_lookup = {}
    for _, row in obs_df.iterrows():
        obs_lookup[row["date"]] = int(row["tmax_f"])
    return forecast_lookup, obs_lookup


def _scenario_configs(fill_rate, slippage, sigma_v2):
    old_cfg = BacktestConfig(
        name="A: Old (sigma=1.2, 8%)",
        sigma=1.2,
        min_edge=0.08,
        forecast_bias=0.0,
        use_confidence_gates=False,
        use_dynamic_edge=False,
        boundary_z_boost=False,
        use_kelly=False,
        bankroll=50.0,
        maker_fill_rate=fill_rate,
        slippage_cents=slippage,
        max_risk_per_trade=5.0,
        min_contracts=5,
    )
    new_fixed_cfg = BacktestConfig(
        name="C: V2-fixed",
        sigma=sigma_v2,
        min_edge=0.08,
        forecast_bias=0.0,
        use_confidence_gates=False,
        use_dynamic_edge=False,
        boundary_z_boost=False,
        use_kelly=True,
        bankroll=50.0,
        maker_fill_rate=fill_rate,
        slippage_cents=slippage,
        max_risk_per_trade=5.0,
        min_contracts=5,
    )
    new_var_cfg = BacktestConfig(
        name="D: V2-variable",
        sigma=0.0,
        min_edge=0.08,
        forecast_bias=0.0,
        use_confidence_gates=False,
        use_dynamic_edge=False,
        boundary_z_boost=False,
        use_kelly=True,
        bankroll=50.0,
        maker_fill_rate=fill_rate,
        slippage_cents=slippage,
        max_risk_per_trade=5.0,
        min_contracts=5,
        variable_sigma=True,
        sigma_base=1.2,
    )
    return old_cfg, new_fixed_cfg, new_var_cfg


def main():
    parser = argparse.ArgumentParser(description="Robustness sweep for old vs new model")
    parser.add_argument("--sigma-ens", type=float, default=1.3, help="Average ensemble sigma estimate")
    args = parser.parse_args()

    print("=" * 88)
    print("ROBUSTNESS SWEEP -- OLD vs NEW MODEL")
    print("=" * 88)

    trades_df = load_trades()
    forecasts_df = load_forecasts()
    obs_df = load_observations()
    if trades_df is None or forecasts_df is None or obs_df is None:
        print("Missing required datasets.")
        return

    trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"]).dt.date
    forecast_lookup, obs_lookup = _build_lookups(forecasts_df, obs_df)

    sigma_v2 = max(1.2, config.ENSEMBLE_ALPHA * args.sigma_ens)
    print("V2 sigma used: %.2f" % sigma_v2)
    print("")

    scenarios = [
        {"name": "baseline", "fill_rate": 0.70, "slippage": 0},
        {"name": "mild_conservative", "fill_rate": 0.60, "slippage": 1},
        {"name": "conservative", "fill_rate": 0.50, "slippage": 2},
        {"name": "stress", "fill_rate": 0.40, "slippage": 3},
        {"name": "optimistic", "fill_rate": 0.85, "slippage": 0},
    ]

    rows = []
    for sc in scenarios:
        old_cfg, fixed_cfg, var_cfg = _scenario_configs(sc["fill_rate"], sc["slippage"], sigma_v2)
        old_trades, _ = run_backtest(old_cfg, trades_df, forecast_lookup, obs_lookup)
        fixed_trades, _ = run_backtest(fixed_cfg, trades_df, forecast_lookup, obs_lookup)
        var_trades, _ = run_backtest(var_cfg, trades_df, forecast_lookup, obs_lookup)
        old_stats = compute_stats(old_trades)
        fixed_stats = compute_stats(fixed_trades)
        var_stats = compute_stats(var_trades)

        row = {
            "scenario": sc["name"],
            "fill_rate": sc["fill_rate"],
            "extra_slippage_cents": sc["slippage"],
            "old_trades": old_stats["n"],
            "fixed_trades": fixed_stats["n"],
            "var_trades": var_stats["n"],
            "old_win_rate": old_stats["win_rate"],
            "fixed_win_rate": fixed_stats["win_rate"],
            "var_win_rate": var_stats["win_rate"],
            "old_pnl": old_stats["total_pnl"],
            "fixed_pnl": fixed_stats["total_pnl"],
            "var_pnl": var_stats["total_pnl"],
            "fixed_delta": fixed_stats["total_pnl"] - old_stats["total_pnl"],
            "var_delta": var_stats["total_pnl"] - old_stats["total_pnl"],
            "old_drawdown": old_stats["max_drawdown"],
            "fixed_drawdown": fixed_stats["max_drawdown"],
            "var_drawdown": var_stats["max_drawdown"],
            "var_beats_old": var_stats["total_pnl"] > old_stats["total_pnl"],
        }
        rows.append(row)

    print("  %-18s %6s %5s %10s %10s %10s %8s %8s %8s" % (
        "Scenario", "Fill", "Slip", "Old PnL", "Fixed PnL", "Var PnL",
        "Δ-fixed", "Δ-var", "Winner"
    ))
    print("  " + "-" * 95)
    for row in rows:
        winner = "VAR" if row["var_beats_old"] else "OLD"
        print("  %-18s %6.2f %5d %10.2f %10.2f %10.2f %+8.2f %+8.2f %8s" % (
            row["scenario"],
            row["fill_rate"],
            row["extra_slippage_cents"],
            row["old_pnl"],
            row["fixed_pnl"],
            row["var_pnl"],
            row["fixed_delta"],
            row["var_delta"],
            winner,
        ))

    wins = sum(1 for row in rows if row["var_beats_old"])
    print("")
    print("V2-variable beats Old in %d/%d scenarios." % (wins, len(rows)))

    out = {
        "sigma_v2": sigma_v2,
        "scenarios": rows,
        "new_wins_scenarios": wins,
        "total_scenarios": len(rows),
    }
    out_path = config.PROJECT_ROOT / "reports" / "robustness_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved: %s" % out_path)


if __name__ == "__main__":
    main()

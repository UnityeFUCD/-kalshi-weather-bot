"""
paper_tracker.py -- Structured paper trading logger with fill tracking and settlement.

Logs every paper trade signal to JSONL, simulates fills by checking the orderbook,
and reconciles PnL after market settlement.

Records are append-only in reports/paper_trades.jsonl with record_type:
  "signal"     -- paper trade generated
  "fill"       -- simulated fill or expiration
  "settlement" -- PnL after market settles
"""

import sys
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from model import parse_bucket_title, compute_fee

logger = logging.getLogger("paper_tracker")


class PaperTracker:
    """Paper trade logger with fill simulation and settlement reconciliation."""

    def __init__(self, kalshi_client=None):
        self.trades_path = config.PAPER_TRADES_PATH
        self.pending_path = config.PAPER_PENDING_PATH
        self.daily_dir = config.PAPER_DAILY_DIR
        self.kalshi = kalshi_client
        self.max_cycles = config.PAPER_FILL_TIMEOUT_CYCLES

        # Ensure directories exist
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)

    # --- JSONL I/O -----------------------------------------------------------

    def _append_jsonl(self, record):
        """Append one JSON record to the trades JSONL file."""
        with open(self.trades_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _read_all_records(self):
        """Read all JSONL records. Returns list of dicts."""
        if not self.trades_path.exists():
            return []
        records = []
        with open(self.trades_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def _load_pending(self):
        """Load pending (unfilled) paper trades."""
        if not self.pending_path.exists():
            return []
        with open(self.pending_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_pending(self, pending):
        """Save pending trades atomically."""
        tmp = self.pending_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pending, f, indent=2, default=str)
        tmp.replace(self.pending_path)

    # --- Signal Logging ------------------------------------------------------

    def log_paper_trade(self, signal, contracts, fee, risk,
                        forecast_temp, target_date, mu, sigma, market=None,
                        market_config=None):
        """
        Log a paper trade signal to JSONL and add to pending fills.

        Args:
            signal: Signal object from model.py
            contracts: int, number of contracts
            fee: float, estimated fee in dollars
            risk: float, dollars at risk
            forecast_temp: int, NWS forecast high (F)
            target_date: date, market target date
            mu: float, model mean
            sigma: float, model std dev
            market: Market object (optional, for bid/ask data)
            market_config: MarketConfig (optional, for series_ticker / ghcn)
        """
        trade_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)

        # Extract bid/ask from Market object
        market_bid = None
        market_ask = None
        midpoint = None
        if market is not None:
            if signal.side == "buy_yes":
                market_bid = market.yes_bid
                market_ask = market.yes_ask
            else:
                market_bid = market.no_bid
                market_ask = market.no_ask
            if market_bid is not None and market_ask is not None:
                midpoint = (market_bid + market_ask) / 2.0

        # Compute edge vs actual bid/ask (not midpoint)
        edge_vs_bid = None
        edge_vs_ask = None
        if signal.side == "buy_yes":
            if market_bid is not None:
                edge_vs_bid = signal.model_prob - (market_bid / 100.0)
            if market_ask is not None:
                edge_vs_ask = signal.model_prob - (market_ask / 100.0)
        else:  # buy_no
            no_prob = 1.0 - signal.model_prob
            if market_bid is not None:
                edge_vs_bid = no_prob - (market_bid / 100.0)
            if market_ask is not None:
                edge_vs_ask = no_prob - (market_ask / 100.0)

        record = {
            "record_type": "signal",
            "trade_id": trade_id,
            "timestamp": now.isoformat(),
            "series_ticker": market_config.series_ticker if market_config else config.SERIES_TICKER,
            "ghcn_station_id": market_config.ghcn_station_id if market_config else None,
            "ticker": signal.bucket.ticker,
            "side": signal.side,
            "model_prob": round(signal.model_prob, 4),
            "market_price": round(signal.market_price, 4),
            "market_bid": market_bid,
            "market_ask": market_ask,
            "midpoint_price": round(midpoint, 1) if midpoint else None,
            "edge": round(signal.edge, 4),
            "edge_vs_bid": round(edge_vs_bid, 4) if edge_vs_bid is not None else None,
            "edge_vs_ask": round(edge_vs_ask, 4) if edge_vs_ask is not None else None,
            "ev_per_contract": round(signal.ev_per_contract, 4),
            "suggested_price": signal.suggested_price,
            "contracts": contracts,
            "estimated_fee": round(fee, 4),
            "risk_dollars": round(risk, 4),
            "forecast_high": forecast_temp,
            "target_date": str(target_date),
            "mu": round(mu, 2),
            "sigma": round(sigma, 2),
        }

        self._append_jsonl(record)

        # Add to pending fills
        pending = self._load_pending()
        pending.append({
            "trade_id": trade_id,
            "ticker": signal.bucket.ticker,
            "side": signal.side,
            "suggested_price": signal.suggested_price,
            "contracts": contracts,
            "estimated_fee": round(fee, 4),
            "created_at": now.isoformat(),
            "cycles_checked": 0,
            "max_cycles": self.max_cycles,
            "target_date": str(target_date),
        })
        self._save_pending(pending)

        logger.info("  PAPER logged: %s %s @ %dc (trade_id=%s)",
                     signal.side, signal.bucket.ticker,
                     signal.suggested_price, trade_id)
        return trade_id

    # --- Fill Checking -------------------------------------------------------

    def check_fills(self):
        """
        Check pending paper trades against current orderbook.
        Returns (fills_count, expirations_count).
        """
        pending = self._load_pending()
        if not pending:
            return 0, 0

        if self.kalshi is None:
            logger.warning("No Kalshi client -- cannot check fills")
            return 0, 0

        fills = 0
        expirations = 0
        remaining = []

        for trade in pending:
            trade["cycles_checked"] += 1

            try:
                ob = self.kalshi.get_orderbook(trade["ticker"])
            except Exception as e:
                logger.warning("  Orderbook fetch failed for %s: %s",
                               trade["ticker"], e)
                remaining.append(trade)
                continue

            filled, fill_price = self._check_single_fill(trade, ob)

            if filled:
                fills += 1
                self._append_jsonl({
                    "record_type": "fill",
                    "trade_id": trade["trade_id"],
                    "fill_timestamp": datetime.now(timezone.utc).isoformat(),
                    "fill_price": fill_price,
                    "fill_reason": "market_moved_through",
                    "cycles_waited": trade["cycles_checked"],
                    "orderbook_snapshot": {
                        "best_yes_bid": ob.best_yes_bid,
                        "best_yes_ask": ob.best_yes_ask,
                        "best_no_bid": ob.best_no_bid,
                        "best_no_ask": ob.best_no_ask,
                        "spread_cents": ob.spread_cents,
                    },
                })
                logger.info("  FILL: %s %s @ %dc (waited %d cycles)",
                            trade["side"], trade["ticker"],
                            fill_price, trade["cycles_checked"])

            elif trade["cycles_checked"] >= trade["max_cycles"]:
                expirations += 1
                self._append_jsonl({
                    "record_type": "fill",
                    "trade_id": trade["trade_id"],
                    "fill_timestamp": datetime.now(timezone.utc).isoformat(),
                    "fill_price": None,
                    "fill_reason": "expired",
                    "cycles_waited": trade["cycles_checked"],
                    "orderbook_snapshot": {
                        "best_yes_bid": ob.best_yes_bid,
                        "best_yes_ask": ob.best_yes_ask,
                        "best_no_bid": ob.best_no_bid,
                        "best_no_ask": ob.best_no_ask,
                        "spread_cents": ob.spread_cents,
                    },
                })
                logger.info("  EXPIRED: %s %s (no fill after %d cycles)",
                            trade["side"], trade["ticker"],
                            trade["cycles_checked"])
            else:
                remaining.append(trade)

        self._save_pending(remaining)

        if fills or expirations:
            logger.info("Fill check: %d filled, %d expired, %d still pending",
                        fills, expirations, len(remaining))

        return fills, expirations

    def _check_single_fill(self, pending, orderbook):
        """Check if a pending paper order would have filled."""
        if pending["side"] == "buy_yes":
            best_ask = orderbook.best_yes_ask
            if best_ask is not None and best_ask <= pending["suggested_price"]:
                return True, best_ask
        elif pending["side"] == "buy_no":
            best_ask = orderbook.best_no_ask
            if best_ask is not None and best_ask <= pending["suggested_price"]:
                return True, best_ask
        return False, None

    # --- Settlement Reconciliation -------------------------------------------

    def reconcile_settlements(self):
        """
        Reconcile filled paper trades against actual settlement.
        Returns list of settlement results.
        """
        records = self._read_all_records()

        # Find all filled trades (not expired)
        filled_trades = {}
        signal_records = {}
        settled_ids = set()

        for r in records:
            if r["record_type"] == "signal":
                signal_records[r["trade_id"]] = r
            elif r["record_type"] == "fill" and r.get("fill_reason") == "market_moved_through":
                filled_trades[r["trade_id"]] = r
            elif r["record_type"] == "settlement":
                settled_ids.add(r["trade_id"])

        # Filter to unsettled fills
        unsettled = {
            tid: fill for tid, fill in filled_trades.items()
            if tid not in settled_ids and tid in signal_records
        }

        if not unsettled:
            logger.info("No unsettled filled trades to reconcile")
            return []

        logger.info("Reconciling %d filled trades...", len(unsettled))

        results = []
        cumulative_pnl = self._get_cumulative_pnl(records)

        for trade_id, fill in unsettled.items():
            signal = signal_records[trade_id]
            target_date = signal["target_date"]

            # Look up per-market config for this signal (falls back to defaults
            # for old records that don't have series_ticker)
            sig_ticker = signal.get("series_ticker")
            sig_ghcn_path = None
            if sig_ticker:
                from market_registry import get_market
                mc = get_market(sig_ticker)
                if mc:
                    sig_ghcn_path = mc.ghcn_parquet_path

            actual_tmax = self._get_actual_tmax(
                target_date,
                series_ticker=sig_ticker,
                ghcn_parquet_path=sig_ghcn_path,
            )

            if actual_tmax is None:
                logger.info("  No settlement data yet for %s (trade %s)",
                            target_date, trade_id)
                continue

            pnl_result = self._compute_trade_pnl(signal, fill, actual_tmax)
            if pnl_result is None:
                continue

            cumulative_pnl += pnl_result["pnl_dollars"]

            settlement = {
                "record_type": "settlement",
                "trade_id": trade_id,
                "settlement_timestamp": datetime.now(timezone.utc).isoformat(),
                "actual_tmax": actual_tmax,
                "tmax_source": pnl_result["tmax_source"],
                "bucket_hit": pnl_result["bucket_hit"],
                "payout_dollars": pnl_result["payout_dollars"],
                "cost_dollars": pnl_result["cost_dollars"],
                "pnl_dollars": pnl_result["pnl_dollars"],
                "cumulative_pnl": round(cumulative_pnl, 4),
            }

            self._append_jsonl(settlement)
            results.append(settlement)

            won = "WON" if pnl_result["pnl_dollars"] > 0 else "LOST"
            logger.info("  SETTLED: %s %s actual=%dF %s PnL=$%+.2f (cum=$%+.2f)",
                        signal["side"], signal["ticker"],
                        actual_tmax, won,
                        pnl_result["pnl_dollars"], cumulative_pnl)

        return results

    def _compute_trade_pnl(self, signal_record, fill_record, actual_tmax):
        """Compute PnL for a filled paper trade given actual temperature."""
        bucket = parse_bucket_title(signal_record["ticker"], "")
        if bucket is None:
            logger.warning("  Could not parse bucket for %s", signal_record["ticker"])
            return None

        bucket_hit = bucket.settles_yes(actual_tmax)
        contracts = signal_record["contracts"]
        entry_cents = fill_record.get("fill_price") or signal_record["suggested_price"]
        cost_per_contract = entry_cents / 100.0
        fee = signal_record["estimated_fee"]

        if signal_record["side"] == "buy_yes":
            payout = 1.00 * contracts if bucket_hit else 0.0
        else:
            payout = 1.00 * contracts if not bucket_hit else 0.0

        cost = cost_per_contract * contracts + fee
        pnl = payout - cost

        return {
            "bucket_hit": bucket_hit,
            "payout_dollars": round(payout, 4),
            "cost_dollars": round(cost, 4),
            "pnl_dollars": round(pnl, 4),
            "tmax_source": "ghcn",
        }

    def _get_actual_tmax(self, target_date_str, series_ticker=None,
                         ghcn_parquet_path=None):
        """Get actual TMAX for a target date. Tries Kalshi settled markets, then GHCN.

        Args:
            target_date_str: date string or date object
            series_ticker: override for config.SERIES_TICKER (from signal record)
            ghcn_parquet_path: override for config.GHCN_PARQUET_PATH (from MarketConfig)
        """
        from datetime import date as date_type

        if isinstance(target_date_str, str):
            try:
                target_date = date_type.fromisoformat(target_date_str)
            except ValueError:
                return None
        else:
            target_date = target_date_str

        ticker_to_use = series_ticker or config.SERIES_TICKER

        # Try Kalshi settled markets first
        if self.kalshi is not None:
            try:
                markets = self.kalshi.get_markets(
                    series_ticker=ticker_to_use,
                    status="settled"
                )
                # Find markets for this target date
                date_str = target_date.strftime("%y") + \
                           target_date.strftime("%b").upper() + \
                           target_date.strftime("%d")
                for m in markets:
                    if date_str in m.ticker and m.result is not None:
                        # We found a settled market but can't directly get TMAX
                        # from the result field alone. Fall through to GHCN.
                        break
            except Exception as e:
                logger.debug("Kalshi settled market lookup failed: %s", e)

        # Fall back to GHCN
        ghcn_path = ghcn_parquet_path or config.GHCN_PARQUET_PATH
        # Also check parent dir for nested repo structure
        if not ghcn_path.exists():
            alt = config.PROJECT_ROOT.parent / "data" / "raw" / "weather" / \
                  "observations" / "USW00094728_daily.parquet"
            if alt.exists():
                ghcn_path = alt

        if ghcn_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(ghcn_path)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                row = df[df["date"] == target_date]
                if not row.empty:
                    tmax = row.iloc[0]["tmax_f"]
                    if pd.notna(tmax):
                        return int(tmax)
            except Exception as e:
                logger.warning("GHCN lookup failed: %s", e)

        return None

    def _get_cumulative_pnl(self, records):
        """Sum up PnL from all existing settlement records."""
        total = 0.0
        for r in records:
            if r["record_type"] == "settlement":
                total += r.get("pnl_dollars", 0.0)
        return total

    # --- Daily Summary -------------------------------------------------------

    def generate_daily_summary(self, date_str=None):
        """Generate a daily paper trading summary report."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        records = self._read_all_records()

        # Filter to this date
        signals = [r for r in records
                    if r["record_type"] == "signal"
                    and r.get("target_date") == date_str]
        fills = [r for r in records
                 if r["record_type"] == "fill"
                 and r.get("trade_id") in {s["trade_id"] for s in signals}]
        settlements = [r for r in records
                       if r["record_type"] == "settlement"
                       and r.get("trade_id") in {s["trade_id"] for s in signals}]

        actual_fills = [f for f in fills if f.get("fill_reason") == "market_moved_through"]
        expired = [f for f in fills if f.get("fill_reason") == "expired"]

        # Compute PnL
        day_pnl = sum(s.get("pnl_dollars", 0) for s in settlements)
        all_settlements = [r for r in records if r["record_type"] == "settlement"]
        cum_pnl = sum(s.get("pnl_dollars", 0) for s in all_settlements)

        # Compute fill rate
        total_signals = len(signals)
        fill_count = len(actual_fills)
        fill_rate = (fill_count / total_signals * 100) if total_signals > 0 else 0

        # Avg edge at fill
        avg_edge = 0
        if actual_fills:
            fill_ids = {f["trade_id"] for f in actual_fills}
            filled_signals = [s for s in signals if s["trade_id"] in fill_ids]
            if filled_signals:
                avg_edge = sum(s.get("edge", 0) for s in filled_signals) / len(filled_signals)

        lines = []
        lines.append("=" * 60)
        lines.append("PAPER TRADING SUMMARY -- %s" % date_str)
        lines.append("=" * 60)

        lines.append("")
        lines.append("SIGNALS")
        lines.append("  Total signals found:  %d" % total_signals)
        lines.append("  Fills (simulated):    %d" % fill_count)
        lines.append("  Expired (no fill):    %d" % len(expired))
        lines.append("  Fill rate:            %.0f%%" % fill_rate)
        if avg_edge:
            lines.append("  Avg edge at fill:     %.1f%%" % (avg_edge * 100))

        if actual_fills:
            lines.append("")
            lines.append("FILLED TRADES")
            for f in actual_fills:
                sig = next((s for s in signals if s["trade_id"] == f["trade_id"]), None)
                if sig:
                    lines.append("  %s %s @ %dc x%d  (edge=%.1f%%, waited %d cycles)" % (
                        sig["side"], sig["ticker"], sig["suggested_price"],
                        sig["contracts"], sig.get("edge", 0) * 100,
                        f.get("cycles_waited", 0)))

        if settlements:
            lines.append("")
            lines.append("SETTLEMENT")
            for s in settlements:
                sig = next((sg for sg in signals if sg["trade_id"] == s["trade_id"]), None)
                ticker = sig["ticker"] if sig else "?"
                side = sig["side"] if sig else "?"
                won = "WON" if s["pnl_dollars"] > 0 else "LOST"
                lines.append("  %s %s  actual=%dF  %s  PnL=$%+.2f" % (
                    side, ticker, s["actual_tmax"], won, s["pnl_dollars"]))

        lines.append("")
        lines.append("PNL")
        lines.append("  Today:       $%+.2f" % day_pnl)
        lines.append("  Cumulative:  $%+.2f" % cum_pnl)

        # Key diagnostic metrics
        lines.append("")
        lines.append("KEY METRICS TO WATCH")
        lines.append("  Fill rate target:  >50%% (backtest assumed 70%%)")
        lines.append("  Avg edge target:   >10%% at fill price")
        if fill_rate < 50 and total_signals > 0:
            lines.append("  WARNING: Fill rate below 50%% -- backtest likely overstated")
        if avg_edge and avg_edge < 0.10:
            lines.append("  WARNING: Avg edge below 10%% -- consider taker orders for high-confidence signals")

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)

        # Write daily file
        summary_path = self.daily_dir / ("paper_summary_%s.txt" % date_str.replace("-", ""))
        summary_path.write_text(report, encoding="utf-8")
        logger.info("Daily summary written to %s", summary_path)
        print(report)

        return report

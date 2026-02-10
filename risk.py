"""
risk.py - Risk management.

Enforces:
- Max risk per trade
- Max daily exposure
- Max open positions
- Daily/weekly loss halts

Adds:
- Same-event position-correlation guard to catch highly overlapping bets.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import config

logger = logging.getLogger(__name__)


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _event_from_ticker(ticker: str) -> str:
    """Extract event ticker from market ticker."""
    parts = ticker.split("-")
    if len(parts) >= 2:
        return "%s-%s" % (parts[0], parts[1])
    return ticker


@dataclass
class DailyStats:
    date: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    max_exposure: float = 0.0
    current_exposure: float = 0.0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    """
    Risk gatekeeper. Every trade must pass pre_trade_check() first.
    """

    def __init__(self, bankroll: float = config.BANKROLL):
        self.bankroll = bankroll
        self.today = DailyStats(date=self._today_str())
        self.weekly_pnl = 0.0
        self.open_positions: dict[str, float] = {}  # ticker -> dollars at risk
        self.open_positions_detail: list[dict[str, Any]] = []

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _check_new_day(self):
        """Reset daily stats if new day."""
        today = self._today_str()
        if today != self.today.date:
            logger.info("New day: %s (yesterday PnL: $%.2f)", today, self.today.pnl)
            self.today = DailyStats(date=today)
            self.open_positions.clear()
            self.open_positions_detail.clear()

    def pre_trade_check(self, risk_dollars: float) -> tuple[bool, str]:
        """
        Check if a trade is allowed.

        Args:
            risk_dollars: Maximum dollars at risk for this trade

        Returns:
            (allowed, reason)
        """
        self._check_new_day()

        if self.today.halted:
            return False, "HALTED: %s" % self.today.halt_reason

        if self.today.pnl <= -config.DAILY_LOSS_LIMIT:
            self.today.halted = True
            self.today.halt_reason = "Daily loss limit ($%s)" % config.DAILY_LOSS_LIMIT
            logger.warning("HALT: %s", self.today.halt_reason)
            return False, self.today.halt_reason

        if self.weekly_pnl <= -config.WEEKLY_LOSS_LIMIT:
            self.today.halted = True
            self.today.halt_reason = "Weekly loss limit ($%s)" % config.WEEKLY_LOSS_LIMIT
            logger.warning("HALT: %s", self.today.halt_reason)
            return False, self.today.halt_reason

        if risk_dollars > config.MAX_RISK_PER_TRADE:
            return False, "Risk $%.2f > max $%s" % (risk_dollars, config.MAX_RISK_PER_TRADE)

        if len(self.open_positions) >= config.MAX_OPEN_POSITIONS:
            return False, "Max positions (%s) reached" % config.MAX_OPEN_POSITIONS

        new_exposure = self.today.current_exposure + risk_dollars
        if new_exposure > config.MAX_DAILY_EXPOSURE:
            return False, "Exposure $%.2f > max $%s" % (new_exposure, config.MAX_DAILY_EXPOSURE)

        return True, "OK"

    def get_open_positions_for_event(self, event_ticker: str) -> list[dict[str, Any]]:
        """Return currently open detailed positions for one event."""
        self._check_new_day()
        return [p for p in self.open_positions_detail if p.get("event_ticker") == event_ticker]

    def check_position_correlation(
        self,
        new_trade: dict[str, Any],
        existing_positions: list[dict[str, Any]],
        model_mu: float,
        model_sigma: float,
    ) -> dict[str, Any]:
        """
        Check if a new trade is too correlated with existing same-event positions.

        Correlation thresholds:
        - > 0.5: warn
        - > 0.8: block
        """
        if not existing_positions:
            return {"action": "allow", "correlation": None, "reason": "no_existing_positions"}

        sigma = max(float(model_sigma), 0.05)
        temps = self._temperature_grid(existing_positions, new_trade, model_mu, sigma)
        if not temps:
            return {"action": "allow", "correlation": None, "reason": "no_temperature_grid"}

        weights = [self._temp_prob(t, model_mu, sigma) for t in temps]
        total_w = sum(weights)
        if total_w <= 0:
            return {"action": "allow", "correlation": None, "reason": "zero_probability_mass"}
        weights = [w / total_w for w in weights]

        existing_pnl = [self._portfolio_pnl_at_temp(existing_positions, t) for t in temps]
        new_pnl = [self._trade_pnl_at_temp(new_trade, t) for t in temps]

        corr = self._weighted_corr(existing_pnl, new_pnl, weights)
        if corr is None:
            return {"action": "allow", "correlation": None, "reason": "undefined_correlation"}

        if corr > 0.8:
            return {
                "action": "block",
                "correlation": corr,
                "reason": "High correlation %.3f > 0.80 with existing same-event positions" % corr,
            }
        if corr > 0.5:
            return {
                "action": "warn",
                "correlation": corr,
                "reason": "Moderate correlation %.3f > 0.50 with existing same-event positions" % corr,
            }
        return {
            "action": "allow",
            "correlation": corr,
            "reason": "Correlation %.3f within limits" % corr,
        }

    def _temperature_grid(
        self,
        existing_positions: list[dict[str, Any]],
        new_trade: dict[str, Any],
        mu: float,
        sigma: float,
    ) -> list[int]:
        """Build integer temperature grid for weighted scenario analysis."""
        bounds = []
        all_trades = list(existing_positions) + [new_trade]
        for trade in all_trades:
            bucket = trade.get("bucket")
            if bucket is None:
                continue
            low = getattr(bucket, "low", None)
            high = getattr(bucket, "high", None)
            if low is not None:
                bounds.append(float(low))
            if high is not None:
                bounds.append(float(high))

        if bounds:
            t_min = int(math.floor(min(bounds) - 6))
            t_max = int(math.ceil(max(bounds) + 6))
        else:
            t_min = int(math.floor(mu - 6 * sigma))
            t_max = int(math.ceil(mu + 6 * sigma))

        t_min = max(-50, t_min)
        t_max = min(150, t_max)
        if t_max < t_min:
            return []
        return list(range(t_min, t_max + 1))

    def _temp_prob(self, temp_int: int, mu: float, sigma: float) -> float:
        """Probability mass of integer temperature under N(mu, sigma)."""
        upper = _phi((temp_int + 0.5 - mu) / sigma)
        lower = _phi((temp_int - 0.5 - mu) / sigma)
        return max(0.0, upper - lower)

    def _trade_pnl_at_temp(self, trade: dict[str, Any], temp_int: int) -> float:
        """P/L for one trade at one integer temperature scenario."""
        bucket = trade.get("bucket")
        if bucket is None:
            return 0.0

        side = trade.get("side", "buy_yes")
        contracts = int(trade.get("contracts", 0) or 0)
        entry_cents = float(trade.get("entry_price_cents", 0) or 0.0)
        fee_dollars = float(trade.get("fee_dollars", 0.0) or 0.0)

        cost = (entry_cents / 100.0) * contracts + fee_dollars
        yes_hit = bool(bucket.settles_yes(temp_int))
        if side == "buy_yes":
            payout = 1.0 * contracts if yes_hit else 0.0
        else:
            payout = 1.0 * contracts if not yes_hit else 0.0
        return payout - cost

    def _portfolio_pnl_at_temp(self, trades: list[dict[str, Any]], temp_int: int) -> float:
        return sum(self._trade_pnl_at_temp(t, temp_int) for t in trades)

    def _weighted_corr(self, x: list[float], y: list[float], w: list[float]) -> float | None:
        """Weighted correlation coefficient."""
        if not x or len(x) != len(y) or len(x) != len(w):
            return None

        mean_x = sum(wx * xv for wx, xv in zip(w, x))
        mean_y = sum(wx * yv for wx, yv in zip(w, y))

        cov = sum(wx * (xv - mean_x) * (yv - mean_y) for wx, xv, yv in zip(w, x, y))
        var_x = sum(wx * (xv - mean_x) ** 2 for wx, xv in zip(w, x))
        var_y = sum(wx * (yv - mean_y) ** 2 for wx, yv in zip(w, y))

        if var_x <= 1e-12 or var_y <= 1e-12:
            return None
        return cov / math.sqrt(var_x * var_y)

    def record_trade_open(
        self,
        ticker: str,
        risk_dollars: float,
        position_detail: dict[str, Any] | None = None,
    ):
        """Record a new open position."""
        self._check_new_day()
        self.open_positions[ticker] = risk_dollars
        self.today.current_exposure += risk_dollars
        self.today.max_exposure = max(self.today.max_exposure, self.today.current_exposure)
        self.today.trades += 1

        detail = {
            "ticker": ticker,
            "event_ticker": _event_from_ticker(ticker),
            "risk_dollars": risk_dollars,
        }
        if position_detail:
            detail.update(position_detail)
            detail.setdefault("event_ticker", _event_from_ticker(ticker))
        self.open_positions_detail.append(detail)

        logger.info(
            "Position opened: %s risk=$%.2f (exposure=$%.2f)",
            ticker,
            risk_dollars,
            self.today.current_exposure,
        )

    def record_trade_close(self, ticker: str, pnl: float):
        """Record a closed position (settlement or exit)."""
        self._check_new_day()
        risk = self.open_positions.pop(ticker, 0.0)
        self.today.current_exposure -= risk
        self.today.pnl += pnl
        self.weekly_pnl += pnl

        # Drop first matching detail record.
        for idx, detail in enumerate(self.open_positions_detail):
            if detail.get("ticker") == ticker:
                self.open_positions_detail.pop(idx)
                break

        if pnl > 0:
            self.today.wins += 1
        elif pnl < 0:
            self.today.losses += 1

        logger.info("Position closed: %s PnL=$%+.2f (daily=$%+.2f)", ticker, pnl, self.today.pnl)

    def summary(self) -> str:
        """Human-readable status."""
        self._check_new_day()
        stats = self.today
        win_rate = (stats.wins / stats.trades * 100) if stats.trades > 0 else 0.0
        text = (
            "Day: %s | Trades: %d | W/L: %d/%d (%.0f%%) | PnL: $%+.2f | Exposure: $%.2f | Weekly: $%+.2f"
            % (
                stats.date,
                stats.trades,
                stats.wins,
                stats.losses,
                win_rate,
                stats.pnl,
                stats.current_exposure,
                self.weekly_pnl,
            )
        )
        if stats.halted:
            text += " | HALTED: %s" % stats.halt_reason
        return text

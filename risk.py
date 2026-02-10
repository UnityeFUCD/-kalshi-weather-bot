"""
risk.py — Risk Management.

Enforces:
- Max risk per trade ($5)
- Max daily exposure ($20)
- Max open positions (4)
- Daily loss limit ($10) → halt
- Weekly loss limit ($20) → halt
- Don't trade within 1 hour of settlement
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)


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
        self.open_positions: dict[str, float] = {}  # ticker → dollars at risk

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _check_new_day(self):
        """Reset daily stats if new day."""
        today = self._today_str()
        if today != self.today.date:
            logger.info(f"New day: {today} (yesterday PnL: ${self.today.pnl:.2f})")
            self.today = DailyStats(date=today)

    def pre_trade_check(self, risk_dollars: float) -> tuple[bool, str]:
        """
        Check if a trade is allowed.
        
        Args:
            risk_dollars: Maximum dollars at risk for this trade
        
        Returns:
            (allowed: bool, reason: str)
        """
        self._check_new_day()

        # Kill switch
        if self.today.halted:
            return False, f"HALTED: {self.today.halt_reason}"

        # Daily loss limit
        if self.today.pnl <= -config.DAILY_LOSS_LIMIT:
            self.today.halted = True
            self.today.halt_reason = f"Daily loss limit (${config.DAILY_LOSS_LIMIT})"
            logger.warning(f"HALT: {self.today.halt_reason}")
            return False, self.today.halt_reason

        # Weekly loss limit
        if self.weekly_pnl <= -config.WEEKLY_LOSS_LIMIT:
            self.today.halted = True
            self.today.halt_reason = f"Weekly loss limit (${config.WEEKLY_LOSS_LIMIT})"
            logger.warning(f"HALT: {self.today.halt_reason}")
            return False, self.today.halt_reason

        # Per-trade risk
        if risk_dollars > config.MAX_RISK_PER_TRADE:
            return False, f"Risk ${risk_dollars:.2f} > max ${config.MAX_RISK_PER_TRADE}"

        # Max positions
        if len(self.open_positions) >= config.MAX_OPEN_POSITIONS:
            return False, f"Max positions ({config.MAX_OPEN_POSITIONS}) reached"

        # Total exposure
        new_exposure = self.today.current_exposure + risk_dollars
        if new_exposure > config.MAX_DAILY_EXPOSURE:
            return False, f"Exposure ${new_exposure:.2f} > max ${config.MAX_DAILY_EXPOSURE}"

        return True, "OK"

    def record_trade_open(self, ticker: str, risk_dollars: float):
        """Record a new open position."""
        self._check_new_day()
        self.open_positions[ticker] = risk_dollars
        self.today.current_exposure += risk_dollars
        self.today.max_exposure = max(self.today.max_exposure, self.today.current_exposure)
        self.today.trades += 1
        logger.info(f"Position opened: {ticker} risk=${risk_dollars:.2f} "
                    f"(exposure=${self.today.current_exposure:.2f})")

    def record_trade_close(self, ticker: str, pnl: float):
        """Record a closed position (settlement or exit)."""
        self._check_new_day()
        risk = self.open_positions.pop(ticker, 0)
        self.today.current_exposure -= risk
        self.today.pnl += pnl
        self.weekly_pnl += pnl

        if pnl > 0:
            self.today.wins += 1
        elif pnl < 0:
            self.today.losses += 1

        logger.info(f"Position closed: {ticker} PnL=${pnl:+.2f} "
                    f"(daily=${self.today.pnl:+.2f})")

    def summary(self) -> str:
        """Human-readable status."""
        self._check_new_day()
        s = self.today
        wr = (s.wins / s.trades * 100) if s.trades > 0 else 0
        return (
            f"Day: {s.date} | Trades: {s.trades} | W/L: {s.wins}/{s.losses} "
            f"({wr:.0f}%) | PnL: ${s.pnl:+.2f} | Exposure: ${s.current_exposure:.2f} "
            f"| Weekly: ${self.weekly_pnl:+.2f}"
            + (f" | ⚠️ HALTED: {s.halt_reason}" if s.halted else "")
        )

"""
bot.py -- Kalshi Weather Trading Bot.

Scans NHIGH (NYC high temperature) markets, compares NWS forecast-implied
probabilities to Kalshi prices, and trades edges above threshold.

Usage:
    python bot.py scan          # Scan only, no trading (safe to run anytime)
    python bot.py paper         # Paper trading (logs what WOULD trade)
    python bot.py live          # Live trading with real money
    python bot.py live --once   # Single cycle, then exit
"""

import sys
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

from kalshi_auth import KalshiAuth
from kalshi_client import KalshiClient
from nws import NWSClient
from model import parse_bucket_title, compute_signals, compute_fee, Signal
from risk import RiskManager
import config

# --- Logging Setup -----------------------------------------------------------

def setup_logging(level=config.LOG_LEVEL):
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


# --- Bot ---------------------------------------------------------------------

class WeatherBot:
    """
    Main trading bot for Kalshi NHIGH (NYC High Temperature) markets.
    """

    def __init__(self, mode="scan"):
        self.mode = mode
        self.logger = logging.getLogger("bot")

        # Initialize NWS client (no auth needed)
        self.nws = NWSClient()

        # ALWAYS create auth -- production requires it even for public endpoints
        auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
        self.kalshi = KalshiClient(auth=auth)

        if mode == "live":
            self.logger.info("LIVE MODE -- real money at risk")
        else:
            self.logger.info("Mode: %s", mode)

        self.risk = RiskManager()
        self._last_forecast_temp = None
        self._last_forecast_time = None

    def run_cycle(self):
        """
        Execute one scan-and-trade cycle.

        1. Fetch NWS forecast for Central Park (correct date!)
        2. Fetch NHIGH markets from Kalshi
        3. Parse buckets and compute model probabilities
        4. Compare to market prices
        5. Generate signals
        6. Execute trades (if mode allows)
        """
        self.logger.info("-" * 50)
        self.logger.info("Cycle start -- %s", datetime.now(timezone.utc).isoformat())

        # -- Step 1: NWS Forecast (for the correct target date) --
        et_offset = timezone(timedelta(hours=-5))
        now_et = datetime.now(et_offset)

        if now_et.hour >= 8:
            # After 8 AM: trade tomorrow's market, need tomorrow's forecast
            target_date = (now_et + timedelta(days=1)).date()
            forecast_temp, period_name = self.nws.get_high_forecast_for_date(target_date)
            sigma = config.SIGMA_1DAY  # 1-day ahead = more uncertainty
        else:
            # Before 8 AM: trade today's market
            target_date = now_et.date()
            forecast_temp, period_name = self.nws.get_high_forecast_for_date(target_date)
            # Same-day sigma depends on time
            if now_et.hour < 8:
                sigma = config.SIGMA_1DAY
            elif now_et.hour < 14:
                sigma = config.SIGMA_SAMEDAY_AM
            else:
                sigma = config.SIGMA_SAMEDAY_PM

        if forecast_temp is None:
            # Fallback to first daytime period
            self.logger.warning("Could not get forecast for %s -- trying fallback", target_date)
            forecast_temp = self.nws.get_today_high_forecast()

        if forecast_temp is None:
            self.logger.error("Failed to get NWS forecast -- skipping cycle")
            return []

        self._last_forecast_temp = forecast_temp
        self._last_forecast_time = datetime.now(timezone.utc)

        mu = forecast_temp - config.FORECAST_BIAS

        self.logger.info("NWS Forecast for %s: %dF | Model: mu=%.1f, sigma=%.1f",
                        target_date, forecast_temp, mu, sigma)

        # -- Step 2: Fetch Markets --
        markets = self.kalshi.get_markets(
            series_ticker=config.SERIES_TICKER,
            status="open"
        )

        if not markets:
            self.logger.warning("No open NHIGH markets found")
            return []

        self.logger.info("Found %d total NHIGH markets", len(markets))

        # -- Step 2b: Filter to target date --
        today_str = now_et.strftime("%y") + now_et.strftime("%b").upper() + now_et.strftime("%d")
        tomorrow = now_et + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%y") + tomorrow.strftime("%b").upper() + tomorrow.strftime("%d")

        self.logger.info("Today=%s, Tomorrow=%s, Hour=%d ET", today_str, tomorrow_str, now_et.hour)

        if now_et.hour >= 8:
            target_str = tomorrow_str
            self.logger.info("After 8AM ET -- targeting tomorrow: %s", target_str)
        else:
            target_str = today_str
            self.logger.info("Before 8AM ET -- targeting today: %s", target_str)

        tradeable = [m for m in markets if target_str in m.ticker]

        if not tradeable:
            self.logger.warning("No markets matching %s -- trying all %d markets", target_str, len(markets))
            for m in markets[:3]:
                self.logger.info("  Available ticker: %s", m.ticker)
            tradeable = markets
        else:
            self.logger.info("Filtered to %d markets for %s (from %d total)",
                           len(tradeable), target_str, len(markets))

        markets = tradeable

        # -- Step 3: Parse Buckets --
        buckets = []
        market_prices = {}

        for m in markets:
            bucket = parse_bucket_title(m.ticker, m.title)
            if bucket is None:
                self.logger.warning("Skipping unparseable: %s '%s'", m.ticker, m.title)
                continue

            buckets.append(bucket)

            # Use midpoint as market price, or yes_ask if no bid
            if m.yes_bid is not None and m.yes_ask is not None:
                price = (m.yes_bid + m.yes_ask) / 200.0
            elif m.yes_ask is not None:
                price = m.yes_ask / 100.0
            elif m.yes_bid is not None:
                price = m.yes_bid / 100.0
            elif m.last_price is not None:
                price = m.last_price / 100.0
            else:
                self.logger.warning("No price for %s -- skipping", m.ticker)
                continue

            market_prices[m.ticker] = price

            bid_str = "%3d" % m.yes_bid if m.yes_bid is not None else " --"
            ask_str = "%3d" % m.yes_ask if m.yes_ask is not None else " --"
            p_model = bucket.probability(mu, sigma)
            self.logger.info("  %-40s bid=%sc ask=%sc vol=%5d P=%.3f",
                           m.ticker, bid_str, ask_str, m.volume, p_model)

        # -- Step 4: Generate Signals --
        # Filter out near-settled markets (bid >= 95c or ask <= 5c)
        filtered_prices = {}
        for ticker, price in market_prices.items():
            if 0.05 <= price <= 0.95:
                filtered_prices[ticker] = price
            else:
                self.logger.info("  Skipping %s: price=%.2f (near-settled)", ticker, price)

        signals = compute_signals(buckets, filtered_prices, mu, sigma)

        if not signals:
            self.logger.info("No signals -- no edge above threshold")
            return []

        self.logger.info(">>> %d signal(s) found:", len(signals))
        for s in signals:
            self.logger.info("  %s %s: edge=%+.3f (model=%.3f vs mkt=%.2f)",
                           s.side, s.bucket.ticker, s.edge, s.model_prob, s.market_price)

        # -- Step 5: Execute --
        if self.mode == "scan":
            self.logger.info("SCAN mode -- no trades placed")
        elif self.mode == "paper":
            self._paper_trade(signals)
        elif self.mode == "live":
            self._live_trade(signals, markets)

        # Log risk summary
        self.logger.info("Risk: %s", self.risk.summary())

        return signals

    def _paper_trade(self, signals):
        """Simulate trading -- log what we would do."""
        for s in signals:
            count = self._compute_position_size(s)
            if count == 0:
                continue

            risk = (s.suggested_price / 100.0) * count
            fee = compute_fee(s.suggested_price, count, is_maker=True)

            allowed, reason = self.risk.pre_trade_check(risk)
            if not allowed:
                self.logger.info("  PAPER BLOCKED: %s", reason)
                continue

            self.logger.info("  PAPER: %s %dx %s @ %dc (risk=$%.2f, fee=$%.2f)",
                           s.side, count, s.bucket.ticker, s.suggested_price, risk, fee)
            self.risk.record_trade_open(s.bucket.ticker, risk)

    def _live_trade(self, signals, markets):
        """Execute real trades on Kalshi."""
        for s in signals:
            count = self._compute_position_size(s)
            if count < config.MIN_CONTRACTS:
                self.logger.info("  Skip %s: count=%d < min=%d",
                               s.bucket.ticker, count, config.MIN_CONTRACTS)
                continue

            risk = (s.suggested_price / 100.0) * count

            # Risk check
            allowed, reason = self.risk.pre_trade_check(risk)
            if not allowed:
                self.logger.info("  BLOCKED: %s", reason)
                continue

            # Verify orderbook hasn't changed (post-only check)
            ob = self.kalshi.get_orderbook(s.bucket.ticker)

            if s.side == "buy_yes":
                best_ask = ob.best_yes_ask
                if best_ask is not None and s.suggested_price >= best_ask:
                    adjusted = best_ask - 1
                    if adjusted < 1:
                        self.logger.info("  Skip %s: can't post-only", s.bucket.ticker)
                        continue
                    self.logger.info("  Post-only adjust: %dc -> %dc", s.suggested_price, adjusted)
                    s.suggested_price = adjusted

                try:
                    result = self.kalshi.place_order(
                        ticker=s.bucket.ticker,
                        side="yes",
                        action="buy",
                        count=count,
                        price_cents=s.suggested_price,
                    )
                    self.risk.record_trade_open(s.bucket.ticker, risk)
                    self.logger.info("  OK ORDER PLACED: %s", result)
                except Exception as e:
                    self.logger.error("  FAIL ORDER: %s", e)

            elif s.side == "buy_no":
                best_ask = ob.best_no_ask
                if best_ask is not None and s.suggested_price >= best_ask:
                    adjusted = best_ask - 1
                    if adjusted < 1:
                        self.logger.info("  Skip %s: can't post-only", s.bucket.ticker)
                        continue
                    s.suggested_price = adjusted

                try:
                    result = self.kalshi.place_order(
                        ticker=s.bucket.ticker,
                        side="no",
                        action="buy",
                        count=count,
                        price_cents=s.suggested_price,
                    )
                    self.risk.record_trade_open(s.bucket.ticker, risk)
                    self.logger.info("  OK ORDER PLACED: %s", result)
                except Exception as e:
                    self.logger.error("  FAIL ORDER: %s", e)

    def _compute_position_size(self, signal):
        """
        How many contracts to buy, respecting risk limits.

        Risk per contract = price in dollars (if buying YES)
        Max risk per trade = $5.00
        Min contracts = 5 (below this, fees dominate)
        """
        price_dollars = signal.suggested_price / 100.0
        if price_dollars <= 0:
            return 0

        max_contracts = int(config.MAX_RISK_PER_TRADE / price_dollars)
        count = max(config.MIN_CONTRACTS, min(max_contracts, 50))

        while count * price_dollars > config.MAX_RISK_PER_TRADE and count > 0:
            count -= 1

        return count

    def run_loop(self, once=False):
        """Main bot loop. Runs until interrupted."""
        self.logger.info("=" * 60)
        self.logger.info("Kalshi Weather Bot -- %s mode", self.mode.upper())
        self.logger.info("Target: %s (NYC High Temperature)", config.SERIES_TICKER)
        self.logger.info("Bankroll: $%.0f", config.BANKROLL)
        self.logger.info("Min edge: %d%%", config.MIN_EDGE * 100)
        self.logger.info("=" * 60)

        if self.mode == "live":
            try:
                balance = self.kalshi.get_balance()
                self.logger.info("Account balance: $%.2f", balance)
            except Exception as e:
                self.logger.error("Failed to get balance: %s", e)

        cycle = 0
        while True:
            cycle += 1
            try:
                self.logger.info("\n%s CYCLE %d %s", "=" * 20, cycle, "=" * 20)
                self.run_cycle()
            except KeyboardInterrupt:
                self.logger.info("\nInterrupted by user")
                break
            except Exception as e:
                self.logger.error("Cycle error: %s", e, exc_info=True)

            if once:
                break

            self.logger.info("Sleeping %ds...", config.SCAN_INTERVAL_SECONDS)
            try:
                time.sleep(config.SCAN_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                self.logger.info("\nInterrupted by user")
                break

        self.logger.info("\nFinal status: %s", self.risk.summary())


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalshi Weather Trading Bot")
    parser.add_argument(
        "mode",
        choices=["scan", "paper", "live"],
        help="scan=read-only, paper=simulated trades, live=real money"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single cycle then exit"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else config.LOG_LEVEL)

    bot = WeatherBot(mode=args.mode)
    bot.run_loop(once=args.once)


if __name__ == "__main__":
    main()
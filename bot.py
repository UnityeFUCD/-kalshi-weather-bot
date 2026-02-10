"""
bot.py -- Kalshi Weather Trading Bot.

Scans weather high-temperature markets, compares NWS forecast-implied
probabilities to Kalshi prices, and trades edges above threshold.

Supports multiple markets via market_registry.py.

Phase 4/5 enhancements:
  - Hourly-derived μ (more accurate than period high)
  - Forecast revision tracking with delta features
  - METAR/ASOS real-time observations with μ nudges
  - Confidence scoring with dynamic MIN_EDGE
  - Ensemble σ from Open-Meteo (ECMWF + GFS)

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
from market_registry import get_enabled_markets
from delta_tracker import DeltaTracker
from metar_obs import MetarObserver
from confidence import (compute_confidence, compute_dynamic_min_edge,
                        compute_boundary_z, extract_bucket_boundaries,
                        passes_predawn_gates)
from ensemble import EnsembleForecaster
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
    Main trading bot for Kalshi weather high-temperature markets.
    """

    def __init__(self, mode="scan"):
        self.mode = mode
        self.logger = logging.getLogger("bot")

        # Multi-market: one NWS client per enabled market
        self._enabled_markets = get_enabled_markets()
        self._nws_clients = {}
        self._delta_trackers = {}
        self._metar_observers = {}
        self._ensemble_forecasters = {}

        for mc in self._enabled_markets:
            self._nws_clients[mc.series_ticker] = NWSClient(market_config=mc)
            self._delta_trackers[mc.series_ticker] = DeltaTracker(mc.series_ticker)

            # METAR observer
            station = config.METAR_STATIONS.get(mc.series_ticker, mc.nws_station)
            self._metar_observers[mc.series_ticker] = MetarObserver(
                station, nws_client=self._nws_clients[mc.series_ticker])

            # Ensemble forecaster (uses lat/lon from NWS config)
            lat_lon = self._get_lat_lon(mc)
            self._ensemble_forecasters[mc.series_ticker] = EnsembleForecaster(*lat_lon)

        # Backward compat: self.nws points to first market's client
        if self._enabled_markets:
            self.nws = self._nws_clients[self._enabled_markets[0].series_ticker]
        else:
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
        self._last_target_date = None
        self._last_mu = None
        self._last_sigma = None
        self._last_confidence = None
        self._last_dynamic_edge = None

        # Paper trading tracker
        if mode == "paper":
            from paper_tracker import PaperTracker
            self.paper_tracker = PaperTracker(kalshi_client=self.kalshi)
            self.logger.info("Paper tracker initialized -- logging to %s",
                             config.PAPER_TRADES_PATH)

        # Accumulate signal report sections across markets
        self._report_sections = []

    def _get_lat_lon(self, mc):
        """Get latitude/longitude for a market config."""
        # Map NWS grid offices to approximate lat/lon
        office_coords = {
            "OKX": (40.7831, -73.9712),   # NYC
            "LOT": (41.9742, -87.9073),   # Chicago O'Hare
        }
        return office_coords.get(mc.nws_grid_office, (40.7831, -73.9712))

    def run_cycle(self):
        """
        Execute one scan-and-trade cycle across all enabled markets.
        """
        self.logger.info("-" * 50)
        self.logger.info("Cycle start -- %s", datetime.now(timezone.utc).isoformat())

        # -- Paper mode: check pending fills from previous cycles --
        if self.mode == "paper" and hasattr(self, 'paper_tracker'):
            fills, expirations = self.paper_tracker.check_fills()
            if fills or expirations:
                self.logger.info("Fill check: %d filled, %d expired", fills, expirations)

        # Reset report sections for this cycle
        self._report_sections = []

        all_signals = []
        for mc in self._enabled_markets:
            nws = self._nws_clients[mc.series_ticker]
            signals = self._run_market_cycle(mc, nws)
            all_signals.extend(signals)

        # Write combined signals.txt report
        self._flush_signal_report()

        return all_signals

    def _run_market_cycle(self, mc, nws_client):
        """
        Run a single scan-and-trade cycle for one market.

        Enhanced with Phase 4/5:
          - Hourly μ from delta tracker
          - METAR observation nudges
          - Confidence scoring
          - Dynamic MIN_EDGE
          - Ensemble σ
        """
        self.logger.info("=== Market: %s (%s) ===", mc.series_ticker, mc.display_name)

        delta_tracker = self._delta_trackers[mc.series_ticker]
        metar_obs = self._metar_observers[mc.series_ticker]
        ensemble = self._ensemble_forecasters[mc.series_ticker]

        # -- Step 1: Determine target date and base sigma --
        et_offset = timezone(timedelta(hours=-5))
        now_et = datetime.now(et_offset)
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

        # -- Step 1b: Hourly-derived μ (Phase 4A) --
        # Try hourly forecast first (more accurate), fall back to period high
        snapshot, is_revision = delta_tracker.process_hourly_forecast(nws_client, target_date)

        hourly_mu = delta_tracker.get_hourly_mu(target_date)
        forecast_temp, period_name = nws_client.get_high_forecast_for_date(target_date)

        if forecast_temp is None:
            self.logger.warning("Could not get forecast for %s -- trying fallback", target_date)
            forecast_temp = nws_client.get_today_high_forecast()

        if forecast_temp is None and hourly_mu is None:
            self.logger.error("Failed to get any forecast for %s -- skipping", mc.series_ticker)
            return []

        # Use hourly max as primary μ, period high as fallback
        if hourly_mu is not None:
            mu = hourly_mu - mc.forecast_bias
            self.logger.info("Using hourly-derived mu=%.1f (hourly_max=%.0f, period_high=%s)",
                            mu, hourly_mu, forecast_temp)
        else:
            mu = forecast_temp - mc.forecast_bias
            self.logger.info("Using period-high mu=%.1f (forecast=%d)", mu, forecast_temp)

        if forecast_temp is None:
            forecast_temp = int(round(hourly_mu)) if hourly_mu else 0

        # Log revision info
        if is_revision:
            delta = delta_tracker.latest_delta(target_date)
            self.logger.info("FORECAST REVISION #%d: delta_tmax=%.1f",
                            delta.revision_number, delta.delta_tmax_hourly)

        self._last_forecast_temp = forecast_temp
        self._last_forecast_time = datetime.now(timezone.utc)
        self._last_target_date = target_date

        # -- Step 1c: METAR observation nudge (Phase 4B) --
        obs_temp, obs_source, obs_time = metar_obs.get_current_temp()
        residual_ewma = None

        if obs_temp is not None:
            self.logger.info("METAR obs: %.1fF from %s", obs_temp, obs_source)
            # Only apply nudge and compute residual for same-day trading
            if target_date == now_et.date():
                mu_before = mu
                mu = metar_obs.compute_mu_nudge(mu)
                if abs(mu - mu_before) > 0.01:
                    self.logger.info("Observation nudge: mu %.1f -> %.1f", mu_before, mu)
                _, residual_ewma = metar_obs.compute_residual(mu)
            else:
                # Next-day: obs don't inform forecast, skip residual for confidence
                self.logger.info("Next-day trade: skipping obs residual for confidence")

        self._last_mu = mu

        # -- Step 1d: Ensemble σ (Phase 5) --
        sigma = sigma_base
        if config.ENSEMBLE_ENABLED:
            try:
                sigma_ens, n_members, _ = ensemble.get_ensemble_sigma(target_date)
                rev_vol = delta_tracker.revision_volatility(target_date)

                sigma = ensemble.compose_sigma(
                    sigma_base=sigma_base,
                    sigma_ens=sigma_ens,
                    revision_volatility=rev_vol,
                )
                if sigma != sigma_base:
                    self.logger.info("Ensemble sigma: base=%.2f -> composed=%.2f "
                                    "(ens=%.2f, %d members, rev_vol=%.2f)",
                                    sigma_base, sigma, sigma_ens or 0, n_members, rev_vol)
            except Exception as e:
                self.logger.warning("Ensemble fetch failed, using base sigma: %s", e)
                sigma = sigma_base

        self._last_sigma = sigma

        self.logger.info("NWS Forecast for %s: %dF | Model: mu=%.1f, sigma=%.2f",
                        target_date, forecast_temp, mu, sigma)

        # -- Step 2: Fetch Markets --
        markets = self.kalshi.get_markets(
            series_ticker=mc.series_ticker,
            status="open"
        )

        if not markets:
            self.logger.warning("No open %s markets found", mc.series_ticker)
            return []

        self.logger.info("Found %d total %s markets", len(markets), mc.series_ticker)

        # -- Step 2b: Filter to target date --
        today_str = now_et.strftime("%y") + now_et.strftime("%b").upper() + now_et.strftime("%d")
        tomorrow = now_et + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%y") + tomorrow.strftime("%b").upper() + tomorrow.strftime("%d")

        self.logger.info("Today=%s, Tomorrow=%s, Hour=%d ET", today_str, tomorrow_str, hour_et)

        if hour_et >= 8:
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

        # -- Step 3b: Confidence scoring (Phase 4C) --
        boundaries = extract_bucket_boundaries(buckets)
        boundary_z = compute_boundary_z(mu, sigma, boundaries)

        forecast_age = delta_tracker.forecast_age_minutes(target_date)
        obs_age = metar_obs.get_obs_age_minutes()

        confidence, gate_scores = compute_confidence(
            forecast_age_min=forecast_age,
            obs_age_min=obs_age,
            residual_ewma=residual_ewma,
            boundary_z=boundary_z,
            hour_et=hour_et,
        )

        dynamic_min_edge = compute_dynamic_min_edge(hour_et, confidence)

        # Compose boundary_risk for ensemble sigma if near boundary
        if boundary_z is not None and boundary_z < 1.5 and config.ENSEMBLE_ENABLED:
            boundary_risk = max(0, 1.5 - boundary_z)
            sigma_with_boundary = ensemble.compose_sigma(
                sigma_base=sigma,
                boundary_risk=boundary_risk,
            )
            if sigma_with_boundary > sigma:
                self.logger.info("Boundary risk sigma boost: %.2f -> %.2f (z=%.2f)",
                                sigma, sigma_with_boundary, boundary_z)
                sigma = sigma_with_boundary
                self._last_sigma = sigma

        self._last_confidence = confidence
        self._last_dynamic_edge = dynamic_min_edge

        self.logger.info("Confidence: %.3f | Gates: %s | Dynamic MIN_EDGE: %.1f%% | Boundary z: %s",
                        confidence,
                        {k: "%.2f" % v for k, v in gate_scores.items()},
                        dynamic_min_edge * 100,
                        "%.2f" % boundary_z if boundary_z is not None else "N/A")

        # Pre-dawn gate check
        if hour_et < 6:
            ok, reason = passes_predawn_gates(confidence, dynamic_min_edge, boundary_z)
            if not ok:
                self.logger.info("Pre-dawn gate BLOCKED: %s", reason)
                self._write_signal_report(mc, target_date, forecast_temp, mu, sigma,
                                          buckets, market_prices, [],
                                          confidence=confidence, dynamic_edge=dynamic_min_edge,
                                          boundary_z=boundary_z)
                return []

        # Confidence floor check
        if confidence < config.MIN_CONFIDENCE_TO_TRADE:
            self.logger.info("Confidence %.3f < %.3f minimum -- no trading",
                            confidence, config.MIN_CONFIDENCE_TO_TRADE)
            self._write_signal_report(mc, target_date, forecast_temp, mu, sigma,
                                      buckets, market_prices, [],
                                      confidence=confidence, dynamic_edge=dynamic_min_edge,
                                      boundary_z=boundary_z)
            return []

        # -- Step 4: Generate Signals (with dynamic MIN_EDGE) --
        filtered_prices = {}
        for ticker, price in market_prices.items():
            if 0.05 <= price <= 0.95:
                filtered_prices[ticker] = price
            else:
                self.logger.info("  Skipping %s: price=%.2f (near-settled)", ticker, price)

        signals = compute_signals(buckets, filtered_prices, mu, sigma,
                                  min_edge=dynamic_min_edge)

        if not signals:
            self.logger.info("No signals -- no edge above %.1f%% dynamic threshold",
                            dynamic_min_edge * 100)
            self._write_signal_report(mc, target_date, forecast_temp, mu, sigma,
                                      buckets, market_prices, [],
                                      confidence=confidence, dynamic_edge=dynamic_min_edge,
                                      boundary_z=boundary_z)
            return []

        self.logger.info(">>> %d signal(s) found:", len(signals))
        for s in signals:
            self.logger.info("  %s %s: edge=%+.3f (model=%.3f vs mkt=%.2f)",
                           s.side, s.bucket.ticker, s.edge, s.model_prob, s.market_price)

        # -- Step 5: Execute --
        if self.mode == "scan":
            self.logger.info("SCAN mode -- no trades placed")
        elif self.mode == "paper":
            self._paper_trade(signals, markets, market_config=mc)
        elif self.mode == "live":
            self._live_trade(signals, markets)

        # Log risk summary
        self.logger.info("Risk: %s", self.risk.summary())

        # Accumulate signal report section for this market
        self._write_signal_report(mc, target_date, forecast_temp, mu, sigma,
                                  buckets, market_prices, signals,
                                  confidence=confidence, dynamic_edge=dynamic_min_edge,
                                  boundary_z=boundary_z)

        return signals

    def _write_signal_report(self, market_config, target_date, forecast_temp,
                              mu, sigma, buckets, market_prices, signals,
                              confidence=None, dynamic_edge=None, boundary_z=None):
        """Accumulate a signal report section for one market."""
        try:
            lines = []
            lines.append("")
            lines.append("-" * 60)
            lines.append("MARKET: %s (%s)" % (market_config.series_ticker, market_config.display_name))
            lines.append("Target Date: %s" % target_date)
            lines.append("-" * 60)

            lines.append("")
            lines.append("FORECAST")
            lines.append("  NWS Forecast High: %dF" % forecast_temp)
            lines.append("  Model Mean (mu):   %.1fF" % mu)
            lines.append("  Model Sigma:       %.2fF" % sigma)

            if confidence is not None:
                lines.append("")
                lines.append("CONFIDENCE")
                lines.append("  Score:            %.3f" % confidence)
                lines.append("  Dynamic MIN_EDGE: %.1f%%" % (dynamic_edge * 100 if dynamic_edge else 0))
                lines.append("  Boundary z:       %s" % (
                    "%.2f" % boundary_z if boundary_z is not None else "N/A"))

            lines.append("")
            lines.append("MARKET PRICES & MODEL PROBABILITIES")
            lines.append("  %-40s %8s %8s %8s" % ("Ticker", "Mkt Price", "Model P", "Edge"))
            lines.append("  " + "-" * 68)

            for b in buckets:
                price = market_prices.get(b.ticker)
                p_model = b.probability(mu, sigma)
                if price is not None:
                    edge = p_model - price
                    lines.append("  %-40s %7.0fc %7.1f%% %+7.1f%%" % (
                        b.ticker, price * 100, p_model * 100, edge * 100))
                else:
                    lines.append("  %-40s %8s %7.1f%%" % (b.ticker, "no price", p_model * 100))

            lines.append("")
            if signals:
                lines.append("SIGNALS FOUND: %d" % len(signals))
                lines.append("")
                for s in signals:
                    count = self._compute_position_size(s)
                    risk = (s.suggested_price / 100.0) * count
                    fee = compute_fee(s.suggested_price, count, is_maker=True)
                    lines.append("  %s %s" % (s.side.upper(), s.bucket.ticker))
                    lines.append("    Model: %.1f%%  |  Market: %.0fc  |  Edge: %+.1f%%" % (
                        s.model_prob * 100, s.market_price * 100, s.edge * 100))
                    lines.append("    Suggested: %dc x %d contracts  |  Risk: $%.2f  |  Fee: $%.2f" % (
                        s.suggested_price, count, risk, fee))
                    lines.append("")
            else:
                edge_pct = dynamic_edge * 100 if dynamic_edge else config.MIN_EDGE * 100
                lines.append("NO SIGNALS -- no edge above %.1f%% threshold" % edge_pct)

            self._report_sections.append("\n".join(lines))

        except Exception as e:
            self.logger.error("Failed to build signal report section for %s: %s",
                              market_config.series_ticker, e)

    def _flush_signal_report(self):
        """Write the combined signal report (all markets) to signals.txt."""
        try:
            mt = timezone(timedelta(hours=-7))
            now_mt = datetime.now(mt)

            header = []
            header.append("=" * 60)
            header.append("KALSHI WEATHER BOT -- SIGNAL REPORT")
            header.append("Generated: %s MT" % now_mt.strftime("%Y-%m-%d %I:%M %p"))
            tickers = ", ".join(mc.series_ticker for mc in self._enabled_markets)
            header.append("Markets: %s" % tickers)
            header.append("=" * 60)

            footer = []
            footer.append("")
            footer.append("Risk: %s" % self.risk.summary())
            footer.append("Mode: %s" % self.mode.upper())
            if self._last_confidence is not None:
                footer.append("Last confidence: %.3f | Dynamic edge: %.1f%%" % (
                    self._last_confidence,
                    self._last_dynamic_edge * 100 if self._last_dynamic_edge else 0))
            footer.append("")

            report = "\n".join(header) + "\n" + "\n".join(self._report_sections) + "\n" + "\n".join(footer)
            config.SIGNAL_REPORT_PATH.write_text(report, encoding="utf-8")
            self.logger.info("Signal report written to %s", config.SIGNAL_REPORT_PATH)

        except Exception as e:
            self.logger.error("Failed to write signal report: %s", e)

    def _paper_trade(self, signals, markets=None, market_config=None):
        """Simulate trading -- log what we would do, with JSONL tracking."""
        # Build ticker -> Market lookup for bid/ask data
        market_lookup = {}
        if markets:
            for m in markets:
                market_lookup[m.ticker] = m

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

            # Log to structured JSONL
            if hasattr(self, 'paper_tracker'):
                market = market_lookup.get(s.bucket.ticker)
                self.paper_tracker.log_paper_trade(
                    signal=s,
                    contracts=count,
                    fee=fee,
                    risk=risk,
                    forecast_temp=self._last_forecast_temp,
                    target_date=self._last_target_date,
                    mu=self._last_mu,
                    sigma=self._last_sigma,
                    market=market,
                    market_config=market_config,
                )

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

    def run_loop(self, once=False, until_hour_mt=None):
        """
        Main bot loop. Runs until interrupted or past until_hour_mt.

        Args:
            once: Run single cycle then exit
            until_hour_mt: Auto-exit after this Mountain Time hour (e.g. 16 = 4 PM MT)
        """
        self.logger.info("=" * 60)
        self.logger.info("Kalshi Weather Bot -- %s mode", self.mode.upper())
        tickers = ", ".join(mc.series_ticker for mc in self._enabled_markets)
        self.logger.info("Enabled markets: %s", tickers)
        self.logger.info("Bankroll: $%.0f", config.BANKROLL)
        self.logger.info("Base MIN_EDGE: %d%% (dynamic scaling active)", config.MIN_EDGE * 100)
        self.logger.info("Ensemble: %s", "ENABLED" if config.ENSEMBLE_ENABLED else "DISABLED")
        if until_hour_mt is not None:
            self.logger.info("Auto-exit at: %d:00 MT", until_hour_mt)
        self.logger.info("=" * 60)

        if self.mode == "live":
            try:
                balance = self.kalshi.get_balance()
                self.logger.info("Account balance: $%.2f", balance)
            except Exception as e:
                self.logger.error("Failed to get balance: %s", e)

        cycle = 0
        while True:
            # Market hours check
            if until_hour_mt is not None:
                mt = timezone(timedelta(hours=-7))
                now_mt = datetime.now(mt)
                if now_mt.hour >= until_hour_mt:
                    self.logger.info("Past %d:00 MT -- auto-exiting market window", until_hour_mt)
                    break

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
        choices=["scan", "paper", "live", "reconcile"],
        help="scan=read-only, paper=simulated trades, live=real money, reconcile=settle paper trades"
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
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date for reconcile (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--until",
        type=int,
        default=None,
        help="Auto-exit at this MT hour (e.g. --until 16 = stop at 4 PM MT)"
    )
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else config.LOG_LEVEL)

    if args.mode == "reconcile":
        from paper_tracker import PaperTracker
        auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
        kalshi = KalshiClient(auth=auth)
        tracker = PaperTracker(kalshi_client=kalshi)
        logging.getLogger("bot").info("Running paper trade reconciliation...")
        results = tracker.reconcile_settlements()
        tracker.generate_daily_summary(date_str=args.date)
        logging.getLogger("bot").info("Reconciliation complete: %d trades settled", len(results))
        return

    bot = WeatherBot(mode=args.mode)
    bot.run_loop(once=args.once, until_hour_mt=args.until)


if __name__ == "__main__":
    main()

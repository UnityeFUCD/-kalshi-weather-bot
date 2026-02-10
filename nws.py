"""
nws.py -- National Weather Service API client.

Fetches temperature forecasts and observations for Central Park, NY.
This is the SAME source that determines NHIGH contract settlement.

NWS API: https://api.weather.gov
No API key needed. Rate limit: ~1 req/sec with User-Agent header.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


class NWSClient:
    """
    NWS API client for Central Park, NY temperature data.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.NWS_USER_AGENT,
            "Accept": "application/geo+json",
        })
        self.grid_office = config.NWS_GRID_OFFICE
        self.grid_x = config.NWS_GRID_X
        self.grid_y = config.NWS_GRID_Y

    def get_forecast(self):
        """
        Get the 7-day forecast for Central Park.
        Returns the raw NWS forecast response (periods array).
        """
        url = ("https://api.weather.gov/gridpoints/"
               "%s/%d,%d/forecast" % (self.grid_office, self.grid_x, self.grid_y))
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._archive("forecast", data)
            return data
        except requests.RequestException as e:
            logger.error("NWS forecast fetch failed: %s", e)
            return None

    def get_hourly_forecast(self):
        """Get hourly forecast for next 156 hours."""
        url = ("https://api.weather.gov/gridpoints/"
               "%s/%d,%d/forecast/hourly" % (self.grid_office, self.grid_x, self.grid_y))
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._archive("hourly_forecast", data)
            return data
        except requests.RequestException as e:
            logger.error("NWS hourly forecast failed: %s", e)
            return None

    def get_latest_observation(self, station="KNYC"):
        """Get latest observation from a specific station."""
        url = "https://api.weather.gov/stations/%s/observations/latest" % station
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._archive("observation", data)
            return data
        except requests.RequestException as e:
            logger.error("NWS observation fetch failed: %s", e)
            return None

    def get_today_high_forecast(self):
        """
        Extract today's forecasted high temperature (F) from the daily forecast.
        Returns the first daytime period's temperature.
        """
        forecast = self.get_forecast()
        if not forecast:
            return None

        periods = forecast.get("properties", {}).get("periods", [])

        for period in periods:
            if period.get("isDaytime", False):
                temp = period.get("temperature")
                name = period.get("name", "")
                logger.info("NWS forecast: %s high = %dF", name, temp)
                return temp

        logger.warning("No daytime period found in NWS forecast")
        return None

    def get_high_forecast_for_date(self, target_date):
        """
        Get the forecasted high for a SPECIFIC date from the 7-day forecast.

        Args:
            target_date: datetime.date object for the target day

        Returns:
            (temperature_F, period_name) tuple, or (None, None) if not found

        The NWS 7-day forecast has periods like:
            "Today" (daytime), "Tonight" (nighttime),
            "Monday" (daytime), "Monday Night" (nighttime), etc.
        Each period has a startTime with the actual date.
        """
        forecast = self.get_forecast()
        if not forecast:
            return None, None

        periods = forecast.get("properties", {}).get("periods", [])

        for period in periods:
            if not period.get("isDaytime", False):
                continue

            # Parse the startTime to get the date
            start_str = period.get("startTime", "")
            try:
                # NWS format: "2026-02-10T06:00:00-05:00"
                start_dt = datetime.fromisoformat(start_str)
                period_date = start_dt.date()
            except (ValueError, TypeError):
                continue

            if period_date == target_date:
                temp = period.get("temperature")
                name = period.get("name", "")
                logger.info("NWS forecast for %s (%s): high = %dF",
                           target_date, name, temp)
                return temp, name

        # If exact date not found, log what we have
        logger.warning("No forecast found for %s", target_date)
        logger.info("Available daytime periods:")
        for period in periods:
            if period.get("isDaytime", False):
                logger.info("  %s: %dF (start=%s)",
                           period.get("name", "?"),
                           period.get("temperature", 0),
                           period.get("startTime", "?"))

        return None, None

    def get_current_temp(self):
        """Get current observed temperature (F) from Central Park station."""
        obs = self.get_latest_observation("KNYC")
        if not obs:
            return None

        props = obs.get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        if temp_c is not None:
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            logger.info("Current Central Park temp: %.1fF", temp_f)
            return temp_f
        return None

    def _archive(self, data_type, data):
        """Save raw API response with timestamp for backtest integrity."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")

        out_dir = config.FORECAST_SNAPSHOTS_DIR if "forecast" in data_type \
            else config.OBSERVATIONS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = "%s_%s.json" % (data_type, ts)
        filepath = out_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump({
                    "fetched_at": now.isoformat(),
                    "data_type": data_type,
                    "response": data,
                }, f, indent=2)
            logger.debug("Archived %s -> %s", data_type, filepath)
        except IOError as e:
            logger.warning("Failed to archive %s: %s", data_type, e)


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("NWS CLIENT TEST")
    print("=" * 60)

    nws = NWSClient()

    print("\n[1] Today's forecast...")
    high = nws.get_today_high_forecast()
    if high is not None:
        print("  Today's forecasted high: %dF" % high)

    print("\n[2] Tomorrow's forecast...")
    et_offset = timezone(timedelta(hours=-5))
    tomorrow = (datetime.now(et_offset) + timedelta(days=1)).date()
    temp, name = nws.get_high_forecast_for_date(tomorrow)
    if temp is not None:
        print("  Tomorrow (%s, %s) forecasted high: %dF" % (tomorrow, name, temp))
    else:
        print("  Could not get tomorrow's forecast")

    print("\n[3] Current observation...")
    temp = nws.get_current_temp()
    if temp is not None:
        print("  Current temp: %.1fF" % temp)

    print("\n" + "=" * 60)
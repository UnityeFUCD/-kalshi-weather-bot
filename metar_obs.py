"""
metar_obs.py -- METAR/ASOS Real-Time Observation Integration (Phase 4B).

Fetches real-time surface observations from multiple sources with failover:
  1. AWC Data API (aviationweather.gov) -- primary, most reliable
  2. Iowa Environmental Mesonet (IEM) JSON -- secondary, fast
  3. NWS Station Observations -- tertiary (already in nws.py)

Provides:
  - Current temperature (°F)
  - Max temperature so far today (°F)
  - Residual (obs - forecast)
  - Residual EWMA for observation-validated μ nudges
"""

import logging
import math
from datetime import datetime, timezone, timedelta
from collections import deque

import requests

import config

logger = logging.getLogger(__name__)


class MetarObserver:
    """
    Real-time METAR/ASOS observation client with multi-source failover.

    Usage:
        obs = MetarObserver("KNYC")
        temp = obs.get_current_temp()
        max_so_far = obs.get_max_so_far_today()
        nudge = obs.compute_mu_nudge(forecast_mu)
    """

    def __init__(self, station_id, nws_client=None):
        self.station_id = station_id
        self.nws_client = nws_client
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.NWS_USER_AGENT,
        })

        # Observation history for today (deque of (datetime, temp_f))
        self._obs_history = deque(maxlen=200)
        self._today_date = None

        # Residual EWMA state
        self._residual_ewma = None
        self._ewma_alpha = 2.0 / (config.OBS_RESIDUAL_EWMA_SPAN + 1)

    def get_current_temp(self):
        """
        Get current temperature (°F) from the best available source.
        Returns (temp_f, source_name, obs_time) or (None, None, None).
        """
        # Source 1: AWC Data API
        result = self._fetch_awc()
        if result:
            self._record_obs(result[0], result[2])
            return result

        # Source 2: IEM JSON
        result = self._fetch_iem()
        if result:
            self._record_obs(result[0], result[2])
            return result

        # Source 3: NWS station observation
        result = self._fetch_nws()
        if result:
            self._record_obs(result[0], result[2])
            return result

        logger.warning("All METAR sources failed for %s", self.station_id)
        return None, None, None

    def get_max_so_far_today(self):
        """Get the maximum observed temperature today."""
        self._check_day_rollover()
        if not self._obs_history:
            return None
        return max(t for _, t in self._obs_history)

    def get_obs_age_minutes(self):
        """Minutes since last successful observation."""
        if not self._obs_history:
            return float("inf")
        last_time = self._obs_history[-1][0]
        age = (datetime.now(timezone.utc) - last_time).total_seconds() / 60.0
        return age

    def compute_residual(self, forecast_mu):
        """
        Compute residual = current_obs - forecast_mu.
        Updates EWMA. Returns (residual, residual_ewma) or (None, None).
        """
        if not self._obs_history:
            return None, None

        last_temp = self._obs_history[-1][1]
        residual = last_temp - forecast_mu

        # Update EWMA
        if self._residual_ewma is None:
            self._residual_ewma = residual
        else:
            self._residual_ewma = (self._ewma_alpha * residual +
                                   (1 - self._ewma_alpha) * self._residual_ewma)

        return residual, self._residual_ewma

    def compute_mu_nudge(self, forecast_mu):
        """
        Compute observation-validated μ nudge.

        Two components:
        1. Residual-confirmed shift: k1 * residual_ewma (if consistent direction)
        2. Max-so-far floor: if max_so_far > forecast_mu, nudge μ up

        Returns adjusted_mu.
        """
        adjusted = forecast_mu

        # Component 1: Residual EWMA nudge
        residual, ewma = self.compute_residual(forecast_mu)
        if ewma is not None and abs(ewma) > 0.3:
            nudge = config.OBS_NUDGE_K1 * ewma
            adjusted += nudge
            logger.debug("μ nudge from residual EWMA: %+.2f (ewma=%.2f)", nudge, ewma)

        # Component 2: Max-so-far floor
        max_so_far = self.get_max_so_far_today()
        if max_so_far is not None and max_so_far > adjusted:
            logger.debug("μ floor from max-so-far: %.1f -> %.1f", adjusted, max_so_far)
            adjusted = max_so_far

        return adjusted

    def _record_obs(self, temp_f, obs_time):
        """Record an observation for today's history."""
        self._check_day_rollover()
        if obs_time is None:
            obs_time = datetime.now(timezone.utc)
        self._obs_history.append((obs_time, temp_f))

    def _check_day_rollover(self):
        """Reset observation history at midnight ET."""
        et_offset = timezone(timedelta(hours=-5))
        today = datetime.now(et_offset).date()
        if self._today_date != today:
            self._obs_history.clear()
            self._residual_ewma = None
            self._today_date = today

    # --- Source 1: AWC Data API (aviationweather.gov) -------------------------

    def _fetch_awc(self):
        """Fetch latest METAR from Aviation Weather Center."""
        url = "https://aviationweather.gov/api/data/metar"
        params = {
            "ids": self.station_id,
            "format": "json",
            "hours": 2,
        }
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            # Most recent observation first
            obs = data[0]
            temp_c = obs.get("temp")
            if temp_c is None:
                return None

            temp_f = temp_c * 9.0 / 5.0 + 32.0
            obs_time_str = obs.get("reportTime") or obs.get("obsTime")
            obs_time = None
            if obs_time_str:
                try:
                    obs_time = datetime.fromisoformat(
                        obs_time_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    obs_time = datetime.now(timezone.utc)

            logger.debug("AWC obs: %.1fF at %s", temp_f, obs_time)
            return temp_f, "awc", obs_time

        except Exception as e:
            logger.debug("AWC fetch failed for %s: %s", self.station_id, e)
            return None

    # --- Source 2: Iowa Environmental Mesonet (IEM) ---------------------------

    def _fetch_iem(self):
        """Fetch latest observation from IEM JSON service."""
        url = ("https://mesonet.agron.iastate.edu/json/current.py"
               "?station=%s&network=NY_ASOS" % self.station_id)
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            last_ob = data.get("last_ob", {})
            temp_f = last_ob.get("airtemp[F]")
            if temp_f is None:
                # Try alternate key format
                for key in last_ob:
                    if "airtemp" in key.lower() and "f" in key.lower():
                        temp_f = last_ob[key]
                        break

            if temp_f is None:
                return None

            obs_time_str = last_ob.get("local_valid") or last_ob.get("utc_valid")
            obs_time = None
            if obs_time_str:
                try:
                    obs_time = datetime.fromisoformat(obs_time_str)
                    if obs_time.tzinfo is None:
                        obs_time = obs_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    obs_time = datetime.now(timezone.utc)

            logger.debug("IEM obs: %.1fF at %s", temp_f, obs_time)
            return float(temp_f), "iem", obs_time

        except Exception as e:
            logger.debug("IEM fetch failed for %s: %s", self.station_id, e)
            return None

    # --- Source 3: NWS Station Observation ------------------------------------

    def _fetch_nws(self):
        """Fetch latest observation from NWS API (via nws_client or direct)."""
        if self.nws_client:
            obs = self.nws_client.get_latest_observation(station=self.station_id)
        else:
            url = "https://api.weather.gov/stations/%s/observations/latest" % self.station_id
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                obs = resp.json()
            except Exception as e:
                logger.debug("NWS obs fetch failed for %s: %s", self.station_id, e)
                return None

        if obs is None:
            return None

        props = obs.get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        if temp_c is None:
            return None

        temp_f = temp_c * 9.0 / 5.0 + 32.0

        obs_time_str = props.get("timestamp")
        obs_time = None
        if obs_time_str:
            try:
                obs_time = datetime.fromisoformat(
                    obs_time_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                obs_time = datetime.now(timezone.utc)

        logger.debug("NWS obs: %.1fF at %s", temp_f, obs_time)
        return temp_f, "nws", obs_time


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("METAR OBSERVER TEST")
    print("=" * 60)

    station = "KNYC"
    print("\nStation: %s" % station)

    obs = MetarObserver(station)

    print("\n[1] Current temperature...")
    temp, source, obs_time = obs.get_current_temp()
    if temp is not None:
        print("  %.1fF from %s at %s" % (temp, source, obs_time))
    else:
        print("  No observation available")

    print("\n[2] Max so far today...")
    max_t = obs.get_max_so_far_today()
    if max_t is not None:
        print("  Max today: %.1fF" % max_t)
    else:
        print("  No observations yet today")

    print("\n[3] Residual (using mock forecast of 35F)...")
    residual, ewma = obs.compute_residual(35.0)
    if residual is not None:
        print("  Residual: %+.1fF, EWMA: %+.2fF" % (residual, ewma))

    print("\n[4] μ nudge (forecast=35F)...")
    adjusted = obs.compute_mu_nudge(35.0)
    print("  Original μ: 35.0, Adjusted μ: %.1f" % adjusted)

    print("\n" + "=" * 60)

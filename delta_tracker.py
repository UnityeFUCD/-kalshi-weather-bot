"""
delta_tracker.py -- Forecast Revision Tracking (Phase 4A).

Tracks NWS forecast revisions via Last-Modified headers and computes
delta features between consecutive snapshots.  Persists state to JSON
so the bot can detect when forecasts change between scan cycles.

Delta features per revision event:
  - delta_tmax_hourly: change in hourly-derived daily max
  - delta_t13: change in 1 PM forecast temp
  - delta_t15: change in 3 PM forecast temp
  - delta_peak_ramp: change in max hour-to-hour ramp rate
  - revision_count: cumulative revisions for this target date
"""

import json
import logging
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class ForecastSnapshot:
    """A single point-in-time forecast snapshot."""

    def __init__(self, fetched_at, last_modified, target_date,
                 period_high, hourly_temps, hourly_max, t13, t15, peak_ramp):
        self.fetched_at = fetched_at          # datetime (UTC)
        self.last_modified = last_modified    # str or None
        self.target_date = target_date        # date
        self.period_high = period_high        # int (NWS period forecast)
        self.hourly_temps = hourly_temps      # list of (hour, temp_f)
        self.hourly_max = hourly_max          # float (max of hourly)
        self.t13 = t13                        # float (1 PM temp)
        self.t15 = t15                        # float (3 PM temp)
        self.peak_ramp = peak_ramp            # float (max hour-to-hour delta)

    def to_dict(self):
        return {
            "fetched_at": self.fetched_at.isoformat(),
            "last_modified": self.last_modified,
            "target_date": str(self.target_date),
            "period_high": self.period_high,
            "hourly_max": self.hourly_max,
            "t13": self.t13,
            "t15": self.t15,
            "peak_ramp": self.peak_ramp,
        }


class DeltaEvent:
    """A detected forecast revision with computed deltas."""

    def __init__(self, timestamp, target_date, delta_tmax_hourly,
                 delta_t13, delta_t15, delta_peak_ramp, revision_number):
        self.timestamp = timestamp
        self.target_date = target_date
        self.delta_tmax_hourly = delta_tmax_hourly
        self.delta_t13 = delta_t13
        self.delta_t15 = delta_t15
        self.delta_peak_ramp = delta_peak_ramp
        self.revision_number = revision_number

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "target_date": str(self.target_date),
            "delta_tmax_hourly": self.delta_tmax_hourly,
            "delta_t13": self.delta_t13,
            "delta_t15": self.delta_t15,
            "delta_peak_ramp": self.delta_peak_ramp,
            "revision_number": self.revision_number,
        }


class DeltaTracker:
    """
    Tracks forecast revisions for a single market.

    Usage:
        tracker = DeltaTracker("KXHIGHNY")
        snapshot = tracker.process_forecast(nws_client, target_date)
        if tracker.has_revision(target_date):
            delta = tracker.latest_delta(target_date)
    """

    def __init__(self, series_ticker):
        self.series_ticker = series_ticker
        self._state_path = config.DATA_DIR / ("delta_state_%s.json" % series_ticker)
        self._snapshots = {}   # target_date_str -> list[ForecastSnapshot]
        self._deltas = {}      # target_date_str -> list[DeltaEvent]
        self._last_modified = {}  # endpoint_key -> last_modified_str
        self._load_state()

    def _load_state(self):
        if self._state_path.exists():
            try:
                with open(self._state_path, "r") as f:
                    state = json.load(f)
                self._last_modified = state.get("last_modified", {})
                # We only persist last_modified; snapshots are transient per day
            except (json.JSONDecodeError, IOError):
                pass

    def _save_state(self):
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "last_modified": self._last_modified,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.replace(self._state_path)

    def process_hourly_forecast(self, nws_client, target_date):
        """
        Fetch hourly forecast and build a snapshot for target_date.
        Returns (ForecastSnapshot, is_revision: bool).

        The nws_client should already be configured with correct grid coords.
        """
        # Fetch hourly with Last-Modified tracking
        hourly_data, last_mod = self._fetch_with_last_modified(nws_client)
        if hourly_data is None:
            # 304 Not Modified or error -- fall back to direct fetch without caching
            logger.debug("Last-Modified fetch returned None, trying direct fetch")
            hourly_data = nws_client.get_hourly_forecast()
            if hourly_data is None:
                return None, False

        # Extract hourly temps for the target date
        periods = hourly_data.get("properties", {}).get("periods", [])
        hourly_temps = []

        for period in periods:
            start_str = period.get("startTime", "")
            try:
                start_dt = datetime.fromisoformat(start_str)
                if start_dt.date() != target_date:
                    continue
                hour = start_dt.hour
                temp = period.get("temperature")
                if temp is not None:
                    hourly_temps.append((hour, temp))
            except (ValueError, TypeError):
                continue

        if not hourly_temps:
            logger.warning("No hourly temps found for %s", target_date)
            return None, False

        # Compute features
        hourly_max = max(t for _, t in hourly_temps)
        t13 = self._get_temp_at_hour(hourly_temps, 13)
        t15 = self._get_temp_at_hour(hourly_temps, 15)
        peak_ramp = self._compute_peak_ramp(hourly_temps)

        # Get period high for comparison
        period_high = self._get_period_high(nws_client, target_date)

        now = datetime.now(timezone.utc)
        snapshot = ForecastSnapshot(
            fetched_at=now,
            last_modified=last_mod,
            target_date=target_date,
            period_high=period_high,
            hourly_temps=hourly_temps,
            hourly_max=hourly_max,
            t13=t13,
            t15=t15,
            peak_ramp=peak_ramp,
        )

        # Check for revision
        date_key = str(target_date)
        is_revision = False

        if date_key not in self._snapshots:
            self._snapshots[date_key] = []

        prev_snapshots = self._snapshots[date_key]

        if prev_snapshots:
            prev = prev_snapshots[-1]
            # Detect revision: Last-Modified changed OR hourly_max changed
            if (last_mod and last_mod != prev.last_modified) or \
               abs(hourly_max - prev.hourly_max) >= 0.5:
                is_revision = True
                rev_num = len(self._deltas.get(date_key, [])) + 1

                delta = DeltaEvent(
                    timestamp=now,
                    target_date=target_date,
                    delta_tmax_hourly=hourly_max - prev.hourly_max,
                    delta_t13=(t13 - prev.t13) if (t13 and prev.t13) else None,
                    delta_t15=(t15 - prev.t15) if (t15 and prev.t15) else None,
                    delta_peak_ramp=(peak_ramp - prev.peak_ramp) if (peak_ramp and prev.peak_ramp) else None,
                    revision_number=rev_num,
                )

                if date_key not in self._deltas:
                    self._deltas[date_key] = []
                self._deltas[date_key].append(delta)

                logger.info("REVISION #%d for %s: ΔTmax=%.1f, ΔT13=%s, ΔT15=%s",
                           rev_num, target_date, delta.delta_tmax_hourly,
                           "%.1f" % delta.delta_t13 if delta.delta_t13 else "N/A",
                           "%.1f" % delta.delta_t15 if delta.delta_t15 else "N/A")

        self._snapshots[date_key].append(snapshot)
        # Keep only last 20 snapshots per date
        if len(self._snapshots[date_key]) > 20:
            self._snapshots[date_key] = self._snapshots[date_key][-20:]

        self._save_state()
        return snapshot, is_revision

    def get_hourly_mu(self, target_date):
        """
        Get the best μ estimate from hourly forecast data.
        Returns hourly_max if available, else None.
        """
        date_key = str(target_date)
        snapshots = self._snapshots.get(date_key, [])
        if snapshots:
            return snapshots[-1].hourly_max
        return None

    def has_revision(self, target_date):
        """Check if any revision was detected for this target date."""
        return bool(self._deltas.get(str(target_date)))

    def latest_delta(self, target_date):
        """Get the most recent DeltaEvent for this target date."""
        deltas = self._deltas.get(str(target_date), [])
        return deltas[-1] if deltas else None

    def revision_count(self, target_date):
        """Number of revisions detected for this target date."""
        return len(self._deltas.get(str(target_date), []))

    def revision_volatility(self, target_date):
        """
        Compute revision volatility = stdev of delta_tmax_hourly values.
        Returns 0.0 if fewer than 2 revisions.
        """
        deltas = self._deltas.get(str(target_date), [])
        vals = [d.delta_tmax_hourly for d in deltas if d.delta_tmax_hourly is not None]
        if len(vals) < 2:
            return 0.0
        return statistics.stdev(vals)

    def latest_snapshot(self, target_date):
        """Get the most recent ForecastSnapshot for this target date."""
        snapshots = self._snapshots.get(str(target_date), [])
        return snapshots[-1] if snapshots else None

    def forecast_age_minutes(self, target_date):
        """Minutes since last forecast fetch for this target date."""
        snap = self.latest_snapshot(target_date)
        if snap is None:
            return float("inf")
        age = (datetime.now(timezone.utc) - snap.fetched_at).total_seconds() / 60.0
        return age

    def _fetch_with_last_modified(self, nws_client):
        """
        Fetch hourly forecast with If-Modified-Since / Last-Modified tracking.
        Returns (data_dict, last_modified_str).
        """
        endpoint_key = "hourly_%s_%d_%d" % (
            nws_client.grid_office, nws_client.grid_x, nws_client.grid_y)

        url = ("https://api.weather.gov/gridpoints/"
               "%s/%d,%d/forecast/hourly" % (
                   nws_client.grid_office, nws_client.grid_x, nws_client.grid_y))

        headers = {}
        prev_lm = self._last_modified.get(endpoint_key)
        if prev_lm:
            headers["If-Modified-Since"] = prev_lm

        try:
            resp = nws_client.session.get(url, timeout=15, headers=headers)

            if resp.status_code == 304:
                logger.debug("Hourly forecast not modified since %s", prev_lm)
                # Return cached data if we have it
                return None, prev_lm

            resp.raise_for_status()

            last_mod = resp.headers.get("Last-Modified")
            if last_mod:
                self._last_modified[endpoint_key] = last_mod

            data = resp.json()
            return data, last_mod

        except Exception as e:
            logger.error("Hourly forecast fetch with Last-Modified failed: %s", e)
            return None, None

    def _get_temp_at_hour(self, hourly_temps, target_hour):
        """Get temperature at a specific hour, or nearest."""
        for hour, temp in hourly_temps:
            if hour == target_hour:
                return temp
        # Find nearest
        nearest = min(hourly_temps, key=lambda x: abs(x[0] - target_hour), default=None)
        return nearest[1] if nearest else None

    def _compute_peak_ramp(self, hourly_temps):
        """Compute max hour-to-hour temperature change."""
        if len(hourly_temps) < 2:
            return 0.0
        sorted_temps = sorted(hourly_temps, key=lambda x: x[0])
        max_ramp = 0.0
        for i in range(1, len(sorted_temps)):
            ramp = abs(sorted_temps[i][1] - sorted_temps[i - 1][1])
            max_ramp = max(max_ramp, ramp)
        return max_ramp

    def _get_period_high(self, nws_client, target_date):
        """Get the NWS 7-day period high for comparison."""
        temp, _ = nws_client.get_high_forecast_for_date(target_date)
        return temp


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from nws import NWSClient
    from market_registry import get_enabled_markets

    print("=" * 60)
    print("DELTA TRACKER TEST")
    print("=" * 60)

    markets = get_enabled_markets()
    if not markets:
        print("No enabled markets")
        sys.exit(1)

    mc = markets[0]
    nws = NWSClient(market_config=mc)
    tracker = DeltaTracker(mc.series_ticker)

    # Get target date
    now_et = datetime.now(config.MARKET_TZ)
    if now_et.hour >= 8:
        target = (now_et + timedelta(days=1)).date()
    else:
        target = now_et.date()

    print("\nTarget date: %s" % target)
    print("Fetching hourly forecast...")

    snapshot, is_revision = tracker.process_hourly_forecast(nws, target)

    if snapshot:
        print("\nSnapshot:")
        print("  Period high:  %s" % snapshot.period_high)
        print("  Hourly max:   %.0f" % snapshot.hourly_max)
        print("  T13 (1PM):    %s" % snapshot.t13)
        print("  T15 (3PM):    %s" % snapshot.t15)
        print("  Peak ramp:    %.1f" % snapshot.peak_ramp)
        print("  Is revision:  %s" % is_revision)

        hourly_mu = tracker.get_hourly_mu(target)
        print("\n  Hourly-derived μ: %.1f" % hourly_mu)
    else:
        print("  No hourly data available")

    print("\n" + "=" * 60)

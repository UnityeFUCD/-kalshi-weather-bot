"""
ensemble.py -- Ensemble σ from Open-Meteo (Phase 5).

Fetches ensemble forecast data from the Open-Meteo Ensemble API:
  - ECMWF IFS 0.25° (51 members)
  - GFS Seamless (31 members)

Computes ensemble spread (σ_ens) from member Tmax values and composes
the final σ using multiple uncertainty sources:

  σ_final = max(
      σ_base,
      α × σ_ens,
      σ_base + β × revision_volatility,
      σ_base + γ × boundary_risk
  )
"""

import logging
import statistics
import math
from datetime import datetime, timezone, timedelta

import requests

import config

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Fetches ensemble Tmax forecasts and computes calibrated σ.

    Usage:
        ens = EnsembleForecaster(lat=40.7831, lon=-73.9712)
        sigma_ens = ens.get_ensemble_sigma(target_date)
        sigma_final = ens.compose_sigma(sigma_base, sigma_ens, rev_vol, boundary_z)
    """

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.session = requests.Session()
        self._cache = {}  # target_date_str -> (sigma_ens, member_maxes, fetched_at)

    def get_ensemble_sigma(self, target_date):
        """
        Fetch ensemble forecasts and compute σ_ens for target_date.

        Returns (sigma_ens, n_members, member_maxes) or (None, 0, []).
        """
        date_key = str(target_date)

        # Check cache
        if date_key in self._cache:
            cached = self._cache[date_key]
            age_min = (datetime.now(timezone.utc) - cached[2]).total_seconds() / 60
            if age_min < config.ENSEMBLE_CACHE_MINUTES:
                logger.debug("Ensemble cache hit for %s (%.0f min old)", date_key, age_min)
                return cached[0], len(cached[1]), cached[1]

        all_maxes = []

        for model in config.ENSEMBLE_MODELS:
            maxes = self._fetch_model_ensemble(model, target_date)
            all_maxes.extend(maxes)

        if len(all_maxes) < 5:
            logger.warning("Only %d ensemble members for %s -- insufficient",
                          len(all_maxes), target_date)
            return None, len(all_maxes), all_maxes

        sigma_ens = statistics.stdev(all_maxes)
        mean_ens = statistics.mean(all_maxes)

        logger.info("Ensemble: %d members, mean=%.1fF, σ_ens=%.2fF",
                    len(all_maxes), mean_ens, sigma_ens)

        # Cache
        self._cache[date_key] = (sigma_ens, all_maxes, datetime.now(timezone.utc))

        return sigma_ens, len(all_maxes), all_maxes

    def compose_sigma(self, sigma_base, sigma_ens=None, revision_volatility=0.0,
                      boundary_risk=0.0):
        """
        Compose final σ from multiple uncertainty sources.

        σ_final = max(
            σ_base,
            α × σ_ens,                        (ensemble spread)
            σ_base + β × revision_volatility,  (forecast instability)
            σ_base + γ × boundary_risk          (proximity to bucket edge)
        )
        """
        candidates = [sigma_base]

        if sigma_ens is not None and sigma_ens > 0:
            candidates.append(config.ENSEMBLE_ALPHA * sigma_ens)

        if revision_volatility > 0:
            candidates.append(sigma_base + config.ENSEMBLE_BETA * revision_volatility)

        if boundary_risk > 0:
            candidates.append(sigma_base + config.ENSEMBLE_GAMMA * boundary_risk)

        sigma_final = max(candidates)

        if sigma_final > sigma_base:
            logger.info("σ composed: base=%.2f -> final=%.2f (ens=%s, rev=%.2f, bnd=%.2f)",
                       sigma_base, sigma_final,
                       "%.2f" % sigma_ens if sigma_ens else "N/A",
                       revision_volatility, boundary_risk)

        return sigma_final

    def _fetch_model_ensemble(self, model_name, target_date):
        """
        Fetch ensemble member Tmax values for a specific model.
        Returns list of Tmax (°F) values, one per member.
        """
        url = "https://ensemble-api.open-meteo.com/v1/ensemble"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": model_name,
        }

        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            # Response has daily -> temperature_2m_max -> list per member
            daily = data.get("daily", {})
            maxes = []

            # Open-Meteo returns member data as temperature_2m_max_member01, etc
            # or as a single list if only one date
            for key, values in daily.items():
                if "temperature_2m_max" in key and key != "time":
                    if isinstance(values, list) and len(values) > 0:
                        val = values[0]  # first (only) date
                        if val is not None:
                            maxes.append(float(val))

            if not maxes:
                # Try alternate response format
                tmax_data = daily.get("temperature_2m_max")
                if isinstance(tmax_data, list):
                    for val in tmax_data:
                        if val is not None:
                            maxes.append(float(val))

            logger.debug("Ensemble %s: %d members for %s",
                        model_name, len(maxes), target_date)
            return maxes

        except Exception as e:
            logger.warning("Ensemble fetch failed for %s/%s: %s",
                          model_name, target_date, e)
            return []

    def get_ensemble_mean(self, target_date):
        """Get ensemble mean Tmax for target_date."""
        _, _, maxes = self.get_ensemble_sigma(target_date)
        if maxes:
            return statistics.mean(maxes)
        return None


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("ENSEMBLE FORECASTER TEST")
    print("=" * 60)

    # NYC Central Park coordinates
    ens = EnsembleForecaster(lat=40.7831, lon=-73.9712)

    et_offset = timezone(timedelta(hours=-5))
    now_et = datetime.now(et_offset)
    tomorrow = (now_et + timedelta(days=1)).date()

    print("\nTarget date: %s" % tomorrow)
    print("Fetching ensemble forecasts...")

    sigma_ens, n_members, maxes = ens.get_ensemble_sigma(tomorrow)

    if sigma_ens is not None:
        print("\nResults:")
        print("  Members: %d" % n_members)
        print("  σ_ens: %.2fF" % sigma_ens)
        print("  Mean: %.1fF" % statistics.mean(maxes))
        print("  Min:  %.1fF" % min(maxes))
        print("  Max:  %.1fF" % max(maxes))

        # Test composition
        print("\nσ composition (base=1.2):")
        sigma_final = ens.compose_sigma(
            sigma_base=1.2,
            sigma_ens=sigma_ens,
            revision_volatility=0.5,
            boundary_risk=0.3,
        )
        print("  Final σ: %.2fF" % sigma_final)
    else:
        print("  Could not fetch ensemble data")

    print("\n" + "=" * 60)

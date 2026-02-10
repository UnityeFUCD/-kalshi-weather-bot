"""
scripts/calibrate_from_history.py -- Compute REAL Sigma and Bias from Historical Data

This script does NOT depend on your NWS forecast snapshots.
Instead it uses:
  1. Open-Meteo Historical Forecast API (archived NWS/GFS model runs)
  2. Your GHCN-Daily actual observations for Central Park

This gives us HUNDREDS of forecast-vs-actual pairs across seasons,
enough to compute statistically meaningful sigma and bias values.

The Open-Meteo Historical Forecast API archives past weather model
outputs, so we can ask "what did the forecast say on Jan 1 about
Jan 2?" and compare it to what actually happened on Jan 2.

Usage:
    cd C:\\Users\\fycin\\Desktop\\kelshi-weather-bot\\-kalshi-weather-bot
    py calibrate_from_history.py
    py calibrate_from_history.py --months 6
"""

import sys
import argparse
import logging
import math
from pathlib import Path
from datetime import datetime, timedelta, date

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_historical_forecasts(lat, lon, start_date, end_date):
    """
    Fetch archived forecast model outputs from Open-Meteo.
    
    The Historical Forecast API returns what the weather model
    predicted at the time, NOT the actual outcome. This is exactly
    what we need — past forecasts that we can compare to observations.
    
    API: https://open-meteo.com/en/docs/historical-forecast-api
    """
    import requests
    import pandas as pd
    
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    # Fetch in chunks of 90 days to avoid timeouts
    all_records = []
    chunk_start = start_date
    chunk_size = timedelta(days=90)
    
    while chunk_start < end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        
        logger.info("  Fetching forecasts %s to %s...", chunk_start, chunk_end)
        
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            dates = data.get("daily", {}).get("time", [])
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            
            for d, t in zip(dates, temps):
                if t is not None:
                    all_records.append({
                        "date": datetime.strptime(d, "%Y-%m-%d").date(),
                        "forecast_high_f": round(t),
                    })
            
            logger.info("    Got %d days", len(dates))
            
        except Exception as e:
            logger.warning("    Chunk failed: %s", e)
        
        chunk_start = chunk_end + timedelta(days=1)
    
    df = pd.DataFrame(all_records)
    logger.info("Total historical forecasts: %d days", len(df))
    return df


def fetch_historical_observations(lat, lon, start_date, end_date):
    """
    Fetch actual observed temperatures from Open-Meteo archive API.
    
    We use this as a cross-check and gap-filler for GHCN-Daily.
    The archive API serves ERA5 reanalysis which is ground-truth quality.
    """
    import requests
    import pandas as pd
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    all_records = []
    chunk_start = start_date
    chunk_size = timedelta(days=90)
    
    while chunk_start < end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        
        logger.info("  Fetching observations %s to %s...", chunk_start, chunk_end)
        
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            dates = data.get("daily", {}).get("time", [])
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            
            for d, t in zip(dates, temps):
                if t is not None:
                    all_records.append({
                        "date": datetime.strptime(d, "%Y-%m-%d").date(),
                        "actual_high_f": round(t),
                    })
            
        except Exception as e:
            logger.warning("    Chunk failed: %s", e)
        
        chunk_start = chunk_end + timedelta(days=1)
    
    df = pd.DataFrame(all_records)
    logger.info("Total historical observations: %d days", len(df))
    return df


def load_ghcn_observations(ghcn_station="USW00094728"):
    """Try to load GHCN-Daily parquet if available."""
    import pandas as pd

    # Search for the file
    pattern = "%s_daily.parquet" % ghcn_station
    for p in PROJECT_ROOT.rglob(pattern):
        logger.info("Found GHCN data: %s", p)
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.rename(columns={"tmax_f": "actual_high_f"})
        df["source"] = "ghcn"
        return df[["date", "actual_high_f", "source"]].dropna(subset=["actual_high_f"])

    return None


def build_calibration(forecasts_df, observations_df):
    """Join forecasts to observations and compute error stats."""
    import pandas as pd
    
    merged = forecasts_df.merge(observations_df[["date", "actual_high_f"]], on="date", how="inner")
    
    merged["error"] = merged["actual_high_f"] - merged["forecast_high_f"]
    merged["abs_error"] = merged["error"].abs()
    
    # Time features
    merged["month"] = pd.to_datetime(merged["date"]).dt.month
    merged["season"] = merged["month"].map({
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    })
    
    return merged


def print_full_analysis(cal_df, current_config=None):
    """Print comprehensive calibration analysis."""
    
    n = len(cal_df)
    
    print("\n" + "=" * 65)
    print("HISTORICAL CALIBRATION ANALYSIS")
    print("  Based on %d forecast-vs-observation pairs" % n)
    print("  Date range: %s to %s" % (cal_df["date"].min(), cal_df["date"].max()))
    print("=" * 65)
    
    # ── Overall ──────────────────────────────────────────────────
    print("\n── OVERALL ERROR DISTRIBUTION ──")
    bias = cal_df["error"].mean()
    sigma = cal_df["error"].std()
    mae = cal_df["abs_error"].mean()
    
    print("  Mean error (bias):   %+.2f°F" % bias)
    print("  Std dev (sigma):     %.2f°F" % sigma)
    print("  Mean absolute error: %.2f°F" % mae)
    print("  Max absolute error:  %d°F" % cal_df["abs_error"].max())
    print("  Median error:        %+.1f°F" % cal_df["error"].median())
    
    # Percentiles
    errors = cal_df["error"]
    print("\n  Error percentiles:")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        val = errors.quantile(pct / 100)
        print("    %3dth: %+.1f°F" % (pct, val))
    
    # ── By Season ────────────────────────────────────────────────
    print("\n── BY SEASON ──")
    print("  %-8s %8s %8s %8s %6s" % ("Season", "Bias", "Sigma", "MAE", "N"))
    print("  " + "-" * 42)
    
    season_sigmas = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        subset = cal_df[cal_df["season"] == season]
        if len(subset) < 5:
            continue
        s_bias = subset["error"].mean()
        s_sigma = subset["error"].std()
        s_mae = subset["abs_error"].mean()
        season_sigmas[season] = s_sigma
        print("  %-8s %+7.2f %7.2f %7.2f %6d" % (
            season, s_bias, s_sigma, s_mae, len(subset)))
    
    # ── By Month ─────────────────────────────────────────────────
    print("\n── BY MONTH ──")
    print("  %-10s %8s %8s %8s %6s" % ("Month", "Bias", "Sigma", "MAE", "N"))
    print("  " + "-" * 44)
    
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    for m in range(1, 13):
        subset = cal_df[cal_df["month"] == m]
        if len(subset) < 3:
            continue
        print("  %-10s %+7.2f %7.2f %7.2f %6d" % (
            month_names[m], subset["error"].mean(), subset["error"].std(),
            subset["abs_error"].mean(), len(subset)))
    
    # ── Distribution Shape ───────────────────────────────────────
    print("\n── DISTRIBUTION SHAPE (is Normal a good fit?) ──")
    
    skew = errors.skew()
    kurt = errors.kurtosis()  # excess kurtosis
    
    print("  Skewness:         %.3f  (Normal = 0)" % skew)
    print("  Excess kurtosis:  %.3f  (Normal = 0)" % kurt)
    
    if abs(skew) < 0.5 and abs(kurt) < 1.0:
        print("  ✅ Approximately Normal — your Gaussian model is a good fit")
    else:
        if abs(kurt) > 1.0:
            print("  ⚠️  Heavier tails than Normal (kurtosis=%.1f)" % kurt)
            print("     Extreme errors are more common than Normal predicts")
            print("     Consider Student-t distribution for better tail modeling")
        if abs(skew) > 0.5:
            print("  ⚠️  Skewed distribution (%.2f)" % skew)
            print("     NWS has a systematic directional tendency")
    
    # ── What Fraction Falls in Each Error Band ───────────────────
    print("\n── ERROR FREQUENCY ──")
    print("  (How often is the forecast within X degrees?)")
    for threshold in [1, 2, 3, 4, 5]:
        frac = (cal_df["abs_error"] <= threshold).mean()
        print("  Within ±%d°F: %.0f%%" % (threshold, frac * 100))
    
    # ── Comparison to Current Config ─────────────────────────────
    if current_config:
        print("\n" + "=" * 65)
        print("CONFIG RECOMMENDATIONS")
        print("=" * 65)
        
        print("\n  Current config.py → Recommended (from %d days of data):" % n)
        print()
        
        # Note: Open-Meteo historical forecasts approximate a 1-day lead
        # time since they're the model run from the day before.
        # For same-day adjustments, we scale from the 1-day sigma.
        
        sigma_1day = sigma  # The historical forecast API gives ~1-day-ahead forecasts
        
        # Same-day estimates: scale down from 1-day
        # Rule of thumb from meteorology: sigma scales roughly as sqrt(lead_time)
        # Same-day AM (~12h lead) ≈ sigma_1day * sqrt(0.5) ≈ sigma_1day * 0.71
        # Same-day PM (~4h lead) ≈ sigma_1day * sqrt(0.17) ≈ sigma_1day * 0.41
        sigma_am = sigma_1day * 0.71
        sigma_pm = sigma_1day * 0.41
        
        cur_1day = current_config.get("SIGMA_1DAY", "?")
        cur_am = current_config.get("SIGMA_SAMEDAY_AM", "?")
        cur_pm = current_config.get("SIGMA_SAMEDAY_PM", "?")
        cur_bias = current_config.get("FORECAST_BIAS", "?")
        
        print("  SIGMA_1DAY      = %.1f   (was %s)" % (sigma_1day, cur_1day))
        print("  SIGMA_SAMEDAY_AM = %.1f   (was %s)  [estimated: σ_1day × 0.71]" % (sigma_am, cur_am))
        print("  SIGMA_SAMEDAY_PM = %.1f   (was %s)  [estimated: σ_1day × 0.41]" % (sigma_pm, cur_pm))
        print("  FORECAST_BIAS   = %.1f   (was %s)" % (bias, cur_bias))
        
        print("\n  ⚠️  Same-day estimates are scaled from 1-day sigma using √t rule.")
        print("     They'll get refined once you collect 14+ days of real NWS snapshots")
        print("     at different times of day.")
        
        # Season-specific sigma
        if season_sigmas:
            print("\n  OPTIONAL — Season-specific sigma (for future config upgrade):")
            for season, s in sorted(season_sigmas.items()):
                print("    %s: σ = %.2f°F" % (season, s))
            
            current_season = "DJF"  # February
            if current_season in season_sigmas:
                s_now = season_sigmas[current_season]
                print("\n  Current season (%s): σ = %.2f°F" % (current_season, s_now))
                if abs(s_now - sigma_1day) > 0.3:
                    print("  This differs from the yearly average (%.2f). Consider using" % sigma_1day)
                    print("  the seasonal value for better calibration right now.")
    
    # ── Sample of Largest Errors (what goes wrong) ───────────────
    print("\n── LARGEST ERRORS (learn from these) ──")
    worst = cal_df.nlargest(10, "abs_error")[["date", "forecast_high_f", "actual_high_f", "error", "season"]]
    print("  %-12s %8s %8s %6s %8s" % ("Date", "Fcst", "Actual", "Error", "Season"))
    print("  " + "-" * 46)
    for _, row in worst.iterrows():
        print("  %-12s %7d°F %6d°F %+5d°F %8s" % (
            row["date"], row["forecast_high_f"], row["actual_high_f"],
            row["error"], row["season"]))


def main():
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Historical forecast calibration")
    parser.add_argument("--months", type=int, default=6,
                        help="Months of history to analyze (default: 6)")
    parser.add_argument("--lat", type=float, default=40.7831,
                        help="Latitude (default: 40.7831 Central Park NYC)")
    parser.add_argument("--lon", type=float, default=-73.9712,
                        help="Longitude (default: -73.9712 Central Park NYC)")
    parser.add_argument("--ghcn-station", type=str, default="USW00094728",
                        help="GHCN station ID (default: USW00094728 Central Park)")
    parser.add_argument("--label", type=str, default="KNYC",
                        help="Label for output files (default: KNYC)")
    args = parser.parse_args()

    print("=" * 65)
    print("HISTORICAL CALIBRATION — Open-Meteo Forecasts vs Observations")
    print("=" * 65)
    print("This uses REAL historical data (not your NWS snapshots).")
    print("We're computing what sigma and bias SHOULD be based on")
    print("%d months of actual forecast-vs-outcome data." % args.months)
    print("Location: %.4f, %.4f  GHCN: %s  Label: %s" % (
        args.lat, args.lon, args.ghcn_station, args.label))
    print()

    lat, lon = args.lat, args.lon
    
    # Date range: end at 5 days ago (to ensure observations are finalized)
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=args.months * 30)
    
    # Step 1: Fetch historical forecasts
    print("[1] Fetching historical forecast archive from Open-Meteo...")
    print("    (%d months, %s to %s)" % (args.months, start_date, end_date))
    forecasts = fetch_historical_forecasts(lat, lon, start_date, end_date)
    
    if forecasts.empty:
        print("❌ Failed to fetch historical forecasts.")
        return
    
    # Step 2: Load observations
    print("\n[2] Loading observations...")
    
    # Try GHCN first (station-level, more accurate)
    ghcn = load_ghcn_observations(ghcn_station=args.ghcn_station)
    
    if ghcn is not None and len(ghcn) > 100:
        observations = ghcn
        print("  Using GHCN-Daily (station-level) observations: %d days" % len(observations))
    else:
        # Fall back to Open-Meteo archive (ERA5 reanalysis)
        print("  GHCN not available or too small, fetching from Open-Meteo archive...")
        observations = fetch_historical_observations(lat, lon, start_date, end_date)
    
    if observations.empty:
        print("❌ Failed to get observations.")
        return
    
    # Step 3: Build calibration table
    print("\n[3] Joining %d forecasts to %d observations..." % (len(forecasts), len(observations)))
    cal = build_calibration(forecasts, observations)
    
    if cal.empty:
        print("❌ No matching dates. Check date formats.")
        return
    
    print("  Matched: %d forecast-observation pairs" % len(cal))
    
    # Step 4: Save
    output_dir = PROJECT_ROOT / "data" / "curated" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("historical_calibration_%s.parquet" % args.label)
    cal.to_parquet(output_path, index=False)
    print("  Saved: %s" % output_path)
    
    # Step 5: Full analysis
    try:
        import config
        current_config = {
            "SIGMA_1DAY": config.SIGMA_1DAY,
            "SIGMA_SAMEDAY_AM": config.SIGMA_SAMEDAY_AM,
            "SIGMA_SAMEDAY_PM": config.SIGMA_SAMEDAY_PM,
            "FORECAST_BIAS": config.FORECAST_BIAS,
        }
    except ImportError:
        current_config = {
            "SIGMA_1DAY": 2.5,
            "SIGMA_SAMEDAY_AM": 1.5,
            "SIGMA_SAMEDAY_PM": 0.8,
            "FORECAST_BIAS": 0.0,
        }
    
    print_full_analysis(cal, current_config)
    
    print("\n" + "=" * 65)
    print("DONE — Update config.py with the recommended values above.")
    print()
    print("Next: Keep running 'py bot.py scan --once' daily.")
    print("After 14+ days, run build_calibration_dataset.py to refine")
    print("these estimates with YOUR NWS forecast snapshots.")
    print("=" * 65)


if __name__ == "__main__":
    main()

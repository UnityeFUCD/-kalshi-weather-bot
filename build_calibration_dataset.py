"""
build_calibration_dataset.py -- Join Forecasts to Settled Observations (v3)

IMPORTANT: Only uses GHCN-Daily USW00094728 as ground truth.
Never uses forecast endpoints as "actuals."
If no settled overlap exists, exits cleanly.

Usage:
    cd C:\\Users\\fycin\\Desktop\\kelshi-weather-bot\\-kalshi-weather-bot
    py build_calibration_dataset.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta, date

PROJECT_ROOT = Path(__file__).resolve().parent
# If we're in scripts/, go up one level
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_observation_file():
    """Search common locations for the GHCN observation parquet."""
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "weather" / "observations" / "USW00094728_daily.parquet",
        PROJECT_ROOT / "data" / "USW00094728_daily.parquet",
        PROJECT_ROOT / "USW00094728_daily.parquet",
    ]
    # Also search recursively
    for p in PROJECT_ROOT.rglob("USW00094728_daily.parquet"):
        if p not in candidates:
            candidates.append(p)
    
    for p in candidates:
        if p.exists():
            logger.info("Found observations at: %s", p)
            return p
    
    logger.warning("No observation parquet found. Searched:")
    for p in candidates[:3]:
        logger.warning("  %s", p)
    return None


def fetch_recent_from_open_meteo(start_date, end_date):
    """
    DISABLED -- Open-Meteo gap-fill removed per Phase 4 calibration patch.

    Settlement-only calibration: we ONLY use GHCN-Daily (official NWS data)
    as ground truth.  Open-Meteo ERA5/reanalysis data does NOT match NWS
    settlement methodology (different station, different rounding, different
    measurement window).  Using it introduces systematic bias into sigma
    calibration.

    This function now returns an empty list.  The gap between GHCN-Daily
    availability and forecast dates is handled by waiting for GHCN to update.
    """
    logger.info("Open-Meteo gap-fill DISABLED (settlement-only calibration)")
    return []


def load_settled_observations(obs_parquet_path):
    """
    Load ONLY settled GHCN-Daily observations.
    No forecast endpoints. No gap-filling with non-observation data.
    Returns a pandas DataFrame with columns: date, tmax_f, source
    Also returns the latest settled date for gating.
    """
    import pandas as pd

    if not obs_parquet_path or not obs_parquet_path.exists():
        logger.error("No GHCN observation file found at: %s", obs_parquet_path)
        return pd.DataFrame(columns=["date", "tmax_f", "source"]), None

    ghcn = pd.read_parquet(obs_parquet_path)
    ghcn["source"] = "ghcn_daily"
    ghcn["date"] = pd.to_datetime(ghcn["date"]).dt.date
    ghcn = ghcn[["date", "tmax_f", "source"]].dropna(subset=["tmax_f"])
    ghcn = ghcn.sort_values("date").reset_index(drop=True)

    latest_settled = ghcn["date"].max()
    logger.info("GHCN-Daily: %d days, latest settled date: %s", len(ghcn), latest_settled)

    return ghcn, latest_settled


def load_forecast_snapshots():
    """
    Load all archived NWS forecast snapshots.
    Searches multiple locations for the JSON files.
    """
    import pandas as pd
    import config
    
    # Search locations
    search_dirs = [
        config.FORECAST_SNAPSHOTS_DIR,
        PROJECT_ROOT / "data" / "raw" / "weather" / "nws_snapshots",
        PROJECT_ROOT / "data" / "forecast_snapshots",
    ]
    
    json_files = []
    for d in search_dirs:
        if d.exists():
            found = list(d.glob("forecast_*.json"))
            if found:
                logger.info("Found %d forecast files in %s", len(found), d)
                json_files.extend(found)
    
    # Deduplicate by filename
    seen = set()
    unique_files = []
    for f in json_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)
    json_files = sorted(unique_files)
    
    if not json_files:
        logger.warning("No forecast snapshot files found anywhere!")
        return pd.DataFrame()
    
    logger.info("Processing %d unique forecast files...", len(json_files))
    
    records = []
    for filepath in json_files:
        try:
            with open(filepath, "r") as f:
                wrapper = json.load(f)
            
            fetched_at = wrapper.get("fetched_at", "")
            data = wrapper.get("response", wrapper)
            
            periods = data.get("properties", {}).get("periods", [])
            
            for period in periods:
                if not period.get("isDaytime", False):
                    continue
                
                start_str = period.get("startTime", "")
                try:
                    start_dt = datetime.fromisoformat(start_str)
                    target_date = start_dt.date()
                except (ValueError, TypeError):
                    continue
                
                temp = period.get("temperature")
                if temp is None:
                    continue
                
                # Parse fetched_at
                try:
                    if fetched_at:
                        fetched_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                    else:
                        ts_str = filepath.stem.split("_", 1)[1]
                        fetched_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                except (ValueError, IndexError):
                    fetched_dt = None
                
                # Lead time
                lead_hours = None
                if fetched_dt and target_date:
                    settlement_approx = datetime(
                        target_date.year, target_date.month, target_date.day,
                        22, 0, 0, tzinfo=timezone.utc
                    )
                    lead_hours = (settlement_approx - fetched_dt).total_seconds() / 3600.0
                
                records.append({
                    "target_date": target_date,
                    "forecast_high_f": int(temp),
                    "fetched_at": fetched_dt.isoformat() if fetched_dt else None,
                    "lead_time_hours": round(lead_hours, 1) if lead_hours else None,
                    "period_name": period.get("name", ""),
                    "source_file": filepath.name,
                })
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Skipping %s: %s", filepath.name, e)
    
    df = pd.DataFrame(records)
    if not df.empty:
        logger.info("Extracted %d forecast records for %d unique dates",
                    len(df), df["target_date"].nunique())
        
        # Show what dates we have forecasts for
        dates = sorted(df["target_date"].unique())
        logger.info("Forecast target dates: %s", [str(d) for d in dates])
    
    return df


def build_calibration_table(forecasts_df, observations_df):
    """Join forecasts to observations on target_date."""
    import pandas as pd
    
    if forecasts_df.empty or observations_df.empty:
        logger.error("Cannot build calibration table: missing data")
        return pd.DataFrame()
    
    # Debug: show date ranges
    fc_dates = sorted(forecasts_df["target_date"].unique())
    obs_dates = sorted(observations_df["date"].unique())
    
    logger.info("Forecast dates:    %s to %s (%d dates)", fc_dates[0], fc_dates[-1], len(fc_dates))
    logger.info("Observation dates: %s to %s (%d dates)", obs_dates[0], obs_dates[-1], len(obs_dates))
    
    # Find overlap
    fc_set = set(fc_dates)
    obs_set = set(obs_dates)
    overlap = fc_set & obs_set
    logger.info("Overlapping dates: %d", len(overlap))
    
    if not overlap:
        # Show the gap
        logger.warning("NO OVERLAP between forecast and observation dates!")
        logger.info("  Latest observation: %s", obs_dates[-1])
        logger.info("  Earliest forecast target: %s", fc_dates[0])
        logger.info("  Gap: %d days", (fc_dates[0] - obs_dates[-1]).days)
        return pd.DataFrame()
    
    logger.info("Matching dates: %s", sorted(overlap))
    
    # Ensure date types match
    forecasts_df = forecasts_df.copy()
    observations_df = observations_df.copy()
    forecasts_df["target_date"] = pd.to_datetime(forecasts_df["target_date"]).dt.date
    observations_df["date"] = pd.to_datetime(observations_df["date"]).dt.date
    
    # Join
    merged = forecasts_df.merge(
        observations_df[["date", "tmax_f"]].rename(
            columns={"date": "target_date", "tmax_f": "actual_high_f"}
        ),
        on="target_date",
        how="inner"
    )
    
    # Compute error
    merged["error"] = merged["actual_high_f"] - merged["forecast_high_f"]
    merged["abs_error"] = merged["error"].abs()
    
    # Time features
    merged["target_date_dt"] = pd.to_datetime(merged["target_date"])
    merged["month"] = merged["target_date_dt"].dt.month
    merged["day_of_week"] = merged["target_date_dt"].dt.dayofweek
    merged["season"] = merged["month"].map({
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    })
    
    # Lead time bins
    if "lead_time_hours" in merged.columns:
        import pandas as pd
        merged["lead_time_bin"] = pd.cut(
            merged["lead_time_hours"],
            bins=[-999, 6, 12, 24, 48, 168],
            labels=["0-6h", "6-12h", "12-24h", "24-48h", "48h+"]
        )
    
    merged = merged.drop(columns=["target_date_dt"], errors="ignore")
    
    logger.info("Calibration table: %d rows, %d unique dates",
                len(merged), merged["target_date"].nunique())
    
    return merged


def analyze_calibration(cal_df):
    """Compute REAL sigma and bias values."""
    
    if cal_df.empty:
        logger.error("No calibration data to analyze")
        return
    
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    
    # Show raw data first
    print("\nRaw forecast vs actual:")
    print("  %-12s %8s %8s %6s %10s %8s" % ("Date", "Forecast", "Actual", "Error", "Lead(hrs)", "Source"))
    print("  " + "-" * 60)
    
    shown = set()
    for _, row in cal_df.sort_values("target_date").iterrows():
        key = (row["target_date"], row.get("lead_time_hours", ""))
        if key not in shown:
            shown.add(key)
            lead = "%.0f" % row["lead_time_hours"] if row.get("lead_time_hours") else "?"
            print("  %-12s %7d°F %6d°F %+5d°F %9sh" % (
                row["target_date"], row["forecast_high_f"],
                row["actual_high_f"], row["error"], lead))
    
    # Overall stats
    n = len(cal_df)
    print("\nOverall forecast error statistics (n=%d):" % n)
    print("  Mean error (bias): %+.2f°F" % cal_df["error"].mean())
    print("  Std dev (sigma):   %.2f°F" % cal_df["error"].std())
    print("  Mean abs error:    %.2f°F" % cal_df["abs_error"].mean())
    print("  Max abs error:     %d°F" % cal_df["abs_error"].max())
    
    # By lead time
    if "lead_time_bin" in cal_df.columns:
        try:
            import config
            config_map = {
                "0-6h": ("SIGMA_SAMEDAY_PM", config.SIGMA_SAMEDAY_PM),
                "6-12h": ("SIGMA_SAMEDAY_AM", config.SIGMA_SAMEDAY_AM),
                "12-24h": ("SIGMA_1DAY", config.SIGMA_1DAY),
                "24-48h": ("SIGMA_1DAY", config.SIGMA_1DAY),
                "48h+": ("(none)", None),
            }
        except ImportError:
            config_map = {}
        
        print("\n" + "-" * 60)
        print("SIGMA BY LEAD TIME:")
        print("-" * 60)
        print("  %-12s %8s %8s %5s %8s" % ("Lead Time", "Bias", "Sigma", "N", "Config"))
        print("  " + "-" * 46)
        
        for bin_label in ["0-6h", "6-12h", "12-24h", "24-48h", "48h+"]:
            subset = cal_df[cal_df["lead_time_bin"] == bin_label]
            if len(subset) < 1:
                continue
            
            bias = subset["error"].mean()
            sigma = subset["error"].std() if len(subset) > 1 else float("nan")
            
            config_info = config_map.get(bin_label, ("?", None))
            config_str = "%.1f" % config_info[1] if config_info[1] else "---"
            
            print("  %-12s %+7.2f %7.2f %5d %7s" % (
                bin_label, bias, sigma, len(subset), config_str))
    
    # Warning about sample size
    if n < 30:
        print("\n⚠️  IMPORTANT: Only %d data points!" % n)
        print("   These sigma estimates are NOT reliable yet.")
        print("   You need at least 30 forecast-observation pairs for")
        print("   statistically meaningful calibration.")
        print("   Keep running 'py bot.py scan --once' daily to collect more.")
        print("   Re-run this script weekly to update estimates.")
    
    return cal_df


def main():
    import pandas as pd

    print("=" * 60)
    print("BUILD CALIBRATION DATASET (v3 -- GHCN-only)")
    print("=" * 60)
    print("Project root: %s" % PROJECT_ROOT)

    # Step 1: Load forecast snapshots
    print("\n[1] Loading forecast snapshots...")
    forecasts = load_forecast_snapshots()

    if forecasts.empty:
        print("\nNo forecast snapshots found.")
        print("Run: py bot.py scan --once")
        print("This archives NWS forecasts. Collect 7-14 days, then re-run.")
        return

    # Step 2: Load ONLY settled GHCN observations (no forecast gap-fill)
    print("\n[2] Loading settled observations (GHCN-Daily only)...")
    obs_path = find_observation_file()
    observations, latest_settled = load_settled_observations(obs_path)

    if observations.empty or latest_settled is None:
        print("\nNo GHCN observations available.")
        print("Run first: py fetch_ghcn_daily.py")
        return

    # Step 3: Gate forecasts to only settled dates
    print("\n[3] Filtering forecasts to settled dates (<= %s)..." % latest_settled)
    forecasts_settled = forecasts[forecasts["target_date"] <= latest_settled].copy()

    if forecasts_settled.empty:
        fc_dates = sorted(forecasts["target_date"].unique())
        print("\nNo settled observations for forecast target dates yet.")
        print("  Earliest forecast target: %s" % fc_dates[0])
        print("  Latest settled GHCN date: %s" % latest_settled)
        print("  Gap: %d day(s)" % (fc_dates[0] - latest_settled).days)
        print("Re-run tomorrow after the day closes / GHCN updates.")
        return

    print("  %d forecast rows for %d settled target dates (of %d total)" % (
        len(forecasts_settled),
        forecasts_settled["target_date"].nunique(),
        forecasts["target_date"].nunique(),
    ))

    # Step 4: Build calibration table
    print("\n[4] Joining forecasts to observations...")
    cal = build_calibration_table(forecasts_settled, observations)

    if cal.empty:
        print("\nNo matching dates after join.")
        print("Re-run tomorrow after the day closes / GHCN updates.")
        return

    # Save
    output_dir = PROJECT_ROOT / "data" / "curated" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "forecast_vs_observed_KNYC.parquet"
    cal.to_parquet(output_path, index=False)
    logger.info("Saved: %s", output_path)

    # Also save CSV for easy inspection
    csv_path = output_dir / "forecast_vs_observed_KNYC.csv"
    cal.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    # Analyze
    analyze_calibration(cal)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
scripts/fetch_ghcn_daily.py -- Download Historical Temperature Observations

Downloads GHCN-Daily data for Central Park (USW00094728) directly from NCEI.
This is the SAME station that determines KXHIGHNY contract settlement.

No API key needed -- this is a direct CSV download from NOAA's public archive.

The file contains daily TMAX, TMIN, PRCP, SNOW going back decades.
We convert it to a clean Parquet file for fast querying.

Usage:
    cd C:\\Users\\Unitye\\Desktop\\kalshi-weather-bot
    py scripts\\fetch_ghcn_daily.py
    py scripts\\fetch_ghcn_daily.py --station USW00094728 --years 10
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
# If we're in scripts/, go up one level
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Station mapping: GHCN-Daily ID -> Kalshi series
STATIONS = {
    "USW00094728": {
        "name": "NY CITY CENTRAL PARK",
        "kalshi_series": "KXHIGHNY",
        "nws_station": "KNYC",
        "note": "Settlement station for NYC High Temperature markets",
    },
    # Add more as we expand:
    # "USW00094846": {"name": "CHICAGO OHARE INTL AP", "kalshi_series": "KXHIGHCHIM", "nws_station": "KORD"},
    # "USW00023174": {"name": "LOS ANGELES INTL AP", "kalshi_series": "KXHIGHLA", "nws_station": "KLAX"},
    # "USW00013874": {"name": "PHILADELPHIA INTL AP", "kalshi_series": "KXHIGHPHIL", "nws_station": "KPHL"},
}


def download_ghcn_daily(station_id, output_dir):
    """
    Download GHCN-Daily CSV for a station directly from NCEI.
    
    Source: https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/
    Format: CSV with columns like DATE, TMAX, TMIN, PRCP, etc.
    TMAX/TMIN are in tenths of degrees Celsius.
    """
    import requests
    
    url = (
        "https://www.ncei.noaa.gov/data/"
        "global-historical-climatology-network-daily/access/%s.csv" % station_id
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / ("%s_ghcn_raw.csv" % station_id)
    
    logger.info("Downloading GHCN-Daily for %s...", station_id)
    logger.info("  URL: %s", url)
    logger.info("  This may take 30-60 seconds (file is ~5-15 MB)")
    
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        
        # Stream to file with progress
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(raw_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    print("\r  Downloading: %.1f%% (%d KB)" % (pct, downloaded // 1024), end="")
        print()  # newline after progress
        
        size_mb = raw_path.stat().st_size / (1024 * 1024)
        logger.info("  Downloaded: %s (%.1f MB)", raw_path.name, size_mb)
        return raw_path
        
    except requests.RequestException as e:
        logger.error("Download failed: %s", e)
        logger.info("  Fallback: try downloading manually from:")
        logger.info("  %s", url)
        return None


def process_ghcn_to_parquet(raw_csv_path, station_id, years_back=10):
    """
    Convert raw GHCN-Daily CSV to a clean Parquet file.
    
    Raw GHCN format:
    - DATE: YYYY-MM-DD
    - TMAX: tenths of degrees Celsius (e.g., 256 = 25.6°C)
    - TMIN: tenths of degrees Celsius
    - PRCP: tenths of mm
    - SNOW: mm
    - Various quality flags
    
    We convert to:
    - date: date
    - tmax_f: integer (Fahrenheit, matching NWS reporting)
    - tmin_f: integer (Fahrenheit)
    - prcp_in: float (inches)
    - snow_in: float (inches)
    """
    import pandas as pd
    
    logger.info("Processing GHCN-Daily CSV -> Parquet...")
    
    # Read CSV
    df = pd.read_csv(raw_csv_path, low_memory=False)
    logger.info("  Raw records: %d", len(df))
    
    # Filter to recent years
    cutoff = datetime.now() - timedelta(days=years_back * 365)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    df = df[df["DATE"] >= cutoff_str].copy()
    logger.info("  After filtering to last %d years: %d records", years_back, len(df))
    
    # Convert date
    df["date"] = pd.to_datetime(df["DATE"]).dt.date
    
    # Convert TMAX: tenths of °C -> °F (rounded to integer, matching NWS)
    if "TMAX" in df.columns:
        # TMAX is in tenths of degrees C, e.g., 256 = 25.6°C
        df["tmax_c"] = df["TMAX"] / 10.0
        df["tmax_f"] = (df["tmax_c"] * 9.0 / 5.0 + 32.0).round(0).astype("Int64")
        tmax_count = df["tmax_f"].notna().sum()
        logger.info("  TMAX records: %d (%.1f%% coverage)", tmax_count, tmax_count * 100 / len(df))
    else:
        logger.warning("  No TMAX column found!")
        df["tmax_f"] = None
    
    # Convert TMIN
    if "TMIN" in df.columns:
        df["tmin_c"] = df["TMIN"] / 10.0
        df["tmin_f"] = (df["tmin_c"] * 9.0 / 5.0 + 32.0).round(0).astype("Int64")
    else:
        df["tmin_f"] = None
    
    # Convert PRCP: tenths of mm -> inches
    if "PRCP" in df.columns:
        df["prcp_in"] = (df["PRCP"] / 10.0) / 25.4  # mm -> inches
    else:
        df["prcp_in"] = None
    
    # Convert SNOW: mm -> inches
    if "SNOW" in df.columns:
        df["snow_in"] = df["SNOW"] / 25.4
    else:
        df["snow_in"] = None
    
    # Quality flags
    df["tmax_qflag"] = df.get("TMAX_ATTRIBUTES", ",,").apply(
        lambda x: str(x).split(",")[1] if isinstance(x, str) and "," in x else ""
    )
    
    # Select and clean
    result = df[["date", "tmax_f", "tmin_f", "prcp_in", "snow_in", "tmax_qflag"]].copy()
    result["station_id"] = station_id
    
    # Save as Parquet
    output_dir = PROJECT_ROOT / "data" / "raw" / "weather" / "observations"
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / ("%s_daily.parquet" % station_id)
    
    result.to_parquet(parquet_path, index=False)
    size_kb = parquet_path.stat().st_size / 1024
    logger.info("  Saved: %s (%.1f KB)", parquet_path.name, size_kb)
    
    # Summary stats
    logger.info("\n  SUMMARY for %s:", station_id)
    logger.info("  Date range: %s to %s", result["date"].min(), result["date"].max())
    logger.info("  Total days: %d", len(result))
    
    tmax = result["tmax_f"].dropna()
    if len(tmax) > 0:
        logger.info("  TMAX range: %d°F to %d°F", tmax.min(), tmax.max())
        logger.info("  TMAX mean:  %.1f°F", tmax.mean())
        logger.info("  TMAX std:   %.1f°F", tmax.std())
    
    return parquet_path


def verify_with_duckdb(parquet_path):
    """Quick verification that the Parquet file works with DuckDB."""
    import duckdb
    
    logger.info("\nVerifying with DuckDB...")
    con = duckdb.connect()
    
    # Basic stats
    result = con.execute("""
        SELECT 
            count(*) as total_days,
            min(date) as first_date,
            max(date) as last_date,
            avg(tmax_f) as avg_tmax,
            stddev(tmax_f) as std_tmax,
            count(tmax_f) as tmax_count
        FROM '%s'
    """ % parquet_path).fetchone()
    
    logger.info("  Total days:  %d", result[0])
    logger.info("  Date range:  %s to %s", result[1], result[2])
    logger.info("  Avg TMAX:    %.1f°F", result[3])
    logger.info("  Std TMAX:    %.1f°F", result[4])
    logger.info("  TMAX count:  %d", result[5])
    
    # Monthly breakdown (useful for seasonal sigma calibration)
    print("\n  Monthly TMAX distribution:")
    print("  %-10s %8s %8s %8s %8s" % ("Month", "Avg", "Std", "Min", "Max"))
    print("  " + "-" * 46)
    
    rows = con.execute("""
        SELECT 
            strftime(date, '%%m') as month,
            round(avg(tmax_f), 1) as avg_t,
            round(stddev(tmax_f), 1) as std_t,
            min(tmax_f) as min_t,
            max(tmax_f) as max_t
        FROM '%s'
        WHERE tmax_f IS NOT NULL
        GROUP BY month
        ORDER BY month
    """ % parquet_path).fetchall()
    
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for row in rows:
        m = int(row[0])
        print("  %-10s %7.1f %7.1f %7d %7d" % (month_names[m], row[1], row[2], row[3], row[4]))
    
    con.close()
    logger.info("\n  ✅ DuckDB verification complete")


def main():
    parser = argparse.ArgumentParser(description="Download GHCN-Daily observations")
    parser.add_argument(
        "--station",
        default="USW00094728",
        help="GHCN station ID (default: Central Park NYC)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Years of history to keep (default: 10)"
    )
    args = parser.parse_args()
    
    station_id = args.station
    
    if station_id in STATIONS:
        info = STATIONS[station_id]
        logger.info("Station: %s (%s)", info["name"], station_id)
        logger.info("Kalshi series: %s (NWS: %s)", info["kalshi_series"], info["nws_station"])
        logger.info("Note: %s", info["note"])
    else:
        logger.info("Station: %s (not in known mapping)", station_id)
    
    print()
    
    # Step 1: Download
    raw_dir = PROJECT_ROOT / "data" / "raw" / "weather" / "observations"
    raw_path = download_ghcn_daily(station_id, raw_dir)
    
    if raw_path is None:
        logger.error("Download failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Process to Parquet
    parquet_path = process_ghcn_to_parquet(raw_path, station_id, years_back=args.years)
    
    # Step 3: Verify
    verify_with_duckdb(parquet_path)
    
    print("\n" + "=" * 60)
    print("DONE! Historical observations saved to:")
    print("  %s" % parquet_path)
    print("\nThis is your calibration ground truth for KXHIGHNY.")
    print("Next step: py scripts\\fetch_kalshi_trades.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

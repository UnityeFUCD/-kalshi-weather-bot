"""
fetch_historical_ensemble.py -- Fetch historical ensemble σ for all backtest dates.

Strategy:
1. Get 6-model deterministic forecasts from Open-Meteo previous-runs API (93 days)
2. Get real ensemble σ from ensemble API for overlap dates (last ~5 days)
3. Calibrate: scale multi-model stdev to match real ensemble σ
4. Save per-day σ_ens to parquet for backtest use
"""

import sys
import json
import time
import logging
import statistics
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import requests
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from backtest import load_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NYC_LAT = 40.7831
NYC_LON = -73.9712

MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless,jma_seamless,meteofrance_seamless"
ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs_seamless"]

OUTPUT_PATH = PROJECT_ROOT / "data" / "curated" / "ensemble_history.parquet"


def fetch_multimodel_forecasts(past_days=92):
    """Get deterministic Tmax from 6 models for past N days."""
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "past_days": past_days,
        "forecast_days": 1,
        "models": MODELS,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    times = daily.get("time", [])
    model_keys = [k for k in daily if k != "time"]

    rows = []
    for i, date_str in enumerate(times):
        vals = []
        for k in model_keys:
            v = daily[k][i]
            if v is not None:
                vals.append(float(v))
        if len(vals) >= 3:
            rows.append({
                "date": date_str,
                "multimodel_stdev": statistics.stdev(vals),
                "multimodel_mean": statistics.mean(vals),
                "n_models": len(vals),
            })
    return rows


def fetch_real_ensemble_sigma():
    """Get real ensemble σ from the ensemble API (recent dates only)."""
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    all_members = defaultdict(list)

    for model in ENSEMBLE_MODELS:
        params = {
            "latitude": NYC_LAT,
            "longitude": NYC_LON,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "past_days": 7,
            "forecast_days": 1,
            "models": model,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            times = daily.get("time", [])
            member_keys = [k for k in daily if "temperature_2m_max" in k and k != "time"]

            for i, date_str in enumerate(times):
                for k in member_keys:
                    v = daily[k][i] if i < len(daily[k]) else None
                    if v is not None:
                        all_members[date_str].append(float(v))
        except Exception as e:
            logger.warning("Ensemble fetch failed for %s: %s", model, e)

    rows = []
    for date_str in sorted(all_members.keys()):
        vals = all_members[date_str]
        if len(vals) >= 10:
            rows.append({
                "date": date_str,
                "real_sigma_ens": statistics.stdev(vals),
                "real_mean": statistics.mean(vals),
                "n_members": len(vals),
            })
    return rows


def main():
    print("=" * 70)
    print("FETCH HISTORICAL ENSEMBLE σ FOR BACKTEST")
    print("=" * 70)
    print()

    # Step 1: Multi-model historical forecasts
    print("[1] Fetching 6-model deterministic forecasts (93 days)...")
    multi_rows = fetch_multimodel_forecasts(92)
    print("  Got %d dates" % len(multi_rows))

    # Step 2: Real ensemble σ (recent)
    print("[2] Fetching real ensemble σ (recent dates)...")
    real_rows = fetch_real_ensemble_sigma()
    print("  Got %d dates with real ensemble data" % len(real_rows))

    # Step 3: Calibrate - find scaling factor
    multi_df = pd.DataFrame(multi_rows)
    multi_df["date"] = pd.to_datetime(multi_df["date"]).dt.date

    real_df = pd.DataFrame(real_rows)
    real_df["date"] = pd.to_datetime(real_df["date"]).dt.date

    overlap = pd.merge(multi_df, real_df, on="date", how="inner")
    print()
    print("[3] Calibration overlap: %d dates" % len(overlap))

    if len(overlap) > 0:
        print()
        print("  Overlap data:")
        print("  %-12s  multi_stdev  real_σ_ens  ratio" % "Date")
        print("  " + "-" * 50)
        ratios = []
        for _, row in overlap.iterrows():
            if row["multimodel_stdev"] > 0:
                ratio = row["real_sigma_ens"] / row["multimodel_stdev"]
                ratios.append(ratio)
                print("  %-12s  %.3f        %.3f      %.2f" % (
                    row["date"], row["multimodel_stdev"],
                    row["real_sigma_ens"], ratio))

        if ratios:
            scale_factor = statistics.median(ratios)
            print()
            print("  Scale factor (median ratio): %.2f" % scale_factor)
            print("  (real ensemble σ ≈ %.2f × multi-model stdev)" % scale_factor)
        else:
            scale_factor = 1.0
            print("  WARNING: No valid ratios, using scale_factor=1.0")
    else:
        # Fallback: based on observation that 6-model stdev ≈ 0.5-1.5
        # and real ensemble σ ≈ 0.7-2.0, typical scale ≈ 1.0-1.5
        scale_factor = 1.2
        print("  No overlap dates — using default scale_factor=%.1f" % scale_factor)

    # Step 4: Compute calibrated σ_ens for all dates
    print()
    print("[4] Computing calibrated σ_ens for all %d dates..." % len(multi_df))

    multi_df["sigma_ens"] = multi_df["multimodel_stdev"] * scale_factor

    # For dates where we have real data, use real
    real_lookup = {row["date"]: row["real_sigma_ens"] for _, row in real_df.iterrows()}
    for i, row in multi_df.iterrows():
        if row["date"] in real_lookup:
            multi_df.at[i, "sigma_ens"] = real_lookup[row["date"]]

    # Compute V2 σ for each day
    multi_df["sigma_v2"] = multi_df["sigma_ens"].apply(
        lambda x: max(1.2, config.ENSEMBLE_ALPHA * x))

    # Stats
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("σ_ens distribution (calibrated):")
    print("  Min:    %.3f" % multi_df["sigma_ens"].min())
    print("  25th:   %.3f" % multi_df["sigma_ens"].quantile(0.25))
    print("  Median: %.3f" % multi_df["sigma_ens"].quantile(0.50))
    print("  75th:   %.3f" % multi_df["sigma_ens"].quantile(0.75))
    print("  Max:    %.3f" % multi_df["sigma_ens"].max())
    print("  Mean:   %.3f" % multi_df["sigma_ens"].mean())
    print()
    print("V2 composed σ distribution:")
    print("  Min:    %.3f" % multi_df["sigma_v2"].min())
    print("  25th:   %.3f" % multi_df["sigma_v2"].quantile(0.25))
    print("  Median: %.3f" % multi_df["sigma_v2"].quantile(0.50))
    print("  75th:   %.3f" % multi_df["sigma_v2"].quantile(0.75))
    print("  Max:    %.3f" % multi_df["sigma_v2"].max())
    print()

    threshold = 1.2 / config.ENSEMBLE_ALPHA
    n_widen = (multi_df["sigma_ens"] > threshold).sum()
    n_same = (multi_df["sigma_ens"] <= threshold).sum()
    print("Days where V2 σ > 1.2 (widens): %d / %d (%.0f%%)" % (
        n_widen, len(multi_df), n_widen / len(multi_df) * 100))
    print("Days where V2 σ = 1.2 (same as Old): %d / %d (%.0f%%)" % (
        n_same, len(multi_df), n_same / len(multi_df) * 100))

    # Save
    out_df = multi_df[["date", "sigma_ens", "multimodel_stdev", "multimodel_mean",
                        "n_models", "sigma_v2"]].copy()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print()
    print("Saved: %s (%d rows)" % (OUTPUT_PATH, len(out_df)))

    # Show sample
    print()
    print("Sample (first 15 days):")
    print("  %-12s  σ_ens   σ_v2   multi_stdev  models" % "Date")
    print("  " + "-" * 55)
    for _, row in out_df.head(15).iterrows():
        print("  %-12s  %.3f   %.3f   %.3f        %d" % (
            row["date"], row["sigma_ens"], row["sigma_v2"],
            row["multimodel_stdev"], row["n_models"]))


if __name__ == "__main__":
    main()

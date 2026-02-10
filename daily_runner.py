"""
daily_runner.py -- One-command daily runner for Kalshi Weather Bot calibration.

Runs all daily tasks in sequence:
  1. Scan markets (py bot.py scan --once) -- collects forecast snapshots
  2. Refresh GHCN observations cache (download latest USW00094728)
  3. Run calibration builder (only produces output when settled overlap exists)

Idempotent: safe to rerun multiple times per day without duplicating data.
Tracks state in data/runner_state.json.

Usage:
    py daily_runner.py              # Full daily run
    py daily_runner.py --skip-scan  # Skip bot scan (just refresh obs + calibrate)
    py daily_runner.py --dry-run    # Show what would happen, don't execute

After 14 days of data, generates a summary report.
"""

import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta, date

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "daily_runner.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("daily_runner")

STATE_FILE = PROJECT_ROOT / "data" / "runner_state.json"
CALIBRATION_DIR = PROJECT_ROOT / "data" / "curated" / "calibration"
REPORT_PATH = CALIBRATION_DIR / "calibration_summary_report.txt"
GHCN_PARQUET = PROJECT_ROOT / "data" / "raw" / "weather" / "observations" / "USW00094728_daily.parquet"


def load_state():
    """Load or initialize runner state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "calibration_test_start_date": None,
        "completed_dates": [],
        "total_executions": 0,
        "last_run_timestamp": None,
        "last_run_date": None,
        "calibration_rows_by_lead_time": {},
        "settled_dates_count": 0,
        "daily_run_log": [],
    }


def save_state(state):
    """Save runner state atomically."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_FILE)


def run_bot_scan():
    """Run py bot.py scan --once to collect forecast snapshot."""
    logger.info("[1/3] Running bot scan...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "bot.py"), "scan", "--once"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.warning("Bot scan exited with code %d", result.returncode)
        if result.stderr:
            logger.warning("  stderr: %s", result.stderr[:500])
    else:
        logger.info("  Bot scan completed successfully")
    return result.returncode == 0


def refresh_ghcn():
    """Download/update GHCN-Daily observations for USW00094728."""
    logger.info("[2/3] Refreshing GHCN observations...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "fetch_ghcn_daily.py"),
         "--station", "USW00094728", "--years", "10"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        logger.warning("GHCN refresh exited with code %d", result.returncode)
        if result.stderr:
            logger.warning("  stderr: %s", result.stderr[:500])
        return False

    # Report latest date
    if GHCN_PARQUET.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(GHCN_PARQUET)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            latest = df["date"].max()
            logger.info("  GHCN updated. Latest settled date: %s", latest)
        except Exception as e:
            logger.warning("  Could not read parquet to check latest date: %s", e)
    else:
        logger.warning("  GHCN parquet not found after refresh")

    return result.returncode == 0


def run_calibration():
    """Run calibration builder. Returns (success, rows_count, details)."""
    logger.info("[3/3] Running calibration builder...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "build_calibration_dataset.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr
    rows = 0
    settled_count = 0

    # Check for the clean "no overlap" exit
    if "No settled observations for forecast target dates yet" in output:
        logger.info("  No settled overlap yet -- this is expected early on")
        return True, 0, {}

    if "No forecast snapshots found" in output:
        logger.info("  No forecast snapshots found -- run bot scan first")
        return True, 0, {}

    # Try to parse calibration output
    cal_parquet = CALIBRATION_DIR / "forecast_vs_observed_KNYC.parquet"
    if cal_parquet.exists():
        try:
            import pandas as pd
            cal = pd.read_parquet(cal_parquet)
            rows = len(cal)
            settled_count = cal["target_date"].nunique()

            # Count rows by lead time
            details = {}
            if "lead_time_bin" in cal.columns:
                for bin_label, group in cal.groupby("lead_time_bin", observed=True):
                    details[str(bin_label)] = len(group)

            logger.info("  Calibration: %d rows, %d settled dates", rows, settled_count)
            return True, rows, details
        except Exception as e:
            logger.warning("  Could not read calibration output: %s", e)

    return result.returncode == 0, rows, {}


def generate_14day_report():
    """Generate summary report after 14+ days of data."""
    cal_parquet = CALIBRATION_DIR / "forecast_vs_observed_KNYC.parquet"
    if not cal_parquet.exists():
        return False

    try:
        import pandas as pd
        import math
        cal = pd.read_parquet(cal_parquet)
    except Exception:
        return False

    n_dates = cal["target_date"].nunique() if "target_date" in cal.columns else 0
    if n_dates < 14:
        return False

    lines = []
    lines.append("=" * 60)
    lines.append("CALIBRATION SUMMARY REPORT (14-Day Window)")
    lines.append("Generated: %s UTC" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    lines.append("=" * 60)

    lines.append("")
    lines.append("OVERVIEW")
    lines.append("  Total calibration rows: %d" % len(cal))
    lines.append("  Unique settled dates:   %d" % n_dates)
    date_range = sorted(cal["target_date"].unique())
    lines.append("  Date range: %s to %s" % (date_range[0], date_range[-1]))

    lines.append("")
    lines.append("ERROR DISTRIBUTION")
    if "error" in cal.columns:
        lines.append("  Mean error (bias): %+.2fF" % cal["error"].mean())
        lines.append("  Std dev (sigma):   %.2fF" % cal["error"].std())
        lines.append("  Mean abs error:    %.2fF" % cal["abs_error"].mean())
        lines.append("  Median abs error:  %.2fF" % cal["abs_error"].median())
        lines.append("  Max abs error:     %dF" % cal["abs_error"].max())

    # Rows per lead-time bucket
    if "lead_time_bin" in cal.columns:
        lines.append("")
        lines.append("ROWS PER LEAD-TIME BUCKET")
        for bin_label in ["0-6h", "6-12h", "12-24h", "24-48h", "48h+"]:
            subset = cal[cal["lead_time_bin"] == bin_label]
            if len(subset) > 0:
                lines.append("  %-10s %d rows" % (bin_label, len(subset)))

        lines.append("")
        lines.append("SIGMA BY LEAD TIME")
        lines.append("  %-10s %8s %8s %5s" % ("Lead Time", "Bias", "Sigma", "N"))
        lines.append("  " + "-" * 35)
        for bin_label in ["0-6h", "6-12h", "12-24h", "24-48h", "48h+"]:
            subset = cal[cal["lead_time_bin"] == bin_label]
            if len(subset) >= 5:
                bias = subset["error"].mean()
                sigma = subset["error"].std()
                lines.append("  %-10s %+7.2f %7.2f %5d" % (bin_label, bias, sigma, len(subset)))
            elif len(subset) > 0:
                lines.append("  %-10s  insufficient data (n=%d, need 5+)" % (bin_label, len(subset)))

    # Reliability table (simple: what % of actuals fell within 1-sigma of forecast)
    if "error" in cal.columns and len(cal) >= 5:
        sigma = cal["error"].std()
        within_1s = (cal["abs_error"] <= sigma).sum()
        within_2s = (cal["abs_error"] <= 2 * sigma).sum()
        lines.append("")
        lines.append("RELIABILITY CHECK (sigma=%.2fF)" % sigma)
        lines.append("  Within 1-sigma: %d/%d (%.0f%%) -- expect ~68%%" % (
            within_1s, len(cal), within_1s * 100.0 / len(cal)))
        lines.append("  Within 2-sigma: %d/%d (%.0f%%) -- expect ~95%%" % (
            within_2s, len(cal), within_2s * 100.0 / len(cal)))

    # Brier score if we have enough data
    if len(cal) >= 10 and "model_prob" in cal.columns and "actual_high_f" in cal.columns:
        lines.append("")
        lines.append("  (Brier score requires bucket-level outcome data -- not yet available)")

    if n_dates < 30:
        lines.append("")
        lines.append("NOTE: %d settled dates is useful but still limited." % n_dates)
        lines.append("30+ dates recommended for reliable sigma estimates.")
        lines.append("Keep collecting daily snapshots.")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("14-day summary report written to %s", REPORT_PATH)
    print(report)
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily runner for Kalshi Weather Bot calibration")
    parser.add_argument("--skip-scan", action="store_true", help="Skip bot scan (just refresh obs + calibrate)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without executing")
    args = parser.parse_args()

    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KALSHI WEATHER BOT -- DAILY RUNNER")
    print("Time: %s UTC" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    state = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Initialize start date on first run
    if state["calibration_test_start_date"] is None:
        state["calibration_test_start_date"] = today
        logger.info("First run -- calibration test window starts today: %s", today)

    # Migrate old state format if needed
    if "completed_dates" not in state:
        state["completed_dates"] = []
    if "total_executions" not in state:
        state["total_executions"] = state.get("runs_completed", 0)

    days_completed = len(set(state["completed_dates"]))

    # Idempotency: check if already ran today
    if state["last_run_date"] == today:
        logger.info("Already ran today (%s). Re-running is safe (idempotent).", today)

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        print("  1. py bot.py scan --once")
        print("  2. py fetch_ghcn_daily.py --station USW00094728 --years 10")
        print("  3. py build_calibration_dataset.py")
        print("\nState: %d unique days completed, %d total executions (since %s)" % (
            days_completed, state["total_executions"], state["calibration_test_start_date"]))
        return

    # Step 1: Bot scan
    scan_ok = True
    if not args.skip_scan:
        scan_ok = run_bot_scan()
    else:
        logger.info("[1/3] Skipping bot scan (--skip-scan)")

    # Step 2: Refresh GHCN
    ghcn_ok = refresh_ghcn()

    # Step 3: Calibration
    cal_ok, cal_rows, lead_time_details = run_calibration()

    # Step 4: Check for 14-day report
    generated_report = generate_14day_report()

    # Update state
    state["total_executions"] += 1
    state["last_run_timestamp"] = datetime.now(timezone.utc).isoformat()
    state["last_run_date"] = today
    state["calibration_rows_by_lead_time"] = lead_time_details
    if cal_rows > 0:
        state["settled_dates_count"] = cal_rows

    # Track unique calendar days (deduplicated)
    if today not in state["completed_dates"]:
        state["completed_dates"].append(today)
    days_completed = len(set(state["completed_dates"]))

    # Append to daily run log (keep last 30 entries)
    state["daily_run_log"].append({
        "date": today,
        "scan_ok": scan_ok,
        "ghcn_ok": ghcn_ok,
        "cal_ok": cal_ok,
        "cal_rows": cal_rows,
    })
    state["daily_run_log"] = state["daily_run_log"][-30:]

    save_state(state)

    # Summary
    print("\n" + "=" * 60)
    print("DAILY RUN COMPLETE")
    print("  Scan:        %s" % ("OK" if scan_ok else "FAILED"))
    print("  GHCN update: %s" % ("OK" if ghcn_ok else "FAILED"))
    print("  Calibration: %s (%d rows)" % ("OK" if cal_ok else "FAILED", cal_rows))
    print("  Days completed: %d / 14  (executions: %d, since %s)" % (
        days_completed, state["total_executions"], state["calibration_test_start_date"]))
    if generated_report:
        print("  14-day report: GENERATED -> %s" % REPORT_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()

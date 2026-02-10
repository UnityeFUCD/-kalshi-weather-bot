"""
cli_scraper.py -- NWS Daily Climate Report Scraper (Phase 4.5).

Parses the NWS Daily Climate Report (CLI product) to extract official
settlement max temperature.  This is the GROUND TRUTH for Kalshi NHIGH
settlement.

The CLI report is published by NWS offices and contains:
  - Maximum temperature
  - Minimum temperature
  - Precipitation
  - etc.

Also maintains a simple JSON settlement database for historical lookups.

Usage:
    py cli_scraper.py                    # Fetch today's CLI report
    py cli_scraper.py --date 2026-02-09  # Fetch specific date
    py cli_scraper.py --list             # Show settlement DB
"""

import re
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

import requests

import config

logger = logging.getLogger(__name__)

# NWS CLI product URLs by office/city
CLI_URLS = {
    "KXHIGHNY": {
        "url": "https://forecast.weather.gov/product.php?site=OKX&issuedby=NYC&product=CLI",
        "office": "OKX",
        "city": "NYC",
    },
    "KXHIGHCHI": {
        "url": "https://forecast.weather.gov/product.php?site=LOT&issuedby=ORD&product=CLI",
        "office": "LOT",
        "city": "ORD",
    },
}


class SettlementDB:
    """Simple JSON-based settlement database."""

    def __init__(self, path=None):
        self.path = path or config.SETTLEMENT_DB_PATH
        self._data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"settlements": {}, "updated_at": None}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self.path)

    def record(self, series_ticker, target_date, tmax, source="cli"):
        """Record a settlement value."""
        key = "%s_%s" % (series_ticker, target_date)
        self._data["settlements"][key] = {
            "series_ticker": series_ticker,
            "date": str(target_date),
            "tmax_f": tmax,
            "source": source,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        logger.info("Settlement recorded: %s %s = %dF (%s)", series_ticker, target_date, tmax, source)

    def get(self, series_ticker, target_date):
        """Look up a settlement value. Returns tmax_f or None."""
        key = "%s_%s" % (series_ticker, target_date)
        entry = self._data["settlements"].get(key)
        if entry:
            return entry["tmax_f"]
        return None

    def list_all(self):
        """Return all settlements sorted by date."""
        items = list(self._data["settlements"].values())
        items.sort(key=lambda x: x["date"])
        return items


def fetch_cli_report(series_ticker="KXHIGHNY"):
    """
    Fetch the latest NWS CLI (Daily Climate Report) for a market.
    Returns raw HTML/text of the report.
    """
    cli_config = CLI_URLS.get(series_ticker)
    if not cli_config:
        logger.error("No CLI URL configured for %s", series_ticker)
        return None

    url = cli_config["url"]
    try:
        resp = requests.get(url, timeout=20, headers={
            "User-Agent": config.NWS_USER_AGENT,
        })
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error("CLI fetch failed for %s: %s", series_ticker, e)
        return None


def parse_cli_report(html_text):
    """
    Parse the NWS CLI report to extract max temperature and report date.

    The CLI report format varies slightly but generally contains lines like:
        MAXIMUM TEMPERATURE (F)     36
    or:
        ...THE MAXIMUM TEMPERATURE...36...

    Returns (report_date: date, tmax: int) or (None, None).
    """
    if not html_text:
        return None, None

    # Extract text content (strip HTML tags)
    text = re.sub(r'<[^>]+>', '', html_text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')

    # Find report date
    # Patterns like "CLIMATE REPORT FOR FEBRUARY 09 2026" or "...FOR 02/09/2026"
    report_date = None

    # Pattern 1: "FOR MONTH DAY YEAR"
    m = re.search(r'FOR\s+(\w+)\s+(\d{1,2})\s+(\d{4})', text, re.IGNORECASE)
    if m:
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        try:
            report_date = datetime.strptime(
                "%s %s %s" % (month_str, day_str, year_str), "%B %d %Y").date()
        except ValueError:
            pass

    # Pattern 2: "FOR MM/DD/YYYY"
    if report_date is None:
        m = re.search(r'FOR\s+(\d{1,2})/(\d{1,2})/(\d{4})', text)
        if m:
            try:
                report_date = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
            except ValueError:
                pass

    # Pattern 3: Date in the product header
    if report_date is None:
        m = re.search(r'(\d{3,4})\s+(AM|PM)\s+\w+\s+\w+\s+(\w+)\s+(\d{1,2})\s+(\d{4})',
                      text, re.IGNORECASE)
        if m:
            month_str, day_str, year_str = m.group(3), m.group(4), m.group(5)
            try:
                report_date = datetime.strptime(
                    "%s %s %s" % (month_str, day_str, year_str), "%b %d %Y").date()
            except ValueError:
                pass

    # Find max temperature
    tmax = None

    # Pattern 1: "MAXIMUM TEMPERATURE (F)   36" or "MAXIMUM TEMPERATURE...36"
    m = re.search(r'MAXIMUM\s+TEMPERATURE.*?(\d{1,3})', text, re.IGNORECASE)
    if m:
        tmax = int(m.group(1))

    # Pattern 2: "MAX TEMP...36" or "HIGH...36"
    if tmax is None:
        m = re.search(r'(?:MAX\s+TEMP|HIGHEST).*?(\d{1,3})', text, re.IGNORECASE)
        if m:
            tmax = int(m.group(1))

    # Sanity check: temperature should be in reasonable range
    if tmax is not None and (tmax < -50 or tmax > 150):
        logger.warning("Parsed tmax=%d seems unreasonable, discarding", tmax)
        tmax = None

    return report_date, tmax


def scrape_and_record(series_ticker="KXHIGHNY", db=None):
    """
    Fetch CLI report, parse it, and record settlement.
    Returns (date, tmax) or (None, None).
    """
    if db is None:
        db = SettlementDB()

    html = fetch_cli_report(series_ticker)
    if html is None:
        return None, None

    report_date, tmax = parse_cli_report(html)

    if report_date is None or tmax is None:
        logger.warning("Could not parse CLI report for %s", series_ticker)
        return None, None

    # Record in DB
    db.record(series_ticker, report_date, tmax, source="cli")

    return report_date, tmax


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NWS CLI Scraper for settlement data")
    parser.add_argument("--market", default="KXHIGHNY", help="Market ticker")
    parser.add_argument("--date", type=str, default=None, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--list", action="store_true", help="List all settlements")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db = SettlementDB()

    if args.list:
        print("=" * 60)
        print("SETTLEMENT DATABASE")
        print("=" * 60)
        settlements = db.list_all()
        if not settlements:
            print("  (empty)")
        else:
            print("  %-12s %-12s %6s %8s" % ("Market", "Date", "Tmax", "Source"))
            print("  " + "-" * 44)
            for s in settlements:
                print("  %-12s %-12s %5dF %8s" % (
                    s["series_ticker"], s["date"], s["tmax_f"], s["source"]))
        return

    print("=" * 60)
    print("NWS CLI SCRAPER")
    print("Market: %s" % args.market)
    print("=" * 60)

    report_date, tmax = scrape_and_record(args.market, db)

    if report_date and tmax is not None:
        print("\nReport date: %s" % report_date)
        print("Max temp:    %dF" % tmax)

        # Check if requested date matches
        if args.date:
            req_date = date.fromisoformat(args.date)
            if req_date != report_date:
                print("\nWARNING: Requested %s but report is for %s" % (req_date, report_date))
                print("CLI report may not be available yet for requested date.")
    else:
        print("\nCould not extract settlement data.")
        print("CLI report may not be published yet.")


if __name__ == "__main__":
    main()

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

    def get_entry(self, series_ticker, target_date):
        """Look up a full settlement entry dict or None."""
        key = "%s_%s" % (series_ticker, target_date)
        return self._data["settlements"].get(key)

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


# --- Unified Settlement Lookup + Reconciliation -----------------------------

def _coerce_date(value):
    """Convert string/date-like input to datetime.date."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _series_from_station(station: str) -> str:
    """Map station code to series ticker."""
    station = (station or "").upper()
    if station in ("KNYC", "USW00094728"):
        return "KXHIGHNY"
    if station in ("KORD", "USW00094846"):
        return "KXHIGHCHI"
    return config.SERIES_TICKER


def _ghcn_tmax_for_series(target_date, series_ticker):
    """Fallback lookup from local GHCN parquet."""
    try:
        from market_registry import get_market
        mc = get_market(series_ticker)
        if mc is None:
            return None
        ghcn_path = mc.ghcn_parquet_path
        if not ghcn_path.exists():
            return None

        import pandas as pd
        df = pd.read_parquet(ghcn_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        row = df[df["date"] == target_date]
        if row.empty:
            return None
        tmax = row.iloc[0]["tmax_f"]
        if pd.isna(tmax):
            return None
        return int(tmax)
    except Exception as e:
        logger.warning("GHCN fallback failed for %s %s: %s", series_ticker, target_date, e)
        return None


def get_settlement_value(target_date, station="KNYC", series_ticker=None, db=None):
    """
    Get settlement Tmax with source priority:
    1) Local settlement DB
    2) NWS CLI scrape
    3) GHCN fallback
    """
    target_date = _coerce_date(target_date)
    resolved_series = series_ticker or _series_from_station(station)
    if db is None:
        db = SettlementDB()

    # 1) Local DB
    entry = db.get_entry(resolved_series, target_date)
    if entry:
        return {
            "date": str(target_date),
            "tmax_f": int(entry["tmax_f"]),
            "source": entry.get("source", "settlement_db"),
            "report_issued_utc": entry.get("recorded_at"),
            "confidence": "official",
        }

    # 2) CLI scrape (official settlement source)
    report_date, tmax = scrape_and_record(series_ticker=resolved_series, db=db)
    if report_date == target_date and tmax is not None:
        return {
            "date": str(target_date),
            "tmax_f": int(tmax),
            "source": "cli_report",
            "report_issued_utc": datetime.now(timezone.utc).isoformat(),
            "confidence": "official",
        }

    # 3) GHCN fallback (provisional)
    ghcn_tmax = _ghcn_tmax_for_series(target_date, resolved_series)
    if ghcn_tmax is not None:
        return {
            "date": str(target_date),
            "tmax_f": int(ghcn_tmax),
            "source": "ghcn_fallback",
            "report_issued_utc": None,
            "confidence": "provisional",
        }

    return None


def _read_jsonl(path: Path):
    """Read JSONL file into list of dict rows."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows):
    """Rewrite JSONL file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)


def _parse_bucket_from_ticker_for_settlement(ticker):
    """Parse B-type bucket from ticker. Returns (kind, low, high) or None.

    We only score B-type buckets (midpoint encoded) because T-type direction
    cannot be resolved from ticker alone without title context.
    """
    if not ticker:
        return None
    parts = str(ticker).split("-")
    if len(parts) < 3:
        return None
    bucket_code = parts[-1]

    m = re.match(r"B(\d+\.?\d*)", bucket_code)
    if m:
        midpoint = float(m.group(1))
        low = int(midpoint - 0.5)
        high = int(midpoint + 0.5)
        return ("between", low, high)
    return None


def _signal_correct(best_edge_obj, actual_tmax):
    """Evaluate correctness of a logged best-edge signal."""
    if not best_edge_obj or actual_tmax is None:
        return None
    ticker = best_edge_obj.get("bucket")
    side = best_edge_obj.get("side")
    parsed = _parse_bucket_from_ticker_for_settlement(ticker)
    if parsed is None:
        return None

    _, low, high = parsed
    bucket_yes = low <= actual_tmax <= high
    if side == "buy_yes":
        return bool(bucket_yes)
    if side == "buy_no":
        return bool(not bucket_yes)
    return None


def _best_edge_from_model_blob(model_blob):
    """Extract side-aware best-edge signal from model comparison blobs."""
    if not isinstance(model_blob, dict):
        return None

    direct = model_blob.get("best_edge")
    if isinstance(direct, dict) and direct.get("bucket") and direct.get("side"):
        return {
            "bucket": direct.get("bucket"),
            "side": direct.get("side"),
            "edge": direct.get("edge"),
        }

    details = model_blob.get("signal_details") or []
    if details:
        best = max(details, key=lambda row: float(row.get("edge", 0.0)))
        bucket = best.get("bucket") or best.get("ticker")
        side = best.get("side")
        if bucket and side:
            return {
                "bucket": bucket,
                "side": side,
                "edge": best.get("edge"),
            }

    return None


def _update_nbm_shadow_records(db):
    """Fill settled outcomes for reports/nbm_shadow.jsonl."""
    path = config.NBM_SHADOW_PATH
    rows = _read_jsonl(path)
    if not rows:
        return {"total": 0, "updated": 0, "settled_rows": 0}

    updated = 0
    settled_rows = 0
    for row in rows:
        series = row.get("series_ticker", config.SERIES_TICKER)
        target = row.get("target_date")
        if not target:
            continue

        settlement = get_settlement_value(target, series_ticker=series, db=db)
        if settlement is None:
            continue
        settled_rows += 1
        actual = settlement["tmax_f"]

        changed = False
        if row.get("actual_tmax") is None:
            row["actual_tmax"] = actual
            row["actual_source"] = settlement["source"]
            row["actual_confidence"] = settlement["confidence"]
            changed = True

        v2_correct = _signal_correct(row.get("v2_best_edge"), actual)
        nbm_correct = _signal_correct(row.get("nbm_best_edge"), actual)

        if row.get("v2_signal_correct") is None and v2_correct is not None:
            row["v2_signal_correct"] = v2_correct
            changed = True
        if row.get("nbm_signal_correct") is None and nbm_correct is not None:
            row["nbm_signal_correct"] = nbm_correct
            changed = True

        if changed:
            updated += 1

    if updated > 0:
        _write_jsonl(path, rows)

    return {"total": len(rows), "updated": updated, "settled_rows": settled_rows}


def _update_model_comparison_records(db):
    """Fill actual_tmax + signal correctness in model comparison log."""
    path = config.PROJECT_ROOT / "reports" / "model_comparison.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return {"total": 0, "updated": 0, "settled_rows": 0}

    updated = 0
    settled_rows = 0
    for row in rows:
        target = row.get("target_date")
        if not target:
            continue
        settlement = get_settlement_value(target, series_ticker=config.SERIES_TICKER, db=db)
        if settlement is None:
            continue
        settled_rows += 1

        changed = False
        if row.get("actual_tmax") is None:
            row["actual_tmax"] = settlement["tmax_f"]
            row["actual_source"] = settlement["source"]
            row["actual_confidence"] = settlement["confidence"]
            changed = True

        actual = row.get("actual_tmax")
        if actual is not None:
            old_best = _best_edge_from_model_blob(row.get("old_model"))
            new_best = _best_edge_from_model_blob(row.get("new_model"))
            old_hit = _signal_correct(old_best, int(actual)) if old_best else None
            new_hit = _signal_correct(new_best, int(actual)) if new_best else None

            if row.get("old_signal_correct") is None and old_hit is not None:
                row["old_signal_correct"] = old_hit
                changed = True
            if row.get("new_signal_correct") is None and new_hit is not None:
                row["new_signal_correct"] = new_hit
                changed = True

            if old_best and not row.get("old_best_edge"):
                row["old_best_edge"] = old_best
                changed = True
            if new_best and not row.get("new_best_edge"):
                row["new_best_edge"] = new_best
                changed = True

        if changed:
            updated += 1

    if updated > 0:
        _write_jsonl(path, rows)
    return {"total": len(rows), "updated": updated, "settled_rows": settled_rows}


def _write_nbm_vs_v2_summary():
    """Generate summary file comparing V2 vs NBM on settled shadow rows."""
    rows = _read_jsonl(config.NBM_SHADOW_PATH)
    settled = [r for r in rows if r.get("actual_tmax") is not None]
    if not settled:
        return None

    v2_err = []
    nbm_err = []
    v2_in_1s = 0
    nbm_in_1s = 0
    v2_cov_n = 0
    nbm_cov_n = 0
    disagree_rows = []
    prob_sums = []

    for row in settled:
        actual = float(row["actual_tmax"])
        v2_mu = row.get("v2_mu")
        v2_sigma = row.get("v2_sigma")
        nbm_mu = row.get("nbm_mu")
        nbm_sigma = row.get("nbm_sigma")

        if v2_mu is not None:
            v2_err.append(abs(float(v2_mu) - actual))
        if nbm_mu is not None:
            nbm_err.append(abs(float(nbm_mu) - actual))

        if v2_mu is not None and v2_sigma is not None and float(v2_sigma) > 0:
            v2_cov_n += 1
            if abs(float(v2_mu) - actual) <= float(v2_sigma):
                v2_in_1s += 1
        if nbm_mu is not None and nbm_sigma is not None and float(nbm_sigma) > 0:
            nbm_cov_n += 1
            if abs(float(nbm_mu) - actual) <= float(nbm_sigma):
                nbm_in_1s += 1

        nbm_probs = row.get("nbm_bucket_probs") or {}
        if nbm_probs:
            prob_sums.append(sum(float(v) for v in nbm_probs.values()))

        v2_be = row.get("v2_best_edge") or {}
        nbm_be = row.get("nbm_best_edge") or {}
        if v2_be and nbm_be and (v2_be.get("bucket") != nbm_be.get("bucket") or v2_be.get("side") != nbm_be.get("side")):
            disagree_rows.append(row)

    v2_mae = (sum(v2_err) / len(v2_err)) if v2_err else None
    nbm_mae = (sum(nbm_err) / len(nbm_err)) if nbm_err else None
    v2_cov = (v2_in_1s / v2_cov_n) if v2_cov_n else None
    nbm_cov = (nbm_in_1s / nbm_cov_n) if nbm_cov_n else None

    v2_disagree_wins = 0
    nbm_disagree_wins = 0
    counted = 0
    for row in disagree_rows:
        v2_hit = row.get("v2_signal_correct")
        nbm_hit = row.get("nbm_signal_correct")
        if v2_hit is None or nbm_hit is None:
            continue
        counted += 1
        if v2_hit and not nbm_hit:
            v2_disagree_wins += 1
        elif nbm_hit and not v2_hit:
            nbm_disagree_wins += 1

    criteria = []
    if v2_mae is not None and nbm_mae is not None:
        criteria.append(("mu_accuracy", nbm_mae <= v2_mae))
    if v2_cov is not None and nbm_cov is not None:
        criteria.append(("sigma_calibration", abs(nbm_cov - 0.68) <= abs(v2_cov - 0.68)))
    if counted > 0:
        criteria.append(("edge_quality", nbm_disagree_wins >= v2_disagree_wins))
    if prob_sums:
        in_range = sum(1 for s in prob_sums if 0.98 <= s <= 1.02)
        criteria.append(("probability_sum", (in_range / len(prob_sums)) >= 0.8))

    wins = sum(1 for _, ok in criteria if ok)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settled_rows": len(settled),
        "v2_mu_mae": round(v2_mae, 4) if v2_mae is not None else None,
        "nbm_mu_mae": round(nbm_mae, 4) if nbm_mae is not None else None,
        "v2_sigma_coverage_1s": round(v2_cov, 4) if v2_cov is not None else None,
        "nbm_sigma_coverage_1s": round(nbm_cov, 4) if nbm_cov is not None else None,
        "disagree_count_scored": counted,
        "v2_disagree_wins": v2_disagree_wins,
        "nbm_disagree_wins": nbm_disagree_wins,
        "nbm_prob_sum_in_0p98_1p02_rate": round(
            (sum(1 for s in prob_sums if 0.98 <= s <= 1.02) / len(prob_sums)), 4
        ) if prob_sums else None,
        "criteria": [{"name": name, "nbm_wins": ok} for name, ok in criteria],
        "nbm_wins_criteria": wins,
        "swap_recommendation": "nbm_if_3_of_4" if wins >= 3 else "keep_v2",
    }

    summary_path = config.PROJECT_ROOT / "reports" / "nbm_vs_v2_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def reconcile_all_shadow_records(run_paper_reconcile=True):
    """
    Nightly reconciliation for paper + shadow logs.

    Updates:
    - reports/paper_trades.jsonl via PaperTracker reconciliation
    - reports/nbm_shadow.jsonl actual_tmax + correctness fields
    - reports/model_comparison.jsonl actual_tmax field
    - reports/nbm_vs_v2_summary.json
    """
    db = SettlementDB()

    # Reconcile paper trades (existing flow)
    paper_settled = []
    if run_paper_reconcile:
        try:
            from paper_tracker import PaperTracker
            tracker = PaperTracker()
            paper_settled = tracker.reconcile_settlements()
        except Exception as e:
            logger.warning("Paper reconciliation failed: %s", e)

    nbm_stats = _update_nbm_shadow_records(db)
    comparison_stats = _update_model_comparison_records(db)
    summary = _write_nbm_vs_v2_summary()

    out = {
        "paper_settled_count": len(paper_settled),
        "nbm_shadow": nbm_stats,
        "model_comparison": comparison_stats,
        "summary_generated": summary is not None,
        "summary": summary,
    }
    logger.info(
        "Shadow reconcile: paper=%d nbm_updated=%d model_cmp_updated=%d",
        len(paper_settled),
        nbm_stats.get("updated", 0),
        comparison_stats.get("updated", 0),
    )
    return out


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NWS CLI Scraper for settlement data")
    parser.add_argument("--market", default="KXHIGHNY", help="Market ticker")
    parser.add_argument("--date", type=str, default=None, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--list", action="store_true", help="List all settlements")
    parser.add_argument(
        "--reconcile-shadow",
        action="store_true",
        help="Run phase 4.5 nightly reconciliation for paper + shadow logs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db = SettlementDB()

    if args.reconcile_shadow:
        result = reconcile_all_shadow_records(run_paper_reconcile=True)
        print(json.dumps(result, indent=2))
        return

    if args.list:
        print("=" * 60)
        print("SETTLEMENT DATABASE")
        print("=" * 60)
        settlements = db.list_all()
        if not settlements:
            print("  (empty)")
        else:
            print("  %-12s %-12s %6s %12s" % ("Market", "Date", "Tmax", "Source"))
            print("  " + "-" * 48)
            for s in settlements:
                print("  %-12s %-12s %5dF %12s" % (
                    s["series_ticker"], s["date"], s["tmax_f"], s["source"]))
        return

    print("=" * 60)
    print("NWS CLI SCRAPER")
    print("Market: %s" % args.market)
    print("=" * 60)

    if args.date:
        target_date = date.fromisoformat(args.date)
        settlement = get_settlement_value(
            target_date=target_date,
            series_ticker=args.market,
            db=db,
        )
        if settlement:
            print("\nRequested date: %s" % target_date)
            print("Max temp:       %dF" % settlement["tmax_f"])
            print("Source:         %s (%s)" % (
                settlement.get("source", "unknown"),
                settlement.get("confidence", "unknown"),
            ))
        else:
            print("\nNo settlement available for %s yet." % target_date)
            print("Checked DB, live CLI report, and GHCN fallback.")
        return

    report_date, tmax = scrape_and_record(args.market, db)

    if report_date and tmax is not None:
        print("\nReport date: %s" % report_date)
        print("Max temp:    %dF" % tmax)
    else:
        print("\nCould not extract settlement data.")
        print("CLI report may not be published yet.")


if __name__ == "__main__":
    main()

"""
nbm_shadow.py -- NBM + HRRR shadow prediction logger.

Runs alongside the main bot and records parallel model outputs without
changing live/paper trading behavior.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

import config
from model import parse_bucket_title

logger = logging.getLogger("nbm_shadow")

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

_ET = timezone(timedelta(hours=-5))
_NBM_PERCENTILE_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_max_p10",
    "temperature_2m_max_p25",
    "temperature_2m_max_p50",
    "temperature_2m_max_p75",
    "temperature_2m_max_p90",
]
_ENSEMBLE_MODELS_FALLBACK = [
    "bom_access_global_ensemble",
    "ecmwf_ifs025",
    "gfs_seamless",
]
_LAT_LON_BY_SERIES = {
    "KXHIGHNY": (40.7831, -73.9712),
    "KXHIGHCHI": (41.9742, -87.9073),
}
_MIN_SIGMA = 0.05


def _utc_now_iso() -> str:
    """UTC timestamp with Z suffix for JSON logs."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _http_get_json(url: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    """Simple JSON GET wrapper so tests can patch one seam."""
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_daily_row(data: dict[str, Any], target_date: date) -> dict[str, float] | None:
    """Extract one daily row by date from Open-Meteo response."""
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return None

    target_iso = target_date.isoformat()
    try:
        idx = times.index(target_iso)
    except ValueError:
        return None

    row: dict[str, float] = {}
    for key, values in daily.items():
        if key == "time" or not isinstance(values, list) or idx >= len(values):
            continue
        value = values[idx]
        if value is None:
            continue
        try:
            row[key] = float(value)
        except (TypeError, ValueError):
            continue
    return row


def _first_present(row: dict[str, float], candidates: list[str]) -> float | None:
    """Return first matching non-null value."""
    for key in candidates:
        if key in row:
            return row[key]
    return None


def _quantile(values: list[float], q: float) -> float:
    """Linear interpolation quantile in [0, 1]."""
    if not values:
        raise ValueError("values is empty")
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def _fetch_nbm_row(lat: float, lon: float, target_date: date) -> dict[str, float] | None:
    """Try Open-Meteo forecast endpoint with NBM model and percentile fields."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(_NBM_PERCENTILE_DAILY_VARS),
        "models": "nbm_conus",
        "forecast_days": 3,
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    data = _http_get_json(OPEN_METEO_FORECAST_URL, params=params, timeout=20)
    return _extract_daily_row(data, target_date)


def _extract_member_tmax_values(daily: dict[str, Any], date_idx: int) -> list[float]:
    """Extract ensemble member max values from one Open-Meteo daily payload."""
    values: list[float] = []
    for key, arr in daily.items():
        if key == "time":
            continue
        if "temperature_2m_max" not in key:
            continue
        if not isinstance(arr, list) or date_idx >= len(arr):
            continue
        raw = arr[date_idx]
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def _fetch_ensemble_percentiles(lat: float, lon: float, target_date: date) -> dict[str, float] | None:
    """
    Fallback: derive percentiles from Open-Meteo ensemble members.

    Uses multiple models and pools member Tmax values.
    """
    all_members: list[float] = []
    for model_name in _ENSEMBLE_MODELS_FALLBACK:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "models": model_name,
        }
        try:
            data = _http_get_json(OPEN_METEO_ENSEMBLE_URL, params=params, timeout=20)
        except Exception as exc:
            logger.warning("NBM shadow: ensemble fallback failed for %s: %s", model_name, exc)
            continue

        daily = data.get("daily") or {}
        times = daily.get("time") or []
        if not times:
            continue
        try:
            idx = times.index(target_date.isoformat())
        except ValueError:
            idx = 0
        all_members.extend(_extract_member_tmax_values(daily, idx))

    if len(all_members) < 5:
        return None

    values = sorted(all_members)
    p10 = _quantile(values, 0.10)
    p25 = _quantile(values, 0.25)
    p50 = _quantile(values, 0.50)
    p75 = _quantile(values, 0.75)
    p90 = _quantile(values, 0.90)
    sigma = max((p90 - p10) / 3.29, _MIN_SIGMA)

    return {
        "nbm_mu": statistics.mean(values),
        "nbm_p10": p10,
        "nbm_p25": p25,
        "nbm_p50": p50,
        "nbm_p75": p75,
        "nbm_p90": p90,
        "nbm_sigma": sigma,
        "n_members": len(values),
    }


def fetch_nbm_forecast(lat: float, lon: float, target_date: date) -> dict[str, Any] | None:
    """
    Fetch NBM expected high and percentiles for target_date.

    Fallback chain:
    1) Open-Meteo NBM daily with percentile fields
    2) Open-Meteo ensemble-derived percentiles
    3) None
    """
    fetched_at = _utc_now_iso()

    row = None
    try:
        row = _fetch_nbm_row(lat, lon, target_date)
    except Exception as exc:
        logger.warning("NBM shadow: Open-Meteo NBM fetch failed: %s", exc)

    if row:
        nbm_mu = _first_present(row, ["temperature_2m_max", "temperature_2m_max_p50"])
        p10 = _first_present(row, ["temperature_2m_max_p10"])
        p25 = _first_present(row, ["temperature_2m_max_p25"])
        p50 = _first_present(row, ["temperature_2m_max_p50", "temperature_2m_max"])
        p75 = _first_present(row, ["temperature_2m_max_p75"])
        p90 = _first_present(row, ["temperature_2m_max_p90"])

        if nbm_mu is not None and p10 is not None and p90 is not None and p90 >= p10:
            return {
                "nbm_mu": nbm_mu,
                "nbm_p10": p10,
                "nbm_p25": p25,
                "nbm_p50": p50,
                "nbm_p75": p75,
                "nbm_p90": p90,
                "nbm_sigma": max((p90 - p10) / 3.29, _MIN_SIGMA),
                "fetched_at_utc": fetched_at,
                "source": "open-meteo-nbm",
            }

    fallback = _fetch_ensemble_percentiles(lat, lon, target_date)
    if fallback is None:
        return None

    if row:
        nbm_point_mu = _first_present(row, ["temperature_2m_max"])
        if nbm_point_mu is not None:
            fallback["nbm_mu"] = nbm_point_mu

    fallback["fetched_at_utc"] = fetched_at
    fallback["source"] = "open-meteo-ensemble-fallback"
    return fallback


def fetch_hrrr_tmax(lat: float, lon: float, target_date: date) -> dict[str, Any] | None:
    """
    Fetch HRRR hourly temperatures and derive same-day expected Tmax.

    Returns None for non-same-day targets.
    """
    today_et = datetime.now(_ET).date()
    if target_date != today_et:
        return None

    fetched_at = _utc_now_iso()
    for model_name in ("hrrr", "hrrr_conus"):
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "models": model_name,
            "forecast_hours": 18,
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        try:
            data = _http_get_json(OPEN_METEO_FORECAST_URL, params=params, timeout=20)
        except Exception as exc:
            logger.warning("NBM shadow: HRRR fetch failed for %s: %s", model_name, exc)
            continue

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        if not times or not temps:
            continue

        hrrr_hours: list[tuple[str, float]] = []
        for ts, temp in zip(times, temps):
            if temp is None:
                continue
            try:
                dt = datetime.fromisoformat(ts)
                temp_f = float(temp)
            except (TypeError, ValueError):
                continue
            if dt.date() != target_date:
                continue
            hrrr_hours.append((ts, temp_f))

        if not hrrr_hours:
            continue

        hrrr_mu = max(temp for _, temp in hrrr_hours)
        return {
            "hrrr_mu": hrrr_mu,
            "hrrr_hours": hrrr_hours,
            "hrrr_run_time": (times[0] if times else None),
            "fetched_at_utc": fetched_at,
        }

    return None


def compute_nbm_bucket_probs(nbm_mu: float, nbm_sigma: float, buckets: list[Any]) -> dict[str, float]:
    """Compute bucket probabilities from an N(mu, sigma) forecast."""
    sigma = max(float(nbm_sigma), _MIN_SIGMA)
    probs: dict[str, float] = {}
    total = 0.0

    for bucket in buckets:
        p = float(bucket.probability(nbm_mu, sigma))
        probs[bucket.ticker] = p
        total += p

    if total < 0.95 or total > 1.05:
        logger.warning("NBM shadow: bucket probabilities sum to %.3f (expected ~1.0)", total)

    return probs


def _best_edge(prob_by_ticker: dict[str, float], market_prices: dict[str, float]) -> dict[str, Any] | None:
    """Find best directional edge from model probabilities vs market prices."""
    best: dict[str, Any] | None = None
    for ticker, prob in prob_by_ticker.items():
        price = market_prices.get(ticker)
        if price is None:
            continue

        yes_edge = prob - price
        no_edge = price - prob
        if yes_edge >= no_edge:
            edge = yes_edge
            side = "buy_yes"
        else:
            edge = no_edge
            side = "buy_no"

        if best is None or edge > best["edge"]:
            best = {"bucket": ticker, "edge": float(edge), "side": side}

    return best


def _build_market_prices(markets: list[Any]) -> tuple[list[Any], dict[str, float]]:
    """Parse buckets and derive market YES prices from market snapshots."""
    buckets = []
    market_prices: dict[str, float] = {}

    for market in markets:
        bucket = parse_bucket_title(market.ticker, market.title)
        if bucket is None:
            continue
        buckets.append(bucket)

        price = None
        if market.yes_bid is not None and market.yes_ask is not None:
            price = (market.yes_bid + market.yes_ask) / 200.0
        elif market.yes_ask is not None:
            price = market.yes_ask / 100.0
        elif market.yes_bid is not None:
            price = market.yes_bid / 100.0
        elif market.last_price is not None:
            price = market.last_price / 100.0

        if price is not None:
            market_prices[market.ticker] = float(price)

    return buckets, market_prices


def _fetch_market_state(market_config: Any, target_date: date) -> tuple[list[Any], dict[str, float]]:
    """Fallback market fetch for standalone invocation (not used by bot integration)."""
    from kalshi_auth import KalshiAuth
    from kalshi_client import KalshiClient

    auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
    client = KalshiClient(auth=auth)
    markets = client.get_markets(series_ticker=market_config.series_ticker, status="open")
    target_token = target_date.strftime("%y") + target_date.strftime("%b").upper() + target_date.strftime("%d")
    selected = [m for m in markets if target_token in m.ticker] or markets
    return _build_market_prices(selected)


def _market_lat_lon(market_config: Any) -> tuple[float, float]:
    """Resolve market coordinates."""
    if market_config.series_ticker in _LAT_LON_BY_SERIES:
        return _LAT_LON_BY_SERIES[market_config.series_ticker]
    return config.NWS_LAT, config.NWS_LON


def _json_ready_prices(market_prices: dict[str, float]) -> dict[str, float]:
    """Round market prices for compact JSONL records."""
    return {ticker: round(price, 4) for ticker, price in market_prices.items()}


def _json_ready_probs(probs: dict[str, float]) -> dict[str, float]:
    """Round probabilities for compact JSONL records."""
    return {ticker: round(prob, 6) for ticker, prob in probs.items()}


def _shadow_log_path() -> Path:
    """Output path for NBM shadow records."""
    path = getattr(config, "NBM_SHADOW_PATH", config.PROJECT_ROOT / "reports" / "nbm_shadow.jsonl")
    return Path(path)


def log_nbm_shadow(
    target_date: date,
    market_config: Any,
    current_v2_mu: float,
    current_v2_sigma: float,
    buckets: list[Any] | None = None,
    market_prices: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """
    Fetch NBM + HRRR, compute bucket probabilities, and append one JSONL record.

    Designed to be non-fatal and side-effect free for live trading logic.
    """
    if buckets is None or market_prices is None:
        buckets, market_prices = _fetch_market_state(market_config, target_date)

    if not buckets:
        logger.warning("NBM shadow: no buckets available for %s %s", market_config.series_ticker, target_date)
        return None

    lat, lon = _market_lat_lon(market_config)
    nbm = fetch_nbm_forecast(lat, lon, target_date)
    hrrr = fetch_hrrr_tmax(lat, lon, target_date)

    v2_probs = compute_nbm_bucket_probs(current_v2_mu, current_v2_sigma, buckets)
    nbm_probs: dict[str, float] = {}
    if nbm is not None:
        nbm_probs = compute_nbm_bucket_probs(nbm["nbm_mu"], nbm["nbm_sigma"], buckets)

    v2_best_edge = _best_edge(v2_probs, market_prices)
    nbm_best_edge = _best_edge(nbm_probs, market_prices) if nbm_probs else None

    if v2_best_edge is not None:
        v2_best_edge["edge"] = round(v2_best_edge["edge"], 6)
    if nbm_best_edge is not None:
        nbm_best_edge["edge"] = round(nbm_best_edge["edge"], 6)

    record = {
        "timestamp_utc": _utc_now_iso(),
        "target_date": str(target_date),
        "series_ticker": market_config.series_ticker,
        "v2_mu": round(float(current_v2_mu), 4),
        "v2_sigma": round(float(current_v2_sigma), 4),
        "v2_bucket_probs": _json_ready_probs(v2_probs),
        "nbm_mu": (round(float(nbm["nbm_mu"]), 4) if nbm else None),
        "nbm_sigma": (round(float(nbm["nbm_sigma"]), 4) if nbm else None),
        "nbm_p10": (round(float(nbm["nbm_p10"]), 4) if nbm and nbm.get("nbm_p10") is not None else None),
        "nbm_p25": (round(float(nbm["nbm_p25"]), 4) if nbm and nbm.get("nbm_p25") is not None else None),
        "nbm_p50": (round(float(nbm["nbm_p50"]), 4) if nbm and nbm.get("nbm_p50") is not None else None),
        "nbm_p75": (round(float(nbm["nbm_p75"]), 4) if nbm and nbm.get("nbm_p75") is not None else None),
        "nbm_p90": (round(float(nbm["nbm_p90"]), 4) if nbm and nbm.get("nbm_p90") is not None else None),
        "nbm_bucket_probs": _json_ready_probs(nbm_probs),
        "nbm_source": (nbm.get("source") if nbm else None),
        "nbm_fetched_at_utc": (nbm.get("fetched_at_utc") if nbm else None),
        "hrrr_mu": (round(float(hrrr["hrrr_mu"]), 4) if hrrr else None),
        "hrrr_run_time": (hrrr.get("hrrr_run_time") if hrrr else None),
        "market_prices": _json_ready_prices(market_prices),
        "v2_best_edge": v2_best_edge,
        "nbm_best_edge": nbm_best_edge,
        "actual_tmax": None,
        "v2_signal_correct": None,
        "nbm_signal_correct": None,
    }

    out_path = _shadow_log_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")

    return record

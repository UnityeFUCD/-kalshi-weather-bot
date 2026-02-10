"""
market_registry.py -- Multi-market configuration registry.

Each MarketConfig holds everything needed to trade a single Kalshi weather
market: NWS grid coordinates, GHCN station for settlement, model sigma values,
and data paths.  Adding a new city is just a new entry here.
"""

from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class MarketConfig:
    series_ticker: str
    display_name: str
    enabled: bool

    # NWS grid / station
    nws_grid_office: str
    nws_grid_x: int
    nws_grid_y: int
    nws_station: str

    # Model parameters
    sigma_1day: float
    sigma_sameday_am: float
    sigma_sameday_pm: float
    forecast_bias: float

    # GHCN settlement data
    ghcn_station_id: str

    # --- derived paths ---

    @property
    def ghcn_parquet_path(self) -> Path:
        return (config.DATA_DIR / "raw" / "weather" / "observations"
                / ("%s_daily.parquet" % self.ghcn_station_id))

    @property
    def forecast_snapshots_dir(self) -> Path:
        return config.FORECAST_SNAPSHOTS_DIR / self.series_ticker

    @property
    def observations_dir(self) -> Path:
        return config.OBSERVATIONS_DIR / self.series_ticker


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MARKET_REGISTRY: dict[str, MarketConfig] = {
    "KXHIGHNY": MarketConfig(
        series_ticker="KXHIGHNY",
        display_name="NYC High Temperature",
        enabled=True,
        nws_grid_office="OKX",
        nws_grid_x=33,
        nws_grid_y=37,
        nws_station="KNYC",
        sigma_1day=1.2,
        sigma_sameday_am=0.9,
        sigma_sameday_pm=0.5,
        forecast_bias=0.0,
        ghcn_station_id="USW00094728",
    ),
    "KXHIGHCHI": MarketConfig(
        series_ticker="KXHIGHCHI",
        display_name="Chicago High Temperature",
        enabled=False,
        nws_grid_office="LOT",
        nws_grid_x=65,
        nws_grid_y=76,
        nws_station="KORD",
        sigma_1day=1.3,       # calibrated from 179 days O'Hare data (2025-08 to 2026-02)
        sigma_sameday_am=0.9,  # σ_1day × 0.71 (√t scaling)
        sigma_sameday_pm=0.5,  # σ_1day × 0.41 (√t scaling)
        forecast_bias=0.1,     # slight warm bias (+0.15°F rounded)
        ghcn_station_id="USW00094846",
    ),
}


def get_enabled_markets() -> list[MarketConfig]:
    """Return all enabled MarketConfig objects, preserving insertion order."""
    return [mc for mc in MARKET_REGISTRY.values() if mc.enabled]


def get_market(series_ticker: str) -> MarketConfig | None:
    """Look up a MarketConfig by series ticker, or None if not found."""
    return MARKET_REGISTRY.get(series_ticker)

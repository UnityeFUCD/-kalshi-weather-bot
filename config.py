"""
config.py -- All configuration for the Kalshi Weather Bot.

FOCUS: NYC High Temperature (NHIGH / KXHIGHNY) -- the ONLY weather market
with meaningful liquidity on Kalshi (~$400+/day volume).
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
KEYS_DIR = PROJECT_ROOT / "keys"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
FORECAST_SNAPSHOTS_DIR = DATA_DIR / "forecast_snapshots"
OBSERVATIONS_DIR = DATA_DIR / "observations"

# Signal report -- written every scan so you can check from any device
# Drop this in OneDrive/Google Drive for mobile access
SIGNAL_REPORT_PATH = PROJECT_ROOT / "signals.txt"

# --- Owner Location ----------------------------------------------------------
# Aurora, CO 80013 -- Mountain Time (UTC-7 standard, UTC-6 DST)
OWNER_TIMEZONE = "America/Denver"
# Note: Bot trades NYC weather, but owner is in Mountain Time
# All user-facing times are shown in MT, all market logic uses ET

# --- Kalshi API (PRODUCTION) -------------------------------------------------
KALSHI_API_KEY_ID = os.environ.get(
    "KALSHI_API_KEY_ID",
    "37f5c54f-05a2-4e71-aae3-e0f426de5c98"
)
KALSHI_PRIVATE_KEY_PATH = os.environ.get(
    "KALSHI_PRIVATE_KEY_PATH",
    str(KEYS_DIR / "kalshi_private_key.pem")
)

# Both read and trade on PRODUCTION
KALSHI_PROD_URL = "https://api.elections.kalshi.com"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co"
KALSHI_READ_URL = KALSHI_PROD_URL
KALSHI_TRADE_URL = KALSHI_PROD_URL

KALSHI_API_PATH = "/trade-api/v2"

# --- Target Market: NHIGH (NYC High Temperature) ----------------------------
# Default; see market_registry.py for multi-market configuration
SERIES_TICKER = "KXHIGHNY"

# From NHIGH.pdf rules:
# - Underlying: max temp from NWS Daily Climate Report for Central Park, NY
# - Settlement: NWS observed max temp (F), reported as integer
# - "between" buckets: >= lower AND <= upper (BOTH INCLUSIVE)
# - "greater than" buckets: STRICTLY > threshold
# - "less than" buckets: STRICTLY < threshold
# - Expiration: 7:00 or 8:00 AM ET after data release
# - Position limit: $25,000 per member
# - Minimum tick: $0.01

# --- NWS API -----------------------------------------------------------------
NWS_USER_AGENT = "KalshiWeatherBot (github.com/yasir)"
# NYC defaults; per-market values in market_registry.py
NWS_LAT = 40.7831
NWS_LON = -73.9712
NWS_GRID_OFFICE = "OKX"
NWS_GRID_X = 33
NWS_GRID_Y = 37

# --- Pricing Model -----------------------------------------------------------
SIGMA_1DAY       = 1.2
SIGMA_SAMEDAY_AM = 0.9    # σ_1day × 0.71 (√t scaling)
SIGMA_SAMEDAY_PM = 0.5    # σ_1day × 0.41 (√t scaling)
FORECAST_BIAS    = 0.0    # Keep — bias is negligible   

# --- Trading Thresholds ------------------------------------------------------
MIN_EDGE = 0.08          # 8% minimum edge over market price
MIN_CONTRACTS = 5        # Below this, fees dominate
MAX_RISK_PER_TRADE = 5.0 # $5.00 max risk per trade (10% of $50 bankroll)
MAX_DAILY_EXPOSURE = 15.0 # $15 max open at once (30% of bankroll)
MAX_OPEN_POSITIONS = 3
MAX_SPREAD = 0.12        # Don't trade if bid-ask spread > 12 cents
MIN_VOLUME_24H = 50      # Minimum 24h volume in contracts

# Fee structure
TAKER_FEE_MULTIPLIER = 0.07
MAKER_FEE_MULTIPLIER = 0.0175

# --- Risk Management ---------------------------------------------------------
BANKROLL = 50.0
DAILY_LOSS_LIMIT = 8.0    # Halt if daily loss exceeds $8
WEEKLY_LOSS_LIMIT = 15.0   # Halt if weekly loss exceeds $15
NO_TRADE_WITHIN_HOURS_OF_SETTLEMENT = 1

# --- Bot Behavior ------------------------------------------------------------
SCAN_INTERVAL_SECONDS = 120
ORDER_TIMEOUT_SECONDS = 600
MAX_CANCELS_PER_MINUTE = 3
USE_MAKER_ORDERS_ONLY = True

# --- Paper Trading -----------------------------------------------------------
PAPER_TRADES_PATH = PROJECT_ROOT / "reports" / "paper_trades.jsonl"
PAPER_PENDING_PATH = PROJECT_ROOT / "reports" / "paper_pending.json"
PAPER_DAILY_DIR = PROJECT_ROOT / "reports" / "daily"
PAPER_FILL_TIMEOUT_CYCLES = 15    # ~30 min at 2-min scan interval
GHCN_PARQUET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "weather" / "observations" / "USW00094728_daily.parquet"
)

# --- Logging -----------------------------------------------------------------
LOG_FILE = LOGS_DIR / "bot.log"
LOG_LEVEL = "INFO"
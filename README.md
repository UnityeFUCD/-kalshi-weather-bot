# Kalshi Weather Trading Bot

Automated trading bot for **NHIGH** (NYC High Temperature) markets on Kalshi.

## Reality Check

From your market data scrape: **NYC high temp is the ONLY weather market with
any liquidity** (~$400/day). Everything else - Denver, Chicago, LA, rain, snow,
low temps - shows $0 volume and empty order books. This bot focuses exclusively
on NHIGH.

## Architecture

```
NWS Forecast API                    Kalshi API
     │                                  │
     ▼                                  ▼
┌──────────┐    ┌─────────┐    ┌──────────────┐
│  nws.py  │───▶│model.py │───▶│  bot.py      │
│ Central  │    │ μ,σ →   │    │ scan/paper/  │
│ Park, NY │    │ P(bucket)│    │ live modes   │
└──────────┘    └─────────┘    └──────┬───────┘
                                      │
                               ┌──────▼───────┐
                               │kalshi_client.py│
                               │+ kalshi_auth.py│
                               │ RSA-PSS auth  │
                               └──────┬───────┘
                                      │
                               ┌──────▼───────┐
                               │  risk.py      │
                               │ $5/trade max  │
                               │ $10/day halt  │
                               └───────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings, thresholds, API keys |
| `kalshi_auth.py` | RSA-PSS request signing (from Kalshi docs) |
| `kalshi_client.py` | Kalshi REST API client (public + authenticated) |
| `nws.py` | NWS forecast + observation fetching |
| `model.py` | Forecast → probability model + signal generation |
| `risk.py` | Position limits, daily loss tracking, kill switch |
| `bot.py` | Main loop tying everything together |

> Live safety: `bot.py live` requires environment variable `LIVE_TRADING=true`.

## Quick Start

```bash
# 1. Install dependencies
py -m pip install requests cryptography

# 2. Test auth (generates a signature - doesn't hit the API)
cd kalshi-weather-bot
py kalshi_auth.py

# 3. Test the pricing model (no API calls)
py model.py

# 4. Test Kalshi connectivity (public endpoint, no auth)
py kalshi_client.py

# 5. Test NWS forecast fetch
py nws.py

# 6. Run one scan cycle (read-only, no trades)
py bot.py scan --once

# 7. Run paper trading (logs what it would trade)
py bot.py paper --once

# 8. Run continuous paper trading
py bot.py paper

# 9. GO LIVE (real money - only after paper trading validates)
$env:LIVE_TRADING="true"
py bot.py live --once    # single cycle
py bot.py live           # continuous
```

## NHIGH Contract Rules (from your PDF)

- **Underlying**: Max temp from NWS Daily Climate Report, Central Park NY
- **Settlement source**: https://www.weather.gov/wrh/climate?wfo=okx
- **"Between A and B"**: A ≤ T ≤ B (both inclusive)
- **"Greater than A"**: T > A (strictly, e.g., "greater than 56" excludes 56)
- **"Less than A"**: T < A (strictly)
- **Expiration**: 7:00 or 8:00 AM ET after data release
- **Delayed to 11 AM** if inconsistent with METAR or final < earlier reports
- **Position limit**: $25,000 per member
- **Min tick**: $0.01

## Strategy

1. Fetch NWS forecast for Central Park → get predicted high temp
2. Model observed temp as: `T ~ Normal(forecast - bias, σ_error)`
3. For each bucket, compute P(bucket) using normal CDF
4. Compare to Kalshi market prices
5. Trade when `model_prob - market_price > dynamic MIN_EDGE` (base config entry is 15%, scaled by confidence/time)
6. Always use **limit orders** (maker fees 4x cheaper than taker)
7. Hold to settlement (round-tripping doubles costs)

## Risk Limits

| Parameter | Value |
|-----------|-------|
| Max risk per trade | $2.00 |
| Max daily exposure | $10.00 |
| Max open positions | 3 |
| Daily loss halt | $8.00 |
| Weekly loss halt | $15.00 |
| Max trades per run | 5 |
| Min contracts/trade | 5 |
| Min edge to trade | Dynamic (base 15%) |

## Milestone Checklist

- [ ] `py kalshi_auth.py` - prints signature without error
- [ ] `py kalshi_client.py` - shows real NHIGH markets and prices
- [ ] `py nws.py` - shows today's NYC forecast
- [ ] `py model.py` - shows probability distribution
- [ ] `py bot.py scan --once` - full cycle, no trades
- [ ] `py bot.py paper` - run 3+ days, review logs
- [ ] Win rate > 55% in paper trading?
- [ ] `$env:LIVE_TRADING="true"; py bot.py live --once` - first real trade
- [ ] Monitor settlement next morning

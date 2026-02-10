# Kalshi Weather Trading Bot

Automated trading bot for **NHIGH** (NYC High Temperature) markets on Kalshi.

## Reality Check

From your market data scrape: **NYC high temp is the ONLY weather market with
any liquidity** (~$400/day). Everything else — Denver, Chicago, LA, rain, snow,
low temps — shows $0 volume and empty order books. This bot focuses exclusively
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

## Quick Start

```bash
# 1. Install dependencies
pip install requests cryptography

# 2. Test auth (generates a signature — doesn't hit the API)
cd kalshi-weather-bot
python kalshi_auth.py

# 3. Test the pricing model (no API calls)
python model.py

# 4. Test Kalshi connectivity (public endpoint, no auth)
python kalshi_client.py

# 5. Test NWS forecast fetch
python nws.py

# 6. Run one scan cycle (read-only, no trades)
python bot.py scan --once

# 7. Run paper trading (logs what it would trade)
python bot.py paper --once

# 8. Run continuous paper trading
python bot.py paper

# 9. GO LIVE (real money — only after paper trading validates)
python bot.py live --once    # single cycle
python bot.py live           # continuous
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
5. Trade when `model_prob - market_price > 8%` (after fees)
6. Always use **limit orders** (maker fees 4x cheaper than taker)
7. Hold to settlement (round-tripping doubles costs)

## Risk Limits

| Parameter | Value |
|-----------|-------|
| Max risk per trade | $5.00 (5% of $100 bankroll) |
| Max daily exposure | $20.00 |
| Max open positions | 4 |
| Daily loss halt | $10.00 |
| Weekly loss halt | $20.00 |
| Min contracts/trade | 5 |
| Min edge to trade | 8% |

## Milestone Checklist

- [ ] `python kalshi_auth.py` — prints signature without error
- [ ] `python kalshi_client.py` — shows real NHIGH markets and prices
- [ ] `python nws.py` — shows today's NYC forecast
- [ ] `python model.py` — shows probability distribution
- [ ] `python bot.py scan --once` — full cycle, no trades
- [ ] `python bot.py paper` — run 3+ days, review logs
- [ ] Win rate > 55% in paper trading?
- [ ] `python bot.py live --once` — first real trade
- [ ] Monitor settlement next morning

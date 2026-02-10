# Kalshi Weather Bot -- Research Handoff Bundle

> Generated: 2026-02-10
> Purpose: Enable a research AI to run standardized optimization passes without touching live trading logic.

---

## 1. System Overview

The bot trades Kalshi's **KXHIGHNY** (NYC High Temperature) market. It predicts the probability that tomorrow's high temperature falls in specific 2-degree buckets, compares those probabilities to market prices, and places maker limit orders when edge exceeds a threshold.

**Data pipeline:** NWS forecast -> Normal distribution model -> Bucket probabilities -> Edge vs market -> Trade signals -> Maker orders on Kalshi.

**Live components (DO NOT MODIFY):**
- `bot.py` -- main trading loop, order execution, paper tracking
- `kalshi_api.py` -- API client, authentication, order placement
- `paper_tracker.py` -- paper trade simulation and reconciliation
- `daily_runner.py` -- scheduled task orchestrator
- `config.py` -- live config (read only for constants)

**Research-safe components (OK to modify/extend):**
- `test_backtest_ab.py` -- A/B/C/D/E backtest harness
- `test_robustness_sweep.py` -- stress test across fill/slippage scenarios
- `diagnose_diff.py` -- trade-by-trade comparison diagnostic
- `fetch_historical_ensemble.py` -- historical ensemble data acquisition
- `backtest.py` -- original backtester (reference implementation)

---

## 2. Probability Model

### Core Formula

```
T_observed ~ Normal(mu, sigma)

mu = NWS_forecast_high - forecast_bias
sigma = max(sigma_base, alpha * sigma_ens)   # V2 ensemble composition
```

### Bucket Probability

For bucket `[a, b]` (between type, both inclusive):
```
P(bucket) = Phi((b + 0.5 - mu) / sigma) - Phi((a - 0.5 - mu) / sigma)
```

Where `Phi` is the standard normal CDF: `0.5 * (1 + erf(x / sqrt(2)))`

The `+/- 0.5` is a continuity correction because NWS reports integer temperatures.

**Above bucket** (T > threshold):
```
P = 1 - Phi((threshold + 0.5 - mu) / sigma)
```

**Below bucket** (T < threshold):
```
P = Phi((threshold - 0.5 - mu) / sigma)
```

### Signal Generation

```
yes_edge = model_prob - market_price
no_edge  = market_price - model_prob   # equivalently: (1-model_prob) - (1-market_price)

if yes_edge > MIN_EDGE:  -> BUY_YES signal
if no_edge  > MIN_EDGE:  -> BUY_NO signal
```

Signals sorted by edge strength, top 3 per day taken.

### Fee Formula

```python
fee_cents = ceil(multiplier * count * price * (1 - price) * 100)
# Maker multiplier: 0.0175
# Taker multiplier: 0.07
```

Fee is quadratic in price -- maximum at 50c, zero at extremes.

---

## 3. Current Model Parameters (V2 -- Production)

| Parameter | Value | Source |
|-----------|-------|--------|
| `SIGMA_1DAY` (sigma_base) | 1.20 F | Calibrated from GHCN vs Open-Meteo historical forecasts |
| `FORECAST_BIAS` | 0.0 F | Calibration showed negligible bias |
| `MIN_EDGE` | 8% (0.08) | Flat threshold, no time-of-day scaling |
| `ENSEMBLE_ALPHA` | 1.1 | Scale factor on ensemble spread |
| `BANKROLL` | $50.00 | Current account size |
| `MAX_RISK_PER_TRADE` | $5.00 | 10% of bankroll |
| `MIN_CONTRACTS` | 5 | Below this, fees dominate |
| `MAKER_FILL_RATE` | 70% | Conservative assumption for backtest |
| `MAX_OPEN_POSITIONS` | 3 | Per day cap |

### Sigma Composition (V2)

```python
sigma_final = max(sigma_base, ENSEMBLE_ALPHA * sigma_ens)
# sigma_base = 1.2
# ENSEMBLE_ALPHA = 1.1
# sigma_ens = stdev of 82 ensemble members (51 ECMWF + 31 GFS)
```

**Key insight:** `sigma_ens` must exceed `1.2 / 1.1 = 1.09` to trigger widening. From historical data, **95% of days sigma_ens < 1.09**, so V2 = Old on 95% of days. The 5% "stormy" days are where V2 adds value by widening sigma.

### Position Sizing (V2 -- Half-Kelly)

```python
kelly_fraction = edge / payout_complement * 0.5  # half-Kelly
kelly_dollars = min(kelly_fraction * bankroll, max_risk_per_trade)
contracts = int(kelly_dollars / price_per_contract)
contracts = clamp(contracts, min_contracts, 50)
```

Where `payout_complement`:
- BUY_YES: `1.0 - market_price`
- BUY_NO: `market_price`

### What Was Tried and Rejected

| Feature | Why Rejected | Backtest Evidence |
|---------|-------------|-------------------|
| Confidence gates (4-gate scoring) | Blocked good AND bad trades equally | V1 PnL $1382 vs Old $1544 |
| Dynamic edge (20% -> 8% by time) | Overcorrected, killed morning trades | Same as above |
| Boundary z-boost (inflate sigma near edges) | Inflated sigma, destroyed edge | Same as above |
| Sigma stacking (base + beta*rev_vol + gamma*boundary) | Combined inflation killed all edge | V2-stacked << Old |
| Dynamic edge scaling by confidence | Net negative -- reduced both bad and good | Backtest showed worse Sharpe |

---

## 4. Backtest Infrastructure

### Data Files

| File | Location | Contents |
|------|----------|----------|
| Trade history | `data/raw/kalshi/KXHIGHNY_trades.parquet` | 611K+ trade prints from Kalshi API |
| Forecasts | `data/curated/historical_calibration_KNYC.parquet` | Open-Meteo historical 1-day forecasts |
| Observations | `data/raw/weather/observations/USW00094728_daily.parquet` | GHCN-Daily Central Park TMAX |
| Ensemble history | `data/curated/ensemble_history.parquet` | 93 days of calibrated per-day sigma_ens |

### Parquet Schemas

**KXHIGHNY_trades.parquet:**
```
ticker: str           # e.g. "KXHIGHNY-26FEB10-B38.5"
trade_date: date      # settlement date
created_time: str     # ISO timestamp
yes_price: float      # cents (1-99)
count: int            # contracts
```

**historical_calibration_KNYC.parquet:**
```
date: date
forecast_high_f: float   # Open-Meteo predicted Tmax in F
```

**USW00094728_daily.parquet:**
```
date: date
tmax_f: float   # actual observed Tmax in F (integer-valued)
```

**ensemble_history.parquet:**
```
date: date
sigma_ens: float          # calibrated ensemble spread (scaled from multi-model stdev)
multimodel_stdev: float   # raw 6-model forecast stdev
multimodel_mean: float    # 6-model forecast mean
n_models: int             # number of models with data (typically 6)
sigma_v2: float           # pre-composed max(1.2, 1.1 * sigma_ens)
```

### Market Price Reconstruction

```python
# VWAP from morning trades (11-15 UTC = 6-10 AM ET)
market_price = sum(yes_price * count) / sum(count)  # per ticker
# Falls back to all-day VWAP if < 10 morning trades
```

### Deterministic Fill Simulation

```python
def _stable_uniform(*parts):
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed).random()

# Fill happens if _stable_uniform(target_date, ticker, side) <= maker_fill_rate
```

This ensures identical results across runs -- critical for comparing models.

### BacktestConfig Dataclass

```python
@dataclass
class BacktestConfig:
    name: str
    sigma: float               # fixed sigma (0.0 if using variable/historical)
    min_edge: float            # minimum edge threshold (0.08 = 8%)
    forecast_bias: float       # mu adjustment (0.0)
    use_confidence_gates: bool # V1 only
    use_dynamic_edge: bool     # V1 only
    boundary_z_boost: bool     # V1 only
    use_kelly: bool            # half-Kelly sizing (V2)
    bankroll: float            # $50
    maker_fill_rate: float     # 0.70
    slippage_cents: int        # 0 (additional adverse)
    max_risk_per_trade: float  # $5.00
    min_contracts: int         # 5
    variable_sigma: bool       # per-day simulated sigma (Model D)
    historical_sigma: bool     # per-day real sigma from parquet (Model E)
    sigma_base: float          # 1.2 (for variable/historical modes)
```

---

## 5. Current Backtest Results (Feb 10, 2026)

### Model Comparison (fill=70%, slippage=0c)

| Metric | A: Old | B: V1 | C: V2-fixed | D: V2-var | E: V2-hist |
|--------|--------|-------|-------------|-----------|------------|
| Total trades | 179 | 165 | 174 | 180 | 179 |
| Win rate | 77% | 73% | 76% | 76% | 77% |
| **Total PnL** | **$1544.22** | $1382.01 | $1540.95 | $1460.37 | **$1548.49** |
| Avg PnL/trade | $8.63 | $8.38 | $8.86 | $8.11 | $8.65 |
| Max drawdown | $14.01 | $14.01 | $17.23 | $14.01 | $14.01 |
| Brier score | 0.1408 | 0.1338 | 0.1402 | 0.1389 | 0.1408 |
| Sharpe (ann.) | 12.81 | 12.19 | 12.68 | 12.27 | 12.86 |

**Winner: E (V2-hist)** -- +$4.27 over Old, same drawdown, best Sharpe.

### Robustness Sweep (5 stress scenarios)

| Scenario | Old PnL | V2-fixed PnL | V2-var PnL |
|----------|---------|-------------|------------|
| Base (70% fill, 0c slip) | $1544 | $1541 | $1460 |
| Low fill (50%) | $1094 | $1089 | $1020 |
| High slip (2c) | $1393 | $1390 | $1310 |
| Pessimistic (50% fill + 2c) | $1001 | $998 | $926 |
| Optimistic (90% fill) | $1871 | $1864 | $1788 |

Old wins all 5 when V2 uses static or random sigma. V2-hist not yet tested in sweep (TODO).

### Trade-by-Trade Diagnosis (Old vs V2-fixed with static σ=1.43)

| Category | Count | Old PnL | V2 PnL | Delta |
|----------|-------|---------|--------|-------|
| BOTH_WIN | 126 | $1457.09 | $1454.61 | -$2.48 |
| BOTH_LOSE | 36 | -$124.88 | -$120.81 | +$4.07 |
| OLD_ONLY_WIN | 12 | +$42.13 | $0.00 | -$42.13 |
| OLD_ONLY_LOSE | 5 | -$7.86 | $0.00 | +$7.86 |
| V2_ONLY_WIN | 7 | $0.00 | +$37.51 | +$37.51 |
| V2_ONLY_LOSE | 2 | $0.00 | -$8.10 | -$8.10 |

**Root cause:** V2 with wider sigma shifts probability from center buckets to wings. Old wins 12 center-bucket BUY_YES trades V2 misses; V2 wins 7 wing-bucket BUY_NO trades Old misses. Net signal selection: -$4.86 for V2 with static sigma. With historical sigma (95% same as Old), this penalty vanishes.

---

## 6. Experiment Matrix -- What to Optimize

### Experiment 1: Sigma Base Calibration

**Question:** Is 1.2 truly optimal, or would 1.1 or 1.3 be better?

```
py -X utf8 test_backtest_ab.py --sigma-ens 1.0
# Then manually edit configs in test_backtest_ab.py:
# Change sigma_base from 1.2 to test values: [1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4]
```

**Metric to optimize:** Total PnL at 70% fill, with Sharpe as tiebreaker.

### Experiment 2: Ensemble Alpha Sweep

**Question:** Is α=1.1 the right scale factor for ensemble spread?

```python
# In config.py, ENSEMBLE_ALPHA currently = 1.1
# Test values: [0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
# For each alpha:
#   sigma_v2 = max(sigma_base, alpha * sigma_ens)
#   Rebuild ensemble_history.parquet (column sigma_v2 changes)
#   Re-run Model E backtest
```

**Note:** Must re-run `fetch_historical_ensemble.py` after changing alpha, OR compute in-test:
```python
sigma = max(sigma_base, new_alpha * row["sigma_ens"])  # use raw sigma_ens column
```

### Experiment 3: MIN_EDGE Optimization

**Question:** Is 8% the sweet spot, or would 6% or 10% be better?

```
# Edit min_edge in BacktestConfig for models A and E
# Test values: [0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
# Run full A/E comparison for each
```

**Constraint:** Lower edge = more trades = more fee drag. Higher edge = fewer trades = more variance.

### Experiment 4: Kelly Fraction

**Question:** Is 0.5 (half-Kelly) optimal, or would 0.25 or 0.75 be better?

```python
# In test_backtest_ab.py, kelly_contracts() function:
kelly_f = edge / payout_complement * FRACTION  # currently 0.5
# Test: [0.25, 0.33, 0.5, 0.67, 0.75, 1.0]
```

**Metric:** PnL AND drawdown. Full Kelly (1.0) maximizes long-run growth but has brutal drawdowns.

### Experiment 5: Fill Rate Sensitivity

**Question:** How sensitive are results to fill rate assumptions?

```
py -X utf8 test_backtest_ab.py --fill-rate 0.50
py -X utf8 test_backtest_ab.py --fill-rate 0.60
py -X utf8 test_backtest_ab.py --fill-rate 0.70
py -X utf8 test_backtest_ab.py --fill-rate 0.80
py -X utf8 test_backtest_ab.py --fill-rate 0.90
```

### Experiment 6: Slippage Sensitivity

```
py -X utf8 test_backtest_ab.py --slippage 0
py -X utf8 test_backtest_ab.py --slippage 1
py -X utf8 test_backtest_ab.py --slippage 2
py -X utf8 test_backtest_ab.py --slippage 3
```

### Experiment 7: Combined Sweep (fill x slippage x edge)

```
py -X utf8 test_robustness_sweep.py
```

Currently tests 5 scenarios. Extend to include Model E (V2-hist).

### Experiment 8: Distribution Shape

**Question:** Is Normal the right distribution, or would Student-t (heavier tails) fit better?

```python
# Replace phi() with Student-t CDF (df=5 or df=10):
from scipy.stats import t as t_dist

def bucket_prob_t(low, high, mu, sigma, df=5):
    return t_dist.cdf((high + 0.5 - mu) / sigma, df) - \
           t_dist.cdf((low - 0.5 - mu) / sigma, df)
```

**Evidence needed:** Compare Brier scores. Lower Brier = better calibration.

### Experiment 9: Forecast Bias by Temperature Range

**Question:** Does NWS have asymmetric bias (overforecasts warm days, underforecasts cold)?

```python
# From calibration data, compute bias = forecast - actual, grouped by temp range:
# Range 20-30F: bias = ?
# Range 30-40F: bias = ?
# Range 40-50F: bias = ?
# Then apply per-range bias correction to mu
```

### Experiment 10: Sigma by Lead Time

**Question:** Should sigma shrink as we approach settlement?

Currently we use a single sigma for all trades. The bot scans every 2 minutes from 6 AM to 4 PM ET. Morning trades have ~18 hours of uncertainty; afternoon trades have ~6 hours.

```python
# Existing config (not used in backtest):
SIGMA_1DAY       = 1.2   # before day starts
SIGMA_SAMEDAY_AM = 0.9   # morning (sqrt-t scaling)
SIGMA_SAMEDAY_PM = 0.5   # afternoon
```

**Challenge:** Backtest trade timestamps are UTC, need to convert to ET and assign sigma by time slot. Would require modifying `run_backtest()` to use trade hours.

---

## 7. Metrics Schema

Every experiment should report these metrics:

```
{
    "experiment_id": "exp_003_minedge_006",
    "model_name": "E: V2-hist",
    "parameters": {
        "sigma_base": 1.2,
        "ensemble_alpha": 1.1,
        "min_edge": 0.06,
        "kelly_fraction": 0.5,
        "fill_rate": 0.70,
        "slippage_cents": 0
    },
    "results": {
        "total_trades": 179,
        "unique_days": 71,
        "win_rate": 0.77,
        "total_pnl": 1548.49,
        "avg_pnl_per_trade": 8.65,
        "total_fees": 12.34,
        "max_drawdown": 14.01,
        "brier_score": 0.1408,
        "annualized_sharpe": 12.86,
        "avg_edge": 0.156,
        "pnl_per_day": 21.81,
        "buy_yes_trades": 120,
        "buy_no_trades": 59,
        "buy_yes_winrate": 0.82,
        "buy_no_winrate": 0.68
    }
}
```

### Key Decision Metrics (in priority order)

1. **Total PnL** -- primary objective
2. **Max Drawdown** -- risk constraint (must stay < $25 for $50 bankroll)
3. **Sharpe Ratio** -- risk-adjusted returns
4. **Win Rate** -- secondary (high win rate = less variance)
5. **Brier Score** -- model calibration quality (lower = better)
6. **Trade Count** -- more trades = more statistical significance

---

## 8. Exact Commands

### Run the full A/B/C/D/E backtest
```bash
cd C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot
py -X utf8 test_backtest_ab.py
```

### Run robustness sweep
```bash
py -X utf8 test_robustness_sweep.py
```

### Run trade-by-trade diagnosis
```bash
py -X utf8 diagnose_diff.py
```

### Fetch fresh historical ensemble data (refreshes parquet)
```bash
py -X utf8 fetch_historical_ensemble.py
```

### Run original backtester with parameter sweep
```bash
py -X utf8 backtest.py --sweep
py -X utf8 backtest.py --sigma 1.2 --edge 0.08
```

### Run live signal comparison (requires API)
```bash
py -X utf8 test_compare_live.py
```

### Run model.py self-test
```bash
py -X utf8 model.py
```

### Run ensemble.py self-test
```bash
py -X utf8 ensemble.py
```

---

## 9. File Map

```
-kalshi-weather-bot/
  config.py                  # All constants (READ ONLY for research)
  model.py                   # Probability model (Bucket, Signal, phi, compute_signals, compute_fee)
  ensemble.py                # Ensemble sigma (EnsembleForecaster, compose_sigma)
  confidence.py              # Confidence scoring (4 gates -- logging only, not blocking)
  backtest.py                # Original backtester (reference)
  bot.py                     # LIVE TRADING (DO NOT MODIFY)
  market_registry.py         # MarketConfig dataclass, per-market parameters
  nws.py                     # NWS API forecast fetcher
  delta_tracker.py           # Forecast revision tracking (Phase 4A)
  metar_obs.py               # METAR/ASOS observations (Phase 4B)
  cli_scraper.py             # NWS CLI settlement report parser (Phase 4.5)
  nbm_shadow.py              # NBM shadow predictions (logging only)
  arb_scanner.py             # Bucket arbitrage detection (detection only)

  test_backtest_ab.py        # A/B/C/D/E model comparison backtest
  test_robustness_sweep.py   # Stress test across scenarios
  test_compare_live.py       # Live market comparison
  diagnose_diff.py           # Trade-by-trade diagnostic
  fetch_historical_ensemble.py  # Historical ensemble data fetcher

  data/
    raw/kalshi/KXHIGHNY_trades.parquet        # 611K trade prints
    raw/weather/observations/USW00094728_daily.parquet  # GHCN TMAX
    curated/historical_calibration_KNYC.parquet  # Forecast history
    curated/ensemble_history.parquet           # 93-day ensemble sigma history

  reports/
    paper_trades.jsonl        # Paper trade log
    ensemble_history.jsonl    # Live ensemble sigma log
    nbm_shadow.jsonl          # NBM shadow predictions
    arb_opportunities.jsonl   # Detected arbitrage
```

---

## 10. Known Issues & Open Questions

1. **Ensemble history is only 93 days** -- Open-Meteo previous-runs API maxes out at ~93 days. The calibration (multi-model stdev -> real ensemble sigma) uses only 5 overlap days. More data would improve confidence.

2. **Backtest doesn't vary sigma by time of day** -- All trades use the same sigma regardless of when they'd execute. Real bot sees different uncertainty at 6 AM vs 3 PM.

3. **NBM API returns 400** -- Open-Meteo doesn't support `nbm_conus` with percentile fields. Ensemble fallback works but gives different mean than NWS (35.5F vs 39F on test day).

4. **Only NYC market tested** -- Chicago (KXHIGHCHI) is calibrated (sigma=1.3, bias=0.1) but disabled. No backtest data for Chicago.

5. **Trade data starts Nov 2025** -- Limited to ~3 months of history. Summer/extreme weather patterns not represented.

6. **VWAP reconstruction assumes no front-running** -- We use morning VWAP as "price we'd see." In reality, our orders would move the market slightly.

7. **Maker fill rate is assumed constant** -- Real fill rates depend on spread, volume, time of day. A model of fill probability as a function of these factors could improve backtest realism.

---

## 11. Research Rules

1. **Never modify live trading files** (`bot.py`, `kalshi_api.py`, `paper_tracker.py`, `daily_runner.py`)
2. **Always use deterministic fills** (`_stable_uniform()`) for reproducibility
3. **Report full metrics schema** for every experiment
4. **Compare against Model A (Old) as baseline** -- it's the simplest and most robust
5. **Improvements must beat Old on PnL AND not increase drawdown by >50%**
6. **Run robustness sweep** on any candidate change before recommending
7. **Use `py -X utf8`** on Windows to avoid encoding errors
8. **All new test scripts** should go in the project root alongside existing test files

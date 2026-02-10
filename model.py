"""
model.py -- Temperature Forecast -> Bucket Probability Model.

Core idea:
    T_observed ~ Normal(mu_forecast - bias, sigma_error)

Where:
    mu_forecast = NWS predicted high temp
    bias = systematic NWS over/under-prediction (starts at 0, calibrate)
    sigma_error = forecast error std dev (~2.5F for 1-day, ~1.5F same-day AM)

For each Kalshi bucket [a, b], compute:
    P(bucket) = Phi((b+0.5 - mu) / sigma) - Phi((a-0.5 - mu) / sigma)

The +/-0.5 continuity correction accounts for integer-valued NWS observations.

NHIGH bucket semantics (from PDF):
- "between A and B": A <= T <= B  (both inclusive)
- "greater than A": T > A (strictly)
- "less than A": T < A (strictly)
"""

import math
import logging
import re
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger(__name__)


def phi(x):
    """Standard normal CDF. Uses math.erf for accuracy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Bucket:
    """A single temperature bucket from a Kalshi market."""
    ticker: str          # Market ticker
    title: str           # e.g., "30 to 31" or "27 or below" or "34 or above"
    bucket_type: str     # "between", "above", "below"
    low: Optional[int]   # Lower bound (None for "below" type)
    high: Optional[int]  # Upper bound (None for "above" type)

    def probability(self, mu, sigma):
        """
        Compute P(this bucket hits) given T ~ N(mu, sigma).
        
        Uses continuity correction because NWS reports integer temps.
        """
        if self.bucket_type == "between":
            # "between A and B" -> A <= T <= B (both inclusive)
            # With continuity correction: P(A-0.5 < T < B+0.5)
            p = phi((self.high + 0.5 - mu) / sigma) - \
                phi((self.low - 0.5 - mu) / sigma)
        elif self.bucket_type == "above":
            # "greater than A" -> T > A (strictly)
            # With continuity correction: P(T > A+0.5)
            p = 1.0 - phi((self.low + 0.5 - mu) / sigma)
        elif self.bucket_type == "below":
            # "less than A" -> T < A (strictly)
            # With continuity correction: P(T < A-0.5)
            p = phi((self.high - 0.5 - mu) / sigma)
        else:
            logger.error("Unknown bucket type: %s", self.bucket_type)
            p = 0.0
        
        return max(0.0, min(1.0, p))


def parse_bucket_title(ticker, title):
    """
    Parse a Kalshi market into a Bucket.
    
    Real title formats from the API:
    - "Will the **high temp in NYC** be 38-39 on Feb 10, 2026?"
    - "Will the **high temp in NYC** be >39 on Feb 10, 2026?"
    - "Will the **high temp in NYC** be <32 on Feb 10, 2026?"
    
    Note: titles may contain degree symbols in various Unicode forms,
    or no degree symbol at all. We match on digits only to be safe.
    
    Real ticker formats:
    - KXHIGHNY-26FEB10-B38.5  -> between 38-39 (midpoint encoding)
    - KXHIGHNY-26FEB10-T39    -> greater than 39
    - KXHIGHNY-26FEB10-T32    -> less than 32 (determined by title)
    
    We try title first, then fall back to ticker parsing.
    """
    title_clean = title.strip()
    
    # ---- Strategy 1: Parse from title text (degree-sign agnostic) ----
    # We strip everything that's not digits, <, >, -, or whitespace
    # to avoid Unicode degree symbol issues
    
    # "be >X" (above) -- check this BEFORE the range pattern
    m = re.search(r'be\s*>\s*(\d+)', title_clean)
    if m:
        return Bucket(
            ticker=ticker, title=title_clean, bucket_type="above",
            low=int(m.group(1)), high=None
        )
    
    # "be <X" (below) -- check this BEFORE the range pattern
    m = re.search(r'be\s*<\s*(\d+)', title_clean)
    if m:
        return Bucket(
            ticker=ticker, title=title_clean, bucket_type="below",
            low=None, high=int(m.group(1))
        )
    
    # "be X-Y" (between) -- the dash could be various Unicode dashes
    m = re.search(r'be\s+(\d+)\s*[-\u2013\u2014]\s*(\d+)', title_clean)
    if m:
        return Bucket(
            ticker=ticker, title=title_clean, bucket_type="between",
            low=int(m.group(1)), high=int(m.group(2))
        )
    
    # ---- Strategy 2: Parse from ticker ------------------------------------
    # Ticker format: KXHIGHNY-26FEB10-B38.5 or KXHIGHNY-26FEB10-T39
    
    ticker_parts = ticker.split("-")
    if len(ticker_parts) >= 3:
        bucket_code = ticker_parts[-1]  # e.g., "B38.5" or "T39"
        
        # B{midpoint} = between (midpoint - 0.5) and (midpoint + 0.5)
        m = re.match(r'B(\d+\.?\d*)', bucket_code)
        if m:
            midpoint = float(m.group(1))
            low = int(midpoint - 0.5)
            high = int(midpoint + 0.5)
            return Bucket(
                ticker=ticker, title=title_clean, bucket_type="between",
                low=low, high=high
            )
        
        # T{threshold} = above or below (need title to determine direction)
        m = re.match(r'T(\d+\.?\d*)', bucket_code)
        if m:
            threshold = int(float(m.group(1)))
            # Check title for direction clues
            if "<" in title_clean or "below" in title_clean.lower():
                return Bucket(
                    ticker=ticker, title=title_clean, bucket_type="below",
                    low=None, high=threshold
                )
            else:
                # Default: "greater than"
                return Bucket(
                    ticker=ticker, title=title_clean, bucket_type="above",
                    low=threshold, high=None
                )
    
    # ---- Strategy 3: Legacy simple formats --------------------------------
    m = re.match(r"(\d+)\s*to\s*(\d+)", title_clean)
    if m:
        return Bucket(
            ticker=ticker, title=title_clean, bucket_type="between",
            low=int(m.group(1)), high=int(m.group(2))
        )
    
    logger.warning("Could not parse bucket: ticker='%s' title='%s'", ticker, title_clean)
    return None


@dataclass
class Signal:
    """A trading signal when model probability diverges from market price."""
    bucket: Bucket
    model_prob: float      # Our computed probability
    market_price: float    # Kalshi yes price (in dollars, 0-1)
    edge: float            # model_prob - market_price (positive = underpriced)
    side: str              # "buy_yes" or "buy_no"
    suggested_price: int   # Price in cents for limit order


def compute_signals(buckets, market_prices, mu, sigma, min_edge=None):
    """
    Compare model probabilities to market prices.
    Generate signals where edge exceeds threshold.
    
    Args:
        buckets: List of parsed Bucket objects
        market_prices: Dict of ticker -> current yes price (0.0 to 1.0)
        mu: Model mean (forecast - bias)
        sigma: Model std dev
        min_edge: Minimum edge to generate signal
    
    Returns:
        List of Signal objects sorted by edge (strongest first)
    """
    if min_edge is None:
        min_edge = config.MIN_EDGE
    
    signals = []
    prob_sum = 0.0
    
    for bucket in buckets:
        prob = bucket.probability(mu, sigma)
        prob_sum += prob
        
        price = market_prices.get(bucket.ticker)
        if price is None:
            continue
        
        # Edge on YES side: we think it's more likely than market says
        yes_edge = prob - price
        # Edge on NO side: we think it's less likely than market says
        no_edge = (1.0 - prob) - (1.0 - price)  # = price - prob
        
        if yes_edge > min_edge:
            suggested = max(1, int(price * 100) - 1)
            signals.append(Signal(
                bucket=bucket,
                model_prob=prob,
                market_price=price,
                edge=yes_edge,
                side="buy_yes",
                suggested_price=suggested,
            ))
        elif no_edge > min_edge:
            no_price = 1.0 - price
            suggested = max(1, int(no_price * 100) - 1)
            signals.append(Signal(
                bucket=bucket,
                model_prob=prob,
                market_price=price,
                edge=no_edge,
                side="buy_no",
                suggested_price=suggested,
            ))
    
    # Sanity check: probabilities should sum to ~1.0
    if abs(prob_sum - 1.0) > 0.05:
        logger.warning("Bucket probabilities sum to %.3f (expected ~1.0)", prob_sum)
    
    signals.sort(key=lambda s: s.edge, reverse=True)
    
    return signals


def compute_fee(price_cents, count, is_maker=True):
    """
    Compute Kalshi fee in dollars.
    
    Formula: roundup(multiplier * count * price * (1-price))
    where price is in [0, 1] range.
    
    Maker fees are 4x cheaper than taker fees.
    """
    p = price_cents / 100.0
    multiplier = config.MAKER_FEE_MULTIPLIER if is_maker else config.TAKER_FEE_MULTIPLIER
    fee_cents = math.ceil(multiplier * count * p * (1.0 - p) * 100)
    return fee_cents / 100.0


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("=" * 60)
    print("MODEL TEST -- NYC High Temp")
    print("=" * 60)
    
    # Test parsing REAL API titles
    print("\nParsing real API titles:")
    real_tests = [
        ("KXHIGHNY-26FEB10-T39",   "Will the **high temp in NYC** be >39\u00b0 on Feb 10, 2026?"),
        ("KXHIGHNY-26FEB10-T32",   "Will the **high temp in NYC** be <32\u00b0 on Feb 10, 2026?"),
        ("KXHIGHNY-26FEB10-B38.5", "Will the **high temp in NYC** be 38-39\u00b0 on Feb 10, 2026?"),
        ("KXHIGHNY-26FEB10-B36.5", "Will the **high temp in NYC** be 36-37\u00b0 on Feb 10, 2026?"),
        ("KXHIGHNY-26FEB10-B34.5", "Will the **high temp in NYC** be 34-35\u00b0 on Feb 10, 2026?"),
        ("KXHIGHNY-26FEB10-B32.5", "Will the **high temp in NYC** be 32-33\u00b0 on Feb 10, 2026?"),
    ]
    
    all_passed = True
    for ticker, title in real_tests:
        b = parse_bucket_title(ticker, title)
        if b:
            print("  OK %s: type=%s low=%s high=%s" % (ticker, b.bucket_type, b.low, b.high))
        else:
            print("  FAIL: %s" % ticker)
            all_passed = False
    
    if all_passed:
        print("  All 6 parsed successfully!")
    
    # Original probability test
    mu = 31.0
    sigma = 2.5
    
    test_buckets = [
        Bucket("B1", "27 or below", "below", None, 28),
        Bucket("B2", "28 to 29", "between", 28, 29),
        Bucket("B3", "30 to 31", "between", 30, 31),
        Bucket("B4", "32 to 33", "between", 32, 33),
        Bucket("B5", "34 to 35", "between", 34, 35),
        Bucket("B6", "36 or above", "above", 35, None),
    ]
    
    print("\nForecast: mu=%.1fF, sigma=%.1fF" % (mu, sigma))
    print("%-18s %10s %12s %8s" % ("Bucket", "P(model)", "Example mkt", "Edge"))
    print("-" * 52)
    
    fake_prices = {
        "B1": 0.03,
        "B2": 0.12,
        "B3": 0.53,
        "B4": 0.20,
        "B5": 0.08,
        "B6": 0.04,
    }
    
    total_p = 0.0
    for b in test_buckets:
        p = b.probability(mu, sigma)
        total_p += p
        mkt = fake_prices.get(b.ticker, 0)
        edge = p - mkt
        flag = " <-- SIGNAL" if abs(edge) > 0.08 else ""
        print("%-18s %10.3f %11.2fc %+7.3f%s" % (b.title, p, mkt, edge, flag))
    
    print("\nTotal probability: %.4f" % total_p)
    
    print("\nFee examples (10 contracts):")
    for price in [10, 25, 50, 75, 90]:
        maker = compute_fee(price, 10, is_maker=True)
        taker = compute_fee(price, 10, is_maker=False)
        print("  @ %dc: maker=$%.2f, taker=$%.2f" % (price, maker, taker))
    
    print("\nSignals (min_edge=%.2f):" % config.MIN_EDGE)
    signals = compute_signals(test_buckets, fake_prices, mu, sigma)
    for s in signals:
        print("  %s %s: edge=%+.3f, model=%.3f, mkt=%.2f" % (
            s.side, s.bucket.title, s.edge, s.model_prob, s.market_price))
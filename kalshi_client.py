"""
kalshi_client.py — Kalshi Exchange API Client.

Handles:
- Public endpoints (no auth): markets, events, orderbooks, series
- Authenticated endpoints: portfolio, orders, positions

Key insight from market data: Only KXHIGHNY (NYC high temp) has liquidity.
Everything else is $0 volume ghost towns.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests

from kalshi_auth import KalshiAuth
import config

logger = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Market:
    """A single Kalshi market (one bucket in a weather event)."""
    ticker: str              # e.g., "KXHIGHNY-26FEB09-B30-31"
    event_ticker: str        # e.g., "KXHIGHNY-26FEB09"
    title: str               # e.g., "30° to 31°"
    subtitle: str
    status: str              # "active", "closed", "settled"
    yes_bid: Optional[int]   # in cents (None if no bid)
    yes_ask: Optional[int]   # in cents
    no_bid: Optional[int]
    no_ask: Optional[int]
    last_price: Optional[int]  # in cents
    volume: int
    volume_24h: int
    open_interest: int
    close_time: Optional[str]
    expiration_time: Optional[str]
    result: Optional[str]    # "yes", "no", or None

    @property
    def yes_bid_dollars(self) -> Optional[float]:
        return self.yes_bid / 100.0 if self.yes_bid is not None else None

    @property
    def yes_ask_dollars(self) -> Optional[float]:
        return self.yes_ask / 100.0 if self.yes_ask is not None else None

    @property
    def midpoint_dollars(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 200.0
        return None

    @property
    def spread_cents(self) -> Optional[int]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None


@dataclass
class OrderBookLevel:
    price_cents: int
    quantity: int

    @property
    def price_dollars(self) -> float:
        return self.price_cents / 100.0


@dataclass
class OrderBook:
    """Orderbook for a single market."""
    yes_levels: list[OrderBookLevel] = field(default_factory=list)
    no_levels: list[OrderBookLevel] = field(default_factory=list)

    @property
    def best_yes_bid(self) -> Optional[int]:
        return self.yes_levels[0].price_cents if self.yes_levels else None

    @property
    def best_no_bid(self) -> Optional[int]:
        return self.no_levels[0].price_cents if self.no_levels else None

    @property
    def best_yes_ask(self) -> Optional[int]:
        """YES ask = 100 - best NO bid (binary market identity)."""
        return (100 - self.best_no_bid) if self.best_no_bid is not None else None

    @property
    def best_no_ask(self) -> Optional[int]:
        return (100 - self.best_yes_bid) if self.best_yes_bid is not None else None

    @property
    def spread_cents(self) -> Optional[int]:
        bid = self.best_yes_bid
        ask = self.best_yes_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None

    @property
    def total_yes_depth(self) -> int:
        return sum(l.quantity for l in self.yes_levels)

    @property
    def total_no_depth(self) -> int:
        return sum(l.quantity for l in self.no_levels)


@dataclass
class Series:
    """A Kalshi series (template for recurring events)."""
    ticker: str
    title: str
    category: str
    frequency: str
    fee_type: str          # "quadratic"
    fee_multiplier: int    # e.g., 7 → 0.07 taker fee
    contract_url: str
    volume: int


# ─── Client ──────────────────────────────────────────────────────────────────

class KalshiClient:
    """
    Kalshi REST API client.
    
    Public endpoints (markets, events, orderbooks) need NO auth.
    Portfolio/order endpoints need RSA-PSS signed headers.
    """

    def __init__(self, auth: Optional[KalshiAuth] = None):
        self.read_url = config.KALSHI_READ_URL    # Production (real data)
        self.trade_url = config.KALSHI_TRADE_URL   # Demo (safe trading)
        self.auth = auth
        self.session = requests.Session()
        # Set default timeout
        self.session.headers.update({"Accept": "application/json"})
        logger.info(f"KalshiClient initialized — read={self.read_url}, trade={self.trade_url}")

    def _public_get(self, path, params=None):
        """GET request. Uses auth headers if available (production requires it)."""
        url = "%s%s" % (self.read_url, path)
        headers = {}
        if self.auth:
            headers = self.auth.headers("GET", path)
        try:
            resp = self.session.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error("GET %s failed: %s", path, e)
            raise

    def _auth_get(self, path: str, params: dict = None) -> dict:
        """GET request to authenticated endpoint. Uses DEMO for safe testing."""
        if not self.auth:
            raise RuntimeError("Auth required but not configured")
        url = f"{self.trade_url}{path}"
        headers = self.auth.headers("GET", path)
        try:
            resp = self.session.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Auth GET {path} failed: {e}")
            raise

    def _auth_post(self, path: str, payload: dict) -> dict:
        """POST request to authenticated endpoint. Uses DEMO for safe testing."""
        if not self.auth:
            raise RuntimeError("Auth required but not configured")
        url = f"{self.trade_url}{path}"
        headers = self.auth.headers("POST", path)
        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Auth POST {path} failed: {e}")
            raise

    def _auth_delete(self, path: str) -> dict:
        """DELETE request to authenticated endpoint. Uses DEMO for safe testing."""
        if not self.auth:
            raise RuntimeError("Auth required but not configured")
        url = f"{self.trade_url}{path}"
        headers = self.auth.headers("DELETE", path)
        try:
            resp = self.session.delete(url, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Auth DELETE {path} failed: {e}")
            raise

    # ── Public Endpoints ─────────────────────────────────────────────────────

    def get_series(self, series_ticker: str) -> Series:
        """Get series metadata (fee structure, settlement sources, etc.)."""
        path = f"{config.KALSHI_API_PATH}/series/{series_ticker}"
        data = self._public_get(path)
        s = data.get("series", {})
        return Series(
            ticker=s.get("ticker", ""),
            title=s.get("title", ""),
            category=s.get("category", ""),
            frequency=s.get("frequency", ""),
            fee_type=s.get("fee_type", ""),
            fee_multiplier=s.get("fee_multiplier", 7),
            contract_url=s.get("contract_url", ""),
            volume=s.get("volume", 0),
        )

    def get_markets(self, series_ticker: str = "", event_ticker: str = "",
                    status: str = "open", limit: int = 100) -> list[Market]:
        """
        Fetch markets. Filter by series or event ticker.
        
        Note from API docs: status filter accepts 'open' but response
        returns 'active'. Both work.
        """
        path = f"{config.KALSHI_API_PATH}/markets"
        params = {"limit": limit}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker

        data = self._public_get(path, params=params)
        markets = []
        for m in data.get("markets", []):
            markets.append(Market(
                ticker=m.get("ticker", ""),
                event_ticker=m.get("event_ticker", ""),
                title=m.get("title", ""),
                subtitle=m.get("subtitle", ""),
                status=m.get("status", ""),
                yes_bid=m.get("yes_bid"),
                yes_ask=m.get("yes_ask"),
                no_bid=m.get("no_bid"),
                no_ask=m.get("no_ask"),
                last_price=m.get("last_price"),
                volume=m.get("volume", 0),
                volume_24h=m.get("volume_24h", 0),
                open_interest=m.get("open_interest", 0),
                close_time=m.get("close_time"),
                expiration_time=m.get("expiration_time"),
                result=m.get("result"),
            ))
        return markets

    def get_event(self, event_ticker: str, with_nested_markets: bool = True) -> dict:
        """Get event with all its markets."""
        path = f"{config.KALSHI_API_PATH}/events/{event_ticker}"
        params = {}
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        return self._public_get(path, params=params)

    def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """
        Get order book for a specific market ticker.
        
        Response format (from docs):
        {
            "orderbook": {
                "yes": [[price_cents, quantity], ...],  # sorted best→worst
                "no": [[price_cents, quantity], ...]
            }
        }
        """
        path = f"{config.KALSHI_API_PATH}/markets/{ticker}/orderbook"
        params = {"depth": depth}
        data = self._public_get(path, params=params)
        ob = data.get("orderbook", {})

        # Demo API can return None instead of [] for empty sides
        raw_yes = ob.get("yes") or []
        raw_no = ob.get("no") or []

        yes_levels = [
            OrderBookLevel(price_cents=int(level[0]), quantity=int(level[1]))
            for level in raw_yes
            if len(level) >= 2
        ]
        no_levels = [
            OrderBookLevel(price_cents=int(level[0]), quantity=int(level[1]))
            for level in raw_no
            if len(level) >= 2
        ]

        # Should already be sorted best-first from API, but ensure it
        yes_levels.sort(key=lambda l: l.price_cents, reverse=True)
        no_levels.sort(key=lambda l: l.price_cents, reverse=True)

        return OrderBook(yes_levels=yes_levels, no_levels=no_levels)

    # ── Authenticated Endpoints ──────────────────────────────────────────────

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        path = f"{config.KALSHI_API_PATH}/portfolio/balance"
        data = self._auth_get(path)
        # Balance returned in cents
        return data.get("balance", 0) / 100.0

    def get_positions(self, event_ticker: str = "") -> list[dict]:
        """Get current positions, optionally filtered by event."""
        path = f"{config.KALSHI_API_PATH}/portfolio/positions"
        params = {}
        if event_ticker:
            params["event_ticker"] = event_ticker
        data = self._auth_get(path, params=params)
        return data.get("market_positions", [])

    def get_orders(self, status: str = "resting") -> list[dict]:
        """Get orders. Status: resting, canceled, executed."""
        path = f"{config.KALSHI_API_PATH}/portfolio/orders"
        params = {"status": status}
        data = self._auth_get(path, params=params)
        return data.get("orders", [])

    def place_order(self, ticker: str, side: str, action: str,
                    count: int, price_cents: int,
                    order_type: str = "limit") -> dict:
        """
        Place an order.

        Args:
            ticker: Market ticker (e.g., "KXHIGHNY-26FEB09-B30-31")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price_cents: Limit price in cents (1-99)
            order_type: "limit" (always for us — never market)

        Returns:
            Order response from API
        """
        path = f"{config.KALSHI_API_PATH}/portfolio/orders"
        payload = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "type": order_type,
            "count": count,
            "client_order_id": str(uuid.uuid4()),
        }
        # Set price on the appropriate side
        if side == "yes":
            payload["yes_price"] = price_cents
        else:
            payload["no_price"] = price_cents

        logger.info(
            f"ORDER: {action} {count}x {side} @ {price_cents}¢ on {ticker}"
        )
        result = self._auth_post(path, payload)
        order = result.get("order", {})
        logger.info(f"  → order_id={order.get('order_id', '?')}")
        return result

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting order."""
        path = f"{config.KALSHI_API_PATH}/portfolio/orders/{order_id}"
        logger.info(f"CANCEL: order_id={order_id}")
        return self._auth_delete(path)


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("KALSHI CLIENT TEST")
    print("=" * 60)

    # No auth needed for public endpoints
    auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
    client = KalshiClient(auth=auth)

    # 1. Get series info
    print("\n[1] Fetching KXHIGHNY series...")
    try:
        series = client.get_series(config.SERIES_TICKER)
        print(f"  Title: {series.title}")
        print(f"  Category: {series.category}")
        print(f"  Fee type: {series.fee_type}, multiplier: {series.fee_multiplier}")
        print(f"  Total volume: {series.volume}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Get today's markets
    print(f"\n[2] Fetching open markets for {config.SERIES_TICKER}...")
    try:
        markets = client.get_markets(series_ticker=config.SERIES_TICKER, status="open")
        if not markets:
            print("  No open markets found. Trying status='active'...")
            markets = client.get_markets(series_ticker=config.SERIES_TICKER, status="active")
        if not markets:
            print("  No markets found at all. Trying without status filter...")
            markets = client.get_markets(series_ticker=config.SERIES_TICKER, status="")

        for m in markets:
            bid = f"{m.yes_bid}¢" if m.yes_bid is not None else "---"
            ask = f"{m.yes_ask}¢" if m.yes_ask is not None else "---"
            print(f"  {m.ticker}: {m.title}")
            print(f"    Bid={bid}  Ask={ask}  Vol={m.volume}  Status={m.status}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. Get orderbook for first market
    if markets:
        print(f"\n[3] Fetching orderbook for {markets[0].ticker}...")
        try:
            ob = client.get_orderbook(markets[0].ticker)
            print(f"  YES bids ({len(ob.yes_levels)} levels):")
            for level in ob.yes_levels[:3]:
                print(f"    {level.price_cents}¢ × {level.quantity}")
            print(f"  NO bids ({len(ob.no_levels)} levels):")
            for level in ob.no_levels[:3]:
                print(f"    {level.price_cents}¢ × {level.quantity}")
            print(f"  Spread: {ob.spread_cents}¢")
        except Exception as e:
            print(f"  ERROR: {e}")

    # 4. Test auth (balance check)
    print(f"\n[4] Testing authenticated endpoint (balance)...")
    try:
        auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
        auth_client = KalshiClient(auth=auth)
        balance = auth_client.get_balance()
        print(f"  Balance: ${balance:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
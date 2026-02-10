import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import config
from arb_scanner import scan_bucket_arbitrage


@dataclass
class FakeMarket:
    ticker: str
    event_ticker: str
    yes_bid: int | None
    yes_ask: int | None
    no_bid: int | None
    no_ask: int | None
    volume_24h: int = 100


class TestArbScanner(unittest.TestCase):
    def test_detect_buy_all_yes_and_within_bucket(self):
        markets = [
            FakeMarket("KXHIGHNY-26FEB11-B36.5", "KXHIGHNY-26FEB11", 28, 30, 74, 72, 120),
            FakeMarket("KXHIGHNY-26FEB11-B38.5", "KXHIGHNY-26FEB11", 29, 30, 73, 71, 110),
            FakeMarket("KXHIGHNY-26FEB11-B40.5", "KXHIGHNY-26FEB11", 27, 30, 75, 73, 90),
        ]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "arb_opportunities.jsonl"
            with patch.object(config, "ARB_OPPORTUNITIES_PATH", out_path):
                rows = scan_bucket_arbitrage(
                    series_ticker="KXHIGHNY",
                    kalshi_client=object(),
                    markets=markets,
                )

            self.assertTrue(out_path.exists())
            file_rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), len(file_rows))
            self.assertTrue(any(r["type"] == "buy_all_yes" for r in rows))
            self.assertTrue(any(r["type"] == "within_bucket" for r in rows))

    def test_detect_stale_price(self):
        markets = [
            FakeMarket("KXHIGHNY-26FEB12-B36.5", "KXHIGHNY-26FEB12", 52, 50, 48, 50, 80),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "arb_opportunities.jsonl"
            with patch.object(config, "ARB_OPPORTUNITIES_PATH", out_path):
                rows = scan_bucket_arbitrage(
                    series_ticker="KXHIGHNY",
                    kalshi_client=object(),
                    markets=markets,
                )

            self.assertTrue(any(r["type"] == "stale_price" for r in rows))


if __name__ == "__main__":
    unittest.main()

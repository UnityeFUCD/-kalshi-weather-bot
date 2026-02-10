import unittest

from model import Bucket
from risk import RiskManager


class TestRiskCorrelation(unittest.TestCase):
    def test_no_existing_positions_allows(self):
        rm = RiskManager()
        b = Bucket(
            ticker="KXHIGHNY-26FEB11-B34.5",
            title="34 to 35",
            bucket_type="between",
            low=34,
            high=35,
        )
        new_trade = {
            "ticker": b.ticker,
            "event_ticker": "KXHIGHNY-26FEB11",
            "side": "buy_yes",
            "entry_price_cents": 40,
            "contracts": 10,
            "bucket": b,
        }
        out = rm.check_position_correlation(new_trade, [], model_mu=35.0, model_sigma=1.0)
        self.assertEqual(out["action"], "allow")
        self.assertIsNone(out["correlation"])

    def test_identical_trade_blocks(self):
        rm = RiskManager()
        b = Bucket(
            ticker="KXHIGHNY-26FEB11-B34.5",
            title="34 to 35",
            bucket_type="between",
            low=34,
            high=35,
        )
        existing = [{
            "ticker": b.ticker,
            "event_ticker": "KXHIGHNY-26FEB11",
            "side": "buy_yes",
            "entry_price_cents": 40,
            "contracts": 10,
            "bucket": b,
        }]
        new_trade = {
            "ticker": b.ticker,
            "event_ticker": "KXHIGHNY-26FEB11",
            "side": "buy_yes",
            "entry_price_cents": 41,
            "contracts": 8,
            "bucket": b,
        }
        out = rm.check_position_correlation(new_trade, existing, model_mu=35.0, model_sigma=1.0)
        self.assertEqual(out["action"], "block")
        self.assertIsNotNone(out["correlation"])
        self.assertGreater(out["correlation"], 0.8)

    def test_opposite_side_same_bucket_not_blocked(self):
        rm = RiskManager()
        b = Bucket(
            ticker="KXHIGHNY-26FEB11-B34.5",
            title="34 to 35",
            bucket_type="between",
            low=34,
            high=35,
        )
        existing = [{
            "ticker": b.ticker,
            "event_ticker": "KXHIGHNY-26FEB11",
            "side": "buy_yes",
            "entry_price_cents": 40,
            "contracts": 10,
            "bucket": b,
        }]
        new_trade = {
            "ticker": b.ticker,
            "event_ticker": "KXHIGHNY-26FEB11",
            "side": "buy_no",
            "entry_price_cents": 60,
            "contracts": 10,
            "bucket": b,
        }
        out = rm.check_position_correlation(new_trade, existing, model_mu=35.0, model_sigma=1.0)
        self.assertIn(out["action"], ("allow", "warn"))
        if out["correlation"] is not None:
            self.assertLess(out["correlation"], 0.8)


if __name__ == "__main__":
    unittest.main()

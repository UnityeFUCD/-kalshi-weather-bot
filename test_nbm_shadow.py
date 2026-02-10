import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import config
from market_registry import get_market
from model import Bucket
import nbm_shadow


class TestNBMShadow(unittest.TestCase):
    def test_compute_nbm_bucket_probs_complements(self):
        buckets = [
            Bucket(
                ticker="TEST-LOW",
                title="Will temp be <32?",
                bucket_type="below",
                low=None,
                high=32,
            ),
            Bucket(
                ticker="TEST-HIGH",
                title="Will temp be >31?",
                bucket_type="above",
                low=31,
                high=None,
            ),
        ]

        probs = nbm_shadow.compute_nbm_bucket_probs(40.0, 2.0, buckets)
        self.assertEqual(set(probs.keys()), {"TEST-LOW", "TEST-HIGH"})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    @patch("nbm_shadow._http_get_json")
    def test_fetch_nbm_forecast_uses_nbm_percentiles(self, mock_get_json):
        target_date = date(2026, 2, 11)
        mock_get_json.return_value = {
            "daily": {
                "time": [target_date.isoformat()],
                "temperature_2m_max": [40.0],
                "temperature_2m_max_p10": [37.0],
                "temperature_2m_max_p25": [38.0],
                "temperature_2m_max_p50": [40.0],
                "temperature_2m_max_p75": [41.0],
                "temperature_2m_max_p90": [43.0],
            }
        }

        out = nbm_shadow.fetch_nbm_forecast(40.7831, -73.9712, target_date)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["source"], "open-meteo-nbm")
        self.assertAlmostEqual(out["nbm_mu"], 40.0)
        self.assertAlmostEqual(out["nbm_sigma"], (43.0 - 37.0) / 3.29, places=6)

    @patch("nbm_shadow._http_get_json")
    def test_fetch_nbm_forecast_falls_back_to_ensemble(self, mock_get_json):
        target_date = date(2026, 2, 11)

        def fake_get_json(url, params, timeout=20):
            if url == nbm_shadow.OPEN_METEO_FORECAST_URL:
                return {
                    "daily": {
                        "time": [target_date.isoformat()],
                        "temperature_2m_max": [40.0],
                    }
                }

            model_name = params.get("models")
            if model_name == "bom_access_global_ensemble":
                vals = [38.0, 39.0]
            elif model_name == "ecmwf_ifs025":
                vals = [40.0, 41.0]
            else:
                vals = [42.0, 43.0]

            return {
                "daily": {
                    "time": [target_date.isoformat()],
                    "temperature_2m_max_member01": [vals[0]],
                    "temperature_2m_max_member02": [vals[1]],
                }
            }

        mock_get_json.side_effect = fake_get_json

        out = nbm_shadow.fetch_nbm_forecast(40.7831, -73.9712, target_date)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["source"], "open-meteo-ensemble-fallback")
        self.assertAlmostEqual(out["nbm_mu"], 40.0)  # Keep NBM point forecast mean
        self.assertGreater(out["nbm_sigma"], 0.0)
        self.assertGreater(out["nbm_p90"], out["nbm_p10"])

    @patch("nbm_shadow.fetch_hrrr_tmax")
    @patch("nbm_shadow.fetch_nbm_forecast")
    def test_log_nbm_shadow_writes_jsonl(self, mock_fetch_nbm, mock_fetch_hrrr):
        target_date = date(2026, 2, 11)
        mock_fetch_nbm.return_value = {
            "nbm_mu": 39.5,
            "nbm_p10": 37.0,
            "nbm_p25": 38.0,
            "nbm_p50": 39.5,
            "nbm_p75": 40.5,
            "nbm_p90": 42.0,
            "nbm_sigma": (42.0 - 37.0) / 3.29,
            "fetched_at_utc": "2026-02-11T12:30:00Z",
            "source": "open-meteo-nbm",
        }
        mock_fetch_hrrr.return_value = {
            "hrrr_mu": 39.9,
            "hrrr_hours": [("2026-02-11T12:00", 39.9)],
            "hrrr_run_time": "2026-02-11T12:00",
            "fetched_at_utc": "2026-02-11T12:31:00Z",
        }

        buckets = [
            Bucket(
                ticker="TEST-LOW",
                title="Will temp be <32?",
                bucket_type="below",
                low=None,
                high=32,
            ),
            Bucket(
                ticker="TEST-HIGH",
                title="Will temp be >31?",
                bucket_type="above",
                low=31,
                high=None,
            ),
        ]
        market_prices = {
            "TEST-LOW": 0.10,
            "TEST-HIGH": 0.90,
        }
        mc = get_market("KXHIGHNY")
        self.assertIsNotNone(mc)
        assert mc is not None

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "nbm_shadow.jsonl"
            with patch.object(config, "NBM_SHADOW_PATH", out_path):
                rec = nbm_shadow.log_nbm_shadow(
                    target_date=target_date,
                    market_config=mc,
                    current_v2_mu=40.0,
                    current_v2_sigma=2.0,
                    buckets=buckets,
                    market_prices=market_prices,
                )

            self.assertIsNotNone(rec)
            self.assertTrue(out_path.exists())
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["series_ticker"], "KXHIGHNY")
            self.assertEqual(payload["target_date"], "2026-02-11")
            self.assertIn("v2_bucket_probs", payload)
            self.assertIn("nbm_bucket_probs", payload)
            self.assertIn("v2_best_edge", payload)
            self.assertIn("nbm_best_edge", payload)


if __name__ == "__main__":
    unittest.main()

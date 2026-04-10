from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from Python.data_engineering.kalshi_dataload import _filter_to_2pm_central, fetch_candles_using_market_times


def _ts(y, m, d, h, minute=0):
    return int(datetime(y, m, d, h, minute, tzinfo=timezone.utc).timestamp())


class TestKalshiDataload(unittest.TestCase):
    class _Resp:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = {} if payload is None else payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    def test_filters_to_last_point_at_or_before_2pm_central(self):
        df = pd.DataFrame(
            [
                # Winter (CST, UTC-6): 20:00 UTC = 2pm CT
                {"decision_key": "d1", "ticker": "K1", "candle_ts": _ts(2026, 1, 15, 19), "close_price": 0.10},
                {"decision_key": "d1", "ticker": "K1", "candle_ts": _ts(2026, 1, 15, 20), "close_price": 0.20},
                {"decision_key": "d1", "ticker": "K1", "candle_ts": _ts(2026, 1, 15, 21), "close_price": 0.30},
                # Summer (CDT, UTC-5): 19:00 UTC = 2pm CT
                {"decision_key": "d1", "ticker": "K1", "candle_ts": _ts(2026, 7, 15, 18), "close_price": 0.40},
                {"decision_key": "d1", "ticker": "K1", "candle_ts": _ts(2026, 7, 15, 19), "close_price": 0.50},
                {"decision_key": "d1", "ticker": "K1", "candle_ts": _ts(2026, 7, 15, 20), "close_price": 0.60},
            ]
        )

        out = _filter_to_2pm_central(df)

        self.assertEqual(list(out["candle_ts"]), [_ts(2026, 1, 15, 20), _ts(2026, 7, 15, 19)])
        self.assertEqual(list(out["close_price"]), [0.20, 0.50])

    def test_handles_dst_boundary_days(self):
        df = pd.DataFrame(
            [
                # Before DST starts (still CST)
                {"decision_key": "d2", "ticker": "K2", "candle_ts": _ts(2026, 3, 7, 20), "close_price": 0.11},
                {"decision_key": "d2", "ticker": "K2", "candle_ts": _ts(2026, 3, 7, 21), "close_price": 0.12},
                # DST day and after switch to CDT
                {"decision_key": "d2", "ticker": "K2", "candle_ts": _ts(2026, 3, 8, 19), "close_price": 0.21},
                {"decision_key": "d2", "ticker": "K2", "candle_ts": _ts(2026, 3, 8, 20), "close_price": 0.22},
            ]
        )

        out = _filter_to_2pm_central(df)

        self.assertEqual(list(out["candle_ts"]), [_ts(2026, 3, 7, 20), _ts(2026, 3, 8, 19)])

    @patch("Python.data_engineering.kalshi_dataload.time.sleep")
    @patch("Python.data_engineering.kalshi_dataload.fetch_market_times")
    @patch("Python.data_engineering.kalshi_dataload.requests.get")
    def test_fetch_candles_chunks_range_and_combines_results(self, mock_get, mock_market_times, mock_sleep):
        start_ts, end_ts = 100, 172900
        first_chunk_end = start_ts + 86400
        mock_market_times.return_value = (start_ts, end_ts)
        requested_ranges = []

        def _side_effect(url, params=None, timeout=15):
            requested_ranges.append((params["start_ts"], params["end_ts"]))
            return self._Resp(
                status_code=200,
                payload={"candlesticks": [{"end_period_ts": params["end_ts"], "price": {"close": 42}}]},
            )

        mock_get.side_effect = _side_effect

        out = fetch_candles_using_market_times("KXFEDDECISION-26JAN-H0")

        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["candlesticks"]), 2)
        self.assertEqual(requested_ranges, [(start_ts, first_chunk_end), (first_chunk_end, end_ts)])
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("Python.data_engineering.kalshi_dataload.time.sleep")
    @patch("Python.data_engineering.kalshi_dataload.fetch_market_times")
    @patch("Python.data_engineering.kalshi_dataload.requests.get")
    def test_fetch_candles_retries_after_429(self, mock_get, mock_market_times, mock_sleep):
        mock_market_times.return_value = (100, 3700)
        mock_get.side_effect = [
            self._Resp(status_code=429, payload={}),
            self._Resp(status_code=200, payload={"candlesticks": [{"end_period_ts": 3700, "price": {"close": 10}}]}),
        ]

        out = fetch_candles_using_market_times("KXFEDDECISION-26JAN-H0")

        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["candlesticks"]), 1)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()

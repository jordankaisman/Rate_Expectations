from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from Python.data_engineering.polymarket_dataload import _filter_to_2pm_central


def _ts(y, m, d, h, minute=0):
    return int(datetime(y, m, d, h, minute, tzinfo=timezone.utc).timestamp())


class TestPolymarketDataload(unittest.TestCase):
    def test_filters_to_last_point_at_or_before_2pm_central(self):
        df = pd.DataFrame(
            [
                # Winter (CST, UTC-6): 20:00 UTC = 2pm CT
                {"event_slug": "fed-a", "market_id": "m1", "observed_ts": _ts(2026, 1, 15, 19), "observed_day": "2026-01-15", "close_price": 0.10},
                {"event_slug": "fed-a", "market_id": "m1", "observed_ts": _ts(2026, 1, 15, 20), "observed_day": "2026-01-15", "close_price": 0.20},
                {"event_slug": "fed-a", "market_id": "m1", "observed_ts": _ts(2026, 1, 15, 21), "observed_day": "2026-01-15", "close_price": 0.30},
                # Summer (CDT, UTC-5): 19:00 UTC = 2pm CT
                {"event_slug": "fed-a", "market_id": "m1", "observed_ts": _ts(2026, 7, 15, 18), "observed_day": "2026-07-15", "close_price": 0.40},
                {"event_slug": "fed-a", "market_id": "m1", "observed_ts": _ts(2026, 7, 15, 19), "observed_day": "2026-07-15", "close_price": 0.50},
                {"event_slug": "fed-a", "market_id": "m1", "observed_ts": _ts(2026, 7, 15, 20), "observed_day": "2026-07-15", "close_price": 0.60},
            ]
        )

        out = _filter_to_2pm_central(df)

        self.assertEqual(list(out["observed_ts"]), [_ts(2026, 1, 15, 20), _ts(2026, 7, 15, 19)])
        self.assertEqual(list(out["close_price"]), [0.20, 0.50])
        self.assertEqual(list(out["observed_day"]), ["2026-01-15", "2026-07-15"])

    def test_handles_dst_boundary_days(self):
        df = pd.DataFrame(
            [
                # Before DST starts (still CST)
                {"event_slug": "fed-b", "market_id": "m2", "observed_ts": _ts(2026, 3, 7, 20), "observed_day": "2026-03-07", "close_price": 0.11},
                {"event_slug": "fed-b", "market_id": "m2", "observed_ts": _ts(2026, 3, 7, 21), "observed_day": "2026-03-07", "close_price": 0.12},
                # DST day and after switch to CDT
                {"event_slug": "fed-b", "market_id": "m2", "observed_ts": _ts(2026, 3, 8, 19), "observed_day": "2026-03-08", "close_price": 0.21},
                {"event_slug": "fed-b", "market_id": "m2", "observed_ts": _ts(2026, 3, 8, 20), "observed_day": "2026-03-08", "close_price": 0.22},
            ]
        )

        out = _filter_to_2pm_central(df)

        self.assertEqual(list(out["observed_ts"]), [_ts(2026, 3, 7, 20), _ts(2026, 3, 8, 19)])
        self.assertEqual(list(out["observed_day"]), ["2026-03-07", "2026-03-08"])


if __name__ == "__main__":
    unittest.main()

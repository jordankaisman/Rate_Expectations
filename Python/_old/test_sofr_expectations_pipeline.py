import unittest
from datetime import date, timedelta

from Python.data_engineering.sofr_expectations_pipeline import (
    build_curve,
    build_monthly_expectations,
    discretize_changes,
    map_meetings,
    run_daily,
)


class SOFRPipelineTests(unittest.TestCase):
    def setUp(self):
        self.market_data = {
            "valuation_date": "2026-01-15",
            "overnight_rate": 0.05,
            "ois_swaps": [
                {"tenor_months": 1, "rate": 0.05},
                {"tenor_months": 3, "rate": 0.05},
                {"tenor_months": 6, "rate": 0.05},
                {"tenor_months": 12, "rate": 0.05},
            ],
            "futures": [
                {"month": "2026-01", "price": 95.0},
                {"month": "2026-02", "price": 95.1},
            ],
        }

    def test_build_curve_returns_reasonable_average_rate(self):
        curve = build_curve(self.market_data)
        start = date(2026, 2, 1)
        end = date(2026, 3, 1)
        avg = curve.average_rate(start, end)
        self.assertGreater(avg, 0.0)
        self.assertAlmostEqual(avg, 0.05, places=3)

    def test_monthly_expectations_prefers_futures_when_available(self):
        curve = build_curve(self.market_data)
        monthly = build_monthly_expectations(curve, self.market_data["futures"])
        jan = next(r for r in monthly if r["month"] == "2026-01")
        mar = next(r for r in monthly if r["month"] == "2026-03")
        self.assertEqual(jan["source"], "futures")
        self.assertAlmostEqual(jan["expected_rate"], 0.05, places=8)
        self.assertEqual(mar["source"], "ois")

    def test_discretize_changes_matches_step_rule(self):
        expectations = [
            {
                "date": "2026-01-15",
                "meeting_date": "2026-03-18",
                "expected_change": 0.00125,
                "method": "raw_futures_ois",
            },
            {
                "date": "2026-01-15",
                "meeting_date": "2026-05-06",
                "expected_change": -0.00375,
                "method": "raw_futures_ois",
            },
        ]
        probs = discretize_changes(expectations)
        self.assertAlmostEqual(probs[0]["p_plus_25"], 0.5, places=8)
        self.assertAlmostEqual(probs[0]["p_0"], 0.5, places=8)
        self.assertAlmostEqual(probs[1]["p_minus_25"], 0.5, places=8)
        self.assertAlmostEqual(probs[1]["p_minus_50"], 0.5, places=8)

    def test_map_meetings_uses_between_meeting_windows(self):
        curve = build_curve(self.market_data)
        monthly = [
            {"date": "2026-01-15", "month": "2026-01", "expected_rate": 0.04, "source": "futures"},
            {"date": "2026-01-15", "month": "2026-02", "expected_rate": 0.05, "source": "futures"},
            {"date": "2026-01-15", "month": "2026-03", "expected_rate": 0.06, "source": "futures"},
        ]
        meetings = [date(2026, 2, 10), date(2026, 3, 20)]

        mapped = map_meetings(monthly, curve, meetings)

        first = mapped[0]
        second = mapped[1]
        jan_days = (date(2026, 2, 1) - date(2026, 1, 15)).days
        feb_pre_days = (date(2026, 2, 10) - date(2026, 2, 1)).days
        feb_post_days = (date(2026, 3, 1) - date(2026, 2, 10)).days
        mar_days = (date(2026, 3, 20) - date(2026, 3, 1)).days
        self.assertAlmostEqual(first["pre_rate"], (jan_days * 0.04 + feb_pre_days * 0.05) / (jan_days + feb_pre_days), places=8)
        self.assertAlmostEqual(first["post_rate"], (feb_post_days * 0.05 + mar_days * 0.06) / (feb_post_days + mar_days), places=8)
        self.assertAlmostEqual(second["pre_rate"], first["post_rate"], places=8)
        self.assertGreater(first["expected_change"], 0.0)

    def test_run_daily_emits_all_variants(self):
        meetings = [
            date(2026, 1, 28),
            date(2026, 3, 18),
            date(2026, 5, 6),
            date(2028, 2, 1),  # out of configured range
        ]
        results = run_daily(self.market_data, meetings)

        methods = {row["method"] for row in results["expectations"]}
        self.assertEqual(methods, {"raw_futures_ois", "ois_anchor_short", "ois_anchor_long"})

        for row in results["probabilities"]:
            total = row["p_minus_50"] + row["p_minus_25"] + row["p_0"] + row["p_plus_25"] + row["p_plus_50"]
            self.assertAlmostEqual(total, 1.0, places=8)

        # 3 in-range meetings x 3 variants
        self.assertEqual(len(results["policy_path"]), 9)


if __name__ == "__main__":
    unittest.main()
 
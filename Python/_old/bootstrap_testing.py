import unittest
from datetime import date, timedelta
import math
from pathlib import Path
import os
import sys

# Ensure repo root is on sys.path (nice for consistency when clicking Play)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Python.data_engineering.sofr_expectations_pipeline import (
    build_curve,
    build_monthly_expectations,
    discretize_changes,
    map_meetings,
    run_daily,
)
from Python.data_engineering.sofr_ois_expectations import (
    add_months,
    bootstrap_ois_discount_curve,
    fixed_leg_schedule,
    year_fraction_act360,
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
        self.assertEqual(jan["source"], "ois_calibrated")
        self.assertAlmostEqual(jan["expected_rate"], 0.05, places=8)
        self.assertEqual(mar["source"], "ois_calibrated")

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
        spot_days = 1
        jan_forward_days = (date(2026, 2, 1) - date(2026, 1, 16)).days
        feb_pre_days = (date(2026, 2, 11) - date(2026, 2, 1)).days
        feb_post_days = (date(2026, 3, 1) - date(2026, 2, 11)).days
        mar_days = (date(2026, 3, 20) - date(2026, 3, 1)).days
        forward_pre = (jan_forward_days * 0.04 + feb_pre_days * 0.05) / (jan_forward_days + feb_pre_days)
        self.assertEqual(first["effective_date"], "2026-02-11")
        self.assertEqual(first["spot_days"], spot_days)
        self.assertAlmostEqual(
            first["pre_rate"],
            (spot_days * first["spot_rate"] + (jan_forward_days + feb_pre_days) * forward_pre)
            / (spot_days + jan_forward_days + feb_pre_days),
            places=6,
        )
        self.assertAlmostEqual(first["post_rate"], (feb_post_days * 0.05 + mar_days * 0.06) / (feb_post_days + mar_days), places=8)
        self.assertEqual(second["effective_date"], "2026-03-21")
        self.assertEqual(second["spot_days"], 0)
        second_feb_days = (date(2026, 3, 1) - date(2026, 2, 10)).days
        second_mar_days = (date(2026, 3, 21) - date(2026, 3, 1)).days
        self.assertAlmostEqual(
            second["pre_rate"],
            (second_feb_days * 0.05 + second_mar_days * 0.06) / (second_feb_days + second_mar_days),
            places=8,
        )
        self.assertGreater(first["expected_change"], 0.0)

    def test_step_forward_is_flat_between_meetings(self):
        curve = build_curve(self.market_data, meetings=[date(2026, 2, 10), date(2026, 3, 20)])
        self.assertAlmostEqual(
            curve.instantaneous_forward(date(2026, 1, 25)),
            curve.instantaneous_forward(date(2026, 2, 5)),
            places=10,
        )
        self.assertNotAlmostEqual(
            curve.instantaneous_forward(date(2026, 2, 5)),
            curve.instantaneous_forward(date(2026, 2, 20)),
            places=6,
        )

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

    def test_bootstrap_deposit_discount_factor_formula(self):
        valuation_date = date(2026, 1, 15)
        maturity = valuation_date + timedelta(days=1)
        rate = 0.0366
        _, dfs = bootstrap_ois_discount_curve(
            valuation_date,
            [{"maturity_date": maturity, "rate": rate, "kind": "deposit", "maturity_months": 0}],
        )
        accrual = year_fraction_act360(valuation_date, maturity)
        expected_df = 1.0 / (1.0 + rate * accrual)
        self.assertAlmostEqual(dfs[-1], expected_df, places=14)

    def test_bootstrap_reprices_par_swap_and_is_monotone(self):
        valuation_date = date(2026, 1, 15)
        flat_rate = 0.04
        instruments = []
        for months in range(1, 6):
            maturity = add_months(valuation_date, months)
            accrual = year_fraction_act360(valuation_date, maturity)
            target_df = math.exp(-flat_rate * accrual)
            deposit_rate = (1.0 / target_df - 1.0) / accrual
            instruments.append(
                {"maturity_date": maturity, "rate": deposit_rate, "kind": "deposit", "maturity_months": months}
            )

        maturity_6m = add_months(valuation_date, 6)
        pay_dates, accruals = fixed_leg_schedule(valuation_date, maturity_6m, 6)
        annuity = sum(
            a * math.exp(-flat_rate * year_fraction_act360(valuation_date, d))
            for a, d in zip(accruals, pay_dates)
        )
        par_rate_6m = (1.0 - math.exp(-flat_rate * year_fraction_act360(valuation_date, maturity_6m))) / annuity
        instruments.append(
            {"maturity_date": maturity_6m, "rate": par_rate_6m, "kind": "ois_swap", "maturity_months": 6}
        )

        node_dates, node_dfs = bootstrap_ois_discount_curve(valuation_date, instruments)
        self.assertTrue(all(node_dfs[i] <= node_dfs[i - 1] for i in range(1, len(node_dfs))))

        df_6m = node_dfs[node_dates.index(maturity_6m)]
        expected_df_6m = math.exp(-flat_rate * year_fraction_act360(valuation_date, maturity_6m))
        self.assertAlmostEqual(df_6m, expected_df_6m, places=10)

    def test_build_curve_discount_factors_are_monotone(self):
        curve = build_curve(self.market_data)
        self.assertTrue(
            all(curve.discount_factors[i] <= curve.discount_factors[i - 1] for i in range(1, len(curve.discount_factors)))
        )


if __name__ == "__main__":
    unittest.main()
 

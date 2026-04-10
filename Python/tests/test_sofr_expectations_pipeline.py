"""Tests for bootstrapped OIS curve integration and diagnostics."""
from __future__ import annotations

import math
import unittest
from datetime import date
from pathlib import Path
import sys
import tempfile
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Python.data_engineering.sofr_ois_expectations import (
    add_months,
    bootstrap_ois_discount_curve,
    check_curve_parity,
    fixed_leg_schedule,
    year_fraction_act360,
)
from Python.data_engineering.sofr_expectations_pipeline import OISCurve, build_curve
from Python.data_engineering.sofr_expectations_pipeline import merge_sofr_into_prediction_csv
from Python.data_engineering.sofr_ois_expectations import OISForwardChain, SR1ChainEstimator


class TestDepositDF(unittest.TestCase):
    """test_deposit_df: DF = 1 / (1 + r * a) for deposit instruments."""

    def test_overnight_deposit(self):
        val = date(2026, 1, 15)
        maturity = date(2026, 1, 16)
        rate = 0.0366
        _, dfs = bootstrap_ois_discount_curve(
            val,
            [{"maturity_date": maturity, "rate": rate, "kind": "deposit", "maturity_months": 0}],
        )
        a = year_fraction_act360(val, maturity)
        expected = 1.0 / (1.0 + rate * a)
        self.assertAlmostEqual(dfs[-1], expected, places=14)

    def test_one_week_deposit(self):
        val = date(2026, 1, 15)
        maturity = date(2026, 1, 22)  # 7 days
        rate = 0.0350
        _, dfs = bootstrap_ois_discount_curve(
            val,
            [{"maturity_date": maturity, "rate": rate, "kind": "deposit", "maturity_months": 0}],
        )
        a = year_fraction_act360(val, maturity)
        expected = 1.0 / (1.0 + rate * a)
        self.assertAlmostEqual(dfs[-1], expected, places=14)


class TestParReproduction(unittest.TestCase):
    """test_par_reproduction: bootstrapped curve reproduces input swap par rates."""

    def _make_flat_instruments(self, valuation_date: date, flat_rate: float, maturities_months: list[int]):
        """Create par-rate deposit and swap instruments consistent with a flat OIS curve."""
        instruments = []
        for months in maturities_months:
            maturity = add_months(valuation_date, months)
            if months <= 1:
                accrual = year_fraction_act360(valuation_date, maturity)
                deposit_rate = (math.exp(flat_rate * accrual) - 1.0) / accrual
                instruments.append(
                    {"maturity_date": maturity, "rate": deposit_rate, "kind": "deposit", "maturity_months": months}
                )
            else:
                pay_dates, accruals = fixed_leg_schedule(valuation_date, maturity, months)
                annuity = sum(
                    a * math.exp(-flat_rate * year_fraction_act360(valuation_date, d))
                    for a, d in zip(accruals, pay_dates)
                )
                df_mat = math.exp(-flat_rate * year_fraction_act360(valuation_date, maturity))
                par_rate = (1.0 - df_mat) / annuity
                instruments.append(
                    {"maturity_date": maturity, "rate": par_rate, "kind": "ois_swap", "maturity_months": months}
                )
        return instruments

    def test_par_swap_reproduced_within_tolerance(self):
        val = date(2026, 1, 15)
        flat_rate = 0.04
        instruments = self._make_flat_instruments(val, flat_rate, [1, 3, 6, 12])
        node_dates, node_dfs = bootstrap_ois_discount_curve(val, instruments)

        for inst in instruments:
            if inst["kind"] == "ois_swap":
                maturity = inst["maturity_date"]
                months = inst["maturity_months"]
                pay_dates, accruals = fixed_leg_schedule(val, maturity, months)

                from Python.data_engineering.sofr_ois_expectations import discount_factor_from_nodes
                annuity = sum(
                    a * discount_factor_from_nodes(d, node_dates, node_dfs, val)
                    for a, d in zip(accruals, pay_dates)
                )
                df_mat = discount_factor_from_nodes(maturity, node_dates, node_dfs, val)
                implied_rate = (1.0 - df_mat) / annuity
                self.assertAlmostEqual(implied_rate, inst["rate"], places=9,
                                       msg=f"Par rate not reproduced at {months}M")


class TestMonotonicity(unittest.TestCase):
    """test_monotonicity: node_dfs are non-increasing."""

    def test_build_curve_monotone(self):
        market_data = {
            "valuation_date": "2026-01-15",
            "overnight_rate": 0.05,
            "ois_swaps": [
                {"tenor_months": 1, "rate": 0.05},
                {"tenor_months": 3, "rate": 0.049},
                {"tenor_months": 6, "rate": 0.048},
                {"tenor_months": 12, "rate": 0.046},
                {"tenor_months": 24, "rate": 0.043},
            ],
        }
        curve = build_curve(market_data)
        dfs = curve.discount_factors
        for i in range(1, len(dfs)):
            self.assertLessEqual(dfs[i], dfs[i - 1],
                                 msg=f"Non-monotone at index {i}: {dfs[i]} > {dfs[i-1]}")

    def test_bootstrap_monotone_with_coupon_nodes(self):
        val = date(2026, 1, 15)
        instruments = []
        for months in [1, 6, 12, 24, 36]:
            maturity = add_months(val, months)
            instruments.append(
                {"maturity_date": maturity, "rate": 0.045, "kind": "ois_swap", "maturity_months": months}
            )
        _, node_dfs = bootstrap_ois_discount_curve(val, instruments)
        for i in range(1, len(node_dfs)):
            self.assertLessEqual(node_dfs[i], node_dfs[i - 1],
                                 msg=f"Non-monotone at index {i}")


class TestAverageRateAct360(unittest.TestCase):
    """test_average_rate_act360: average_rate uses ACT/360 consistently."""

    def test_flat_curve_average_rate(self):
        """On a flat curve, average_rate should return approximately the flat rate."""
        val = date(2026, 1, 15)
        flat_rate = 0.05
        # Build curve with flat 5% rate
        market_data = {
            "valuation_date": val.isoformat(),
            "overnight_rate": flat_rate,
            "ois_swaps": [
                {"tenor_months": m, "rate": flat_rate}
                for m in [1, 3, 6, 12]
            ],
        }
        curve = build_curve(market_data)
        start = date(2026, 2, 1)
        end = date(2026, 3, 1)
        avg = curve.average_rate(start, end)
        self.assertAlmostEqual(avg, flat_rate, places=3)

    def test_average_rate_uses_act360(self):
        """average_rate result matches manual ACT/360 calculation."""
        val = date(2026, 1, 15)
        # Create a simple two-node curve
        start_node = date(2026, 1, 16)
        end_node = date(2027, 1, 15)
        t1 = year_fraction_act360(val, start_node)
        t2 = year_fraction_act360(val, end_node)
        flat_rate = 0.04
        df1 = math.exp(-flat_rate * t1)
        df2 = math.exp(-flat_rate * t2)
        curve = OISCurve(val, [start_node, end_node], [df1, df2], "log_linear_discount")

        start = date(2026, 3, 1)
        end = date(2026, 6, 1)
        avg = curve.average_rate(start, end)
        t_s = year_fraction_act360(val, start)
        t_e = year_fraction_act360(val, end)
        expected = flat_rate  # flat curve => average rate == flat_rate
        self.assertAlmostEqual(avg, expected, places=6)

        # Verify the formula directly: -ln(df_end/df_start) / (t_end - t_start)
        df_s = curve.discount_factor(start)
        df_e = curve.discount_factor(end)
        manual = -math.log(df_e / df_s) / (t_e - t_s)
        self.assertAlmostEqual(avg, manual, places=12)


class TestCheckCurveParity(unittest.TestCase):
    """Tests for check_curve_parity diagnostic function."""

    def test_parity_small_error_on_flat_curve(self):
        val = date(2026, 1, 15)
        flat_rate = 0.04
        instruments = []
        for months in [1, 6, 12]:
            maturity = add_months(val, months)
            if months <= 1:
                a = year_fraction_act360(val, maturity)
                r = (math.exp(flat_rate * a) - 1.0) / a
                instruments.append(
                    {"maturity_date": maturity, "rate": r, "kind": "deposit", "maturity_months": months}
                )
            else:
                pay_dates, accruals = fixed_leg_schedule(val, maturity, months)
                annuity = sum(
                    a * math.exp(-flat_rate * year_fraction_act360(val, d))
                    for a, d in zip(accruals, pay_dates)
                )
                df_m = math.exp(-flat_rate * year_fraction_act360(val, maturity))
                r = (1.0 - df_m) / annuity
                instruments.append(
                    {"maturity_date": maturity, "rate": r, "kind": "ois_swap", "maturity_months": months}
                )

        node_dates, node_dfs = bootstrap_ois_discount_curve(val, instruments)
        _, max_error = check_curve_parity(val, instruments, node_dates, node_dfs)
        self.assertLess(max_error, 1e-6, msg=f"Max parity error too large: {max_error:.4e}")

    def test_parity_returns_per_instrument_results(self):
        val = date(2026, 1, 15)
        maturity = date(2026, 2, 15)
        rate = 0.05
        instruments = [{"maturity_date": maturity, "rate": rate, "kind": "deposit", "maturity_months": 1}]
        node_dates, node_dfs = bootstrap_ois_discount_curve(val, instruments)
        results, _ = check_curve_parity(val, instruments, node_dates, node_dfs)
        self.assertEqual(len(results), 1)
        self.assertIn("maturity_date", results[0])
        self.assertIn("quoted_rate", results[0])
        self.assertIn("implied_rate", results[0])
        self.assertIn("error", results[0])


class TestBuildCurveFallback(unittest.TestCase):
    """Test that build_curve falls back gracefully on bad data."""

    def test_fallback_on_bad_rate(self):
        """build_curve should not raise even if instruments cause bootstrap failure."""
        market_data = {
            "valuation_date": "2026-01-15",
            "overnight_rate": 0.05,
            "ois_swaps": [
                # Extremely large rate that would cause non-positive DF
                {"tenor_months": 12, "rate": 99.9},
            ],
        }
        # Should not raise
        try:
            curve = build_curve(market_data)
            # If it succeeds, discount factors should still be positive
            self.assertGreater(curve.discount_factors[-1], 0.0)
        except Exception:
            self.fail("build_curve raised an unexpected exception on bad data")


class TestTradeFocusedEstimators(unittest.TestCase):
    def test_sr1_chain_recursive_jump(self):
        estimator = SR1ChainEstimator(
            valuation_date=date(2026, 1, 10),
            monthly_rates={"2026-01": 0.0400, "2026-02": 0.0410, "2026-03": 0.0420},
            meetings=[date(2026, 2, 18), date(2026, 3, 18)],
        )
        rows = estimator.estimate()
        self.assertEqual(len(rows), 2)
        feb = rows[0]
        self.assertEqual(feb["meeting_date"], "2026-02-18")
        self.assertAlmostEqual(feb["r_pre"], 0.0400, places=10)
        self.assertAlmostEqual(feb["r_post"], 0.0428, places=10)
        self.assertAlmostEqual(feb["jump_sr1"], 0.0028, places=10)
        self.assertAlmostEqual(feb["sensitivity_sr1"], 10 / 28, places=10)

    def test_ois_forward_chain_jump(self):
        estimator = OISForwardChain(
            valuation_date=date(2026, 1, 1),
            ois_quotes=[
                {"tenor_months": 1, "rate": 0.0400},
                {"tenor_months": 2, "rate": 0.0410},
                {"tenor_months": 3, "rate": 0.0420},
            ],
            meetings=[date(2026, 3, 15)],
        )
        rows = estimator.estimate()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["meeting_date"], "2026-03-15")
        self.assertAlmostEqual(row["sensitivity_ois"], 16 / 31, places=10)
        self.assertNotEqual(row["jump_ois"], 0.0)

    def test_sr1_chain_filters_low_sensitivity_month(self):
        estimator = SR1ChainEstimator(
            valuation_date=date(2026, 1, 1),
            monthly_rates={"2026-01": 0.0500, "2026-02": 0.0510},
            meetings=[date(2026, 2, 27)],
        )
        rows = estimator.estimate()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["sensitivity_sr1"], 0.0, places=12)
        self.assertAlmostEqual(rows[0]["jump_sr1"], 0.0, places=12)

    def test_ois_forward_chain_filters_low_sensitivity_gap(self):
        estimator = OISForwardChain(
            valuation_date=date(2026, 1, 1),
            ois_quotes=[
                {"tenor_months": 1, "rate": 0.0400},
                {"tenor_months": 2, "rate": 0.0420},
            ],
            meetings=[date(2026, 2, 27)],
        )
        rows = estimator.estimate()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["sensitivity_ois"], 0.0, places=12)
        self.assertAlmostEqual(rows[0]["jump_ois"], 0.0, places=12)


class TestTradeMergeColumns(unittest.TestCase):
    def test_merge_adds_trade_jump_columns(self):
        prediction = pd.DataFrame(
            [{"decision_date": "2026-03-15", "observed_day_pst": "2026-01-01", "x": 1}]
        )
        sofr = pd.DataFrame(
            [
                {
                    "decision_date": "2026-03-15",
                    "observed_day_pst": "2026-01-01",
                    "sofr_method": "trade_sr1_chain",
                    "sofr_expected_change": 0.001,
                },
                {
                    "decision_date": "2026-03-15",
                    "observed_day_pst": "2026-01-01",
                    "sofr_method": "trade_ois_forward_chain",
                    "sofr_expected_change": -0.002,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_path = Path(tmp_dir) / "prediction.csv"
            sofr_path = Path(tmp_dir) / "sofr.csv"
            out_path = Path(tmp_dir) / "out.csv"
            prediction.to_csv(pred_path, index=False)
            sofr.to_csv(sofr_path, index=False)
            merge_sofr_into_prediction_csv(str(pred_path), str(sofr_path), str(out_path))
            out = pd.read_csv(out_path)
            self.assertIn("jump_sr1", out.columns)
            self.assertIn("jump_ois", out.columns)
            self.assertAlmostEqual(float(out.iloc[0]["jump_sr1"]), 0.001, places=10)
            self.assertAlmostEqual(float(out.iloc[0]["jump_ois"]), -0.002, places=10)

    def test_merge_excludes_legacy_non_trade_methods(self):
        prediction = pd.DataFrame(
            [{"decision_date": "2026-03-15", "observed_day_pst": "2026-01-01", "x": 1}]
        )
        sofr = pd.DataFrame(
            [
                {
                    "decision_date": "2026-03-15",
                    "observed_day_pst": "2026-01-01",
                    "sofr_method": "raw_futures_ois",
                    "sofr_expected_change": 0.010,
                },
                {
                    "decision_date": "2026-03-15",
                    "observed_day_pst": "2026-01-01",
                    "sofr_method": "trade_sr1_chain",
                    "sofr_expected_change": 0.001,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_path = Path(tmp_dir) / "prediction.csv"
            sofr_path = Path(tmp_dir) / "sofr.csv"
            out_path = Path(tmp_dir) / "out.csv"
            prediction.to_csv(pred_path, index=False)
            sofr.to_csv(sofr_path, index=False)
            merge_sofr_into_prediction_csv(str(pred_path), str(sofr_path), str(out_path))
            out = pd.read_csv(out_path)
            self.assertTrue((out["sofr_method"] == "trade_sr1_chain").all())


if __name__ == "__main__":
    unittest.main()

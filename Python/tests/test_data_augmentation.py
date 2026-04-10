from __future__ import annotations

import unittest

import pandas as pd

from Python.data_engineering.data_augmentation import add_probability_bps_columns, _build_butterfly, _build_curve_spread


class TestDataAugmentation(unittest.TestCase):
    def test_probability_columns_are_converted_to_bps(self):
        df = pd.DataFrame(
            [
                {
                    "polymarket_H50": 0.10,
                    "polymarket_C25": 0.20,
                    "kalshi_H25": 0.15,
                    "kalshi_C50+": 0.05,
                }
            ]
        )
        out = add_probability_bps_columns(df)
        self.assertAlmostEqual(float(out.iloc[0]["polymarket_H50_bps"]), 5.0, places=10)
        self.assertAlmostEqual(float(out.iloc[0]["polymarket_C25_bps"]), -5.0, places=10)
        self.assertAlmostEqual(float(out.iloc[0]["kalshi_H25_bps"]), 3.75, places=10)
        self.assertAlmostEqual(float(out.iloc[0]["kalshi_C50+_bps"]), -2.5, places=10)

    def test_butterfly_uses_standard_wing_belly_wing_weights(self):
        curve = pd.DataFrame(
            [
                {"observed_day_pst": "2026-01-01", "maturity_months": 0.0, "implied_rate_bps": 100.0},
                {"observed_day_pst": "2026-01-01", "maturity_months": 6.0, "implied_rate_bps": 120.0},
                {"observed_day_pst": "2026-01-01", "maturity_months": 12.0, "implied_rate_bps": 130.0},
            ]
        )
        out = _build_butterfly(curve, "maturity_months", "implied_rate_bps", "sr1_butterfly_bps")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["sr1_butterfly_bps"]), 10.0, places=10)

    def test_butterfly_supports_wider_high_variance_span(self):
        curve = pd.DataFrame(
            [
                {"observed_day_pst": "2026-01-01", "maturity_months": 3.0, "implied_rate_bps": 100.0},
                {"observed_day_pst": "2026-01-01", "maturity_months": 12.0, "implied_rate_bps": 150.0},
                {"observed_day_pst": "2026-01-01", "maturity_months": 24.0, "implied_rate_bps": 180.0},
            ]
        )
        out = _build_butterfly(
            curve,
            "maturity_months",
            "implied_rate_bps",
            "sr1_butterfly_bps_hvar",
            front_months=3.0,
            belly_months=12.0,
            back_months=24.0,
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["sr1_butterfly_bps_hvar"]), 20.0, places=10)

    def test_curve_spread_uses_dv01_style_weights(self):
        curve = pd.DataFrame(
            [
                {"observed_day_pst": "2026-01-01", "maturity_months": 2.0, "implied_rate_bps": 100.0},
                {"observed_day_pst": "2026-01-01", "maturity_months": 12.0, "implied_rate_bps": 130.0},
            ]
        )
        out = _build_curve_spread(curve, "maturity_months", "implied_rate_bps", "sr1_steepener_bps")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["sr1_steepener_bps"]), 470.0, places=10)

    def test_curve_spread_supports_aggressive_dv01_weighting(self):
        curve = pd.DataFrame(
            [
                {"observed_day_pst": "2026-01-01", "maturity_months": 1.0, "implied_rate_bps": 100.0},
                {"observed_day_pst": "2026-01-01", "maturity_months": 24.0, "implied_rate_bps": 180.0},
            ]
        )
        out = _build_curve_spread(
            curve,
            "maturity_months",
            "implied_rate_bps",
            "sr1_steepener_bps_hvar",
            short_months=1.0,
            long_months=24.0,
            dv01_weight_power=1.5,
        )
        self.assertEqual(len(out), 1)
        expected = (24.0 / 1.0) ** 1.5 * 100.0 - 180.0
        self.assertAlmostEqual(float(out.iloc[0]["sr1_steepener_bps_hvar"]), expected, places=8)


if __name__ == "__main__":
    unittest.main()

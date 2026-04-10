from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Python.data_engineering.sofr_ois_expectations import (
    SR1MatrixEstimator,
    OISMatrixEstimator,
    _business_days_between,
    parse_ois_tenor,
    tenor_to_maturity,
    year_fraction_act360,
)
from Python.data_engineering.sofr_expectations_pipeline import (
    merge_sofr_into_prediction_csv,
)


class TestMatrixEstimators(unittest.TestCase):
    def test_sr1_matrix_supports_sr3_and_returns_weights(self):
        estimator = SR1MatrixEstimator(
            valuation_date=date(2026, 3, 1),
            monthly_rates={
                "SR1:2026-03": 0.0400,
                "SR3:2026-03-18": 0.0410,
            },
            meetings=[date(2026, 3, 17)],
        )
        records, portfolio_weights = estimator.estimate()

        self.assertEqual(len(records), 1)
        self.assertIn("jump", records[0])
        weights = portfolio_weights["2026-03-17"]
        self.assertIn("SR1:2026-03", weights)
        self.assertIn("SR3:2026-03-18", weights)

    def test_ois_matrix_filters_irrelevant_maturities(self):
        estimator = OISMatrixEstimator(
            valuation_date=date(2026, 1, 1),
            ois_quotes=[
                {"tenor_months": 1.0, "rate": 0.0400},   # excluded: no overlap with meeting effective date
                {"tenor_months": 3.0, "rate": 0.0410},   # included
                {"tenor_months": 10.0, "rate": 0.0420},  # excluded: >9 months
            ],
            meetings=[date(2026, 2, 1)],
        )
        records, portfolio_weights = estimator.estimate()

        self.assertEqual(len(records), 1)
        weights = portfolio_weights["2026-02-01"]
        self.assertEqual(len(weights), 1)
        self.assertTrue(next(iter(weights)).startswith("OIS_3M_"))

    def test_business_days_use_us_holidays(self):
        # 2026-07-03 is observed Independence Day holiday; window is [start, end)
        count = _business_days_between(date(2026, 7, 2), date(2026, 7, 7))
        self.assertEqual(count, 2)

    def test_tenor_to_maturity_uses_t_plus_2_and_holiday_calendar(self):
        tenor = parse_ois_tenor("USDSROISON=")
        self.assertIsNotNone(tenor)
        # 2026-07-03 is observed holiday; T+2 from 2026-07-01 settles on 2026-07-06.
        maturity = tenor_to_maturity(date(2026, 7, 1), tenor)
        self.assertEqual(maturity, date(2026, 7, 7))

class TestMergeIncludesPortfolioWeights(unittest.TestCase):
    def test_merge_preserves_portfolio_weight_columns(self):
        prediction = pd.DataFrame(
            [{"decision_date": "2026-03-17", "observed_day_pst": "2026-01-01", "x": 1}]
        )
        sofr = pd.DataFrame(
            [
                {
                    "decision_date": "2026-03-17",
                    "observed_day_pst": "2026-01-01",
                    "jump_sr1": 0.001,
                    "jump_ois": -0.002,
                    "jump_sr1_portfolio_weights": json.dumps({"SR1:2026-03": 0.5}),
                    "jump_ois_portfolio_weights": json.dumps({"OIS_3M_0": 1.0}),
                }
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

            self.assertIn("jump_sr1_portfolio_weights", out.columns)
            self.assertIn("jump_ois_portfolio_weights", out.columns)
            self.assertEqual(out.iloc[0]["jump_sr1_portfolio_weights"], json.dumps({"SR1:2026-03": 0.5}))

    def test_merge_adds_grouped_prediction_assets_and_effr(self):
        prediction = pd.DataFrame(
            [
                {
                    "decision_date": "2026-03-15",
                    "observed_day_pst": "2026-01-01",
                    "polymarket_H25": 0.20,
                    "polymarket_H50": 0.10,
                    "polymarket_C25": 0.05,
                    "polymarket_C50": 0.02,
                    "kalshi_H25": 0.30,
                    "kalshi_H50": 0.10,
                    "kalshi_C25": 0.04,
                    "kalshi_C50+": 0.01,
                }
            ]
        )
        sofr = pd.DataFrame(
            [
                {
                    "decision_date": "2026-03-15",
                    "observed_day_pst": "2026-01-01",
                    "jump_sr1": 0.001,
                    "jump_ois": -0.002,
                    "jump_sr1_portfolio_weights": "{}",
                    "jump_ois_portfolio_weights": "{}",
                }
            ]
        )
        effr = pd.DataFrame(
            [
                {"Date_": "2026-01-01", "Settlement": 95.00, "LastTrdDate": "2026-02-27", "DSMnem": "A"},
                {"Date_": "2026-01-01", "Settlement": 94.90, "LastTrdDate": "2026-03-31", "DSMnem": "B"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_path = Path(tmp_dir) / "prediction.csv"
            sofr_path = Path(tmp_dir) / "sofr.csv"
            effr_path = Path(tmp_dir) / "effr.csv"
            out_path = Path(tmp_dir) / "out.csv"
            prediction.to_csv(pred_path, index=False)
            sofr.to_csv(sofr_path, index=False)
            effr.to_csv(effr_path, index=False)

            merge_sofr_into_prediction_csv(
                str(pred_path),
                str(sofr_path),
                str(out_path),
                effr_futures_path=str(effr_path),
            )
            out = pd.read_csv(out_path)
            row = out.iloc[0]

            self.assertAlmostEqual(row["jump_sr1_bps"], 10.0, places=10)
            self.assertAlmostEqual(row["jump_ois_bps"], -20.0, places=10)
            self.assertAlmostEqual(row["Poly_Hike"], 0.30, places=10)
            self.assertAlmostEqual(row["Poly_Cut"], 0.07, places=10)
            self.assertAlmostEqual(row["Poly_Hike_bps"], 10.0, places=10)
            self.assertAlmostEqual(row["Poly_Cut_bps"], -2.25, places=10)
            self.assertAlmostEqual(row["Kalshi_Hike_bps"], 12.5, places=10)
            self.assertAlmostEqual(row["Kalshi_Cut_bps"], -1.5, places=10)
            self.assertTrue(pd.notna(row["effr_expected_bps"]))


if __name__ == "__main__":
    unittest.main()

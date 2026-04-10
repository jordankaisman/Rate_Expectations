from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Python.stat_arb.Moment_PCA import PREDICTION_BPS_MAP
from Python.stat_arb.visuals import run_pca_visuals


class TestPCAVisuals(unittest.TestCase):
    def test_run_pca_visuals_generates_top3_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "panel.csv"
            out_dir = Path(tmpdir) / "figures"
            panel = pd.DataFrame(
                {
                    "decision_date": [pd.Timestamp("2026-01-28")] * 12,
                    "observed_day_pst": pd.date_range("2025-12-01", periods=12, freq="D"),
                    "jump_sr1": [0.001 + (i * 0.0001) for i in range(12)],
                    "jump_ois": [0.0012 + (i * 0.0001) for i in range(12)],
                    "effr_expected_bps": [5.0 + (i * 0.1) for i in range(12)],
                    "polymarket_H25": [0.20 + (i * 0.001) for i in range(12)],
                    "polymarket_C25": [0.10 + (i * 0.001) for i in range(12)],
                    "kalshi_H25": [0.25 + (i * 0.001) for i in range(12)],
                    "kalshi_C25": [0.12 + (i * 0.001) for i in range(12)],
                }
            )
            panel.to_csv(data_path, index=False)
            outputs = run_pca_visuals(
                path=str(data_path),
                output_dir=str(out_dir),
                panel_mode="prediction_moments",
                n_components=3,
            )
            self.assertIn("factor_scores", outputs)
            self.assertIn("loadings", outputs)
            self.assertIn("explained_variance", outputs)
            self.assertTrue((out_dir / "pca_explained_variance_top3.png").exists())
            self.assertTrue((out_dir / "pca_loadings_top3.png").exists())
            self.assertTrue((out_dir / "pca_factor_scores_top3.png").exists())

    def test_run_pca_visuals_prediction_all_excludes_raw_probability_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "panel.csv"
            out_dir = Path(tmpdir) / "figures"
            panel = pd.DataFrame(
                {
                    "decision_date": [pd.Timestamp("2026-01-28")] * 12,
                    "observed_day_pst": pd.date_range("2025-12-01", periods=12, freq="D"),
                    "jump_sr1": [0.001 + (i * 0.0001) for i in range(12)],
                    "jump_ois": [0.0012 + (i * 0.0001) for i in range(12)],
                    "effr_expected_bps": [5.0 + (i * 0.1) for i in range(12)],
                    "polymarket_H50": [0.10 + (i * 0.002) for i in range(12)],
                    "polymarket_C25": [0.15 + (i * 0.001) for i in range(12)],
                    "kalshi_H50": [0.20 + (i * 0.0015) for i in range(12)],
                    "kalshi_C25": [0.13 + (i * 0.0012) for i in range(12)],
                }
            )
            panel.to_csv(data_path, index=False)
            outputs = run_pca_visuals(
                path=str(data_path),
                output_dir=str(out_dir),
                panel_mode="prediction_all",
                n_components=3,
            )
            loading_assets = set(outputs["loadings"].index.tolist())
            self.assertIn("kalshi_H50_bps", loading_assets)
            self.assertNotIn("kalshi_H50", loading_assets)
            self.assertTrue(loading_assets.isdisjoint(set(PREDICTION_BPS_MAP.keys())))


if __name__ == "__main__":
    unittest.main()

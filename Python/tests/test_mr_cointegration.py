from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path
import os
import sys

# Ensure repo root is on sys.path (nice for consistency when clicking Play)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


from Python.stat_arb.Moment_PCA import (
    _cointegration_residuals,
    _rolling_bands_by_decision,
    StrategyConfig,
    build_asset_panel,
    filter_residuals,
    plot_factor_mimicking_returns,
    run_ou_strategy,
    run_pca,
    run_full_experiment,
)
from Python.stat_arb.variance_skew_strategies import (
    build_skew_arb_panel,
    build_variance_arb_panel,
    run_full_experiment as run_variance_skew_full_experiment,
    run_skew_arb_experiment,
    run_variance_arb_experiment,
)


class TestMRCointegrationWorkflow(unittest.TestCase):
    def _sample_panel(self) -> pd.DataFrame:
        days = pd.date_range("2026-01-01", periods=40, freq="D")
        base = np.linspace(0, 1, len(days))
        return pd.DataFrame(
            {
                "decision_date": pd.Timestamp("2026-02-20"),
                "observed_day_pst": days,
                "effr_expected_bps": 10 + base,
                "jump_sr1_bps": 11 + base * 1.1,
                "jump_ois_bps": 9 + base * 0.9,
                "polymarket_H25": 0.2 + base * 0.01,
                "polymarket_H50": 0.1 + base * 0.01,
                "polymarket_C25": 0.05,
                "polymarket_C50": 0.03,
                "kalshi_H25": 0.25 + base * 0.01,
                "kalshi_H50": 0.12 + base * 0.01,
                "kalshi_C25": 0.03,
                "kalshi_C50+": 0.02,
                "sr1_butterfly_bps": 0.1 + base * 0.15,
                "effr_butterfly_bps": 0.08 + base * 0.10,
                "ois_butterfly_bps": 0.12 + base * 0.14,
                "sr1_steepener_bps": 0.3 + base * 0.20,
                "effr_steepener_bps": 0.25 + base * 0.18,
                "ois_steepener_bps": 0.35 + base * 0.22,
            }
        )

    def test_build_panel_modes(self):
        df = self._sample_panel()
        expected = build_asset_panel(df, "prediction_expected")
        grouped = build_asset_panel(df, "prediction_grouped")
        full = build_asset_panel(df, "prediction_all")
        moments = build_asset_panel(df, "prediction_moments")
        self.assertIn("polymarket_expected_bps", expected.columns)
        self.assertIn("Poly_Hike_bps", grouped.columns)
        self.assertTrue(pd.notna(grouped.iloc[0]["Poly_Hike_bps"]))
        self.assertTrue(any(c.startswith("polymarket_") and c.endswith("_bps") for c in full.columns))
        self.assertIn("polymarket_variance_asset", moments.columns)
        self.assertIn("kalshi_variance_asset", moments.columns)

    def test_pca_filter_and_ou_strategy_runs(self):
        panel = build_asset_panel(self._sample_panel(), "prediction_grouped")
        _, _, residuals, r2 = run_pca(panel, n_components=3)
        tradable = filter_residuals(
            residuals,
            r2,
            min_r2=-1.0,
            max_rho=1.0,
            adf_alpha=1.0,
            max_half_life_days=999.0,
            variance_threshold=-1.0,
        )
        self.assertIsInstance(tradable, pd.DataFrame)
        kept = [c for c in tradable.columns if bool(tradable[c].any())]
        self.assertGreater(len(kept), 0)

        config = StrategyConfig(ou_window=10, no_trade_days_before_decision=2)
        trades, metrics = run_ou_strategy(panel, residuals, kept[:2], config, tradable=tradable[kept[:2]])
        self.assertIsInstance(trades, pd.DataFrame)
        self.assertIsInstance(metrics, pd.DataFrame)

    def test_run_pca_normalization_changes_factor_balance(self):
        days = pd.date_range("2026-01-01", periods=80, freq="D")
        t = np.linspace(0, 4 * np.pi, len(days))
        panel = pd.DataFrame(
            {
                "decision_date": pd.Timestamp("2026-03-20"),
                "observed_day_pst": days,
                "asset_small_scale": np.sin(t),
                "asset_large_scale": 1000.0 * np.cos(t),
            }
        )
        pca_raw, _, _, _ = run_pca(panel, n_components=2, normalize=False)
        pca_norm, _, _, _ = run_pca(panel, n_components=2, normalize=True)
        self.assertGreater(float(pca_raw.explained_variance_ratio_[0]), 0.99)
        self.assertLess(float(pca_norm.explained_variance_ratio_[0]), 0.8)

    def test_variance_arb_outputs(self):
        df = self._sample_panel()
        var_panel = build_variance_arb_panel(df)
        self.assertIn("polymarket_variance_asset", var_panel.columns)
        self.assertIn("sr1_butterfly_bps", var_panel.columns)

        config = StrategyConfig(ou_window=10, no_trade_days_before_decision=2, min_r2=-1.0)
        outputs = run_variance_arb_experiment(df, config=config, min_r2=0.0)
        self.assertIn("variance_metrics", outputs)
        self.assertIsInstance(outputs["variance_residuals"], pd.DataFrame)
        self.assertIn("variance_trades", outputs)
        self.assertEqual(len(outputs["variance_metrics"]), outputs["variance_residuals"].shape[1])

    def test_variance_and_skew_panels_daily_reindex_and_rates_only_ffill(self):
        decision_date = pd.Timestamp("2026-02-20")
        df = pd.DataFrame(
            {
                "decision_date": [decision_date] * 3,
                "observed_day_pst": pd.to_datetime(["2026-01-02", "2026-01-06", "2026-01-08"]),
                "polymarket_H25": [0.2, 0.2, 0.2],
                "polymarket_H50": [0.1, 0.1, 0.1],
                "polymarket_C25": [0.05, 0.05, 0.05],
                "polymarket_C50": [0.03, 0.03, 0.03],
                "kalshi_H25": [0.2, 0.2, 0.2],
                "kalshi_H50": [0.1, 0.1, 0.1],
                "kalshi_C25": [0.05, 0.05, 0.05],
                "kalshi_C50+": [0.03, 0.03, 0.03],
                "sr1_butterfly_bps": [1.0, np.nan, 2.0],
                "effr_butterfly_bps": [1.0, np.nan, 2.0],
                "ois_butterfly_bps": [1.0, np.nan, 2.0],
                "sr1_steepener_bps": [1.0, np.nan, 2.0],
                "effr_steepener_bps": [1.0, np.nan, 2.0],
                "ois_steepener_bps": [1.0, np.nan, 2.0],
            }
        )
        variance_panel = build_variance_arb_panel(df)
        skew_panel = build_skew_arb_panel(df)
        saturday_row = variance_panel[variance_panel["observed_day_pst"] == pd.Timestamp("2026-01-03")].iloc[0]
        tuesday_row = variance_panel[variance_panel["observed_day_pst"] == pd.Timestamp("2026-01-06")].iloc[0]
        self.assertEqual(float(saturday_row["sr1_butterfly_bps"]), 1.0)
        self.assertTrue(pd.isna(tuesday_row["sr1_butterfly_bps"]))
        self.assertTrue(pd.isna(saturday_row["polymarket_variance_asset"]))
        skew_saturday_row = skew_panel[skew_panel["observed_day_pst"] == pd.Timestamp("2026-01-03")].iloc[0]
        skew_tuesday_row = skew_panel[skew_panel["observed_day_pst"] == pd.Timestamp("2026-01-06")].iloc[0]
        self.assertEqual(float(skew_saturday_row["sr1_steepener_bps"]), 1.0)
        self.assertTrue(pd.isna(skew_tuesday_row["sr1_steepener_bps"]))

    def test_variance_and_skew_panels_can_use_high_variance_instruments(self):
        df = self._sample_panel()
        df["sr1_butterfly_bps_hvar"] = df["sr1_butterfly_bps"] * 10.0
        df["effr_butterfly_bps_hvar"] = df["effr_butterfly_bps"] * 10.0
        df["ois_butterfly_bps_hvar"] = df["ois_butterfly_bps"] * 10.0
        df["sr1_steepener_bps_hvar"] = df["sr1_steepener_bps"] * 10.0
        df["effr_steepener_bps_hvar"] = df["effr_steepener_bps"] * 10.0
        df["ois_steepener_bps_hvar"] = df["ois_steepener_bps"] * 10.0
        df["sr1_flattener_bps_hvar"] = -df["sr1_steepener_bps_hvar"]
        df["effr_flattener_bps_hvar"] = -df["effr_steepener_bps_hvar"]
        df["ois_flattener_bps_hvar"] = -df["ois_steepener_bps_hvar"]

        var_panel = build_variance_arb_panel(df, use_high_variance_instruments=True)
        skew_panel = build_skew_arb_panel(df, use_high_variance_instruments=True)
        self.assertIn("sr1_butterfly_bps_hvar", var_panel.columns)
        self.assertIn("sr1_steepener_bps_hvar", skew_panel.columns)

    def test_cointegration_residuals_support_normalization_and_rolling(self):
        days = pd.date_range("2026-01-01", periods=120, freq="D")
        decision_dates = np.where(np.arange(120) < 60, pd.Timestamp("2026-03-01"), pd.Timestamp("2026-05-01"))
        market = np.linspace(1.0, 8.0, len(days))
        pred = 1000.0 + (market * 30.0) + np.sin(np.arange(len(days)) / 3.0)
        pred[60:] = 5000.0 + (market[60:] * 120.0) + np.sin(np.arange(60, len(days)) / 3.0)
        panel = pd.DataFrame(
            {
                "decision_date": decision_dates,
                "observed_day_pst": days,
                "pred_asset": pred,
                "market_asset": market,
            }
        )
        residuals, r2 = _cointegration_residuals(
            panel=panel,
            prediction_assets=["pred_asset"],
            market_assets=["market_asset"],
            min_r2=-1.0,
            normalize_within_decision_date=True,
            rolling_window_days=30,
        )
        key = "pred_asset_vs_market_asset"
        self.assertIn(key, residuals.columns)
        self.assertIn(key, r2.index)
        self.assertGreater(residuals[key].notna().sum(), 20)

    def test_skew_arb_outputs(self):
        df = self._sample_panel()
        skew_panel = build_skew_arb_panel(df)
        self.assertIn("polymarket_positive_skew_asset", skew_panel.columns)
        self.assertIn("sr1_flattener_bps", skew_panel.columns)

        config = StrategyConfig(ou_window=10, no_trade_days_before_decision=2, min_r2=-1.0)
        outputs = run_skew_arb_experiment(df, config=config, min_r2=0.0)
        self.assertIn("skew_metrics", outputs)
        self.assertIsInstance(outputs["skew_residuals"], pd.DataFrame)
        self.assertEqual(len(outputs["skew_metrics"]), outputs["skew_residuals"].shape[1])

    def test_skew_arb_relaxes_r2_when_empty(self):
        df = self._sample_panel()
        for col in [
            "sr1_steepener_bps",
            "effr_steepener_bps",
            "ois_steepener_bps",
            "sr1_flattener_bps",
            "effr_flattener_bps",
            "ois_flattener_bps",
        ]:
            df[col] = np.nan
        outputs = run_skew_arb_experiment(df, config=StrategyConfig(ou_window=10), min_r2=0.99)
        self.assertIn("skew_min_r2_used", outputs)
        self.assertEqual(float(outputs["skew_min_r2_used"]["min_r2_used"].iloc[0]), 0.0)
        self.assertIsInstance(outputs["skew_residuals"], pd.DataFrame)

    def test_rolling_bands_exposes_z_score(self):
        panel = self._sample_panel()
        strategy = StrategyConfig(ou_window=10)
        ddf = panel[["decision_date", "observed_day_pst"]].copy()
        residual = pd.Series(np.linspace(0.0, 1.0, len(ddf)), index=ddf.index)
        bands = _rolling_bands_by_decision(ddf, residual, strategy)
        self.assertIn("z", bands.columns)
        self.assertGreater(bands["z"].notna().sum(), 0)

    def test_ou_strategy_does_not_carry_position_across_decision_dates(self):
        decision_a = pd.Timestamp("2026-03-01")
        decision_b = pd.Timestamp("2026-06-01")

        panel_a = pd.DataFrame(
            {
                "decision_date": [decision_a] * 12,
                "observed_day_pst": pd.date_range("2026-01-01", periods=12, freq="D"),
            }
        )
        panel_b = pd.DataFrame(
            {
                "decision_date": [decision_b] * 12,
                "observed_day_pst": pd.date_range("2026-01-13", periods=12, freq="D"),
            }
        )
        panel = pd.concat([panel_a, panel_b], ignore_index=True)

        # First decision date has enough variation to generate an entry near the end;
        # second decision date is flat and should not be used to close date-A positions.
        residual_values = [0.0, 0.2, 0.3, 0.35, 0.33, 0.4, 0.42, 0.41, 0.43, 0.45] + [3.0, 3.0] + [0.0] * 12
        residuals = pd.DataFrame({"test_asset": residual_values}, index=panel.index)

        trades, _ = run_ou_strategy(
            panel=panel,
            residuals=residuals,
            assets=["test_asset"],
            config=StrategyConfig(ou_window=10, entry_sigma=1.0, exit_sigma=0.5, no_trade_days_before_decision=0),
        )

        self.assertFalse(trades.empty)

        entries_a = trades[
            (trades["decision_date"] == decision_a)
            & (trades["event"].str.startswith("enter"))
        ]
        exits_b = trades[
            (trades["decision_date"] == decision_b)
            & (trades["event"].str.contains("exit"))
        ]
        self.assertGreater(len(entries_a), 0)
        self.assertEqual(len(exits_b), 0)
        panel_map = panel.groupby("observed_day_pst")["decision_date"].apply(set).to_dict()
        for row in trades.itertuples(index=False):
            self.assertIn(row.decision_date, panel_map[row.observed_day_pst])

    def test_factor_mimicking_returns_generation(self):
        panel = build_asset_panel(self._sample_panel(), "prediction_moments")
        pca, _, _, _ = run_pca(panel, n_components=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            returns = plot_factor_mimicking_returns(panel, pca, output_path=str(Path(tmpdir) / "factors.png"))
            self.assertTrue((Path(tmpdir) / "factors.png").exists())
            self.assertTrue({"decision_date", "F1"}.issubset(returns.columns))

    def test_full_experiment_diagnostics_and_plot_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_full_experiment(
                path="Data/Merged/Prediction_all_augmented.csv",
                config=StrategyConfig(panel_mode="prediction_moments", n_components=2, pca_rolling_window_days=30, min_r2=0.1, ou_window=10),
                output_dir=tmpdir,
                generate_plots=False,
            )
            self.assertIn("model_fit_table", outputs)
            self.assertIn("performance_table", outputs)
            self.assertIn("half_life_table", outputs)
            self.assertIn("walkforward_plots", outputs)
            self.assertIn("cumulative_pnl", outputs)
            self.assertIn("factor_mimicking_returns", outputs)
            self.assertIn("all_strategy_trades", outputs)
            self.assertFalse(outputs["performance_table"].empty)
            self.assertTrue(
                {
                    "Decision_Date",
                    "Strategy_Type",
                    "R_Squared",
                    "ADF_Statistic",
                    "ADF_p_value",
                    "OU_Kappa",
                    "ou_windows_used",
                    "ou_kappa_mean",
                }.issubset(outputs["model_fit_table"].columns)
            )
            self.assertTrue({"Half_Life_Days", "Time_To_FOMC_Days", "Risk_Flag"}.issubset(outputs["half_life_table"].columns))
            level_entry_trades = outputs["trades"][outputs["trades"]["event"].str.startswith("enter")]
            self.assertGreaterEqual(level_entry_trades["observed_day_pst"].dt.to_period("M").nunique(), 2)
            self.assertTrue((Path(tmpdir) / "cumulative_pnl_by_strategy.png").exists())
            self.assertTrue((Path(tmpdir) / "results" / "all_strategy_trades.csv").exists())

    def test_all_strategy_trades_have_balanced_events_per_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_full_experiment(
                path="Data/Merged/Prediction_all_augmented.csv",
                config=StrategyConfig(panel_mode="prediction_moments", n_components=2, min_r2=0.1, ou_window=10),
                output_dir=tmpdir,
                generate_plots=False,
            )
            all_trades = outputs["all_strategy_trades"].sort_values(
                ["strategy_type", "asset", "decision_date", "observed_day_pst"]
            )
            self.assertFalse(all_trades.empty)
            for _, group in all_trades.groupby(["strategy_type", "asset", "decision_date"]):
                open_positions = 0
                for event in group["event"]:
                    if event.startswith("enter"):
                        open_positions += 1
                    elif "exit" in event:
                        self.assertGreater(open_positions, 0)
                        open_positions -= 1

    def test_variance_skew_full_experiment_runs_independently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_variance_skew_full_experiment(
                path="Data/Merged/Prediction_all_augmented.csv",
                config=StrategyConfig(panel_mode="prediction_moments", n_components=2, min_r2=0.1, ou_window=10),
                output_dir=tmpdir,
                generate_plots=False,
            )
            self.assertIn("variance_trades", outputs)
            self.assertIn("skew_trades", outputs)
            self.assertIn("performance_table", outputs)
            self.assertTrue((Path(tmpdir) / "cumulative_pnl_by_strategy.png").exists())


if __name__ == "__main__":
    unittest.main()

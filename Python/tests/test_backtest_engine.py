from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Python.stat_arb.Moment_PCA import (
    StrategyConfig,
    _finalize_global_metrics_for_export,
    _load_ois_underlying_prices,
    build_asset_panel,
    calculate_transaction_costs,
    compute_residual_basket_weights,
    filter_residuals,
    performance_metrics,
    run_basket_backtest,
    run_full_experiment,
    run_ou_strategy,
    run_pca,
    sharpe_breakdowns,
)


class TestBacktestEngine(unittest.TestCase):
    def test_build_asset_panel_retains_underlying_mapping_inputs(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")],
                "observed_day_pst": [pd.Timestamp("2026-03-10")],
                "effr_expected_bps": [5.0],
                "jump_sr1_bps": [6.0],
                "jump_ois_bps": [7.0],
                "polymarket_H25": [0.3],
                "kalshi_H50": [0.2],
                "jump_sr1_portfolio_weights": [json.dumps({"SR1:2026-03": 1.0})],
                "jump_ois_portfolio_weights": [json.dumps({"OIS_3M_0": 1.0})],
            }
        )
        out = build_asset_panel(panel, "prediction_grouped")
        self.assertIn("jump_sr1_portfolio_weights", out.columns)
        self.assertIn("jump_ois_portfolio_weights", out.columns)
        self.assertIn("polymarket_H25", out.columns)
        self.assertIn("kalshi_H50", out.columns)

    def test_load_ois_underlying_prices_prefers_csv_and_normalizes_tickers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "SOFR OIS.csv"
            pd.DataFrame(
                {
                    "observed_day_pst": ["2026-03-10", "2026-03-10", "2026-03-11", "2026-03-11"],
                    "ticker": ["OIS_2.2666666M_3", "OIS_3.0M_0", "OIS_2.26667M_3", "OIS_3M_0"],
                    "rate": [0.04, 0.05, 0.041, 0.051],
                }
            ).to_csv(csv_path, index=False)
            loaded = _load_ois_underlying_prices(str(Path(tmpdir) / "SOFR OIS.xlsx"))
            self.assertIn("OIS_2.26667M_3", loaded.columns)
            self.assertIn("OIS_3M_0", loaded.columns)
            self.assertAlmostEqual(float(loaded.loc[pd.Timestamp("2026-03-11"), "OIS_2.26667M_3"]), 0.041, places=9)

    def test_basket_backtest_uses_entry_weights_and_underlying_prices(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 3,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]),
                "asset_a": [100.0, 103.0, 104.0],
                "asset_b": [200.0, 198.0, 197.0],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["asset_a", "asset_a"],
                "decision_date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-12"]),
                "event": ["enter_long", "exit_long"],
                "position": [1, 0],
                "pnl_bps": [0.0, 0.0],
            }
        )
        # Include a different non-entry weight so test confirms entry-time weight lock.
        basket_weights = {
            (0, "asset_a"): np.array([1.0, -0.5], dtype=float),
            (2, "asset_a"): np.array([10.0, -10.0], dtype=float),
        }
        daily_pnl, trade_log, cumulative_pnl = run_basket_backtest(
            panel=panel,
            signal_trades=signal_trades,
            basket_weights=basket_weights,
            asset_cols=["asset_a", "asset_b"],
        )
        self.assertEqual(len(trade_log), 1)
        expected_trade_pnl = (104.0 - 100.0) + (-0.5) * (197.0 - 200.0)
        self.assertAlmostEqual(float(trade_log.iloc[0]["trade_pnl"]), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(daily_pnl["daily_pnl"].sum()), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(cumulative_pnl.iloc[-1]["cumulative_pnl"]), expected_trade_pnl, places=9)
        logged_weights = json.loads(trade_log.iloc[0]["entry_weights"])
        self.assertAlmostEqual(float(logged_weights["asset_a"]), 1.0)
        self.assertAlmostEqual(float(logged_weights["asset_b"]), -0.5)
        self.assertTrue(bool(trade_log.iloc[0]["weights_constant_during_trade"]))

    def test_basket_backtest_ignores_assets_without_finite_entry_prices(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 3,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]),
                "asset_a": [100.0, 103.0, 104.0],
                "asset_b": [200.0, 198.0, 197.0],
                "asset_sparse": [np.nan, np.nan, np.nan],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["asset_a", "asset_a"],
                "decision_date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-12"]),
                "event": ["enter_long", "exit_long"],
                "position": [1, 0],
                "pnl_bps": [0.0, 0.0],
            }
        )
        basket_weights = {
            (0, "asset_a"): np.array([1.0, -0.5, 2.0], dtype=float),
            (2, "asset_a"): np.array([10.0, -10.0, 1.0], dtype=float),
        }
        daily_pnl, trade_log, cumulative_pnl = run_basket_backtest(
            panel=panel,
            signal_trades=signal_trades,
            basket_weights=basket_weights,
            asset_cols=["asset_a", "asset_b", "asset_sparse"],
        )
        self.assertEqual(len(trade_log), 1)
        expected_trade_pnl = (104.0 - 100.0) + (-0.5) * (197.0 - 200.0)
        self.assertAlmostEqual(float(trade_log.iloc[0]["trade_pnl"]), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(daily_pnl["daily_pnl"].sum()), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(cumulative_pnl.iloc[-1]["cumulative_pnl"]), expected_trade_pnl, places=9)

    def test_basket_backtest_maps_composites_to_underlying_legs_for_mtm(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 3,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]),
                "jump_sr1_bps": [10.0, 11.0, 9.0],
                "effr_expected_bps": [8.0, 9.5, 7.5],
                "Poly_Hike_bps": [6.0, 7.0, 6.5],
                "polymarket_H25": [0.30, 0.35, 0.32],
                "jump_sr1_portfolio_weights": [
                    json.dumps({"SR1:2026-03": 1.5, "SR1:2026-04": -0.5}),
                    json.dumps({"SR1:2026-03": 1.4, "SR1:2026-04": -0.4}),
                    json.dumps({"SR1:2026-03": 1.3, "SR1:2026-04": -0.3}),
                ],
                "jump_ois_portfolio_weights": [None, None, None],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["jump_sr1_bps", "jump_sr1_bps"],
                "decision_date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-12"]),
                "event": ["enter_long", "exit_long"],
                "position": [1, 0],
                "pnl_bps": [0.0, 0.0],
            }
        )
        basket_weights = {
            (0, "jump_sr1_bps"): np.array([1.0, 2.0, -0.5], dtype=float),
            (2, "jump_sr1_bps"): np.array([20.0, -20.0, 2.0], dtype=float),
        }
        underlying_prices = pd.DataFrame(
            {
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]),
                "SR1:2026-03": [0.0400, 0.0410, 0.0390],
                "SR1:2026-04": [0.0450, np.nan, 0.0440],  # missing day should ffill
                "EFFR_2026_02": [0.0420, 0.0430, 0.0410],
                "EFFR_2026_03": [0.0440, 0.0460, 0.0450],
                "polymarket_H25": [0.30, 0.35, 0.32],
            }
        )
        daily_pnl, trade_log, cumulative_pnl = run_basket_backtest(
            panel=panel,
            signal_trades=signal_trades,
            basket_weights=basket_weights,
            asset_cols=["jump_sr1_bps", "effr_expected_bps", "Poly_Hike_bps"],
            underlying_prices=underlying_prices,
        )
        self.assertEqual(len(trade_log), 1)
        entry_weights = json.loads(trade_log.iloc[0]["entry_weights"])
        self.assertIn("SR1:2026-03", entry_weights)
        self.assertIn("SR1:2026-04", entry_weights)
        self.assertIn("EFFR_2026_02", entry_weights)
        self.assertIn("EFFR_2026_03", entry_weights)
        self.assertIn("polymarket_H25", entry_weights)
        self.assertNotIn("jump_sr1_bps", entry_weights)
        effr_scale = 31.0 / 13.0
        expected_weights = {
            "SR1:2026-03": 15000.0,
            "SR1:2026-04": -5000.0,
            "EFFR_2026_02": 2.0 * (-10000.0 * effr_scale),
            "EFFR_2026_03": 2.0 * (10000.0 * effr_scale),
            "polymarket_H25": -12.5,
        }
        for ticker, expected in expected_weights.items():
            self.assertAlmostEqual(float(entry_weights[ticker]), expected, places=9)
        expected_trade_pnl = sum(
            expected_weights[ticker] * (float(underlying_prices.iloc[2][ticker]) - float(underlying_prices.iloc[0][ticker]))
            for ticker in expected_weights
        )
        self.assertAlmostEqual(float(trade_log.iloc[0]["trade_pnl"]), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(daily_pnl["daily_pnl"].sum()), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(cumulative_pnl.iloc[-1]["cumulative_pnl"]), expected_trade_pnl, places=9)

    def test_basket_backtest_maps_prediction_moments2_expected_to_raw_contracts(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 2,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "polymarket_expected_bps": [2.5, 3.0],
                "polymarket_C25": [0.20, 0.18],
                "polymarket_H0": [0.35, 0.34],
                "polymarket_H25": [0.45, 0.48],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["polymarket_expected_bps", "polymarket_expected_bps"],
                "decision_date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "event": ["enter_long", "exit_long"],
                "position": [1, 0],
                "pnl_bps": [0.0, 0.0],
            }
        )
        underlying_prices = panel[["observed_day_pst", "polymarket_C25", "polymarket_H0", "polymarket_H25"]].copy()
        daily_pnl, trade_log, _ = run_basket_backtest(
            panel=panel,
            signal_trades=signal_trades,
            basket_weights={(0, "polymarket_expected_bps"): np.array([1.0], dtype=float)},
            asset_cols=["polymarket_expected_bps"],
            underlying_prices=underlying_prices,
        )
        self.assertEqual(len(trade_log), 1)
        entry_weights = json.loads(trade_log.iloc[0]["entry_weights"])
        self.assertEqual(set(entry_weights), {"polymarket_C25", "polymarket_H0", "polymarket_H25"})
        self.assertAlmostEqual(float(entry_weights["polymarket_C25"]), -25.0, places=9)
        self.assertAlmostEqual(float(entry_weights["polymarket_H0"]), 0.0, places=9)
        self.assertAlmostEqual(float(entry_weights["polymarket_H25"]), 25.0, places=9)
        self.assertGreater(abs(float(daily_pnl["daily_pnl"].sum())), 0.0)

    def test_basket_backtest_maps_prediction_moments2_tail_weight_to_tail_contracts(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 2,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "polymarket_tail_weight_bps": [5.0, 5.2],
                "polymarket_C75+": [0.05, 0.04],
                "polymarket_C50": [0.10, 0.11],
                "polymarket_C25": [0.20, 0.21],
                "polymarket_H0": [0.25, 0.24],
                "polymarket_H25": [0.20, 0.20],
                "polymarket_H50": [0.12, 0.13],
                "polymarket_H75": [0.08, 0.07],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["polymarket_tail_weight_bps", "polymarket_tail_weight_bps"],
                "decision_date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "event": ["enter_long", "exit_long"],
                "position": [1, 0],
                "pnl_bps": [0.0, 0.0],
            }
        )
        underlying_prices = panel[
            [
                "observed_day_pst",
                "polymarket_C75+",
                "polymarket_C50",
                "polymarket_C25",
                "polymarket_H0",
                "polymarket_H25",
                "polymarket_H50",
                "polymarket_H75",
            ]
        ].copy()
        _, trade_log, _ = run_basket_backtest(
            panel=panel,
            signal_trades=signal_trades,
            basket_weights={(0, "polymarket_tail_weight_bps"): np.array([1.0], dtype=float)},
            asset_cols=["polymarket_tail_weight_bps"],
            underlying_prices=underlying_prices,
        )
        self.assertEqual(len(trade_log), 1)
        entry_weights = json.loads(trade_log.iloc[0]["entry_weights"])
        self.assertNotIn("polymarket_H25", entry_weights)
        self.assertNotIn("polymarket_H0", entry_weights)
        self.assertAlmostEqual(float(entry_weights["polymarket_C75+"]), (75.0**2) / 35.0, places=9)
        self.assertAlmostEqual(float(entry_weights["polymarket_H75"]), (75.0**2) / 35.0, places=9)

    def test_basket_backtest_normalizes_ois_ticker_precision_for_weights(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 2,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "jump_ois_bps": [1.0, 1.2],
                "jump_ois_portfolio_weights": [
                    json.dumps({"OIS_2.2666666M_0": 1.0}),
                    json.dumps({"OIS_2.2666666M_0": 1.0}),
                ],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["jump_ois_bps", "jump_ois_bps"],
                "decision_date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "event": ["enter_long", "exit_long"],
                "position": [1, 0],
                "pnl_bps": [0.0, 0.0],
            }
        )
        underlying_prices = pd.DataFrame(
            {
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "OIS_2.26667M_0": [0.0400, 0.0415],
            }
        )
        daily_pnl, trade_log, cumulative_pnl = run_basket_backtest(
            panel=panel,
            signal_trades=signal_trades,
            basket_weights={(0, "jump_ois_bps"): np.array([1.0], dtype=float)},
            asset_cols=["jump_ois_bps"],
            underlying_prices=underlying_prices,
        )
        self.assertEqual(len(trade_log), 1)
        expected_trade_pnl = 10000.0 * (0.0415 - 0.0400)
        self.assertAlmostEqual(float(trade_log.iloc[0]["trade_pnl"]), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(daily_pnl["daily_pnl"].sum()), expected_trade_pnl, places=9)
        self.assertAlmostEqual(float(cumulative_pnl.iloc[-1]["cumulative_pnl"]), expected_trade_pnl, places=9)

    def test_performance_metrics_reports_net_fields_with_transaction_costs(self):
        daily_pnl = pd.DataFrame(
            {
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "daily_pnl": [15.0, 5.0],
            }
        )
        trade_log = pd.DataFrame(
            [
                {
                    "asset": "mix_asset",
                    "decision_date": pd.Timestamp("2026-03-18"),
                    "entry_time": pd.Timestamp("2026-03-10"),
                    "exit_time": pd.Timestamp("2026-03-11"),
                    "position": 1,
                    "entry_weights": json.dumps({"polymarket_H25": 10.0, "SR1:2026-03": 2.0}, sort_keys=True),
                    "entry_prices": json.dumps({"polymarket_H25": 0.40, "SR1:2026-03": 0.95}, sort_keys=True),
                    "trade_pnl": 20.0,
                    "weights_constant_during_trade": True,
                }
            ]
        )
        cost_summary = calculate_transaction_costs(trade_log)
        metrics = performance_metrics(daily_pnl, trade_log, pd.DataFrame())
        row = metrics.iloc[0]
        self.assertIn("total_commissions", metrics.columns)
        self.assertIn("net_pnl", metrics.columns)
        self.assertIn("net_sharpe", metrics.columns)
        self.assertIn("gross_total_return", metrics.columns)
        self.assertIn("net_total_return", metrics.columns)
        self.assertIn("gross_max_drawdown_per_day", metrics.columns)
        self.assertIn("net_max_drawdown_per_day", metrics.columns)
        self.assertIn("gross_win_rate", metrics.columns)
        self.assertIn("net_win_rate", metrics.columns)
        self.assertIn("avg_holding_period_per_trade_days", metrics.columns)
        self.assertGreater(float(row["total_commissions"]), 0.0)
        self.assertAlmostEqual(float(row["total_commissions"]), float(cost_summary["total_commissions"]), places=9)
        self.assertAlmostEqual(float(row["net_pnl"]), float(row["total_pnl"]) - float(row["total_commissions"]), places=9)
        self.assertIn("commission_by_platform", cost_summary)
        self.assertIn("commission_by_trade", cost_summary)
        self.assertTrue({"polymarket", "cme"}.issubset(set(cost_summary["commission_by_platform"].index)))
        self.assertEqual(len(cost_summary["commission_by_trade"]), len(trade_log))
        self.assertAlmostEqual(float(row["gross_total_return"]), float(row["total_pnl"]), places=9)
        self.assertAlmostEqual(float(row["net_total_return"]), float(row["net_pnl"]), places=9)
        self.assertLessEqual(float(row["gross_max_drawdown_per_day"]), 0.0)
        self.assertLessEqual(float(row["net_max_drawdown_per_day"]), 0.0)

    def test_sharpe_breakdowns_include_gross_and_net_levels(self):
        daily_pnl = pd.DataFrame(
            {
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13"]),
                "daily_pnl": [3.0, 1.0, -1.0, -3.0],
            }
        )
        trade_log = pd.DataFrame(
            [
                {"asset": "a", "decision_date": pd.Timestamp("2026-03-18"), "trade_pnl": 3.0},
                {"asset": "a", "decision_date": pd.Timestamp("2026-03-18"), "trade_pnl": 1.0},
                {"asset": "b", "decision_date": pd.Timestamp("2026-04-15"), "trade_pnl": -3.0},
                {"asset": "b", "decision_date": pd.Timestamp("2026-04-15"), "trade_pnl": -1.0},
            ]
        )
        trade_costs = pd.Series(0.0, index=trade_log.index)
        by_asset, by_meeting, by_year = sharpe_breakdowns(daily_pnl, trade_log, commission_by_trade=trade_costs, annualization=16.0)
        self.assertTrue({"asset", "gross_sharpe", "net_sharpe"}.issubset(by_asset.columns))
        self.assertTrue({"decision_date", "gross_sharpe", "net_sharpe"}.issubset(by_meeting.columns))
        self.assertTrue({"year", "gross_sharpe", "net_sharpe"}.issubset(by_year.columns))
        expected_annualized = 8.0
        self.assertAlmostEqual(float(by_asset.loc[by_asset["asset"] == "a", "gross_sharpe"].iloc[0]), expected_annualized, places=9)
        self.assertAlmostEqual(float(by_meeting.loc[by_meeting["decision_date"] == pd.Timestamp("2026-03-18"), "gross_sharpe"].iloc[0]), expected_annualized, places=9)
        self.assertAlmostEqual(float(by_year.loc[by_year["year"] == 2026, "gross_sharpe"].iloc[0]), 0.0, places=9)

    def test_finalize_global_metrics_for_export_keeps_requested_fields_and_platform_breakdown(self):
        metrics = pd.DataFrame(
            [
                {
                    "gross_total_return": 0.1,
                    "net_total_return": 0.09,
                    "gross_total_profit": 100.0,
                    "net_total_profit": 90.0,
                    "gross_annualized_return": 0.12,
                    "net_annualized_return": 0.1,
                    "gross_annualized_sharpe": 1.5,
                    "net_annualized_sharpe": 1.2,
                    "total_transaction_costs": 10.0,
                    "gross_max_drawdown_per_day": -0.02,
                    "net_max_drawdown_per_day": -0.03,
                    "gross_trade_win_rate": 0.6,
                    "net_trade_win_rate": 0.55,
                    "number_of_trades": 20,
                    "avg_holding_period_per_trade_days": 2.5,
                    "gross_sharpe": 0.2,
                }
            ]
        )
        tc_breakdown = pd.DataFrame(
            {
                "platform": ["cme", "polymarket"],
                "commission": [4.0, 6.0],
            }
        )
        out = _finalize_global_metrics_for_export(metrics, tc_breakdown)
        self.assertEqual(list(out.columns[:2]), ["transaction_costs_cme", "transaction_costs_polymarket"])
        self.assertAlmostEqual(float(out.iloc[0]["transaction_costs_cme"]), 4.0, places=9)
        self.assertAlmostEqual(float(out.iloc[0]["transaction_costs_polymarket"]), 6.0, places=9)
        self.assertNotIn("gross_sharpe", out.columns)
        self.assertEqual(
            set(out.columns),
            {
                "transaction_costs_cme",
                "transaction_costs_polymarket",
                "gross_total_return",
                "net_total_return",
                "gross_total_profit",
                "net_total_profit",
                "gross_annualized_return",
                "net_annualized_return",
                "gross_annualized_sharpe",
                "net_annualized_sharpe",
                "total_transaction_costs",
                "gross_max_drawdown_per_day",
                "net_max_drawdown_per_day",
                "gross_trade_win_rate",
                "net_trade_win_rate",
                "number_of_trades",
                "avg_holding_period_per_trade_days",
            },
        )

    def test_three_layer_filter_and_ou_strategy_emit_diagnostics_without_adf_gate(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-04-15")] * 120,
                "observed_day_pst": pd.date_range("2026-01-01", periods=120, freq="D"),
            }
        )
        rng = np.random.default_rng(7)
        vals = np.zeros(120, dtype=float)
        for i in range(1, 120):
            vals[i] = 0.6 * vals[i - 1] + rng.normal(scale=0.2)
        residuals = pd.DataFrame(
            {
                "asset_a": vals,
            },
            index=panel.index,
        )
        tradable = filter_residuals(
            residuals=residuals,
            r2_scores=pd.Series({"asset_a": 0.95}),
            min_r2=0.1,
            max_rho=1.0,
            adf_alpha=0.0,
            max_half_life_days=5.0,
            variance_threshold=0.05,
        )
        self.assertTrue(bool(tradable.loc[tradable.index[-1], "asset_a"]))
        kept = [c for c in tradable.columns if bool(tradable[c].any())]
        self.assertIn("asset_a", kept)
        trades, metrics, diagnostics = run_ou_strategy(
            panel=panel,
            residuals=residuals,
            assets=kept,
            tradable=tradable[kept],
            config=StrategyConfig(
                ou_window=20,
                entry_sigma=0.8,
                exit_sigma=0.2,
                adf_alpha=0.0,
                no_trade_days_before_decision=0,
            ),
            return_diagnostics=True,
        )
        self.assertFalse(trades.empty)
        self.assertGreater(int(metrics.loc[metrics["asset"] == "asset_a", "entries"].iloc[0]), 0)
        self.assertFalse(diagnostics.empty)
        self.assertTrue({"ADF_stat", "ADF_pvalue", "OU_kappa", "OU_mu", "OU_sigma", "half_life", "spread_std"}.issubset(diagnostics.columns))

    def test_filter_residuals_diagnostics_include_gate_flags(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-04-15")] * 40,
                "observed_day_pst": pd.date_range("2026-02-01", periods=40, freq="D"),
            }
        )
        residuals = pd.DataFrame({"asset_a": np.linspace(-1.0, 1.0, 40)}, index=panel.index)
        _, diag = filter_residuals(
            residuals=residuals,
            r2_scores=pd.Series({"asset_a": 0.95}),
            min_r2=0.1,
            max_rho=1.0,
            adf_alpha=0.05,
            max_half_life_days=30.0,
            variance_threshold=0.01,
            return_diagnostics=True,
            estimation_window=20,
        )
        self.assertFalse(diag.empty)
        self.assertTrue({"r2_pass", "half_life_pass", "variance_pass", "selected"}.issubset(diag.columns))

    def test_ou_strategy_applies_time_stop_exit(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-04-15")] * 50,
                "observed_day_pst": pd.date_range("2026-01-01", periods=50, freq="D"),
            }
        )
        residual_path = np.concatenate(
            [
                np.array([0.0, 0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.04, -0.04, 0.02, -0.02, 0.03]),
                np.linspace(1.5, 6.0, 38),
            ]
        )
        residuals = pd.DataFrame({"asset_a": residual_path}, index=panel.index)
        tradable = pd.DataFrame(True, index=panel.index, columns=["asset_a"])
        trades, metrics = run_ou_strategy(
            panel=panel,
            residuals=residuals,
            assets=["asset_a"],
            tradable=tradable,
            config=StrategyConfig(
                ou_window=12,
                entry_sigma=0.5,
                exit_sigma=0.1,
                no_trade_days_before_decision=0,
                max_holding_days=3,
            ),
        )
        self.assertFalse(trades.empty)
        self.assertGreaterEqual(int((trades["event"] == "time_stop_exit").sum()), 1)
        self.assertGreaterEqual(int(metrics.loc[metrics["asset"] == "asset_a", "entries"].iloc[0]), 1)

    def test_basket_backtest_validates_weight_shape_alignment(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 2,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                "asset_a": [100.0, 101.0],
                "asset_b": [200.0, 201.0],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["asset_a"],
                "decision_date": [pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10"]),
                "event": ["enter_long"],
                "position": [1],
                "pnl_bps": [0.0],
            }
        )
        with self.assertRaises(AssertionError):
            run_basket_backtest(panel, signal_trades, {(0, "asset_a"): np.array([1.0])}, ["asset_a", "asset_b"])

    def test_basket_backtest_applies_execution_lag_to_fill_times_and_prices(self):
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-03-18")] * 4,
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13"]),
                "asset_a": [100.0, 101.0, 102.0, 103.0],
            }
        )
        signal_trades = pd.DataFrame(
            {
                "asset": ["asset_a", "asset_a"],
                "decision_date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-18")],
                "observed_day_pst": pd.to_datetime(["2026-03-10", "2026-03-12"]),
                "event": ["enter_long", "exit_long"],
                "position": [1, 0],
                "pnl_bps": [0.0, 0.0],
            }
        )
        daily_pnl, trade_log, _ = run_basket_backtest(
            panel=panel,
            signal_trades=signal_trades,
            basket_weights={(0, "asset_a"): np.array([1.0], dtype=float)},
            asset_cols=["asset_a"],
            execution_lag_days=1,
        )
        self.assertEqual(len(trade_log), 1)
        self.assertEqual(pd.Timestamp(trade_log.iloc[0]["entry_time"]), pd.Timestamp("2026-03-11"))
        self.assertEqual(pd.Timestamp(trade_log.iloc[0]["exit_time"]), pd.Timestamp("2026-03-13"))
        self.assertAlmostEqual(float(trade_log.iloc[0]["trade_pnl"]), 2.0, places=9)
        self.assertAlmostEqual(float(daily_pnl["daily_pnl"].sum()), 2.0, places=9)

    def test_compute_residual_basket_weights_reuses_run_pca_weights(self):
        days = pd.date_range("2026-01-01", periods=40, freq="D")
        base = np.linspace(0.0, 1.0, len(days))
        panel = pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-02-20")] * len(days),
                "observed_day_pst": days,
                "effr_expected_bps": 10 + base,
                "jump_sr1_bps": 11 + 1.1 * base,
                "jump_ois_bps": 9 + 0.9 * base,
                "polymarket_H25": 0.2 + 0.01 * base,
                "polymarket_H50": 0.1 + 0.01 * base,
                "polymarket_C25": 0.05,
                "polymarket_C50": 0.03,
                "kalshi_H25": 0.25 + 0.01 * base,
                "kalshi_H50": 0.12 + 0.01 * base,
                "kalshi_C25": 0.03,
                "kalshi_C50+": 0.02,
            }
        )
        asset_panel = build_asset_panel(panel, "prediction_grouped")
        _, _, _, _, pca_weights, pca_assets = run_pca(asset_panel, n_components=3, rolling_window_days=15)
        basket_weights, basket_assets = compute_residual_basket_weights(asset_panel, n_components=3, rolling_window_days=15)
        self.assertEqual(pca_assets, basket_assets)
        self.assertSetEqual(set(pca_weights.keys()), set(basket_weights.keys()))
        sample_key = next(iter(pca_weights))
        np.testing.assert_allclose(pca_weights[sample_key], basket_weights[sample_key], atol=1e-12, rtol=0.0)

    def test_full_experiment_writes_backtest_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_full_experiment(
                path="Data/Merged/Prediction_all_augmented.csv",
                config=StrategyConfig(panel_mode="prediction_all", n_components=1, pca_rolling_window_days=30, min_r2=0.1, ou_window=10),
                output_dir=tmpdir,
                generate_plots=True,
            )
            out_dir = Path(tmpdir) / "outputs"
            self.assertTrue((out_dir / "daily_pnl.csv").exists())
            self.assertTrue((out_dir / "trade_log.csv").exists())
            self.assertTrue((out_dir / "cumulative_pnl.csv").exists())
            self.assertTrue((out_dir / "residual_diagnostics.csv").exists())
            self.assertTrue((out_dir / "strategy_metrics.csv").exists())
            self.assertTrue((out_dir / "per_asset_metrics.csv").exists())
            self.assertTrue((out_dir / "diagnostics_cointegration.csv").exists())
            self.assertTrue((out_dir / "diagnostics_spread_model.csv").exists())
            self.assertTrue((Path(tmpdir) / "cumulative_pnl.png").exists())
            self.assertTrue((Path(tmpdir) / "spread_selection.png").exists())
            self.assertTrue((Path(tmpdir) / "trade_pnl_histogram.png").exists())
            self.assertTrue((Path(tmpdir) / "pca_variance.png").exists())
            self.assertTrue((Path(tmpdir) / "cumulative_pnl_by_strategy.png").exists())
            self.assertTrue((Path(tmpdir) / "factor_mimicking_returns.png").exists())
            #self.assertTrue(any((Path(tmpdir) / "walkforward").glob("*.png")))
            self.assertIn("trade_log", outputs)
            self.assertIn("daily_pnl", outputs)
            self.assertIn("diagnostics_cointegration", outputs)
            self.assertIn("diagnostics_spread_model", outputs)
            self.assertIn("global_metrics", outputs)
            self.assertIn("per_asset_metrics", outputs)
            self.assertIn("strategy_metrics", outputs)
            self.assertTrue(
                {
                    "asset",
                    "decision_date",
                    "entry_time",
                    "exit_time",
                    "position",
                    "entry_weights",
                    "entry_prices",
                    "trade_pnl",
                    "weights_constant_during_trade",
                }.issubset(outputs["trade_log"].columns)
            )

    def test_full_experiment_applies_start_date_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "panel.csv"
            days = pd.date_range("2026-01-01", periods=90, freq="D")
            base = np.linspace(0.0, 1.0, len(days))
            raw = pd.DataFrame(
                {
                    "decision_date": [pd.Timestamp("2026-05-20")] * len(days),
                    "observed_day_pst": days,
                    "effr_expected_bps": 5.0 + base,
                    "jump_sr1": 0.0005 + 0.00005 * base,
                    "jump_ois": 0.0004 + 0.00005 * base,
                    "polymarket_H25": 0.30 + 0.02 * base,
                    "polymarket_C25": 0.20 - 0.01 * base,
                    "kalshi_H50": 0.25 + 0.01 * base,
                    "kalshi_C50": 0.15 - 0.01 * base,
                }
            )
            raw.to_csv(data_path, index=False)
            outputs = run_full_experiment(
                path=str(data_path),
                config=StrategyConfig(
                    panel_mode="prediction_grouped",
                    n_components=1,
                    pca_rolling_window_days=20,
                    ou_window=10,
                    min_r2=0.1,
                    start_date="2026-02-15",
                ),
                output_dir=tmpdir,
                generate_plots=False,
            )
            self.assertGreaterEqual(outputs["panel"]["observed_day_pst"].min(), pd.Timestamp("2026-02-15"))


if __name__ == "__main__":
    unittest.main()

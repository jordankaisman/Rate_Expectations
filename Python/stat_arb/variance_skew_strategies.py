from __future__ import annotations

import numpy as np
import pandas as pd

from Python.stat_arb.Moment_PCA import (
    PREDICTION_BPS_MAP,
    StrategyConfig,
    _calculate_moments,
    build_half_life_table,
    build_model_fit_table,
    build_performance_table,
    generate_residual_walkforward_plots,
    load_prediction_panel,
    plot_cumulative_pnl,
    run_ou_strategy,
)


def _tail_skew_asset(df: pd.DataFrame, prefix: str, side: str) -> pd.Series:
    cols = [c for c in PREDICTION_BPS_MAP if c.startswith(prefix) and c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    probs = df[cols].apply(pd.to_numeric, errors="coerce")
    strikes = pd.Series({c: PREDICTION_BPS_MAP[c] for c in cols}, dtype=float)
    if side == "positive":
        weights = strikes.clip(lower=0.0) ** 2
    elif side == "negative":
        weights = (-strikes.clip(upper=0.0)) ** 2
    else:
        raise ValueError(f"Unknown skew side: {side}")
    if float(weights.sum()) <= 0:
        return pd.Series(np.nan, index=df.index)
    return probs.mul(weights, axis=1).sum(axis=1, min_count=1)

def _cointegration_residuals(
    panel: pd.DataFrame,
    prediction_assets: list[str],
    market_assets: list[str],
    min_r2: float = 0.2,
    normalize_within_decision_date: bool = False,
    rolling_window_days: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    residuals = pd.DataFrame(index=panel.index)
    r2_scores: dict[str, float] = {}

    def _zscore_by_decision(values: pd.Series, valid_mask: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=panel.index, dtype=float)
        grouped = panel.loc[valid_mask, "decision_date"].groupby(panel.loc[valid_mask, "decision_date"]).groups
        for _, idx in grouped.items():
            v = values.loc[idx]
            std = float(v.std(ddof=0))
            if not np.isfinite(std) or std <= 1e-12:
                continue
            out.loc[idx] = (v - float(v.mean())) / std
        return out

    for pred in prediction_assets:
        if pred not in panel.columns:
            continue
        y = pd.to_numeric(panel[pred], errors="coerce")
        for market in market_assets:
            if market not in panel.columns:
                continue
            x = pd.to_numeric(panel[market], errors="coerce")
            mask = y.notna() & x.notna()
            if mask.sum() < 20:
                continue

            x_work = _zscore_by_decision(x, mask) if normalize_within_decision_date else x.astype(float)
            y_work = _zscore_by_decision(y, mask) if normalize_within_decision_date else y.astype(float)
            fit_mask = mask & x_work.notna() & y_work.notna()
            if fit_mask.sum() < 20:
                continue

            name = f"{pred}_vs_{market}"
            fitted = pd.Series(np.nan, index=panel.index, dtype=float)
            if rolling_window_days is None or rolling_window_days <= 0:
                model = LinearRegression()
                x_vals = x_work.loc[fit_mask].to_numpy().reshape(-1, 1)
                y_vals = y_work.loc[fit_mask].to_numpy()
                model.fit(x_vals, y_vals)
                fitted.loc[fit_mask] = model.predict(x_vals)
            else:
                fit_df = (
                    panel.loc[fit_mask, ["observed_day_pst"]]
                    .assign(x=x_work.loc[fit_mask], y=y_work.loc[fit_mask])
                    .sort_values("observed_day_pst")
                )
                for idx, row in fit_df.iterrows():
                    end_day = row["observed_day_pst"]
                    start_day = end_day - pd.Timedelta(days=int(rolling_window_days))
                    train_mask = (fit_df["observed_day_pst"] >= start_day) & (fit_df["observed_day_pst"] <= end_day)
                    train = fit_df.loc[train_mask]
                    if len(train) < 20:
                        continue
                    model = LinearRegression()
                    train_x = train["x"].to_numpy().reshape(-1, 1)
                    train_y = train["y"].to_numpy()
                    model.fit(train_x, train_y)
                    fitted.loc[idx] = float(model.predict(np.array([[row["x"]]], dtype=float))[0])

            pred_mask = fitted.notna() & y_work.notna()
            if pred_mask.sum() < 20:
                continue
            y_actual = y_work.loc[pred_mask]
            y_fitted = fitted.loc[pred_mask]
            sst = float(((y_actual - y_actual.mean()) ** 2).sum())
            if sst <= 1e-12:
                continue
            sse = float(((y_actual - y_fitted) ** 2).sum())
            r2 = 1.0 - (sse / sst)
            if r2 <= min_r2:
                continue
            residuals[name] = y_work - fitted
            r2_scores[name] = r2
    return residuals, pd.Series(r2_scores, dtype=float)


def _reindex_decision_panel_daily(panel: pd.DataFrame, fill_cols: list[str], fill_limit: int = 3) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for decision_date, grp in panel.groupby("decision_date", sort=False):
        g = grp.sort_values("observed_day_pst").set_index("observed_day_pst")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_idx)
        g.index.name = "observed_day_pst"
        g["decision_date"] = decision_date
        cols_to_fill = [c for c in fill_cols if c in g.columns]
        if cols_to_fill:
            g[cols_to_fill] = g[cols_to_fill].ffill(limit=fill_limit)
        frames.append(g.reset_index())
    if not frames:
        return panel.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def build_variance_arb_panel(df: pd.DataFrame, use_high_variance_instruments: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["polymarket_variance_asset"], _ = _calculate_moments(out, "polymarket_")
    out["kalshi_variance_asset"], _ = _calculate_moments(out, "kalshi_")
    suffix = "_hvar" if use_high_variance_instruments else ""
    market_cols = [f"sr1_butterfly_bps{suffix}", f"effr_butterfly_bps{suffix}", f"ois_butterfly_bps{suffix}"]
    for col in market_cols:
        if col not in out.columns:
            out[col] = np.nan
    panel = out[
        [
            "decision_date",
            "observed_day_pst",
            "polymarket_variance_asset",
            "kalshi_variance_asset",
            *market_cols,
        ]
    ].copy()
    panel = _reindex_decision_panel_daily(panel, fill_cols=market_cols, fill_limit=3)
    return panel.sort_values(["decision_date", "observed_day_pst"])


def build_skew_arb_panel(df: pd.DataFrame, use_high_variance_instruments: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["polymarket_positive_skew_asset"] = _tail_skew_asset(out, "polymarket_", side="positive")
    out["kalshi_positive_skew_asset"] = _tail_skew_asset(out, "kalshi_", side="positive")
    out["polymarket_negative_skew_asset"] = _tail_skew_asset(out, "polymarket_", side="negative")
    out["kalshi_negative_skew_asset"] = _tail_skew_asset(out, "kalshi_", side="negative")
    suffix = "_hvar" if use_high_variance_instruments else ""
    steep_cols = [f"sr1_steepener_bps{suffix}", f"effr_steepener_bps{suffix}", f"ois_steepener_bps{suffix}"]
    flat_cols = [f"sr1_flattener_bps{suffix}", f"effr_flattener_bps{suffix}", f"ois_flattener_bps{suffix}"]
    for prefix in ["sr1", "effr", "ois"]:
        steep_col = f"{prefix}_steepener_bps{suffix}"
        flat_col = f"{prefix}_flattener_bps{suffix}"
        if steep_col not in out.columns:
            out[steep_col] = np.nan
        if flat_col not in out.columns:
            out[flat_col] = -pd.to_numeric(out[steep_col], errors="coerce")
    panel = out[
        [
            "decision_date",
            "observed_day_pst",
            "polymarket_positive_skew_asset",
            "kalshi_positive_skew_asset",
            "polymarket_negative_skew_asset",
            "kalshi_negative_skew_asset",
            *steep_cols,
            *flat_cols,
        ]
    ].copy()
    panel = _reindex_decision_panel_daily(panel, fill_cols=steep_cols + flat_cols, fill_limit=3)
    return panel.sort_values(["decision_date", "observed_day_pst"])


def run_variance_arb_experiment(df: pd.DataFrame, config: StrategyConfig, min_r2: float = 0.2) -> dict[str, pd.DataFrame]:
    variance_panel = build_variance_arb_panel(df, use_high_variance_instruments=config.use_high_variance_instruments)
    suffix = "_hvar" if config.use_high_variance_instruments else ""
    residuals, r2 = _cointegration_residuals(
        variance_panel,
        prediction_assets=["polymarket_variance_asset", "kalshi_variance_asset"],
        market_assets=[f"sr1_butterfly_bps{suffix}", f"effr_butterfly_bps{suffix}", f"ois_butterfly_bps{suffix}"],
        min_r2=min_r2,
        normalize_within_decision_date=True,
        rolling_window_days=config.cointegration_rolling_window_days,
    )
    threshold_used = float(min_r2)
    if residuals.empty and min_r2 > 0:
        threshold_used = 0.0
        residuals, r2 = _cointegration_residuals(
            variance_panel,
            prediction_assets=["polymarket_variance_asset", "kalshi_variance_asset"],
            market_assets=[f"sr1_butterfly_bps{suffix}", f"effr_butterfly_bps{suffix}", f"ois_butterfly_bps{suffix}"],
            min_r2=threshold_used,
            normalize_within_decision_date=True,
            rolling_window_days=config.cointegration_rolling_window_days,
        )
    if residuals.empty:
        return {
            "variance_panel": variance_panel,
            "variance_residuals": residuals,
            "variance_r2": r2.to_frame("r2"),
            "variance_min_r2_used": pd.DataFrame({"min_r2_used": [threshold_used]}),
            "variance_trades": pd.DataFrame(),
            "variance_metrics": pd.DataFrame(columns=["asset", "sharpe", "entries", "exits", "avg_profit_per_trade", "total_profit_bps"]),
        }
    assets = list(residuals.columns)
    trades, metrics = run_ou_strategy(variance_panel, residuals, assets, config)
    return {
        "variance_panel": variance_panel,
        "variance_residuals": residuals,
        "variance_r2": r2.to_frame("r2"),
        "variance_min_r2_used": pd.DataFrame({"min_r2_used": [threshold_used]}),
        "variance_trades": trades,
        "variance_metrics": metrics,
    }


def run_skew_arb_experiment(df: pd.DataFrame, config: StrategyConfig, min_r2: float = 0.2) -> dict[str, pd.DataFrame]:
    skew_panel = build_skew_arb_panel(df, use_high_variance_instruments=config.use_high_variance_instruments)
    suffix = "_hvar" if config.use_high_variance_instruments else ""
    positive_residuals, positive_r2 = _cointegration_residuals(
        skew_panel,
        prediction_assets=["polymarket_positive_skew_asset", "kalshi_positive_skew_asset"],
        market_assets=[f"sr1_steepener_bps{suffix}", f"effr_steepener_bps{suffix}", f"ois_steepener_bps{suffix}"],
        min_r2=min_r2,
        normalize_within_decision_date=True,
        rolling_window_days=config.cointegration_rolling_window_days,
    )
    negative_residuals, negative_r2 = _cointegration_residuals(
        skew_panel,
        prediction_assets=["polymarket_negative_skew_asset", "kalshi_negative_skew_asset"],
        market_assets=[f"sr1_flattener_bps{suffix}", f"effr_flattener_bps{suffix}", f"ois_flattener_bps{suffix}"],
        min_r2=min_r2,
        normalize_within_decision_date=True,
        rolling_window_days=config.cointegration_rolling_window_days,
    )
    residuals = pd.concat([positive_residuals, negative_residuals], axis=1)
    r2 = pd.concat([positive_r2, negative_r2]).sort_index()
    threshold_used = float(min_r2)
    if residuals.empty and min_r2 > 0:
        threshold_used = 0.0
        positive_residuals, positive_r2 = _cointegration_residuals(
            skew_panel,
            prediction_assets=["polymarket_positive_skew_asset", "kalshi_positive_skew_asset"],
            market_assets=[f"sr1_steepener_bps{suffix}", f"effr_steepener_bps{suffix}", f"ois_steepener_bps{suffix}"],
            min_r2=threshold_used,
            normalize_within_decision_date=True,
            rolling_window_days=config.cointegration_rolling_window_days,
        )
        negative_residuals, negative_r2 = _cointegration_residuals(
            skew_panel,
            prediction_assets=["polymarket_negative_skew_asset", "kalshi_negative_skew_asset"],
            market_assets=[f"sr1_flattener_bps{suffix}", f"effr_flattener_bps{suffix}", f"ois_flattener_bps{suffix}"],
            min_r2=threshold_used,
            normalize_within_decision_date=True,
            rolling_window_days=config.cointegration_rolling_window_days,
        )
        residuals = pd.concat([positive_residuals, negative_residuals], axis=1)
        r2 = pd.concat([positive_r2, negative_r2]).sort_index()
    if residuals.empty:
        return {
            "skew_panel": skew_panel,
            "skew_residuals": residuals,
            "skew_r2": r2.to_frame("r2"),
            "skew_min_r2_used": pd.DataFrame({"min_r2_used": [threshold_used]}),
            "skew_trades": pd.DataFrame(),
            "skew_metrics": pd.DataFrame(columns=["asset", "sharpe", "entries", "exits", "avg_profit_per_trade", "total_profit_bps"]),
        }
    assets = list(residuals.columns)
    trades, metrics = run_ou_strategy(skew_panel, residuals, assets, config)
    return {
        "skew_panel": skew_panel,
        "skew_residuals": residuals,
        "skew_r2": r2.to_frame("r2"),
        "skew_min_r2_used": pd.DataFrame({"min_r2_used": [threshold_used]}),
        "skew_trades": trades,
        "skew_metrics": metrics,
    }


def run_full_experiment(
    path: str = "Data/Merged/Prediction_all_augmented.csv",
    config: StrategyConfig = StrategyConfig(),
    output_dir: str = "Data/Outputs/mr_cointegration_variance_skew",
    generate_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    panel_input = load_prediction_panel(path)
    capped_min_r2 = min(config.min_r2, 0.05)
    variance_outputs = run_variance_arb_experiment(panel_input, config=config, min_r2=capped_min_r2)
    skew_outputs = run_skew_arb_experiment(panel_input, config=config, min_r2=capped_min_r2)
    variance_fit = build_model_fit_table(
        variance_outputs["variance_panel"],
        variance_outputs["variance_residuals"],
        variance_outputs["variance_r2"]["r2"] if "r2" in variance_outputs["variance_r2"] else pd.Series(dtype=float),
        strategy_type="Variance",
        ou_window=config.ou_window,
    )
    skew_fit = build_model_fit_table(
        skew_outputs["skew_panel"],
        skew_outputs["skew_residuals"],
        skew_outputs["skew_r2"]["r2"] if "r2" in skew_outputs["skew_r2"] else pd.Series(dtype=float),
        strategy_type="Skew",
        ou_window=config.ou_window,
    )
    model_fit_table = pd.concat([variance_fit, skew_fit], ignore_index=True)
    half_life_table = pd.concat(
        [
            build_half_life_table(variance_outputs["variance_panel"], variance_fit),
            build_half_life_table(skew_outputs["skew_panel"], skew_fit),
        ],
        ignore_index=True,
    )
    performance_table = build_performance_table(
        {
            "Variance": variance_outputs["variance_trades"],
            "Skew": skew_outputs["skew_trades"],
        }
    )
    cumulative_pnl = plot_cumulative_pnl(
        {
            "Variance": variance_outputs["variance_trades"],
            "Skew": skew_outputs["skew_trades"],
        },
        output_path=f"{output_dir}/cumulative_pnl_by_strategy.png",
    )
    walkforward_plots = pd.DataFrame()
    if generate_plots:
        walkforward_plots = pd.concat(
            [
                generate_residual_walkforward_plots(
                    variance_outputs["variance_panel"],
                    variance_outputs["variance_residuals"],
                    variance_outputs["variance_trades"],
                    "variance",
                    config,
                    f"{output_dir}/walkforward",
                ),
                generate_residual_walkforward_plots(
                    skew_outputs["skew_panel"],
                    skew_outputs["skew_residuals"],
                    skew_outputs["skew_trades"],
                    "skew",
                    config,
                    f"{output_dir}/walkforward",
                ),
            ],
            ignore_index=True,
        )
    outputs = {
        "model_fit_table": model_fit_table,
        "half_life_table": half_life_table,
        "performance_table": performance_table,
        "cumulative_pnl": cumulative_pnl,
        "walkforward_plots": walkforward_plots,
    }
    outputs.update(variance_outputs)
    outputs.update(skew_outputs)
    return outputs


if __name__ == "__main__":
    outputs = run_full_experiment()
    print(outputs["performance_table"].to_string(index=False))

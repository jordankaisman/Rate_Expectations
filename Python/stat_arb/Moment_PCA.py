from __future__ import annotations

import json
import math
import os
import re
import warnings
import calendar
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Python" / "pure_arb"))

from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from Python.data_engineering.sofr_ois_expectations import parse_ois_tenor, tenor_to_maturity, year_fraction_act360
from Python.pure_arb.transaction_cost_models import (
    cme_round_trip_cost,
    kalshi_fed_fee,
    kalshi_spread_rule,
    polymarket_fed_fee,
    polymarket_spread_rule,
    prediction_market_one_way_cost,
)

PREDICTION_BPS_MAP = {
    "polymarket_C75+": -75.0, "polymarket_C50+": -50.0, "polymarket_C50": -50.0,
    "polymarket_C25": -25.0, "polymarket_H0": 0.0, "polymarket_H25": 25.0,
    "polymarket_H25+": 25.0, "polymarket_H50": 50.0, "polymarket_H50+": 50.0,
    "polymarket_H75": 75.0, "kalshi_C50+": -50.0, "kalshi_C50": -50.0,
    "kalshi_C25": -25.0, "kalshi_H0": 0.0, "kalshi_H25": 25.0,
    "kalshi_H50": 50.0, "kalshi_H50+": 50.0,
}

OU_BOUNDARY_EPSILON = 1e-8
OU_SINGULARITY_EPSILON = 1e-12
MIN_PCA_ROLLING_OBSERVATIONS = 5
MIN_ROLLING_OU_POINTS = 10
MIN_R2_OBSERVATIONS = 3
MAX_VOLATILITY_SCALE = 5.0
TRADE_EVENT_MARKERS = [
    ("enter_long", "^", "tab:green"), ("enter_short", "v", "tab:red"),
    ("exit_long", "o", "black"), ("exit_short", "o", "black"),
    ("forced_exit", "x", "tab:gray"),
]
TRADE_LOG_COLUMNS = ["asset", "decision_date", "entry_time", "exit_time", "position", "entry_weights", "entry_prices", "trade_pnl", "weights_constant_during_trade"]
FALLBACK_MIN_R2_CAP = 0.1
FALLBACK_MAX_HALF_LIFE_FLOOR = 10.0
FALLBACK_ENTRY_SIGMA_FLOOR = 0.5
FALLBACK_ENTRY_SIGMA_CAP = 1.0
FALLBACK_EXIT_SIGMA_CAP = 0.2
FALLBACK_VAR_SCALE_CAP = 0.1
DEFAULT_CME_ADV_CONTRACTS = 100000.0
MAX_PCA_PARALLEL_WORKERS = 32
MIN_PCA_PARALLEL_WINDOWS = 20
PCA_PARALLEL_WORK_PER_WORKER = 4
NORMAL_PCT_ABS_GT_1 = 0.31731050786291415  # P(|Z| > 1) for standard normal Z
NORMAL_PCT_ABS_GT_2 = 0.04550026389635842  # P(|Z| > 2) for standard normal Z
TOTAL_LOSS_RETURN = -1.0
REPORT_PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

CME_CONTRACT_NOTIONALS = {
    "SR3": 25.00,
    "SR1": 41.67,
    "ZQ": 41.67,
}


def _apply_report_plot_style() -> None:
    plt.rcParams.update(REPORT_PLOT_STYLE)

@dataclass(frozen=True)
class StrategyConfig:
    panel_mode: str = "prediction_grouped"
    start_date: str | None = None
    execution_lag_days: int = 0
    n_components: int = 3
    pca_rolling_window_days: int | None = None
    min_r2: float = 0.01
    max_rho: float = 0.98  # Kept for backward compat with tests
    adf_alpha: float = 0.05
    max_half_life_days: float = 5.0
    variance_threshold: float | None = None
    variance_threshold_scale: float = 0.25
    ou_window: int = 22
    entry_sigma: float = 1.25
    exit_sigma: float = 0.5
    no_trade_days_before_decision: int = 2
    max_holding_days: int = 10
    use_high_variance_instruments: bool = False
    cointegration_rolling_window_days: int | None = None


def _causal_impute_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    # Strictly causal imputation: carry last observation, then use only historical mean.
    ffilled = numeric.ffill()
    hist_mean = ffilled.expanding(min_periods=1).mean().shift(1)
    return ffilled.fillna(hist_mean).fillna(0.0)


def _infer_annualization_periods(dates: pd.Series | pd.Index | None) -> float:
    if dates is None:
        return 252.0
    ts = pd.Series(pd.to_datetime(dates, errors="coerce")).dropna()
    if ts.empty:
        return 252.0
    # Calendar-day panels (incl. weekends) use 365; business-day style uses 252.
    return 365.0 if (ts.dt.dayofweek >= 5).any() else 252.0


def _previous_equity_or_nan(equity: pd.Series, initial_capital: float) -> pd.Series:
    return equity.shift(1).fillna(initial_capital).replace(0.0, np.nan)


def _rolling_r2_series(transformed: pd.DataFrame, residuals: pd.DataFrame, window: int | None) -> pd.DataFrame:
    # Need a small finite sample floor before rolling R² is statistically meaningful.
    min_obs = max(MIN_R2_OBSERVATIONS, MIN_ROLLING_OU_POINTS // 2)
    if window and window > 1:
        den = transformed.pow(2).rolling(window=window, min_periods=min_obs).mean()
        num = residuals.pow(2).rolling(window=window, min_periods=min_obs).mean()
    else:
        den = transformed.pow(2).expanding(min_periods=min_obs).mean()
        num = residuals.pow(2).expanding(min_periods=min_obs).mean()
    return (1.0 - (num / den).replace([np.inf, -np.inf], np.nan)).clip(lower=-10.0, upper=1.0)


def _rolling_window_starts(sorted_dates: pd.Series, window_days: int) -> np.ndarray:
    days = pd.to_datetime(sorted_dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
    starts = np.zeros(len(days), dtype=int)
    if len(days) == 0:
        return starts
    window_delta = np.timedelta64(int(window_days), "D")
    left = 0
    for right in range(len(days)):
        lower = days[right] - window_delta
        while left <= right and days[left] < lower:
            left += 1
        starts[right] = left
    return starts


def load_prediction_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["observed_day_pst"] = pd.to_datetime(df["observed_day_pst"], errors="coerce")
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
    df["jump_sr1_bps"] = pd.to_numeric(df.get("jump_sr1"), errors="coerce") * 10000.0
    df["jump_ois_bps"] = pd.to_numeric(df.get("jump_ois"), errors="coerce") * 10000.0
    return df


def _expected_bps_from_distribution(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    bps = pd.Series({c: PREDICTION_BPS_MAP.get(c, np.nan) for c in cols}).dropna()
    aligned = df[bps.index].apply(pd.to_numeric, errors="coerce")
    return aligned.mul(bps, axis=1).sum(axis=1, min_count=1)


def _calculate_moments(df: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series]:
    cols = [c for c in df.columns if c.startswith(prefix) and c in PREDICTION_BPS_MAP]
    if not cols:
        nan_s = pd.Series(np.nan, index=df.index)
        return nan_s, nan_s

    probs = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sums = probs.sum(axis=1)
    probs_norm = probs.div(row_sums.where(row_sums > 1e-12), axis=0)
    
    bps_vals = np.array([float(PREDICTION_BPS_MAP[c]) for c in cols])
    mu = (probs_norm * bps_vals).sum(axis=1)
    diffs = bps_vals - mu.to_numpy()[:, None]
    
    variance = (probs_norm * (diffs ** 2)).sum(axis=1)
    std = np.sqrt(variance)
    skew = (probs_norm * (diffs ** 3)).sum(axis=1) / (std ** 3)
    
    return variance.where(row_sums > 1e-12, np.nan), skew.where((row_sums > 1e-12) & (std > 1e-12), np.nan)


def _tail_abs_weight(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix) and c in PREDICTION_BPS_MAP]
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if not cols: return out

    probs = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    strikes = pd.Series({c: float(PREDICTION_BPS_MAP[c]) for c in cols}, dtype=float)

    for idx, p in probs.iterrows():
        total_mass = float(p.sum())
        if total_mass <= 1e-12: continue
        
        mu = float((p * strikes).sum() / total_mass)
        diffs = strikes - mu
        below, above = diffs[diffs < 0], diffs[diffs > 0]
        
        exclude = {c for c in [below.idxmax() if not below.empty else None, 
                               above.idxmin() if not above.empty else None] if c}
        tail_cols = [c for c in cols if c not in exclude]
        
        if tail_cols:
            out.loc[idx] = float(((p[tail_cols] * abs(strikes[tail_cols]) ** 2) / 35).sum())
        else:
            out.loc[idx] = 0.0
    return out


def build_asset_panel(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    
    for prefix in ["polymarket_", "kalshi_"]:
        out[f"{prefix}expected_bps"] = _expected_bps_from_distribution(out, prefix)
        out[f"{prefix}variance_asset"], _ = _calculate_moments(out, prefix)
        out[f"{prefix}tail_weight_bps"] = _tail_abs_weight(out, prefix)

    #base_assets = ["effr_expected_bps", "jump_sr1_bps", "jump_ois_bps"]
    base_assets = ["effr_expected_bps", "jump_sr1_bps"]
    prediction_all = [c for c in out.columns if c in PREDICTION_BPS_MAP]
    
    for col in prediction_all:
        out[col + "_bps"] = pd.to_numeric(out[col], errors="coerce") * PREDICTION_BPS_MAP[col]

    for plat in ["Poly", "Kalshi"]:
        plat_lower = "polymarket" if plat == "Poly" else "kalshi"
        out[f"{plat}_Hike_bps"] = out[[f"{c}_bps" for c in prediction_all if c.startswith(plat_lower) and PREDICTION_BPS_MAP[c] > 0]].sum(axis=1, min_count=1)
        out[f"{plat}_Cut_bps"] = out[[f"{c}_bps" for c in prediction_all if c.startswith(plat_lower) and PREDICTION_BPS_MAP[c] < 0]].sum(axis=1, min_count=1)

    mode_map = {
        "prediction_expected": base_assets + ["polymarket_expected_bps", "kalshi_expected_bps"],
        "prediction_grouped": base_assets + ["Poly_Hike_bps", "Poly_Cut_bps", "Kalshi_Hike_bps", "Kalshi_Cut_bps"],
        "prediction_all": base_assets + [c + "_bps" for c in prediction_all],
        "prediction_moments": base_assets + ["polymarket_expected_bps", "kalshi_expected_bps", "polymarket_variance_asset", "kalshi_variance_asset"],
        "prediction_moments2": base_assets + ["polymarket_expected_bps", "kalshi_expected_bps", "polymarket_tail_weight_bps", "kalshi_tail_weight_bps"]
    }
    
    if mode not in mode_map: raise ValueError(f"Unknown panel mode: {mode}")
    support_cols = [
        c
        for c in (["jump_sr1_portfolio_weights", "jump_ois_portfolio_weights"] + list(PREDICTION_BPS_MAP.keys()))
        if c in out.columns
    ]
    selected_cols = ["decision_date", "observed_day_pst"] + mode_map[mode]
    keep_cols = list(dict.fromkeys(selected_cols + support_cols))
    return out[keep_cols].copy().sort_values(["decision_date", "observed_day_pst"])


def _residual_weight_matrix(comps: np.ndarray, n_assets: int) -> np.ndarray:
    k = min(2, comps.shape[0])
    c = comps[:k, :]
    proj = c.T @ c if k > 0 else np.zeros((n_assets, n_assets), dtype=float)
    weights = np.eye(n_assets, dtype=float) - proj
    row_mean = np.nanmean(weights, axis=1, keepdims=True)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    return weights - row_mean


def run_pca(panel: pd.DataFrame, n_components: int = 3, normalize: bool = False, rolling_window_days: int | None = None):
    asset_cols = [c for c in panel.columns if c not in {"decision_date", "observed_day_pst"}]
    x_filled = _causal_impute_numeric(panel[asset_cols]).dropna(axis=1, how="all")
    if x_filled.empty: raise ValueError("No valid numeric assets available for PCA.")

    assets, n_factors = list(x_filled.columns), min(n_components, x_filled.shape[1])
    weight_map: dict[tuple[object, str], np.ndarray] = {}

    if not rolling_window_days or rolling_window_days <= 0:
        x_centered = x_filled - x_filled.mean()
        if normalize: x_centered = x_centered.div(x_centered.std(ddof=0).replace(0, 1.0), axis=1)
        
        pca = PCA(n_components=n_factors).fit(x_centered.values)
        factors = pca.transform(x_centered.values)
        factors_df = pd.DataFrame(factors, index=panel.index, columns=[f"F{i + 1}" for i in range(factors.shape[1])])
        recon = factors[:, :min(2, factors.shape[1])] @ pca.components_[:min(2, factors.shape[1]), :]
        residuals = pd.DataFrame(x_centered.values - recon, index=panel.index, columns=x_centered.columns)
        r2 = _rolling_r2_series(x_centered, residuals, window=None)
        static = _residual_weight_matrix(pca.components_, len(assets))
        for idx in panel.index:
            for asset_pos, asset_name in enumerate(assets):
                weight_map[(idx, asset_name)] = static[asset_pos, :].copy()
        return pca, factors_df, residuals, r2, weight_map, assets

    # Rolling PCA logic
    ordered = (
        panel[["observed_day_pst"]]
        .copy()
        .assign(orig_index=panel.index)
        .dropna(subset=["observed_day_pst"])
        .sort_values("observed_day_pst")
    )
    factor_rows = pd.DataFrame(np.nan, index=panel.index, columns=[f"F{i + 1}" for i in range(n_factors)])
    transformed = pd.DataFrame(np.nan, index=panel.index, columns=x_filled.columns)
    residuals = pd.DataFrame(np.nan, index=panel.index, columns=x_filled.columns)
    pca_last = None
    ordered_idx = ordered["orig_index"].to_numpy()
    starts = _rolling_window_starts(ordered["observed_day_pst"], int(rolling_window_days))
    ordered_pos = x_filled.index.get_indexer(ordered_idx)
    if (ordered_pos < 0).any():
        raise ValueError("Panel index alignment failed for rolling PCA.")
    x_values = x_filled.to_numpy(dtype=float)
    weight_cube = np.full((len(ordered_idx), len(assets), len(assets)), np.nan, dtype=float)

    def _rolling_snapshot(pos: int):
        row_idx = ordered_idx[pos]
        row_pos = int(ordered_pos[pos])
        window_pos = ordered_pos[starts[pos] : pos + 1]
        if len(window_pos) < max(MIN_PCA_ROLLING_OBSERVATIONS, n_factors):
            return None
        w_x = x_values[window_pos, :]
        w_mean = w_x.mean(axis=0)
        w_trans = w_x - w_mean
        if normalize:
            scale = w_trans.std(axis=0, ddof=0)
            scale = np.where(scale > 0.0, scale, 1.0)
            w_trans = w_trans / scale
            curr_trans = (x_values[row_pos, :] - w_mean) / scale
        else:
            curr_trans = x_values[row_pos, :] - w_mean

        pca_i = PCA(n_components=min(n_factors, w_trans.shape[1])).fit(w_trans)
        score = pca_i.transform(curr_trans.reshape(1, -1))[0]
        recon_factors = min(2, pca_i.components_.shape[0])
        residual = curr_trans - (score[:recon_factors] @ pca_i.components_[:recon_factors, :])
        weights = _residual_weight_matrix(pca_i.components_, len(assets))
        return pos, row_idx, score, curr_trans, residual, pca_i, weights

    n_workers = min(MAX_PCA_PARALLEL_WORKERS, max(1, (os.cpu_count() or 1)))
    if len(ordered_idx) >= max(MIN_PCA_PARALLEL_WINDOWS, PCA_PARALLEL_WORK_PER_WORKER * n_workers) and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            snapshots = list(executor.map(_rolling_snapshot, range(len(ordered_idx))))
    else:
        snapshots = [_rolling_snapshot(pos) for pos in range(len(ordered_idx))]

    for snapshot in snapshots:
        if snapshot is None:
            continue
        pos, row_idx, score, curr_trans, residual, pca_i, weights = snapshot
        residuals.loc[row_idx] = residual
        transformed.loc[row_idx] = curr_trans
        factor_rows.loc[row_idx, [f"F{i + 1}" for i in range(len(score))]] = score
        weight_cube[pos, :, :] = weights
        pca_last = pca_i

    if pca_last is None:
        pca_last = PCA(n_components=n_factors).fit(x_filled - x_filled.mean())

    for pos, row_idx in enumerate(ordered_idx):
        row_weights = weight_cube[pos]
        if not np.isfinite(row_weights).any():
            continue
        for asset_pos, asset_name in enumerate(assets):
            weight_map[(row_idx, asset_name)] = row_weights[asset_pos, :].copy()

    r2 = _rolling_r2_series(transformed, residuals, window=max(2, rolling_window_days or 0))
    return pca_last, factor_rows, residuals, r2, weight_map, assets


def _estimate_ou_params(window_values: np.ndarray) -> tuple[float, float, float] | None:
    s_t, s_t1 = window_values[:-1], window_values[1:]
    denom = np.sum((s_t - s_t.mean()) ** 2)
    if denom < OU_SINGULARITY_EPSILON: return None
    b = np.sum((s_t - s_t.mean()) * (s_t1 - s_t1.mean())) / denom
    if not np.isfinite(b) or b <= 0 or b >= (1.0 - OU_BOUNDARY_EPSILON): return None
    a = float(s_t1.mean() - b * s_t.mean())
    theta, mu = -math.log(b), (a / (1 - b))
    sigma_eps = float(np.std(s_t1 - (a + b * s_t), ddof=0))
    sigma = sigma_eps * math.sqrt(2 * theta / max(OU_SINGULARITY_EPSILON, 1 - b**2))
    return theta, mu, sigma


def filter_residuals(residuals: pd.DataFrame, r2_scores: pd.Series | pd.DataFrame, min_r2: float, max_rho: float, adf_alpha: float, max_half_life_days: float = 5.0, variance_threshold: float | None = None, variance_threshold_scale: float = 0.25, return_diagnostics: bool = False, estimation_window: int | None = None):
    window = max(MIN_ROLLING_OU_POINTS, int(estimation_window) if estimation_window and estimation_window > 1 else 2 * MIN_ROLLING_OU_POINTS)
    spread_std = residuals.rolling(window=window, min_periods=MIN_ROLLING_OU_POINTS).std(ddof=0)
    half_life = pd.DataFrame(np.nan, index=residuals.index, columns=residuals.columns, dtype=float)
    ou_sigma = pd.DataFrame(np.nan, index=residuals.index, columns=residuals.columns, dtype=float)
    diag_rows = []

    if isinstance(r2_scores, pd.DataFrame):
        r2_df = r2_scores.reindex(index=residuals.index, columns=residuals.columns)
    else:
        r2_df = pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float)
        for col in residuals.columns:
            r2_df[col] = float(r2_scores.get(col, np.nan))

    for col in residuals.columns:
        s = residuals[col]
        for end_idx in range(len(s)):
            w_vals = s.iloc[max(0, end_idx - window + 1) : end_idx + 1].dropna().to_numpy(dtype=float)
            if len(w_vals) < MIN_ROLLING_OU_POINTS:
                continue
            params = _estimate_ou_params(w_vals)
            if params and params[0] > 0:
                half_life.iloc[end_idx, half_life.columns.get_loc(col)] = float(np.log(2.0) / params[0])
                ou_sigma.iloc[end_idx, ou_sigma.columns.get_loc(col)] = float(params[2])

    if variance_threshold is not None:
        var_gate = spread_std > variance_threshold
    else:
        cross_med = spread_std.median(axis=1)
        var_gate = spread_std.gt(variance_threshold_scale * cross_med, axis=0)
    tradable = ((r2_df > min_r2) & (half_life < max_half_life_days) & var_gate).fillna(False)

    for col in residuals.columns:
        s_eval = residuals[col].dropna().iloc[-window:]
        adf_stat, adf_p = (np.nan, np.nan)
        if len(s_eval) >= MIN_ROLLING_OU_POINTS:
            try:
                adf_stat, adf_p = adfuller(s_eval.values, maxlag=1, autolag=None)[:2]
            except Exception:
                pass
        hl_last = half_life[col].dropna()
        ou_sigma_last = ou_sigma[col].dropna()
        std_last = spread_std[col].dropna()
        r2_last = r2_df[col].dropna()
        r2_last_val = float(r2_last.iloc[-1]) if len(r2_last) else np.nan
        hl_last_val = float(hl_last.iloc[-1]) if len(hl_last) else np.nan
        std_last_val = float(std_last.iloc[-1]) if len(std_last) else np.nan
        r2_pass = bool(pd.notna(r2_last_val) and r2_last_val > min_r2)
        hl_pass = bool(pd.notna(hl_last_val) and hl_last_val < max_half_life_days)
        if variance_threshold is not None:
            var_pass = bool(pd.notna(std_last_val) and std_last_val > variance_threshold)
        else:
            med_last = float(spread_std.iloc[-1].median()) if len(spread_std) else np.nan
            var_pass = bool(pd.notna(std_last_val) and pd.notna(med_last) and std_last_val > (variance_threshold_scale * med_last))
        sel_last = bool(tradable[col].iloc[-1]) if len(tradable) else False
        diag_rows.append(
            {
                "asset": col,
                "R2": r2_last_val,
                "OU_kappa": float(np.log(2.0) / hl_last.iloc[-1]) if len(hl_last) and hl_last.iloc[-1] > 0 else np.nan,
                "OU_sigma": float(ou_sigma_last.iloc[-1]) if len(ou_sigma_last) else np.nan,
                "half_life": hl_last_val,
                "spread_std": std_last_val,
                "ADF_stat": float(adf_stat) if pd.notna(adf_stat) else np.nan,
                "ADF_pvalue": float(adf_p) if pd.notna(adf_p) else np.nan,
                "r2_pass": r2_pass,
                "half_life_pass": hl_pass,
                "variance_pass": var_pass,
                "selected": sel_last,
            }
        )

    return (tradable, pd.DataFrame(diag_rows)) if return_diagnostics else tradable


def run_ou_strategy(panel: pd.DataFrame, residuals: pd.DataFrame, assets: list[str], config: StrategyConfig, tradable: pd.DataFrame | None = None, return_diagnostics: bool = False):
    trades, diags = [], []
    work = panel[["decision_date", "observed_day_pst"]].copy().sort_values(["decision_date", "observed_day_pst"])
    work["days_to_decision"] = (work["decision_date"] - work["observed_day_pst"]).dt.days
    decision_keys = work["decision_date"]
    daily_asset_pnl_rows = []
    annualization = _infer_annualization_periods(work["observed_day_pst"])
    min_hist_points = max(10, config.ou_window // 2)
    pair_window = max(1, config.ou_window - 1)
    min_pair_points = max(1, min_hist_points - 1)

    s_all = residuals.reindex(work.index)[assets].apply(pd.to_numeric, errors="coerce")
    x_pairs = s_all.groupby(decision_keys).shift(1)
    y_pairs = s_all
    
    x_mean = x_pairs.groupby(decision_keys).rolling(window=pair_window, min_periods=min_pair_points).mean().reset_index(level=0, drop=True)
    y_mean = y_pairs.groupby(decision_keys).rolling(window=pair_window, min_periods=min_pair_points).mean().reset_index(level=0, drop=True)
    cov_xy = (x_pairs * y_pairs).groupby(decision_keys).rolling(window=pair_window, min_periods=min_pair_points).mean().reset_index(level=0, drop=True) - (x_mean * y_mean)
    var_x = (x_pairs * x_pairs).groupby(decision_keys).rolling(window=pair_window, min_periods=min_pair_points).mean().reset_index(level=0, drop=True) - (x_mean * x_mean)
    var_y = (y_pairs * y_pairs).groupby(decision_keys).rolling(window=pair_window, min_periods=min_pair_points).mean().reset_index(level=0, drop=True) - (y_mean * y_mean)
    
    b_raw = cov_xy.div(var_x.where(var_x >= OU_SINGULARITY_EPSILON))
    valid_b = np.isfinite(b_raw) & (b_raw > 0.0) & (b_raw < (1.0 - OU_BOUNDARY_EPSILON))
    b_raw = b_raw.where(valid_b)
    a_raw = (y_mean - b_raw * x_mean).where(valid_b)
    theta_raw = (-np.log(b_raw)).where(valid_b)
    mu_raw = (a_raw / (1.0 - b_raw)).where(valid_b)
    
    sigma_eps_sq = (var_y - b_raw * cov_xy).clip(lower=0.0)
    sigma_eps = np.sqrt(sigma_eps_sq)
    sigma_raw = (sigma_eps * np.sqrt((2.0 * theta_raw).clip(lower=0.0) / (1.0 - b_raw.pow(2)).clip(lower=OU_SINGULARITY_EPSILON))).where(valid_b)

    mu = mu_raw.groupby(decision_keys).shift(1)
    sigma = sigma_raw.groupby(decision_keys).shift(1)
    kappa = theta_raw.groupby(decision_keys).shift(1)
    
    upper = mu + config.entry_sigma * sigma
    lower = mu - config.entry_sigma * sigma
    ex_upper = mu + config.exit_sigma * sigma
    ex_lower = mu - config.exit_sigma * sigma

    grouped_indices = [
        work.loc[idx].sort_values("observed_day_pst").index.to_numpy()
        for idx in work.groupby("decision_date").groups.values()
    ]

    for asset in assets:
        s = residuals[asset].reindex(work.index)
        tradable_s = (
            tradable[asset].reindex(work.index).fillna(False)
            if tradable is not None and asset in tradable.columns
            else pd.Series(True, index=work.index)
        )
        for ddf_idx in grouped_indices:
            pos, entry, entry_time = 0.0, np.nan, pd.NaT
            prev_mtm = 0.0
            
            for i in ddf_idx:
                curr = s.loc[i]
                if pd.isna(curr): continue

                if pos != 0 and pd.notna(entry):
                    mtm = float(pos * (curr - entry))
                    daily_asset_pnl_rows.append({"asset": asset, "observed_day_pst": work.at[i, "observed_day_pst"], "daily_pnl": mtm - prev_mtm})
                    prev_mtm = mtm
                
                if work.at[i, "days_to_decision"] <= config.no_trade_days_before_decision:
                    if pos != 0 and pd.notna(entry):
                        trades.append({"asset": asset, "decision_date": work.at[i, "decision_date"], "observed_day_pst": work.at[i, "observed_day_pst"], "event": "forced_exit", "position": 0, "pnl_bps": prev_mtm})
                        pos, entry, entry_time, prev_mtm = 0.0, np.nan, pd.NaT, 0.0
                    continue

                mu_i, sigma_i = mu.at[i, asset], sigma.at[i, asset]
                kappa_i = kappa.at[i, asset]
                
                # Check for valid OU parameters before considering entry
                if not (pd.notna(mu_i) and pd.notna(sigma_i) and pd.notna(kappa_i) and sigma_i > 0 and kappa_i > 0):
                    continue

                timed_out = (
                    pos != 0
                    and pd.notna(entry_time)
                    and config.max_holding_days > 0
                    and (pd.Timestamp(work.at[i, "observed_day_pst"]) - pd.Timestamp(entry_time)).days >= config.max_holding_days
                )
                if timed_out:
                    trades.append({"asset": asset, "decision_date": work.at[i, "decision_date"], "observed_day_pst": work.at[i, "observed_day_pst"], "event": "time_stop_exit", "position": 0, "pnl_bps": prev_mtm})
                    pos, entry, entry_time, prev_mtm = 0.0, np.nan, pd.NaT, 0.0
                    continue

                if pos == 0:
                    if not bool(tradable_s.loc[i]):
                        continue
                    # Fixed unit size: 1.0
                    if curr >= upper.at[i, asset]: pos, entry, ev = -1.0, curr, "enter_short"
                    elif curr <= lower.at[i, asset]: pos, entry, ev = 1.0, curr, "enter_long"
                    if pos != 0:
                        entry_time = pd.Timestamp(work.at[i, "observed_day_pst"])
                        prev_mtm = 0.0
                        trades.append({"asset": asset, "decision_date": work.at[i, "decision_date"], "observed_day_pst": work.at[i, "observed_day_pst"], "event": ev, "position": pos, "pnl_bps": 0.0})
                
                elif (pos > 0 and curr >= ex_lower.at[i, asset]) or (pos < 0 and curr <= ex_upper.at[i, asset]):
                    trades.append({"asset": asset, "decision_date": work.at[i, "decision_date"], "observed_day_pst": work.at[i, "observed_day_pst"], "event": "exit_long" if pos > 0 else "exit_short", "position": 0, "pnl_bps": prev_mtm})
                    pos, entry, entry_time, prev_mtm = 0.0, np.nan, pd.NaT, 0.0
                
            if pos != 0 and pd.notna(entry):
                last_idx = ddf_idx[-1]
                trades.append({"asset": asset, "decision_date": work.at[last_idx, "decision_date"], "observed_day_pst": work.at[last_idx, "observed_day_pst"], "event": "forced_exit", "position": 0, "pnl_bps": prev_mtm})

            if return_diagnostics:
                hist_arr = s.loc[ddf_idx].dropna().to_numpy(dtype=float)
                if len(hist_arr) >= config.ou_window:
                    hist_arr = hist_arr[-config.ou_window:]
                diag_idx = None
                for idx in ddf_idx[::-1]:
                    if pd.notna(mu.at[idx, asset]) and pd.notna(kappa.at[idx, asset]) and pd.notna(sigma.at[idx, asset]):
                        diag_idx = idx
                        break
                if diag_idx is not None:
                    adf_stat, adf_p = np.nan, np.nan
                    try:
                        if len(hist_arr) >= 10 and np.nanstd(hist_arr) > OU_SINGULARITY_EPSILON:
                            adf_stat, adf_p = adfuller(hist_arr, maxlag=1, autolag="AIC")[:2]
                    except Exception:
                        pass
                    diags.append(
                        {
                            "asset": asset,
                            "decision_date": work.at[ddf_idx[-1], "decision_date"],
                            "ADF_stat": float(adf_stat) if pd.notna(adf_stat) else np.nan,
                            "ADF_pvalue": float(adf_p) if pd.notna(adf_p) else np.nan,
                            "OU_kappa": float(kappa.at[diag_idx, asset]),
                            "OU_mu": float(mu.at[diag_idx, asset]),
                            "OU_sigma": float(sigma.at[diag_idx, asset]),
                            "half_life": float(math.log(2) / kappa.at[diag_idx, asset]) if kappa.at[diag_idx, asset] > 0 else np.nan,
                            "spread_std": float(np.nanstd(hist_arr)),
                        }
                    )

    trades_df = pd.DataFrame(trades)
    metrics = pd.DataFrame({"asset": assets, "sharpe": np.nan, "annualized_sharpe": np.nan, "entries": 0, "exits": 0, "avg_profit_per_trade": 0.0, "total_profit_bps": 0.0})
    
    if not trades_df.empty:
        entries = trades_df[trades_df["event"].str.startswith("enter")].groupby("asset").size()
        exits = trades_df[trades_df["event"].str.contains("exit")].groupby("asset")["pnl_bps"].agg(["count", "mean", "sum", "std"])
        metrics["entries"] = metrics["asset"].map(entries).fillna(0).astype(int)
        metrics["exits"] = metrics["asset"].map(exits["count"]).fillna(0).astype(int)
        metrics["avg_profit_per_trade"] = metrics["asset"].map(exits["mean"]).fillna(0.0)
        metrics["total_profit_bps"] = metrics["asset"].map(exits["sum"]).fillna(0.0)
    
    if daily_asset_pnl_rows:
        daily_asset = pd.DataFrame(daily_asset_pnl_rows).groupby(["asset", "observed_day_pst"], as_index=False)["daily_pnl"].sum()
        sharpe_map, ann_map = {}, {}
        for asset, g in daily_asset.groupby("asset"):
            s_pnl = g["daily_pnl"].astype(float)
            daily_std = float(s_pnl.std(ddof=0))
            sh = float(s_pnl.mean() / daily_std) if len(s_pnl) > 1 and daily_std > 0 else 0.0
            sharpe_map[asset] = sh
            ann_map[asset] = float(sh * math.sqrt(annualization))
        metrics["sharpe"] = metrics["asset"].map(sharpe_map).fillna(0.0)
        metrics["annualized_sharpe"] = metrics["asset"].map(ann_map).fillna(0.0)

    return (trades_df, metrics, pd.DataFrame(diags)) if return_diagnostics else (trades_df, metrics)

def compute_residual_basket_weights(panel: pd.DataFrame, n_components: int = 3, normalize: bool = False, rolling_window_days: int | None = None):
    try:
        _, _, _, _, weight_map, assets = run_pca(
            panel=panel,
            n_components=n_components,
            normalize=False,
            rolling_window_days=rolling_window_days,
        )
    except ValueError:
        return {}, []
    if normalize:
        normalized_map: dict[tuple[object, str], np.ndarray] = {}
        for key, w in weight_map.items():
            scale = np.nansum(np.abs(w))
            normalized_map[key] = (w / scale) if scale > OU_SINGULARITY_EPSILON else w
        return normalized_map, assets
    return weight_map, assets


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_parse_weight_dict(raw: object) -> dict[str, float]:
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
    else:
        return {}
    out: dict[str, float] = {}
    for k, v in parsed.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out[_normalize_ois_ticker(str(k))] = f
    return out


_OIS_TICKER_PATTERN = re.compile(r"^OIS_([^M]+)M_([0-9]+)$")


def _format_ois_tenor_months(value: float, precision: int = 5) -> str:
    rounded = float(np.round(float(value), precision))
    text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _normalize_ois_ticker(ticker: str) -> str:
    match = _OIS_TICKER_PATTERN.match(str(ticker))
    if not match:
        return str(ticker)
    tenor_raw, idx_raw = match.groups()
    try:
        tenor_val = float(tenor_raw)
        idx_val = int(idx_raw)
    except (TypeError, ValueError):
        return str(ticker)
    return f"OIS_{_format_ois_tenor_months(tenor_val)}M_{idx_val}"


def _load_ois_underlying_prices_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    csv = pd.read_csv(path, low_memory=False)
    if csv.empty:
        return pd.DataFrame()

    if {"observed_day_pst", "ticker", "rate"}.issubset(csv.columns):
        csv["observed_day_pst"] = pd.to_datetime(csv["observed_day_pst"], errors="coerce")
        csv["ticker"] = csv["ticker"].astype(str).map(_normalize_ois_ticker)
        csv["rate"] = pd.to_numeric(csv["rate"], errors="coerce")
        csv = csv.dropna(subset=["observed_day_pst", "ticker", "rate"])
        if csv.empty:
            return pd.DataFrame()
        out = csv.pivot(index="observed_day_pst", columns="ticker", values="rate").sort_index()
    else:
        if "observed_day_pst" not in csv.columns:
            csv = csv.rename(columns={csv.columns[0]: "observed_day_pst"})
        csv["observed_day_pst"] = pd.to_datetime(csv["observed_day_pst"], errors="coerce")
        csv = csv.dropna(subset=["observed_day_pst"]).set_index("observed_day_pst")
        csv.columns = [_normalize_ois_ticker(c) for c in csv.columns]
        out = csv.apply(pd.to_numeric, errors="coerce").sort_index()
    out.columns.name = None
    return out


def _effr_contract_weights(decision_date: pd.Timestamp, late_month_threshold: float = 0.67) -> dict[str, float]:
    meeting = pd.Timestamp(decision_date)
    if pd.isna(meeting):
        return {}
    m_day, m_year, m_month = meeting.day, meeting.year, meeting.month
    days_in_month = calendar.monthrange(m_year, m_month)[1]
    late_month = (m_day / days_in_month) > late_month_threshold
    if late_month:
        next_year, next_month = (m_year + 1, 1) if m_month == 12 else (m_year, m_month + 1)
        return {
            f"EFFR_{m_year:04d}_{m_month:02d}": -10000.0,
            f"EFFR_{next_year:04d}_{next_month:02d}": 10000.0,
        }

    denom = days_in_month - m_day
    if denom <= 0:
        return {}
    scale = days_in_month / denom
    prev_year, prev_month = (m_year - 1, 12) if m_month == 1 else (m_year, m_month - 1)
    return {
        f"EFFR_{prev_year:04d}_{prev_month:02d}": -10000.0 * scale,
        f"EFFR_{m_year:04d}_{m_month:02d}": 10000.0 * scale,
    }


def _prediction_contract_weights(asset: str, panel_columns: list[str]) -> dict[str, float]:
    if asset in PREDICTION_BPS_MAP:
        return {asset: 1.0}
    if asset.endswith("_bps") and asset[:-4] in PREDICTION_BPS_MAP:
        leg = asset[:-4]
        return {leg: float(PREDICTION_BPS_MAP[leg])}

    def _legs(prefix: str, sign: str | None = None) -> dict[str, float]:
        out = {}
        for c in panel_columns:
            if c.startswith(prefix) and c in PREDICTION_BPS_MAP:
                strike = float(PREDICTION_BPS_MAP[c])
                if sign == "positive" and strike <= 0:
                    continue
                if sign == "negative" and strike >= 0:
                    continue
                out[c] = strike
        return out

    if asset == "polymarket_expected_bps":
        return _legs("polymarket_")
    if asset == "kalshi_expected_bps":
        return _legs("kalshi_")
    if asset == "Poly_Hike_bps":
        return _legs("polymarket_", sign="positive")
    if asset == "Poly_Cut_bps":
        return _legs("polymarket_", sign="negative")
    if asset == "Kalshi_Hike_bps":
        return _legs("kalshi_", sign="positive")
    if asset == "Kalshi_Cut_bps":
        return _legs("kalshi_", sign="negative")
    return {}


def _prediction_tail_contract_weights(asset: str, row: pd.Series) -> dict[str, float]:
    if asset == "polymarket_tail_weight_bps":
        prefix = "polymarket_"
    elif asset == "kalshi_tail_weight_bps":
        prefix = "kalshi_"
    else:
        return {}

    cols = [c for c in row.index if c.startswith(prefix) and c in PREDICTION_BPS_MAP]
    if not cols:
        return {}
    probs = pd.to_numeric(row[cols], errors="coerce")
    total_mass = float(probs.fillna(0.0).sum())
    if total_mass <= OU_SINGULARITY_EPSILON:
        return {}
    strikes = pd.Series({c: float(PREDICTION_BPS_MAP[c]) for c in cols}, dtype=float)
    mu = float((probs.fillna(0.0) * strikes).sum() / total_mass)
    diffs = strikes - mu
    below, above = diffs[diffs < 0], diffs[diffs > 0]
    exclude = {c for c in [below.idxmax() if not below.empty else None, above.idxmin() if not above.empty else None] if c}
    tail_cols = [c for c in cols if c not in exclude]
    return {c: float((abs(strikes[c]) ** 2) / 35.0) for c in tail_cols}


def _map_composite_to_underlyings(asset: str, row: pd.Series) -> dict[str, float]:
    if asset == "jump_sr1_bps":
        return {k: 10000.0 * v for k, v in _safe_parse_weight_dict(row.get("jump_sr1_portfolio_weights")).items()}
    if asset == "jump_ois_bps":
        return {k: 10000.0 * v for k, v in _safe_parse_weight_dict(row.get("jump_ois_portfolio_weights")).items()}
    if asset == "effr_expected_bps":
        return _effr_contract_weights(pd.Timestamp(row.get("decision_date")))
    tail = _prediction_tail_contract_weights(asset, row)
    if tail:
        return tail
    pred = _prediction_contract_weights(asset, list(row.index))
    if pred:
        return pred
    return {asset: 1.0}


@lru_cache(maxsize=2)
def _load_sr1_underlying_prices(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    sr1 = pd.read_csv(path, usecols=["Date_", "Settlement", "LastTrdDate", "Volume", "OpenInterest"], low_memory=False)
    sr1["observed_day_pst"] = pd.to_datetime(sr1["Date_"], errors="coerce")
    sr1["last_trade"] = pd.to_datetime(sr1["LastTrdDate"], errors="coerce")
    sr1["price"] = (100.0 - pd.to_numeric(sr1["Settlement"], errors="coerce")) / 100.0
    sr1["Volume"] = pd.to_numeric(sr1["Volume"], errors="coerce")
    sr1["OpenInterest"] = pd.to_numeric(sr1["OpenInterest"], errors="coerce")
    sr1 = sr1.dropna(subset=["observed_day_pst", "last_trade", "price"])
    sr1["ticker"] = "SR1:" + sr1["last_trade"].dt.to_period("M").astype(str)
    sr1 = sr1.sort_values(["observed_day_pst", "ticker", "Volume", "OpenInterest"]).groupby(["observed_day_pst", "ticker"], as_index=False).tail(1)
    out = sr1.pivot(index="observed_day_pst", columns="ticker", values="price").sort_index()
    out.columns.name = None
    return out


@lru_cache(maxsize=2)
def _load_effr_underlying_prices(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    effr = pd.read_csv(path, usecols=["Date_", "Settlement", "LastTrdDate", "Volume", "OpenInterest"], low_memory=False)
    effr["observed_day_pst"] = pd.to_datetime(effr["Date_"], errors="coerce")
    effr["last_trade"] = pd.to_datetime(effr["LastTrdDate"], errors="coerce")
    effr["price"] = (100.0 - pd.to_numeric(effr["Settlement"], errors="coerce")) / 100.0
    effr["Volume"] = pd.to_numeric(effr["Volume"], errors="coerce")
    effr["OpenInterest"] = pd.to_numeric(effr["OpenInterest"], errors="coerce")
    effr = effr.dropna(subset=["observed_day_pst", "last_trade", "price"])
    effr["ticker"] = "EFFR_" + effr["last_trade"].dt.strftime("%Y_%m")
    effr = effr.sort_values(["observed_day_pst", "ticker", "Volume", "OpenInterest"]).groupby(["observed_day_pst", "ticker"], as_index=False).tail(1)
    out = effr.pivot(index="observed_day_pst", columns="ticker", values="price").sort_index()
    out.columns.name = None
    return out


@lru_cache(maxsize=2)
def _load_ois_underlying_prices(path: str) -> pd.DataFrame:
    csv_path = str(Path(path).with_suffix(".csv"))
    csv_prices = _load_ois_underlying_prices_csv(csv_path)
    if not csv_prices.empty:
        return csv_prices
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        ois_raw = pd.read_excel(path, header=None)
    except ImportError:
        warnings.warn(
            f"Unable to import Excel dependencies for {path}; skipping OIS underlying prices. "
            "Add Data/SOFR/SOFR OIS.csv to avoid this dependency and improve load speed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()
    records: list[dict[str, float | pd.Timestamp]] = []
    for col in range(max(0, ois_raw.shape[1] - 1)):
        code = ois_raw.iloc[0, col]
        if ois_raw.iloc[3, col + 1] != "CLOSE":
            continue
        tenor = parse_ois_tenor(code)
        if tenor is None:
            continue
        dates = pd.to_datetime(ois_raw.iloc[4:, col], errors="coerce")
        rates = pd.to_numeric(ois_raw.iloc[4:, col + 1], errors="coerce") / 100.0
        normalized_dates = dates.dt.normalize()
        unique_dates = pd.DatetimeIndex(normalized_dates.dropna().unique())
        tenor_months_map = {
            ts: 12.0 * year_fraction_act360(ts.date(), tenor_to_maturity(ts.date(), tenor))
            for ts in unique_dates
        }
        frame = pd.DataFrame(
            {
                "observed_day_pst": normalized_dates,
                "tenor_months": normalized_dates.map(tenor_months_map),
                "rate": rates,
            }
        ).dropna(subset=["observed_day_pst", "tenor_months", "rate"])
        if frame.empty:
            continue
        records.extend(frame.to_dict(orient="records"))
    if not records:
        return pd.DataFrame()
    ois = pd.DataFrame(records).sort_values(["observed_day_pst", "tenor_months"])
    ois["idx"] = ois.groupby("observed_day_pst").cumcount()
    ois["ticker"] = ois.apply(
        lambda row: f"OIS_{_format_ois_tenor_months(float(row['tenor_months']))}M_{int(row['idx'])}",
        axis=1,
    )
    out = ois.pivot(index="observed_day_pst", columns="ticker", values="rate").sort_index()
    out.columns.name = None
    return out


def _build_underlying_price_lookup(panel: pd.DataFrame, asset_cols: list[str], underlying_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    panel_with_dates = panel.copy()
    panel_with_dates["observed_day_pst"] = pd.to_datetime(panel_with_dates["observed_day_pst"], errors="coerce")
    panel_with_dates = panel_with_dates.dropna(subset=["observed_day_pst"])

    panel_base_cols = [c for c in set(asset_cols) | set(PREDICTION_BPS_MAP.keys()) if c in panel_with_dates.columns]
    panel_prices = (
        panel_with_dates.sort_values(["decision_date", "observed_day_pst"])
        .groupby("observed_day_pst", as_index=True)[panel_base_cols]
        .last()
        .apply(pd.to_numeric, errors="coerce")
        if panel_base_cols
        else pd.DataFrame(index=panel_with_dates["observed_day_pst"].drop_duplicates().sort_values())
    )

    if underlying_prices is not None:
        supplied = underlying_prices.copy()
        if "observed_day_pst" in supplied.columns:
            supplied["observed_day_pst"] = pd.to_datetime(supplied["observed_day_pst"], errors="coerce")
            supplied = supplied.dropna(subset=["observed_day_pst"]).set_index("observed_day_pst")
        supplied.index = pd.to_datetime(supplied.index, errors="coerce")
        supplied = supplied[~supplied.index.isna()].apply(pd.to_numeric, errors="coerce")
        overlap = [c for c in supplied.columns if c in panel_prices.columns]
        if overlap:
            panel_prices = panel_prices.drop(columns=overlap)
        out = panel_prices.join(supplied, how="outer")
        return out.sort_index().ffill()

    need_sr1 = "jump_sr1_bps" in asset_cols
    need_ois = "jump_ois_bps" in asset_cols
    need_effr = "effr_expected_bps" in asset_cols

    raw_frames = []
    repo = _repo_root()
    if need_sr1:
        raw_frames.append(_load_sr1_underlying_prices(str(repo / "Data" / "SOFR" / "SR1.csv")))
    if need_ois:
        raw_frames.append(_load_ois_underlying_prices(str(repo / "Data" / "SOFR" / "SOFR OIS.xlsx")))
    if need_effr:
        raw_frames.append(_load_effr_underlying_prices(str(repo / "Data" / "EFFR_Futures" / "effr_futures.csv")))

    out = panel_prices
    for frame in raw_frames:
        if frame.empty:
            continue
        out = out.join(frame, how="outer")
    return out.sort_index().ffill()


def run_basket_backtest(
    panel: pd.DataFrame,
    signal_trades: pd.DataFrame,
    basket_weights: dict,
    asset_cols: list[str],
    underlying_prices: pd.DataFrame | None = None,
    execution_lag_days: int = 0,
):
    if signal_trades.empty or not asset_cols:
        return pd.DataFrame(columns=["observed_day_pst", "daily_pnl"]), pd.DataFrame(), pd.DataFrame(columns=["observed_day_pst", "cumulative_pnl"])
    if execution_lag_days < 0:
        raise ValueError("execution_lag_days must be >= 0")

    ordered = panel.sort_values(["decision_date", "observed_day_pst"])
    price_lookup = _build_underlying_price_lookup(ordered, asset_cols, underlying_prices=underlying_prices)
    aligned_prices = price_lookup.reindex(pd.to_datetime(ordered["observed_day_pst"], errors="coerce")).ffill()
    aligned_prices.index = ordered.index
    prices = aligned_prices.to_dict("index")
    key_idx = {(pd.Timestamp(r.decision_date), pd.Timestamp(r.observed_day_pst)): int(r.Index) for r in ordered.itertuples()}
    date_idx = {pd.Timestamp(d): g.sort_values("observed_day_pst").index.tolist() for d, g in ordered.groupby("decision_date")}

    def _resolve_execution_index(decision_date: pd.Timestamp, signal_time: pd.Timestamp) -> int | None:
        idx_path = date_idx.get(decision_date, [])
        if not idx_path:
            return None
        target_time = pd.Timestamp(signal_time) + pd.Timedelta(days=int(execution_lag_days))
        for i in idx_path:
            obs = pd.Timestamp(ordered.loc[i, "observed_day_pst"])
            if pd.isna(obs):
                continue
            if obs >= target_time:
                return int(i)
        return None

    trade_rows, daily_rows, active = [], [], {}
    warned_ois_mapping = False

    for row in signal_trades.sort_values(["asset", "decision_date", "observed_day_pst"]).itertuples():
        key = (str(row.asset), pd.Timestamp(row.decision_date))
        
        if str(row.event).startswith("enter"):
            signal_idx = key_idx.get((key[1], pd.Timestamp(row.observed_day_pst)))
            execution_idx = _resolve_execution_index(key[1], pd.Timestamp(row.observed_day_pst))
            weights = None if signal_idx is None else basket_weights.get((signal_idx, key[0]))
            if signal_idx is None or execution_idx is None or weights is None:
                continue

            weights_arr = np.array(weights, dtype=float)
            assert weights_arr.shape == (len(asset_cols),), "Basket weights must align with asset_count."
            assert np.isfinite(weights_arr).any(), "Basket weights must include at least one finite value."
            row_data = ordered.loc[signal_idx]
            entry_map = prices.get(execution_idx, {})
            underlying_qty: dict[str, float] = {}
            for comp_asset, comp_weight in zip(asset_cols, weights_arr):
                if not np.isfinite(comp_weight) or abs(float(comp_weight)) <= OU_SINGULARITY_EPSILON:
                    continue
                leg_weights = _map_composite_to_underlyings(str(comp_asset), row_data)
                if (
                    not warned_ois_mapping
                    and str(comp_asset) == "jump_ois_bps"
                    and leg_weights
                    and not set(leg_weights).intersection(entry_map.keys())
                ):
                    warnings.warn(
                        "No jump_ois_bps leg tickers matched available underlying OIS prices. "
                        "Check OIS ticker normalization between portfolio weights and price columns.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_ois_mapping = True
                for ticker, leg_weight in leg_weights.items():
                    if not np.isfinite(leg_weight):
                        continue
                    underlying_qty[ticker] = underlying_qty.get(ticker, 0.0) + float(comp_weight) * float(leg_weight)

            valid_qty = {
                ticker: qty
                for ticker, qty in underlying_qty.items()
                if np.isfinite(qty) and np.isfinite(entry_map.get(ticker, np.nan))
            }
            if valid_qty:
                active[key] = {
                    "asset": key[0],
                    "decision_date": key[1],
                    "entry_time": pd.Timestamp(ordered.loc[execution_idx, "observed_day_pst"]),
                    "position": int(row.position),
                    "quantities": valid_qty,
                    "entry_prices": {ticker: float(entry_map[ticker]) for ticker in valid_qty},
                    "prev_mtm": 0.0,
                }
        
        elif ("exit" in str(row.event)) and key in active:
            t = active.pop(key)
            execution_idx = _resolve_execution_index(key[1], pd.Timestamp(row.observed_day_pst))
            if execution_idx is None:
                active[key] = t
                continue
            exit_time = pd.Timestamp(ordered.loc[execution_idx, "observed_day_pst"])
            if exit_time < t["entry_time"]:
                active[key] = t
                continue
            path = [i for i in date_idx.get(key[1], []) if t["entry_time"] <= ordered.loc[i, "observed_day_pst"] <= exit_time]
            
            last_mtm = 0.0
            for i in path:
                cur_day_prices = prices.get(i, {})
                mtm_components = 0.0
                for ticker, qty in t["quantities"].items():
                    entry_price = float(t["entry_prices"][ticker])
                    current_price = cur_day_prices.get(ticker, np.nan)
                    if not np.isfinite(current_price):
                        current_price = entry_price
                    mtm_components += float(qty) * (float(current_price) - entry_price)
                mtm = float(t["position"] * mtm_components)
                daily_rows.append({"observed_day_pst": ordered.loc[i, "observed_day_pst"], "daily_pnl": mtm - t["prev_mtm"]})
                t["prev_mtm"], last_mtm = mtm, mtm
                
            trade_rows.append({"asset": t["asset"], "decision_date": t["decision_date"], "entry_time": t["entry_time"], "exit_time": exit_time, "position": t["position"], "entry_weights": json.dumps(t["quantities"], sort_keys=True), "entry_prices": json.dumps(t["entry_prices"], sort_keys=True), "trade_pnl": last_mtm, "weights_constant_during_trade": True})

    daily_pnl = pd.DataFrame(daily_rows).groupby("observed_day_pst", as_index=False)["daily_pnl"].sum().sort_values("observed_day_pst") if daily_rows else pd.DataFrame(columns=["observed_day_pst", "daily_pnl"])
    cum_pnl = daily_pnl.assign(cumulative_pnl=daily_pnl["daily_pnl"].cumsum())[["observed_day_pst", "cumulative_pnl"]] if not daily_pnl.empty else pd.DataFrame(columns=["observed_day_pst", "cumulative_pnl"])
    
    trade_log = pd.DataFrame(trade_rows, columns=TRADE_LOG_COLUMNS) if trade_rows else pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    return daily_pnl, trade_log, cum_pnl


def calculate_transaction_costs(trade_log: pd.DataFrame) -> dict[str, object]:
    if trade_log.empty:
        empty_series = pd.Series(dtype=float)
        return {
            "total_commissions": 0.0,
            "commission_by_day": empty_series,
            "commission_by_ticker": empty_series,
            "commission_by_platform": empty_series,
            "commission_by_trade": empty_series,
            "investigation": pd.DataFrame(),
        }

    events: list[dict[str, object]] = []
    for trade_id, row in enumerate(trade_log.itertuples()):
        qty_map = _safe_parse_weight_dict(getattr(row, "entry_weights", None))
        price_map = _safe_parse_weight_dict(getattr(row, "entry_prices", None))
        if not qty_map:
            continue
        position = int(getattr(row, "position", 0))
        decision_date = pd.Timestamp(getattr(row, "decision_date", pd.NaT))
        entry_time = pd.Timestamp(getattr(row, "entry_time", pd.NaT))
        exit_time = pd.Timestamp(getattr(row, "exit_time", pd.NaT))
        for ticker, qty in qty_map.items():
            signed_qty = float(position) * float(qty)
            ref_price = float(price_map.get(ticker, 0.5))
            if pd.notna(entry_time):
                events.append(
                    {
                        "trade_id": trade_id,
                        "time": entry_time,
                        "decision_date": decision_date,
                        "ticker": ticker,
                        "delta_qty": signed_qty,
                        "ref_price": ref_price,
                    }
                )
            if pd.notna(exit_time):
                events.append(
                    {
                        "trade_id": trade_id,
                        "time": exit_time,
                        "decision_date": decision_date,
                        "ticker": ticker,
                        "delta_qty": -signed_qty,
                        "ref_price": ref_price,
                    }
                )
    if not events:
        empty_series = pd.Series(dtype=float)
        return {
            "total_commissions": 0.0,
            "commission_by_day": empty_series,
            "commission_by_ticker": empty_series,
            "commission_by_platform": empty_series,
            "commission_by_trade": empty_series,
            "investigation": pd.DataFrame(),
        }

    events_df = pd.DataFrame(events)
    events_df["time"] = pd.to_datetime(events_df["time"], errors="coerce")
    events_df["decision_date"] = pd.to_datetime(events_df["decision_date"], errors="coerce")
    events_df = events_df.dropna(subset=["time"])
    if events_df.empty:
        empty_series = pd.Series(dtype=float)
        return {
            "total_commissions": 0.0,
            "commission_by_day": empty_series,
            "commission_by_ticker": empty_series,
            "commission_by_platform": empty_series,
            "commission_by_trade": empty_series,
            "investigation": pd.DataFrame(),
        }

    events_df["event_day"] = events_df["time"].dt.normalize()
    events_df["turnover"] = events_df["delta_qty"].abs()
    events_df["ref_price"] = pd.to_numeric(events_df["ref_price"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    events_df["days_to_decision"] = (events_df["decision_date"] - events_df["time"]).dt.days.clip(lower=0).fillna(0).astype(float)

    def _commission_for_event(event_row: pd.Series) -> float:
        ticker = str(event_row["ticker"])
        qty = float(event_row["turnover"])
        if qty <= OU_SINGULARITY_EPSILON:
            return 0.0

        # Prediction Markets: Qty is already physical contracts
        if ticker.startswith(("polymarket_", "kalshi_")):
            is_poly = ticker.startswith("polymarket_")
            trip_spread = polymarket_spread_rule(event_row["days_to_decision"]) if is_poly else kalshi_spread_rule(event_row["days_to_decision"])
            spread = trip_spread / 2.0  # Round-trip spread is double the one-way spread
            fee_func = polymarket_fed_fee if is_poly else kalshi_fed_fee
            rounded_contracts = max(1, int(round(qty)))
            fee = fee_func(rounded_contracts, float(event_row["ref_price"]))
            return float(prediction_market_one_way_cost(num_contracts=qty, spread=spread, fee=fee))

        # CME Rates: Qty is in per-unit-rate sensitivity; convert to BPV-risk units.
        risk_per_bp = qty / 10000.0

        contract_type = "SR1" if ticker.startswith("SR1:") else "ZQ"
        # NOTE: cme_round_trip_cost in this project already parameterizes size in BPV-risk units.
        # Dividing by contract BPV again materially underestimates costs.
        return float(cme_round_trip_cost(num_contracts=risk_per_bp,
                                        adv_contracts=DEFAULT_CME_ADV_CONTRACTS, 
                                        contract=contract_type) / 2.0)

    events_df["commission"] = events_df.apply(_commission_for_event, axis=1)
    events_df["platform"] = np.where(
        events_df["ticker"].astype(str).str.startswith("polymarket_"),
        "polymarket",
        np.where(
            events_df["ticker"].astype(str).str.startswith("kalshi_"),
            "kalshi",
            "cme",
        ),
    )
    commission_by_day = events_df.groupby("event_day")["commission"].sum().sort_index()
    commission_by_ticker = events_df.groupby("ticker")["commission"].sum().sort_values(ascending=False)
    commission_by_platform = events_df.groupby("platform")["commission"].sum().sort_values(ascending=False)
    commission_by_trade = events_df.groupby("trade_id")["commission"].sum().sort_index()
    investigation = (
        events_df.groupby("platform")
        .agg(
            events=("commission", "size"),
            total_turnover=("turnover", "sum"),
            mean_turnover=("turnover", "mean"),
            median_turnover=("turnover", "median"),
            total_commission=("commission", "sum"),
            avg_commission_per_event=("commission", "mean"),
        )
        .reset_index()
        .sort_values("total_commission", ascending=False)
    )
    total = float(events_df["commission"].sum())
    return {
        "total_commissions": total,
        "commission_by_day": commission_by_day,
        "commission_by_ticker": commission_by_ticker,
        "commission_by_platform": commission_by_platform,
        "commission_by_trade": commission_by_trade,
        "investigation": investigation,
    }


GLOBAL_METRICS_REPORT_COLUMNS = [
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
]


def _finalize_global_metrics_for_export(global_metrics: pd.DataFrame, transaction_costs_by_platform: pd.DataFrame) -> pd.DataFrame:
    row_source = global_metrics.iloc[0] if isinstance(global_metrics, pd.DataFrame) and not global_metrics.empty else pd.Series(dtype=float)
    platform_totals: dict[str, float] = {}
    if isinstance(transaction_costs_by_platform, pd.DataFrame) and not transaction_costs_by_platform.empty:
        if {"platform", "commission"}.issubset(transaction_costs_by_platform.columns):
            grouped = (
                transaction_costs_by_platform.assign(
                    platform=transaction_costs_by_platform["platform"].astype(str),
                    commission=pd.to_numeric(transaction_costs_by_platform["commission"], errors="coerce").fillna(0.0),
                )
                .groupby("platform")["commission"]
                .sum()
                .sort_index()
            )
            platform_totals = {f"transaction_costs_{platform}": float(value) for platform, value in grouped.items()}

    report_row: dict[str, float | int | object] = {**platform_totals}
    for col in GLOBAL_METRICS_REPORT_COLUMNS:
        value = row_source[col] if col in row_source.index else np.nan
        if col == "total_transaction_costs" and (pd.isna(value) or value is None):
            value = float(sum(platform_totals.values()))
        report_row[col] = value
    return pd.DataFrame([report_row])


def performance_metrics(daily_pnl: pd.DataFrame, trade_log: pd.DataFrame, per_asset_metrics: pd.DataFrame) -> pd.DataFrame:
    pnl = daily_pnl["daily_pnl"].astype(float) if not daily_pnl.empty else pd.Series(dtype=float)
    cost_summary = calculate_transaction_costs(trade_log)
    commissions = float(cost_summary["total_commissions"])
    gross_daily = (
        daily_pnl.assign(observed_day_pst=pd.to_datetime(daily_pnl["observed_day_pst"], errors="coerce"))
        .dropna(subset=["observed_day_pst"])
        .groupby("observed_day_pst")["daily_pnl"]
        .sum()
        .sort_index()
        if not daily_pnl.empty and "observed_day_pst" in daily_pnl.columns
        else pd.Series(dtype=float)
    )
    net_daily = gross_daily.sub(cost_summary["commission_by_day"], fill_value=0.0).sort_index()
    initial_capital = 10000
    gross_equity = (initial_capital + gross_daily.cumsum()).clip(lower=0.0)
    net_equity = (initial_capital + net_daily.cumsum()).clip(lower=0.0)
    gross_prev = _previous_equity_or_nan(gross_equity, initial_capital)
    net_prev = _previous_equity_or_nan(net_equity, initial_capital)
    gross_returns = gross_daily.div(gross_prev).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    net_returns = net_daily.div(net_prev).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe = float((gross_returns.mean() / gross_returns.std(ddof=0)) if len(gross_returns) > 1 and gross_returns.std(ddof=0) > 0 else 0.0)
    net_sharpe = float((net_returns.mean() / net_returns.std(ddof=0)) if len(net_returns) > 1 and net_returns.std(ddof=0) > 0 else 0.0)
    annualization = _infer_annualization_periods(daily_pnl["observed_day_pst"] if "observed_day_pst" in daily_pnl.columns else None)
    annualized_sharpe = float(sharpe * math.sqrt(annualization))
    net_annualized_sharpe = float(net_sharpe * math.sqrt(annualization))
    total_pnl = float(pnl.sum()) if len(pnl) else 0.0
    net_pnl = float(total_pnl - commissions)
    n_trades = int(len(trade_log))
    avg_trade_pnl = float(trade_log["trade_pnl"].mean()) if n_trades else 0.0
    gross_vol_annualized = float(gross_returns.std(ddof=0) * math.sqrt(annualization)) if len(gross_returns) > 1 else 0.0
    net_vol_annualized = float(net_returns.std(ddof=0) * math.sqrt(annualization)) if len(net_returns) > 1 else 0.0
    gross_total_return = float(gross_equity.iloc[-1] / initial_capital - 1.0) if len(gross_equity) else 0.0
    net_total_return = float(net_equity.iloc[-1] / initial_capital - 1.0) if len(net_equity) else 0.0
    gross_annualized_return = 0.0
    if len(gross_daily):
        if float(gross_equity.iloc[-1]) > 0.0:
            gross_annualized_return = float((gross_equity.iloc[-1] / initial_capital) ** (annualization / len(gross_daily)) - 1.0)
        else:
            gross_annualized_return = TOTAL_LOSS_RETURN

    net_annualized_return = 0.0
    if len(net_daily):
        if float(net_equity.iloc[-1]) > 0.0:
            net_annualized_return = float((net_equity.iloc[-1] / initial_capital) ** (annualization / len(net_daily)) - 1.0)
        else:
            net_annualized_return = TOTAL_LOSS_RETURN
    gross_win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    net_win_rate = float((net_daily > 0).mean()) if len(net_daily) else 0.0
    trade_win_rate_gross = float((trade_log["trade_pnl"].astype(float) > 0).mean()) if n_trades else 0.0
    trade_win_rate_net = float(((trade_log["trade_pnl"].astype(float) - (commissions / max(1, n_trades))) > 0).mean()) if n_trades else 0.0
    gross_dd = gross_equity.div(gross_equity.cummax()).sub(1.0)
    net_dd = net_equity.div(net_equity.cummax()).sub(1.0)
    gross_max_drawdown = float(gross_dd.min()) if len(gross_dd) else 0.0
    net_max_drawdown = float(net_dd.min()) if len(net_dd) else 0.0
    gross_yearly_dd = (
        gross_dd.groupby(gross_dd.index.year).min().min()
        if len(gross_dd)
        else 0.0
    )
    net_yearly_dd = (
        net_dd.groupby(net_dd.index.year).min().min()
        if len(net_dd)
        else 0.0
    )
    gross_yearly_dd = float(gross_yearly_dd) if pd.notna(gross_yearly_dd) else 0.0
    net_yearly_dd = float(net_yearly_dd) if pd.notna(net_yearly_dd) else 0.0
    avg_holding_period = 0.0
    if not trade_log.empty and {"entry_time", "exit_time"}.issubset(trade_log.columns):
        spans = (
            pd.to_datetime(trade_log["exit_time"], errors="coerce")
            - pd.to_datetime(trade_log["entry_time"], errors="coerce")
        ).dt.total_seconds() / 86400.0
        spans = spans.replace([np.inf, -np.inf], np.nan).dropna()
        avg_holding_period = float(spans.mean()) if len(spans) else 0.0
    return pd.DataFrame(
        [
            {
                "gross_total_return": gross_total_return,
                "net_total_return": net_total_return,
                "gross_total_profit": total_pnl,
                "net_total_profit": net_pnl,
                "gross_annualized_return": gross_annualized_return,
                "net_annualized_return": net_annualized_return,
                "gross_volatility": gross_vol_annualized,
                "net_volatility": net_vol_annualized,
                "gross_sharpe": sharpe,
                "net_sharpe": net_sharpe,
                "gross_annualized_sharpe": annualized_sharpe,
                "net_annualized_sharpe": net_annualized_sharpe,
                "total_transaction_costs": commissions,
                "gross_max_drawdown_per_day": gross_max_drawdown,
                "net_max_drawdown_per_day": net_max_drawdown,
                "gross_max_drawdown_per_year": gross_yearly_dd,
                "net_max_drawdown_per_year": net_yearly_dd,
                "gross_win_rate": gross_win_rate,
                "net_win_rate": net_win_rate,
                "gross_trade_win_rate": trade_win_rate_gross,
                "net_trade_win_rate": trade_win_rate_net,
                "number_of_trades": n_trades,
                "avg_profit_per_trade": avg_trade_pnl,
                "avg_holding_period_per_trade_days": avg_holding_period,
                # Backward-compatible aliases
                "total_pnl": total_pnl,
                "net_pnl": net_pnl,
                "sharpe": sharpe,
                "annualized_sharpe": annualized_sharpe,
                "avg_trade_pnl": avg_trade_pnl,
                "total_commissions": commissions,
            }
        ]
    )


def diagnostic_reports(panel: pd.DataFrame, residuals: pd.DataFrame, filtered_assets: list[str], trades: pd.DataFrame, trade_log: pd.DataFrame, per_asset_metrics: pd.DataFrame) -> pd.DataFrame:
    trade_pairs = trade_log[["entry_time", "exit_time"]].dropna() if not trade_log.empty else pd.DataFrame(columns=["entry_time", "exit_time"])
    hp = ((trade_pairs["exit_time"] - trade_pairs["entry_time"]).dt.days if not trade_pairs.empty else pd.Series(dtype=float))
    out = pd.DataFrame([{
        "signals_generated": int(len(trades[trades["event"].str.startswith("enter")])) if not trades.empty else 0,
        "signals_filtered": int(max(0, residuals.shape[1] - len(filtered_assets))),
        "trades_executed": int(len(trade_log)),
        "forced_exits": int((trades["event"] == "forced_exit").sum()) if not trades.empty else 0,
        "avg_holding_period": float(hp.mean()) if len(hp) else 0.0,
    }])
    print(f"Assets in panel: {panel.shape[1] - 2}")
    print(f"Residual spreads generated: {residuals.shape[1]}")
    print(f"Spreads passing filter: {len(filtered_assets)}")
    print(f"Trades executed: {int(len(trade_log))}")
    if not per_asset_metrics.empty and {"asset", "half_life", "spread_std"}.issubset(per_asset_metrics.columns):
        unavailable = int((pd.to_numeric(per_asset_metrics["half_life"], errors="coerce").isna()).sum())
        print(f"Spreads without stable OU estimates (insufficient/degenerate history): {unavailable}")
    return out


def _compute_sharpe(series: pd.Series, annualization: float = 252.0) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    std = float(s.std(ddof=0)) if len(s) else 0.0
    if len(s) <= 1 or std <= 0.0:
        return 0.0
    return float((s.mean() / std) * math.sqrt(float(annualization)))


def sharpe_breakdowns(
    daily_pnl: pd.DataFrame,
    trade_log: pd.DataFrame,
    commission_by_trade: pd.Series | None = None,
    commission_by_day: pd.Series | None = None,
    annualization: float = 252.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_meeting = pd.DataFrame(columns=["decision_date", "gross_sharpe", "net_sharpe"])
    if not trade_log.empty and {"decision_date", "trade_pnl"}.issubset(trade_log.columns):
        costs = commission_by_trade if commission_by_trade is not None else pd.Series(0.0, index=trade_log.index)
        costs = costs.reindex(trade_log.index).fillna(0.0)
        net_trade_pnl = pd.to_numeric(trade_log["trade_pnl"], errors="coerce").fillna(0.0) - costs
        rows = []
        grouped_dates = pd.to_datetime(trade_log["decision_date"], errors="coerce")
        for meeting, g in trade_log.groupby(grouped_dates):
            idx = g.index
            rows.append(
                {
                    "decision_date": meeting,
                    "gross_sharpe": _compute_sharpe(g["trade_pnl"], annualization=annualization),
                    "net_sharpe": _compute_sharpe(net_trade_pnl.loc[idx], annualization=annualization),
                }
            )
        by_meeting = pd.DataFrame(rows).sort_values("decision_date")
        for col in ["decision_date", "gross_sharpe", "net_sharpe"]:
            if col not in by_meeting.columns:
                by_meeting[col] = np.nan

    by_year = pd.DataFrame(columns=["year", "gross_sharpe", "net_sharpe"])
    if not daily_pnl.empty and {"observed_day_pst", "daily_pnl"}.issubset(daily_pnl.columns):
        work = daily_pnl.copy()
        work["observed_day_pst"] = pd.to_datetime(work["observed_day_pst"], errors="coerce")
        work = work.dropna(subset=["observed_day_pst"])
        if not work.empty:
            net_daily = (
                work.set_index("observed_day_pst")["daily_pnl"]
                .astype(float)
                .sub(commission_by_day if commission_by_day is not None else pd.Series(dtype=float), fill_value=0.0)
                .sort_index()
            )
            yr = work["observed_day_pst"].dt.year
            rows = []
            for year, g in work.groupby(yr):
                rows.append(
                    {
                        "year": int(year),
                        "gross_sharpe": _compute_sharpe(g["daily_pnl"], annualization=annualization),
                        "net_sharpe": _compute_sharpe(net_daily[net_daily.index.year == int(year)], annualization=annualization),
                    }
                )
            by_year = pd.DataFrame(rows).sort_values("year")
            for col in ["year", "gross_sharpe", "net_sharpe"]:
                if col not in by_year.columns:
                    by_year[col] = np.nan

    by_asset = pd.DataFrame(columns=["asset", "gross_sharpe", "net_sharpe"])
    if not trade_log.empty and {"asset", "trade_pnl"}.issubset(trade_log.columns):
        costs = commission_by_trade if commission_by_trade is not None else pd.Series(0.0, index=trade_log.index)
        costs = costs.reindex(trade_log.index).fillna(0.0)
        net_trade_pnl = pd.to_numeric(trade_log["trade_pnl"], errors="coerce").fillna(0.0) - costs
        rows = []
        for asset, g in trade_log.groupby("asset"):
            idx = g.index
            rows.append(
                {
                    "asset": asset,
                    "gross_sharpe": _compute_sharpe(g["trade_pnl"], annualization=annualization),
                    "net_sharpe": _compute_sharpe(net_trade_pnl.loc[idx], annualization=annualization),
                }
            )
        by_asset = pd.DataFrame(rows).sort_values("asset")
        for col in ["asset", "gross_sharpe", "net_sharpe"]:
            if col not in by_asset.columns:
                by_asset[col] = np.nan
    return by_asset, by_meeting, by_year


def summarize_ou_signal_distribution(residuals: pd.DataFrame, assets: list[str], config: StrategyConfig) -> pd.DataFrame:
    z_rows: list[pd.DataFrame] = []
    for asset in assets:
        if asset not in residuals.columns:
            continue
        s = pd.to_numeric(residuals[asset], errors="coerce")
        mu = s.rolling(max(2, config.ou_window), min_periods=max(2, config.ou_window // 2)).mean()
        sd = s.rolling(max(2, config.ou_window), min_periods=max(2, config.ou_window // 2)).std(ddof=0)
        z = (s - mu).div(sd.where(sd > OU_SINGULARITY_EPSILON))
        z_rows.append(pd.DataFrame({"asset": asset, "zscore": z}))
    if not z_rows:
        return pd.DataFrame(columns=["metric", "value"])
    z_df = pd.concat(z_rows, ignore_index=True)
    z_series = pd.to_numeric(z_df["zscore"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if z_series.empty:
        return pd.DataFrame(columns=["metric", "value"])
    pct_abs1 = float((z_series.abs() > 1.0).mean())
    pct_abs2 = float((z_series.abs() > 2.0).mean())
    out = pd.DataFrame(
        [
            {"metric": "count", "value": float(len(z_series))},
            {"metric": "mean", "value": float(z_series.mean())},
            {"metric": "std", "value": float(z_series.std(ddof=0))},
            {"metric": "skew", "value": float(z_series.skew())},
            {"metric": "kurtosis", "value": float(z_series.kurtosis())},
            {"metric": "pct_abs_gt_1", "value": pct_abs1},
            {"metric": "pct_abs_gt_1_normal", "value": NORMAL_PCT_ABS_GT_1},
            {"metric": "pct_abs_gt_2", "value": pct_abs2},
            {"metric": "pct_abs_gt_2_normal", "value": NORMAL_PCT_ABS_GT_2},
        ]
    )
    return out


def plot_cumulative_pnl(daily_pnl: pd.DataFrame, output_path: str, net_daily: pd.Series | None = None):
    _apply_report_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    if not daily_pnl.empty:
        x = pd.to_datetime(daily_pnl["observed_day_pst"], errors="coerce")
        gross = pd.to_numeric(daily_pnl["daily_pnl"], errors="coerce").fillna(0.0).cumsum()
        ax.plot(x, gross, label="Gross cumulative PnL", linewidth=1.3)
        if net_daily is not None and len(net_daily):
            net_series = pd.to_numeric(net_daily, errors="coerce").fillna(0.0).cumsum()
            net_series = net_series.reindex(pd.Index(x)).ffill().fillna(0.0)
            ax.plot(x, net_series, label="Net cumulative PnL", linewidth=1.3)
        ax.legend()
    ax.set_title("Cumulative PnL (Gross/Net)")
    ax.set_ylabel("PnL")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax.grid(alpha=0.25)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame()

def plot_factor_mimicking_returns(panel, pca, output_path, n_factors=3):
    _apply_report_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ev = np.array(getattr(pca, "explained_variance_ratio_", []), dtype=float)
    if len(ev):
        x = np.arange(1, len(ev) + 1)
        ax.bar(x, ev)
        ax.set_xticks(x)
    ax.set_title("PCA Variance Explained")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return pd.DataFrame()


def plot_spread_selection(diag: pd.DataFrame, output_path: str):
    _apply_report_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    data = diag.copy()
    sel = data["selected"].fillna(False) if "selected" in data.columns else pd.Series(False, index=data.index)
    axs[0].scatter(data.loc[~sel, "R2"], data.loc[~sel, "half_life"], s=15, alpha=0.7, label="filtered")
    axs[0].scatter(data.loc[sel, "R2"], data.loc[sel, "half_life"], s=20, alpha=0.9, label="selected")
    axs[0].set_xlabel("R2"); axs[0].set_ylabel("half_life")
    axs[1].scatter(data.loc[~sel, "spread_std"], data.loc[~sel, "half_life"], s=15, alpha=0.7)
    axs[1].scatter(data.loc[sel, "spread_std"], data.loc[sel, "half_life"], s=20, alpha=0.9)
    axs[1].set_xlabel("spread_std"); axs[1].set_ylabel("half_life")
    axs[0].legend(loc="best")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output_path); plt.close(fig)


def plot_trade_pnl_histogram(trade_log: pd.DataFrame, output_path: str):
    _apply_report_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = trade_log["trade_pnl"].astype(float) if "trade_pnl" in trade_log.columns else pd.Series(dtype=float)
    if len(vals):
        bins = min(35, max(8, int(np.sqrt(len(vals)))))
        ax.hist(vals, bins=bins, alpha=0.75, edgecolor="white")
        ax.axvline(float(vals.mean()), color="tab:red", linestyle="--", linewidth=1.2, label=f"mean={float(vals.mean()):.3f}")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Per-Trade PnL Distribution")
    ax.set_xlabel("Trade PnL")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_asset_profit_and_sharpe(per_asset_metrics: pd.DataFrame, output_path: str):
    _apply_report_plot_style()
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    data = per_asset_metrics.copy()
    if not data.empty and "asset" in data.columns:
        data = data.sort_values("asset")
        x = np.arange(len(data))
        pnl_col = "total_pnl" if "total_pnl" in data.columns else "total_profit_bps"
        sharpe_col = "annualized_sharpe" if "annualized_sharpe" in data.columns else "sharpe"
        axs[0].bar(x, pd.to_numeric(data.get(pnl_col, 0.0), errors="coerce").fillna(0.0), color="tab:blue")
        axs[0].set_ylabel("Total profit")
        axs[0].set_title("Total Profit by Asset")
        axs[1].bar(x, pd.to_numeric(data.get(sharpe_col, 0.0), errors="coerce").fillna(0.0), color="tab:green")
        axs[1].set_ylabel("Sharpe")
        axs[1].set_title("Sharpe by Asset")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(data["asset"], rotation=45, ha="right")
    for ax in axs:
        ax.grid(alpha=0.2, axis="y")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spread_with_trades(panel: pd.DataFrame, residuals: pd.DataFrame, trades: pd.DataFrame, asset: str, config: StrategyConfig, output_path: str, decision_date: str | pd.Timestamp | None = None):
    if asset not in residuals.columns: return
    _apply_report_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    series = residuals[asset].astype(float)
    roll_mu = series.rolling(max(2, config.ou_window), min_periods=max(2, config.ou_window // 2)).mean()
    roll_sd = series.rolling(max(2, config.ou_window), min_periods=max(2, config.ou_window // 2)).std(ddof=0)
    s_score = (series - roll_mu).div(roll_sd.where(roll_sd > OU_SINGULARITY_EPSILON))
    x = pd.to_datetime(panel["observed_day_pst"], errors="coerce")
    decision = pd.to_datetime(panel["decision_date"], errors="coerce")
    mask = pd.Series(True, index=panel.index)
    title_suffix = ""
    if decision_date is not None:
        target = pd.Timestamp(decision_date)
        mask = decision == target
        title_suffix = f" | meeting {target.date()}"
    x = x[mask]
    s_score = s_score[mask]
    ax.plot(x, s_score, label="OU s-score", linewidth=1.1)
    ax.axhline(config.entry_sigma, linestyle="--", color="tab:red", linewidth=1.0, label="entry bands")
    ax.axhline(-config.entry_sigma, linestyle="--", color="tab:red", linewidth=1.0)
    ax.axhline(config.exit_sigma, linestyle=":", color="tab:gray", linewidth=1.0, label="exit bands")
    ax.axhline(-config.exit_sigma, linestyle=":", color="tab:gray", linewidth=1.0)
    t = trades[trades["asset"] == asset] if not trades.empty else pd.DataFrame()
    if decision_date is not None and not t.empty:
        t = t[pd.to_datetime(t["decision_date"], errors="coerce") == pd.Timestamp(decision_date)]
    lookup = {(pd.Timestamp(panel.loc[i, "decision_date"]), pd.Timestamp(panel.loc[i, "observed_day_pst"])): float(s_score.loc[i]) for i in panel.index if i in s_score.index and pd.notna(s_score.loc[i])}
    for ev, marker, color in TRADE_EVENT_MARKERS:
        pts = t[t["event"] == ev]
        if not pts.empty:
            y = pts.apply(lambda r: lookup.get((pd.Timestamp(r["decision_date"]), pd.Timestamp(r["observed_day_pst"])), np.nan), axis=1).to_numpy(dtype=float)
            ax.scatter(pts["observed_day_pst"], y, marker=marker, c=color, s=20, label=ev)
    ax.set_title(f"{asset} OU s-score with trading bands{title_suffix}")
    ax.set_ylabel("s-score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def run_full_experiment(
    path: str,
    config: StrategyConfig,
    output_dir: str,
    generate_plots: bool = True,
    generate_report_plots: bool = True,
    report_meeting_date: str = "2026-01-28",
):
    raw_panel = load_prediction_panel(path)
    if config.start_date is not None:
        start_ts = pd.Timestamp(config.start_date)
        observed = pd.to_datetime(raw_panel["observed_day_pst"], errors="coerce")
        raw_panel = raw_panel.loc[observed >= start_ts].copy()

    panel = build_asset_panel(raw_panel, config.panel_mode)
    passthrough_cols = {"jump_sr1_portfolio_weights", "jump_ois_portfolio_weights"} | set(PREDICTION_BPS_MAP.keys())
    model_panel = panel.drop(columns=[c for c in passthrough_cols if c in panel.columns])

    pca, _, residuals, r2, basket_weights, basket_assets = run_pca(model_panel, config.n_components, False, config.pca_rolling_window_days)
    tradable, diag = filter_residuals(residuals, r2, config.min_r2, config.max_rho, config.adf_alpha, config.max_half_life_days, config.variance_threshold, config.variance_threshold_scale, True, config.ou_window)
    filtered_assets = [c for c in tradable.columns if bool(tradable[c].any())]
    trades, metrics, diags = run_ou_strategy(model_panel, residuals[filtered_assets], filtered_assets, config, tradable=tradable[filtered_assets] if filtered_assets else tradable, return_diagnostics=True)
    daily_pnl, trade_log, cum_pnl = run_basket_backtest(panel, trades, basket_weights, basket_assets, execution_lag_days=config.execution_lag_days)
    if trade_log.empty or abs(float(daily_pnl["daily_pnl"].sum())) <= OU_SINGULARITY_EPSILON:
        relaxed_min_r2 = max(0.0, min(config.min_r2, FALLBACK_MIN_R2_CAP))
        relaxed_half_life = max(config.max_half_life_days, FALLBACK_MAX_HALF_LIFE_FLOOR)
        relaxed_var_scale = min(config.variance_threshold_scale, FALLBACK_VAR_SCALE_CAP)
        relaxed_entry_sigma = max(FALLBACK_ENTRY_SIGMA_FLOOR, min(config.entry_sigma, FALLBACK_ENTRY_SIGMA_CAP))
        relaxed_exit_sigma = min(config.exit_sigma, FALLBACK_EXIT_SIGMA_CAP)
        relaxed = replace(config, min_r2=relaxed_min_r2, max_half_life_days=relaxed_half_life, variance_threshold=None, variance_threshold_scale=relaxed_var_scale, entry_sigma=relaxed_entry_sigma, exit_sigma=relaxed_exit_sigma)
        tradable, diag = filter_residuals(residuals, r2, relaxed.min_r2, relaxed.max_rho, relaxed.adf_alpha, relaxed.max_half_life_days, relaxed.variance_threshold, relaxed.variance_threshold_scale, True, relaxed.ou_window)
        filtered_assets = [c for c in tradable.columns if bool(tradable[c].any())]
        trades, metrics, diags = run_ou_strategy(model_panel, residuals[filtered_assets], filtered_assets, relaxed, tradable=tradable[filtered_assets] if filtered_assets else tradable, return_diagnostics=True)
        daily_pnl, trade_log, cum_pnl = run_basket_backtest(panel, trades, basket_weights, basket_assets, execution_lag_days=config.execution_lag_days)

    out = Path(output_dir) / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    daily_pnl.to_csv(out / "daily_pnl.csv", index=False)
    trade_log.to_csv(out / "trade_log.csv", index=False)
    cum_pnl.to_csv(out / "cumulative_pnl.csv", index=False)
    diags.to_csv(out / "diagnostics_cointegration.csv", index=False)
    diags.to_csv(out / "diagnostics_spread_model.csv", index=False) # Simplified overlap
    diag.to_csv(out / "residual_diagnostics.csv", index=False)
    per_asset_metrics = metrics.rename(columns={"entries": "number_of_entries", "exits": "number_of_exits", "avg_profit_per_trade": "avg_pnl_per_trade", "total_profit_bps": "total_pnl"})
    if not diag.empty:
        per_asset_metrics = per_asset_metrics.merge(diag[["asset", "half_life", "spread_std"]], on="asset", how="left")
    per_asset_metrics.to_csv(out / "per_asset_metrics.csv", index=False)
    strategy_diag = diagnostic_reports(panel, residuals, filtered_assets, trades, trade_log, per_asset_metrics)
    strategy_diag.to_csv(out / "strategy_metrics.csv", index=False)
    global_metrics = performance_metrics(daily_pnl, trade_log, per_asset_metrics)
    cost_summary = calculate_transaction_costs(trade_log)
    gross_daily_series = (
        daily_pnl.assign(observed_day_pst=pd.to_datetime(daily_pnl["observed_day_pst"], errors="coerce"))
        .dropna(subset=["observed_day_pst"])
        .groupby("observed_day_pst")["daily_pnl"]
        .sum()
        .sort_index()
        if not daily_pnl.empty and "observed_day_pst" in daily_pnl.columns
        else pd.Series(dtype=float)
    )
    net_daily_series = gross_daily_series.sub(cost_summary["commission_by_day"], fill_value=0.0).sort_index()
    by_asset, by_meeting, by_year = sharpe_breakdowns(
        daily_pnl,
        trade_log,
        commission_by_trade=cost_summary.get("commission_by_trade"),
        commission_by_day=cost_summary.get("commission_by_day"),
        annualization=_infer_annualization_periods(daily_pnl.get("observed_day_pst") if isinstance(daily_pnl, pd.DataFrame) else None),
    )
    if not by_asset.empty and not per_asset_metrics.empty and {"asset", "annualized_sharpe", "sharpe"}.issubset(per_asset_metrics.columns):
        by_asset = by_asset.merge(
            per_asset_metrics[["asset", "sharpe", "annualized_sharpe"]].rename(
                columns={"sharpe": "ou_sharpe", "annualized_sharpe": "ou_annualized_sharpe"}
            ),
            on="asset",
            how="left",
        )
    by_asset.to_csv(out / "sharpe_by_asset.csv", index=False)
    by_meeting.to_csv(out / "sharpe_by_meeting.csv", index=False)
    by_year.to_csv(out / "sharpe_by_year.csv", index=False)
    tc_breakdown = cost_summary["commission_by_platform"].rename("commission").reset_index().rename(columns={"platform": "platform"})
    tc_breakdown.to_csv(out / "transaction_costs_by_platform.csv", index=False)
    global_metrics = _finalize_global_metrics_for_export(global_metrics, tc_breakdown)
    cost_investigation = cost_summary.get("investigation", pd.DataFrame())
    if isinstance(cost_investigation, pd.DataFrame) and not cost_investigation.empty:
        cost_investigation.to_csv(out / "transaction_cost_investigation.csv", index=False)
    ou_signal_summary = summarize_ou_signal_distribution(residuals, filtered_assets, config)
    ou_signal_summary.to_csv(out / "ou_signal_summary.csv", index=False)
    global_metrics.to_csv(out / "global_metrics.csv", index=False)
    print(f"Gross Total Return: {float(global_metrics.iloc[0]['gross_total_return']):.6f}")
    print(f"Total Transaction Costs: {float(global_metrics.iloc[0]['total_transaction_costs']):.6f}")
    print(f"Net Total Return: {float(global_metrics.iloc[0]['net_total_return']):.6f}")
    print(f"Gross Annualized Sharpe: {float(global_metrics.iloc[0]['gross_annualized_sharpe']):.6f}")
    print(f"Net Annualized Sharpe: {float(global_metrics.iloc[0]['net_annualized_sharpe']):.6f}")

    if generate_plots:
        plot_cumulative_pnl(daily_pnl, str(Path(output_dir) / "cumulative_pnl.png"), net_daily=net_daily_series)
        plot_spread_selection(diag, str(Path(output_dir) / "spread_selection.png"))
        plot_trade_pnl_histogram(trade_log, str(Path(output_dir) / "trade_pnl_histogram.png"))
        plot_factor_mimicking_returns(panel, pca, str(Path(output_dir) / "pca_variance.png"))
        plot_asset_profit_and_sharpe(per_asset_metrics, str(Path(output_dir) / "profit_sharpe_by_asset.png"))
        for asset in sorted(filtered_assets):
            plot_spread_with_trades(panel, residuals, trades, asset, config, str(Path(output_dir) / f"spread_trades_{asset}.png"))
            if generate_report_plots:
                plot_spread_with_trades(
                    panel,
                    residuals,
                    trades,
                    asset,
                    config,
                    str(Path(output_dir) / f"spread_trades_{asset}_{pd.Timestamp(report_meeting_date).date()}.png"),
                    decision_date=report_meeting_date,
                )
        plot_cumulative_pnl(daily_pnl, str(Path(output_dir) / "cumulative_pnl_by_strategy.png"), net_daily=net_daily_series)
        plot_factor_mimicking_returns(panel, pca, str(Path(output_dir) / "factor_mimicking_returns.png"))

    r2_frame = r2 if isinstance(r2, pd.DataFrame) else r2.to_frame("r2")
    return {
        "panel": panel, "residuals": residuals, "r2": r2_frame, "trades": trades, 
        "metrics": metrics, "trade_log": trade_log, "daily_pnl": daily_pnl, 
        "diagnostics_cointegration": diags, "cumulative_pnl": cum_pnl, 
        "selection_diagnostics": diag,
        "diagnostics_spread_model": diags,
        "per_asset_metrics": per_asset_metrics,
        "strategy_metrics": strategy_diag,
        "global_metrics": global_metrics,
        "sharpe_by_asset": by_asset,
        "sharpe_by_meeting": by_meeting,
        "sharpe_by_year": by_year,
        "transaction_costs_by_platform": tc_breakdown,
        "ou_signal_summary": ou_signal_summary,
    }

if __name__ == "__main__":
    run_full_experiment("Data/Merged/Prediction_all_with_sofr.csv", StrategyConfig(execution_lag_days=0, panel_mode="prediction_expected", n_components=1, max_half_life_days=5, pca_rolling_window_days=90, ou_window=30, max_holding_days=100), "Data/Outputs/mr_cointegration")
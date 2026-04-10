from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from fredapi import Fred
from statsmodels.tsa.stattools import adfuller, coint
import logging

key = "ea3673cec79ae8f1be6187d0a3a08d95"

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MeetingConfig:
    meeting_date: pd.Timestamp
    csv_path: Path


MEETINGS: tuple[MeetingConfig, ...] = (
    MeetingConfig(pd.Timestamp("2025-10-29"), BASE_DIR / "polymarket-price-data-29-10-2025-(19-06-2025-30-10-2025).csv"),
    MeetingConfig(pd.Timestamp("2025-12-10"), BASE_DIR / "polymarket-price-data-10-12-2025-(31-07-2025-09-12-2025).csv"),
    MeetingConfig(pd.Timestamp("2026-01-28"), BASE_DIR / "polymarket-price-data-28-01-2026-(17-09-2025-23-01-2026).csv"),
    MeetingConfig(pd.Timestamp("2026-03-18"), BASE_DIR / "polymarket-price-data-18-03-2026-(29-10-2025-23-01-2026).csv"),
    MeetingConfig(pd.Timestamp("2026-04-29"), BASE_DIR / "polymarket-price-data-29-04-2026-(13-11-2025-23-01-2026).csv"),
)


FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "sofr": "SOFR",
    "t_bill_1m": "DGS1MO",
    "t_bill_3m": "DTB3",
    "t_bill_6m": "DTB6",
    "t_bill_1y": "DTB1YR",
    "t_note_2y": "DGS2",
    "t_note_3y": "DGS3",
    "t_note_5y": "DGS5",
}

MATURITY_MAP = {
    "overnight": 1 / 365,
    "t_bill_1m": 1 / 12,
    "t_bill_3m": 0.25,
    "t_bill_6m": 0.5,
    "t_bill_1y": 1.0,
    "t_note_2y": 2.0,
    "t_note_3y": 3.0,
    "t_note_5y": 5.0,
}

ZERO_CURVE_ORDER = list(MATURITY_MAP.keys())


OUTCOME_BPS_MAP = {
    "50+ bps decrease": -0.005,
    "25 bps decrease": -0.0025,
    "No change": 0.0,
    "25+ bps increase": 0.0025,
}


def fred_client(api_key: Optional[str] = None) -> Fred:
    key = "ea3673cec79ae8f1be6187d0a3a08d95"
    #key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY must be set in environment or provided.")
    return Fred(api_key=key)

def load_polymarket_expectations(meetings: Iterable[MeetingConfig] = MEETINGS) -> pd.DataFrame:
    frames = []
    for meeting in meetings:
        df = pd.read_csv(meeting.csv_path)
        df["date"] = pd.to_datetime(df["Date (UTC)"], format="%m-%d-%Y %H:%M")
        df = df.sort_values("date").set_index("date")
        full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(full_range).ffill()
        available = [col for col in OUTCOME_BPS_MAP if col in df.columns]
        if not available:
            raise ValueError(f"No matching outcome columns in {meeting.csv_path.name}")
        probabilities = df[available].astype(float)
        total_prob = probabilities.sum(axis=1)
        if (total_prob <= 0).any() or total_prob.isna().any():
            raise ValueError(f"Invalid probability totals in {meeting.csv_path.name}")
        probabilities = probabilities.div(total_prob.replace(0, np.nan), axis=0).fillna(0.0)
        outcome_values = pd.Series({col: OUTCOME_BPS_MAP[col] for col in available})
        expected_change = probabilities.mul(outcome_values).sum(axis=1)
        centered = outcome_values.to_numpy() - expected_change.to_numpy()[:, None]
        variance = (probabilities.to_numpy() * centered**2).sum(axis=1)
        variance = pd.Series(variance, index=probabilities.index)
        variance_safe = variance.replace(0, np.nan)
        skew = (probabilities.to_numpy() * centered**3).sum(axis=1) / variance_safe.pow(1.5)
        skew = pd.Series(skew, index=probabilities.index).fillna(0.0)
        dominant = probabilities.idxmax(axis=1)
        days_to_meeting = (meeting.meeting_date - probabilities.index).days
        frames.append(
            pd.DataFrame(
                {
                    "date": probabilities.index,
                    "meeting_date": meeting.meeting_date,
                    "expected_change": expected_change,
                    "prob_variance": variance,
                    "prob_skew": skew,
                    "dominant_outcome": dominant,
                    "days_to_meeting": days_to_meeting,
                    **{f"prob_{col}": probabilities[col] for col in probabilities.columns},
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return result.set_index(["date", "meeting_date"]).sort_index()


def identify_regimes(poly: pd.DataFrame, variance_threshold: float, late_cutoff: int = 10) -> pd.Series:
    def classify(row: pd.Series) -> str:
        if row["days_to_meeting"] <= late_cutoff:
            return "late"
        if row["prob_variance"] > variance_threshold:
            return "high_dispersion"
        if row["dominant_outcome"] == "25+ bps increase":
            return "hike"
        if row["dominant_outcome"] in {"25 bps decrease", "50+ bps decrease"}:
            return "cut"
        return "neutral"

    return poly.apply(classify, axis=1)


def fetch_curve_points(start_date: str, fred_api_key: Optional[str] = None) -> pd.DataFrame:
    fred = fred_client(fred_api_key)
    frames = {}
    for label, series in FRED_SERIES.items():
        series_data = fred.get_series(series, observation_start=start_date)
        frames[label] = series_data
    curve = pd.DataFrame(frames)
    curve.index = pd.to_datetime(curve.index)
    return curve.sort_index()

def build_discount_factors(zero_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert zero rates into discount factors.
    zero_df: DataFrame with maturities as columns and zero rates in decimal.
    """
    dfs = zero_df.copy()
    for col in zero_df.columns:
        t = MATURITY_MAP[col]
        dfs[col] = np.exp(-zero_df[col] * t)  # continuous discount factors
    return dfs

def interp_discount_factor(row: pd.Series, target_t: float) -> float:
    """
    Interpolate discount factors on a continuous maturity grid.
    row: discount factors for named maturities
    target_t: target time in years
    """
    grid_t = np.array([MATURITY_MAP[c] for c in row.index])
    grid_df = row.values
    return np.interp(target_t, grid_t, grid_df)

def forward_rate_from_dfs(df_prev: float, df_next: float, dt: float) -> float:
    """
    Compute forward rate from discount factors over dt.
    """
    return (df_prev / df_next - 1) / dt

def meeting_implied_forward(
    dfs: pd.DataFrame,
    meeting_date: pd.Timestamp,
    next_meeting_date: pd.Timestamp
) -> pd.Series:
    """
    Return a Series of meeting-to-next meeting forward SOFR expectations.
    dfs: discount factor DataFrame indexed by date
    meeting_date: current meeting
    next_meeting_date: subsequent meeting
    """
    results = []

    for date, row in dfs.iterrows():
        if date >= meeting_date:
            results.append(np.nan)
            continue

        t_start = max((meeting_date - date).days / 365.0, 1 / 365)
        t_end = max((next_meeting_date - date).days / 365.0, t_start + 1/365)

        df_start = interp_discount_factor(row, t_start)
        df_end = interp_discount_factor(row, t_end)

        # forward rate over [t_start, t_end]
        fwd = forward_rate_from_dfs(df_start, df_end, t_end - t_start)
        results.append(fwd)

    return pd.Series(results, index=dfs.index)

def year_frac(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """ACT/360 year fraction."""
    return (end - start).days / 360.0

def bootstrap_from_sofr_futures(
    futures_quotes: list[dict],
    valuation_date: pd.Timestamp
) -> dict[pd.Timestamp, float]:
    """
    futures_quotes:
        [
          {
            "start": pd.Timestamp,
            "end": pd.Timestamp,
            "price": float  # e.g. 94.87
          }
        ]
    Returns: {date: discount_factor}
    """
    dfs = {valuation_date: 1.0}

    for q in sorted(futures_quotes, key=lambda x: x["end"]):
        start = q["start"]
        end = q["end"]
        price = q["price"]

        rate = (100.0 - price) / 100.0
        dt = year_frac(start, end)

        df_start = dfs[start]
        df_end = df_start / (1.0 + rate * dt)
        dfs[end] = df_end

    return dfs

def bootstrap_from_ois_swaps(
    dfs: dict[pd.Timestamp, float],
    ois_quotes: list[dict],
    valuation_date: pd.Timestamp
) -> dict[pd.Timestamp, float]:
    """
    ois_quotes:
        [
          {
            "maturity": pd.Timestamp,
            "fixed_rate": float  # decimal, e.g. 0.0535
            "payment_dates": list[pd.Timestamp]
          }
        ]
    """
    for q in sorted(ois_quotes, key=lambda x: x["maturity"]):
        maturity = q["maturity"]
        fixed = q["fixed_rate"]
        pay_dates = q["payment_dates"]

        accruals = []
        known_leg = 0.0

        prev = valuation_date
        for d in pay_dates:
            dt = year_frac(prev, d)
            if d in dfs:
                known_leg += dfs[d] * dt
            else:
                accruals.append((d, dt))
            prev = d

        # Only last DF should be unknown
        if len(accruals) != 1:
            raise ValueError("Bootstrap expects exactly one unknown DF")

        last_date, last_dt = accruals[0]

        df_T = (1.0 - fixed * known_leg) / (1.0 + fixed * last_dt)
        dfs[last_date] = df_T

    return dfs

def discount_factors_to_zero_curve(
    dfs: dict[pd.Timestamp, float],
    valuation_date: pd.Timestamp
) -> pd.DataFrame:
    rows = []
    for d, df in dfs.items():
        if d <= valuation_date:
            continue
        t = year_frac(valuation_date, d)
        zero = -np.log(df) / t
        rows.append({"date": d, "t": t, "zero_rate": zero})

    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)



def threshold_ecm(spread: pd.Series, alpha: float = 0.5, level: Optional[float] = None) -> pd.Series:
    """Threshold error-correction adjustment when spreads exceed the alpha quantile."""
    spread = spread.dropna()
    if spread.empty:
        return spread
    lagged = spread.shift(1)
    level_value = level if level is not None else lagged.abs().quantile(alpha)
    adjustment = np.where(lagged.abs() >= level_value, -lagged, 0.0)
    return pd.Series(adjustment, index=spread.index)


def build_rates_implied_expectations(
    zero_curve: pd.DataFrame, meetings: Iterable[MeetingConfig] = MEETINGS
) -> pd.DataFrame:
    frames = []
    meetings_list = sorted(meetings, key=lambda m: m.meeting_date)
    meeting_dates = [m.meeting_date for m in meetings_list]
    for idx, meeting in enumerate(meetings_list):
        next_date = meeting_dates[idx + 1] if idx + 1 < len(meeting_dates) else None
        meeting_series = meeting_forward_implied_change(zero_curve, meeting.meeting_date, next_date)
        frames.append(
            pd.DataFrame(
                {
                    "date": zero_curve.index,
                    "meeting_date": meeting.meeting_date,
                    "expected_change": meeting_series,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return result.set_index(["date", "meeting_date"]).sort_index()


def align_expectations(poly: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    merged = poly.join(rates, how="inner", lsuffix="_poly", rsuffix="_rates")
    merged = merged.dropna(subset=["expected_change_poly", "expected_change_rates"])
    return merged


def run_cointegration_tests(aligned: pd.DataFrame, train_window: int = 60) -> pd.DataFrame:
    results = []
    for meeting_date, data in aligned.groupby(level="meeting_date"):
        series_a = data["expected_change_poly"].droplevel("meeting_date")
        series_b = data["expected_change_rates"].droplevel("meeting_date")
        if len(series_a) < max(train_window, 20):
            continue
        score, pvalue, _ = coint(series_a.iloc[:train_window], series_b.iloc[:train_window])
        x_train = series_b.iloc[:train_window].to_numpy()
        y_train = series_a.iloc[:train_window].to_numpy()
        beta = np.linalg.lstsq(x_train[:, None], y_train, rcond=None)[0][0]
        residuals = series_a.iloc[train_window:] - beta * series_b.iloc[train_window:]
        oos_pvalue = np.nan
        if len(residuals.dropna()) >= 20:
            try:
                oos_pvalue = adfuller(residuals.dropna())[1]
            except ValueError:
                oos_pvalue = np.nan
        results.append(
            {
                "meeting_date": meeting_date,
                "score": score,
                "beta": beta,
                "pvalue": pvalue,
                "oos_pvalue": oos_pvalue,
            }
        )
    return pd.DataFrame(results).set_index("meeting_date")


def lead_lag_regression(aligned: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    results = []
    for meeting_date, data in aligned.groupby(level="meeting_date"):
        df = data.copy().droplevel("meeting_date").sort_index()
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                shifted = df["expected_change_rates"].rename("expected_change_rates")
            else:
                shifted = df["expected_change_rates"].shift(lag).rename("expected_change_rates")
            valid = df[["expected_change_poly"]].join(shifted, how="inner").dropna()
            if len(valid) < 20:
                continue
            x = valid["expected_change_rates"].to_numpy()
            y = valid["expected_change_poly"].to_numpy()
            beta = np.linalg.lstsq(x[:, None], y, rcond=None)[0][0]
            resid = y - beta * x
            results.append(
                {
                    "meeting_date": meeting_date,
                    "lag": lag,
                    "beta": beta,
                    "resid_std": np.std(resid),
                }
            )
    return pd.DataFrame(results).set_index(["meeting_date", "lag"])


def estimate_spread(
    aligned: pd.DataFrame,
    train_window: int = 60,
    hike_percentile: float = 0.6,
    other_percentile: float = 0.4,
) -> pd.DataFrame:
    outputs = []
    for meeting_date, data in aligned.groupby(level="meeting_date"):
        df = data.copy().droplevel("meeting_date")
        df = df.sort_index()
        if len(df) < train_window:
            continue
        x = df["expected_change_rates"].to_numpy()
        y = df["expected_change_poly"].to_numpy()
        beta = np.linalg.lstsq(x[:train_window, None], y[:train_window], rcond=None)[0][0]
        spread = y - beta * x
        variance = df["prob_variance"].to_numpy()
        variance_threshold = df["prob_variance"].iloc[:train_window].quantile(0.75)
        regime = identify_regimes(df, variance_threshold)
        hike_threshold = np.nanpercentile(spread[:train_window], hike_percentile * 100)
        other_threshold = np.nanpercentile(spread[:train_window], other_percentile * 100)
        threshold = np.where(regime == "hike", hike_threshold, other_threshold)
        adjusted_spread = spread - threshold
        in_sample = np.arange(len(df)) < train_window
        ecm_level = pd.Series(spread, index=df.index).shift(1).iloc[:train_window].abs().quantile(0.5)
        ecm_adjustment = threshold_ecm(pd.Series(spread, index=df.index), level=ecm_level)
        outputs.append(
            pd.DataFrame(
                {
                    "date": df.index,
                    "meeting_date": meeting_date,
                    "spread": spread,
                    "adjusted_spread": adjusted_spread,
                    "beta": beta,
                    "prob_variance": variance,
                    "regime": regime.values,
                    "in_sample": in_sample,
                    "ecm_adjustment": ecm_adjustment.values,
                }
            )
        )
    result = pd.concat(outputs, ignore_index=True)
    return result.set_index(["date", "meeting_date"]).sort_index()


def backtest_mean_reversion(
    spread: pd.DataFrame,
    entry_z: float = 1.25,
    exit_z: float = 0.25,
    variance_adjustment: bool = False,
) -> pd.DataFrame:
    results = []
    for meeting_date, data in spread.groupby(level="meeting_date"):
        df = data.copy().droplevel("meeting_date").sort_index()
        base_series = df["spread"]
        rolling_mean = base_series.rolling(60, min_periods=20).mean()
        rolling_std = base_series.rolling(60, min_periods=20).std()
        z_score = (base_series - rolling_mean) / rolling_std
        variance_threshold = df.loc[df["in_sample"], "prob_variance"].median()
        threshold = np.full(len(df), entry_z)
        if variance_adjustment:
            threshold = np.where(df["prob_variance"] > variance_threshold, entry_z * 1.2, entry_z)
        position = np.zeros(len(df))
        for i in range(1, len(df)):
            if df["in_sample"].iloc[i]:
                position[i] = position[i - 1]
                continue
            if np.isnan(z_score.iloc[i]):
                position[i] = position[i - 1]
                continue
            if position[i - 1] == 0:
                if z_score.iloc[i] > threshold[i]:
                    position[i] = -1
                elif z_score.iloc[i] < -threshold[i]:
                    position[i] = 1
            else:
                if abs(z_score.iloc[i]) < exit_z:
                    position[i] = 0
                else:
                    position[i] = position[i - 1]
        # Close any open positions at the end of the sample
        if position[-1] != 0:
            position[-1] = 0

        position_series = pd.Series(position, index=df.index)
        pnl = position_series.shift(1).fillna(0.0) * base_series.diff().fillna(0.0)
        results.append(
            pd.DataFrame(
                {
                    "date": df.index,
                    "meeting_date": meeting_date,
                    "spread": base_series.values,
                    "z_score": z_score.values,
                    "position": position,
                    "pnl": pnl,
                }
            )
        )
    result = pd.concat(results, ignore_index=True)
    return result.set_index(["date", "meeting_date"]).sort_index()

def plot_cointegration_test(aligned: pd.DataFrame, results: pd.DataFrame):
    for meeting_date, data in aligned.groupby(level="meeting_date"):
        series_a = data["expected_change_poly"].droplevel("meeting_date")
        series_b = data["expected_change_rates"].droplevel("meeting_date")
        residuals = series_a - results.loc[meeting_date, "beta"] * series_b

        plt.figure(figsize=(12, 6))
        plt.plot(series_a, label="Expected Change Poly", alpha=0.7)
        plt.plot(series_b, label="Expected Change Rates", alpha=0.7)
        plt.plot(residuals, label="Residuals", linestyle="--", alpha=0.7)
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.title(f"Cointegration Test for Meeting Date: {meeting_date}")
        plt.legend()
        plt.show()


def backtest_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for meeting_date, data in backtest.groupby(level="meeting_date"):
        trades = data["position"].diff().fillna(0).abs().sum() / 2
        closed_trades = data[data["position"].diff() != 0]
        profitable_trades = closed_trades[closed_trades["pnl"] > 0]
        open_trades = data["position"].iloc[-1] != 0
        pnl_profitable = profitable_trades["pnl"].sum()
        total_pnl = data["pnl"].sum()

        summary.append(
            {
                "meeting_date": meeting_date,
                "total_trades": trades,
                "profitable_trades": len(profitable_trades),
                "pnl_profitable_trades": pnl_profitable,
                "open_trades": int(open_trades),
                "total_pnl": total_pnl,
            }
        )
    return pd.DataFrame(summary).set_index("meeting_date")


def plot_backtest_results(backtest: pd.DataFrame):
    for meeting_date, data in backtest.groupby(level="meeting_date"):
        df = data.copy().droplevel("meeting_date")
        plt.figure(figsize=(12, 6))

        # Plot spread and z-score
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df["spread"], label="Spread", alpha=0.7)
        plt.plot(df.index, df["z_score"], label="Z-Score", alpha=0.7)
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.title(f"Spread and Z-Score for Meeting Date: {meeting_date}")
        plt.legend()

        # Plot positions and PnL
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df["position"], label="Position", alpha=0.7)
        plt.plot(df.index, df["pnl"].cumsum(), label="Cumulative PnL", alpha=0.7)
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.title(f"Positions and Cumulative PnL for Meeting Date: {meeting_date}")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    fred_key = os.getenv("FRED_API_KEY")
    curve = fetch_curve_points("2024-01-01", fred_key)
    df = build_discount_factors(curve)
    zero_curve = meeting_implied_forward(curve_to_zero_curve(curve))
    poly = load_polymarket_expectations()
    rates = build_rates_implied_expectations(zero_curve)
    aligned = align_expectations(poly, rates)

    # Run cointegration tests and plot results
    cointegration = run_cointegration_tests(aligned)
    plot_cointegration_test(aligned, cointegration)

    # Estimate spread and backtest
    spread = estimate_spread(aligned)
    backtest = backtest_mean_reversion(spread, variance_adjustment=True)

    # Generate summary statistics for backtest
    summary = backtest_summary(backtest)
    print(summary)

    # Plot backtest results
    plot_backtest_results(backtest)

    print(cointegration.head())
    print(backtest.groupby("meeting_date")["pnl"].sum())
    print(spread.head())
    print(rates)


## 12/3

def plot_factor_mimicking_returns(
    panel: pd.DataFrame,
    pca: PCA,
    output_path: str,
    n_factors: int = 3,
) -> pd.DataFrame:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    asset_cols = [c for c in panel.columns if c not in {"decision_date", "observed_day_pst"}]
    if not asset_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No factor returns available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return pd.DataFrame()
    x = panel[asset_cols].apply(pd.to_numeric, errors="coerce")
    x = x.dropna(axis=1, how="all")
    if x.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No factor returns available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return pd.DataFrame()
    x = x.fillna(x.mean())
    factor_count = min(n_factors, pca.components_.shape[0])
    weights = pca.components_[:factor_count, :]
    factor_values = x.to_numpy() @ weights.T
    factor_df = pd.DataFrame(
        factor_values,
        index=panel.index,
        columns=[f"F{i + 1}" for i in range(weights.shape[0])],
    )
    factor_df["decision_date"] = panel["decision_date"].values
    decision_returns = factor_df.groupby("decision_date").agg(lambda s: s.iloc[-1] - s.iloc[0]).sort_index()
    decision_returns.index = pd.to_datetime(decision_returns.index)

    fig, ax = plt.subplots(figsize=(10, 5))
    if decision_returns.empty:
        ax.text(0.5, 0.5, "No factor returns available", ha="center", va="center")
    else:
        decision_returns.plot(ax=ax, marker="o", linewidth=1.4)
    ax.set_title("Factor Mimicking Portfolio Returns by Decision Date")
    ax.set_ylabel("Return (bps)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return decision_returns.reset_index()


    def plot_cumulative_pnl(strategy_trades: dict[str, pd.DataFrame], output_path: str) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for strategy_type, trades in strategy_trades.items():
        if trades.empty:
            series_map[strategy_type] = pd.Series(dtype=float)
            continue
        exits = trades[trades["event"].astype(str).str.contains("exit")].copy()
        if exits.empty:
            series_map[strategy_type] = pd.Series(dtype=float)
            continue
        daily = exits.groupby("observed_day_pst")["pnl_bps"].sum().sort_index().cumsum()
        series_map[strategy_type] = daily

    curve_df = pd.DataFrame(series_map).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    if curve_df.empty:
        ax.text(0.5, 0.5, "No exit trades available", ha="center", va="center")
    else:
        curve_df.plot(ax=ax, linewidth=1.8)
    ax.set_title("Cumulative PnL by Strategy Class")
    ax.set_ylabel("Cumulative bps")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return curve_df.reset_index().rename(columns={"index": "observed_day_pst"})
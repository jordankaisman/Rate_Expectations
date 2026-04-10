from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_monthly_pnl(
    trades: pd.DataFrame,
    pnl_col: str,
    date_col: str = "exit_date",
    label: str | None = None,
) -> pd.DataFrame:
    if trades.empty or pnl_col not in trades.columns:
        return pd.DataFrame(columns=["month", "pnl", "cum_pnl", "drawdown", "label"])

    frame = trades.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[pnl_col] = pd.to_numeric(frame[pnl_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, pnl_col]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["month", "pnl", "cum_pnl", "drawdown", "label"])

    monthly = (
        frame.groupby(frame[date_col].dt.to_period("M"))[pnl_col]
        .sum()
        .rename("pnl")
        .reset_index()
    )
    monthly["month"] = monthly[date_col].dt.to_timestamp("M")
    monthly = monthly.drop(columns=[date_col])
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["cum_pnl"] = monthly["pnl"].cumsum()
    monthly["cum_peak"] = monthly["cum_pnl"].cummax()
    monthly["drawdown"] = monthly["cum_pnl"] - monthly["cum_peak"]
    monthly["label"] = label if label is not None else pnl_col
    return monthly


def _annualized_sharpe(series: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    sigma = clean.std(ddof=1)
    if sigma <= 0:
        return np.nan
    return (clean.mean() / sigma) * math.sqrt(periods_per_year)


def _annualized_sortino(series: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    downside = clean[clean < 0]
    if len(downside) == 0:
        return np.nan
    downside_sigma = downside.std(ddof=1)
    if pd.isna(downside_sigma) or downside_sigma <= 0:
        return np.nan
    return (clean.mean() / downside_sigma) * math.sqrt(periods_per_year)


def compute_research_stats(
    trades: pd.DataFrame,
    pnl_col: str,
    date_col: str = "exit_date",
    periods_per_year: int = 12,
) -> dict[str, float]:
    if trades.empty or pnl_col not in trades.columns:
        return {
            "n_trades": 0.0,
            "total_pnl": 0.0,
            "mean_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "std_trade_pnl": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_holding_days": np.nan,
            "monthly_mean_pnl": np.nan,
            "monthly_vol_pnl": np.nan,
            "monthly_sharpe_ann": np.nan,
            "monthly_sortino_ann": np.nan,
            "max_monthly_drawdown": np.nan,
            "best_month_pnl": np.nan,
            "worst_month_pnl": np.nan,
            "n_months": 0.0,
        }

    frame = trades.copy()
    frame[pnl_col] = pd.to_numeric(frame[pnl_col], errors="coerce")
    frame = frame.dropna(subset=[pnl_col]).copy()
    if frame.empty:
        return {
            "n_trades": 0.0,
            "total_pnl": 0.0,
            "mean_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "std_trade_pnl": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_holding_days": np.nan,
            "monthly_mean_pnl": np.nan,
            "monthly_vol_pnl": np.nan,
            "monthly_sharpe_ann": np.nan,
            "monthly_sortino_ann": np.nan,
            "max_monthly_drawdown": np.nan,
            "best_month_pnl": np.nan,
            "worst_month_pnl": np.nan,
            "n_months": 0.0,
        }

    monthly = build_monthly_pnl(frame, pnl_col=pnl_col, date_col=date_col)
    wins = frame[pnl_col] > 0
    gross_profit = frame.loc[frame[pnl_col] > 0, pnl_col].sum()
    gross_loss = frame.loc[frame[pnl_col] < 0, pnl_col].sum()
    profit_factor = np.nan if gross_loss == 0 else float(gross_profit / abs(gross_loss))

    stats = {
        "n_trades": float(len(frame)),
        "total_pnl": float(frame[pnl_col].sum()),
        "mean_trade_pnl": float(frame[pnl_col].mean()),
        "median_trade_pnl": float(frame[pnl_col].median()),
        "std_trade_pnl": float(frame[pnl_col].std(ddof=1)) if len(frame) > 1 else np.nan,
        "win_rate": float(wins.mean()) if len(frame) > 0 else np.nan,
        "profit_factor": profit_factor,
        "avg_holding_days": float(pd.to_numeric(frame.get("days_held"), errors="coerce").mean())
        if "days_held" in frame.columns
        else np.nan,
        "monthly_mean_pnl": float(monthly["pnl"].mean()) if not monthly.empty else np.nan,
        "monthly_vol_pnl": float(monthly["pnl"].std(ddof=1)) if len(monthly) > 1 else np.nan,
        "monthly_sharpe_ann": float(_annualized_sharpe(monthly["pnl"], periods_per_year=periods_per_year))
        if not monthly.empty
        else np.nan,
        "monthly_sortino_ann": float(_annualized_sortino(monthly["pnl"], periods_per_year=periods_per_year))
        if not monthly.empty
        else np.nan,
        "max_monthly_drawdown": float(monthly["drawdown"].min()) if not monthly.empty else np.nan,
        "best_month_pnl": float(monthly["pnl"].max()) if not monthly.empty else np.nan,
        "worst_month_pnl": float(monthly["pnl"].min()) if not monthly.empty else np.nan,
        "n_months": float(len(monthly)),
    }
    return stats


def summarize_strategies(reports: dict[str, dict[str, float]]) -> pd.DataFrame:
    if not reports:
        return pd.DataFrame()
    out = pd.DataFrame(reports).T
    out.index.name = "strategy"
    return out


def plot_cumulative_and_drawdown(
    monthly: pd.DataFrame,
    title_prefix: str,
    y_label: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    if monthly.empty:
        axes[0].set_title(f"{title_prefix} - Cumulative PnL")
        axes[1].set_title(f"{title_prefix} - Drawdown")
        return fig, axes

    axes[0].plot(monthly["month"], monthly["cum_pnl"], color="navy", linewidth=2)
    axes[0].set_title(f"{title_prefix} - Cumulative PnL")
    axes[0].set_ylabel(y_label)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(monthly["month"], monthly["drawdown"], 0.0, color="firebrick", alpha=0.25)
    axes[1].plot(monthly["month"], monthly["drawdown"], color="firebrick", linewidth=1.5)
    axes[1].set_title(f"{title_prefix} - Drawdown")
    axes[1].set_ylabel(y_label)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_monthly_bars(monthly: pd.DataFrame, title: str, y_label: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    if monthly.empty:
        ax.set_title(title)
        return fig, ax

    colors = np.where(monthly["pnl"] >= 0, "seagreen", "indianred")
    ax.bar(monthly["month"], monthly["pnl"], color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_trade_pnl_histogram(trades: pd.DataFrame, pnl_col: str, title: str, x_label: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    if trades.empty or pnl_col not in trades.columns:
        ax.set_title(title)
        return fig, ax

    data = pd.to_numeric(trades[pnl_col], errors="coerce").dropna()
    if data.empty:
        ax.set_title(title)
        return fig, ax

    ax.hist(data, bins=20, color="steelblue", alpha=0.85, edgecolor="white")
    ax.axvline(data.mean(), color="orange", linestyle="--", linewidth=2, label=f"mean={data.mean():.2f}")
    ax.axvline(data.median(), color="purple", linestyle="-.", linewidth=2, label=f"median={data.median():.2f}")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax

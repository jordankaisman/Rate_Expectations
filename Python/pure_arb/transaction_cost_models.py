from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pure_arb_pipeline import EFFR_FUTURES_PATH, KALSHI_PATH, parse_contrdate


CME_CONTRACT_NOTIONALS = {
    "SR3": 25.00,
    "SR1": 41.67,
    "ZQ": 41.67,
}

CME_BPV_USD_PER_BP = 41.67
DEFAULT_CONTRACT = "ZQ"


def polymarket_fed_fee(num_contracts: int, price: float) -> float:
    return 0.0


def kalshi_fed_fee(num_contracts: int, price: float) -> float:
    raw = 0.07 * num_contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def polymarket_spread_rule(days_to_decision: float) -> float:
    if days_to_decision < 0:
        raise ValueError("days_to_decision must be non-negative")
    if days_to_decision < 45:
        return 0.01
    if days_to_decision < 85:
        return 0.03
    if days_to_decision < 95:
        return 0.04
    return 0.05

def kalshi_spread_rule(days_to_decision: float) -> float:
    if days_to_decision < 0:
        raise ValueError("days_to_decision must be non-negative")
    if days_to_decision < 45:
        return 0.01
    if days_to_decision < 85:
        return 0.03
    if days_to_decision < 95:
        return 0.04
    return 0.05

def prediction_market_one_way_cost(
    num_contracts: float,
    spread: float,
    fee: float,
    transfer: float = 0.0,
) -> float:
    return num_contracts * spread + fee + transfer


def cme_participation_cost(
    num_contracts: float,
    adv_contracts: float,
    contract: str = DEFAULT_CONTRACT,
    contract_notional: float | None = None,
) -> dict[str, float | str]:
    key = contract.upper()
    if contract_notional is None:
        if key not in CME_CONTRACT_NOTIONALS:
            raise ValueError(
                f"Unknown contract '{contract}'. "
                f"Use one of {list(CME_CONTRACT_NOTIONALS)} or pass contract_notional."
            )
        contract_notional = CME_CONTRACT_NOTIONALS[key]

    safe_adv = max(float(adv_contracts), 1.0)
    dollar_traded = float(num_contracts) * float(contract_notional)
    adv_dollar = safe_adv * float(contract_notional)
    participation_rate = max(dollar_traded / adv_dollar, 0.0)
    c_bps = 2.0 + 15.0 * math.sqrt(participation_rate)
    tc_round_trip = 2.0 * (c_bps / 10000.0) * dollar_traded

    return {
        "contract": key,
        "num_contracts": float(num_contracts),
        "contract_notional": float(contract_notional),
        "dollar_traded": dollar_traded,
        "adv_contracts": safe_adv,
        "adv_dollar": adv_dollar,
        "participation_rate": participation_rate,
        "c_bps": c_bps,
        "tc_round_trip": tc_round_trip,
    }


def cme_round_trip_cost(
    num_contracts: float,
    adv_contracts: float,
    contract: str = DEFAULT_CONTRACT,
) -> float:
    details = cme_participation_cost(num_contracts, adv_contracts, contract=contract)
    return float(details["tc_round_trip"])


def build_adv_lookup(
    effr_futures_path: Path = EFFR_FUTURES_PATH,
) -> tuple[dict[tuple[str, int, int], float], dict[tuple[int, int], float]]:
    futures = pd.read_csv(effr_futures_path)
    futures["obs_date"] = pd.to_datetime(futures["Date_"], errors="coerce")
    futures["Volume"] = pd.to_numeric(futures["Volume"], errors="coerce")
    parsed = futures["ContrDate"].apply(parse_contrdate)
    futures["year"] = [p[0] for p in parsed]
    futures["month"] = [p[1] for p in parsed]
    valid = futures.dropna(subset=["obs_date", "year", "month", "Volume"]).copy()

    lookup = {
        (r.obs_date.strftime("%Y-%m-%d"), int(r.year), int(r.month)): float(r.Volume)
        for r in valid.itertuples()
    }
    monthly_median = valid.groupby(["year", "month"])["Volume"].median().to_dict()
    return lookup, monthly_median


def get_cme_adv_contracts(
    entry_date: pd.Timestamp,
    meeting_date: pd.Timestamp,
    vol_lookup: dict[tuple[str, int, int], float],
    monthly_median_lookup: dict[tuple[int, int], float],
) -> float:
    key = (entry_date.strftime("%Y-%m-%d"), meeting_date.year, meeting_date.month)
    if key in vol_lookup:
        return max(float(vol_lookup[key]), 1.0)
    return max(float(monthly_median_lookup.get((meeting_date.year, meeting_date.month), 100000.0)), 1.0)


def load_clean_kalshi_spread_data(kalshi_path: Path = KALSHI_PATH) -> pd.DataFrame:
    raw = pd.read_csv(kalshi_path)
    raw["spread"] = raw["yes_ask_close"] - raw["yes_bid_close"]
    raw["days_to_decision"] = (raw["market_close_ts"] - raw["candle_ts"]) / 86400.0
    raw["contract_type"] = raw["ticker"].str.extract(r"-([A-Z]+\d*)$")

    clean = (
        raw.dropna(subset=["spread", "yes_bid_close", "yes_ask_close"])
        .query("0 <= days_to_decision <= 120 and spread >= 0")
        .copy()
    )
    return clean


def build_spread_bins(df_clean: pd.DataFrame, bin_width: int = 10) -> pd.DataFrame:
    bins = np.arange(0, df_clean["days_to_decision"].max() + bin_width, bin_width)
    out = df_clean.copy()
    out["dtd_bin"] = pd.cut(out["days_to_decision"], bins=bins)
    binned = (
        out.groupby("dtd_bin", observed=True)["spread"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    binned["bin_mid"] = binned["dtd_bin"].apply(lambda x: x.mid)
    return binned


def derive_kalshi_three_zone_rule(
    binned: pd.DataFrame,
    t_far: int = 45,
    t_mid: int = 15,
) -> dict[str, float]:
    far_spread = round(float(binned.loc[binned["bin_mid"] > t_far, "median"].median()), 2)
    mid_spread = round(float(binned.loc[(binned["bin_mid"] > t_mid) & (binned["bin_mid"] <= t_far), "median"].median()), 2)
    near_spread = round(float(binned.loc[binned["bin_mid"] <= t_mid, "median"].median()), 2)
    return {
        "t_far": float(t_far),
        "t_mid": float(t_mid),
        "far_spread": far_spread,
        "mid_spread": mid_spread,
        "near_spread": near_spread,
    }


def kalshi_typical_spread(days_to_decision: float, rule: dict[str, float]) -> float:
    if days_to_decision > rule["t_far"]:
        return rule["far_spread"]
    if days_to_decision > rule["t_mid"]:
        return rule["mid_spread"]
    return rule["near_spread"]


def plot_kalshi_spread_panels(df_clean: pd.DataFrame, binned: pd.DataFrame, save_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Kalshi Bid-Ask Spread vs Days to Fed Decision", fontsize=13, fontweight="bold")

    ctypes = df_clean["contract_type"].dropna().unique()
    palette = plt.cm.tab10(np.linspace(0, 1, len(ctypes)))
    cmap = dict(zip(ctypes, palette))

    ax = axes[0]
    for contract_type, grp in df_clean.groupby("contract_type"):
        ax.scatter(
            grp["days_to_decision"],
            grp["spread"],
            color=cmap.get(contract_type, "steelblue"),
            alpha=0.25,
            s=8,
            label=contract_type,
            rasterized=True,
        )
    ax.set_xlabel("Days to Decision")
    ax.set_ylabel("Bid-Ask Spread ($)")
    ax.set_title("All Observations by Contract Type")
    ax.legend(title="Contract", fontsize=7, markerscale=2, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(binned["bin_mid"], binned["q25"], binned["q75"], alpha=0.25, color="steelblue", label="IQR (25-75th pct)")
    ax2.plot(
        binned["bin_mid"],
        binned["median"],
        color="steelblue",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Median spread",
    )
    for _, row in binned.iterrows():
        if row["n"] >= 5:
            ax2.annotate(
                f"n={int(row['n'])}",
                xy=(row["bin_mid"], row["q75"]),
                fontsize=6.5,
                ha="center",
                va="bottom",
                color="grey",
            )
    ax2.set_xlabel("Days to Decision")
    ax2.set_title("Binned Median + IQR (10-day bins)")
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_kalshi_spread_rule_fit(
    binned: pd.DataFrame,
    rule: dict[str, float],
    save_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(binned["bin_mid"], binned["q25"], binned["q75"], alpha=0.20, color="steelblue", label="IQR")
    ax.plot(binned["bin_mid"], binned["median"], color="steelblue", lw=2, marker="o", ms=5, label="Median spread")

    dtd_range = np.linspace(0, 120, 500)
    rule_line = [kalshi_typical_spread(d, rule) for d in dtd_range]
    ax.step(dtd_range, rule_line, color="crimson", lw=2.2, where="post", label="3-rule approximation", linestyle="--")

    for threshold, label in [(rule["t_far"], "45d"), (rule["t_mid"], "15d")]:
        ax.axvline(threshold, color="grey", lw=1, linestyle=":")
        ax.text(threshold + 0.5, ax.get_ylim()[1] * 0.98, label, fontsize=8, color="grey", va="top")

    ax.set_xlabel("Days to Fed Decision")
    ax.set_ylabel("Bid-Ask Spread ($)")
    ax.set_title("Kalshi Spread Rules vs Empirical Data")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax

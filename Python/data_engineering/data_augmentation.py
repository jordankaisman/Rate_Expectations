from __future__ import annotations
from pathlib import Path
import sys
import os
from typing import Any

import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Python.data_engineering.sofr_ois_expectations import parse_ois_tenor, tenor_to_maturity, year_fraction_act360


PREDICTION_BPS_MAP: dict[str, float] = {
    "polymarket_C75+": -75.0,
    "polymarket_C50+": -50.0,
    "polymarket_C50": -50.0,
    "polymarket_C25": -25.0,
    "polymarket_H0": 0.0,
    "polymarket_H25": 25.0,
    "polymarket_H25+": 25.0,
    "polymarket_H50": 50.0,
    "polymarket_H50+": 50.0,
    "polymarket_H75": 75.0,
    "kalshi_C50+": -50.0,
    "kalshi_C50": -50.0,
    "kalshi_C25": -25.0,
    "kalshi_H0": 0.0,
    "kalshi_H25": 25.0,
    "kalshi_H50": 50.0,
    "kalshi_H50+": 50.0,
}


def add_probability_bps_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, strike_bps in PREDICTION_BPS_MAP.items():
        if col in out.columns:
            out[f"{col}_bps"] = pd.to_numeric(out[col], errors="coerce") * strike_bps
    return out


def _month_distance(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).days / 30.4375


def _build_butterfly(
    curve_points: pd.DataFrame,
    maturity_col: str,
    rate_col: str,
    out_col: str,
    front_months: float = 0.0,
    belly_months: float = 6.0,
    back_months: float = 12.0,
) -> pd.DataFrame:
    if curve_points.empty:
        return pd.DataFrame(columns=["observed_day_pst", out_col])

    def _calc(group: pd.DataFrame) -> float:
        valid = group[[maturity_col, rate_col]].dropna()
        if valid.empty:
            return np.nan
        front_idx = (valid[maturity_col] - front_months).abs().idxmin()
        six_idx = (valid[maturity_col] - belly_months).abs().idxmin()
        twelve_idx = (valid[maturity_col] - back_months).abs().idxmin()
        front = float(valid.loc[front_idx, rate_col])
        six = float(valid.loc[six_idx, rate_col])
        twelve = float(valid.loc[twelve_idx, rate_col])
        # Standard butterfly: 2*belly - wing_1 - wing_2.
        return -front + 2.0 * six - twelve

    butterfly = (
        curve_points.groupby("observed_day_pst", as_index=False)
        .apply(lambda grp: pd.Series({out_col: _calc(grp)}), include_groups=False)
    )
    return butterfly


def _build_curve_spread(
    curve_points: pd.DataFrame,
    maturity_col: str,
    rate_col: str,
    out_col: str,
    short_months: float = 2.0,
    long_months: float = 12.0,
    dv01_weight_power: float = 1.0,
) -> pd.DataFrame:
    if curve_points.empty:
        return pd.DataFrame(columns=["observed_day_pst", out_col])

    # DV01-style weighting (duration proxy by maturity) to form a macro steepener.
    short_weight = (long_months / max(short_months, 1e-6)) ** dv01_weight_power
    long_weight = 1.0

    def _calc(group: pd.DataFrame) -> float:
        valid = group[[maturity_col, rate_col]].dropna()
        if valid.empty:
            return np.nan
        short_idx = (valid[maturity_col] - short_months).abs().idxmin()
        long_idx = (valid[maturity_col] - long_months).abs().idxmin()
        short_rate = float(valid.loc[short_idx, rate_col])
        long_rate = float(valid.loc[long_idx, rate_col])
        return short_weight * short_rate - long_weight * long_rate

    spread = (
        curve_points.groupby("observed_day_pst", as_index=False)
        .apply(lambda grp: pd.Series({out_col: _calc(grp)}), include_groups=False)
    )
    return spread


def _sr1_butterfly(sr1_path: str) -> pd.DataFrame:
    if not os.path.exists(sr1_path):
        return pd.DataFrame(columns=["observed_day_pst", "sr1_butterfly_bps"])
    sr1 = pd.read_csv(sr1_path, low_memory=False)
    sr1 = sr1[sr1.get("ExchTickerSymb") == "SR1"].copy()
    sr1["observed_day_pst"] = pd.to_datetime(sr1.get("Date_"), errors="coerce")
    sr1["last_trade"] = pd.to_datetime(sr1.get("LastTrdDate"), errors="coerce")
    sr1["implied_rate_bps"] = (100.0 - pd.to_numeric(sr1.get("Settlement"), errors="coerce")) * 100.0
    sr1 = sr1.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    sr1["maturity_months"] = sr1.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    sr1["observed_day_pst"] = sr1["observed_day_pst"].dt.date.astype(str)
    return _build_butterfly(sr1, "maturity_months", "implied_rate_bps", "sr1_butterfly_bps")


def _sr1_butterfly_hvar(sr1_path: str) -> pd.DataFrame:
    if not os.path.exists(sr1_path):
        return pd.DataFrame(columns=["observed_day_pst", "sr1_butterfly_bps_hvar"])
    sr1 = pd.read_csv(sr1_path, low_memory=False)
    sr1 = sr1[sr1.get("ExchTickerSymb") == "SR1"].copy()
    sr1["observed_day_pst"] = pd.to_datetime(sr1.get("Date_"), errors="coerce")
    sr1["last_trade"] = pd.to_datetime(sr1.get("LastTrdDate"), errors="coerce")
    sr1["implied_rate_bps"] = (100.0 - pd.to_numeric(sr1.get("Settlement"), errors="coerce")) * 100.0
    sr1 = sr1.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    sr1["maturity_months"] = sr1.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    sr1["observed_day_pst"] = sr1["observed_day_pst"].dt.date.astype(str)
    return _build_butterfly(
        sr1,
        "maturity_months",
        "implied_rate_bps",
        "sr1_butterfly_bps_hvar",
        front_months=3.0,
        belly_months=12.0,
        back_months=24.0,
    )


def _effr_butterfly(effr_path: str) -> pd.DataFrame:
    if not os.path.exists(effr_path):
        return pd.DataFrame(columns=["observed_day_pst", "effr_butterfly_bps"])
    effr = pd.read_csv(effr_path, low_memory=False)
    effr["observed_day_pst"] = pd.to_datetime(effr.get("Date_"), errors="coerce")
    effr["last_trade"] = pd.to_datetime(effr.get("LastTrdDate"), errors="coerce")
    effr["implied_rate_bps"] = (100.0 - pd.to_numeric(effr.get("Settlement"), errors="coerce")) * 100.0
    effr = effr.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    effr["maturity_months"] = effr.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    effr["observed_day_pst"] = effr["observed_day_pst"].dt.date.astype(str)
    return _build_butterfly(effr, "maturity_months", "implied_rate_bps", "effr_butterfly_bps")


def _effr_butterfly_hvar(effr_path: str) -> pd.DataFrame:
    if not os.path.exists(effr_path):
        return pd.DataFrame(columns=["observed_day_pst", "effr_butterfly_bps_hvar"])
    effr = pd.read_csv(effr_path, low_memory=False)
    effr["observed_day_pst"] = pd.to_datetime(effr.get("Date_"), errors="coerce")
    effr["last_trade"] = pd.to_datetime(effr.get("LastTrdDate"), errors="coerce")
    effr["implied_rate_bps"] = (100.0 - pd.to_numeric(effr.get("Settlement"), errors="coerce")) * 100.0
    effr = effr.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    effr["maturity_months"] = effr.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    effr["observed_day_pst"] = effr["observed_day_pst"].dt.date.astype(str)
    return _build_butterfly(
        effr,
        "maturity_months",
        "implied_rate_bps",
        "effr_butterfly_bps_hvar",
        front_months=3.0,
        belly_months=12.0,
        back_months=24.0,
    )


def _ois_butterfly(ois_path: str) -> pd.DataFrame:
    if not os.path.exists(ois_path):
        return pd.DataFrame(columns=["observed_day_pst", "ois_butterfly_bps"])
    try:
        ois_raw = pd.read_excel(ois_path, header=None)
    except ImportError:
        return pd.DataFrame(columns=["observed_day_pst", "ois_butterfly_bps"])
    rows: list[dict[str, Any]] = []
    for col in range(ois_raw.shape[1] - 1):
        code = ois_raw.iloc[0, col]
        if ois_raw.iloc[3, col + 1] != "CLOSE":
            continue
        tenor = parse_ois_tenor(code)
        if tenor is None:
            continue
        dates = pd.to_datetime(ois_raw.iloc[4:, col], errors="coerce")
        rates = pd.to_numeric(ois_raw.iloc[4:, col + 1], errors="coerce")
        for obs_dt, rate in zip(dates, rates):
            if pd.isna(obs_dt) or pd.isna(rate):
                continue
            obs_date = obs_dt.date()
            maturity = tenor_to_maturity(obs_date, tenor)
            maturity_months = 12.0 * year_fraction_act360(obs_date, maturity)
            rows.append(
                {
                    "observed_day_pst": str(obs_date),
                    "maturity_months": maturity_months,
                    "implied_rate_bps": float(rate) * 100.0,
                }
            )
    ois = pd.DataFrame(rows)
    return _build_butterfly(ois, "maturity_months", "implied_rate_bps", "ois_butterfly_bps")


def _ois_butterfly_hvar(ois_path: str) -> pd.DataFrame:
    if not os.path.exists(ois_path):
        return pd.DataFrame(columns=["observed_day_pst", "ois_butterfly_bps_hvar"])
    try:
        ois_raw = pd.read_excel(ois_path, header=None)
    except ImportError:
        return pd.DataFrame(columns=["observed_day_pst", "ois_butterfly_bps_hvar"])
    rows: list[dict[str, Any]] = []
    for col in range(ois_raw.shape[1] - 1):
        code = ois_raw.iloc[0, col]
        if ois_raw.iloc[3, col + 1] != "CLOSE":
            continue
        tenor = parse_ois_tenor(code)
        if tenor is None:
            continue
        dates = pd.to_datetime(ois_raw.iloc[4:, col], errors="coerce")
        rates = pd.to_numeric(ois_raw.iloc[4:, col + 1], errors="coerce")
        for obs_dt, rate in zip(dates, rates):
            if pd.isna(obs_dt) or pd.isna(rate):
                continue
            obs_date = obs_dt.date()
            maturity = tenor_to_maturity(obs_date, tenor)
            maturity_months = 12.0 * year_fraction_act360(obs_date, maturity)
            rows.append(
                {
                    "observed_day_pst": str(obs_date),
                    "maturity_months": maturity_months,
                    "implied_rate_bps": float(rate) * 100.0,
                }
            )
    ois = pd.DataFrame(rows)
    return _build_butterfly(
        ois,
        "maturity_months",
        "implied_rate_bps",
        "ois_butterfly_bps_hvar",
        front_months=3.0,
        belly_months=12.0,
        back_months=24.0,
    )


def _sr1_curve_spreads(sr1_path: str) -> pd.DataFrame:
    if not os.path.exists(sr1_path):
        return pd.DataFrame(columns=["observed_day_pst", "sr1_steepener_bps", "sr1_flattener_bps"])
    sr1 = pd.read_csv(sr1_path, low_memory=False)
    sr1 = sr1[sr1.get("ExchTickerSymb") == "SR1"].copy()
    sr1["observed_day_pst"] = pd.to_datetime(sr1.get("Date_"), errors="coerce")
    sr1["last_trade"] = pd.to_datetime(sr1.get("LastTrdDate"), errors="coerce")
    sr1["implied_rate_bps"] = (100.0 - pd.to_numeric(sr1.get("Settlement"), errors="coerce")) * 100.0
    sr1 = sr1.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    sr1["maturity_months"] = sr1.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    sr1["observed_day_pst"] = sr1["observed_day_pst"].dt.date.astype(str)
    steepener = _build_curve_spread(sr1, "maturity_months", "implied_rate_bps", "sr1_steepener_bps")
    steepener["sr1_flattener_bps"] = -steepener["sr1_steepener_bps"]
    return steepener


def _sr1_curve_spreads_hvar(sr1_path: str) -> pd.DataFrame:
    if not os.path.exists(sr1_path):
        return pd.DataFrame(columns=["observed_day_pst", "sr1_steepener_bps_hvar", "sr1_flattener_bps_hvar"])
    sr1 = pd.read_csv(sr1_path, low_memory=False)
    sr1 = sr1[sr1.get("ExchTickerSymb") == "SR1"].copy()
    sr1["observed_day_pst"] = pd.to_datetime(sr1.get("Date_"), errors="coerce")
    sr1["last_trade"] = pd.to_datetime(sr1.get("LastTrdDate"), errors="coerce")
    sr1["implied_rate_bps"] = (100.0 - pd.to_numeric(sr1.get("Settlement"), errors="coerce")) * 100.0
    sr1 = sr1.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    sr1["maturity_months"] = sr1.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    sr1["observed_day_pst"] = sr1["observed_day_pst"].dt.date.astype(str)
    steepener = _build_curve_spread(
        sr1,
        "maturity_months",
        "implied_rate_bps",
        "sr1_steepener_bps_hvar",
        short_months=1.0,
        long_months=24.0,
        dv01_weight_power=1.5,
    )
    steepener["sr1_flattener_bps_hvar"] = -steepener["sr1_steepener_bps_hvar"]
    return steepener


def _effr_curve_spreads(effr_path: str) -> pd.DataFrame:
    if not os.path.exists(effr_path):
        return pd.DataFrame(columns=["observed_day_pst", "effr_steepener_bps", "effr_flattener_bps"])
    effr = pd.read_csv(effr_path, low_memory=False)
    effr["observed_day_pst"] = pd.to_datetime(effr.get("Date_"), errors="coerce")
    effr["last_trade"] = pd.to_datetime(effr.get("LastTrdDate"), errors="coerce")
    effr["implied_rate_bps"] = (100.0 - pd.to_numeric(effr.get("Settlement"), errors="coerce")) * 100.0
    effr = effr.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    effr["maturity_months"] = effr.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    effr["observed_day_pst"] = effr["observed_day_pst"].dt.date.astype(str)
    steepener = _build_curve_spread(effr, "maturity_months", "implied_rate_bps", "effr_steepener_bps")
    steepener["effr_flattener_bps"] = -steepener["effr_steepener_bps"]
    return steepener


def _effr_curve_spreads_hvar(effr_path: str) -> pd.DataFrame:
    if not os.path.exists(effr_path):
        return pd.DataFrame(columns=["observed_day_pst", "effr_steepener_bps_hvar", "effr_flattener_bps_hvar"])
    effr = pd.read_csv(effr_path, low_memory=False)
    effr["observed_day_pst"] = pd.to_datetime(effr.get("Date_"), errors="coerce")
    effr["last_trade"] = pd.to_datetime(effr.get("LastTrdDate"), errors="coerce")
    effr["implied_rate_bps"] = (100.0 - pd.to_numeric(effr.get("Settlement"), errors="coerce")) * 100.0
    effr = effr.dropna(subset=["observed_day_pst", "last_trade", "implied_rate_bps"])
    effr["maturity_months"] = effr.apply(
        lambda row: _month_distance(row["observed_day_pst"], row["last_trade"]),
        axis=1,
    )
    effr["observed_day_pst"] = effr["observed_day_pst"].dt.date.astype(str)
    steepener = _build_curve_spread(
        effr,
        "maturity_months",
        "implied_rate_bps",
        "effr_steepener_bps_hvar",
        short_months=1.0,
        long_months=24.0,
        dv01_weight_power=1.5,
    )
    steepener["effr_flattener_bps_hvar"] = -steepener["effr_steepener_bps_hvar"]
    return steepener


def _ois_curve_spreads(ois_path: str) -> pd.DataFrame:
    if not os.path.exists(ois_path):
        return pd.DataFrame(columns=["observed_day_pst", "ois_steepener_bps", "ois_flattener_bps"])
    try:
        ois_raw = pd.read_excel(ois_path, header=None)
    except ImportError:
        return pd.DataFrame(columns=["observed_day_pst", "ois_steepener_bps", "ois_flattener_bps"])
    rows: list[dict[str, Any]] = []
    for col in range(ois_raw.shape[1] - 1):
        code = ois_raw.iloc[0, col]
        if ois_raw.iloc[3, col + 1] != "CLOSE":
            continue
        tenor = parse_ois_tenor(code)
        if tenor is None:
            continue
        dates = pd.to_datetime(ois_raw.iloc[4:, col], errors="coerce")
        rates = pd.to_numeric(ois_raw.iloc[4:, col + 1], errors="coerce")
        for obs_dt, rate in zip(dates, rates):
            if pd.isna(obs_dt) or pd.isna(rate):
                continue
            obs_date = obs_dt.date()
            maturity = tenor_to_maturity(obs_date, tenor)
            maturity_months = 12.0 * year_fraction_act360(obs_date, maturity)
            rows.append(
                {
                    "observed_day_pst": str(obs_date),
                    "maturity_months": maturity_months,
                    "implied_rate_bps": float(rate) * 100.0,
                }
            )
    ois = pd.DataFrame(rows)
    steepener = _build_curve_spread(ois, "maturity_months", "implied_rate_bps", "ois_steepener_bps")
    steepener["ois_flattener_bps"] = -steepener["ois_steepener_bps"]
    return steepener


def _ois_curve_spreads_hvar(ois_path: str) -> pd.DataFrame:
    if not os.path.exists(ois_path):
        return pd.DataFrame(columns=["observed_day_pst", "ois_steepener_bps_hvar", "ois_flattener_bps_hvar"])
    try:
        ois_raw = pd.read_excel(ois_path, header=None)
    except ImportError:
        return pd.DataFrame(columns=["observed_day_pst", "ois_steepener_bps_hvar", "ois_flattener_bps_hvar"])
    rows: list[dict[str, Any]] = []
    for col in range(ois_raw.shape[1] - 1):
        code = ois_raw.iloc[0, col]
        if ois_raw.iloc[3, col + 1] != "CLOSE":
            continue
        tenor = parse_ois_tenor(code)
        if tenor is None:
            continue
        dates = pd.to_datetime(ois_raw.iloc[4:, col], errors="coerce")
        rates = pd.to_numeric(ois_raw.iloc[4:, col + 1], errors="coerce")
        for obs_dt, rate in zip(dates, rates):
            if pd.isna(obs_dt) or pd.isna(rate):
                continue
            obs_date = obs_dt.date()
            maturity = tenor_to_maturity(obs_date, tenor)
            maturity_months = 12.0 * year_fraction_act360(obs_date, maturity)
            rows.append(
                {
                    "observed_day_pst": str(obs_date),
                    "maturity_months": maturity_months,
                    "implied_rate_bps": float(rate) * 100.0,
                }
            )
    ois = pd.DataFrame(rows)
    steepener = _build_curve_spread(
        ois,
        "maturity_months",
        "implied_rate_bps",
        "ois_steepener_bps_hvar",
        short_months=1.0,
        long_months=24.0,
        dv01_weight_power=1.5,
    )
    steepener["ois_flattener_bps_hvar"] = -steepener["ois_steepener_bps_hvar"]
    return steepener


def augment_prediction_panel(
    prediction_path: str = os.path.join("Data", "Merged", "Prediction_all_with_sofr.csv"),
    out_path: str = os.path.join("Data", "Merged", "Prediction_all_augmented.csv"),
    sr1_path: str = os.path.join("Data", "SOFR", "SR1.csv"),
    effr_path: str = os.path.join("Data", "EFFR_Futures", "effr_futures.csv"),
    ois_path: str = os.path.join("Data", "SOFR", "SOFR OIS.xlsx"),
    use_high_variance_instruments: bool = False,
) -> None:
    merged = pd.read_csv(prediction_path)
    merged = add_probability_bps_columns(merged)

    for curve_aug in (
        _sr1_butterfly(sr1_path),
        _effr_butterfly(effr_path),
        _ois_butterfly(ois_path),
        _sr1_butterfly_hvar(sr1_path),
        _effr_butterfly_hvar(effr_path),
        _ois_butterfly_hvar(ois_path),
        _sr1_curve_spreads(sr1_path),
        _effr_curve_spreads(effr_path),
        _ois_curve_spreads(ois_path),
        _sr1_curve_spreads_hvar(sr1_path),
        _effr_curve_spreads_hvar(effr_path),
        _ois_curve_spreads_hvar(ois_path),
    ):
        merged = merged.merge(curve_aug, on="observed_day_pst", how="left")

    if use_high_variance_instruments:
        for base in [
            "sr1_butterfly_bps",
            "effr_butterfly_bps",
            "ois_butterfly_bps",
            "sr1_steepener_bps",
            "effr_steepener_bps",
            "ois_steepener_bps",
            "sr1_flattener_bps",
            "effr_flattener_bps",
            "ois_flattener_bps",
        ]:
            hvar = f"{base}_hvar"
            if hvar in merged.columns:
                merged[base] = merged[hvar]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)


def main() -> None:
    augment_prediction_panel()


if __name__ == "__main__":
    main()

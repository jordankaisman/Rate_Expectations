from __future__ import annotations
from pathlib import Path
import os
import sys

# Ensure repo root is on sys.path (nice for consistency when clicking Play)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
import json
from typing import Any, Iterable
import numpy as np
import pandas as pd
from Python.data_engineering.EFFR_dataload import merge_effr_into_panel
from Python.data_engineering.sofr_ois_expectations import (
    OISTenor,
    SR1MatrixEstimator,
    OISMatrixEstimator,
    parse_ois_tenor,
    tenor_to_maturity,
    year_fraction_act360,
)


@dataclass(frozen=True)
class PipelineConfig:
    step_size: float = 0.0025
    interpolation: str = "flat_forward_meetings"
    anchor_windows: tuple[int, int] = (1, 3)
    outcomes: tuple[float, ...] = (-0.075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.075)
    futures_ois_diff_threshold: float = 0.0010
    meeting_start: date = date(2023, 2, 1)
    meeting_end: date = date(2028, 1, 31)
    forecast_months: int = 24


CONFIG = PipelineConfig()
def _to_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date value: {value!r}")



def _normalize_rate(rate: float) -> float:
    return rate / 100.0 if rate > 1.0 else rate



def _tenor_months_from_tenor(tenor: OISTenor, valuation_date: date) -> float:
    maturity_date = tenor_to_maturity(valuation_date, tenor)
    return 12.0 * year_fraction_act360(valuation_date, maturity_date)

def build_sofr_expectations_csv(
    futures_path: str,
    ois_path: str,
    merged_path: str,
    out_path: str,
    methods: Iterable[str] | None = None,
) -> None:
    """
    Build trade-focused SOFR-implied meeting expectations and probabilities.

    Trade output methods:
      - trade_sr1_chain
      - trade_ois_forward_chain

    Legacy bootstrap variants are retained only for robustness diagnostics and are
    no longer emitted into the trade output table.
    """

    merged = pd.read_csv(merged_path)
    meeting_col = "meeting_date" if "meeting_date" in merged.columns else "decision_date"
    meetings = sorted({_to_date(d) for d in merged[meeting_col].dropna().unique()})

    resolved_futures_path = futures_path
    if not os.path.exists(resolved_futures_path):
        fallback_path = os.path.join(os.path.dirname(futures_path), "SOFR Futures.csv")
        if os.path.exists(fallback_path):
            resolved_futures_path = fallback_path
    if not os.path.exists(resolved_futures_path):
        raise FileNotFoundError(f"Could not find futures data at {futures_path} or fallback SOFR Futures.csv")
    futures = pd.read_csv(
        resolved_futures_path,
        usecols=["Date_", "Settlement", "LastTrdDate", "ExchTickerSymb", "Volume", "OpenInterest"],
        low_memory=False,
    )
    if futures.empty:
        raise ValueError(f"Futures dataset is empty: {resolved_futures_path}")
    has_sr1 = (futures["ExchTickerSymb"] == "SR1").any()
    has_sr3 = (futures["ExchTickerSymb"] == "SR3").any()
    if not has_sr1 and not has_sr3:
        raise ValueError("Futures dataset must contain SR1 (preferred) or SR3 contracts")
    futures = futures[futures["ExchTickerSymb"].isin(["SR1", "SR3"])].copy()
    futures["observed_day_pst"] = pd.to_datetime(futures["Date_"], errors="coerce").dt.date.astype(str)
    futures["last_trade_date"] = pd.to_datetime(futures["LastTrdDate"], errors="coerce").dt.date
    futures["price"] = pd.to_numeric(futures["Settlement"], errors="coerce")
    futures["Volume"] = pd.to_numeric(futures["Volume"], errors="coerce")
    futures["OpenInterest"] = pd.to_numeric(futures["OpenInterest"], errors="coerce")
    futures = futures.dropna(subset=["observed_day_pst", "last_trade_date", "price"])

    sr1_mask = futures["ExchTickerSymb"] == "SR1"
    futures.loc[sr1_mask, "instrument"] = "SR1:" + pd.to_datetime(
        futures.loc[sr1_mask, "last_trade_date"]
    ).dt.to_period("M").astype(str)
    futures.loc[~sr1_mask, "instrument"] = "SR3:" + pd.to_datetime(
        futures.loc[~sr1_mask, "last_trade_date"]
    ).dt.date.astype(str)
    futures = futures.sort_values(["observed_day_pst", "instrument", "Volume", "OpenInterest"])
    futures = futures.groupby(["observed_day_pst", "instrument"], as_index=False).tail(1)

    ois_raw = pd.read_excel(ois_path, header=None)
    ois_records: list[dict[str, Any]] = []
    for col in range(ois_raw.shape[1] - 1):
        code = ois_raw.iloc[0, col]
        if ois_raw.iloc[3, col + 1] != "CLOSE":
            continue
        tenor = parse_ois_tenor(code)
        if tenor is None:
            continue

        dates = pd.to_datetime(ois_raw.iloc[4:, col], errors="coerce")
        rates = pd.to_numeric(ois_raw.iloc[4:, col + 1], errors="coerce")
        # Precompute per-date tenor months once per OIS tenor column to avoid repeated holiday-calendar work.
        unique_dates = {d.date() for d in dates if pd.notna(d)}
        tenor_months_by_date = {d: _tenor_months_from_tenor(tenor, d) for d in unique_dates}
        tenor_months = dates.dt.date.map(tenor_months_by_date)
        frame = pd.DataFrame(
            {
                "observed_day_pst": dates.dt.date.astype(str),
                "tenor_months": tenor_months,
                "rate": rates / 100.0,
            }
        ).dropna(subset=["observed_day_pst", "rate"])
        if frame.empty:
            continue
        ois_records.extend(frame.to_dict(orient="records"))

    ois = pd.DataFrame(ois_records)
    ois = ois.dropna(subset=["observed_day_pst", "tenor_months", "rate"])

    futures_by_day = {
        day: grp[["instrument", "price"]].to_dict(orient="records")
        for day, grp in futures.groupby("observed_day_pst")
    }
    ois_by_day = {day: grp for day, grp in ois.groupby("observed_day_pst")}

    requested_methods = set(methods) if methods is not None else None

    rows: list[dict[str, Any]] = []
    observed_days = sorted(set(merged["observed_day_pst"].dropna().unique()) & set(ois_by_day.keys()))
    for observed_day in observed_days:
        ois_day = ois_by_day[observed_day].sort_values("tenor_months")
        ois_swaps = [
            {"tenor_months": float(r["tenor_months"]), "rate": float(r["rate"])}
            for _, r in ois_day.iterrows()
        ]
        if not ois_swaps:
            continue

        overnight_rate = float(ois_day.iloc[0]["rate"])
        valuation_date = _to_date(observed_day)
        sr1_instruments = {
            str(r["instrument"]): _normalize_rate(100.0 - float(r["price"]))
            for r in futures_by_day.get(observed_day, [])
        }
        sr1_chain = SR1MatrixEstimator(valuation_date, sr1_instruments, meetings)
        sr1_rows, sr1_weights = sr1_chain.estimate()

        ois_chain = OISMatrixEstimator(
            valuation_date,
            [{"tenor_months": float(r["tenor_months"]), "rate": float(r["rate"])} for _, r in ois_day.iterrows()],
            meetings,
        )
        ois_rows, ois_weights = ois_chain.estimate()

        # Map results by meeting_date so we emit one row per meeting_date + observed_day
        sr1_map = {r["meeting_date"]: r for r in sr1_rows}
        ois_map = {r["meeting_date"]: r for r in ois_rows}
        meeting_keys = sorted(set(list(sr1_map.keys()) + list(ois_map.keys())))

        for meeting_date in meeting_keys:
            sr1_r = sr1_map.get(meeting_date)
            ois_r = ois_map.get(meeting_date)
            out_row: dict[str, Any] = {
                "decision_date": meeting_date,
                "observed_day_pst": observed_day,
                "sofr_pre_rate": None,
                "sofr_post_rate": None,
                "jump_sr1": None,
                "jump_ois": None,
                "jump_sr1_sensitivity": None,
                "jump_ois_sensitivity": None,
                "jump_sr1_hedge_1bp_contract": None,
                "jump_ois_hedge_1bp_contract": None,
                "jump_sr1_portfolio_weights": None,
                "jump_ois_portfolio_weights": None,
            }
            if sr1_r is not None:
                out_row.update(
                    {
                        "sofr_pre_rate": sr1_r.get("r_pre"),
                        "sofr_post_rate": sr1_r.get("r_post"),
                        "jump_sr1": sr1_r.get("jump_sr1", sr1_r.get("jump")),
                        "jump_sr1_sensitivity": sr1_r.get("sensitivity_sr1"),
                        "jump_sr1_hedge_1bp_contract": sr1_r.get("hedge_1bp_sr1_contract"),
                        "jump_sr1_portfolio_weights": json.dumps(sr1_weights.get(meeting_date, {}), sort_keys=True),
                    }
                )
            if ois_r is not None:
                # prefer OIS pre/post rates if present (they should be similar)
                out_row.update(
                    {
                        "sofr_pre_rate": ois_r.get("r_pre") if ois_r.get("r_pre") is not None else out_row.get("sofr_pre_rate"),
                        "sofr_post_rate": ois_r.get("r_post") if ois_r.get("r_post") is not None else out_row.get("sofr_post_rate"),
                        "jump_ois": ois_r.get("jump_ois", ois_r.get("jump")),
                        "jump_ois_sensitivity": ois_r.get("sensitivity_ois"),
                        "jump_ois_hedge_1bp_contract": ois_r.get("hedge_1bp_ois_contract"),
                        "jump_ois_portfolio_weights": json.dumps(ois_weights.get(meeting_date, {}), sort_keys=True),
                    }
                )
            rows.append(out_row)

    # ensure consistent column order and presence
    out_df = pd.DataFrame(rows)
    if "sofr_expected_change" not in out_df.columns:
        # prefer jump_ois then jump_sr1 for expected change
        out_df["sofr_expected_change"] = out_df.get("jump_ois").fillna(out_df.get("jump_sr1"))
    out_df = out_df.sort_values(["decision_date", "observed_day_pst"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)


def merge_sofr_into_prediction_csv(
    prediction_merged_path: str,
    sofr_expectations_path: str,
    out_path: str,
    effr_futures_path: str | None = None,
) -> None:
    import pandas as pd

    prediction = pd.read_csv(prediction_merged_path)
    sofr = pd.read_csv(sofr_expectations_path)
    # New format: expectations file contains `jump_sr1` and `jump_ois` columns.
    # Build a wide table with the last-observed jump values per decision/observed day
    trade_wide = (
        sofr.groupby(["decision_date", "observed_day_pst"], as_index=False)
        .agg(
            {
                "jump_sr1": "last",
                "jump_ois": "last",
                "jump_sr1_portfolio_weights": "last",
                "jump_ois_portfolio_weights": "last",
            }
        )
    )
    merged = prediction.merge(trade_wide, on=["decision_date", "observed_day_pst"], how="left")
    merged["jump_sr1_bps"] = pd.to_numeric(merged["jump_sr1"], errors="coerce") * 10000.0
    merged["jump_ois_bps"] = pd.to_numeric(merged["jump_ois"], errors="coerce") * 10000.0

    def _sum_cols(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
        found = [c for c in candidates if c in frame.columns]
        if not found:
            return pd.Series(np.nan, index=frame.index)
        return frame[found].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

    def _col_or(frame: pd.DataFrame, first: str, fallback: str) -> pd.Series:
        if first in frame.columns:
            return frame[first]
        return frame.get(fallback)

    merged["Poly_Hike"] = _sum_cols(merged, ["polymarket_H25", "polymarket_H50"])
    merged["Poly_Cut"] = _sum_cols(merged, ["polymarket_C25", "polymarket_C50", "polymarket_C50+"])
    merged["Kalshi_Hike"] = _sum_cols(merged, ["kalshi_H25", "kalshi_H50"])
    merged["Kalshi_Cut"] = _sum_cols(merged, ["kalshi_C25", "kalshi_C50", "kalshi_C50+"])

    merged["Poly_Hike_bps"] = (
        25.0 * pd.to_numeric(merged.get("polymarket_H25"), errors="coerce")
        + 50.0 * pd.to_numeric(merged.get("polymarket_H50"), errors="coerce")
    )
    merged["Poly_Cut_bps"] = (
        -25.0 * pd.to_numeric(merged.get("polymarket_C25"), errors="coerce")
        - 50.0
        * pd.to_numeric(_col_or(merged, "polymarket_C50", "polymarket_C50+"), errors="coerce")
    )
    merged["Kalshi_Hike_bps"] = (
        25.0 * pd.to_numeric(merged.get("kalshi_H25"), errors="coerce")
        + 50.0 * pd.to_numeric(merged.get("kalshi_H50"), errors="coerce")
    )
    merged["Kalshi_Cut_bps"] = (
        -25.0 * pd.to_numeric(merged.get("kalshi_C25"), errors="coerce")
        - 50.0
        * pd.to_numeric(_col_or(merged, "kalshi_C50", "kalshi_C50+"), errors="coerce")
    )

    if effr_futures_path and os.path.exists(effr_futures_path):
        merged = merge_effr_into_panel(merged, effr_futures_path)
    else:
        merged["effr_expected_bps"] = np.nan
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)


def main() -> None:
    default_sr1_path = os.path.join("Data", "SOFR", "SR1.csv")
    futures_path = default_sr1_path if os.path.exists(default_sr1_path) else os.path.join("Data", "SOFR", "SOFR Futures.csv")
    ois_path = os.path.join("Data", "SOFR", "SOFR OIS.xlsx")
    effr_path = os.path.join("Data", "EFFR_Futures", "effr_futures.csv")
    prediction_merged_path = os.path.join("Data", "Merged", "Prediction_all.csv")
    sofr_out_path = os.path.join("Data", "SOFR", "SOFR_expectations.csv")
    merged_out_path = os.path.join("Data", "Merged", "Prediction_all_with_sofr.csv")

    build_sofr_expectations_csv(
        futures_path=futures_path,
        ois_path=ois_path,
        merged_path=prediction_merged_path,
        out_path=sofr_out_path,
    )
    merge_sofr_into_prediction_csv(
        prediction_merged_path=prediction_merged_path,
        sofr_expectations_path=sofr_out_path,
        out_path=merged_out_path,
        effr_futures_path=effr_path,
    )


if __name__ == "__main__":
    main()

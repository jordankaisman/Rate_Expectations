from __future__ import annotations

import calendar
from typing import Iterable

import numpy as np
import pandas as pd


LATE_MONTH_THRESHOLD = 0.67


def _build_effr_lookup(effr_path: str) -> dict[tuple[str, int, int], float]:
    effr = pd.read_csv(
        effr_path,
        usecols=["Date_", "Settlement", "LastTrdDate", "DSMnem"],
        low_memory=False,
    )
    effr["obs_date"] = pd.to_datetime(effr["Date_"], errors="coerce")
    effr["Settlement"] = pd.to_numeric(effr["Settlement"], errors="coerce")
    effr = effr.dropna(subset=["obs_date", "Settlement"])
    effr["implied_rate"] = (100.0 - effr["Settlement"]) / 100.0

    effr["last_trade"] = pd.to_datetime(effr["LastTrdDate"], errors="coerce")
    effr = effr.dropna(subset=["last_trade"])
    effr["contr_year"] = effr["last_trade"].dt.year
    effr["contr_month"] = effr["last_trade"].dt.month
    effr["obs_str"] = effr["obs_date"].dt.strftime("%Y-%m-%d")

    effr = effr.sort_values("obs_date")
    effr = effr.drop_duplicates(subset=["obs_str", "contr_year", "contr_month"], keep="last")

    return {
        (row["obs_str"], int(row["contr_year"]), int(row["contr_month"])): float(row["implied_rate"])
        for _, row in effr.iterrows()
    }


def build_effr_expected_changes(
    panel: pd.DataFrame,
    effr_path: str,
    late_month_threshold: float = LATE_MONTH_THRESHOLD,
) -> pd.DataFrame:
    """
    Return EFFR implied expected meeting jumps (in bps) by decision_date/observed_day_pst.
    """
    if panel.empty:
        return pd.DataFrame(columns=["decision_date", "observed_day_pst", "effr_expected_bps"])

    if "decision_date" not in panel.columns or "observed_day_pst" not in panel.columns:
        raise ValueError("panel must contain decision_date and observed_day_pst columns")

    effr_lookup = _build_effr_lookup(effr_path)
    if not effr_lookup:
        return pd.DataFrame(columns=["decision_date", "observed_day_pst", "effr_expected_bps"])

    base = panel[["decision_date", "observed_day_pst"]].dropna().copy()
    base["obs_str"] = pd.to_datetime(base["observed_day_pst"], errors="coerce").dt.strftime("%Y-%m-%d")
    base = base.dropna(subset=["obs_str"])

    effr_rows: list[dict[str, object]] = []
    meeting_dates: Iterable[str] = sorted(base["decision_date"].astype(str).unique())
    for meeting_str in meeting_dates:
        meeting_dt = pd.Timestamp(meeting_str)
        m_day = meeting_dt.day
        m_year = meeting_dt.year
        m_month = meeting_dt.month
        days_in_month = calendar.monthrange(m_year, m_month)[1]

        late_month = (m_day / days_in_month) > late_month_threshold
        if late_month:
            next_year, next_month = (m_year + 1, 1) if m_month == 12 else (m_year, m_month + 1)
            contract_a = (m_year, m_month)
            contract_b = (next_year, next_month)
        else:
            prev_year, prev_month = (m_year - 1, 12) if m_month == 1 else (m_year, m_month - 1)
            contract_a = (prev_year, prev_month)
            contract_b = (m_year, m_month)

        obs_strs = (
            base.loc[base["decision_date"].astype(str) == meeting_str, "obs_str"]
            .sort_values()
            .unique()
        )
        for obs_str in obs_strs:
            rate_a = effr_lookup.get((obs_str, contract_a[0], contract_a[1]))
            rate_b = effr_lookup.get((obs_str, contract_b[0], contract_b[1]))
            if rate_a is None or rate_b is None:
                continue

            if late_month:
                pre_rate = rate_a
                post_rate = rate_b
            else:
                pre_rate = rate_a
                meeting_avg = rate_b
                denom = days_in_month - m_day
                if denom <= 0:
                    continue
                post_rate = (meeting_avg * days_in_month - pre_rate * m_day) / denom

            effr_rows.append(
                {
                    "decision_date": meeting_str,
                    "observed_day_pst": obs_str,
                    "effr_expected_bps": float((post_rate - pre_rate) * 10000.0),
                }
            )

    out = pd.DataFrame(effr_rows)
    if out.empty:
        return pd.DataFrame(columns=["decision_date", "observed_day_pst", "effr_expected_bps"])
    return (
        out.sort_values(["decision_date", "observed_day_pst"])
        .drop_duplicates(subset=["decision_date", "observed_day_pst"], keep="last")
    )


def merge_effr_into_panel(panel: pd.DataFrame, effr_path: str) -> pd.DataFrame:
    merged = panel.copy()
    effr = build_effr_expected_changes(merged, effr_path=effr_path)
    merged = merged.merge(effr, on=["decision_date", "observed_day_pst"], how="left")
    if "effr_expected_bps" not in merged.columns:
        merged["effr_expected_bps"] = np.nan
    return merged


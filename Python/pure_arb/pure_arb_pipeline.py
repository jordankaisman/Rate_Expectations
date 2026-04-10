from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_repo_root() -> Path:
    cwd = Path.cwd()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "Data").exists():
            return candidate
    raise FileNotFoundError("Could not resolve repo root")


REPO_ROOT = resolve_repo_root()
PREDICTION_PATH = REPO_ROOT / "Data" / "Merged" / "Prediction_all_with_sofr.csv"
KALSHI_PATH = REPO_ROOT / "Data" / "Kalshi" / "Kalshi_rates.csv"
EFFR_FUTURES_PATH = REPO_ROOT / "Data" / "EFFR_Futures" / "effr_futures.csv"
MONTHLY_EFFR_PATH = REPO_ROOT / "Data" / "EFFR_Futures" / "monthly_EFFR.csv"
FED_DECISIONS_PATH = REPO_ROOT / "Data" / "EFFR_Futures" / "fed_decisions.csv"
OUT_DIR = REPO_ROOT / "Data" / "Outputs" / "pure_arb"

LATE_MONTH_THRESHOLD = 0.80
MIN_TOTAL_PROB = 0.75

KALSHI_MAP = {
    "kalshi_C50+": -50,
    "kalshi_C25": -25,
    "kalshi_H0": 0,
    "kalshi_H25": 25,
    "kalshi_H50": 50,
    "kalshi_H50+": 50,
}

POLY_MAP = {
    "polymarket_C75+": -75,
    "polymarket_C50+": -50,
    "polymarket_C50": -50,
    "polymarket_C25": -25,
    "polymarket_H0": 0,
    "polymarket_H25": 25,
    "polymarket_H25+": 25,
    "polymarket_H50": 50,
    "polymarket_H50+": 50,
    "polymarket_H75": 75,
}

SUFFIX_TO_BPS = {
    "C25": -25,
    "C>25": -50,
    "C26": -50,
    "C24": -25,
    "H0": 0,
    "H25": 25,
    "H>25": 50,
    "H26": 50,
    "H50": 50,
    "TC25": -25,
    "TH50": 50,
}

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def expected_change_bps(row: pd.Series, col_map: dict[str, float], min_prob: float = 0.01) -> float:
    total_prob = 0.0
    weighted = 0.0
    for col, bps in col_map.items():
        p = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(p):
            total_prob += float(p)
            weighted += float(p) * float(bps)
    return weighted / total_prob if total_prob >= min_prob else np.nan


def parse_kalshi_ticker(ticker: str) -> tuple[int | None, int | None, str | None]:
    m = re.search(r"(\d{2})([A-Z]{3})\d*-([A-Z>\d]+)$", str(ticker))
    if not m:
        return None, None, None
    year = 2000 + int(m.group(1))
    month = MONTH_MAP.get(m.group(2))
    suffix = m.group(3)
    return year, month, suffix


def parse_contrdate(v: float | int | str) -> tuple[int | None, int | None]:
    if pd.isna(v):
        return None, None
    s = str(v).split(".")[0].strip()
    if not s:
        return None, None
    if len(s) <= 3:
        month = int(s[0])
        year = 2000 + int(s[1:])
    else:
        month = int(s[:-2])
        year = 2000 + int(s[-2:])
    return year, month


def load_base_panel(prediction_path: Path = PREDICTION_PATH) -> pd.DataFrame:
    df = pd.read_csv(prediction_path)
    df["observed_day_pst"] = pd.to_datetime(df["observed_day_pst"], errors="coerce")
    df = df.dropna(subset=["decision_date", "observed_day_pst"]).copy()

    df["kalshi_expected_bps"] = df.apply(lambda r: expected_change_bps(r, KALSHI_MAP), axis=1)
    df["poly_expected_bps"] = df.apply(lambda r: expected_change_bps(r, POLY_MAP), axis=1)
    df["sofr_expected_bps"] = pd.to_numeric(df.get("jump_ois"), errors="coerce") * 10000.0
    return df


def merge_kalshi_bid_ask(df: pd.DataFrame, kalshi_path: Path = KALSHI_PATH) -> pd.DataFrame:
    k = pd.read_csv(kalshi_path)
    k["obs_date"] = pd.to_datetime(k["candle_ts"], unit="s", errors="coerce")
    k["obs_str"] = k["obs_date"].dt.strftime("%Y-%m-%d")

    parsed = k["ticker"].apply(parse_kalshi_ticker)
    k["k_year"] = [p[0] for p in parsed]
    k["k_month"] = [p[1] for p in parsed]
    k["k_suffix"] = [p[2] for p in parsed]
    k["k_bps"] = k["k_suffix"].map(SUFFIX_TO_BPS)

    k["yes_bid"] = pd.to_numeric(k["yes_bid_close"], errors="coerce")
    k["yes_ask"] = pd.to_numeric(k["yes_ask_close"], errors="coerce")
    k["volume"] = pd.to_numeric(k["volume"], errors="coerce")
    k = k.dropna(subset=["k_year", "k_month", "k_bps", "obs_date"]).copy()

    meeting_dates = {
        (pd.Timestamp(d).year, pd.Timestamp(d).month): d for d in df["decision_date"].unique()
    }
    k["decision_date"] = k.apply(lambda r: meeting_dates.get((int(r["k_year"]), int(r["k_month"]))), axis=1)
    k = k.dropna(subset=["decision_date"]).copy()

    def _bid_ask(group: pd.DataFrame) -> pd.Series:
        bid_rows = group.dropna(subset=["yes_bid"])
        ask_rows = group.dropna(subset=["yes_ask"])
        bid_weighted = (bid_rows["yes_bid"] * bid_rows["k_bps"]).sum()
        bid_total = bid_rows["yes_bid"].sum()
        ask_weighted = (ask_rows["yes_ask"] * ask_rows["k_bps"]).sum()
        ask_total = ask_rows["yes_ask"].sum()
        return pd.Series(
            {
                "kalshi_bid_bps": bid_weighted / bid_total if bid_total > 0.01 else np.nan,
                "kalshi_ask_bps": ask_weighted / ask_total if ask_total > 0.01 else np.nan,
                "kalshi_volume": group["volume"].sum(),
                "kalshi_avg_spread": (group["yes_ask"] - group["yes_bid"]).mean(),
            }
        )

    out = df.copy()
    out["obs_str"] = out["observed_day_pst"].dt.strftime("%Y-%m-%d")

    if k.empty:
        for col in ("kalshi_bid_bps", "kalshi_ask_bps", "kalshi_volume", "kalshi_avg_spread"):
            out[col] = np.nan
        return out

    grouped = k.groupby(["decision_date", "obs_str"], group_keys=False)
    try:
        k_ba = grouped.apply(_bid_ask, include_groups=False).reset_index()
    except TypeError:
        k_ba = grouped.apply(_bid_ask).reset_index()

    return out.merge(k_ba, on=["decision_date", "obs_str"], how="left")


def apply_forward_fill(df: pd.DataFrame, min_total_prob: float = MIN_TOTAL_PROB) -> pd.DataFrame:
    out = df.copy()
    poly_cols = [c for c in out.columns if c.startswith("polymarket_")]
    kalshi_cols = [c for c in out.columns if c.startswith("kalshi_")]

    for meeting in out["decision_date"].dropna().unique():
        idx = out.loc[out["decision_date"] == meeting].sort_values("observed_day_pst").index
        out.loc[idx, poly_cols] = out.loc[idx, poly_cols].ffill(limit=1)
        out.loc[idx, kalshi_cols] = out.loc[idx, kalshi_cols].ffill(limit=1)

    out["kalshi_expected_bps_ff"] = out.apply(
        lambda r: expected_change_bps(r, KALSHI_MAP, min_prob=min_total_prob), axis=1
    )
    return out


def add_effr_expected_bps(
    df: pd.DataFrame,
    effr_futures_path: Path = EFFR_FUTURES_PATH,
    late_month_threshold: float = LATE_MONTH_THRESHOLD,
) -> pd.DataFrame:
    futures = pd.read_csv(effr_futures_path)
    futures["obs_date"] = pd.to_datetime(futures["Date_"], errors="coerce")
    futures["Settlement"] = pd.to_numeric(futures["Settlement"], errors="coerce")
    futures["implied_rate"] = 100.0 - futures["Settlement"]
    futures["last_trade"] = pd.to_datetime(futures["LastTrdDate"], errors="coerce")
    futures = futures.dropna(subset=["obs_date", "last_trade", "implied_rate"]).copy()
    futures["contr_year"] = futures["last_trade"].dt.year
    futures["contr_month"] = futures["last_trade"].dt.month
    futures["obs_str"] = futures["obs_date"].dt.strftime("%Y-%m-%d")

    futures = futures.sort_values("obs_date")
    futures = futures.drop_duplicates(subset=["obs_str", "contr_year", "contr_month"], keep="last")
    effr_lookup = {
        (r["obs_str"], int(r["contr_year"]), int(r["contr_month"])): float(r["implied_rate"])
        for _, r in futures.iterrows()
    }

    out = df.copy()
    out["obs_str"] = out["observed_day_pst"].dt.strftime("%Y-%m-%d")
    effr_rows: list[dict[str, float | str]] = []

    for meeting_str in sorted(out["decision_date"].dropna().unique()):
        meeting_dt = pd.Timestamp(meeting_str)
        m_day = meeting_dt.day
        m_year = meeting_dt.year
        m_month = meeting_dt.month
        days_in_month = int(meeting_dt.days_in_month)
        late_month = (m_day / days_in_month) > late_month_threshold

        if late_month:
            next_year, next_month = (m_year + 1, 1) if m_month == 12 else (m_year, m_month + 1)
            need_a = (m_year, m_month)
            need_b = (next_year, next_month)
        else:
            prev_year, prev_month = (m_year - 1, 12) if m_month == 1 else (m_year, m_month - 1)
            need_a = (prev_year, prev_month)
            need_b = (m_year, m_month)

        obs_strs = out.loc[out["decision_date"] == meeting_str, "obs_str"].sort_values().unique()
        for obs_str in obs_strs:
            rate_a = effr_lookup.get((obs_str, need_a[0], need_a[1]))
            rate_b = effr_lookup.get((obs_str, need_b[0], need_b[1]))
            if rate_a is None or rate_b is None:
                continue

            if late_month:
                pre_rate, post_rate = rate_a, rate_b
            else:
                pre_rate = rate_a
                month_avg = rate_b
                denom = days_in_month - m_day
                if denom <= 0:
                    continue
                post_rate = (month_avg * days_in_month - pre_rate * m_day) / denom

            effr_rows.append(
                {
                    "decision_date": meeting_str,
                    "obs_str": obs_str,
                    "effr_expected_bps": (post_rate - pre_rate) * 100.0,
                }
            )

    effr_df = pd.DataFrame(effr_rows)
    if "effr_expected_bps" in out.columns:
        out = out.drop(columns=["effr_expected_bps"])
    if not effr_df.empty:
        out = out.merge(effr_df, on=["decision_date", "obs_str"], how="left")
    else:
        out["effr_expected_bps"] = np.nan
    return out


def build_realized_change(
    df: pd.DataFrame,
    monthly_effr_path: Path = MONTHLY_EFFR_PATH,
    late_month_threshold: float = LATE_MONTH_THRESHOLD,
) -> dict[str, float]:
    monthly = pd.read_csv(monthly_effr_path)
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    monthly["EFFR"] = pd.to_numeric(monthly["EFFR"], errors="coerce")
    actual_rate = {
        (r["date"].year, r["date"].month): r["EFFR"] / 100.0
        for _, r in monthly.dropna(subset=["date", "EFFR"]).iterrows()
    }

    out: dict[str, float] = {}
    for meeting_str in df["decision_date"].dropna().unique():
        meeting = pd.Timestamp(meeting_str)
        day, year, month, days_in_month = meeting.day, meeting.year, meeting.month, int(meeting.days_in_month)
        late_month = (day / days_in_month) > late_month_threshold

        if late_month:
            next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
            rate_a, rate_b = actual_rate.get((year, month)), actual_rate.get((next_year, next_month))
            if rate_a is not None and rate_b is not None:
                out[meeting_str] = (rate_b - rate_a) * 10000.0
        else:
            prev_year, prev_month = (year - 1, 12) if month == 1 else (year, month - 1)
            rate_a, rate_b = actual_rate.get((prev_year, prev_month)), actual_rate.get((year, month))
            if rate_a is not None and rate_b is not None and (days_in_month - day) > 0:
                post = (rate_b * days_in_month - rate_a * day) / (days_in_month - day)
                out[meeting_str] = (post - rate_a) * 10000.0
    return out


def build_actual_decision(df: pd.DataFrame, fed_decisions_path: Path = FED_DECISIONS_PATH) -> dict[str, float]:
    fed = pd.read_csv(fed_decisions_path)
    fed["date"] = pd.to_datetime(fed["date"], errors="coerce")
    fed["rate_change_bps"] = pd.to_numeric(fed["rate_change_bps"], errors="coerce")

    out: dict[str, float] = {}
    today = pd.Timestamp.now().normalize()
    for meeting_str in df["decision_date"].dropna().unique():
        meeting = pd.Timestamp(meeting_str)
        if meeting > today:
            continue
        close = fed.loc[(fed["date"] - meeting).abs() <= pd.Timedelta(days=2)]
        out[meeting_str] = float(close.iloc[0]["rate_change_bps"]) if not close.empty else 0.0
    return out


def build_panel_with_targets() -> tuple[pd.DataFrame, dict[str, float], dict[str, float], list[str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_base_panel()
    panel = merge_kalshi_bid_ask(panel)
    panel = apply_forward_fill(panel)
    panel = add_effr_expected_bps(panel)

    realized_change = build_realized_change(panel)
    actual_decision = build_actual_decision(panel)
    effr_meetings = sorted(panel.loc[panel["effr_expected_bps"].notna(), "decision_date"].unique())
    return panel, realized_change, actual_decision, effr_meetings

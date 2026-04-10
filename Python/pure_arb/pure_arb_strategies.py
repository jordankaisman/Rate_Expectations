from __future__ import annotations

import numpy as np
import pandas as pd

from pure_arb_pipeline import POLY_MAP
from transaction_cost_models import (
    CME_BPV_USD_PER_BP,
    cme_round_trip_cost,
    get_cme_adv_contracts,
    polymarket_fed_fee,
    polymarket_spread_rule,
    prediction_market_one_way_cost,
)


FIXED_TC_BPS = 4.0
N_CME_DEFAULT = 1


def run_fixed_bps_strategy(
    df: pd.DataFrame,
    meetings: list[str],
    realized_change: dict[str, float],
    actual_decision: dict[str, float],
    tc_bps: float = FIXED_TC_BPS,
) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    for meeting in meetings:
        mdf = df[df["decision_date"] == meeting].sort_values("observed_day_pst")
        mdf = mdf[mdf["kalshi_expected_bps_ff"].notna() & mdf["effr_expected_bps"].notna()].copy()
        if mdf.empty or meeting not in realized_change:
            continue

        spread_settle = (
            actual_decision.get(meeting) - realized_change.get(meeting)
            if (meeting in actual_decision and meeting in realized_change)
            else np.nan
        )

        for _, row in mdf.iterrows():
            spread = float(row["kalshi_expected_bps_ff"] - row["effr_expected_bps"])
            if abs(spread) <= tc_bps:
                continue
            is_short = spread > 0
            pnl_theoretical = abs(spread) - tc_bps
            pnl_actual = (
                (spread - spread_settle - tc_bps) if is_short else (spread_settle - spread - tc_bps)
            ) if pd.notna(spread_settle) else np.nan

            trades.append(
                {
                    "meeting": meeting,
                    "entry_date": pd.Timestamp(row["observed_day_pst"]),
                    "exit_date": pd.Timestamp(meeting),
                    "spread_at_entry_bps": spread,
                    "direction": "Short Kalshi / Long CME" if is_short else "Long Kalshi / Short CME",
                    "pnl_theoretical_bps": pnl_theoretical,
                    "pnl_actual_bps": pnl_actual,
                    "spread_at_settle_bps": spread_settle,
                    "days_held": int((pd.Timestamp(meeting) - pd.Timestamp(row["observed_day_pst"])).days),
                }
            )

    return pd.DataFrame(trades).sort_values("entry_date") if trades else pd.DataFrame()


def build_weighted_pm_portfolio(
    row: pd.Series,
    side: int,
    n_cme: int,
    base_hike_bp: float = 25.0,
) -> dict[str, object]:
    base_shares = CME_BPV_USD_PER_BP * base_hike_bp * n_cme
    prices: dict[str, float] = {}
    shares: dict[str, float] = {}
    weights: dict[str, float] = {}

    for col, bps in POLY_MAP.items():
        if col not in row.index:
            continue
        p = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(p):
            w = bps / base_hike_bp
            q = side * base_shares * w
            prices[col] = float(p)
            shares[col] = float(q)
            weights[col] = float(w)

    return {
        "prices": prices,
        "shares": shares,
        "weights": weights,
        "total_abs_shares": float(sum(abs(q) for q in shares.values())),
    }


def nearest_outcome_col(actual_bps: float, cols: list[str]) -> str | None:
    if len(cols) == 0 or pd.isna(actual_bps):
        return None
    return min(cols, key=lambda c: abs(POLY_MAP[c] - actual_bps))


def run_dynamic_usd_strategy(
    df: pd.DataFrame,
    meetings: list[str],
    realized_change: dict[str, float],
    actual_decision: dict[str, float],
    vol_lookup: dict[tuple[str, int, int], float],
    monthly_adv_lookup: dict[tuple[int, int], float],
    n_cme: int = N_CME_DEFAULT,
) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    for meeting in meetings:
        mdf = df[df["decision_date"] == meeting].sort_values("observed_day_pst")
        mdf = mdf[mdf["poly_expected_bps"].notna() & mdf["effr_expected_bps"].notna()].copy()
        if mdf.empty:
            continue

        meeting_ts = pd.Timestamp(meeting)
        has_settlement = (meeting in actual_decision) and (meeting in realized_change)

        for _, row in mdf.iterrows():
            entry_date = pd.Timestamp(row["observed_day_pst"])
            spread_entry = float(row["poly_expected_bps"] - row["effr_expected_bps"])

            is_short_pm = spread_entry > 0
            pm_side = -1 if is_short_pm else 1
            cme_side = 1 if is_short_pm else -1
            direction = "Short PM / Long CME" if is_short_pm else "Long PM / Short CME"

            pm_port = build_weighted_pm_portfolio(row, side=pm_side, n_cme=n_cme)
            if pm_port["total_abs_shares"] <= 0:
                continue

            dtd = max((meeting_ts - entry_date).days, 0)
            pm_spread = polymarket_spread_rule(dtd)
            pm_avg_price = float(np.nanmean(list(pm_port["prices"].values()))) if pm_port["prices"] else 0.5
            pm_fee_one_way = polymarket_fed_fee(int(round(pm_port["total_abs_shares"])), pm_avg_price)
            pm_tc_round_trip = 2.0 * prediction_market_one_way_cost(
                num_contracts=float(pm_port["total_abs_shares"]),
                spread=pm_spread,
                fee=pm_fee_one_way,
            )

            cme_adv = get_cme_adv_contracts(entry_date, meeting_ts, vol_lookup, monthly_adv_lookup)
            cme_tc_round_trip = cme_round_trip_cost(n_cme, cme_adv, contract="ZQ")
            total_tc_usd = pm_tc_round_trip + cme_tc_round_trip

            gross_theoretical = abs(spread_entry) * CME_BPV_USD_PER_BP * n_cme
            if gross_theoretical <= total_tc_usd:
                continue
            net_theoretical = gross_theoretical - total_tc_usd

            pm_leg = np.nan
            cme_leg = np.nan
            gross_actual = np.nan
            net_actual = np.nan
            spread_settle = np.nan

            if has_settlement:
                actual_bps = float(actual_decision[meeting])
                cme_settle_bps = float(realized_change[meeting])
                spread_settle = actual_bps - cme_settle_bps

                winner_col = nearest_outcome_col(actual_bps, list(pm_port["shares"].keys()))
                pm_leg_val = 0.0
                for col, quantity in pm_port["shares"].items():
                    payoff = 1.0 if col == winner_col else 0.0
                    pm_leg_val += quantity * (payoff - pm_port["prices"][col])
                pm_leg = pm_leg_val

                cme_entry_bps = float(row["effr_expected_bps"])
                cme_leg = cme_side * (cme_settle_bps - cme_entry_bps) * CME_BPV_USD_PER_BP * n_cme
                gross_actual = pm_leg + cme_leg
                net_actual = gross_actual - total_tc_usd

            trades.append(
                {
                    "meeting": meeting,
                    "entry_date": entry_date,
                    "exit_date": meeting_ts,
                    "direction": direction,
                    "spread_entry_bps": spread_entry,
                    "spread_settle_bps": spread_settle,
                    "days_held": int((meeting_ts - entry_date).days),
                    "n_cme": n_cme,
                    "pm_total_abs_shares": pm_port["total_abs_shares"],
                    "pm_tc_round_trip_usd": pm_tc_round_trip,
                    "cme_tc_round_trip_usd": cme_tc_round_trip,
                    "total_tc_usd": total_tc_usd,
                    "gross_theoretical_usd": gross_theoretical,
                    "net_theoretical_usd": net_theoretical,
                    "pm_leg_pnl_usd": pm_leg,
                    "cme_leg_pnl_usd": cme_leg,
                    "gross_actual_usd": gross_actual,
                    "net_actual_usd": net_actual,
                    "pm_portfolio_weights": pm_port["weights"],
                }
            )

    return pd.DataFrame(trades).sort_values("entry_date") if trades else pd.DataFrame()

"""
report_plots.py
---------------
Generates publication-quality PDF plots for two FOMC meetings:
    2025-09-17  and  2025-07-30

Outputs saved to Data/Outputs/pure_arb/report_figures/:
    01_expected_rate_<meeting>.pdf   — Polymarket + Kalshi + CME expected rate
    02_spread_shaded_<meeting>.pdf   — Spread vs CME, shaded region
    03_tc_profiles_<meeting>.pdf     — Transaction cost profiles by venue
    04_spread_entries_<meeting>.pdf  — Spread + TC bands + trade entry arrows
    05_pnl_summary.pdf               — Net PnL bar chart (actual vs theoretical)

Run from Python/pure_arb/:
    python report_plots.py           # full pipeline + save cache + plots
    python report_plots.py --replot  # load cached data and replot only (fast)
    python report_plots.py --save-cache  # rebuild cache without plotting

The cache is saved to Data/Outputs/pure_arb/report_figures/report_cache.pkl
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── ensure local modules are importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import pure_arb_pipeline
import pure_arb_strategies
import transaction_cost_models
import transaction_cost_plots

importlib.reload(pure_arb_pipeline)
importlib.reload(transaction_cost_models)
importlib.reload(pure_arb_strategies)
importlib.reload(transaction_cost_plots)

from pure_arb_pipeline import OUT_DIR, build_panel_with_targets
from pure_arb_strategies import N_CME_DEFAULT, run_dynamic_usd_strategy, run_kalshi_dynamic_strategy
from transaction_cost_models import CME_BPV_USD_PER_BP, build_adv_lookup
from transaction_cost_plots import compute_tc_profiles

# ── config ────────────────────────────────────────────────────────────────────
MEETINGS   = ["2025-07-30", "2025-09-17"]
REPORT_DIR = OUT_DIR / "report_figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = REPORT_DIR / "report_cache.pkl"

DV01 = CME_BPV_USD_PER_BP * N_CME_DEFAULT  # $41.67

# matplotlib style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

USD_FMT = mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
BPS_FMT = mticker.FuncFormatter(lambda v, _: f"{v:+.1f} bps")

COLORS = {
    "poly":   "#CC0000",
    "kalshi": "#0055CC",
    "cme":    "#E07800",
    "settle": "#000000",
    "cme_settle": "#009900",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(REPORT_DIR.parent.parent.parent)}")


def _fmt_meeting(m: str) -> str:
    return pd.Timestamp(m).strftime("%B %Y")


# ── 1. Expected rate change (absolute lines) ──────────────────────────────────
def plot_expected_rate(mdf: pd.DataFrame, meeting: str,
                       realized_change: dict, actual_decision: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    p_mask    = mdf["poly_expected_bps"].notna()
    k_mask    = mdf["kalshi_expected_bps_ff"].notna()
    effr_mask = mdf["effr_expected_bps"].notna()

    if p_mask.any():
        ax.plot(mdf.loc[p_mask, "observed_day_pst"],
                mdf.loc[p_mask, "poly_expected_bps"] * DV01,
                color=COLORS["poly"], lw=1.6, label="Polymarket")
    if k_mask.any():
        ax.plot(mdf.loc[k_mask, "observed_day_pst"],
                mdf.loc[k_mask, "kalshi_expected_bps_ff"] * DV01,
                color=COLORS["kalshi"], lw=1.6, label="Kalshi")
    if effr_mask.any():
        ax.plot(mdf.loc[effr_mask, "observed_day_pst"],
                mdf.loc[effr_mask, "effr_expected_bps"] * DV01,
                color=COLORS["cme"], lw=2.0, label="CME Fed Funds")

    if meeting in realized_change:
        ax.axhline(realized_change[meeting] * DV01, color=COLORS["cme_settle"],
                   lw=1.2, ls="--", label=f"CME settle (${realized_change[meeting]*DV01:,.0f})")
    if meeting in actual_decision:
        ax.axhline(actual_decision[meeting] * DV01, color=COLORS["settle"],
                   lw=1.2, ls=":", label=f"Fed decision (${actual_decision[meeting]*DV01:,.0f})")

    ax.axhline(0, color="#AAAAAA", lw=0.8)
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Expected Rate Change — {_fmt_meeting(meeting)}")
    ax.set_ylabel("Expected Change ($, 1 ZQ contract)")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 2. Shaded spread ──────────────────────────────────────────────────────────
def plot_spread_shaded(mdf: pd.DataFrame, meeting: str,
                       realized_change: dict, actual_decision: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    effr_mask = mdf["effr_expected_bps"].notna()

    if effr_mask.any():
        ax.plot(mdf.loc[effr_mask, "observed_day_pst"],
                mdf.loc[effr_mask, "effr_expected_bps"] * DV01,
                color=COLORS["cme"], lw=2.2, label="CME", zorder=5)

    p_mask = mdf["poly_expected_bps"].notna() & effr_mask
    if p_mask.any():
        xd = mdf.loc[p_mask, "observed_day_pst"]
        pv = mdf.loc[p_mask, "poly_expected_bps"] * DV01
        pc = mdf.loc[p_mask, "effr_expected_bps"] * DV01
        ax.plot(xd, pv, color=COLORS["poly"], lw=1.4, label="Polymarket", zorder=4)
        ax.fill_between(xd, pc, pv, alpha=0.18, color=COLORS["poly"])

    k_mask = mdf["kalshi_expected_bps_ff"].notna() & effr_mask
    if k_mask.any():
        xd = mdf.loc[k_mask, "observed_day_pst"]
        kv = mdf.loc[k_mask, "kalshi_expected_bps_ff"] * DV01
        kc = mdf.loc[k_mask, "effr_expected_bps"] * DV01
        ax.plot(xd, kv, color=COLORS["kalshi"], lw=1.4, label="Kalshi", zorder=3)
        ax.fill_between(xd, kc, kv, alpha=0.12, color=COLORS["kalshi"])

    if meeting in realized_change:
        ax.plot(pd.Timestamp(meeting), realized_change[meeting] * DV01,
                marker="x", ms=10, mew=2.5, color=COLORS["cme_settle"],
                zorder=6, label="CME settle")
    if meeting in actual_decision:
        ax.plot(pd.Timestamp(meeting), actual_decision[meeting] * DV01,
                marker="o", ms=6, color=COLORS["settle"], zorder=6, label="Fed decision")

    ax.axhline(0, color="#AAAAAA", lw=0.8)
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Expected Rate Change vs CME — {_fmt_meeting(meeting)}")
    ax.set_ylabel("Expected Change ($, 1 ZQ contract)")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 3. TC profiles ────────────────────────────────────────────────────────────
def plot_tc_single(tcd: pd.DataFrame, meeting: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    dates = tcd["observed_day_pst"]

    if "cme_tc_usd" in tcd.columns:
        ax.plot(dates, tcd["cme_tc_usd"], color=COLORS["cme"], lw=1.8, label="CME ZQ")
    if "kalshi_tc_usd" in tcd.columns:
        km = tcd["kalshi_tc_usd"].notna()
        ax.plot(dates[km], tcd.loc[km, "kalshi_tc_usd"],
                color=COLORS["kalshi"], lw=1.8, label="Kalshi")
    if "poly_tc_usd" in tcd.columns:
        pm = tcd["poly_tc_usd"].notna()
        ax.plot(dates[pm], tcd.loc[pm, "poly_tc_usd"],
                color=COLORS["poly"], lw=1.8, ls="--", label="Polymarket")

    ax.yaxis.set_major_formatter(USD_FMT)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"One-Way Transaction Costs — {_fmt_meeting(meeting)}")
    ax.set_ylabel("TC per leg ($, 1 ZQ contract)")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 4. Spread + entries + TC bands ────────────────────────────────────────────
def plot_spread_entries(mdf: pd.DataFrame, meeting: str,
                        tc_data: dict,
                        trades_poly: pd.DataFrame,
                        trades_kalshi: pd.DataFrame,
                        realized_change: dict,
                        actual_decision: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    mdf = mdf.copy()
    mdf["poly_spread"]   = (mdf["poly_expected_bps"] - mdf["effr_expected_bps"]) * DV01
    mdf["kalshi_spread"] = (mdf["kalshi_expected_bps_ff"] - mdf["effr_expected_bps"]) * DV01

    if mdf["poly_spread"].notna().any():
        pm = mdf["poly_spread"].notna()
        ax.plot(mdf.loc[pm, "observed_day_pst"], mdf.loc[pm, "poly_spread"],
                color=COLORS["poly"], lw=1.4, label="Poly spread", zorder=4)
    if mdf["kalshi_spread"].notna().any():
        km = mdf["kalshi_spread"].notna()
        ax.plot(mdf.loc[km, "observed_day_pst"], mdf.loc[km, "kalshi_spread"],
                color=COLORS["kalshi"], lw=1.4, label="Kalshi spread", zorder=4)

    ax.axhline(0, color="#AAAAAA", lw=0.8)

    if meeting in tc_data:
        tcd   = tc_data[meeting]
        dates = tcd["observed_day_pst"]

        poly_tc   = (tcd["cme_tc_usd"] + tcd["poly_tc_usd"]).where(tcd["poly_tc_usd"].notna())
        kalshi_tc = (tcd["cme_tc_usd"] + tcd["kalshi_tc_usd"]).where(tcd["kalshi_tc_usd"].notna())

        pm = poly_tc.notna()
        if pm.any():
            ax.plot(dates[pm],  poly_tc[pm], color=COLORS["poly"], lw=1.0, ls="--", alpha=0.6, label="±Poly TC")
            ax.plot(dates[pm], -poly_tc[pm], color=COLORS["poly"], lw=1.0, ls="--", alpha=0.6)
            ax.fill_between(dates[pm], -poly_tc[pm], poly_tc[pm], color=COLORS["poly"], alpha=0.06, zorder=2)

        km = kalshi_tc.notna()
        if km.any():
            ax.plot(dates[km],  kalshi_tc[km], color=COLORS["kalshi"], lw=1.0, ls="--", alpha=0.6, label="±Kalshi TC")
            ax.plot(dates[km], -kalshi_tc[km], color=COLORS["kalshi"], lw=1.0, ls="--", alpha=0.6)

    def _arrow(ax, d, s, color):
        dy = -abs(s) * 0.38 if s > 0 else abs(s) * 0.38
        ax.annotate("", xy=(d, s + dy), xytext=(d, s),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=13),
                    zorder=7)

    for trades, color_short, color_long in [
        (trades_poly, COLORS["poly"], "#00AA00"),
        (trades_kalshi, COLORS["kalshi"], "#00AA00"),
    ]:
        if not trades.empty:
            for _, tr in trades[trades["meeting"] == meeting].iterrows():
                s = tr["spread_entry_bps"] * DV01
                _arrow(ax, tr["entry_date"], s, color_short if s > 0 else color_long)

    if meeting in actual_decision and meeting in realized_change:
        ss = (actual_decision[meeting] - realized_change[meeting]) * DV01
        ax.plot(pd.Timestamp(meeting), ss, marker="o", ms=7, color=COLORS["settle"],
                zorder=8, label=f"Settle ${ss:+.0f}")

    ax.yaxis.set_major_formatter(USD_FMT)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Spread vs CME with Trade Entries — {_fmt_meeting(meeting)}")
    ax.set_ylabel("Spread ($, 1 ZQ contract)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ── 5. PnL summary (both meetings) ───────────────────────────────────────────
def plot_pnl_summary(trades_poly: pd.DataFrame, trades_kalshi: pd.DataFrame,
                     meetings: list[str]) -> tuple[plt.Figure, str]:

    def _agg(trades_df, pnl_col, gross_col):
        if trades_df.empty:
            return {}
        t = trades_df.dropna(subset=[pnl_col]) if pnl_col in trades_df.columns else pd.DataFrame()
        if t.empty:
            return {}
        return {
            row["meeting"]: {
                "gross": row[gross_col],
                "net":   row[pnl_col],
                "n":     int(n),
            }
            for (_, row), n in zip(
                t.groupby("meeting")[[gross_col, pnl_col]].sum().reset_index().iterrows(),
                t.groupby("meeting").size().values,
            )
        }

    actual_poly   = _agg(trades_poly,   "net_actual_usd",  "gross_actual_usd")
    actual_kalshi = _agg(trades_kalshi, "net_actual_usd",  "gross_actual_usd")
    theo_poly     = _agg(trades_poly,   "net_pnl_usd",     "gross_pnl_usd")
    theo_kalshi   = _agg(trades_kalshi, "net_pnl_usd",     "gross_pnl_usd")

    labels = [_fmt_meeting(m) for m in meetings]
    x      = np.arange(len(meetings))
    w      = 0.18

    fig, ax = plt.subplots(figsize=(6, 6))

    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    series  = [
        (actual_poly,   COLORS["poly"],   "Poly actual",        {}),
        (theo_poly,     COLORS["poly"],   "Poly theoretical",   {"alpha": 0.45}),
        (actual_kalshi, COLORS["kalshi"], "Kalshi actual",      {"hatch": "//"}),
        (theo_kalshi,   COLORS["kalshi"], "Kalshi theoretical", {"hatch": "//", "alpha": 0.40}),
    ]

    detail_lines = []
    for xi, meeting in enumerate(meetings):
        for (data, color, label, kwargs), off in zip(series, offsets):
            v = data.get(meeting, {}).get("net", 0.0)
            bar_kw = dict(color=color, zorder=3, width=w, **kwargs)
            ax.bar(xi + off, v, **bar_kw, label=label if xi == 0 else None)

        # collect detail text
        for tag, data in [("Poly actual", actual_poly), ("Kalshi actual", actual_kalshi)]:
            d = data.get(meeting, {})
            if d:
                detail_lines.append(
                    f"  {_fmt_meeting(meeting)} | {tag}: "
                    f"gross ${d['gross']:+,.0f}  net ${d['net']:+,.0f}  ({d['n']} trades)"
                )

    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Net PnL ($)")
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.set_title("Net PnL by Meeting")
    ax.legend(fontsize=8)
    ax.set_aspect("auto")

    fig.tight_layout()

    detail_str = "Actual PnL detail (PM payoff $1/$0 + CME realized settle − TC):\n" + "\n".join(detail_lines)
    return fig, detail_str


# ── pipeline ──────────────────────────────────────────────────────────────────
def _run_pipeline(execution_lag: int = 1) -> dict:
    """Run the full data pipeline and return a cache dict."""
    print("Loading data…")
    df_ff, realized_change, actual_decision, all_meetings = build_panel_with_targets()

    df_ff           = df_ff[df_ff["decision_date"] >= "2025-01-01"].copy()
    realized_change = {k: v for k, v in realized_change.items() if k >= "2025-01-01"}
    actual_decision = {k: v for k, v in actual_decision.items() if k >= "2025-01-01"}
    effr_meetings   = [m for m in all_meetings if m >= "2025-01-01"]

    print("Building ADV lookup and running strategies…")
    vol_lookup, monthly_adv = build_adv_lookup()

    trades_poly = run_dynamic_usd_strategy(
        df=df_ff, meetings=effr_meetings,
        vol_lookup=vol_lookup, monthly_adv_lookup=monthly_adv,
        realized_change=realized_change, actual_decision=actual_decision,
        n_cme=N_CME_DEFAULT, execution_lag=execution_lag,
    )
    trades_kalshi = run_kalshi_dynamic_strategy(
        df=df_ff, meetings=effr_meetings,
        vol_lookup=vol_lookup, monthly_adv_lookup=monthly_adv,
        realized_change=realized_change, actual_decision=actual_decision,
        n_cme=N_CME_DEFAULT, execution_lag=execution_lag,
    )

    print("Computing TC profiles…")
    tc_data = compute_tc_profiles(df_ff, effr_meetings, vol_lookup, monthly_adv, n_cme=N_CME_DEFAULT)

    return {
        "df_ff":           df_ff,
        "realized_change": realized_change,
        "actual_decision": actual_decision,
        "trades_poly":     trades_poly,
        "trades_kalshi":   trades_kalshi,
        "tc_data":         tc_data,
    }


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"Cache saved → {CACHE_PATH.relative_to(REPORT_DIR.parent.parent.parent)}")


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No cache found at {CACHE_PATH}. Run without --replot first."
        )
    print(f"Loading cache from {CACHE_PATH.relative_to(REPORT_DIR.parent.parent.parent)} …")
    try:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Cache incompatible ({e.__class__.__name__}). Run without --replot to rebuild."
        ) from e


# ── plotting ──────────────────────────────────────────────────────────────────
def _generate_plots(cache: dict) -> None:
    df_ff           = cache["df_ff"]
    realized_change = cache["realized_change"]
    actual_decision = cache["actual_decision"]
    trades_poly     = cache["trades_poly"]
    trades_kalshi   = cache["trades_kalshi"]
    tc_data         = cache["tc_data"]

    print(f"\nSaving figures to {REPORT_DIR}\n")

    for meeting in MEETINGS:
        tag = meeting.replace("-", "")
        mdf = df_ff[df_ff["decision_date"] == meeting].sort_values("observed_day_pst").copy()
        if mdf.empty:
            print(f"  WARNING: no data for {meeting}")
            continue

        print(f"─── {_fmt_meeting(meeting)} ({meeting}) ───")

        fig = plot_expected_rate(mdf, meeting, realized_change, actual_decision)
        _save(fig, REPORT_DIR / f"01_expected_rate_{tag}.pdf")

        fig = plot_spread_shaded(mdf, meeting, realized_change, actual_decision)
        _save(fig, REPORT_DIR / f"02_spread_shaded_{tag}.pdf")

        if meeting in tc_data:
            fig = plot_tc_single(tc_data[meeting], meeting)
            _save(fig, REPORT_DIR / f"03_tc_profiles_{tag}.pdf")

        fig = plot_spread_entries(mdf, meeting, tc_data,
                                  trades_poly, trades_kalshi,
                                  realized_change, actual_decision)
        _save(fig, REPORT_DIR / f"04_spread_entries_{tag}.pdf")

    print("\n─── PnL summary (both meetings) ───")
    fig, detail = plot_pnl_summary(
        trades_poly[trades_poly["meeting"].isin(MEETINGS)],
        trades_kalshi[trades_kalshi["meeting"].isin(MEETINGS)] if not trades_kalshi.empty else trades_kalshi,
        MEETINGS,
    )
    _save(fig, REPORT_DIR / "05_pnl_summary.pdf")
    print(detail)

    print("\nDone.")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate report PDF plots.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--replot",     action="store_true",
                       help="Skip pipeline, load cached data and replot only (fast).")
    group.add_argument("--save-cache", action="store_true",
                       help="Run pipeline and save cache without generating plots.")
    args = parser.parse_args()

    if args.replot:
        cache = _load_cache()
    else:
        cache = _run_pipeline()
        _save_cache(cache)

    if not args.save_cache:
        _generate_plots(cache)


if __name__ == "__main__":
    main()

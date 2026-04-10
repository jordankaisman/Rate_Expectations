"""
pnl_analysis.py
---------------
Publication-quality PnL analysis chart for the CME × Polymarket pure-arb strategy.

Loads pre-computed strategy results from the report_plots cache (fast), or runs
the full pipeline if no cache exists.

Output:
    Data/Outputs/pure_arb/report_figures/pnl_analysis.pdf   — 2×2 multi-panel figure

Usage (from Python/pure_arb/):
    python pnl_analysis.py           # use cache if available, else run pipeline
    python pnl_analysis.py --fresh   # force full pipeline rerun
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

USD_FMT  = mticker.FuncFormatter(lambda v, _: f"${v:+,.0f}")
USD_FMT0 = mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")

WIN_COL  = "#009900"
LOSS_COL = "#CC0000"
THEO_COL = "#E8A0A0"
ZERO_COL = "#AAAAAA"
CME_TC_COL = "#888888"
PM_TC_COL  = "#CC4444"


def _pnl_color(v: float) -> str:
    return WIN_COL if v >= 0 else LOSS_COL


# ── data loading ──────────────────────────────────────────────────────────────
def _load_or_run(fresh: bool = False, execution_lag: int = 1) -> dict:
    from report_plots import CACHE_PATH, _run_pipeline, _save_cache

    if not fresh and CACHE_PATH.exists():
        print(f"Loading cache from {CACHE_PATH.name} …")
        try:
            cache = pickle.load(open(CACHE_PATH, "rb"))
            if cache.get("execution_lag") == execution_lag:
                return cache
            print(f"Cache has lag={cache.get('execution_lag')}, need lag={execution_lag}, rebuilding …")
        except Exception as e:
            print(f"Cache incompatible ({e.__class__.__name__}), rebuilding …")

    print("Running pipeline …")
    cache = _run_pipeline(execution_lag=execution_lag)
    cache["execution_lag"] = execution_lag
    _save_cache(cache)
    return cache


# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt_m(m: str) -> str:
    return pd.Timestamp(m).strftime("%b %Y")


def _meeting_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("meeting").agg(
        n_trades=("net_actual_usd", "count"),
        gross_actual=("gross_actual_usd", "sum"),
        net_actual=("net_actual_usd", "sum"),
        net_theo=("net_pnl_usd", "sum"),
        total_tc=("total_tc_usd", "sum"),
    ).reset_index()
    g["label"] = g["meeting"].apply(_fmt_m)
    return g


def _cumulative(trades: pd.DataFrame, col: str) -> pd.DataFrame:
    t = trades.dropna(subset=[col]).sort_values("entry_date").copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["cum"] = t[col].cumsum()
    return t


def _global_stats(trades: pd.DataFrame) -> dict:
    if trades.empty or "net_actual_usd" not in trades.columns:
        return {}
    t = trades.dropna(subset=["net_actual_usd"])
    n           = len(t)
    wins        = (t["net_actual_usd"] > 0).sum()
    total_gross = t["gross_actual_usd"].sum()
    total_net   = t["net_actual_usd"].sum()
    total_tc    = t["total_tc_usd"].sum()
    avg_days    = t["days_held"].mean() if "days_held" in t.columns else float("nan")
    avg_net     = t["net_actual_usd"].mean()
    std_net     = t["net_actual_usd"].std(ddof=1)
    sharpe      = (avg_net / std_net) if std_net > 0 else float("nan")
    return {
        "n": n, "wins": wins, "win_rate": wins / n,
        "total_gross": total_gross, "total_net": total_net,
        "total_tc": total_tc, "tc_pct_gross": total_tc / total_gross if total_gross else float("nan"),
        "avg_net": avg_net, "std_net": std_net, "sharpe_per_trade": sharpe,
        "avg_days": avg_days,
    }


# ── panel 1: cumulative PnL ───────────────────────────────────────────────────
def _plot_cumulative(ax: plt.Axes, trades_poly: pd.DataFrame) -> None:
    for col, color, label, ls in [
        ("net_actual_usd", WIN_COL,  "Actual",      "-"),
        ("net_pnl_usd",    THEO_COL, "Theoretical", "--"),
    ]:
        if trades_poly.empty:
            continue
        c = _cumulative(trades_poly, col)
        if c.empty:
            continue
        # colour the line green/red based on final value
        final = c["cum"].iloc[-1]
        ax.step(c["entry_date"], c["cum"], where="post",
                color=_pnl_color(final) if col == "net_actual_usd" else color,
                lw=2.0, linestyle=ls, label=label)

    ax.axhline(0, color=ZERO_COL, lw=0.8)
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Cumulative Net PnL Over Time")
    ax.set_ylabel("Cumulative Net PnL ($)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)


# ── panel 2: per-meeting bar ──────────────────────────────────────────────────
def _plot_per_meeting(ax: plt.Axes, trades_poly: pd.DataFrame,
                      all_meetings: list[str]) -> None:
    ms = _meeting_summary(trades_poly)

    base = pd.DataFrame({"meeting": all_meetings})
    base["label"] = base["meeting"].apply(_fmt_m)

    if not ms.empty and "net_actual" in ms.columns:
        pp = base.merge(ms[["meeting", "net_actual", "net_theo"]], on="meeting", how="left").fillna(0)
    else:
        pp = base.copy()
        pp["net_actual"] = 0.0
        pp["net_theo"]   = 0.0

    x, w = np.arange(len(base)), 0.30
    for xi, rp in enumerate(pp.itertuples()):
        act_col  = _pnl_color(rp.net_actual)
        theo_col = WIN_COL if rp.net_theo >= 0 else LOSS_COL
        ax.bar(xi - w*0.55, rp.net_actual, width=w, color=act_col,
               zorder=3, label="Actual"      if xi == 0 else None)
        ax.bar(xi + w*0.55, rp.net_theo,   width=w, color=theo_col,
               zorder=3, label="Theoretical" if xi == 0 else None, alpha=0.45)

    ax.axhline(0, color=ZERO_COL, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(base["label"], rotation=30, ha="right", fontsize=8)
    ax.set_title("Net PnL by FOMC Meeting")
    ax.set_ylabel("Net PnL ($)")
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


# ── panel 3: trade PnL distribution ──────────────────────────────────────────
def _plot_distribution(ax: plt.Axes, trades_poly: pd.DataFrame) -> None:
    if trades_poly.empty or "net_actual_usd" not in trades_poly.columns:
        return
    vals = trades_poly["net_actual_usd"].dropna()
    if vals.empty:
        return

    bins = np.linspace(vals.min() - 1, vals.max() + 1, 35)
    # colour each bin green/red
    counts, edges = np.histogram(vals, bins=bins)
    for i, (left, right, n) in enumerate(zip(edges[:-1], edges[1:], counts)):
        mid = (left + right) / 2
        ax.bar(mid, n, width=(right - left) * 0.9,
               color=_pnl_color(mid), alpha=0.75, edgecolor="white", lw=0.4)

    ax.axvline(0, color="#333333", lw=1.2, ls="--")
    ax.xaxis.set_major_formatter(USD_FMT0)
    ax.set_title("Distribution of Trade Net PnL")
    ax.set_xlabel("Net PnL per Trade ($)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)

    # add win/loss annotation
    n_win  = (vals > 0).sum()
    n_loss = (vals <= 0).sum()
    ax.text(0.97, 0.95, f"Win: {n_win}  Loss: {n_loss}  ({n_win/len(vals):.0%})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=WIN_COL if n_win > n_loss else LOSS_COL)


# ── panel 4: TC decomposition ─────────────────────────────────────────────────
def _plot_tc_decomp(ax: plt.Axes, trades_poly: pd.DataFrame) -> None:
    if trades_poly.empty:
        ax.set_visible(False)
        return

    gross = trades_poly["gross_actual_usd"].sum()
    net   = trades_poly["net_actual_usd"].sum()
    cme_tc = trades_poly["cme_tc_usd"].sum()
    pm_tc  = trades_poly["pm_tc_usd"].sum()
    total_tc = cme_tc + pm_tc

    x, w = np.array([0]), 0.45

    ax.bar(x, cme_tc,  width=w, color=CME_TC_COL, label="CME TC",  zorder=3, alpha=0.85)
    ax.bar(x, pm_tc,   width=w, color=PM_TC_COL,  label="PM TC",   zorder=3, alpha=0.80, bottom=cme_tc)

    # gross line
    ax.plot([x[0] - w*0.6, x[0] + w*0.6], [gross, gross],
            color="#000000", lw=2.5, zorder=5)
    ax.text(x[0], gross + max(gross * 0.025, 2),
            f"Gross  ${gross:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # net line — green or red
    net_color = _pnl_color(net)
    ax.plot([x[0] - w*0.6, x[0] + w*0.6], [net, net],
            color=net_color, lw=2.5, zorder=5, ls="--")
    # text above the net line (not below it)
    ax.text(x[0], net + max(abs(net) * 0.025, 2),
            f"Net  ${net:+,.0f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=net_color)

    ax.set_xticks([])
    ax.set_title("TC Decomposition vs Gross PnL\n(Polymarket)")
    ax.set_ylabel("Amount ($)")
    ax.yaxis.set_major_formatter(USD_FMT0)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    # TC % annotation inside bar
    ax.text(x[0], total_tc / 2,
            f"Total TC\n${total_tc:,.0f}\n({total_tc/gross:.0%} of gross)",
            ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")


# ── print stats ───────────────────────────────────────────────────────────────
def _print_stats(trades_poly: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("  Strategy Analytics Summary — Polymarket")
    print("═" * 60)
    s = _global_stats(trades_poly)
    if not s:
        print("No trades.")
        return
    print(f"  Trades          : {s['n']}  (wins: {s['wins']}, win rate: {s['win_rate']:.0%})")
    print(f"  Total gross PnL : ${s['total_gross']:+,.2f}")
    print(f"  Total TC paid   : ${s['total_tc']:,.2f}  ({s['tc_pct_gross']:.1%} of gross)")
    print(f"  Total net PnL   : ${s['total_net']:+,.2f}")
    print(f"  Avg net / trade : ${s['avg_net']:+,.2f}  (σ = ${s['std_net']:,.2f})")
    print(f"  Sharpe per trade: {s['sharpe_per_trade']:+.2f}")
    print(f"  Avg days held   : {s['avg_days']:.1f}")
    print("═" * 60)


# ── comparison plot (lag 0 vs lag 1) ─────────────────────────────────────────
def _plot_comparison(cache0: dict, cache1: dict) -> plt.Figure:
    tp0 = cache0["trades_poly"]
    tp1 = cache1["trades_poly"]
    all_meetings = sorted(cache0["df_ff"]["decision_date"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pure-Arb Strategy: Same-Day (t+0) vs t+1 Execution — CME × Polymarket",
                 fontsize=13, fontweight="bold", y=1.01)

    # top-left: cumulative PnL both lags
    ax = axes[0, 0]
    for trades, color, label, ls in [
        (tp0, WIN_COL,   "Actual (t+0)",       "-"),
        (tp0, THEO_COL,  "Theoretical (t+0)",  "--"),
        (tp1, "#0055CC", "Actual (t+1)",        "-"),
        (tp1, "#88AAEE", "Theoretical (t+1)",   "--"),
    ]:
        if trades.empty:
            continue
        col = "net_actual_usd" if "Actual" in label else "net_pnl_usd"
        c = _cumulative(trades, col)
        if c.empty:
            continue
        ax.step(c["entry_date"], c["cum"], where="post", color=color, lw=1.8, ls=ls, label=label)
    ax.axhline(0, color=ZERO_COL, lw=0.8)
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Cumulative Net PnL")
    ax.set_ylabel("Cumulative Net PnL ($)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    # top-right: per-meeting bars t+0 vs t+1
    ax = axes[0, 1]
    ms0 = _meeting_summary(tp0)
    ms1 = _meeting_summary(tp1)
    base = pd.DataFrame({"meeting": all_meetings})
    base["label"] = base["meeting"].apply(_fmt_m)

    def _safe_net(ms):
        if ms.empty or "net_actual" not in ms.columns:
            return {}
        return ms.set_index("meeting")["net_actual"].to_dict()

    net0 = _safe_net(ms0)
    net1 = _safe_net(ms1)
    x, w = np.arange(len(base)), 0.30
    for xi, row in base.iterrows():
        m = row["meeting"]
        v0, v1 = net0.get(m, 0.0), net1.get(m, 0.0)
        ax.bar(xi - w*0.55, v0, width=w, color=_pnl_color(v0), zorder=3,
               label="t+0" if xi == 0 else None)
        ax.bar(xi + w*0.55, v1, width=w, color=_pnl_color(v1), zorder=3,
               label="t+1" if xi == 0 else None, alpha=0.55)
    ax.axhline(0, color=ZERO_COL, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(base["label"], rotation=30, ha="right", fontsize=8)
    ax.set_title("Net PnL by Meeting")
    ax.set_ylabel("Net PnL ($)")
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    # bottom-left: distribution overlay
    ax = axes[1, 0]
    for trades, color, label, hatch in [
        (tp0, WIN_COL,   "t+0", ""),
        (tp1, "#0055CC", "t+1", "//"),
    ]:
        if trades.empty:
            continue
        vals = trades["net_actual_usd"].dropna()
        if vals.empty:
            continue
        bins = np.linspace(vals.min() - 1, vals.max() + 1, 30)
        ax.hist(vals, bins=bins, color=color, hatch=hatch, alpha=0.5,
                edgecolor="white", lw=0.4, label=label)
    ax.axvline(0, color="#333333", lw=1.2, ls="--")
    ax.xaxis.set_major_formatter(USD_FMT0)
    ax.set_title("Trade PnL Distribution")
    ax.set_xlabel("Net PnL per Trade ($)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    # bottom-right: summary stats table
    ax = axes[1, 1]
    ax.axis("off")
    s0 = _global_stats(tp0)
    s1 = _global_stats(tp1)

    def _fmt(s, key, fmt=None):
        if not s or key not in s:
            return "—"
        v = s[key]
        if fmt:
            return fmt.format(v)
        return str(v)

    rows = [
        ("Trades",             _fmt(s0,"n"), _fmt(s1,"n")),
        ("Win rate",           _fmt(s0,"win_rate","{:.0%}"), _fmt(s1,"win_rate","{:.0%}")),
        ("Total gross ($)",    _fmt(s0,"total_gross","${:+,.0f}"), _fmt(s1,"total_gross","${:+,.0f}")),
        ("Total TC ($)",       _fmt(s0,"total_tc","${:,.0f}"), _fmt(s1,"total_tc","${:,.0f}")),
        ("TC % of gross",      _fmt(s0,"tc_pct_gross","{:.0%}"), _fmt(s1,"tc_pct_gross","{:.0%}")),
        ("Total net ($)",      _fmt(s0,"total_net","${:+,.0f}"), _fmt(s1,"total_net","${:+,.0f}")),
        ("Avg net / trade ($)",_fmt(s0,"avg_net","${:+,.2f}"), _fmt(s1,"avg_net","${:+,.2f}")),
        ("Sharpe per trade",   _fmt(s0,"sharpe_per_trade","{:+.2f}"), _fmt(s1,"sharpe_per_trade","{:+.2f}")),
        ("Avg days held",      _fmt(s0,"avg_days","{:.1f}"), _fmt(s1,"avg_days","{:.1f}")),
    ]

    tbl = ax.table(
        cellText=[[r[1], r[2]] for r in rows],
        rowLabels=[r[0] for r in rows],
        colLabels=["t+0 (same-day)", "t+1"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DDDDDD")
        elif c >= 0:
            val = rows[r - 1][c + 1]
            if val.startswith("$+") or (val.startswith("+") and not val.startswith("+-")):
                cell.set_facecolor("#D6F5D6")
            elif val.startswith("$-") or val.startswith("-"):
                cell.set_facecolor("#FFD6D6")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("Summary Comparison", pad=14)

    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="Force full pipeline rerun (ignore cache)")
    parser.add_argument("--lag", type=int, default=0, choices=[0, 1],
                        help="Execution lag: 0=same-day, 1=t+1 (default: 0)")
    parser.add_argument("--compare", action="store_true",
                        help="Plot t+0 vs t+1 side-by-side comparison")
    args = parser.parse_args()

    from report_plots import REPORT_DIR

    if args.compare:
        cache0 = _load_or_run(fresh=args.fresh, execution_lag=0)
        cache1 = _load_or_run(fresh=False,       execution_lag=1)
        fig = _plot_comparison(cache0, cache1)
        out = REPORT_DIR / "pnl_analysis_compare.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved → {out.relative_to(out.parent.parent.parent.parent)}")
        print("\n--- t+0 ---")
        _print_stats(cache0["trades_poly"])
        print("\n--- t+1 ---")
        _print_stats(cache1["trades_poly"])
        return

    cache = _load_or_run(fresh=args.fresh, execution_lag=args.lag)
    trades_poly  = cache["trades_poly"]
    df_ff        = cache["df_ff"]
    all_meetings = sorted(df_ff["decision_date"].unique().tolist())

    lag_label = "same-day execution" if args.lag == 0 else "t+1 execution"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Pure-Arb Strategy: PnL Analysis — CME × Polymarket ({lag_label})",
                 fontsize=14, fontweight="bold", y=1.01)

    _plot_cumulative(axes[0, 0], trades_poly)
    _plot_per_meeting(axes[0, 1], trades_poly, all_meetings)
    _plot_distribution(axes[1, 0], trades_poly)
    _plot_tc_decomp(axes[1, 1], trades_poly)

    fig.tight_layout()

    out = REPORT_DIR / "pnl_analysis.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out.relative_to(out.parent.parent.parent.parent)}")

    _print_stats(trades_poly)


if __name__ == "__main__":
    main()

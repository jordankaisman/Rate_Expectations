import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Ensure repo root is on sys.path (nice for consistency when clicking Play)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Canonical labels -> bps move convention for "expected change"
# (Open-ended buckets mapped to their floor, consistent with SOFR pipeline capping.)
CANONICAL_TO_BPS = {
    "H0": 0.0,
    "H25": 25.0,
    "H25+": 25.0,
    "H50": 50.0,
    "H50+": 50.0,
    "H75": 75.0,
    "H75+": 75.0,
    "C25": -25.0,
    "C50": -50.0,
    "C50+": -50.0,
    "C75": -75.0,
    "C75+": -75.0,
}


def _expected_change_bps_from_row(row: pd.Series, venue_prefix: str) -> float | None:
    """
    Compute expected bps change for a venue from a wide row that contains columns like:
      kalshi_H0, kalshi_H25, ...
      polymarket_H0, polymarket_H25, ...

    Returns None if no usable columns are found.
    """
    total = 0.0
    weight_sum = 0.0

    for label, bps in CANONICAL_TO_BPS.items():
        col = f"{venue_prefix}_{label}"
        if col not in row.index:
            continue
        p = row[col]
        if pd.isna(p):
            continue
        p = float(p)
        total += p * bps
        weight_sum += p

    if weight_sum <= 0.0:
        return None

    # If venue prices don't sum to 1 (common), normalize.
    return total / weight_sum


@dataclass(frozen=True)
class Config:
    meeting_date: str = "2026-03-18"
    in_csv: str = "Data/Merged/Prediction_all_augmented.csv"
    out_png: str = "Data/Diagnostics/meeting_expected_change_timeseries.png"
    sofr_methods: tuple[str, ...] = ("trade_ois_forward_chain", "trade_sr1_chain", "effr_expected_bps")


CFG = Config()


def main() -> None:
    meeting_date = os.environ.get("MEETING_DATE", CFG.meeting_date)
    in_csv = Path(os.environ.get("PREDICTION_ALL_WITH_SOFR_CSV", CFG.in_csv))
    out_png = Path(os.environ.get("SOFR_VIZ_OUT", CFG.out_png))

    if not in_csv.is_absolute():
        in_csv = REPO_ROOT / in_csv
    if not out_png.is_absolute():
        out_png = REPO_ROOT / out_png

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)

    required = {"decision_date", "observed_day_pst", "jump_sr1", "jump_ois", "effr_expected_bps"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {in_csv}: {sorted(missing)}")

    # Filter to the meeting
    df = df[df["decision_date"].astype(str) == meeting_date].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for decision_date={meeting_date} in {in_csv}")

    # Parse/sort time axis
    df["observed_day_pst_dt"] = pd.to_datetime(df["observed_day_pst"], errors="coerce")
    df = df.dropna(subset=["observed_day_pst_dt"]).copy()
    df = df.sort_values(["observed_day_pst_dt"]).copy()

    # Convert jumps (decimal) -> bps
    df["jump_sr1_bps"] = pd.to_numeric(df["jump_sr1"], errors="coerce") * 10000.0
    df["jump_ois_bps"] = pd.to_numeric(df["jump_ois"], errors="coerce") * 10000.0
    df["jump_effr_bps"] = pd.to_numeric(df["effr_expected_bps"], errors="coerce")

    # Build SOFR wide time series: one column for each jump type (mean if multiple observations)
    sofr_wide = (
        df.groupby("observed_day_pst_dt", as_index=True)
        [["jump_ois_bps", "jump_sr1_bps", "jump_effr_bps"]]
        .mean()
        .sort_index()
    )

    # Compute venue expected change time series: de-duplicate by observed day
    venue_base = df.drop_duplicates(subset=["observed_day_pst_dt"]).set_index("observed_day_pst_dt")

    kalshi_series = venue_base.apply(lambda r: _expected_change_bps_from_row(r, "kalshi"), axis=1)
    polymarket_series = venue_base.apply(lambda r: _expected_change_bps_from_row(r, "polymarket"), axis=1)

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot SOFR methods (solid lines) using the jump columns produced by the pipeline
    method_to_col = {
        "trade_ois_forward_chain": "jump_ois_bps",
        "trade_sr1_chain": "jump_sr1_bps",
        "effr_expected_bps": "jump_effr_bps",
    }
    plotted_methods: list[str] = []
    for method in CFG.sofr_methods:
        col = method_to_col.get(method)
        if col and col in sofr_wide.columns:
            ax.plot(
                sofr_wide.index,
                sofr_wide[col],
                linewidth=2.2,
                label=f"SOFR {method}",
            )
            plotted_methods.append(method)

    # Plot venues (dashed)
    if kalshi_series.notna().any():
        ax.plot(
            kalshi_series.index,
            kalshi_series.values,
            linestyle="--",
            linewidth=2.2,
            label="Kalshi expected change (bps)",
        )
    if polymarket_series.notna().any():
        ax.plot(
            polymarket_series.index,
            polymarket_series.values,
            linestyle="--",
            linewidth=2.2,
            label="Polymarket expected change (bps)",
        )

    ax.set_title(f"Expected change (bps) over time for meeting {meeting_date}")
    ax.set_xlabel("Observed day (PST)")
    ax.set_ylabel("Expected change (bps)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_png}")
    print(
        f"Rows used: {len(df)} | Days plotted: {len(sofr_wide)} | "
        f"SOFR methods plotted: {plotted_methods} | "
        f"SOFR columns present: {list(sofr_wide.columns)} | "
        f"Kalshi points: {int(kalshi_series.notna().sum())} | "
        f"Polymarket points: {int(polymarket_series.notna().sum())}"
    )


if __name__ == "__main__":
    main()
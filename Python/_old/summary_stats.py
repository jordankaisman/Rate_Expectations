from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Python.stat_arb.Moment_PCA import PREDICTION_BPS_MAP, build_asset_panel, run_pca


DATA_DIR = REPO_ROOT / "Data"
PRESENTATION_DIR = REPO_ROOT / "Presentation"
FIGURES_DIR = PRESENTATION_DIR / "figures"
SUMMARY_PATH = FIGURES_DIR / "summary_stats.json"


def _load_panel() -> pd.DataFrame:
    panel = pd.read_csv(DATA_DIR / "Merged" / "Prediction_all_with_sofr.csv")
    panel["decision_date"] = pd.to_datetime(panel["decision_date"], errors="coerce")
    panel["observed_day_pst"] = pd.to_datetime(panel["observed_day_pst"], errors="coerce")
    return panel


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _coverage_counts(asset_panel: pd.DataFrame) -> pd.DataFrame:
    return (
        asset_panel.groupby("decision_date")
        .agg(
            total_days=("observed_day_pst", "size"),
            polymarket_days=("polymarket_expected_bps", lambda s: int(s.notna().sum())),
            kalshi_days=("kalshi_expected_bps", lambda s: int(s.notna().sum())),
            sr1_days=("jump_sr1_bps", lambda s: int(s.notna().sum())),
            ois_days=("jump_ois_bps", lambda s: int(s.notna().sum())),
            effr_days=("effr_expected_bps", lambda s: int(s.notna().sum())),
        )
        .reset_index()
        .sort_values("decision_date")
    )


def _raw_source_summary() -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    raw_specs = {
        "polymarket": DATA_DIR / "Polymarket" / "Polymarket_rates.csv",
        "kalshi": DATA_DIR / "Kalshi" / "Kalshi_rates.csv",
        "effr_futures": DATA_DIR / "EFFR_Futures" / "effr_futures.csv",
        "sr1": DATA_DIR / "SOFR" / "SR1.csv",
    }
    for name, path in raw_specs.items():
        df = pd.read_csv(path, low_memory=False)
        info: dict[str, object] = {"rows": int(len(df)), "columns": int(len(df.columns))}
        if "decision_date" in df.columns:
            info["decision_dates"] = int(pd.to_datetime(df["decision_date"], errors="coerce").nunique())
        if "observed_day" in df.columns:
            dates = pd.to_datetime(df["observed_day"], errors="coerce")
            info["start_date"] = str(dates.min().date())
            info["end_date"] = str(dates.max().date())
        elif "Date_" in df.columns:
            dates = pd.to_datetime(df["Date_"], errors="coerce")
            info["start_date"] = str(dates.min().date())
            info["end_date"] = str(dates.max().date())
        elif "candle_ts" in df.columns:
            dates = pd.to_datetime(df["candle_ts"], unit="s", errors="coerce")
            info["start_date"] = str(dates.min().date())
            info["end_date"] = str(dates.max().date())
        summary[name] = info
    return summary


def _merged_summary(asset_panel: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, object]:
    series_cols = [
        "polymarket_expected_bps",
        "kalshi_expected_bps",
        "jump_sr1_bps",
        "jump_ois_bps",
        "effr_expected_bps",
        "polymarket_tail_weight_bps",
        "kalshi_tail_weight_bps",
    ]
    stats = {
        col: {
            "non_null_rows": int(asset_panel[col].notna().sum()),
            "coverage_pct": round(float(asset_panel[col].notna().mean() * 100.0), 1),
            "mean": round(float(pd.to_numeric(asset_panel[col], errors="coerce").mean()), 2),
            "std": round(float(pd.to_numeric(asset_panel[col], errors="coerce").std()), 2),
            "meetings": int(asset_panel.loc[asset_panel[col].notna(), "decision_date"].nunique()),
        }
        for col in series_cols
    }
    return {
        "merged_rows": int(len(asset_panel)),
        "meetings": int(asset_panel["decision_date"].nunique()),
        "observation_start": str(asset_panel["observed_day_pst"].min().date()),
        "observation_end": str(asset_panel["observed_day_pst"].max().date()),
        "decision_start": str(asset_panel["decision_date"].min().date()),
        "decision_end": str(asset_panel["decision_date"].max().date()),
        "series": stats,
        "coverage_by_meeting_tail": coverage.tail(5).assign(
            decision_date=lambda d: d["decision_date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }


def _choose_example_meeting(asset_panel: pd.DataFrame) -> str:
    cols = [
        "polymarket_expected_bps",
        "kalshi_expected_bps",
        "effr_expected_bps",
        "jump_sr1_bps",
        "jump_ois_bps",
    ]
    complete = asset_panel.dropna(subset=cols)
    counts = complete.groupby("decision_date").size().sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No meeting has complete cross-market coverage.")
    return counts.index[0].strftime("%Y-%m-%d")


def plot_prediction_market_coverage(coverage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(coverage))
    ax.barh(y - 0.18, coverage["polymarket_days"], height=0.35, label="Polymarket", color="#4e79a7")
    ax.barh(y + 0.18, coverage["kalshi_days"], height=0.35, label="Kalshi", color="#f28e2b")
    ax.set_yticks(y)
    ax.set_yticklabels(coverage["decision_date"].dt.strftime("%Y-%m-%d"))
    ax.invert_yaxis()
    ax.set_xlabel("Observed trading days in merged panel")
    ax.set_title("Prediction-market coverage by FOMC meeting")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_market_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rates_market_coverage(asset_panel: pd.DataFrame, raw_summary: dict[str, dict[str, object]]) -> None:
    merged_stats = {
        "SR1": float(asset_panel["jump_sr1_bps"].notna().mean() * 100.0),
        "OIS": float(asset_panel["jump_ois_bps"].notna().mean() * 100.0),
        "EFFR": float(asset_panel["effr_expected_bps"].notna().mean() * 100.0),
    }
    raw_rows = {
        "SR1": raw_summary["sr1"]["rows"],
        "OIS expectations": len(pd.read_csv(DATA_DIR / "SOFR" / "SOFR_expectations.csv", low_memory=False)),
        "EFFR futures": raw_summary["effr_futures"]["rows"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(list(merged_stats.keys()), list(merged_stats.values()), color=["#59a14f", "#9c755f", "#b07aa1"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Non-missing share in merged panel (%)")
    axes[0].set_title("Rates-side usable coverage")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(list(raw_rows.keys()), list(raw_rows.values()), color=["#59a14f", "#9c755f", "#b07aa1"])
    axes[1].set_ylabel("Rows in source files")
    axes[1].set_title("Raw file depth")
    axes[1].ticklabel_format(style="plain", axis="y")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rates_market_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_example(panel: pd.DataFrame, meeting: str) -> dict[str, object]:
    df = panel.copy()
    df["decision_date"] = df["decision_date"].dt.strftime("%Y-%m-%d")
    df = df[df["decision_date"] == meeting].copy()
    common = df[df.filter(regex="^(polymarket_|kalshi_)").notna().any(axis=1)]
    sample_row = common.sort_values("observed_day_pst").iloc[len(common) // 2]

    poly_cols = [c for c in PREDICTION_BPS_MAP if c.startswith("polymarket_") and c in df.columns]
    kalshi_cols = [c for c in PREDICTION_BPS_MAP if c.startswith("kalshi_") and c in df.columns]

    def _series(cols: list[str]) -> pd.Series:
        s = sample_row[cols].apply(pd.to_numeric, errors="coerce")
        return pd.Series(
            {PREDICTION_BPS_MAP[c]: float(s[c]) for c in cols if pd.notna(s[c])}
        ).sort_index()

    poly = _series(poly_cols)
    kalshi = _series(kalshi_cols)
    strikes = sorted(set(poly.index).union(kalshi.index))
    poly = poly.reindex(strikes, fill_value=0.0)
    kalshi = kalshi.reindex(strikes, fill_value=0.0)

    x = np.arange(len(strikes))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, poly.values, width=0.35, label="Polymarket", color="#4e79a7")
    ax.bar(x + 0.18, kalshi.values, width=0.35, label="Kalshi", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v):+d}" for v in strikes])
    ax.set_xlabel("Rate-change bucket (bps)")
    ax.set_ylabel("Quoted probability")
    ax.set_title(f"Discrete outcome distribution example ({meeting}, {sample_row['observed_day_pst'].date()})")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribution_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "meeting": meeting,
        "observation_date": str(sample_row["observed_day_pst"].date()),
    }


def plot_expectations_example(asset_panel: pd.DataFrame, meeting: str) -> None:
    cols = [
        "polymarket_expected_bps",
        "kalshi_expected_bps",
        "effr_expected_bps",
        "jump_sr1_bps",
        "jump_ois_bps",
    ]
    df = asset_panel[asset_panel["decision_date"] == pd.Timestamp(meeting)].copy()
    df = df.dropna(subset=cols)
    df["days_to_decision"] = (df["decision_date"] - df["observed_day_pst"]).dt.days
    df = df.sort_values("observed_day_pst")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    palette = {
        "polymarket_expected_bps": "#4e79a7",
        "kalshi_expected_bps": "#f28e2b",
        "jump_sr1_bps": "#59a14f",
        "jump_ois_bps": "#9c755f",
        "effr_expected_bps": "#b07aa1",
    }
    labels = {
        "polymarket_expected_bps": "Polymarket",
        "kalshi_expected_bps": "Kalshi",
        "jump_sr1_bps": "SR1",
        "jump_ois_bps": "OIS",
        "effr_expected_bps": "EFFR",
    }
    for col in cols:
        ax.plot(df["days_to_decision"], df[col], label=labels[col], color=palette[col], linewidth=2)
    ax.invert_xaxis()
    ax.set_xlabel("Days to FOMC decision")
    ax.set_ylabel("Expected change (bps)")
    ax.set_title(f"Cross-market convergence example ({meeting})")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "expectations_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_signal_correlation(asset_panel: pd.DataFrame) -> None:
    cols = [
        "polymarket_expected_bps",
        "kalshi_expected_bps",
        "effr_expected_bps",
        "jump_sr1_bps",
        "jump_ois_bps",
    ]
    corr = asset_panel[cols].corr()
    labels = ["Poly", "Kalshi", "EFFR", "SR1", "OIS"]

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0.65, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Expectation signal correlations")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "signal_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca_moments(asset_panel: pd.DataFrame) -> dict[str, object]:
    pca, _, _, r2, _, _ = run_pca(asset_panel, n_components=3)
    asset_cols = [c for c in asset_panel.columns if c not in {"decision_date", "observed_day_pst"}]
    components = pd.DataFrame(
        pca.components_,
        index=[f"PC{i + 1}" for i in range(len(pca.components_))],
        columns=asset_cols,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(["PC1", "PC2", "PC3"], pca.explained_variance_ratio_[:3] * 100.0, color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Moment PCA: variance explained")
    axes[0].grid(axis="y", alpha=0.2)

    heat = axes[1].imshow(components.iloc[:3].values, cmap="coolwarm", aspect="auto", vmin=-0.8, vmax=0.8)
    axes[1].set_xticks(range(len(asset_cols)))
    axes[1].set_xticklabels(
        ["EFFR", "SR1", "OIS", "Poly exp", "Kalshi exp", "Poly var", "Kalshi var"],
        rotation=35,
        ha="right",
    )
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(["PC1", "PC2", "PC3"])
    axes[1].set_title("Moment PCA: loadings")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_prediction_moments.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "pc1_explained_variance_pct": round(float(pca.explained_variance_ratio_[0] * 100.0), 1),
        "pc2_explained_variance_pct": round(float(pca.explained_variance_ratio_[1] * 100.0), 1),
        "pc3_explained_variance_pct": round(float(pca.explained_variance_ratio_[2] * 100.0), 1),
        "r2": {k: round(float(v), 3) for k, v in r2.to_dict().items()},
    }

def plot_pca_moments2(asset_panel: pd.DataFrame) -> dict[str, object]:
    pca, _, _, r2, _, _ = run_pca(asset_panel, n_components=3)

    asset_cols = [c for c in asset_panel.columns if c not in {"decision_date", "observed_day_pst"}]

    components = pd.DataFrame(
        pca.components_,
        index=[f"PC{i + 1}" for i in range(len(pca.components_))],
        columns=asset_cols,
    )

    explained = pca.explained_variance_ratio_[:3] * 100.0

    pretty_labels = [
        "EFFR", "SR1", "OIS", "E: Poly", "E: Kalshi", "Var: Poly", "Var: Kalshi"
    ]

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[4, 1])

    for i in range(3):

        load = components.iloc[i].values
        colors = ["#4e79a7" if v > 0 else "#e15759" for v in load]

        # Loadings bar chart
        ax = fig.add_subplot(gs[i, 0])
        ax.barh(pretty_labels, load, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"PC{i+1} Loadings")
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        # Explained variance bar
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.bar([0], [explained[i]], color="#59a14f")
        ax2.set_ylim(0, 100)
        ax2.set_xticks([])
        ax2.set_ylabel("%")
        ax2.set_title("Explained")

        ax2.text(
            0,
            explained[i],
            f"{explained[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Moment PCA: Loadings and Explained Variance", y=0.98)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "pca_prediction_moments.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "pc1_explained_variance_pct": round(float(explained[0]), 1),
        "pc2_explained_variance_pct": round(float(explained[1]), 1),
        "pc3_explained_variance_pct": round(float(explained[2]), 1),
        "r2": {k: round(float(v), 3) for k, v in r2.to_dict().items()},
    }

def plot_transaction_costs() -> dict[str, float]:
    labels = [
        "Polymarket\n300 contracts",
        "Kalshi\n200 contracts",
        "CME SOFR fut\n4 contracts",
        "SOFR OIS\n50mm 1Y",
        "SOFR OIS\n100mm 5Y",
    ]
    values = [0.345, 7.0, 16.96, 100.0, 900.0]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color=["#4e79a7", "#f28e2b", "#59a14f", "#9c755f", "#b07aa1"])
    ax.set_ylabel("Round-trip explicit cost (USD)")
    ax.set_title("Illustrative transaction-cost scale from notebook examples")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "transaction_costs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "polymarket_300_contracts": 0.345,
        "kalshi_200_contracts": 7.0,
        "cme_sofr_4_contracts": 16.96,
        "sofr_ois_50mm_1y": 100.0,
        "sofr_ois_100mm_5y": 900.0,
    }


def build_summary() -> dict[str, object]:
    _ensure_dirs()
    merged_panel = _load_panel()
    asset_panel = build_asset_panel(merged_panel, "prediction_moments2")
    coverage = _coverage_counts(asset_panel)
    raw_summary = _raw_source_summary()
    example_meeting = _choose_example_meeting(asset_panel)

    plot_prediction_market_coverage(coverage)
    plot_rates_market_coverage(asset_panel, raw_summary)
    distribution_example = plot_distribution_example(merged_panel, example_meeting)
    plot_expectations_example(asset_panel, example_meeting)
    plot_signal_correlation(asset_panel)
    pca_summary = plot_pca_moments2(asset_panel)
    cost_summary = plot_transaction_costs()

    summary = {
        "raw_sources": raw_summary,
        "merged_panel": _merged_summary(asset_panel, coverage),
        "example_meeting": example_meeting,
        "distribution_example": distribution_example,
        "pca": pca_summary,
        "transaction_cost_examples": cost_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    result = build_summary()
    print(json.dumps(result, indent=2))

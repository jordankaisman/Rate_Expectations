from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Ensure repo root is on sys.path (nice for consistency when clicking Play)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Python" / "pure_arb"))

from Python.stat_arb.Moment_PCA import PREDICTION_BPS_MAP, build_asset_panel, load_prediction_panel, run_pca

REPORT_PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

def pretty_label(name: str) -> str:
    # Special cases first
    if name == "effr_expected_bps":
        return "EFFR"
    if name == "jump_ois_bps":
        return "OIS"
    if name == "jump_sr1_bps":
        return "SR1"
    if name == "polymarket_tail_weight_bps":
        return "Polymarket Variance"
    if name == "kalshi_tail_weight_bps":
        return "Kalshi Variance"
    if name == "kalshi_expected_bps":
        return "Kalshi Expected"
    if name == "polymarket_expected_bps":
        return "Polymarket Expected"
    if name == "kalshi_expected":
        return "Kalshi Expected"
    if name == "polymarket_expected":
        return "Polymarket Expected"

    # General pattern: polymarket_H50_bps -> Polymarket H50
    if name.startswith("polymarket_"):
        core = name.replace("polymarket_", "").replace("_bps", "")
        return f"Polymarket {core.upper()}"

    if name.startswith("kalshi_"):
        core = name.replace("kalshi_", "").replace("_bps", "")
        return f"Kalshi {core.upper()}"

    # fallback (just clean underscores / suffix)
    return name.replace("_bps", "").replace("_", " ").title()

def plot_asset_projections(
    pca: PCA,
    asset_names: list[str],
    output_path: str | None = None
) -> None:
    n_factors = min(3, pca.components_.shape[0])  # match style (top 3 PCs)
    comps = pca.components_
    explained_ratio = getattr(pca, "explained_variance_ratio_", None)

    fig, axes = plt.subplots(
        n_factors, 2,
        figsize=(10, 3.5 * n_factors),
        gridspec_kw={"width_ratios": [4, 1]}
    )

    if n_factors == 1:
        axes = np.array([axes])

    for i in range(n_factors):
        ax_load = axes[i, 0]
        ax_exp = axes[i, 1]

        # --- Loadings ---
        pretty_names = [pretty_label(a) for a in asset_names]
        series = pd.Series(comps[i, :], index=pretty_names)        
        series = series.sort_values()

        colors = ["tab:red" if v < 0 else "tab:blue" for v in series.values]

        ax_load.barh(series.index, series.values, color=colors, alpha=0.85)
        ax_load.axvline(0, color="black", linewidth=1)
        ax_load.set_title(f"PC{i+1} Loadings")
        ax_load.tick_params(axis="y", labelsize=9)

        # --- Explained variance ---
        if explained_ratio is not None:
            val = explained_ratio[i] * 100
        else:
            val = 0

        ax_exp.bar([0], [val])
        ax_exp.set_ylim(0, 100)
        ax_exp.set_title("Explained")
        ax_exp.set_xticks([])
        ax_exp.set_ylabel("%")

        ax_exp.text(
            0, val + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11
        )

    fig.suptitle("PCA: Loadings and Explained Variance", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_asset_projections2(pca: PCA, asset_names: list[str], output_path: str | None = None) -> None:
    n_factors = min(6, pca.components_.shape[0])
    explained_var = getattr(pca, "explained_variance_", np.ones(n_factors))
    comps = pca.components_  # shape: (n_factors, n_assets)

    fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, 5))
    if n_factors == 1:
        axes = [axes]

    for i in range(n_factors):
        cov = comps[i, :] * explained_var[i]  # signed covariance between factor i and each asset
        pretty_names = [pretty_label(a) for a in asset_names]
        series = pd.Series(comps[i, :], index=pretty_names)        
        colors = ["tab:blue" if v >= 0 else "tab:orange" for v in series.values]
        axes[i].bar(series.index, series.values, color=colors, alpha=0.85)
        axes[i].set_title(f"Factor {i + 1} covariance with assets")
        axes[i].set_ylabel("Covariance")
        axes[i].tick_params(axis="x", rotation=90, labelsize=8)

    plt.tight_layout()
    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def run_pca_visuals(
    path: str,
    output_dir: str,
    panel_mode: str = "prediction_all",
    n_components: int = 3,
    pca_rolling_window_days: int | None = None,
) -> dict[str, pd.DataFrame]:
    plt.rcParams.update(REPORT_PLOT_STYLE)
    raw = load_prediction_panel(path)
    panel = build_asset_panel(raw, panel_mode)
    passthrough_cols = {"jump_sr1_portfolio_weights", "jump_ois_portfolio_weights"} | set(PREDICTION_BPS_MAP.keys())
    model_panel = panel.drop(columns=[c for c in passthrough_cols if c in panel.columns])
    pca, factors, _, _, _, assets = run_pca(
        model_panel,
        n_components=n_components,
        normalize=False,
        rolling_window_days=pca_rolling_window_days,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    explained = np.array(getattr(pca, "explained_variance_ratio_", []), dtype=float)
    n_show = min(3, len(explained))
    comp_labels = [f"PC{i+1}" for i in range(n_show)]

    fig, ax = plt.subplots(figsize=(8, 4))
    if n_show:
        ax.bar(comp_labels, explained[:n_show], color="#0055CC")
    ax.set_title("PCA Explained Variance (Top 3 Components)")
    ax.set_ylabel("Explained variance ratio")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out / "pca_explained_variance_top3.png", dpi=180)
    plt.close(fig)

    loadings = np.array(getattr(pca, "components_", []), dtype=float)[:n_show, :]
    plot_asset_projections(pca, assets, output_path=str(out / "pca_loadings_top3.png"))

    factors_plot = factors.copy()
    if "observed_day_pst" in model_panel.columns:
        factors_plot["observed_day_pst"] = pd.to_datetime(model_panel["observed_day_pst"], errors="coerce")
    else:
        factors_plot["observed_day_pst"] = pd.NaT
    keep = [c for c in ["F1", "F2", "F3"] if c in factors_plot.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    if keep:
        for c in keep:
            ax.plot(factors_plot["observed_day_pst"], factors_plot[c], label=c, linewidth=1.2)
        ax.legend(loc="best")
    ax.set_title("PCA Factor Scores (Top 3 Components)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "pca_factor_scores_top3.png", dpi=180)
    plt.close(fig)

    loadings_df = pd.DataFrame(loadings.T, index=assets, columns=comp_labels)
    explained_df = pd.DataFrame({"component": comp_labels, "explained_variance_ratio": explained[:n_show]})
    factors_plot.to_csv(out / "pca_factor_scores.csv", index=False)
    loadings_df.to_csv(out / "pca_loadings.csv")
    explained_df.to_csv(out / "pca_explained_variance.csv", index=False)
    return {"factor_scores": factors_plot, "loadings": loadings_df, "explained_variance": explained_df}


if __name__ == "__main__":
    run_pca_visuals(
        path="Data/Merged/Prediction_all_with_sofr.csv",
        output_dir="Data/Outputs/pca_visuals",
        panel_mode="prediction_moments2",
        n_components=3,
    )

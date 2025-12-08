"""
Validation Consolidation Module
===============================

Combines validation outputs from:
    - graphical_lasso
    - ledoit_wolf
    - dcc_garch

Outputs:
    - consolidated_metrics.csv
    - consolidated_plot.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "results/validation"
MODELS = ["graphical_lasso", "ledoit_wolf", "dcc_garch"]


def load_static_metrics(model):
    path = os.path.join(BASE_DIR, model, "static_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing static metrics for {model}: {path}")
    return pd.read_csv(path).assign(Model=model)


def consolidate_metrics():
    print("[INFO] Loading static metrics from each model...")

    frames = []
    for model in MODELS:
        df = load_static_metrics(model)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    out_csv = os.path.join(BASE_DIR, "consolidated_metrics.csv")
    merged.to_csv(out_csv, index=False)

    print(f"[INFO] Saved consolidated CSV → {out_csv}")
    return merged


def make_comparison_plot(metrics_df):
    print("[INFO] Creating consolidated comparison plot...")

    plt.figure(figsize=(14, 8))

    metrics = ["Frobenius", "Spectral", "KL", "TE"]
    colors = ["black", "darkred", "darkblue", "darkgreen"]

    for metric, col in zip(metrics, colors):
        plt.plot(
            metrics_df["Model"],
            metrics_df[metric],
            marker="o",
            linewidth=2,
            label=metric,
            color=col
        )

    plt.title("Model Comparison – Validation Metrics", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_plot = os.path.join(BASE_DIR, "consolidated_plot.png")
    plt.savefig(out_plot)
    plt.close()
    print(f"[INFO] Saved consolidated plot → {out_plot}")

def make_barplot(metrics_df):
    print("[INFO] Creating consolidated barplot...")

    metrics = ["Frobenius", "Spectral", "KL", "TE"]

    # Reshape metrics table for seaborn
    df_long = metrics_df.melt(id_vars="Model", value_vars=metrics,
                              var_name="Metric", value_name="Value")

    plt.figure(figsize=(14, 8))
    import seaborn as sns

    sns.barplot(
        data=df_long,
        x="Metric",
        y="Value",
        hue="Model",
        alpha=0.85
    )

    plt.title("Model Comparison – Barplot of Validation Metrics", fontsize=16)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, "consolidated_barplot.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Saved consolidated barplot → {out_path}")


# --------------------------------------------------------
# 10. Consolidation Pipeline
# --------------------------------------------------------


def run_consolidation():
    print("\n[INFO] ===== STARTING CONSOLIDATION =====\n")

    metrics_df = consolidate_metrics()

    # Line plot comparison
    make_comparison_plot(metrics_df)

    # Barplot comparison
    make_barplot(metrics_df)

    print("\n[INFO] ===== CONSOLIDATION COMPLETED SUCCESSFULLY =====\n")



if __name__ == "__main__":
    run_consolidation()

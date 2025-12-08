"""
05_dcc_garch_multigarch_lowmem.py

Academic-style DCC-GARCH training using the updated 'multigarch' package
with low-memory mode enabled.

Enhancements in this version:
-----------------------------
- The final covariance and correlation matrices are saved with tickers
  preserved as index/columns.
- The fitted DCC model (pickle file) now also stores the tickers so that
  validation and rolling evaluation stages always maintain consistent
  ordering.
- The main user-facing function is now named `dcc_garch_training()` to
  allow standardized usage from the main pipeline.


"""

from __future__ import annotations
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from multigarch import DCC

# Output paths
OUT_DIR = "results/training/dcc_garch"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)


# =====================================================================
# Utility functions
# =====================================================================
def load_and_pivot(path: str) -> Tuple[pd.DataFrame, list]:
    """Load long-format returns and pivot to wide format."""
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])

    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").sort_index()

    tickers = list(wide.columns)
    print(f"[INFO] Pivot completed: shape={wide.shape}, assets={len(tickers)}")
    print(f"[INFO] Example tickers: {tickers[:20]}")

    return wide, tickers


def impute_forward_backward(wide: pd.DataFrame) -> pd.DataFrame:
    """Forward/backward fill to obtain dense matrix for DCC estimation."""
    print("[INFO] Performing forward/backward imputation...")
    filled = wide.ffill().bfill()
    print(f"[INFO] Remaining missing values: {filled.isna().sum().sum()}")
    return filled


def save_matrix(mat: np.ndarray, labels: list, path: str):
    """Save a square matrix with tickers as index/columns."""
    df = pd.DataFrame(mat, index=labels, columns=labels)
    df.to_csv(path)
    print(f"[SAVE] Matrix saved to: {path}")


# =====================================================================
# Plotting helpers
# =====================================================================
def plot_heatmap(mat: np.ndarray, labels: list, outpath: str, title: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[PLOT] Heatmap saved: {outpath}")


def plot_eigen_spectrum(mat: np.ndarray, outpath: str, title: str):
    eigs = np.linalg.eigvalsh(mat)
    eigs_sorted = np.sort(eigs)[::-1]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(eigs_sorted) + 1), eigs_sorted, marker="o")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[PLOT] Eigen spectrum saved: {outpath}")


# =====================================================================
# MAIN TRAINING FUNCTION — renamed to dcc_garch_training()
# =====================================================================
def dcc_garch_training(
    train_csv: str = "data/train_returns.csv",
    outdir: str = OUT_DIR,
    plots_dir: str = PLOTS_DIR,
    horizon_forecast: int = 5
):
    """
    Train a DCC-GARCH(1,1) model in low-memory mode.
    Saves:
      - Final covariance matrix (CSV)
      - Final correlation matrix (CSV)
      - First-step forecast covariance (CSV)
      - Model pickle (INCLUDING tickers)
      - Plots: heatmaps, eigenvalue spectrum
      - Training diagnostics CSV

    Returns:
        dict of output paths.
    """
    # ------------------------------
    # 1. Load + Pivot + Impute
    # ------------------------------
    wide, tickers = load_and_pivot(train_csv)
    wide_filled = impute_forward_backward(wide)

    X = wide_filled.to_numpy()
    T, N = X.shape
    print(f"[INFO] DCC estimation with T={T}, N={N}")

    # ------------------------------
    # 2. Fit DCC-GARCH
    # ------------------------------
    print("[INFO] Initializing DCC-GARCH(1,1) with low_memory=True...")
    dcc = DCC(p=1, q=1, low_memory=True, n_jobs=-1)

    print("[INFO] Fitting DCC model...")
    fitted = dcc.fit(X)
    obj = fitted if fitted is not None else dcc

    # Store tickers INSIDE the model object
    obj.tickers = tickers  
    print("[INFO] Embedded tickers inside the model object.")

    # ------------------------------
    # 3. Extract final matrices
    # ------------------------------
    H_final = obj.H
    R_final = obj.R

    print(f"[INFO] Final covariance matrix shape: {H_final.shape}")
    print(f"[INFO] Final correlation matrix shape: {R_final.shape}")

    # ------------------------------
    # 4. Save matrices
    # ------------------------------
    cov_csv = os.path.join(outdir, "dcc_covariance_final.csv")
    corr_csv = os.path.join(outdir, "dcc_correlation_final.csv")
    save_matrix(H_final, tickers, cov_csv)
    save_matrix(R_final, tickers, corr_csv)

    # ------------------------------
    # 5. Save model pickle WITH TICKERS
    # ------------------------------
    model_pickle = os.path.join(outdir, "dcc_model_lowmem.pkl")
    with open(model_pickle, "wb") as f:
        pickle.dump(obj, f)

    print(f"[SAVE] Pickled DCC model with tickers → {model_pickle}")

    # ------------------------------
    # 6. Short forecast
    # ------------------------------
    print("[INFO] Running short-term forecast...")
    try:
        fc = obj.forecast(horizon=horizon_forecast)
        if isinstance(fc, dict) and "covariance" in fc:
            fc_cov = np.asarray(fc["covariance"])
        else:
            fc_cov = np.asarray(fc)

        if fc_cov.shape[0] > 0:
            fc1 = fc_cov[0]
            fc_path = os.path.join(outdir, "dcc_forecast_tplus1_cov.csv")
            save_matrix(fc1, tickers, fc_path)
            print("[SAVE] First-step forecast covariance saved.")

    except Exception as e:
        print("[WARN] Forecast failed:", e)

    # ------------------------------
    # 7. Plots
    # ------------------------------
    plot_heatmap(H_final, tickers,
                 os.path.join(plots_dir, "dcc_covariance_heatmap.png"),
                 "DCC-GARCH Final Covariance Matrix")

    plot_heatmap(R_final, tickers,
                 os.path.join(plots_dir, "dcc_correlation_heatmap.png"),
                 "DCC-GARCH Final Correlation Matrix")

    plot_eigen_spectrum(H_final,
                        os.path.join(plots_dir, "dcc_eigen_spectrum.png"),
                        "Eigenvalue Spectrum of DCC Covariance")

    # ------------------------------
    # 8. Diagnostics
    # ------------------------------
    cond_num = float(np.linalg.cond(H_final))
    trace_val = float(np.trace(H_final))
    mean_var = float(np.mean(np.diag(H_final)))

    summary_df = pd.DataFrame([
        {"metric": "n_assets", "value": N},
        {"metric": "n_observations", "value": T},
        {"metric": "condition_number", "value": cond_num},
        {"metric": "trace", "value": trace_val},
        {"metric": "mean_variance", "value": mean_var}
    ])

    summary_csv = os.path.join(outdir, "dcc_training_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SAVE] Training summary saved: {summary_csv}")

    # ------------------------------
    # Return paths
    # ------------------------------
    return {
        "model_pickle": model_pickle,
        "cov_csv": cov_csv,
        "corr_csv": corr_csv,
        "summary_csv": summary_csv,
        "plots_dir": plots_dir
    }


# =====================================================================
# Script entry point
# =====================================================================
if __name__ == "__main__":
    print("\n===============================================")
    print("   STARTING DCC-GARCH TRAINING (LOW MEMORY)    ")
    print("===============================================\n")

    results = dcc_garch_training()
    print("\n[DONE] Outputs:", results)

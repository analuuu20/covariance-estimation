"""
05_dcc_garch_multigarch_lowmem.py

Academic-style DCC-GARCH training using the updated 'multigarch' package
with low-memory mode enabled.

This script:
  - Loads long-format log-returns (data/train_returns.csv)
  - Pivots to wide format (T x N)
  - Performs simple forward/backward imputation
  - Fits a DCC-GARCH(1,1) model with low_memory=True
  - Saves model pickle, final covariance & correlation matrices (CSV)
  - Produces publication-quality plots (heatmaps, eigenvalue spectrum)
  - Runs a short 5-step forecast and saves first-step forecasted covariance
  - Prints intermediate progress messages and a final summary table

Notes (academic):
  - Using low_memory=True reduces memory footprint by storing only final
    matrices H and R (n x n) instead of full time-varying arrays (T x n x n).
  - We use forward/backward imputation to obtain a dense matrix required by
    multivariate estimation; this is a pragmatic choice to avoid excessive
    row-dropping and preserve degrees of freedom for assets with sparse misses.

Author: (Your Name)
Date: 2025
"""

from __future__ import annotations
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: multigarch must be the updated version installed in your virtualenv
# (pip install --upgrade git+https://github.com/mechanicpanic/multigarch.git)
from multigarch import DCC

# Output paths
OUT_DIR = "models"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)


# ---------------------------
# Utilities
# ---------------------------
def load_and_pivot(path: str) -> Tuple[pd.DataFrame, list]:
    """
    Load the long-format returns and pivot to a wide panel (Date x Ticker).
    Returns the DataFrame (wide) and the ordered list of tickers (columns).
    """
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    expected_cols = {"Date", "Ticker", "LogReturn"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns {expected_cols}. Found {list(df.columns)}")

    print("[INFO] Pivoting to wide format (Date x Ticker)...")
    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").sort_index()

    print(f"[INFO] Wide matrix shape = {wide.shape} (rows=dates, cols=assets)")
    tickers = list(wide.columns)
    print(f"[INFO] Number of assets: {len(tickers)}")
    # Print first 40 tickers for confirmation (avoid huge log)
    print(f"[INFO] Example tickers (first 40): {tickers[:40]}")

    return wide, tickers


def impute_forward_backward(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Simple forward/backward fill to impute missing values.
    This is a pragmatic approach commonly used prior to multivariate estimation
    when dropna would remove too much data.
    """
    print("[INFO] Performing forward/backward imputation (ffill then bfill)...")
    # use .ffill() / .bfill() to avoid pandas future warning
    filled = wide.ffill().bfill()
    missing_now = filled.isna().sum().sum()
    print(f"[INFO] Remaining missing values after imputation: {missing_now}")
    
    return filled


def save_matrix(mat: np.ndarray, labels: list, path: str):
    """Save square matrix to CSV with tickers as index/columns."""
    df = pd.DataFrame(mat, index=labels, columns=labels)
    df.to_csv(path)
    print(f"[SAVE] Matrix saved to: {path}")


# ---------------------------
# Plotting helpers
# ---------------------------
def plot_heatmap(mat: np.ndarray, labels: list, outpath: str, title: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[PLOT] Saved heatmap: {outpath}")


def plot_eigen_spectrum(mat: np.ndarray, outpath: str, title: str):
    eigs = np.linalg.eigvalsh(mat)
    eigs_sorted = np.sort(eigs)[::-1]
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(eigs_sorted) + 1), eigs_sorted, marker="o")
    plt.xlabel("Eigenvalue index (sorted)")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[PLOT] Saved eigenvalue spectrum: {outpath}")


# ---------------------------
# Main DCC training
# ---------------------------
def train_dcc_lowmem(train_csv: str = "data/train_returns.csv",
                     outdir: str = OUT_DIR,
                     plots_dir: str = PLOTS_DIR,
                     horizon_forecast: int = 5):
    """Train DCC-GARCH in low memory mode and save results + plots."""
    wide, tickers = load_and_pivot(train_csv)

    # Impute missing values for estimation
    wide_filled = impute_forward_backward(wide)
    print(f"[INFO] After imputation: wide shape = {wide_filled.shape}")

    # Convert to numpy (T x N) as expected by multigarch
    returns_np = wide_filled.to_numpy(dtype=float)
    T, N = returns_np.shape
    print(f"[INFO] Using T={T} observations and N={N} assets for DCC estimation.")
    print("[INFO] Caution: DCC estimation may still be computationally intensive for large N.")

    # -----------------------------------------------------------------
    # Fit DCC in low-memory mode: H and R will be (N x N) final matrices
    # -----------------------------------------------------------------
    print("[INFO] Initializing DCC(p=1, q=1) with low_memory=True ...")
    dcc = DCC(p=1, q=1, low_memory=True, n_jobs=-1)

    print("[INFO] Fitting DCC model (low-memory mode). This may take several minutes ...")
    dcc_fit = dcc.fit(returns_np)  # returns dcc object / fitted object depending on API
    print("[INFO] DCC fit completed.")

    # According to updated API: dcc.H and dcc.R are final (N x N)
    # Keep compatibility: if dcc_fit returned an object, prefer that
    obj = dcc_fit if dcc_fit is not None else dcc

    # Extract final covariance & correlation matrices (N x N)
    try:
        H_final = obj.H  # final covariance matrix (N x N)
        R_final = obj.R  # final correlation matrix (N x N)
    except Exception as e:
        raise RuntimeError("Could not extract H/R from fitted DCC object - check API/installation") from e

    print(f"[INFO] Final covariance matrix shape: {H_final.shape}")
    print(f"[INFO] Final correlation matrix shape: {R_final.shape}")

    # Save matrices as CSV
    cov_csv = os.path.join(outdir, "dcc_covariance_final.csv")
    corr_csv = os.path.join(outdir, "dcc_correlation_final.csv")
    save_matrix(H_final, tickers, cov_csv)
    save_matrix(R_final, tickers, corr_csv)

    # Save model pickle (store fitted object)
    model_pickle = os.path.join(outdir, "dcc_model_lowmem.pkl")
    with open(model_pickle, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] Pickled fitted DCC object to: {model_pickle}")

    # ---------------------------
    # Forecasting (demonstrate it still works)
    # ---------------------------
    print(f"[INFO] Running a short forecast (horizon={horizon_forecast}) ...")
    try:
        fc = obj.forecast(horizon=horizon_forecast)
        # fc expected shape: (horizon, N, N)
        if isinstance(fc, dict) and "covariance" in fc:
            fc_cov = np.asarray(fc["covariance"])
        else:
            fc_cov = np.asarray(fc)
        print(f"[INFO] Forecast returned shape: {fc_cov.shape}")
        # Save the t+1 forecasted covariance
        if fc_cov.shape[0] >= 1:
            fc1 = fc_cov[0]
            save_matrix(fc1, tickers, os.path.join(outdir, "dcc_forecast_tplus1_cov.csv"))
            print("[SAVE] Saved forecast t+1 covariance matrix.")
    except Exception as e:
        print("[WARN] Forecasting failed or returned unexpected format:", str(e))

    # ---------------------------
    # Plots (publication-quality)
    # ---------------------------
    print("[INFO] Generating publication-quality plots ...")
    plot_heatmap(H_final, tickers, os.path.join(plots_dir, "dcc_covariance_heatmap.png"),
                 title="Final DCC-GARCH Covariance (Σ_T)")
    plot_heatmap(R_final, tickers, os.path.join(plots_dir, "dcc_correlation_heatmap.png"),
                 title="Final DCC-GARCH Correlation (R_T)")

    plot_eigen_spectrum(H_final, os.path.join(plots_dir, "dcc_eigen_spectrum.png"),
                        title="Eigenvalue Spectrum of Final Covariance Σ_T")

    # Condition number and basic diagnostics
    cond_num = float(np.linalg.cond(H_final))
    trace_val = float(np.trace(H_final))
    diag_means = np.mean(np.diag(H_final))
    print("\n[DIAGNOSTICS]")
    print(f" - Condition number (Σ_T): {cond_num:.3e}")
    print(f" - Trace(Σ_T): {trace_val:.6e}")
    print(f" - Mean diagonal variance: {diag_means:.6e}")

    # Save a short summary CSV
    summary_df = pd.DataFrame([
        {"metric": "n_assets", "value": N},
        {"metric": "n_observations", "value": T},
        {"metric": "condition_number", "value": cond_num},
        {"metric": "trace", "value": trace_val},
        {"metric": "mean_diagonal_variance", "value": diag_means}
    ])
    summary_csv = os.path.join(outdir, "dcc_training_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SAVE] Training summary saved to: {summary_csv}")

    print("\n[INFO] DCC-GARCH low-memory training finished. Outputs saved under:", outdir)
    return {
        "model_pickle": model_pickle,
        "cov_csv": cov_csv,
        "corr_csv": corr_csv,
        "summary_csv": summary_csv,
        "plots_dir": plots_dir
    }


# ---------------------------
# Run as script
# ---------------------------
if __name__ == "__main__":
    print("\n=======================================================")
    print("STARTING DCC-GARCH (LOW-MEMORY) TRAINING PIPELINE")
    print("=======================================================\n")
    results = train_dcc_lowmem()
    print("\n[DONE] Results:", results)

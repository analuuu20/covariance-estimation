"""
05_dcc_garch_multigarch.py

Academic-style implementation of multivariate DCC-GARCH using the multigarch
library provided for the covariance estimation project.

This script:
    - Loads train_returns.csv
    - Converts to wide matrix (T × N)
    - Fits a DCC-GARCH(1,1) model using multigarch
    - Saves the trained model as a pickle
    - Saves summary statistics to CSV
    - Generates relevant diagnostic plots
    - Prints intermediate progress updates

Author: (Your Name)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multigarch import GARCH, CCC, DCC


# ============================================================================
# 1. LOAD DATASET
# ============================================================================
def load_dataset(path):
    print("[INFO] Loading dataset:", path)
    df = pd.read_csv(path)

    print("[INFO] Converting 'Date' column to datetime...")
    df["Date"] = pd.to_datetime(df["Date"])

    print("[INFO] Pivoting to wide asset-return matrix...")
    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").sort_index()

    print("[INFO] Wide matrix shape =", wide.shape)
    print("[INFO] Number of assets =", wide.shape[1])
    print("[INFO] Number of observations =", wide.shape[0])

    return wide


# ============================================================================
# 2. OPTIONAL IMPUTATION (simple forward fill)
# ============================================================================
def simple_imputation(wide):
    print("[INFO] Performing simple forward/backward imputation...")
    filled = wide.fillna(method="ffill").fillna(method="bfill")
    missing_now = filled.isna().sum().sum()

    print(f"[INFO] Remaining missing values after imputation: {missing_now}")
    return filled


# ============================================================================
# 3. FIT DCC-GARCH MODEL
# ============================================================================
def fit_dcc_model(returns_matrix):
    """
    Multigarch requires a numpy array of shape (T, N).
    """
    print("[INFO] Initializing DCC-GARCH(p=1, q=1)...")
    model = DCC(p=1, q=1, n_jobs=-1)

    print("[INFO] Fitting DCC model... this may take several minutes.")
    dcc_fit = model.fit(returns_matrix)

    print("[INFO] DCC-GARCH model successfully fitted.")

    return dcc_fit


# ============================================================================
# 4. EXTRACT SUMMARY STATISTICS
# ============================================================================
def summarize_dcc(dcc_fit, asset_names):
    """
    Extracts and organizes DCC model results.
    """

    print("[INFO] Extracting DCC summary statistics...")

    # DCC outputs
    H = dcc_fit.H        # shape (T, N, N) covariance matrices
    R = dcc_fit.R        # shape (T, N, N) correlation matrices

    avg_corr = R.mean(axis=0)
    avg_var = np.mean(np.diagonal(H, axis1=1, axis2=2))

    summary = pd.DataFrame({
        "Metric": ["Average Variance", "Average Off-Diagonal Correlation"],
        "Value": [avg_var, avg_corr[np.triu_indices_from(avg_corr, k=1)].mean()]
    })

    print(summary)
    return summary, H, R


# ============================================================================
# 5. PLOTS
# ============================================================================
def generate_plots(H, R, asset_names, outdir):

    print("[INFO] Generating plots...")

    # ------------------ Plot 1: Average variance through time ----------------
    avg_var_t = np.mean(np.diagonal(H, axis1=1, axis2=2), axis=1)

    plt.figure(figsize=(10,4))
    plt.plot(avg_var_t)
    plt.title("Average Conditional Variance Through Time (DCC-GARCH)")
    plt.xlabel("Time")
    plt.ylabel("Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dcc_avg_variance.png"))
    plt.close()

    # ------------------ Plot 2: Average correlation through time -------------
    avg_corr_t = R[:, np.triu_indices(len(asset_names), k=1)[0],
                      np.triu_indices(len(asset_names), k=1)[1]].mean(axis=1)

    plt.figure(figsize=(10,4))
    plt.plot(avg_corr_t)
    plt.title("Average Conditional Correlation Through Time (DCC-GARCH)")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dcc_avg_correlation.png"))
    plt.close()

    print("[INFO] Plots saved.")


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================
def main():
    data_path = "data/train_returns.csv"
    outdir = "models"
    os.makedirs(outdir, exist_ok=True)

    print("\n[INFO] Starting DCC-GARCH training pipeline...\n")

    # Step 1: Load
    wide = load_dataset(data_path)

    # Step 2: Impute and convert to numpy
    wide_imputed = simple_imputation(wide)
    returns_np = wide_imputed.to_numpy()

    print(f"[INFO] Number of tickers entering DCC-GARCH: {wide_imputed.shape[1]}")

    # Step 3: Fit multivariate DCC model
    dcc_fit = fit_dcc_model(returns_np)

    # Step 4: Summaries
    summary, H, R = summarize_dcc(dcc_fit, wide_imputed.columns)

    summary.to_csv(os.path.join(outdir, "dcc_summary.csv"), index=False)

    # Step 5: Save model
    with open(os.path.join(outdir, "dcc_model.pkl"), "wb") as f:
        pickle.dump(dcc_fit, f)

    print("[INFO] Saved DCC model to models/dcc_model.pkl")

    # Step 6: Plots
    generate_plots(H, R, wide_imputed.columns, outdir)

    print("\n[INFO] DCC-GARCH pipeline completed successfully.\n")


if __name__ == "__main__":
    main()

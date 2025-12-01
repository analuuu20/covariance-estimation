"""
Ledoit-Wolf Shrinkage Covariance Estimation Module
==================================================

This script:
1. Loads long-format asset returns (train_returns.csv)
2. Converts to wide format (Date x Ticker)
3. Cleans + imputes missing data
4. Fits a Ledoit-Wolf shrinkage covariance estimator
5. Saves:
   - covariance matrix (CSV)
   - trained model (pickle)
   - diagnostic plots
6. Prints main statistics and returns results for main.py integration

Author: 2025
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.covariance import LedoitWolf


# =========================================================
# 1. Load dataset
# =========================================================
def load_data(path):
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    print("[INFO] Dataset loaded. Shape:", df.shape)
    return df


# =========================================================
# 2. Pivot to wide format
# =========================================================
def pivot_to_wide(df):
    print("[INFO] Pivoting to wide format...")
    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn")
    wide = wide.sort_index()
    print("[INFO] Wide matrix shape:", wide.shape)
    return wide


# =========================================================
# 3. Clean + Impute
# =========================================================
def clean_and_impute(wide):
    print("[INFO] Cleaning data and imputing missing values...")

    # Remove assets with too many missing values
    threshold = 0.8 * len(wide)
    wide = wide.dropna(axis=1, thresh=threshold)
    print(f"[INFO] After dropping sparse assets: {wide.shape[1]} assets remain.")

    # Forward + backward fill
    wide = wide.ffill().bfill()

    # Ensure no missing values remain
    assert not wide.isna().any().any(), "[ERROR] Missing values remain!"

    print("[INFO] Imputation complete.")
    return wide


# =========================================================
# 4. Fit Ledoit-Wolf Covariance Model
# =========================================================
def fit_ledoit_wolf(wide):
    print("[INFO] Fitting Ledoit-Wolf shrinkage estimator...")

    X = wide.values  # T x N matrix
    lw = LedoitWolf().fit(X)

    print("[INFO] Ledoit-Wolf fitting completed.")
    print("[INFO] Shrinkage coefficient:", lw.shrinkage_)

    cov = lw.covariance_

    print("[INFO] Covariance matrix shape:", cov.shape)
    return lw, cov


# =========================================================
# 5. Save results
# =========================================================
def save_covariance_matrix(cov, tickers, outdir="results/ledoit_wolf"):
    os.makedirs(outdir, exist_ok=True)

    df_cov = pd.DataFrame(cov, index=tickers, columns=tickers)
    path = os.path.join(outdir, "lw_covariance_matrix.csv")
    df_cov.to_csv(path)

    print("[INFO] Covariance matrix saved:", path)
    return df_cov


def save_model(model, outdir="results/ledoit_wolf"):
    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, "lw_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print("[INFO] Trained Ledoit-Wolf model saved:", path)


def save_plots(cov, tickers, outdir="results/ledoit_wolf"):
    os.makedirs(outdir, exist_ok=True)
    print("[INFO] Generating diagnostic plots...")

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cov, cmap="viridis", xticklabels=tickers, yticklabels=tickers)
    plt.title("Ledoit-Wolf Covariance Matrix (Heatmap)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lw_covariance_heatmap.png"))
    plt.close()

    # Eigenvalue distribution
    eigenvalues = np.linalg.eigvalsh(cov)
    plt.figure(figsize=(8, 4))
    plt.hist(eigenvalues, bins=40, edgecolor="black")
    plt.title("Spectrum of Covariance Matrix (Eigenvalues)")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lw_eigenvalues.png"))
    plt.close()

    print("[INFO] Plots saved.")


# =========================================================
# 6. Main pipeline handler
# =========================================================
def main():
    print("\n[INFO] Starting Ledoit-Wolf Shrinkage Covariance Pipeline...\n")

    df = load_data("data/train_returns.csv")
    wide = pivot_to_wide(df)
    wide = clean_and_impute(wide)

    tickers = wide.columns.tolist()
    print(f"[INFO] Number of assets used: {len(tickers)}")

    model, cov = fit_ledoit_wolf(wide)

    df_cov = save_covariance_matrix(cov, tickers)
    save_model(model)
    save_plots(cov, tickers)

    print("\n[INFO] Pipeline completed successfully.")
    print("[INFO] Shrinkage coefficient:", model.shrinkage_)
    print("[INFO] Covariance matrix summary:")
    print(df_cov.describe())

    return model, cov, tickers


# Allow integration into main.py
if __name__ == "__main__":
    main()

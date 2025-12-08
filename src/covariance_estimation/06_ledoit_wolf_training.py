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
   - trained model + tickers (pickle)
   - diagnostic plots
6. Returns model, covariance matrix and tickers


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

    threshold = 0.8 * len(wide)
    wide = wide.dropna(axis=1, thresh=threshold)
    print(f"[INFO] After dropping sparse assets: {wide.shape[1]} assets remain.")

    wide = wide.ffill().bfill()

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
def save_covariance_matrix(cov, tickers, outdir="results/training/ledoit_wolf"):
    os.makedirs(outdir, exist_ok=True)

    df_cov = pd.DataFrame(cov, index=tickers, columns=tickers)
    path = os.path.join(outdir, "lw_covariance_matrix.csv")
    df_cov.to_csv(path)

    print("[INFO] Covariance matrix saved:", path)
    return df_cov


def save_model_with_tickers(model, tickers, outdir="results/training/ledoit_wolf"):
    """
    Save model AND tickers together so they can be loaded in validation.
    """
    os.makedirs(outdir, exist_ok=True)
    payload = {"model": model, "tickers": tickers}

    path = os.path.join(outdir, "lw_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print("[INFO] Trained Ledoit-Wolf model + tickers saved:", path)


def save_plots(cov, tickers, outdir="results/training/ledoit_wolf"):
    os.makedirs(outdir, exist_ok=True)
    print("[INFO] Generating diagnostic plots...")

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cov, cmap="viridis", xticklabels=False, yticklabels=False)
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
# 6. PUBLIC TRAINING FUNCTION
# =========================================================
def ledoit_wolf_training():
    """
    Public function used by main.py and the validation pipeline.
    Returns:
        model  (sklearn LedoitWolf)
        cov    np.array N x N covariance matrix
        tickers list of asset tickers
    """
    print("\n[INFO] Starting Ledoit-Wolf Shrinkage Covariance Pipeline...\n")

    df = load_data("data/train_returns.csv")
    wide = pivot_to_wide(df)
    wide = clean_and_impute(wide)

    tickers = wide.columns.tolist()
    print(f"[INFO] Number of assets used: {len(tickers)}")

    model, cov = fit_ledoit_wolf(wide)

    df_cov = save_covariance_matrix(cov, tickers)
    save_model_with_tickers(model, tickers)
    save_plots(cov, tickers)

    print("\n[INFO] Pipeline completed successfully.")
    print("[INFO] Shrinkage coefficient:", model.shrinkage_)
    print("[INFO] Covariance matrix summary:")
    print(df_cov.describe())

    return model, cov, tickers


# =========================================================
# 7. Allow running standalone
# =========================================================
if __name__ == "__main__":
    ledoit_wolf_training()

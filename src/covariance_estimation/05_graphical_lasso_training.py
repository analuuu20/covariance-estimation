"""
GRAPHICAL LASSO COVARIANCE ESTIMATION MODULE:

This module fits a Graphical Lasso model to estimate a sparse inverse covariance matrix
from financial asset log-returns. The Graphical Lasso applies L1 regularization to
promote sparsity in the precision matrix, which is useful for high-dimensional
financial data where the number of assets may approach or exceed the number of observations.

The pipeline follows these steps:
1. Loads long-format asset returns (train_returns.csv)
2. Converts to wide format (Date x Ticker)
3. Cleans + imputes missing data
4. Fits a Graphical Lasso sparse inverse covariance estimator
5. It outputs:
   - covariance matrix (CSV)
   - precision (inverse covariance) matrix (CSV)
   - trained model + tickers (pickle)
   - diagnostic plots
6. Returns model, covariance, precision, tickers
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.covariance import GraphicalLassoCV


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
# 4. Fit Graphical Lasso Model
# =========================================================
def fit_graphical_lasso(wide):
    print("[INFO] Fitting Graphical Lasso with CV (automatic alpha selection)...")

    X = wide.values  # T x N matrix
    gl = GraphicalLassoCV(cv=5, n_jobs=-1).fit(X)

    print("[INFO] Graphical Lasso fitting completed.")
    print(f"[INFO] Selected alpha: {gl.alpha_}")

    cov = gl.covariance_
    precision = gl.precision_

    print("[INFO] Covariance matrix shape:", cov.shape)
    print("[INFO] Precision matrix shape:", precision.shape)

    return gl, cov, precision


# =========================================================
# 5. Save results
# =========================================================
def save_covariance_matrix(cov, tickers, outdir="results/training/graphical_lasso"):
    os.makedirs(outdir, exist_ok=True)

    df_cov = pd.DataFrame(cov, index=tickers, columns=tickers)
    path = os.path.join(outdir, "gl_covariance_matrix.csv")
    df_cov.to_csv(path)

    print("[INFO] Covariance matrix saved:", path)
    return df_cov


def save_precision_matrix(precision, tickers, outdir="results/training/graphical_lasso"):
    os.makedirs(outdir, exist_ok=True)

    df_prec = pd.DataFrame(precision, index=tickers, columns=tickers)
    path = os.path.join(outdir, "gl_precision_matrix.csv")
    df_prec.to_csv(path)

    print("[INFO] Precision matrix saved:", path)
    return df_prec


def save_model_with_tickers(model, tickers, outdir="results/training/graphical_lasso"):
    """
    Save Graphical Lasso model + tickers together for validation pipeline.
    """
    os.makedirs(outdir, exist_ok=True)

    payload = {"model": model, "tickers": tickers}

    path = os.path.join(outdir, "gl_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print("[INFO] Trained Graphical Lasso model + tickers saved:", path)


# =========================================================
# 6. Diagnostic Plots
# =========================================================
def save_plots(cov, precision, tickers, outdir="results/training/graphical_lasso"):
    os.makedirs(outdir, exist_ok=True)
    print("[INFO] Generating diagnostic plots...")

    # Heatmap - Covariance
    plt.figure(figsize=(12, 10))
    sns.heatmap(cov, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("Graphical Lasso Covariance Matrix (Heatmap)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gl_cov_heatmap.png"))
    plt.close()

    # Heatmap - Precision
    plt.figure(figsize=(12, 10))
    sns.heatmap(precision, cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
    plt.title("Graphical Lasso Precision Matrix (Heatmap)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gl_precision_heatmap.png"))
    plt.close()

    # Eigenvalue spectrum
    eigenvalues = np.linalg.eigvalsh(cov)
    plt.figure(figsize=(8, 4))
    plt.hist(eigenvalues, bins=40, edgecolor="black")
    plt.title("Spectrum of Covariance Matrix (Eigenvalues)")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gl_eigenvalues.png"))
    plt.close()

    print("[INFO] Plots saved.")


# =========================================================
# 7. FULL PIPELINE FOR GRAPHICAL LASSO TRAINING
# =========================================================
def graphical_lasso_training():
    """
    Function used by main.py and validation pipeline.
    Returns:
        model     GraphicalLassoCV fitted model
        cov       covariance matrix
        precision precision matrix
        tickers   list of tickers
    """
    print("\n[INFO] Starting Graphical Lasso Covariance Pipeline...\n")

    df = load_data("data/train_returns.csv")
    wide = pivot_to_wide(df)
    wide = clean_and_impute(wide)

    tickers = wide.columns.tolist()
    print(f"[INFO] Number of assets used: {len(tickers)}")

    model, cov, precision = fit_graphical_lasso(wide)

    df_cov = save_covariance_matrix(cov, tickers)
    df_prec = save_precision_matrix(precision, tickers)
    save_model_with_tickers(model, tickers)
    save_plots(cov, precision, tickers)

    print("\n[INFO] Pipeline completed successfully.")
    print(f"[INFO] Selected alpha (sparsity penalty): {model.alpha_}")
    print("[INFO] Covariance matrix summary:")
    print(df_cov.describe())

    return model, cov, precision, tickers


# =========================================================
# 8. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    graphical_lasso_training()

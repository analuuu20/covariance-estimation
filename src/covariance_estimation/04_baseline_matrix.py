"""
Baseline Sample Covariance Matrix Computation Module
====================================================

This script computes the baseline *sample covariance matrix* using
pairwise-complete observations from the training set of asset log-returns.

MAIN IMPROVEMENT IN THIS VERSION:
---------------------------------
The saved covariance matrix now includes row/column labels (tickers) 
embedded directly in the CSV file. This ensures full compatibility with 
later validation modules that require consistent asset ordering.

STRUCTURE:
----------
The code is organized into modular functions so it can be imported and
reused within the project's final pipeline (main.py).

OUTPUTS:
--------
- baseline_cov_matrix.csv 
- covariance_heatmap.png
- correlation_heatmap.png
- variance_distribution.png


"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------
def load_data(path):
    """
    Load the raw CSV containing daily asset log returns in long format.

    Parameters
    ----------
    path : str
        Path to train_returns.csv.

    Returns
    -------
    df : pd.DataFrame
        Raw long-format dataframe with columns:
        Date, Ticker, LogReturn, ...
    """
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print("[INFO] Dataset loaded. Shape:", df.shape)
    return df


# ---------------------------------------------------------------------
# 2. PIVOT TO WIDE FORMAT
# ---------------------------------------------------------------------
def pivot_returns(df):
    """
    Pivot the long-format dataset into wide format:

    - Rows   = trading days
    - Columns = tickers (assets)
    - Values  = log returns

    This transformation is required for covariance estimation.

    Returns
    -------
    pivot : pd.DataFrame
        Wide-format matrix (n_days × n_assets)
    """
    print("[INFO] Converting Date column to datetime...")
    df["Date"] = pd.to_datetime(df["Date"])

    print("[INFO] Pivoting dataset to wide format...")
    pivot = df.pivot_table(
        index="Date",
        columns="Ticker",
        values="LogReturn"
    )

    print("[INFO] Pivot completed.")
    print("[INFO] Wide shape (days × assets):", pivot.shape)
    print("[INFO] Number of assets:", pivot.shape[1])
    print("[INFO] Number of observations:", pivot.shape[0])

    return pivot


# ---------------------------------------------------------------------
# 3. COMPUTE SAMPLE COVARIANCE
# ---------------------------------------------------------------------
def compute_covariance(pivot):
    """
    Compute the sample covariance matrix using pairwise-complete observations.

    Parameters
    ----------
    pivot : pd.DataFrame
        Wide-format log returns matrix.

    Returns
    -------
    cov_matrix : pd.DataFrame
        Covariance matrix (n_assets × n_assets)
        Indexed and column-labeled by tickers.
    """
    print("[INFO] Computing sample covariance matrix...")
    cov_matrix = pivot.cov(min_periods=1)

    print("[INFO] Covariance matrix computed. Shape:", cov_matrix.shape)
    return cov_matrix


# ---------------------------------------------------------------------
# 4. SAVE RESULTS (WITH TICKERS INCLUDED)
# ---------------------------------------------------------------------
def save_results(cov_matrix, outdir="results/training/baseline"):
    """
    Save covariance matrix to CSV, preserving tickers in both rows and columns.

    This ensures the validation module can load the correct asset ordering.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
    outdir : str
    """
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "baseline_cov_matrix.csv")

    # Save with index + header → tickers fully preserved
    cov_matrix.to_csv(outfile, index=True)

    print(f"[INFO] Covariance matrix (with tickers) saved to: {outfile}")


# ---------------------------------------------------------------------
# 5. DIAGNOSTIC PLOTS
# ---------------------------------------------------------------------
def generate_plots(cov_matrix, pivot, outdir="results/training/baseline"):
    """
    Generate diagnostic plots:
    - covariance heatmap
    - correlation heatmap
    - variance distribution histogram
    """
    print("[INFO] Generating plots...")
    os.makedirs(outdir, exist_ok=True)

    # 1) Covariance heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(cov_matrix, cmap="viridis")
    plt.title("Sample Covariance Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "covariance_heatmap.png"), dpi=300)
    plt.close()

    # 2) Correlation heatmap
    corr = pivot.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # 3) Variance distribution
    variances = np.diag(cov_matrix.values)
    plt.figure(figsize=(8, 6))
    plt.hist(variances, bins=40, edgecolor="black")
    plt.title("Distribution of Asset Variances")
    plt.xlabel("Variance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "variance_distribution.png"), dpi=300)
    plt.close()

    print(f"[INFO] Plots saved to directory: {outdir}")


# ---------------------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------------------
def baseline_training():
    print("[INFO] Starting baseline covariance computation module...")

    df = load_data("data/train_returns.csv")
    pivot = pivot_returns(df)
    cov_matrix = compute_covariance(pivot)
    save_results(cov_matrix)
    generate_plots(cov_matrix, pivot)

    print("[INFO] Baseline covariance computation completed.")


if __name__ == "__main__":
    baseline_training()

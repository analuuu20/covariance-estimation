"""
BASELINE VALIDATION COVARIANCE MATRIX COMPUTATION MODULE:

This module computes the baseline sample covariance matrix for the
validation dataset, ensuring:

1. Same tickers used in training
2. Same column order as training
3. Same mild imputation procedure
4. Output saved with ticker/column labels for downstream compatibility

The pipeline follows these steps:
1. Load the validation log-returns dataset from:
    data/validation_returns.csv
2. Load the training baseline covariance matrix from:
    results/training/baseline/baseline_cov_matrix.csv
   to extract the list and order of tickers used during training.
3. Pivot the long-format validation dataset into wide format.
4. Align the validation dataset to the training tickers:
   - Retain only tickers present in training
   - Reorder columns to match training order
5. Apply the same mild imputation procedure as used in training:
   - Forward fill
   - Backward fill
   - Column mean imputation for any remaining NaNs
6. Compute the sample covariance matrix using pairwise-complete observations.
7. Save the resulting validation covariance matrix to:
    results/validation/baseline/baseline_cov_matrix_validation.csv

"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# 1. LOAD VALIDATION DATA
# ---------------------------------------------------------------------
def load_validation(path="data/validation_returns.csv"):
    print(f"[INFO] Loading validation dataset: {path}")
    df = pd.read_csv(path)
    print("[INFO] Validation dataset shape:", df.shape)
    return df


# ---------------------------------------------------------------------
# 2. LOAD TRAIN TICKERS (FOR REORDERING AND CHECKS)
# ---------------------------------------------------------------------
def load_train_tickers(train_cov_path="results/training/baseline/baseline_cov_matrix.csv"):
    print(f"[INFO] Loading training covariance matrix from: {train_cov_path}")
    cov = pd.read_csv(train_cov_path, index_col=0)
    tickers = list(cov.columns)
    print(f"[INFO] Number of training tickers: {len(tickers)}")
    return tickers


# ---------------------------------------------------------------------
# 3. PIVOT VALIDATION DATA
# ---------------------------------------------------------------------
def pivot_validation(df):
    print("[INFO] Pivoting validation dataset to wide format...")
    df["Date"] = pd.to_datetime(df["Date"])

    pivot = df.pivot_table(
        index="Date",
        columns="Ticker",
        values="LogReturn"
    )

    print("[INFO] Validation wide matrix shape:", pivot.shape)
    return pivot


# ---------------------------------------------------------------------
# 4. CHECK TICKER CONSISTENCY & ALIGN ORDER
# ---------------------------------------------------------------------
def align_to_train_tickers(pivot_valid, train_tickers):
    valid_tickers = set(pivot_valid.columns)

    missing = [t for t in train_tickers if t not in valid_tickers]
    extra = [t for t in valid_tickers if t not in train_tickers]

    print("[INFO] Checking ticker consistency with training set...")
    print(f"[INFO] Missing tickers in validation: {missing}")
    print(f"[INFO] Extra tickers in validation (ignored): {extra}")

    # keep only the tickers that exist in both
    pivot_valid = pivot_valid.reindex(columns=train_tickers)

    return pivot_valid


# ---------------------------------------------------------------------
# 5. MILD IMPUTATION (same as train)
# ---------------------------------------------------------------------
def mild_impute(pivot):
    print("[INFO] Applying mild imputation...")

    pivot = pivot.copy()

    pivot = pivot.ffill()
    pivot = pivot.bfill()

    col_means = pivot.mean()
    pivot = pivot.fillna(col_means)

    print("[INFO] Mild imputation complete.")
    return pivot


# ---------------------------------------------------------------------
# 6. COMPUTE SAMPLE COVARIANCE
# ---------------------------------------------------------------------
def compute_covariance(pivot):
    print("[INFO] Computing sample covariance matrix for validation...")
    cov = pivot.cov(min_periods=1)
    print("[INFO] Covariance shape:", cov.shape)
    return cov


# ---------------------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------------------
def save_validation_cov(cov, outdir="results/validation/baseline"):
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "baseline_cov_matrix_validation.csv")
    cov.to_csv(outfile, index=True)
    print(f"[INFO] Validation covariance matrix saved to: {outfile}")


# ---------------------------------------------------------------------
# 8. FULL PIPELINE FOR BASELINE VALIDATION
# ---------------------------------------------------------------------
def baseline_validation():
    print("==== BASELINE VALIDATION COVARIANCE MODULE START ====")

    df_valid = load_validation()
    train_tickers = load_train_tickers()
    pivot_valid = pivot_validation(df_valid)
    pivot_valid = align_to_train_tickers(pivot_valid, train_tickers)
    pivot_valid = mild_impute(pivot_valid)
    cov_valid = compute_covariance(pivot_valid)
    save_validation_cov(cov_valid)

    print("==== BASELINE VALIDATION COVARIANCE MODULE COMPLETED ====")


if __name__ == "__main__":
    baseline_validation()

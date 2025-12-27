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
    df = pd.read_csv(path, parse_dates=["Date"])
    print("[INFO] Validation dataset shape:", df.shape)
    return df


# ---------------------------------------------------------------------
# 2. LOAD TRAINING TICKERS
# ---------------------------------------------------------------------
def load_canonical_tickers():
    """
    Load canonical ticker universe derived during training.
    Priority:
      1) DCC-GARCH canonical_tickers_used.txt
      2) Graphical Lasso covariance CSV
      3) Ledoit-Wolf covariance CSV
    """

    print("[INFO] Loading canonical ticker universe from training artifacts...")

    dcc_path = "results/training/dcc_garch/canonical_tickers_used.txt"
    gl_path = "results/training/graphical_lasso/gl_covariance_matrix.csv"
    lw_path = "results/training/ledoit_wolf/lw_covariance_matrix.csv"

    if os.path.exists(dcc_path):
        tickers = pd.read_csv(dcc_path, header=None)[0].astype(str).tolist()
        print(f"[INFO] Canonical tickers loaded from DCC-GARCH: {len(tickers)}")
        return tickers

    if os.path.exists(gl_path):
        tickers = list(pd.read_csv(gl_path, index_col=0).index.astype(str))
        print(f"[INFO] Canonical tickers loaded from Graphical Lasso: {len(tickers)}")
        return tickers

    if os.path.exists(lw_path):
        tickers = list(pd.read_csv(lw_path, index_col=0).index.astype(str))
        print(f"[INFO] Canonical tickers loaded from Ledoit-Wolf: {len(tickers)}")
        return tickers

    raise FileNotFoundError(
        "[ERROR] No canonical ticker source found. "
        "Training must be run before baseline validation."
    )


# ---------------------------------------------------------------------
# 3. PIVOT VALIDATION DATA
# ---------------------------------------------------------------------
def pivot_validation(df):
    print("[INFO] Pivoting validation dataset to wide format...")
    pivot = df.pivot(index="Date", columns="Ticker", values="LogReturn")
    print("[INFO] Validation wide matrix shape:", pivot.shape)
    return pivot


# ---------------------------------------------------------------------
# 4. ALIGN TO CANONICAL TICKERS
# ---------------------------------------------------------------------
def align_to_canonical_tickers(pivot_valid, canonical_tickers):
    valid_tickers = set(pivot_valid.columns)

    missing = [t for t in canonical_tickers if t not in valid_tickers]
    extra = [t for t in valid_tickers if t not in canonical_tickers]

    print("[INFO] Checking ticker consistency with canonical universe...")
    print(f"[INFO] Missing tickers in validation: {len(missing)}")
    print(f"[INFO] Extra tickers in validation (ignored): {len(extra)}")

    if missing:
        raise ValueError(
            "[ERROR] Validation data is missing canonical tickers. "
            "Baseline validation aborted to preserve comparability."
        )

    # Reorder & restrict
    return pivot_valid.loc[:, canonical_tickers]


# ---------------------------------------------------------------------
# 5. MILD IMPUTATION (IDENTICAL TO TRAINING)
# ---------------------------------------------------------------------
def mild_impute(pivot):
    print("[INFO] Applying mild imputation...")
    pivot = pivot.ffill().bfill()

    remaining = pivot.isna().sum().sum()
    if remaining > 0:
        pivot = pivot.fillna(pivot.mean())

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
    cov.to_csv(outfile)
    print(f"[SAVE] Validation covariance matrix saved to: {outfile}")


# ---------------------------------------------------------------------
# 8. FULL PIPELINE
# ---------------------------------------------------------------------
def baseline_validation():
    print("\n==== BASELINE VALIDATION START ====\n")

    df_valid = load_validation()
    canonical_tickers = load_canonical_tickers()
    pivot_valid = pivot_validation(df_valid)
    pivot_valid = align_to_canonical_tickers(pivot_valid, canonical_tickers)
    pivot_valid = mild_impute(pivot_valid)
    cov_valid = compute_covariance(pivot_valid)
    save_validation_cov(cov_valid)

    print("\n==== BASELINE VALIDATION COMPLETED SUCCESSFULLY ====\n")


if __name__ == "__main__":
    baseline_validation()

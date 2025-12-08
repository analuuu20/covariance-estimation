"""
Model Validation Module
=======================

This module evaluates covariance estimation models using the validation dataset.

INPUTS
------
Training Outputs:
    Graphical Lasso
        - results/training/graphical_lasso/gl_covariance_matrix.csv
        - results/training/graphical_lasso/gl_precision_matrix.csv
        - results/training/graphical_lasso/gl_model.pkl

    Ledoit–Wolf
        - results/training/ledoit_wolf/lw_covariance_matrix.csv
        - results/training/ledoit_wolf/lw_model.pkl

    DCC-GARCH
        - results/training/dcc_garch/dcc_covariance_final.csv
        - results/training/dcc_garch/dcc_model_lowmem.pkl

Validation Input:
    >>> validation_csv = "data/validation_returns.csv"
    (NOT covariance. This must be the raw validation returns in long format.)

OUTPUTS
-------
Stored in: results/validation/<model_name>/

    - static_metrics.csv
    - rolling_window_results.csv
    - benchmark_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------
# 0. CONFIG
# --------------------------------------------------------
VALIDATION_CSV = "data/validation_returns.csv"

GL_COV = "results/training/graphical_lasso/gl_covariance_matrix.csv"
LW_COV = "results/training/ledoit_wolf/lw_covariance_matrix.csv"
DCC_COV = "results/training/dcc_garch/dcc_covariance_final.csv"

OUTDIR = "results/validation"


# --------------------------------------------------------
# 1. Load & Pivot Validation Dataset
# --------------------------------------------------------
def load_validation_returns(path):
    print("[INFO] Loading validation dataset (long format)...")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn")

    print("[INFO] Validation wide-format shape:", wide.shape)
    return wide


# --------------------------------------------------------
# 2. Mild Imputation (same as training)
# --------------------------------------------------------
def clean_validation_matrix(wide):
    print("[INFO] Applying mild forward/backward imputation...")

    wide = wide.ffill().bfill()

    if wide.isna().any().any():
        raise ValueError("Missing data remains after imputation.")

    return wide


# --------------------------------------------------------
# 3. Load Model Covariance Matrices
# --------------------------------------------------------
def load_covariance_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df = df.reindex(index=df.index, columns=df.index)
    return df


# --------------------------------------------------------
# 4. Align tickers between validation and training covariance
# --------------------------------------------------------
def align_tickers(val_wide, train_cov):
    print("[INFO] Aligning validation tickers with training set...")

    train_tickers = set(train_cov.index.tolist())
    val_tickers = set(val_wide.columns.tolist())

    # common tickers
    common = list(train_tickers.intersection(val_tickers))
    missing = train_tickers - val_tickers

    if missing:
        print(f"[WARN] Dropping missing training tickers not in validation: {missing}")

    # Align both matrices
    val_aligned = val_wide[common]
    train_aligned = train_cov.loc[common, common]

    return val_aligned, train_aligned


# --------------------------------------------------------
# 5. Compute baseline validation covariance
# --------------------------------------------------------
def compute_validation_cov(val_wide):
    print("[INFO] Computing sample covariance of validation set...")
    return val_wide.cov()


# --------------------------------------------------------
# 6. Metric Functions
# --------------------------------------------------------
def frobenius_distance(A, B):
    return np.linalg.norm(A - B, ord="fro")


def spectral_distance(A, B):
    return np.linalg.norm(A - B, ord=2)


def kl_divergence(A, B):
    invB = np.linalg.inv(B)
    k = A.shape[0]
    term1 = np.trace(invB @ A)
    term2 = np.log(np.linalg.det(B) / np.linalg.det(A))
    return 0.5 * (term1 - k + term2)


def portfolio_tracking_error(cov_train, cov_val):
    ones = np.ones(len(cov_train))
    w = np.linalg.inv(cov_train) @ ones
    w /= ones.T @ np.linalg.inv(cov_train) @ ones

    var_train = w.T @ cov_train @ w
    var_val = w.T @ cov_val @ w

    return np.abs(var_val - var_train)


# --------------------------------------------------------
# 7. Rolling window validation
# --------------------------------------------------------
def rolling_window_validation(val_wide, train_cov, window=60):
    print(f"[INFO] Rolling window validation (window = {window} days)...")

    results = []

    for end in range(window, len(val_wide)):
        win = val_wide.iloc[end-window:end]
        cov_win = win.cov()

        f = frobenius_distance(cov_win.values, train_cov.values)
        s = spectral_distance(cov_win.values, train_cov.values)
        k = kl_divergence(cov_win.values, train_cov.values)
        te = portfolio_tracking_error(train_cov.values, cov_win.values)

        results.append([val_wide.index[end], f, s, k, te])

    df = pd.DataFrame(results, columns=["Date", "Frobenius", "Spectral", "KL", "TE"])
    return df


# --------------------------------------------------------
# 8. Save plots
# --------------------------------------------------------
def save_benchmark_plot(model_name, static_metrics, rolling_df, outdir):
    plt.figure(figsize=(14, 7))

    plt.plot(rolling_df["Date"], rolling_df["Frobenius"], label="Rolling Frobenius")
    plt.axhline(static_metrics["Frobenius"], linestyle="--", label="Static Frobenius")

    plt.title(f"Benchmark Validation – {model_name}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, "benchmark_plot.png"))
    plt.close()


# --------------------------------------------------------
# 9. Validation Pipeline for each model
# --------------------------------------------------------
def validate_model(model_name, cov_path, val_cov, val_wide):
    print(f"\n[INFO] Validating model: {model_name}")

    model_cov = load_covariance_matrix(cov_path)

    # align tickers dynamically per model
    val_aligned, model_aligned = align_tickers(val_wide, model_cov)

    outdir = os.path.join(OUTDIR, model_name)
    os.makedirs(outdir, exist_ok=True)

    # STATIC METRICS
    static = {
        "Frobenius": frobenius_distance(val_cov.loc[val_aligned.columns, val_aligned.columns].values,
                                        model_aligned.values),
        "Spectral": spectral_distance(val_cov.loc[val_aligned.columns, val_aligned.columns].values,
                                      model_aligned.values),
        "KL": kl_divergence(val_cov.loc[val_aligned.columns, val_aligned.columns].values,
                            model_aligned.values),
        "TE": portfolio_tracking_error(model_aligned.values,
                                       val_cov.loc[val_aligned.columns, val_aligned.columns].values),
    }

    pd.DataFrame([static]).to_csv(os.path.join(outdir, "static_metrics.csv"), index=False)

    # ROLLING WINDOW
    rolling = rolling_window_validation(val_aligned, model_aligned)
    rolling.to_csv(os.path.join(outdir, "rolling_window_results.csv"), index=False)

    # PLOT
    save_benchmark_plot(model_name, static, rolling, outdir)


# --------------------------------------------------------
# 10. Main Execution
# --------------------------------------------------------
def run_validation():

    print("\n[INFO] ===== STARTING FULL MODEL VALIDATION =====\n")

    val_wide = load_validation_returns(VALIDATION_CSV)
    val_wide = clean_validation_matrix(val_wide)

    # Load GL tickers as master list
    gl_cov = load_covariance_matrix(GL_COV)
    MASTER_TICKERS = gl_cov.index.tolist()
    print(f"[INFO] Using {len(MASTER_TICKERS)} aligned tickers from training.")

    # Strict alignment for baseline cov
    missing = set(MASTER_TICKERS) - set(val_wide.columns)
    if missing:
        raise ValueError(f"[ERROR] Validation missing required tickers: {missing}")

    val_wide = val_wide[MASTER_TICKERS]

    # Baseline covariance
    val_cov = compute_validation_cov(val_wide)

    # Validate models
    validate_model("graphical_lasso", GL_COV, val_cov, val_wide)
    validate_model("ledoit_wolf", LW_COV, val_cov, val_wide)
    validate_model("dcc_garch", DCC_COV, val_cov, val_wide)

    print("\n[INFO] ===== VALIDATION COMPLETED SUCCESSFULLY =====\n")


if __name__ == "__main__":
    run_validation()

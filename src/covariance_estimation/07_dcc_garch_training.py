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

Note: this variant includes a canonical-universe filtering step that
loads the canonical list of tickers from previously produced training
artifacts (Graphical Lasso or Ledoit-Wolf covariance CSV). This ensures
all estimators are trained/evaluated on the same asset universe (491
tickers) and avoids spurious differences due to mismatched asset sets.
Using detectors derived from already-produced training outputs (not from
future data) reduces look-ahead bias in comparative validation.
"""
# Improved DCC-GARCH Training Script
# With automatic ticker filtering, alignment checks, detailed prints,
# academic English comments, and stronger preprocessing.

from __future__ import annotations
import os
import pickle
from typing import Tuple, List

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
    """Load long-format returns and pivot to wide format.
    Academic-style: ensures chronological ordering and removes duplicated
    index values if present.
    """
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])

    # Remove duplicated rows by Date+Ticker if they exist
    df = df.drop_duplicates(subset=["Date", "Ticker"])

    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").sort_index()

    tickers = list(wide.columns)
    print(f"[INFO] Pivot completed: shape={wide.shape}, assets={len(tickers)}")
    print(f"[INFO] Example tickers: {tickers[:20]}")

    return wide, tickers


def forward_backward_imputation(wide: pd.DataFrame) -> pd.DataFrame:
    """Perform forward/backward fill.
    Academic-style justification: DCC-GARCH requires a dense matrix because
    its likelihood function is undefined when values are missing.
    """
    print("[INFO] Performing forward/backward imputation...")
    filled = wide.ffill().bfill()
    print(f"[INFO] Remaining missing values after imputation: {filled.isna().sum().sum()}")
    return filled


def drop_empty_columns(wide: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    """Remove columns that are entirely missing.
    This produces consistent behaviour versus the validation stage.
    Returns:
      - cleaned DataFrame
      - list of kept tickers
      - list of dropped tickers
    """
    print("[INFO] Checking for empty (all-NaN) tickers...")

    nan_cols = [c for c in wide.columns if wide[c].isna().all()]
    kept_cols = [c for c in wide.columns if c not in nan_cols]

    if nan_cols:
        print(f"[WARN] Dropping {len(nan_cols)} empty tickers:")
        print(nan_cols)
    else:
        print("[INFO] No empty tickers detected.")

    cleaned = wide[kept_cols]
    return cleaned, kept_cols, nan_cols


def save_matrix(mat: np.ndarray, labels: list, path: str):
    """Save a square matrix with labels attached as index and columns."""
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
# MAIN TRAINING FUNCTION
# =====================================================================
def dcc_garch_training(
    train_csv: str = "data/train_returns.csv",
    outdir: str = OUT_DIR,
    plots_dir: str = PLOTS_DIR,
    horizon_forecast: int = 5,
):
    """
    Train a DCC-GARCH(1,1) model in low-memory mode.

    Includes improvements:
    - Automatic removal of fully-NaN tickers.
    - Full alignment reporting.
    - Academic-style detailed comments for methodological clarity.
    - Additional diagnostics.

    This variant will try to restrict the training asset universe to a
    canonical list (491 tickers) derived from previously produced training
    artifacts (Graphical Lasso CSV preferred, fallback to Ledoit-Wolf).
    This ensures consistency across estimators when comparing downstream
    validation/performance metrics.
    """

    print("\n=============================")
    print("  DCC-GARCH TRAINING START  ")
    print("=============================\n")

    # -------------------------------------------------------
    # (1) Load + Pivot
    # -------------------------------------------------------
    wide, tickers_initial = load_and_pivot(train_csv)

    # -------------------------------------------------------------------
    # Canonical ticker filtering: derive canonical universe (491 tickers)
    # from already-trained models to ensure cross-model comparability.
    #
    # Academic justification:
    # - Using the same asset universe across estimators avoids
    #   confounding differences due to different asset sets (a source
    #   of bias when comparing covariance estimates).
    # - We prioritise Graphical Lasso output (assumed present) then
    #   fall back to Ledoit-Wolf. If neither is present, proceed
    #   without filtering but warn the user.
    # -------------------------------------------------------------------

    print("[INFO] Attempting to derive canonical ticker universe from trained models...")

    gl_cov_path = os.path.join("results", "training", "graphical_lasso", "gl_covariance_matrix.csv")
    lw_cov_path = os.path.join("results", "training", "ledoit_wolf", "lw_covariance_matrix.csv")

    canonical_tickers: List[str] = None  # type: ignore
    source_path = None

    # 1) Try Graphical Lasso covariance CSV (preferred)
    if os.path.exists(gl_cov_path):
        try:
            _df = pd.read_csv(gl_cov_path, index_col=0)
            canonical_tickers = list(_df.index.astype(str))
            source_path = gl_cov_path
            print(f"[INFO] Canonical tickers loaded from Graphical Lasso CSV: {gl_cov_path}")
        except Exception as e:
            print(f"[WARN] Failed to read GL CSV ({gl_cov_path}): {e}")

    # 2) Fallback: Ledoit-Wolf covariance CSV
    if canonical_tickers is None and os.path.exists(lw_cov_path):
        try:
            _df = pd.read_csv(lw_cov_path, index_col=0)
            canonical_tickers = list(_df.index.astype(str))
            source_path = lw_cov_path
            print(f"[INFO] Canonical tickers loaded from Ledoit-Wolf CSV: {lw_cov_path}")
        except Exception as e:
            print(f"[WARN] Failed to read LW CSV ({lw_cov_path}): {e}")

    # 3) If we found a canonical list, apply it
    if canonical_tickers is not None:
        n_can = len(canonical_tickers)
        print(f"[INFO] Canonical universe size (from {source_path}): {n_can} tickers")

        # Check expected length (491) and warn if mismatch
        if n_can != 491:
            print(f"[WARN] Canonical ticker list length is {n_can}, expected 491. Proceeding but please verify.")

        # Compute intersection with available tickers in the training wide matrix
        available_in_data = [t for t in canonical_tickers if t in wide.columns]
        missing_from_data = [t for t in canonical_tickers if t not in wide.columns]

        print(f"[INFO] {len(available_in_data)} canonical tickers found in training dataset, "
              f"{len(missing_from_data)} missing.")

        if missing_from_data:
            # Print a short sample (but don't flood the logs)
            sample_missing = missing_from_data if len(missing_from_data) <= 50 else missing_from_data[:50]
            print("[WARN] Missing canonical tickers from training data (sample):", sample_missing)

        # If the intersection is too small, error out to avoid nonsense fits
        if len(available_in_data) < 5:
            raise RuntimeError("[ERROR] Too few canonical tickers present in training data after intersection.")

        # Reorder the wide matrix to the canonical ordering restricted to available tickers
        wide = wide.loc[:, available_in_data]
        print(f"[INFO] Training matrix reduced/reordered to canonical universe: shape={wide.shape}")

        # Persist the final canonical list used for reproducibility
        final_list_path = os.path.join(outdir, "canonical_tickers_used.txt")
        try:
            pd.Series(available_in_data).to_csv(final_list_path, index=False, header=False)
            print(f"[SAVE] Canonical ticker list saved: {final_list_path}")
        except Exception:
            print(f"[WARN] Could not save canonical ticker list to {final_list_path}")

    else:
        # No canonical list found — warn and continue (original behaviour)
        print("[WARN] No trained-model covariance CSV found to derive canonical tickers. "
              "Proceeding without canonical filtering.")

    # -------------------------------------------------------
    # (2) Drop tickers entirely missing (if any) and impute
    # -------------------------------------------------------
    wide_clean, tickers_kept, tickers_dropped = drop_empty_columns(wide)

    # -------------------------------------------------------
    # (3) Imputation
    # -------------------------------------------------------
    wide_filled = forward_backward_imputation(wide_clean)

    # Convert to numpy
    X = wide_filled.to_numpy()
    T, N = X.shape

    print(f"[INFO] Final training matrix T={T}, N={N}")
    print(f"[INFO] Dropped tickers count: {len(tickers_dropped)}")

    # -------------------------------------------------------
    # (4) Fit DCC-GARCH
    # -------------------------------------------------------
    print("[INFO] Initializing DCC-GARCH(1,1) with low_memory=True...")
    dcc = DCC(p=1, q=1, low_memory=True, n_jobs=-1)

    print("[INFO] Fitting DCC model... this may take several minutes...")
    fitted = dcc.fit(X)
    obj = fitted if fitted is not None else dcc

    # Store tickers inside model object
    obj.tickers = tickers_kept
    print("[INFO] Embedded cleaned ticker list inside model object.")

    # -------------------------------------------------------
    # (5) Extract final matrices
    # -------------------------------------------------------
    H_final = obj.H
    R_final = obj.R

    print(f"[INFO] Final covariance matrix shape: {H_final.shape}")
    print(f"[INFO] Final correlation matrix shape: {R_final.shape}")

    # -------------------------------------------------------
    # (6) Save matrices
    # -------------------------------------------------------
    cov_csv = os.path.join(outdir, "dcc_covariance_final.csv")
    corr_csv = os.path.join(outdir, "dcc_correlation_final.csv")

    save_matrix(H_final, tickers_kept, cov_csv)
    save_matrix(R_final, tickers_kept, corr_csv)

    # -------------------------------------------------------
    # (7) Save model
    # -------------------------------------------------------
    model_pickle = os.path.join(outdir, "dcc_model_lowmem.pkl")
    with open(model_pickle, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] Model pickle saved: {model_pickle}")

    # -------------------------------------------------------
    # (8) Forecast (first-step)
    # -------------------------------------------------------
    print("[INFO] Running short-term DCC forecast...")
    try:
        fc = obj.forecast(horizon=horizon_forecast)
        if isinstance(fc, dict) and "covariance" in fc:
            fc_cov = np.asarray(fc["covariance"])
        else:
            fc_cov = np.asarray(fc)

        if fc_cov.shape[0] > 0:
            fc1 = fc_cov[0]
            fc_path = os.path.join(outdir, "dcc_forecast_tplus1_cov.csv")
            save_matrix(fc1, tickers_kept, fc_path)
            print("[SAVE] First-step forecast covariance saved.")
    except Exception as e:
        print("[WARN] Forecast failed:", e)

    # -------------------------------------------------------
    # (9) Plots
    # -------------------------------------------------------
    plot_heatmap(H_final, tickers_kept, os.path.join(plots_dir, "dcc_covariance_heatmap.png"), "DCC-GARCH Final Covariance Matrix")
    plot_heatmap(R_final, tickers_kept, os.path.join(plots_dir, "dcc_correlation_heatmap.png"), "DCC-GARCH Final Correlation Matrix")
    plot_eigen_spectrum(H_final, os.path.join(plots_dir, "dcc_eigen_spectrum.png"), "Eigenvalue Spectrum of DCC Covariance")

    # -------------------------------------------------------
    # (10) Diagnostics
    # -------------------------------------------------------
    cond_num = float(np.linalg.cond(H_final))
    trace_val = float(np.trace(H_final))
    mean_var = float(np.mean(np.diag(H_final)))

    summary_df = pd.DataFrame([
        {"metric": "n_assets", "value": N},
        {"metric": "n_observations", "value": T},
        {"metric": "condition_number", "value": cond_num},
        {"metric": "trace", "value": trace_val},
        {"metric": "mean_variance", "value": mean_var},
        {"metric": "dropped_tickers", "value": len(tickers_dropped)}
    ])

    summary_csv = os.path.join(outdir, "dcc_training_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SAVE] Training summary saved: {summary_csv}")

    print("\n[INFO] DCC-GARCH training completed successfully.\n")

    return {
        "model_pickle": model_pickle,
        "cov_csv": cov_csv,
        "corr_csv": corr_csv,
        "summary_csv": summary_csv,
        "plots_dir": plots_dir,
        "tickers_dropped": tickers_dropped,
        "tickers_kept": tickers_kept,
    }


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    results = dcc_garch_training()
    print("\n[DONE] Outputs:", results)

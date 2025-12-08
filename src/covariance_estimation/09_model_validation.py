# 09_full_validation_consolidated.py
"""
Full validation + consolidation pipeline for covariance estimators (single script).

This script performs:
  - Load validation returns (long format) and pivot to wide matrix (Date x Ticker)
  - Mild imputation identical to training (ffill then bfill)
  - Baseline (realized) validation covariance from validation period
  - Per-model validation for:
      * Graphical Lasso (CSV covariance saved during training)
      * Ledoit-Wolf (CSV covariance saved during training)
      * DCC-GARCH (CSV covariance saved during training)
    For each model:
      - detect and report missing tickers
      - automatically align / reorder matrices by intersecting tickers
      - compute STATIC distances between predicted covariance and realized covariance:
            Frobenius norm: ||A - B||_F = sqrt(sum_ij (A_ij - B_ij)^2)
            Spectral norm (operator 2-norm): ||A - B||_2 = largest singular value of A-B
            KL divergence (multivariate Gaussian): KL(N(0,B) || N(0,A)) = 0.5*(tr(A^{-1} B) - n + ln(det A / det B))
            Portfolio tracking error (long-only GMVP TE)
      - rolling-window validation (window size configurable)
      - save per-model CSV reports and benchmark plots in results/validation/<model>/
  - Consolidate static metrics across models, compute normalized score and ranking
  - Save consolidated CSV and comparison plots in results/validation/
  - Print intermediate progress so user can follow pipeline

Notes (academic):
  - Frobenius norm measures overall matrix element-wise discrepancy.
  - Spectral norm measures the largest distortion in Euclidean operator sense.
  - KL divergence compares implied multivariate Gaussians — sensitive to eigenstructure and log-determinant differences.
  - Portfolio TE measures economic significance: how much a GMVP built on predicted covariance differs ex-post.
  - Consolidated "score" is a simple normalized aggregation (min-max normalization per metric, then average, lower=better).
"""

from __future__ import annotations
import os
import math
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)

# ---------------------------
# CONFIG
# ---------------------------
VALIDATION_CSV = "data/validation_returns.csv"   # raw validation returns (long)
# paths for model covariances created by training modules (these should exist)
GL_COV = "results/training/graphical_lasso/gl_covariance_matrix.csv"
LW_COV = "results/training/ledoit_wolf/lw_covariance_matrix.csv"
DCC_COV = "results/training/dcc_garch/dcc_covariance_final.csv"

OUTDIR = "results/validation"
PLOTS_DIR = os.path.join(OUTDIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Rolling validation params
ROLL_WINDOW = 60   # days used to compute realized rolling cov
ROLL_STEP = 1

# Numerical stability
EPS = 1e-10
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------
# Utilities (math + stability)
# ---------------------------
def nearest_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project symmetric A to nearest PSD by eigenvalue clipping (simple spectral clipping)."""
    B = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals_clipped = np.clip(eigvals, a_min=eps, a_max=None)
    B_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return (B_psd + B_psd.T) / 2.0


def safe_inv(mat: np.ndarray) -> np.ndarray:
    """Stable inverse via eigen-decomposition with clipping to avoid singular matrices."""
    mat = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals_clipped = np.clip(eigvals, a_min=EPS, a_max=None)
    inv = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T
    return (inv + inv.T) / 2.0


def frobenius_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm ||A - B||_F."""
    return float(np.linalg.norm(A - B, ord="fro"))


def spectral_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Spectral norm ||A - B||_2."""
    return float(np.linalg.norm(A - B, ord=2))


def kl_divergence_gaussian(A: np.ndarray, B: np.ndarray) -> float:
    """
    KL( N(0,B) || N(0,A) ) = 0.5 * ( tr(A^{-1} B) - n + ln(det A / det B) ).
    Lower is better. we compute with stability fixes.
    """
    # A is predicted, B is realized in our earlier notation; keep consistent: KL(true || pred)
    # Here we compute KL(B || A).
    A = nearest_psd(A)
    B = nearest_psd(B)
    n = A.shape[0]
    invA = safe_inv(A)
    tr_term = float(np.trace(invA @ B))

    # compute log determinants robustly
    sign_a, logdet_a = np.linalg.slogdet(A)
    sign_b, logdet_b = np.linalg.slogdet(B)
    if sign_a <= 0 or sign_b <= 0:
        # fallback using eigenvalues
        logdet_a = float(np.sum(np.log(np.clip(np.linalg.eigvalsh(A), EPS, None))))
        logdet_b = float(np.sum(np.log(np.clip(np.linalg.eigvalsh(B), EPS, None))))

    kl = 0.5 * (tr_term - n + (logdet_a - logdet_b))
    return float(kl)


def portfolio_tracking_error(cov_pred: np.ndarray, cov_real: np.ndarray) -> float:
    """
    Portfolio tracking error using long-only Global Minimum Variance Portfolio (GMVP).
    Solve w = argmin_w w' cov_pred w s.t. sum(w)=1, w >= 0 (we use closed form unconstrained then project to non-negative).
    For robustness we use unconstrained MV weights and if negative weights occur, fallback to equal weights.
    """
    cov_pred = nearest_psd(cov_pred)
    try:
        inv = safe_inv(cov_pred)
        ones = np.ones(cov_pred.shape[0])
        w = inv @ ones
        denom = float(ones.T @ inv @ ones)
        if denom <= 0 or not np.isfinite(denom):
            raise np.linalg.LinAlgError("Non-positive denom in GMV")
        w = w / denom
        # if negative weights found (shorts) -> fallback to equal weights (long-only approximate)
        if np.any(w < -1e-8):
            w = np.clip(w, 0, None)
            if w.sum() <= 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
    except Exception:
        w = np.ones(cov_pred.shape[0]) / cov_pred.shape[0]

    var_pred = float(w.T @ cov_pred @ w)
    var_real = float(w.T @ cov_real @ w)
    return float(abs(var_real - var_pred))


# ---------------------------
# I/O helpers
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_cov_csv(path: str) -> pd.DataFrame:
    """Load covariance CSV that includes tickers as index/columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Covariance file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    # ensure square and reindexed
    df = df.reindex(index=df.index, columns=df.index)
    return df


# ---------------------------
# Data prep
# ---------------------------
def load_and_prepare_validation(path: str) -> pd.DataFrame:
    """Load validation returns (long) and pivot to wide, then impute mild missing values (ffill, bfill)."""
    print(f"[INFO] Loading validation returns from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"[INFO] Raw validation shape (long): {df.shape}")
    required = {"Date", "Ticker", "LogReturn"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Validation CSV must include columns: {required}. Found: {list(df.columns)}")

    print("[INFO] Pivoting to wide-format (Date x Ticker)...")
    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").sort_index()
    print(f"[INFO] Wide-format shape: {wide.shape} (T x N)")

    print("[INFO] Applying mild imputation (ffill then bfill)...")
    wide = wide.ffill().bfill()
    missing = int(wide.isna().sum().sum())
    if missing > 0:
        raise ValueError(f"[ERROR] Missing values remain after imputation: {missing}")

    return wide


# ---------------------------
# Single-model validation
# ---------------------------
def validate_single_model(model_name: str,
                          model_cov_path: str,
                          val_wide: pd.DataFrame,
                          val_cov_full: pd.DataFrame,
                          roll_window: int = ROLL_WINDOW) -> Dict:
    """
    Validate a single trained-model covariance (CSV) against validation sample.
    Returns a dictionary with paths to saved outputs and summary metrics.
    """
    print(f"\n[INFO] ===== Validating model: {model_name} =====")
    print(f"[INFO] Loading model covariance from: {model_cov_path}")
    model_cov_df = load_cov_csv(model_cov_path)
    model_tickers = set(model_cov_df.index.tolist())
    val_tickers = set(val_wide.columns.tolist())

    print(f"[INFO] Model tickers: {len(model_tickers)}")
    print(f"[INFO] Validation tickers: {len(val_tickers)}")

    missing_in_validation = sorted(list(model_tickers - val_tickers))
    missing_in_model = sorted(list(val_tickers - model_tickers))

    if missing_in_validation:
        print("[WARN] Tickers present in MODEL but missing in VALIDATION:")
        print(missing_in_validation)
    if missing_in_model:
        print("[WARN] Tickers present in VALIDATION but missing in MODEL:")
        print(missing_in_model)

    # intersection (common tickers)
    common = sorted(list(model_tickers.intersection(val_tickers)))
    print(f"[INFO] Using {len(common)} intersecting tickers for metric computations.")

    if len(common) < 5:
        raise ValueError(f"[ERROR] Too few intersecting tickers ({len(common)}). Aborting validation for {model_name}.")

    # align & reorder matrices
    model_aligned = model_cov_df.loc[common, common].astype(float)
    val_aligned_wide = val_wide.loc[:, common].astype(float)
    val_cov_aligned = val_cov_full.loc[common, common].astype(float)

    # ensure PSD for numeric stability
    model_mat = nearest_psd(model_aligned.values)
    val_mat = nearest_psd(val_cov_aligned.values)

    # STATIC metrics
    print("[INFO] Computing static metrics (Frobenius, Spectral, KL, TE)...")
    fro = frobenius_norm(model_mat, val_mat)
    spec = spectral_norm(model_mat, val_mat)
    kl = kl_divergence_gaussian(model_mat, val_mat)
    te = portfolio_tracking_error(model_mat, val_mat)

    static_metrics = {
        "Model": model_name,
        "Tickers Used": len(common),
        "Missing in Validation": len(missing_in_validation),
        "Missing in Model": len(missing_in_model),
        "Frobenius": fro,
        "Spectral": spec,
        "KL": kl,
        "TE": te
    }

    # save static metrics
    outdir = os.path.join(OUTDIR, model_name)
    ensure_dir(outdir)
    static_csv = os.path.join(outdir, "static_metrics.csv")
    pd.DataFrame([static_metrics]).to_csv(static_csv, index=False)
    print(f"[SAVE] Static metrics saved: {static_csv}")

    # ROLLING validation (compute rolling realized cov and distances to model_cov)
    print(f"[INFO] Rolling-window validation (window={roll_window}) ...")
    rolling_records = []
    T = len(val_aligned_wide)
    for end in range(roll_window, T, ROLL_STEP):
        win = val_aligned_wide.iloc[end - roll_window:end]
        cov_win = win.cov().values
        cov_win = nearest_psd(cov_win)
        f_r = frobenius_norm(model_mat, cov_win)
        s_r = spectral_norm(model_mat, cov_win)
        k_r = kl_divergence_gaussian(model_mat, cov_win)
        te_r = portfolio_tracking_error(model_mat, cov_win)
        rolling_records.append({
            "Date": pd.to_datetime(val_aligned_wide.index[end]),
            "Frobenius": f_r,
            "Spectral": s_r,
            "KL": k_r,
            "TE": te_r
        })

    rolling_df = pd.DataFrame(rolling_records)
    rolling_csv = os.path.join(outdir, "rolling_window_results.csv")
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"[SAVE] Rolling results saved: {rolling_csv}")

    # plot benchmark (rolling Frobenius with static line)
    print("[INFO] Generating benchmark plot...")
    plt.figure(figsize=(10, 5))
    if not rolling_df.empty:
        plt.plot(rolling_df["Date"], rolling_df["Frobenius"], label="Rolling Frobenius")
    plt.axhline(fro, color="red", linestyle="--", label="Static Frobenius")
    plt.title(f"Validation Benchmark — {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Frobenius distance")
    plt.legend()
    benchmark_png = os.path.join(outdir, "benchmark_plot.png")
    plt.tight_layout()
    plt.savefig(benchmark_png, dpi=200)
    plt.close()
    print(f"[SAVE] Plot saved: {benchmark_png}")

    return {
        "static": static_metrics,
        "rolling_csv": rolling_csv,
        "static_csv": static_csv,
        "benchmark_plot": benchmark_png
    }


# ---------------------------
# Consolidation & scoring
# ---------------------------
def consolidate_and_score(models_results: List[Dict]) -> pd.DataFrame:
    """
    Build consolidated DataFrame with static metrics for all models,
    normalize metrics (min-max) and compute aggregated score (lower = better).

    Score design (simple, transparent):
      - For each metric M in {Frobenius, Spectral, KL, TE} compute normalized:
            M_norm = (M - min(Ms)) / (max(Ms) - min(Ms) + EPS)
      - Aggregated score = mean(M_norm) (equal weights)
      - Lower aggregated score implies better overall performance across metrics.
    """
    print("[INFO] Consolidating static metrics across models...")
    rows = [res["static"] for res in models_results]
    df = pd.DataFrame(rows).set_index("Model")

    metrics = ["Frobenius", "Spectral", "KL", "TE"]
    # handle edge cases where metric values are identical across models
    norm_df = pd.DataFrame(index=df.index)
    for m in metrics:
        vals = df[m].astype(float).values
        mn = float(np.nanmin(vals))
        mx = float(np.nanmax(vals))
        denom = (mx - mn) if (mx - mn) > 0 else EPS
        norm = (vals - mn) / denom
        norm_df[f"{m}_norm"] = norm

    # aggregated score: average of normalized metrics
    norm_df["score"] = norm_df[[f"{m}_norm" for m in metrics]].mean(axis=1)
    # ranking
    norm_df = norm_df.sort_values("score", ascending=True)
    consolidated = df.join(norm_df)
    consolidated.reset_index(inplace=True)
    consolidated_csv = os.path.join(OUTDIR, "consolidated_metrics.csv")
    consolidated.to_csv(consolidated_csv, index=False)
    print(f"[SAVE] Consolidated metrics saved: {consolidated_csv}")

    # plots: barplot of raw metrics and barplot of scores
    print("[INFO] Generating consolidated plots...")
    plt.figure(figsize=(10, 6))
    plot_metrics = ["Frobenius", "Spectral", "KL", "TE"]
    consolidated_melt = consolidated.melt(id_vars="Model", value_vars=plot_metrics,
                                          var_name="Metric", value_name="Value")
    sns.barplot(data=consolidated_melt, x="Metric", y="Value", hue="Model")
    plt.title("Validation Metrics Comparison (raw values)")
    plt.tight_layout()
    raw_plot = os.path.join(OUTDIR, "consolidated_metrics_barplot.png")
    plt.savefig(raw_plot, dpi=200)
    plt.close()
    print(f"[SAVE] Consolidated raw metrics barplot: {raw_plot}")

    # score barplot
    plt.figure(figsize=(8, 4))
    sns.barplot(data=consolidated, x="Model", y="score")
    plt.title("Aggregated Normalized Score (lower = better)")
    plt.tight_layout()
    score_plot = os.path.join(OUTDIR, "consolidated_score_barplot.png")
    plt.savefig(score_plot, dpi=200)
    plt.close()
    print(f"[SAVE] Consolidated score barplot: {score_plot}")

    return consolidated


# ---------------------------
# Main pipeline
# ---------------------------
def validation():
    print("\n[INFO] ===== STARTING FULL VALIDATION & CONSOLIDATION PIPELINE =====\n")

    # 1) Load validation returns and prepare wide matrix
    val_wide = load_and_prepare_validation(VALIDATION_CSV)

    # 2) Choose master tickers as intersection of training GL cov and validation (ensures baseline comparability)
    #    We require the validation baseline covariance to use the same ordered tickers for later comparisons.
    gl_cov_df = load_cov_csv(GL_COV)
    master_tickers = list(gl_cov_df.index)
    print(f"[INFO] Master tickers (from Graphical Lasso training) = {len(master_tickers)} tickers")

    missing_master = set(master_tickers) - set(val_wide.columns)
    if missing_master:
        raise ValueError(f"[ERROR] Validation missing required tickers from master list: {missing_master}")

    # reorder validation wide to master tickers (ensures baseline uses same ordering)
    val_wide = val_wide.loc[:, master_tickers]
    print("[INFO] Validation wide matrix reordered to master training tickers.")

    # 3) Baseline realized covariance (sample covariance over entire validation period)
    print("[INFO] Computing baseline realized covariance from validation period (pairwise-complete sample cov)...")
    val_cov_full = val_wide.cov()
    baseline_csv = os.path.join(OUTDIR, "baseline_cov_matrix_validation.csv")
    ensure_dir(os.path.dirname(baseline_csv))
    val_cov_full.to_csv(baseline_csv)
    print(f"[SAVE] Baseline validation covariance saved: {baseline_csv}")

    # 4) Per-model validation
    model_paths = [
        ("graphical_lasso", GL_COV),
        ("ledoit_wolf", LW_COV),
        ("dcc_garch", DCC_COV)
    ]
    results = []
    for model_name, cov_path in model_paths:
        try:
            res = validate_single_model(model_name, cov_path, val_wide, val_cov_full, roll_window=ROLL_WINDOW)
            results.append(res)
            print(f"[INFO] Completed validation for {model_name}")
        except Exception as e:
            print(f"[ERROR] Validation failed for {model_name}: {e}")

    if not results:
        raise RuntimeError("No model validations succeeded; aborting consolidation.")

    # 5) Consolidate and score
    consolidated_df = consolidate_and_score(results)
    print("\n[INFO] Final consolidated ranking (lower score = better):")
    print(consolidated_df[["Model", "score"]].sort_values("score"))

    print("\n[INFO] ===== FULL VALIDATION & CONSOLIDATION FINISHED =====\n")
    print(f"[INFO] Outputs located under: {OUTDIR}")

    return {
        "per_model_results": results,
        "consolidated": consolidated_df
    }


if __name__ == "__main__":
    validation()

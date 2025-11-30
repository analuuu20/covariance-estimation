"""
classical_estimators.py

Classical covariance estimators pipeline (pairwise-complete covariance + PSD correction).

This module:
  - loads the training returns CSV (data/train_returns.csv)
  - computes pairwise-complete covariances (use only observations where both assets exist)
  - projects the pairwise covariance matrix to the nearest PSD matrix (numerically stable)
  - fits classical covariance estimators:
      * Raw sample covariance (pairwise)
      * Ledoit-Wolf shrinkage
      * Oracle Approximating Shrinkage (OAS)
      * PCA-based covariance (reconstruct with k components to reach 90% explained variance)
      * Graphical Lasso tuned with GraphicalLassoCV (gamma selection via cross-validation)
  - prints intermediate progress messages (so you can follow execution)
  - generates publication-ready plots and a summary table with key parameters
  - saves outputs under tests/ and tests/plots/

Academic notes (short):
  - Pairwise-complete covariance is standard in finance for panels with missing data:
    covariance_{i,j} is computed using dates where both series i and j are observed.
  - Ledoit–Wolf and OAS provide shrinkage estimators that reduce estimation error
    in high-dimensional covariance estimation (Ledoit & Wolf, 2004; Chen et al., 2010).
  - PCA approximates covariance by low-rank factor structure (economically: common factors).
  - Graphical Lasso estimates a sparse precision matrix (conditional independence graph).

Requirements:
  pip install pandas numpy scikit-learn matplotlib seaborn joblib

Author: (you)
Date: (project)
"""

from __future__ import annotations
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV, GraphicalLasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Visual style (publication-friendly)
sns.set_theme(style="white", context="notebook", font_scale=1.05)

# Paths
TRAIN_PATH = "data/train_returns.csv"
OUT_DIR = "tests"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------
# Utilities
# -------------------------
def load_training_wide(train_csv: str = TRAIN_PATH) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load training returns CSV and pivot to wide panel (Date x Ticker).
    We DO NOT drop rows across all tickers; missing values are preserved.
    Returns:
        wide: DataFrame indexed by Date with columns = tickers, values = LogReturn
        tickers: list of tickers (columns order)
    """
    print(f"[LOAD] Loading training returns from '{train_csv}' ...")
    df = pd.read_csv(train_csv, parse_dates=["Date"])
    required = {"Date", "Ticker", "LogReturn"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Training CSV must contain columns {required}. Found: {list(df.columns)}")
    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn")
    print(f"[LOAD] Wide panel created: {wide.shape[0]} dates x {wide.shape[1]} tickers.")
    return wide, list(wide.columns)


def pairwise_covariance_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise-complete covariance matrix:
      Sigma[i,j] = cov(x_i, x_j) using only observations where both are present.

    This avoids dropping dates with any missingness across the entire panel.
    """
    print("[COV] Computing pairwise-complete covariance matrix...")
    cols = wide.columns
    N = len(cols)
    Sigma = np.zeros((N, N), dtype=float)

    # For numerical stability, convert to numpy arrays once
    # but we need to access masks by column
    for i in range(N):
        xi = wide.iloc[:, i]
        for j in range(i, N):
            xj = wide.iloc[:, j]
            valid = xi.notna() & xj.notna()
            nobs = int(valid.sum())
            if nobs > 1:
                # use np.cov on the aligned non-NA values; ddof=1 (sample covariance)
                cov_ij = np.cov(xi[valid].values, xj[valid].values, ddof=1)[0, 1]
                Sigma[i, j] = cov_ij
                Sigma[j, i] = cov_ij
            else:
                # Not enough overlapping observations -> set to 0 (will be corrected by PSD projection)
                Sigma[i, j] = 0.0
                Sigma[j, i] = 0.0

    Sigma_df = pd.DataFrame(Sigma, index=cols, columns=cols)
    print(f"[COV] Pairwise covariance computed. Some pairs may have few observations; check diagnostics if needed.")
    return Sigma_df


def nearest_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project symmetric matrix A to nearest positive semidefinite matrix by
    eigenvalue clipping. This simple projection is acceptable for producing a PSD matrix
    to feed into downstream algorithms (LedoitWolf/OAS/GLASSO/PCA).
    """
    # Ensure symmetry
    B = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals_clipped = np.clip(eigvals, a_min=eps, a_max=None)
    B_psd = (eigvecs @ np.diag(eigvals_clipped)) @ eigvecs.T
    # Re-enforce symmetry
    B_psd = (B_psd + B_psd.T) / 2.0
    return B_psd


def save_matrix_csv(mat: np.ndarray, labels: List[str], path: str):
    df = pd.DataFrame(mat, index=labels, columns=labels)
    df.to_csv(path)
    print(f"[SAVE] Saved matrix to: {path}")


def plot_heatmap(mat: np.ndarray, labels: List[str], path: str, title: str = None, cmap="viridis"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, xticklabels=labels, yticklabels=labels, cmap=cmap)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved heatmap: {path}")


# -------------------------
# Estimators
# -------------------------
def fit_ledoit_wolf(X: np.ndarray) -> Tuple[LedoitWolf, np.ndarray, float]:
    """
    Fit Ledoit–Wolf shrinkage estimator.
    Returns model object, covariance matrix, and shrinkage factor delta.
    """
    print("[LW] Fitting Ledoit-Wolf shrinkage estimator...")
    model = LedoitWolf().fit(X)
    cov = model.covariance_
    delta = float(model.shrinkage_)
    print(f"[LW] Done. Shrinkage δ = {delta:.6f}")
    return model, cov, delta


def fit_oas(X: np.ndarray) -> Tuple[OAS, np.ndarray, float]:
    """
    Fit Oracle Approximating Shrinkage (OAS).
    """
    print("[OAS] Fitting OAS estimator...")
    model = OAS().fit(X)
    cov = model.covariance_
    delta = float(model.shrinkage_)
    print(f"[OAS] Done. Shrinkage δ = {delta:.6f}")
    return model, cov, delta


def fit_pca_covariance(X: np.ndarray, explained_target: float = 0.90) -> Tuple[PCA, np.ndarray, int]:
    """
    Fit PCA on X (observations x variables) and reconstruct covariance
    using the minimal k components that reach explained_target cumulative variance.
    Returns PCA object, reconstructed covariance, and k.
    """
    print("[PCA] Fitting PCA and selecting components for target explained variance...")
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, explained_target) + 1)
    # Reconstruct covariance: Cov ≈ V_k Λ_k V_k^T (component loadings times variances)
    components_k = pca.components_[:k, :]  # shape k x N
    vars_k = pca.explained_variance_[:k]   # length k
    cov_pca = (components_k.T * vars_k) @ components_k
    print(f"[PCA] Selected k = {k} components to reach {explained_target*100:.0f}% explained variance.")
    return pca, cov_pca, k


def fit_glasso_cv(X: np.ndarray, alphas: List[float] = None) -> Tuple[GraphicalLassoCV, np.ndarray, float]:
    """
    Fit GraphicalLassoCV to select alpha and return fitted model and covariance.
    """
    print("[GLASSO] Running GraphicalLassoCV to find best alpha (this may take time)...")
    if alphas is None:
        alphas = np.logspace(-4, -1, 10)
    gl_cv = GraphicalLassoCV(alphas=alphas, cv=4).fit(X)
    best_alpha = float(gl_cv.alpha_)
    print(f"[GLASSO] Best alpha found by CV: {best_alpha:.6e}. Re-fitting GraphicalLasso with this alpha...")
    glasso = GraphicalLasso(alpha=best_alpha, max_iter=1000).fit(X)
    cov = glasso.covariance_
    return gl_cv, cov, best_alpha


# -------------------------
# Pipeline
# -------------------------
def run_classical_estimators_pipeline(train_csv: str = TRAIN_PATH, outdir: str = OUT_DIR):
    """
    Main pipeline to compute pairwise covariance, PSD-correct it, fit estimators,
    generate plots, and save summary.
    """
    print("\n[PIPELINE] Starting classical estimators pipeline...")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1) Load wide panel
    wide, tickers = load_training_wide(train_csv)

    # 2) Compute pairwise-complete covariance
    Sigma_pairwise = pairwise_covariance_matrix(wide)
    save_matrix_csv(Sigma_pairwise.values, tickers, os.path.join(outdir, "pairwise_cov_raw.csv"))
    plot_heatmap(Sigma_pairwise.values, tickers, os.path.join(PLOTS_DIR, "heatmap_pairwise_raw.png"),
                 title="Pairwise-Complete Raw Covariance")

    # 3) Project to nearest PSD
    print("[COV->PSD] Projecting pairwise matrix to nearest PSD (eigenvalue clipping)...")
    Sigma_psd = nearest_psd(Sigma_pairwise.values)
    save_matrix_csv(Sigma_psd, tickers, os.path.join(outdir, "pairwise_cov_psd.csv"))
    plot_heatmap(Sigma_psd, tickers, os.path.join(PLOTS_DIR, "heatmap_pairwise_psd.png"),
                 title="Pairwise Covariance (PSD-projected)")

    # 4) Prepare data matrix X for estimators
    # Most sklearn estimators expect rows = observations, cols = variables.
    # We need a design matrix X that reasonably represents joint observations.
    # Strategy: use the original wide panel but substitute NaNs with column means (unbiased) OR zero.
    # Here we choose column mean imputation (simple and standard for these estimators).
    print("[PREP] Preparing data matrix X for estimators with column-mean imputation.")
    X = wide.copy()
    col_means = X.mean(axis=0)
    X_imputed = X.fillna(col_means)
    X_values = X_imputed.values  # shape T x N

    # Standardize columns (zero mean) - many estimators assume centered data; shrinkage uses raw second moments, but centering is standard.
    X_centered = X_values - np.nanmean(X_values, axis=0)

    # 5) Raw sample covariance computed from imputed X (for comparison)
    cov_raw_from_imputed = np.cov(X_centered, rowvar=False)
    save_matrix_csv(cov_raw_from_imputed, tickers, os.path.join(outdir, "cov_from_imputed_raw.csv"))
    plot_heatmap(cov_raw_from_imputed, tickers, os.path.join(PLOTS_DIR, "heatmap_cov_imputed_raw.png"),
                 title="Sample Covariance (imputed)")

    # 6) Ledoit-Wolf
    lw_model, lw_cov, lw_delta = fit_ledoit_wolf(X_centered)
    dump(lw_model, os.path.join(outdir, "ledoit_wolf.joblib"))
    save_matrix_csv(lw_cov, tickers, os.path.join(outdir, "cov_ledoit_wolf.csv"))
    # Plot shrinkage
    plt.figure(figsize=(6, 4))
    plt.bar(["Ledoit-Wolf"], [lw_delta])
    plt.title("Ledoit-Wolf shrinkage delta")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lw_shrinkage.png"), dpi=300)
    plt.close()
    print("[LW] Saved Ledoit-Wolf results.")

    # 7) OAS
    oas_model, oas_cov, oas_delta = fit_oas(X_centered)
    dump(oas_model, os.path.join(outdir, "oas_model.joblib"))
    save_matrix_csv(oas_cov, tickers, os.path.join(outdir, "cov_oas.csv"))
    plt.figure(figsize=(6, 4))
    plt.bar(["OAS"], [oas_delta])
    plt.title("OAS shrinkage delta")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "oas_shrinkage.png"), dpi=300)
    plt.close()
    print("[OAS] Saved OAS results.")

    # 8) PCA
    pca_model, pca_cov, k = fit_pca_covariance(X_centered, explained_target=0.90)
    dump(pca_model, os.path.join(outdir, "pca_model.joblib"))
    save_matrix_csv(pca_cov, tickers, os.path.join(outdir, "cov_pca.csv"))
    # PCA scree plot
    ratios = np.cumsum(pca_model.explained_variance_ratio_)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(ratios) + 1), ratios, marker="o")
    plt.axhline(0.9, color="red", linestyle="--", label="90% explained")
    plt.title("PCA cumulative explained variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca_scree.png"), dpi=300)
    plt.close()
    print(f"[PCA] Saved PCA results. k = {k}")

    # 9) Graphical Lasso (CV)
    gl_model_cv, gl_cov, gl_alpha = fit_glasso_cv(X_centered)
    dump(gl_model_cv, os.path.join(outdir, "glasso_cv_model.joblib"))
    save_matrix_csv(gl_cov, tickers, os.path.join(outdir, "cov_glasso.csv"))
    # Sparsity pattern of precision matrix
    precision = gl_model_cv.covariance_precision_ if hasattr(gl_model_cv, "covariance_precision_") else gl_model_cv.precision_
    plt.figure(figsize=(8, 6))
    sns.heatmap((precision != 0).astype(int), cmap="binary")
    plt.title("GLASSO sparsity pattern (precision non-zero)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "glasso_sparsity.png"), dpi=300)
    plt.close()
    print(f"[GLASSO] Saved GLASSO results. Best alpha = {gl_alpha:.6e}")

    # 10) Additional diagnostic plots: condition numbers and eigenvalue traces
    def condnum(mat: np.ndarray) -> float:
        return float(np.linalg.cond(mat))

    mats = {
        "Raw (pairwise)": Sigma_pairwise.values,
        "PSD pairwise": Sigma_psd,
        "Raw (imputed)": cov_raw_from_imputed,
        "Ledoit-Wolf": lw_cov,
        "OAS": oas_cov,
        "PCA": pca_cov,
        "GLASSO": gl_cov
    }

    conds = {name: condnum(mat) for name, mat in mats.items()}
    # Bar plot of condition numbers
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(conds.values()), y=list(conds.keys()), palette="muted")
    plt.xlabel("Condition number")
    plt.title("Condition numbers of candidate covariance matrices")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "condition_numbers.png"), dpi=300)
    plt.close()
    print("[PLOT] Saved condition numbers plot.")

    # Eigenvalue trace over models (sorted)
    eig_traces = {}
    for name, mat in mats.items():
        try:
            eigs = np.linalg.eigvalsh(mat)
            eig_traces[name] = np.sum(eigs)  # trace = sum eigenvalues
        except Exception:
            eig_traces[name] = float("nan")

    trace_df = pd.DataFrame([eig_traces])
    trace_df.to_csv(os.path.join(outdir, "eigenvalue_traces.csv"), index=False)
    print("[SAVE] Saved eigenvalue traces.")

    # 11) Build summary table and save
    summary_rows = [
        {"Model": "Pairwise Raw", "KeyParameter": "N/A", "CondNumber": conds["Raw (pairwise)"], "Notes": "Pairwise covariance (no PSD)"},
        {"Model": "Pairwise PSD", "KeyParameter": "eigenclip", "CondNumber": conds["PSD pairwise"], "Notes": "PSD-projected pairwise matrix"},
        {"Model": "Raw (imputed)", "KeyParameter": "impute=colmean", "CondNumber": conds["Raw (imputed)"], "Notes": "Raw from column-mean imputed panel"},
        {"Model": "Ledoit-Wolf", "KeyParameter": f"δ={lw_delta:.6f}", "CondNumber": conds["Ledoit-Wolf"], "Notes": "Shrinkage estimator"},
        {"Model": "OAS", "KeyParameter": f"δ={oas_delta:.6f}", "CondNumber": conds["OAS"], "Notes": "Oracle-approx shrinkage"},
        {"Model": "PCA", "KeyParameter": f"k={k}", "CondNumber": conds["PCA"], "Notes": "Low-rank factor covariance"},
        {"Model": "GLASSO", "KeyParameter": f"alpha={gl_alpha:.6e}", "CondNumber": conds["GLASSO"], "Notes": "Sparse precision (Graphical Lasso)"}
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(outdir, "classical_estimators_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SUMMARY] Summary table saved: {summary_csv}")

    print("\n[SUMMARY TABLE]")
    print(summary_df.to_string(index=False))

    print("\n[PIPELINE] Classical estimators pipeline finished.")

    # Return summary and models for programmatic use
    results = {
        "Sigma_pairwise": Sigma_pairwise,
        "Sigma_psd": Sigma_psd,
        "lw_model": lw_model,
        "oas_model": oas_model,
        "pca_model": pca_model,
        "glasso_cv": gl_model_cv,
        "summary": summary_df
    }
    return results


# If executed as script
if __name__ == "__main__":
    run_classical_estimators_pipeline()

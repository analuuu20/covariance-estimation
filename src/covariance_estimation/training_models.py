"""
covariance_training.py

Training / calibration module for covariance estimators (TRAINING-STAGE ONLY).

- Input: data/train_returns.csv (panel with columns: Date, Ticker, LogReturn)
- Output:
    - pickles for trained models (tests/models/)
    - CSVs of covariance matrices (tests/covariances/)
    - plots (results/plots/)
    - summary table with calibration & diagnostics (results/covariance_training_summary.csv)

This version attempts to use mgarch or mvgarch for a DCC-GARCH baseline.
If neither library is installed, the script falls back to using the
sample covariance as the DCC baseline but still trains the other estimators.

Academic notes:
- DCC-GARCH is estimated on a seeded random subset of 30 tickers to balance
  representativeness and computational feasibility.
- Other estimators (Ledoit-Wolf, OAS, PCA, GLASSO) operate on the full cleaned training set.
"""

from __future__ import annotations
import os
import warnings
import joblib
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.covariance import LedoitWolf, OAS, GraphicalLasso
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# Try to import multivariate GARCH libraries that implement DCC/GARCH.
# We try a couple of known names; availability depends on your environment.
DCC_LIB = None
DCC_LIB_NAME = None
try:
    import mvgarch  # if installed
    DCC_LIB = mvgarch
    DCC_LIB_NAME = "mvgarch"
except Exception:
    try:
        import mgarch  # if installed
        DCC_LIB = mgarch
        DCC_LIB_NAME = "mgarch"
    except Exception:
        DCC_LIB = None
        DCC_LIB_NAME = None

if DCC_LIB_NAME:
    print(f"[INFO] Multivariate-GARCH library available: {DCC_LIB_NAME}")
else:
    warnings.warn("[WARN] No multivariate GARCH library found (mvgarch/mgarch not installed). DCC baseline will be fallback (sample covariance).")


# ---------------------------
# Helper I/O & plotting
# ---------------------------

def ensure_dirs():
    os.makedirs("tests/models", exist_ok=True)
    os.makedirs("tests/covariances", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def load_training_returns(path: str = "data/train_returns.csv") -> pd.DataFrame:
    """
    Load the training returns panel and validate expected columns.
    """
    print(f"[STEP] Loading training returns from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    expected = {"Date", "Ticker", "LogReturn"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns {expected}. Found: {list(df.columns)}")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"[STEP] Loaded training panel: {df['Ticker'].nunique()} tickers, {df['Date'].nunique()} unique dates.")
    return df


def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot panel to a wide matrix Date x Ticker of LogReturn.
    """
    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn")
    wide = wide.sort_index(axis=1)
    return wide


def plot_and_save_heatmap(mat: np.ndarray, labels: List[str], filename: str, annotate_limit: int = 40, title: Optional[str] = None):
    """
    Save heatmap with sensible defaults. If many tickers, omit axis labels for legibility.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, xticklabels=(labels if len(labels) <= annotate_limit else False),
                yticklabels=(labels if len(labels) <= annotate_limit else False),
                cmap="vlag", center=0)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[PLOT] Saved heatmap to {filename}")


def plot_scree(eigvals: np.ndarray, filename: str, title: str = "Scree plot (eigvals)"):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(eigvals) + 1), np.sort(eigvals)[::-1], marker="o")
    plt.title(title)
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[PLOT] Saved scree to {filename}")


# ---------------------------
# Estimators: training functions
# ---------------------------

def estimate_sample_cov(X: np.ndarray) -> np.ndarray:
    """Sample covariance (population MLE: ddof=0)."""
    return np.cov(X, rowvar=False, bias=True)


def estimate_ledoit_wolf(X: np.ndarray) -> Tuple[np.ndarray, Any]:
    lw = LedoitWolf().fit(X)
    return lw.covariance_, lw


def estimate_oas(X: np.ndarray) -> Tuple[np.ndarray, Any]:
    oas = OAS().fit(X)
    return oas.covariance_, oas


def estimate_pca_cov(X: np.ndarray, explained_target: float = 0.90) -> Tuple[np.ndarray, Dict[str, Any]]:
    pca_full = PCA(n_components=min(X.shape[1], X.shape[0]))
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, explained_target) + 1)
    pca_k = PCA(n_components=k).fit(X)
    cov_rec = pca_k.get_covariance()
    meta = {"n_components": k, "explained_variance": float(cumvar[k - 1])}
    return cov_rec, meta


def estimate_glasso_cv(X: np.ndarray, alphas: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simple GLASSO alpha grid search using sklearn GraphicalLasso.score as selection criterion."""
    from sklearn.covariance import GraphicalLasso
    if alphas is None:
        alphas = list(np.logspace(-4, -1, 10))
    best_score = -np.inf
    best_alpha = None
    best_model = None
    for a in alphas:
        try:
            model = GraphicalLasso(alpha=a, max_iter=200)
            model.fit(X)
            score = model.score(X)
            print(f"[GLASSO] alpha={a:.4g} score={score:.4f}")
            if score > best_score:
                best_score = score
                best_alpha = a
                best_model = model
        except Exception as e:
            warnings.warn(f"GLASSO alpha {a} failed: {e}")
    if best_model is None:
        raise RuntimeError("GLASSO calibration failed for all alphas")
    return best_model.covariance_, {"best_alpha": best_alpha, "score": best_score, "model": best_model}


# ---------------------------
# Multivariate GARCH / DCC baseline
# ---------------------------

def estimate_dcc_with_mvgarch(wide_df: pd.DataFrame, subset_tickers: List[str], random_state: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Try to estimate DCC-GARCH using an installed multivariate-GARCH package.

    Strategy:
    - If `mvgarch` is available, use its API.
    - Else if `mgarch` is available, use its API.
    - Else, fallback to the sample covariance and record fallback in metadata.

    Important:
    - Different packages have different APIs; here we attempt common patterns
      but also include clear error handling and informative prints.
    """
    metadata: Dict[str, Any] = {}
    if DCC_LIB is None:
        warnings.warn("No multivariate GARCH library found; falling back to sample covariance for DCC baseline.")
        return estimate_sample_cov(wide_df[subset_tickers].values), {"dcc_used": False, "reason": "no library"}

    # deterministic random pick (seed) – selection already performed by caller ideally
    np.random.seed(random_state)
    X_sub = wide_df[subset_tickers].dropna()
    print(f"[DCC] Subset for DCC: {X_sub.shape[0]} timepoints x {X_sub.shape[1]} tickers")

    try:
        if DCC_LIB_NAME == "mvgarch":
            # Example usage pattern for mvgarch (note: actual API may differ; adjust if necessary)
            # This block tries to be robust: we call common names and catch exceptions.
            print("[DCC] Attempting to fit DCC via 'mvgarch' package...")
            model = DCC_LIB.MVGARCH(X_sub)  # hypothetical: MVGARCH wrapper
            res = model.fit()               # hypothetical usage
            # Suppose res has attribute covariances: (T, N, N)
            cov_ts = getattr(res, "covariances", None)
            if cov_ts is None:
                # try alternative attribute names or fallback
                cov_ts = getattr(res, "Sigma_t", None)
            if cov_ts is None:
                raise RuntimeError("mvgarch fit returned no covariance series attribute - please inspect the library API.")
            avg_cov = np.mean(cov_ts, axis=0)
            metadata.update({"dcc_used": True, "library": "mvgarch", "n_time": cov_ts.shape[0]})
            print("[DCC] mvgarch DCC fit complete.")
            return avg_cov, metadata

        elif DCC_LIB_NAME == "mgarch":
            print("[DCC] Attempting to fit DCC via 'mgarch' package...")
            # Hypothetical API usage for mgarch
            model = DCC_LIB.DCC(X_sub)
            res = model.fit()
            cov_ts = getattr(res, "covariances", None) or getattr(res, "Sigma_t", None)
            if cov_ts is None:
                raise RuntimeError("mgarch result lacks covariance time-series attribute.")
            avg_cov = np.mean(cov_ts, axis=0)
            metadata.update({"dcc_used": True, "library": "mgarch", "n_time": cov_ts.shape[0]})
            print("[DCC] mgarch DCC fit complete.")
            return avg_cov, metadata

        else:
            raise RuntimeError("Unknown DCC_LIB_NAME encountered.")
    except Exception as e:
        warnings.warn(f"DCC fit via {DCC_LIB_NAME} failed: {e}. Falling back to sample covariance for subset.")
        return estimate_sample_cov(X_sub.values), {"dcc_used": False, "error": str(e)}


# ---------------------------
# Main training routine
# ---------------------------

def train_all_models(train_returns_path: str = "data/train_returns.csv",
                     dcc_subset_n: int = 30,
                     random_state: int = 42,
                     pca_explained: float = 0.90,
                     glasso_alphas: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Master training pipeline that fits the DCC baseline (on subset) and the
    static/regularized estimators on the full cleaned training set.
    """

    ensure_dirs()
    print("[START] Loading training data...")
    panel = load_training_returns(train_returns_path)

    # Pivot and conservative cleaning: drop tickers with >25% missing and drop any NA rows
    wide = pivot_wide(panel)
    na_frac = wide.isna().mean()
    keep = na_frac[na_frac <= 0.25].index.tolist()
    print(f"[CLEAN] Keeping {len(keep)} tickers (<=25% missing).")
    wide = wide[keep]
    wide = wide.dropna(axis=0, how="any")
    print(f"[CLEAN] After dropna: {wide.shape[0]} dates x {wide.shape[1]} tickers.")

    # Full matrix for static estimators
    X = wide.values  # T x N

    # Save sample covariance (benchmark)
    cov_sample = estimate_sample_cov(X)
    pd.DataFrame(cov_sample, index=wide.columns, columns=wide.columns).to_csv("tests/covariances/cov_sample.csv")
    joblib.dump({"type": "sample_cov"}, "tests/models/sample_meta.pkl")
    print("[SAVE] Sample covariance saved.")

    # DCC baseline: select seeded random subset of tickers
    print(f"[DCC] Selecting {dcc_subset_n} tickers (seed={random_state}) for DCC baseline...")
    rng = np.random.RandomState(random_state)
    if dcc_subset_n >= len(wide.columns):
        subset = list(wide.columns)
    else:
        subset = list(rng.choice(wide.columns, size=dcc_subset_n, replace=False))
    print(f"[DCC] Subset selected: {subset[:8]}... (total {len(subset)})")

    cov_dcc, meta_dcc = estimate_dcc_with_mvgarch(wide, subset, random_state=random_state)
    pd.DataFrame(cov_dcc, index=subset, columns=subset).to_csv("tests/covariances/cov_dcc.csv")
    joblib.dump(meta_dcc, "tests/models/dcc_meta.pkl")
    print("[SAVE] DCC baseline covariance and meta saved (or fallback).")

    # Ledoit-Wolf
    print("[TRAIN] Estimating Ledoit-Wolf shrinkage...")
    cov_lw, lw_model = estimate_ledoit_wolf(X)
    pd.DataFrame(cov_lw, index=wide.columns, columns=wide.columns).to_csv("tests/covariances/cov_ledoitwolf.csv")
    joblib.dump(lw_model, "tests/models/ledoitwolf_model.pkl")
    print("[SAVE] Ledoit-Wolf model saved.")

    # OAS
    print("[TRAIN] Estimating OAS shrinkage...")
    cov_oas, oas_model = estimate_oas(X)
    pd.DataFrame(cov_oas, index=wide.columns, columns=wide.columns).to_csv("tests/covariances/cov_oas.csv")
    joblib.dump(oas_model, "tests/models/oas_model.pkl")
    print("[SAVE] OAS model saved.")

    # PCA
    print(f"[TRAIN] Fitting PCA to reach {pca_explained*100:.0f}% explained variance...")
    cov_pca, meta_pca = estimate_pca_cov(X, explained_target=pca_explained)
    pd.DataFrame(cov_pca, index=wide.columns, columns=wide.columns).to_csv("tests/covariances/cov_pca.csv")
    joblib.dump(meta_pca, "tests/models/pca_meta.pkl")
    print(f"[SAVE] PCA covariance and meta saved. Components: {meta_pca['n_components']}")

    # GLASSO
    print("[TRAIN] Calibrating Graphical Lasso (alpha grid)...")
    cov_glasso, meta_glasso = estimate_glasso_cv(X, alphas=glasso_alphas)
    pd.DataFrame(cov_glasso, index=wide.columns, columns=wide.columns).to_csv("tests/covariances/cov_glasso.csv")
    try:
        joblib.dump(meta_glasso["model"], "tests/models/glasso_model.pkl")
    except Exception:
        joblib.dump(meta_glasso, "tests/models/glasso_meta.pkl")
    print(f"[SAVE] GLASSO covariance and model saved. Best alpha: {meta_glasso.get('best_alpha')}")

    # -------------------------
    # Diagnostics & plots
    # -------------------------
    print("[PLOTS] Generating diagnostic plots...")

    # Scree from sample covariance
    eigvals = np.linalg.eigvalsh(cov_sample)[::-1]
    plot_scree(eigvals, "results/plots/scree_sample.png")

    # Heatmaps (limit labeling if many tickers)
    plot_and_save_heatmap(cov_sample, list(wide.columns), "results/plots/cov_sample.png", title="Sample Covariance (training)")
    plot_and_save_heatmap(cov_lw, list(wide.columns), "results/plots/cov_ledoitwolf.png", title="Ledoit-Wolf Covariance (training)")
    plot_and_save_heatmap(cov_oas, list(wide.columns), "results/plots/cov_oas.png", title="OAS Covariance (training)")
    plot_and_save_heatmap(cov_pca, list(wide.columns), "results/plots/cov_pca.png", title="PCA Reconstructed Covariance (training)")
    plot_and_save_heatmap(cov_glasso, list(wide.columns), "results/plots/cov_glasso.png", title="GLASSO Covariance (training)")

    # GLASSO sparsity if available
    try:
        precision = meta_glasso["model"].precision_
        plot_and_save_heatmap((precision != 0).astype(int), list(wide.columns), "results/plots/glasso_sparsity.png", title="GLASSO Precision Sparsity (1=nonzero)")
    except Exception:
        print("[PLOT] GLASSO precision not available to plot sparsity.")

    # Condition numbers plot
    conds = {
        "Sample": np.linalg.cond(cov_sample),
        "LedoitWolf": np.linalg.cond(cov_lw),
        "OAS": np.linalg.cond(cov_oas),
        "PCA": np.linalg.cond(cov_pca),
        "GLASSO": np.linalg.cond(cov_glasso),
        "DCC-GARCH": (np.nan if not meta_dcc.get("dcc_used", False) else np.linalg.cond(cov_dcc))
    }
    # simple bar horizontal plot
    plt.figure(figsize=(7, max(4, 0.4 * len(conds))))
    sns.barplot(x=list(conds.values()), y=list(conds.keys()), palette="muted")
    plt.xlabel("Condition Number")
    plt.title("Condition numbers by estimator")
    plt.tight_layout()
    plt.savefig("results/plots/condition_numbers.png")
    plt.close()
    print("[PLOT] Saved condition numbers plot.")

    # -------------------------
    # Summary table
    # -------------------------
    print("[SUMMARY] Building summary table of diagnostics and meta...")

    rows = []
    models = {
        "Sample": (cov_sample, {}),
        "DCC-GARCH": (cov_dcc, meta_dcc),
        "LedoitWolf": (cov_lw, {"model": "LedoitWolf"}),
        "OAS": (cov_oas, {"model": "OAS"}),
        "PCA": (cov_pca, meta_pca),
        "GLASSO": (cov_glasso, {"meta": meta_glasso})
    }

    for name, (covm, meta) in models.items():
        eigs = np.linalg.eigvalsh(covm)
        posdef = bool(np.all(eigs > 0))
        cond = float(np.linalg.cond(covm))
        N = covm.shape[0]
        ew_var = float(np.ones(N) @ covm @ np.ones(N) / (N**2))
        frob_to_sample = float(np.linalg.norm(covm - cov_sample, ord="fro"))
        rows.append({
            "Model": name,
            "PositiveDefinite": posdef,
            "ConditionNumber": cond,
            "EqualWeightPortfolioVar": ew_var,
            "FrobeniusToSample": frob_to_sample,
            "Meta": str(meta)
        })

    summary_df = pd.DataFrame(rows)
    summary_csv = "results/covariance_training_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    joblib.dump({"meta_dcc": meta_dcc, "meta_glasso": meta_glasso, "meta_pca": meta_pca}, "tests/models/all_models_meta.pkl")
    print(f"[SAVE] Summary CSV saved to {summary_csv} and metadata pickled.")

    print("[DONE] Training completed successfully.")
    return summary_df


# ---------------------------
# Run when executed
# ---------------------------
if __name__ == "__main__":
    summary = train_all_models()
    print("\n=== Training summary ===")
    print(summary.to_string(index=False))

# src/covariance_estimation/08_full_validation.py
"""
Full validation pipeline for covariance estimators.

Validates the following trained models (trained on train_returns.csv):
 - DCC-GARCH (low-memory / final covariance)
 - Ledoit-Wolf shrinkage estimator
 - Graphical Lasso (alpha selected via CV)

This script performs:
 1) Static validation over the entire validation sample
 2) Rolling-window (1-step) validation with forecast horizon H (default 5 days)
 3) Computes distance metrics between predicted covariance and realized covariance:
      - Frobenius norm
      - Spectral norm (operator norm)
      - KL divergence (multivariate Gaussian)
 4) Portfolio-based validation: long-only Global Minimum Variance Portfolio (GMVP)
    computed from each predicted covariance; measure realized out-of-sample
    portfolio variance over the forecast horizon.
 5) Saves results, diagnostic plots, and a final summary table.

Outputs saved to: results/validation/

Author: Generated (academic-style comments). 2025
"""

from __future__ import annotations
import os
import sys
import math
import pickle
import warnings
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estimators
from sklearn.covariance import LedoitWolf
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

# Quadratic programming for long-only GMVP
try:
    import cvxpy as cp
except Exception:
    cp = None

# Visual style
sns.set_theme(style="whitegrid", font_scale=1.05)

# ---------------------------
# Config / file locations
# ---------------------------
VALIDATION_CSV = "data/validation_returns.csv"  # input
OUTDIR = "results/validation"
PLOTS_DIR = os.path.join(OUTDIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Trained-model candidate paths to try (update if your paths differ)
DCC_PICKLE_PATHS = [
    "models/dcc_model_lowmem.pkl",
    "models/dcc_model.pkl",
    "tests/dcc_garch_model.pkl",
    "models/dcc_fit.pkl"
]
DCC_COV_CSVS = [
    "models/dcc_covariance_final.csv",
    "tests/dcc_garch_covariance_final.csv",
    "models/dcc_covariance.csv",
]

LW_MODEL_PATHS = [
    "results/ledoit_wolf/lw_model.pkl",
    "results/ledoit_wolf/ledoit_wolf.joblib",
    "tests/ledoit_wolf.joblib",
    "tests/lw_cov.pkl"
]

GL_MODEL_PATHS = [
    "results/graphical_lasso/gl_model.pkl",
    "results/graphical_lasso/glasso_cv_model.joblib",
    "tests/glasso_model.pkl",
]

# Rolling validation parameters
ROLLING_HORIZON = 5   # forecast horizon (days) to evaluate realized covariance
ROLLING_STEP = 1      # step in days to move rolling window
MIN_OBS_REALIZED = 3  # minimal days required to compute realized cov for horizon

# Numerical stability
EPS = 1e-8

# ---------------------------
# Utilities
# ---------------------------
def nearest_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project symmetric A to nearest PSD by eigenvalue clipping."""
    B = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals_clipped = np.clip(eigvals, a_min=eps, a_max=None)
    B_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return (B_psd + B_psd.T) / 2.0

def safe_inv(mat: np.ndarray) -> np.ndarray:
    """Compute stable inverse via eigen-decomposition with clipping."""
    mat = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals_clipped = np.clip(eigvals, a_min=EPS, a_max=None)
    inv = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T
    return (inv + inv.T) / 2.0

def kl_divergence_gaussian(S_pred: np.ndarray, S_true: np.ndarray) -> float:
    """
    KL(N(0,S_true) || N(0,S_pred)) = 0.5*( tr(S_pred^{-1} S_true) - n + ln(det S_pred / det S_true) ).
    We compute a symmetric-ish measure by returning that scalar (lower is better).
    """
    n = S_true.shape[0]
    S_pred = nearest_psd(S_pred)
    S_true = nearest_psd(S_true)
    inv_pred = safe_inv(S_pred)
    tr_term = float(np.trace(inv_pred @ S_true))
    sign_p, logdet_p = np.linalg.slogdet(S_pred)
    sign_t, logdet_t = np.linalg.slogdet(S_true)
    if sign_p <= 0 or sign_t <= 0:
        # fallback: use eigenvalues to approximate logdet
        logdet_p = float(np.sum(np.log(np.clip(np.linalg.eigvalsh(S_pred), EPS, None))))
        logdet_t = float(np.sum(np.log(np.clip(np.linalg.eigvalsh(S_true), EPS, None))))
    kl = 0.5 * (tr_term - n + (logdet_p - logdet_t))
    return float(kl)

def frobenius_norm(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro"))

def spectral_norm(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord=2))

def load_pickle_any(paths: List[str]):
    for p in paths:
        if os.path.exists(p):
            print(f"[LOAD] Found model file: {p}")
            with open(p, "rb") as f:
                try:
                    obj = pickle.load(f)
                except Exception:
                    # try joblib if pickle fails
                    try:
                        import joblib
                        obj = joblib.load(p)
                    except Exception as e:
                        print(f"[WARN] Could not load {p}: {e}")
                        continue
            return obj, p
    print("[WARN] No model file found in provided paths.")
    return None, None

def try_load_csv_any(paths: List[str]):
    for p in paths:
        if os.path.exists(p):
            print(f"[LOAD] Found CSV file: {p}")
            df = pd.read_csv(p, index_col=0)
            return df, p
    return None, None

# ---------------------------
# Load trained models
# ---------------------------
def load_trained_models():
    print("[STEP] Attempting to load trained models from disk...")

    # DCC
    dcc_obj, dcc_path = load_pickle_any(DCC_PICKLE_PATHS)
    if dcc_obj is None:
        # try loading final covariance CSV as fallback
        dcc_cov_df, dcc_cov_path = try_load_csv_any(DCC_COV_CSVS)
        if dcc_cov_df is not None:
            print("[INFO] DCC covariance CSV loaded as fallback.")
            dcc_obj = {"covariance_final": dcc_cov_df.values, "cov_csv": dcc_cov_path}
        else:
            print("[WARN] DCC not found. DCC validations will be skipped.")
            dcc_obj = None

    # Ledoit-Wolf
    lw_obj, lw_path = load_pickle_any(LW_MODEL_PATHS)
    if lw_obj is None:
        # try reading covariance CSV if model missing
        lw_cov_df, lw_cov_path = try_load_csv_any([
            "results/ledoit_wolf/lw_covariance_matrix.csv",
            "results/ledoit_wolf/cov_ledoit_wolf.csv",
            "tests/raw_cov.csv"
        ])
        if lw_cov_df is not None:
            lw_obj = {"covariance": lw_cov_df.values, "cov_csv": lw_cov_path}
        else:
            print("[WARN] Ledoit-Wolf model not found. LW validations will be skipped.")
            lw_obj = None

    # Graphical Lasso
    gl_obj, gl_path = load_pickle_any(GL_MODEL_PATHS)
    if gl_obj is None:
        gl_cov_df, gl_cov_path = try_load_csv_any([
            "results/graphical_lasso/gl_covariance_matrix.csv",
            "results/graphical_lasso/cov_glasso.csv"
        ])
        if gl_cov_df is not None:
            gl_obj = {"covariance": gl_cov_df.values, "cov_csv": gl_cov_path}
        else:
            print("[WARN] Graphical Lasso model not found. GLASSO validation will be skipped.")
            gl_obj = None

    return {"dcc": (dcc_obj, dcc_path), "lw": (lw_obj, lw_path), "gl": (gl_obj, gl_path)}

# ---------------------------
# Load validation data
# ---------------------------
def load_validation_matrix(path=VALIDATION_CSV) -> Tuple[pd.DataFrame, List[str]]:
    print(f"[STEP] Loading validation returns from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "LogReturn"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Validation CSV must include columns: {required}. Found: {list(df.columns)}")
    wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").sort_index()
    print(f"[STEP] Validation wide matrix shape: {wide.shape} (T x N)")
    print(f"[INFO] Example tickers (first 40): {list(wide.columns[:40])}")
    return wide, list(wide.columns)

# ---------------------------
# Portfolio: long-only GMVP via cvxpy
# ---------------------------
def solve_long_only_gmvp(S: np.ndarray) -> np.ndarray:
    """
    Solve long-only GMVP: minimize w' S w subject to sum(w)=1, w >= 0.
    Returns weights as numpy array (n,).
    Requires cvxpy.
    """
    n = S.shape[0]
    S = nearest_psd(S)
    if cp is None:
        raise RuntimeError("cvxpy is required for constrained GMVP. Install with `pip install cvxpy`.")
    w = cp.Variable(shape=(n,))
    objective = cp.quad_form(w, S)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        # fallback: equal weights
        print("[WARN] cvxpy failed to find solution; falling back to equal weights.")
        return np.ones(n) / n
    return np.array(w.value).flatten()

# ---------------------------
# Helper: predicted covariance extractor
# ---------------------------
def get_predicted_covariance(model_obj, model_name: str, assets: List[str], reference_index: int = 0):
    """
    Extract a predicted covariance matrix for the set of assets.
    model_obj can be:
      - sklearn estimator object with .covariance_ attribute (LedoitWolf, GraphicalLasso)
      - pickle object from multigarch with attributes .H (final covariance) or .forecast(...)
      - dict fallback containing 'covariance'
    """
    if model_obj is None:
        return None

    # If it's a dict loaded from CSV fallback
    if isinstance(model_obj, dict) and "covariance" in model_obj:
        cov = np.asarray(model_obj["covariance"])
        return cov

    # If it's a multigarch DCC object pickled
    # Try common attribute names for final covariance: .H (final in low_memory), .H_ (maybe), or attribute 'covariance_final'
    if model_name == "dcc":
        # If the pickled object is an actual DCC instance
        if hasattr(model_obj, "H"):
            H_attr = getattr(model_obj, "H")
            # H might be final matrix (n,n) or (T,n,n)
            H = np.asarray(H_attr)
            if H.ndim == 2:
                return H
            elif H.ndim == 3:
                # return final slice
                return H[-1]
        # Try forecast(1)
        if hasattr(model_obj, "forecast"):
            try:
                fc = model_obj.forecast(horizon=1)
                # many implementations return dict with "covariance" key or an array
                if isinstance(fc, dict) and "covariance" in fc:
                    cov = np.asarray(fc["covariance"][0])
                    return cov
                else:
                    arr = np.asarray(fc)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        return arr[0]
            except Exception as e:
                print(f"[WARN] DCC forecast failed: {e}. Trying fallback attributes.")
        # Try dictionary fallback keys
        if isinstance(model_obj, dict) and "covariance_final" in model_obj:
            return np.asarray(model_obj["covariance_final"])
        if isinstance(model_obj, dict) and "Sigma_T" in model_obj:
            return np.asarray(model_obj["Sigma_T"])
        # Try loading CSV if provided in model_obj
        if isinstance(model_obj, dict) and "cov_csv" in model_obj:
            try:
                df = pd.read_csv(model_obj["cov_csv"], index_col=0)
                return df.values
            except Exception:
                pass
        print("[WARN] Could not extract covariance from DCC object - returning None.")
        return None

    # For sklearn-like models
    if hasattr(model_obj, "covariance_"):
        try:
            cov = np.asarray(model_obj.covariance_)
            return cov
        except Exception:
            pass

    # Joblib objects sometimes store estimator in attribute 'estimator_' or are dicts
    if isinstance(model_obj, dict) and "covariance" in model_obj:
        return np.asarray(model_obj["covariance"])

    # Last resort: if it's a numpy array stored directly
    if isinstance(model_obj, np.ndarray):
        return model_obj

    print(f"[WARN] Unknown model object type for {model_name}.")
    return None

# ---------------------------
# Realized covariance estimator over a horizon (sample covariance)
# ---------------------------
def realized_covariance(returns_window: np.ndarray) -> np.ndarray:
    """
    Compute realized covariance from returns over the horizon.
    returns_window: (H, n) array of returns across the horizon.
    If H == 1, use outer product r_t r_t^T as a naive realized covariance.
    """
    H, n = returns_window.shape
    if H <= 1:
        # use outer product
        r = returns_window[-1]
        return np.outer(r, r)
    # use sample covariance (ddof=0 for population cov over the horizon)
    return np.cov(returns_window, rowvar=False, ddof=0)

# ---------------------------
# Static validation function (single predicted covariance vs whole validation realized)
# ---------------------------
def static_validation(pred_cov: np.ndarray, val_returns_wide: pd.DataFrame) -> Dict[str, float]:
    """
    Compare predicted covariance to a realized covariance computed from entire validation set
    (pairwise-complete sample covariance).
    """
    # realized sample covariance from validation (pairwise-complete approach)
    realized_cov = val_returns_wide.cov(min_periods=1).values
    realized_cov = nearest_psd(realized_cov)
    pred_cov = nearest_psd(pred_cov)

    metrics = {
        "frobenius": frobenius_norm(pred_cov, realized_cov),
        "spectral": spectral_norm(pred_cov, realized_cov),
        "kl": kl_divergence_gaussian(pred_cov, realized_cov)
    }
    return metrics

# ---------------------------
# Rolling validation procedure
# ---------------------------
def rolling_validation_for_model(model_obj, model_name: str, val_wide: pd.DataFrame,
                                 assets: List[str], rolling_horizon: int = ROLLING_HORIZON):
    """
    Run a rolling evaluation across the validation sample.

    For each time t where t + rolling_horizon <= T:
      - Use the trained model (static) to produce a predicted covariance for horizon t+1
        * For LedoitWolf/GraphicalLasso we use the saved estimator's covariance_ as static prediction
        * For DCC, try to use forecast(horizon=1) if available; else fall back to final Sigma_T
      - Compute realized covariance using returns at t+1 ... t+rolling_horizon (if available)
      - Compute distance metrics and a GMVP (long-only) OOS portfolio variance
    """
    T, N = val_wide.shape
    print(f"[ROLL] Starting rolling validation for {model_name}. T={T}, N={N}, horizon={rolling_horizon}")

    results = []
    # convert to numpy for speed
    R = val_wide.values  # shape T x N

    # Use model-level predicted covariance if model is static (common for LW/GL)
    static_cov = get_predicted_covariance(model_obj, model_name, assets)
    if static_cov is None:
        print(f"[WARN] No static covariance available for {model_name}. Skipping rolling path predictions; will attempt forecast if available.")
    else:
        static_cov = nearest_psd(static_cov)
    # check if model supports dynamic forecast
    supports_forecast = hasattr(model_obj, "forecast") if model_obj is not None else False

    # iterate
    for t in range(0, T - rolling_horizon, ROLLING_STEP):
        # realized returns window for horizon
        window = R[t+1:t+1+rolling_horizon, :]
        if window.shape[0] < MIN_OBS_REALIZED:
            # skip if not enough realized points to compute a meaningful realized cov
            continue
        realized = realized_covariance(window)

        # predicted covariance:
        pred_cov = None
        # 1) try forecast if model supplies a forecasting API (DCC may)
        if supports_forecast:
            try:
                fc = model_obj.forecast(horizon=1)  # many libs accept horizon argument
                if isinstance(fc, dict) and "covariance" in fc:
                    pred_cov = np.asarray(fc["covariance"][0])
                else:
                    arr = np.asarray(fc)
                    if arr.ndim == 3 and arr.shape[0] >= 1:
                        pred_cov = arr[0]
            except Exception:
                # forecasting with pre-trained models may require supplying most recent data;
                # we fallback silently to static_cov
                pred_cov = None

        # 2) fallback to static covariance (Ledoit-Wolf, GLASSO, or DCC final)
        if pred_cov is None:
            if static_cov is not None:
                pred_cov = static_cov
            else:
                # can't produce prediction
                continue

        # ensure PSD
        pred_cov = nearest_psd(pred_cov)
        realized = nearest_psd(realized)

        # compute metrics
        frob = frobenius_norm(pred_cov, realized)
        spec = spectral_norm(pred_cov, realized)
        kl = kl_divergence_gaussian(pred_cov, realized)

        # portfolio: long-only GMVP weights from pred_cov, realized variance over window
        try:
            w = solve_long_only_gmvp(pred_cov)
        except Exception as e:
            print(f"[WARN] GMVP solver failed at t={t}: {e}. Using equal weights.")
            w = np.ones(N) / N

        # compute realized portfolio returns over horizon and variance
        port_rets = (window @ w)  # shape H
        realized_port_var = float(np.var(port_rets, ddof=0))  # population variance over horizon
        results.append({
            "t_index": t,
            "frobenius": frob,
            "spectral": spec,
            "kl": kl,
            "realized_portfolio_variance": realized_port_var
        })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print(f"[WARN] Rolling validation produced no points for {model_name} (maybe validation set too short).")
    else:
        print(f"[ROLL] Completed rolling validation for {model_name}: {len(df_res)} steps.")
    return df_res

# ---------------------------
# Main pipeline
# ---------------------------
def validation():
    print("\n==============================================")
    print("Full validation pipeline (static + rolling)")
    print("==============================================\n")

    # 1) Load models
    models = load_trained_models()
    dcc_obj, _ = models["dcc"]
    lw_obj, _ = models["lw"]
    gl_obj, _ = models["gl"]

    # 2) Load validation returns
    val_wide, assets = load_validation_matrix(VALIDATION_CSV)
    tickers = assets
    N = len(tickers)
    print(f"[INFO] Number of tickers in validation set: {N}")

    # 3) Static validation: compute realized covariance from full validation sample
    # For each model produce a static predicted covariance and compare
    summaries = []
    for name, (obj, path) in [("DCC", (dcc_obj, DCC_PICKLE_PATHS)), ("LedoitWolf", (lw_obj, LW_MODEL_PATHS)), ("GraphicalLasso", (gl_obj, GL_MODEL_PATHS))]:
        print(f"\n[STATIC] Evaluating model: {name} ...")
        model_obj = obj
        if model_obj is None:
            print(f"[STATIC] Model {name} not available - skipping.")
            continue

        pred_cov = get_predicted_covariance(model_obj, name.lower(), tickers)
        if pred_cov is None:
            print(f"[STATIC] Could not extract covariance for {name}; skipping static metrics.")
            continue
        # Align pred_cov to tickers if shape mismatch: attempt to subset or reorder if matrix came from different set
        if pred_cov.shape[0] != N:
            print(f"[WARN] Predicted covariance shape {pred_cov.shape} does not match validation asset count {N}. Attempting to adapt...")
            # try to read CSV fallback with labels
            if isinstance(model_obj, dict) and "cov_csv" in model_obj:
                try:
                    df_model_cov = pd.read_csv(model_obj["cov_csv"], index_col=0)
                    common = [t for t in tickers if t in df_model_cov.index]
                    if len(common) == 0:
                        print("[ERROR] No overlap between model assets and validation tickers.")
                        continue
                    pred_cov = df_model_cov.loc[common, common].values
                    # subset validation accordingly
                    sub_val = val_wide.loc[:, common]
                    metrics = static_validation(pred_cov, sub_val)
                    metrics["model"] = name
                    metrics["n_assets_used"] = len(common)
                    summaries.append(metrics)
                    print(f"[STATIC] {name}: metrics on {len(common)} common assets saved.")
                    continue
                except Exception as e:
                    print(f"[WARN] Could not adapt: {e}. Skipping {name}.")
                    continue
            else:
                print(f"[WARN] No CSV labels available; skipping {name} for static metrics.")
                continue

        metrics = static_validation(pred_cov, val_wide)
        metrics["model"] = name
        metrics["n_assets_used"] = N
        summaries.append(metrics)
        print(f"[STATIC] {name} metrics: Fro={metrics['frobenius']:.6e}, Spec={metrics['spectral']:.6e}, KL={metrics['kl']:.6e}")

        # Save predicted covariance csv for reference
        try:
            df_cov = pd.DataFrame(pred_cov, index=tickers, columns=tickers)
            df_cov.to_csv(os.path.join(OUTDIR, f"{name.lower()}_pred_cov_static.csv"))
            print(f"[SAVE] {name} predicted covariance saved to CSV.")
        except Exception:
            pass

    # Save static summary
    static_df = pd.DataFrame(summaries)
    static_csv = os.path.join(OUTDIR, "static_validation_summary.csv")
    static_df.to_csv(static_csv, index=False)
    print(f"\n[SAVE] Static validation summary saved to: {static_csv}")

    # 4) Rolling validation
    rolling_results = {}
    for name, (obj, path) in [("DCC", (dcc_obj, DCC_PICKLE_PATHS)), ("LedoitWolf", (lw_obj, LW_MODEL_PATHS)), ("GraphicalLasso", (gl_obj, GL_MODEL_PATHS))]:
        if obj is None:
            print(f"[ROLL] Skipping rolling for {name} (model not available).")
            continue
        try:
            df_roll = rolling_validation_for_model(obj, name.lower(), val_wide, tickers, rolling_horizon=ROLLING_HORIZON)
            rolling_results[name] = df_roll
            if not df_roll.empty:
                # Save
                df_roll.to_csv(os.path.join(OUTDIR, f"rolling_{name.lower()}.csv"), index=False)
                print(f"[SAVE] Rolling results for {name} saved.")
        except Exception as e:
            print(f"[ERROR] Rolling validation failed for {name}: {e}")

    # 5) Aggregate rolling metrics into summary table (mean, std)
    rows = []
    for name, df_roll in rolling_results.items():
        if df_roll is None or df_roll.empty:
            continue
        rows.append({
            "model": name,
            "n_steps": len(df_roll),
            "fro_mean": df_roll["frobenius"].mean(),
            "fro_std": df_roll["frobenius"].std(),
            "spec_mean": df_roll["spectral"].mean(),
            "kl_mean": df_roll["kl"].mean(),
            "portvar_mean": df_roll["realized_portfolio_variance"].mean()
        })
    rolling_summary_df = pd.DataFrame(rows)
    rolling_summary_csv = os.path.join(OUTDIR, "rolling_validation_summary.csv")
    rolling_summary_df.to_csv(rolling_summary_csv, index=False)
    print(f"[SAVE] Rolling validation summary saved to: {rolling_summary_csv}")

    # 6) Plots: static comparison and rolling comparison
    print("[PLOT] Generating comparison plots...")

    # Static: bar chart of Frobenius for each model
    if not static_df.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(x="model", y="frobenius", data=static_df)
        plt.title("Static Validation: Frobenius norm (predicted vs realized)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "static_frobenius_comparison.png"))
        plt.close()

    # Rolling: timeseries of realized portfolio variance and frobenius per model
    for metric in ["frobenius", "realized_portfolio_variance"]:
        plt.figure(figsize=(10, 5))
        for name, df_roll in rolling_results.items():
            if df_roll is None or df_roll.empty:
                continue
            plt.plot(df_roll["t_index"], df_roll[metric], label=name)
        plt.xlabel("Rolling step index (t)")
        plt.ylabel(metric)
        plt.title(f"Rolling comparison: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"rolling_{metric}_comparison.png"))
        plt.close()

    # 7) Consolidated output
    final_summary = {
        "static_summary_csv": static_csv,
        "rolling_summary_csv": rolling_summary_csv,
        "rolling_detail_files": {name: os.path.join(OUTDIR, f"rolling_{name.lower()}.csv") for name in rolling_results.keys()},
        "plots_dir": PLOTS_DIR
    }
    meta_path = os.path.join(OUTDIR, "validation_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(final_summary, f)
    print(f"[SAVE] Validation meta saved: {meta_path}")

    print("\n[FINISHED] Full validation pipeline completed.")
    print(f" - Outputs: {OUTDIR} (plots: {PLOTS_DIR})")
    print(" - Static summary:", static_csv)
    print(" - Rolling summary:", rolling_summary_csv)

if __name__ == "__main__":
    validation()

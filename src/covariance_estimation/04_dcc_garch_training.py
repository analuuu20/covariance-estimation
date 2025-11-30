"""
dcc_training_only.py

Manual implementation of the two-step DCC-GARCH(1,1) estimator with MLE calibration 
(Engle, 2002). This script is dedicated solely to training the dynamic baseline model
and saving its parameters for subsequent out-of-sample evaluation.

Workflow:
1. Load and prepare a reproducible subset of returns (N=30 tickers).
2. Fit N univariate GARCH(1,1) models (to get conditional volatilities and standardized residuals).
3. Calibrate the DCC parameters (a, b) via constrained numerical optimization (MLE).
4. Save the estimated DCC parameters and the full time series of GARCH conditional volatilities.

Notes:
- Numerical stabilization (rescaling) is applied to returns to improve MLE convergence.
- The output is the full time-series of GARCH sigmas and the DCC parameters (a, b).
"""

from __future__ import annotations
import os
import warnings
import pickle
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from arch.univariate import ConstantMean, GARCH, Normal
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Config / hyperparameters
# ---------------------------
TRAIN_PATH = "data/train_returns.csv"   # Input panel (log-returns)
OUT_DIR = "tests"                       # Outputs directory
N_TICKERS = 30                          # Subset size for DCC tractability
RANDOM_STATE = 42                       # Reproducibility seed for ticker selection
EPS_STABILITY = 1e-6                    # Small jitter for positive definiteness (PD)
GARCH_PRINT_INTERVAL = 5                # Print progress every k assets
SCALING_FACTOR = 100                    # Numerical stabilization factor (rescales returns)


# ---------------------------
# Utility functions (Simplified for DCC only)
# ---------------------------

def ensure_outdir(outdir: str = OUT_DIR):
    os.makedirs(outdir, exist_ok=True)


def load_panel(path: str = TRAIN_PATH) -> pd.DataFrame:
    """Load training panel and return wide format."""
    print(f"[LOAD] Reading training CSV from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    # [ACADEMIC NOTE]: Rescaling applied for numerical stability of MLE optimizer.
    df['LogReturn'] *= SCALING_FACTOR
    
    print(f"[LOAD] Panel loaded: {df['Ticker'].nunique()} unique tickers.")
    return df


def select_tickers(df: pd.DataFrame, n: int = N_TICKERS, seed: int = RANDOM_STATE) -> List[str]:
    """Select n tickers at random using a fixed seed for reproducibility."""
    np.random.seed(seed)
    unique = np.sort(df["Ticker"].unique())
    if n >= len(unique):
        return list(unique)
    chosen = list(np.random.choice(unique, size=n, replace=False))
    print(f"[SELECT] Selected {n} tickers (seed={seed}). Example slice: {chosen[:6]}...")
    return chosen


def pivot_and_sync(df: pd.DataFrame, tickers: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Build synchronized wide returns matrix for the chosen subset."""
    print("[PREP] Pivoting to wide format and synchronizing dates...")
    sub = df[df["Ticker"].isin(tickers)].copy()
    wide = sub.pivot(index="Date", columns="Ticker", values="LogReturn")
    wide = wide.loc[:, tickers]     
    wide = wide.dropna(axis=0, how="any") # Drop any date with missing data across the subset
    print(f"[PREP] Final synchronized panel dimensions: {wide.shape[0]} dates, {wide.shape[1]} tickers.")
    X = wide.values                 
    tickers_order = list(wide.columns)
    return wide, X, tickers_order


# ---------------------------
# GARCH estimation (univariate) - STEP 1
# ---------------------------

def fit_garch_for_series(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fit a ConstantMean + GARCH(1,1) model for one series."""
    # Fit quietly
    am = ConstantMean(series)
    am.volatility = GARCH(p=1, o=0, q=1)
    am.distribution = Normal()
    res = am.fit(disp="off")
    
    sigma = res.conditional_volatility
    resid = res.resid
    std_resid = resid / sigma

    params = {
        "omega": float(res.params.get("omega", np.nan)),
        "alpha[1]": float(res.params.get("alpha[1]", np.nan)),
        "beta[1]": float(res.params.get("beta[1]", np.nan))
    }
    return sigma.values, std_resid.values, params


def fit_garch_panel(wide: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit GARCH(1,1) for each column of the wide DataFrame (Parallel fitting)."""
    T, N = wide.shape
    print(f"\n[GARCH - STEP 1] Fitting N={N} univariate GARCH(1,1) models...")
    sigmas = np.zeros((T, N))
    std_resids = np.zeros((T, N))
    rows = []
    
    for i, col in enumerate(wide.columns):
        if (i % GARCH_PRINT_INTERVAL) == 0:
            print(f"[GARCH] Fitting asset {i+1}/{N}: {col} ...")
        
        s, std, params = fit_garch_for_series(wide[col])
        sigmas[:, i] = s
        std_resids[:, i] = std
        row = {"Ticker": col, **params}
        rows.append(row)
        
    params_df = pd.DataFrame(rows)
    print("[GARCH - STEP 1] Completed fitting all univariate GARCH models.")
    return sigmas, std_resids, params_df


# ---------------------------
# DCC likelihood and estimation - STEP 2
# ---------------------------

def ensure_pd(mat: np.ndarray, eps: float = EPS_STABILITY) -> np.ndarray:
    """Ensure matrix is positive definite (PD) by adding small jitter on diagonal."""
    mat = 0.5 * (mat + mat.T) # Symmetrize
    jitter = 0.0
    try:
        np.linalg.cholesky(mat)
        return mat
    except np.linalg.LinAlgError:
        jitter = eps
        max_tries = 10
        for _ in range(max_tries):
            try:
                mat_j = mat + np.eye(mat.shape[0]) * jitter
                np.linalg.cholesky(mat_j)
                return mat_j
            except np.linalg.LinAlgError:
                jitter *= 10
        raise np.linalg.LinAlgError("Matrix not positive-definite even after jitter.")


def dcc_neg_loglik(params: np.ndarray, eps: np.ndarray) -> float:
    """Negative log-likelihood for DCC(1,1) based on standardized residuals."""
    a, b = params
    # Constraints check (stationarity constraint: a>=0, b>=0, a+b<1)
    if a < 0 or b < 0 or (a + b) >= 0.999:
        return 1e12

    T, N = eps.shape
    S = np.cov(eps, rowvar=False)  # Unconditional covariance of standardized resid
    Q = S.copy()
    nll = 0.0

    # Main DCC recursion loop
    for t in range(T):
        e = eps[t][:, None]
        Q = (1.0 - a - b) * S + a * (e @ e.T) + b * Q # Q_t update
        
        # Form correlation R_t
        diagQ = np.sqrt(np.diag(Q))
        diagQ = np.where(diagQ <= 0, EPS_STABILITY, diagQ)
        Dinv = np.diag(1.0 / diagQ)
        R = Dinv @ Q @ Dinv
        
        try:
            R = ensure_pd(R)
        except np.linalg.LinAlgError:
            return 1e12 # Heavy penalty

        # Compute log-likelihood contribution
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e12 # Heavy penalty
        invR = np.linalg.inv(R)
        quad = float(eps[t].T @ invR @ eps[t])
        nll += 0.5 * (logdet + quad)

    return nll


def estimate_dcc_params_mle(std_resids: np.ndarray, init: Tuple[float, float] = (0.01, 0.97)) -> Tuple[float, float, Dict[str, Any]]:
    """Estimate (a,b) by minimizing the negative log-likelihood (MLE)."""
    print("\n[DCC - STEP 2] Beginning MLE for (a,b) correlation parameters...")
    
    def callback(xk):
        # Progress print during optimization
        print(f"[DCC-MLE] Iteration update: a={xk[0]:.6f}, b={xk[1]:.6f}")

    # Bounds: a and b must be non-negative and sum to less than 1
    bounds = [(1e-8, 0.9999), (1e-8, 0.9999)] 
    res = minimize(lambda p: dcc_neg_loglik(p, std_resids),
                   x0=np.array(init),
                   method="L-BFGS-B",
                   bounds=bounds,
                   callback=callback,
                   options={"disp": False, "maxiter": 500})
    
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    
    if not res.success:
        print(f"[DCC-MLE] WARNING: Optimizer failed to converge: {res.message}")
    
    print(f"\n[DCC-MLE] Finished. Estimated: a={a_hat:.6f}, b={b_hat:.6f}. Log-Likelihood: {-res.fun:.2f}")
    metadata = {"opt_success": res.success, "opt_message": res.message, "fun": float(res.fun), "nit": res.nit}
    return a_hat, b_hat, metadata


# ---------------------------
# Reconstruction and Save
# ---------------------------

def compute_dcc_time_series(std_resids: np.ndarray,
                            sigmas: np.ndarray,
                            a: float,
                            b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the full time series of DCC covariances Σ_t and correlations R_t."""
    T, N = std_resids.shape
    S = np.cov(std_resids, rowvar=False)
    Q = S.copy()

    Sigma_ts = np.zeros((T, N, N))
    R_ts = np.zeros((T, N, N))

    for t in range(T):
        e = std_resids[t][:, None]
        Q = (1.0 - a - b) * S + a * (e @ e.T) + b * Q
        
        diagQ = np.sqrt(np.diag(Q))
        diagQ = np.where(diagQ <= 0, EPS_STABILITY, diagQ)
        Dinv = np.diag(1.0 / diagQ)
        R_t = Dinv @ Q @ Dinv                  
        R_ts[t] = ensure_pd(R_t)

        D_t = np.diag(sigmas[t])
        Sigma_ts[t] = ensure_pd(D_t @ R_t @ D_t)

    return Sigma_ts, R_ts


def save_dcc_results(model_obj: Dict[str, Any], outdir: str, SCALING_FACTOR: int, tickers_order: List[str]):
    """Saves the final DCC object and summary files."""    
    
    Sigma_T_scaled = model_obj['Sigma_ts'][-1]
    
    # 1. Descale the final covariance matrix (Sigma_T) for absolute interpretation
    Sigma_T = Sigma_T_scaled / (SCALING_FACTOR ** 2)
    
    # Update object with descaled matrices
    model_obj['Sigma_T_descaled'] = Sigma_T
    model_obj['Sigma_ts_descaled'] = model_obj['Sigma_ts'] / (SCALING_FACTOR ** 2)
    
    # 2. Save full model object (PICKLE)
    pickle_path = os.path.join(outdir, "dcc_garch_model.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"\n[SAVE SUCCESS] Pickled DCC model object to: {pickle_path}")

    # 3. Save final covariance matrix (CSV)
    cov_df = pd.DataFrame(Sigma_T, index=tickers_order, columns=tickers_order)
    cov_csv = os.path.join(outdir, "dcc_garch_covariance_final.csv")
    cov_df.to_csv(cov_csv)
    print(f"[SAVE SUCCESS] Saved descaled covariance matrix CSV to: {cov_csv}")


# ---------------------------
# Top-level DCC training pipeline
# ---------------------------

def train_dcc_pipeline(train_csv: str = TRAIN_PATH,
                       outdir: str = OUT_DIR,
                       n_tickers: int = N_TICKERS,
                       seed: int = RANDOM_STATE) -> Dict[str, Any]:
    
    ensure_outdir(outdir)
    
    # 1. Data Prep and Subset Selection
    df = load_panel(train_csv)
    chosen_tickers = select_tickers(df, n=n_tickers, seed=seed)
    wide, X, tickers_order = pivot_and_sync(df, chosen_tickers)

    # 2. Fit Univariate GARCH models (Conditional Volatilities)
    sigmas, std_resids, params_df = fit_garch_panel(wide)

    # Handle numerical issues (from original code)
    if not np.isfinite(std_resids).all():
        warnings.warn("Found non-finite standardized residuals; replacing with zeros (conservative).")
        std_resids = np.nan_to_num(std_resids, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Estimate DCC Parameters (MLE)
    a_hat, b_hat, opt_meta = estimate_dcc_params_mle(std_resids, init=(0.05, 0.90))

    # 4. Compute Full Time Series of Covariance Matrices
    Sigma_ts, R_ts = compute_dcc_time_series(std_resids, sigmas, a_hat, b_hat)

    # 5. Build Final Model Object
    model_obj = {
        "DCC_a": a_hat,
        "DCC_b": b_hat,
        "tickers": tickers_order,
        "garch_params": params_df,
        "Sigma_ts": Sigma_ts,
        "R_ts": R_ts,
        "opt_meta": opt_meta
    }
    
    # 6. Save Descaled Results and Object
    save_dcc_results(model_obj, outdir, SCALING_FACTOR, tickers_order)

    # ------------------------------------------------
    # 7. CALCULAR LOG-LIKELIHOOD DESESCALADO
    # ------------------------------------------------
    
    # [ACADEMIC ADJUSTMENT] Descale the Log-Likelihood (LLF) for absolute interpretation.
    # LLF_original = LLF_scaled - (T * N * ln(c))
    
    # Parámetros del ajuste
    T_obs = X.shape[0]                  # 712
    N_assets = len(tickers_order)       # 30
    log_c = np.log(SCALING_FACTOR)      # ln(100) approx 4.605
    
    # Valor de LLF negativo obtenido de la optimización (el valor 'fun' es la NLLF)
    nll_scaled = opt_meta.get("fun", 0.0) 
    
    # Calcular el Log-Likelihood (positivo) escalado
    llf_scaled = -nll_scaled 
    
    # Calcular el factor de ajuste total
    adjustment_factor = T_obs * N_assets * log_c
    
    # Calcular el Log-Likelihood desescalado
    llf_descaled = llf_scaled - adjustment_factor
    
    print(f"\n[INFO] Descaling Log-Likelihood: {llf_scaled:.2f} (Scaled) -> {llf_descaled:.2f} (Absolute)")

    # 8. Generate Summary Output
    summary = pd.DataFrame([
        {"Parameter": "DCC_a", "Value": a_hat},
        {"Parameter": "DCC_b", "Value": b_hat},
        {"Parameter": "n_tickers", "Value": len(tickers_order)},
        {"Parameter": "T_obs", "Value": X.shape[0]},
        {"Parameter": "Log-Likelihood (Descaled)", "Value": llf_descaled},
        {"Parameter": "Log-Likelihood (Scaled)", "Value": llf_scaled}, # Mantener el valor escalado para referencia
        {"Parameter": "opt_success", "Value": opt_meta.get("opt_success", False)}, 
        {"Parameter": "opt_message", "Value": opt_meta.get("opt_message", "")}
    ])
    
    print("\n[DCC SUMMARY TABLE (CALIBRATION)]")
    print(summary.to_string(index=False))

    return model_obj


# ---------------------------
# Top-level execution
# ---------------------------
if __name__ == "__main__":
    
    print("\n=======================================================")
    print("STARTING DCC-GARCH TRAINING PIPELINE (MANUAL MLE)")
    print("=======================================================")
    
    # The pipeline is now fully self-contained, including rescaling and saving descaled results.
    dcc_results = train_dcc_pipeline()
    
    print("\n[DONE] DCC-GARCH training finished. Model object and matrices saved to 'tests/'.")



 
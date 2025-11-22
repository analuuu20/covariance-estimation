"""
dcc_manual.py

Manual implementation of DCC-GARCH(1,1) estimator with MLE calibration
(Engle, 2002). This script performs the following steps:

1) Load training returns from /data/train_returns.csv (must contain columns: Date, Ticker, LogReturn).
2) Select a reproducible random subset of 30 tickers (random_state=42).
3) Create a synchronized wide matrix (T x N) by pivoting and dropping rows with missing values.
4) Fit univariate GARCH(1,1) for each selected ticker using `arch.univariate` (ConstantMean + GARCH).
   - Extract conditional volatilities σ_{i,t} and standardized residuals ε_{i,t}.
5) Define the DCC(1,1) recursion:
       Q_t = (1 - a - b) Q̄ + a ε_{t-1} ε'_{t-1} + b Q_{t-1},
   convert to correlations R_t and evaluate the Gaussian log-likelihood:
       L(a,b) = 0.5 * sum_t [ ln |R_t| + ε_t' R_t^{-1} ε_t ]  (+ const)
   We maximize the likelihood (equivalently minimize negative log-likelihood).
6) Calibrate (a,b) via constrained numerical optimization (L-BFGS-B with constraints enforced via penalty).
7) Construct the final covariance matrix Σ_T = D_T R_T D_T where D_T = diag(σ_{i,T}).
8) Save model outputs (pickle + CSV) in /tests and print a table of estimated parameters.

Notes:
- We purposely use only 30 tickers (seeded selection) for tractability of DCC estimation.
- The univariate GARCH step is independent across tickers (two-step estimation is standard:
  first fit marginal volatilities, then fit the correlation dynamics).
- The implementation includes numerical stabilizers (small jitter) to ensure positive-definiteness
  and robust inversion when computing likelihoods.
"""

from __future__ import annotations
import os
import warnings
import pickle
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import dump
from arch.univariate import ConstantMean, GARCH, Normal

# ---------------------------
# Config / hyperparameters
# ---------------------------
TRAIN_PATH = "data/train_returns.csv"   # input panel
OUT_DIR = "tests"                        # outputs (pickles and cov csv)
N_TICKERS = 30                           # subset for DCC baseline
RANDOM_STATE = 42                        # reproducibility seed for ticker selection
EPS_STABILITY = 1e-6                     # small jitter for PD-ness
GARCH_PRINT_INTERVAL = 5                 # print progress every k assets



# ---------------------------
# Utility functions
# ---------------------------

def ensure_outdir(outdir: str = OUT_DIR):
    os.makedirs(outdir, exist_ok=True)


def load_panel(path: str = TRAIN_PATH) -> pd.DataFrame:
    """
    Load training panel and perform basic validation.
    Expected columns: Date, Ticker, LogReturn
    """
    print(f"[LOAD] Reading training CSV from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    req = {"Date", "Ticker", "LogReturn"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns {req}. Found: {list(df.columns)}")
    print(f"[LOAD] Panel loaded: {df['Ticker'].nunique()} unique tickers, {df['Date'].nunique()} unique dates.")
    return df


def select_tickers(df: pd.DataFrame, n: int = N_TICKERS, seed: int = RANDOM_STATE) -> List[str]:
    """
    Select n tickers at random using a fixed seed for reproducibility.
    """
    np.random.seed(seed)
    unique = np.sort(df["Ticker"].unique())
    if n >= len(unique):
        print(f"[SELECT] Requested {n} tickers but only {len(unique)} available. Using all tickers.")
        return list(unique)
    chosen = list(np.random.choice(unique, size=n, replace=False))
    print(f"[SELECT] Selected {n} tickers (seed={seed}). Example slice: {chosen[:6]}...")
    return chosen


def pivot_and_sync(df: pd.DataFrame, tickers: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build synchronized wide returns matrix for the chosen tickers.
    Drops dates with any NA across the chosen subset to obtain a synchronous panel.
    Returns:
      - wide_df: DataFrame (index=Date, columns=tickers)
      - X: numpy array (T x N)
      - tickers_order: list of tickers in column order
    """
    print("[PREP] Pivoting to wide format and synchronizing dates (dropna across subset)...")
    sub = df[df["Ticker"].isin(tickers)].copy()
    wide = sub.pivot(index="Date", columns="Ticker", values="LogReturn")
    before_dates = wide.shape[0]
    wide = wide.loc[:, tickers]   # preserve chosen order
    wide = wide.dropna(axis=0, how="any")
    after_dates = wide.shape[0]
    print(f"[PREP] Dates before sync: {before_dates}, after dropna: {after_dates}. Using {wide.shape[1]} tickers.")
    X = wide.values  # T x N
    tickers_order = list(wide.columns)
    return wide, X, tickers_order


# ---------------------------
# GARCH estimation (univariate)
# ---------------------------

def fit_garch_for_series(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Fit a ConstantMean + GARCH(1,1) model for one series using `arch`.
    Returns:
      - sigma: array (T,) conditional volatilities (aligned with input series index)
      - std_resid: array (T,) standardized residuals = resid / sigma
      - params: dict with fitted parameters (omega, alpha, beta)
    """
    # Create model: Constant mean + GARCH(1,1), Normal innovations
    am = ConstantMean(series)
    am.volatility = GARCH(p=1, o=0, q=1)
    am.distribution = Normal()

    # Fit quietly
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
    """
    Fit GARCH(1,1) for each column of the wide DataFrame.
    Returns:
      - sigmas: ndarray (T x N) conditional volatilities
      - std_resids: ndarray (T x N) standardized residuals
      - params_df: DataFrame with per-asset GARCH params
    """
    T, N = wide.shape
    print(f"[GARCH] Fitting univariate GARCH(1,1) for N={N} assets over T={T} dates.")
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
    print("[GARCH] Completed fitting all univariate GARCH models.")
    return sigmas, std_resids, params_df


# ---------------------------
# DCC likelihood and estimation
# ---------------------------

def ensure_pd(mat: np.ndarray, eps: float = EPS_STABILITY) -> np.ndarray:
    """
    Ensure matrix is positive definite by adding small jitter on diagonal if needed.
    """
    # Symmetrize
    mat = 0.5 * (mat + mat.T)
    # Add jitter until PD
    jitter = 0.0
    try:
        # quick check via Cholesky
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
        # if still not PD, raise
        raise np.linalg.LinAlgError("Matrix not positive-definite even after jitter.")


def dcc_neg_loglik(params: np.ndarray, eps: np.ndarray) -> float:
    """
    Negative log-likelihood for DCC(1,1) based on standardized residuals eps (T x N).

    We use the Gaussian conditional log-likelihood (up to additive constants):
      L = 0.5 * sum_t [ ln |R_t| + eps_t' R_t^{-1} eps_t ].

    To enforce stationarity constraint a>=0,b>=0,a+b<1 we return a large penalty
    when constraints are violated.
    """
    a, b = params
    # constraints: a >= 0, b >= 0, a + b < 1
    if a < 0 or b < 0 or (a + b) >= 0.999999:
        return 1e12

    T, N = eps.shape
    S = np.cov(eps, rowvar=False)  # unconditional covariance of standardized resid
    Q = S.copy()
    nll = 0.0

    # iterate time to build Q_t and compute correlation R_t
    for t in range(T):
        # update Q: note here eps[t] is 1D array of length N
        e = eps[t][:, None]   # column vector
        Q = (1.0 - a - b) * S + a * (e @ e.T) + b * Q

        # form correlation R_t from Q (R_t = diag(Q)^{-1/2} Q diag(Q)^{-1/2})
        diagQ = np.sqrt(np.diag(Q))
        # if any diagQ is zero, add tiny jitter
        if np.any(diagQ <= 0):
            diagQ = np.where(diagQ <= 0, EPS_STABILITY, diagQ)
        Dinv = np.diag(1.0 / diagQ)
        R = Dinv @ Q @ Dinv

        # numerical stabilization
        try:
            R = ensure_pd(R)
        except np.linalg.LinAlgError:
            # heavy penalty if non-PD
            return 1e12

        # compute logdet and quadratic form
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e12
        invR = np.linalg.inv(R)
        quad = float(eps[t].T @ invR @ eps[t])
        nll += 0.5 * (logdet + quad)

    return nll  # minimized by optimizer


def estimate_dcc_params_mle(std_resids: np.ndarray, init: Tuple[float, float] = (0.01, 0.97)) -> Tuple[float, float, Dict[str, Any]]:
    """
    Estimate (a,b) by minimizing the negative log-likelihood.
    Returns estimated (a_hat, b_hat) and optimization metadata.
    """
    print("[DCC-MLE] Beginning MLE for (a,b) with initial guess:", init)

    def callback(xk):
        # callback called every iteration; print progress
        print(f"[DCC-MLE] iter params: a={xk[0]:.6f}, b={xk[1]:.6f}")

    # bounds to keep parameters in [1e-6, 0.999]
    bounds = [(1e-8, 0.9999), (1e-8, 0.9999)]
    res = minimize(lambda p: dcc_neg_loglik(p, std_resids),
                   x0=np.array(init),
                   method="L-BFGS-B",
                   bounds=bounds,
                   callback=callback,
                   options={"disp": False, "maxiter": 200})
    if not res.success:
        print("[DCC-MLE] WARNING: optimizer did not converge:", res.message)
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    print(f"[DCC-MLE] Finished. Estimated: a={a_hat:.6f}, b={b_hat:.6f}. Optimizer message: {res.message}")
    metadata = {"opt_success": res.success, "opt_message": res.message, "fun": float(res.fun), "nit": res.nit}
    return a_hat, b_hat, metadata


# ---------------------------
# Reconstruct final covariance and save
# ---------------------------

def build_final_covariance(std_resids: np.ndarray, sigmas: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Compute final (time-averaged or last-period) DCC covariance:
      - run the DCC recursion to produce Q_T and hence R_T
      - use last conditional sigmas to form Σ_T = D_T R_T D_T
    Here we return Σ_T (N x N).
    """
    T, N = std_resids.shape
    S = np.cov(std_resids, rowvar=False)
    Q = S.copy()
    # run recursion up to T
    for t in range(T):
        e = std_resids[t][:, None]
        Q = (1.0 - a - b) * S + a * (e @ e.T) + b * Q

    # correlation
    diagQ = np.sqrt(np.diag(Q))
    diagQ = np.where(diagQ <= 0, EPS_STABILITY, diagQ)
    Dinv = np.diag(1.0 / diagQ)
    R_T = Dinv @ Q @ Dinv
    R_T = ensure_pd(R_T)

    # last conditional sigmas: use the last row of sigmas
    last_sigmas = sigmas[-1]
    D_last = np.diag(last_sigmas)
    Sigma_T = D_last @ R_T @ D_last
    Sigma_T = ensure_pd(Sigma_T)
    return Sigma_T, R_T


# ---------------------------
# Top-level pipeline
# ---------------------------

def train_dcc_pipeline(train_csv: str = TRAIN_PATH,
                       outdir: str = OUT_DIR,
                       n_tickers: int = N_TICKERS,
                       seed: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Run the full DCC-GARCH training pipeline:
     - load panel
     - select tickers
     - pivot & sync
     - fit GARCH univariates
     - estimate DCC (a,b) by MLE
     - build final covariance Σ_T
     - save pickle + covariance CSV + params table
    Returns a dictionary with metadata and filepaths.
    """
    ensure_outdir(outdir)
    df = load_panel(train_csv)
    chosen = select_tickers(df, n=n_tickers, seed=seed)
    wide, X, tickers_order = pivot_and_sync(df, chosen)

    # Fit univariate GARCH on the synchronized wide panel (T x N)
    sigmas, std_resids, params_df = fit_garch_panel(wide)

    # Some standardized residuals may contain nan due to numerical issues; ensure finite
    if not np.isfinite(std_resids).all():
        print("[WARN] Found non-finite standardized residuals; replacing inf/NaN with zeros (conservative).")
        std_resids = np.nan_to_num(std_resids, nan=0.0, posinf=0.0, neginf=0.0)

    # Estimate DCC parameters by MLE
    a_hat, b_hat, opt_meta = estimate_dcc_params_mle(std_resids, init=(0.01, 0.97))

    # Build final covariance (Σ_T) and correlation R_T
    Sigma_T, R_T = build_final_covariance(std_resids, sigmas, a_hat, b_hat)

    # Save results to outdir
    ensure_outdir(outdir)
    meta = {
        "tickers": tickers_order,
        "a": a_hat,
        "b": b_hat,
        "opt_meta": opt_meta
    }

    # Save pickle of metadata and arrays (avoid ultra-large objects)
    model_obj = {"meta": meta, "Sigma_T": Sigma_T, "R_T": R_T, "garch_params": params_df}
    pickle_path = os.path.join(outdir, "dcc_garch_model.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"[SAVE] Pickled model metadata to: {pickle_path}")

    # Save covariance CSV with tickers as columns/rows
    cov_df = pd.DataFrame(Sigma_T, index=tickers_order, columns=tickers_order)
    cov_csv = os.path.join(outdir, "dcc_garch_covariance.csv")
    cov_df.to_csv(cov_csv)
    print(f"[SAVE] Saved DCC covariance matrix CSV to: {cov_csv}")

    # Save garch params for each asset (csv)
    garch_params_csv = os.path.join(outdir, "dcc_univariate_garch_params.csv")
    params_df.to_csv(garch_params_csv, index=False)
    print(f"[SAVE] Saved univariate GARCH parameters to: {garch_params_csv}")

    # Print final summary table
    summary = pd.DataFrame([
        {"Parameter": "DCC_a", "Value": a_hat},
        {"Parameter": "DCC_b", "Value": b_hat},
        {"Parameter": "n_tickers", "Value": len(tickers_order)},
        {"Parameter": "T_obs", "Value": X.shape[0]},
        {"Parameter": "opt_success", "Value": opt_meta.get("opt_success", False)},
        {"Parameter": "opt_message", "Value": opt_meta.get("opt_message", "")}
    ])
    print("\n[DCC SUMMARY TABLE]")
    print(summary.to_string(index=False))

    return {"meta": meta, "pickle": pickle_path, "cov_csv": cov_csv, "garch_params_csv": garch_params_csv}


# ---------------------------
# If executed as script
# ---------------------------
if __name__ == "__main__":
    # Run the pipeline and capture outputs
    results = train_dcc_pipeline()
    print("\n[DONE] DCC-GARCH training pipeline finished. Results stored in 'tests/'.")


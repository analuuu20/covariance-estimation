"""
covariance_training.py

This module estimates covariance matrices from the TRAINING RETURNS dataset
using multiple econometric and statistical models:

    1. Dynamic Conditional Correlation GARCH (DCC-GARCH) – Baseline
    2. Ledoit–Wolf shrinkage estimator
    3. Oracle Approximating Shrinkage (OAS)
    4. Graphical Lasso (GLASSO)
    5. PCA-based covariance estimator
    6. Raw Sample Covariance (benchmark)

The resulting calibrated models and matrices are saved to:

    tests/dcc_model.pkl
    tests/lw_cov.pkl
    tests/oas_cov.pkl
    tests/glasso_model.pkl
    tests/pca_model.pkl
    tests/raw_cov.csv

Plots illustrating model behavior, convergence, or structural properties
are saved in:

    tests/plots/

All code is written with academic-style explanations.
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf, OAS, GraphicalLasso
from sklearn.decomposition import PCA

from arch.univariate import GARCH as UnivariateGARCH, ConstantMean
from arch.multivariate import DCC, GARCH as MultivariateGARCH
import arch.unitroot


plt.style.use("seaborn-v0_8-whitegrid")   # professional visual style


# ========================================================
# 1. Load Training Returns
# ========================================================

def load_training_returns(path="data/train_returns.csv"):
    """
    Loads the training returns dataset, ensuring correct structure.
    """
    print(f"[STEP] Loading train returns from {path} ...")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


# ========================================================
# 2. DCC-GARCH Estimation
# ========================================================

def estimate_dcc_garch(df, save_path_model="tests/dcc_model.pkl"):
    """
    Estimates a DCC-GARCH(1,1) model across all tickers.

    Notes:
    ------
    DCC-GARCH models allow **time-varying covariance**, capturing
    the dynamics of volatility clustering (Engle, 2002).
    """

    print("\n[MODEL] Estimating DCC-GARCH baseline...")

    returns_wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").dropna()
    T, N = returns_wide.shape

    series_list = []
    for col in returns_wide.columns:
        m = ConstantMean(returns_wide[col])
        m.volatility = GARCH(p=1, o=0, q=1)
        res = m.fit(disp="off")
        series_list.append(res)

    dcc = DCC(series_list)
    dcc_res = dcc.fit()

    # Save
    os.makedirs("tests", exist_ok=True)
    with open(save_path_model, "wb") as f:
        pickle.dump(dcc_res, f)

    print("[SUCCESS] DCC-GARCH model saved.")

    # Plot log-likelihood
    plt.figure(figsize=(10, 4))
    plt.plot(dcc_res.loglikelihoods, linewidth=2)
    plt.title("DCC-GARCH Log-Likelihood Convergence", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.tight_layout()
    plt.savefig("tests/plots/dcc_loglikelihood.png")
    plt.close()

    return dcc_res


# ========================================================
# 3. Classical Covariance Estimators
# ========================================================

def estimate_shrinkage_models(df):
    """
    Computes Ledoit–Wolf, OAS, GLASSO, PCA, and raw covariance matrices,
    including visual diagnostics and saving the trained models.
    """

    returns_wide = df.pivot(index="Date", columns="Ticker", values="LogReturn").dropna()
    R = returns_wide.values

    os.makedirs("tests", exist_ok=True)
    os.makedirs("tests/plots", exist_ok=True)

    results = {}

    # ──────────────────────────────────────────────────────────────
    # Raw Covariance
    # ──────────────────────────────────────────────────────────────
    raw_cov = np.cov(R, rowvar=False)
    pd.DataFrame(raw_cov).to_csv("tests/raw_cov.csv", index=False)
    results["Raw"] = raw_cov

    # Plot Condition Number
    plt.figure(figsize=(7, 4))
    plt.title("Condition Number – Raw Covariance")
    plt.bar(["Raw Cov"], [np.linalg.cond(raw_cov)])
    plt.ylabel("Condition Number")
    plt.tight_layout()
    plt.savefig("tests/plots/raw_condition.png")
    plt.close()

    # ──────────────────────────────────────────────────────────────
    # Ledoit–Wolf
    # ──────────────────────────────────────────────────────────────
    lw = LedoitWolf().fit(R)
    results["LW"] = lw.covariance_

    with open("tests/lw_cov.pkl", "wb") as f:
        pickle.dump(lw, f)

    # Plot shrinkage
    plt.figure(figsize=(7, 4))
    plt.title(f"Ledoit–Wolf Shrinkage δ = {lw.shrinkage_:.4f}")
    plt.bar(["Shrinkage"], [lw.shrinkage_])
    plt.ylabel("δ")
    plt.tight_layout()
    plt.savefig("tests/plots/lw_shrinkage.png")
    plt.close()

    # ──────────────────────────────────────────────────────────────
    # OAS
    # ──────────────────────────────────────────────────────────────
    oas = OAS().fit(R)
    results["OAS"] = oas.covariance_

    with open("tests/oas_cov.pkl", "wb") as f:
        pickle.dump(oas, f)

    plt.figure(figsize=(7, 4))
    plt.title(f"OAS Shrinkage δ = {oas.shrinkage_:.4f}")
    plt.bar(["Shrinkage"], [oas.shrinkage_])
    plt.ylabel("δ")
    plt.tight_layout()
    plt.savefig("tests/plots/oas_shrinkage.png")
    plt.close()

    # ──────────────────────────────────────────────────────────────
    # GLASSO
    # ──────────────────────────────────────────────────────────────
    glasso = GraphicalLasso(alpha=0.01).fit(R)
    results["GLASSO"] = glasso.covariance_

    with open("tests/glasso_model.pkl", "wb") as f:
        pickle.dump(glasso, f)

    plt.figure(figsize=(7, 4))
    plt.title("GLASSO – Sparsity of Precision Matrix")
    plt.imshow(glasso.precision_ != 0, cmap="binary", aspect="auto")
    plt.colorbar(label="Non-zero")
    plt.tight_layout()
    plt.savefig("tests/plots/glasso_sparsity.png")
    plt.close()

    # ──────────────────────────────────────────────────────────────
    # PCA
    # ──────────────────────────────────────────────────────────────
    pca = PCA().fit(R)
    results["PCA"] = pca.get_covariance()

    with open("tests/pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    # Plot variance explained
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(0.9, color="red", linestyle="--")
    plt.title("PCA – Cumulative Variance Explained")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Variance")
    plt.tight_layout()
    plt.savefig("tests/plots/pca_variance.png")
    plt.close()

    return results


# ========================================================
# 4. Comparison Table
# ========================================================
"""
def generate_training_summary(results, dcc_model):
    """
    Creates a summary table for the training-set performance and
    calibrated parameters of each covariance model.
    """

    rows = []

    rows.append(["Raw Covariance",
                 np.linalg.cond(results["Raw"]),
                 "N/A",
                 "Static benchmark"])

    rows.append(["Ledoit–Wolf",
                 np.linalg.cond(results["LW"]),
                 "δ = {:.4f}".format(LedoitWolf().fit.__defaults__),
                 "Shrinkage estimator"])

    rows.append(["OAS",
                 np.linalg.cond(results["OAS"]),
                 "δ = {:.4f}".format(OAS().fit.__defaults__),
                 "Improved shrinkage under Gaussianity"])

    rows.append(["GLASSO",
                 np.linalg.cond(results["GLASSO"]),
                 "α = 0.01",
                 "Sparse precision matrix"])

    rows.append(["PCA",
                 np.linalg.cond(results["PCA"]),
                 "Explains 90% variance",
                 "Factor model"])

    rows.append(["DCC-GARCH",
                 "Time-varying",
                 "log-likelihood = {:.2f}".format(np.sum(dcc_model.loglikelihoods)),
                 "Dynamic volatility & correlation"])

    summary = pd.DataFrame(rows, columns=[
        "Model", "Condition / Stability", "Key Parameter", "Interpretation"
    ])

    summary.to_csv("tests/training_summary.csv", index=False)
    print("\n[SUMMARY] Training results saved to tests/training_summary.csv")

    return summary


# ========================================================
# MAIN PIPELINE
# ========================================================

if __name__ == "__main__":

    df = load_training_returns()

    dcc_model = estimate_dcc_garch(df)

    results = estimate_shrinkage_models(df)

    summary = generate_training_summary(results, dcc_model)

    print("\nPipeline complete.")
"""
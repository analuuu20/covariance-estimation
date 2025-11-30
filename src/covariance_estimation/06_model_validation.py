"""
Model Validation Script
-----------------------
This script loads validation return data and evaluates the performance of
previously trained covariance models:
- DCC-GARCH
- Ledoit-Wolf
- OAS
- PCA-based covariance
- Graphical LASSO

It produces:
- Intermediate prints for workflow transparency
- Academic-style English comments
- Summary table comparing validation metrics
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ------------------------------------------------------------
# 1. Load Validation Returns
# ------------------------------------------------------------
print("[INFO] Loading validation returns ...")
val_path = Path("data/validation_returns.csv")
validation_returns = pd.read_csv(val_path, index_col=0, parse_dates=True)
validation_returns = validation_returns.sort_index()
print("[INFO] Validation data shape:", validation_returns.shape)

# ------------------------------------------------------------
# 2. Load Trained Models
# ------------------------------------------------------------
print("[INFO] Loading trained models from disk ...")
models = {}

for name in ["dcc_garch", "ledoit_wolf", "oas", "pca", "glasso"]:
    file = Path(f"models/{name}_model.pkl")
    if file.exists():
        with open(file, "rb") as f:
            models[name] = pickle.load(f)
        print(f"[OK] Loaded {name} model.")
    else:
        print(f"[WARNING] Model file missing: {name}")

# ------------------------------------------------------------
# 3. Define Performance Metrics
# ------------------------------------------------------------
def mse(true_cov, est_cov):
    return np.mean((true_cov - est_cov)**2)

def frobenius(true_cov, est_cov):
    return np.linalg.norm(true_cov - est_cov, ord='fro')

def is_positive_semidefinite(M):
    return np.all(np.linalg.eigvalsh(M) >= -1e-8)

# ------------------------------------------------------------
# 4. Compute Empirical Validation Covariance (Pairwise Complete)
# ------------------------------------------------------------
print("[INFO] Computing empirical covariance (pairwise complete) ...")

emp_cov = validation_returns.cov(min_periods=1)
emp_cov = emp_cov.values

print("[INFO] Empirical covariance PSD?", is_positive_semidefinite(emp_cov))

# ------------------------------------------------------------
# 5. Validate Each Model
# ------------------------------------------------------------
results = []

for name, model in models.items():
    print(f"\n[INFO] Validating model: {name}")

    if name == "dcc_garch":
        print("[INFO] Computing dynamic covariance from DCC-GARCH ...")
        cov_matrix = model.compute_average_covariance(validation_returns)
    else:
        print("[INFO] Using static covariance output ...")
        cov_matrix = model["cov"]

    psd_flag = is_positive_semidefinite(cov_matrix)
    print(" - Positive semidefinite?", psd_flag)

    mse_val = mse(emp_cov, cov_matrix)
    frob_val = frobenius(emp_cov, cov_matrix)

    print(" - MSE:", mse_val)
    print(" - Frobenius norm:", frob_val)

    results.append({
        "Model": name,
        "PSD": psd_flag,
        "MSE": mse_val,
        "Frobenius": frob_val
    })

# ------------------------------------------------------------
# 6. Summary Table
# ------------------------------------------------------------
summary = pd.DataFrame(results)
print("\n================ VALIDATION SUMMARY ================")
print(summary.to_string(index=False))

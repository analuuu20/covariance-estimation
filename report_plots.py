import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# 1. Load results
# --------------------------------------------------
df = pd.read_csv("results/validation/consolidated_metrics.csv")

models = df["Model"]

# --------------------------------------------------
# 2. Create 2x2 figure
# --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# --------------------------------------------------
# (a) Frobenius norm
# --------------------------------------------------
axes[0].bar(models, df["Frobenius"])
axes[0].set_title("(a) Frobenius Norm")
axes[0].set_ylabel("Distance")
axes[0].grid(axis="y", linestyle="--", alpha=0.6)

# --------------------------------------------------
# (b) Spectral norm
# --------------------------------------------------
axes[1].bar(models, df["Spectral"])
axes[1].set_title("(b) Spectral Norm")
axes[1].set_ylabel("Distance")
axes[1].grid(axis="y", linestyle="--", alpha=0.6)

# --------------------------------------------------
# (c) KL divergence (log scale)
# --------------------------------------------------
axes[2].bar(models, df["KL"])
axes[2].set_title("(c) Kullback–Leibler Divergence")
axes[2].set_ylabel("KL Divergence")
axes[2].set_yscale("log")
axes[2].grid(axis="y", linestyle="--", alpha=0.6)

# --------------------------------------------------
# (d) Tracking Error
# --------------------------------------------------
axes[3].bar(models, df["TE"])
axes[3].set_title("(d) Tracking Error")
axes[3].set_ylabel("Tracking Error")
axes[3].grid(axis="y", linestyle="--", alpha=0.6)

# --------------------------------------------------
# 3. Global formatting
# --------------------------------------------------
for ax in axes:
    ax.tick_params(axis="x", rotation=15)

fig.suptitle(
    "Out-of-Sample Validation Metrics for Covariance Estimators",
    fontsize=14
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --------------------------------------------------
# 4. Save for Overleaf
# --------------------------------------------------
plt.savefig(
    "results/validation/figure_validation_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()





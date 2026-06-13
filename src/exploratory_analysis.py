from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from helpers import load_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "exploratory_analysis"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "cluster_id",
    "hour",
    "weekday_number",
    "is_weekend",
    "month",
    "day_of_month",
]
TARGET = "pickup_count"


train_df = load_split("train")
test_df = load_split("test")


# Pearson + Spearman correlations against the target.
correlations = pd.DataFrame(
    [
        {
            "feature": feature,
            "pearson_vs_pickup_count": round(
                train_df[[feature, TARGET]].corr(method="pearson").iloc[0, 1], 4
            ),
            "spearman_vs_pickup_count": round(
                train_df[[feature, TARGET]].corr(method="spearman").iloc[0, 1], 4
            ),
        }
        for feature in NUMERIC_FEATURES
    ]
).sort_values("spearman_vs_pickup_count", key=lambda s: s.abs(), ascending=False)

correlations.to_csv(TABLE_DIR / "table_2a_pearson_correlations.csv", index=False)
print("\nFeature correlations vs pickup_count:")
print(correlations.to_string(index=False))


# Spearman correlation heatmap (features + target).
heatmap_columns = NUMERIC_FEATURES + [TARGET]
matrix = train_df[heatmap_columns].corr(method="spearman")

fig, ax = plt.subplots(figsize=(6.5, 5.0))
im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(heatmap_columns)))
ax.set_yticks(range(len(heatmap_columns)))
ax.set_xticklabels(heatmap_columns, rotation=45, ha="right")
ax.set_yticklabels(heatmap_columns)
for i in range(len(heatmap_columns)):
    for j in range(len(heatmap_columns)):
        ax.text(
            j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=8
        )
fig.colorbar(im, ax=ax, label="Spearman correlation")
ax.set_title("Spearman Correlation Heatmap")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_3a_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Train vs test feature distribution drift via Kolmogorov-Smirnov.
drift_records = []
for feature in NUMERIC_FEATURES + [TARGET]:
    train_values = train_df[feature].to_numpy()
    test_values = test_df[feature].to_numpy()
    statistic, p_value = ks_2samp(train_values, test_values)
    drift_records.append(
        {
            "feature": feature,
            "train_mean": round(float(np.mean(train_values)), 4),
            "test_mean": round(float(np.mean(test_values)), 4),
            "train_std": round(float(np.std(train_values)), 4),
            "test_std": round(float(np.std(test_values)), 4),
            "ks_statistic": round(float(statistic), 4),
            "ks_p_value": round(float(p_value), 6),
            "drift_at_0.05": bool(p_value < 0.05),
        }
    )

drift_metrics = pd.DataFrame(drift_records)
drift_metrics.to_csv(TABLE_DIR / "table_6a_drift_metrics.csv", index=False)
print("\nTrain vs test drift (Kolmogorov-Smirnov):")
print(drift_metrics.to_string(index=False))


# One histogram panel per feature showing train vs test distributions.
n_cols = 3
n_rows = int(np.ceil(len(heatmap_columns) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.0 * n_rows))
axes = axes.flatten()

for ax, feature in zip(axes, heatmap_columns):
    ax.hist(
        train_df[feature], bins=30, density=True, alpha=0.45, label="Train (Apr-Jul)"
    )
    ax.hist(test_df[feature], bins=30, density=True, alpha=0.45, label="Test (Sep)")
    p = drift_metrics.loc[drift_metrics["feature"] == feature, "ks_p_value"].iloc[0]
    ax.set_title(f"{feature}\nKS p={p:.4f}")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

for ax in axes[len(heatmap_columns):]:
    ax.axis("off")

fig.suptitle("Train vs Test Feature Distribution Drift", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_6_train_test_distribution_drift.png", dpi=300, bbox_inches="tight")
plt.close(fig)

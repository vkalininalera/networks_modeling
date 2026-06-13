from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp

from helpers import load_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELING_DIR = PROJECT_ROOT / "artifacts" / "modeling"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Modelling outputs that the report cites.
test_performance = pd.read_csv(MODELING_DIR / "tables" / "table_9_final_test_performance.csv")
test_predictions = pd.read_csv(MODELING_DIR / "predictions" / "test_predictions.csv")

print("Final test performance:")
print(test_performance.to_string(index=False))


# Total-actual vs predicted demand summary.
actual_total = test_predictions["actual_pickup_count"].sum()
predicted_total = test_predictions["predicted_pickup_count"].sum()
print(
    f"\nSeptember demand: actual {actual_total:,.0f}, predicted {predicted_total:,.0f}, "
    f"captured {predicted_total / actual_total * 100:.1f}%"
)


# Drift figure (train vs test on the features most likely to drift).
train_full = load_split("train")
test_full = load_split("test")

drift_features = ["pickup_count", "hour", "day_of_month"]
fig, axes = plt.subplots(1, len(drift_features), figsize=(11, 3.4))
for ax, feature in zip(axes, drift_features):
    ax.hist(train_full[feature], bins=30, density=True, alpha=0.45, label="Train")
    ax.hist(test_full[feature], bins=30, density=True, alpha=0.45, label="Test")
    _, p_val = ks_2samp(train_full[feature], test_full[feature])
    ax.set_title(f"{feature}\nKS p={p_val:.4f}")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

fig.suptitle("Train vs Test Drift on Key Features", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_11a_evaluation_drift.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Cluster x hour bias grid (where does the model under-predict the most?).
bias_grid = (
    test_predictions.assign(
        bias=test_predictions["actual_pickup_count"]
        - test_predictions["predicted_pickup_count"]
    )
    .groupby(["cluster_id", "hour"])["bias"]
    .mean()
    .unstack("hour")
    .round(2)
)
bias_grid.to_csv(TABLE_DIR / "table_16a_bias_cluster_hour.csv")

fig, ax = plt.subplots(figsize=(10, 4.4))
im = ax.imshow(bias_grid.values, cmap="coolwarm", aspect="auto")
ax.set_xticks(range(bias_grid.shape[1]))
ax.set_xticklabels(bias_grid.columns)
ax.set_yticks(range(bias_grid.shape[0]))
ax.set_yticklabels([f"Cluster {c}" for c in bias_grid.index])
ax.set_xlabel("Hour of day")
ax.set_ylabel("Cluster")
ax.set_title("Mean Bias (Actual - Predicted) by Cluster and Hour")
fig.colorbar(im, ax=ax, label="actual - predicted (pickups)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_11b_bias_cluster_hour.png", dpi=300, bbox_inches="tight")
plt.close(fig)

peak_cluster, peak_hour = bias_grid.stack().idxmax()
peak_value = bias_grid.stack().max()
print(
    f"\nBias grid (cluster x hour): {TABLE_DIR / 'table_16a_bias_cluster_hour.csv'}"
)
print(
    f"Peak under-prediction cell: cluster {peak_cluster}, "
    f"hour {peak_hour}, "
    f"bias {peak_value:.1f} pickups"
)

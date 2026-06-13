from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from helpers import load_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "modeling"
TABLE_DIR = OUTPUT_DIR / "tables"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "pickup_count"

GLOBAL_FEATURES = [
    "cluster_id",
    "hour",
    "weekday_number",
    "is_weekend",
    "month",
    "day_of_month",
]
GLOBAL_CATEGORICAL = ["cluster_id", "hour", "weekday_number", "month"]
GLOBAL_NUMERIC = ["is_weekend", "day_of_month"]

CLUSTER_FEATURES = ["hour", "weekday_number", "is_weekend", "month", "day_of_month"]
CLUSTER_CATEGORICAL = ["hour", "weekday_number", "month"]
CLUSTER_NUMERIC = ["is_weekend", "day_of_month"]


def make_preprocessor(categorical, numeric):
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )


def regression_metrics(y_true, y_pred):
    # Negative pickup counts make no business sense; clip at zero.
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def predict_historical_average(train_df, prediction_df, group_cols):
    global_mean = train_df[TARGET].mean()
    group_means = (
        train_df.groupby(group_cols)[TARGET].mean().reset_index(name="prediction")
    )
    pred = prediction_df.merge(group_means, on=group_cols, how="left")["prediction"]
    return pred.fillna(global_mean).to_numpy()


def fit_per_cluster_ridge(train_df):
    models = {}
    for cluster_id in sorted(train_df["cluster_id"].unique()):
        chunk = train_df[train_df["cluster_id"] == cluster_id]
        pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor(CLUSTER_CATEGORICAL, CLUSTER_NUMERIC)),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        pipeline.fit(chunk[CLUSTER_FEATURES], chunk[TARGET])
        models[int(cluster_id)] = pipeline
    return models


def predict_per_cluster(models, prediction_df):
    pred = pd.Series(index=prediction_df.index, dtype=float)
    for cluster_id, model in models.items():
        mask = prediction_df["cluster_id"] == cluster_id
        if mask.sum() == 0:
            continue
        pred.loc[mask] = model.predict(prediction_df.loc[mask, CLUSTER_FEATURES])
    return pred.to_numpy()


train = load_split("train")
validation = load_split("validation")
test = load_split("test")
train_val = pd.concat([train, validation], ignore_index=True)

print(f"Train {len(train):,} / Validation {len(validation):,} / Test {len(test):,}")

X_train, y_train = train[GLOBAL_FEATURES], train[TARGET]
X_val, y_val = validation[GLOBAL_FEATURES], validation[TARGET]


# Baselines
results = []
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
results.append(
    {
        "Model": "Global mean baseline",
        "Model type": "Baseline",
        **regression_metrics(y_val, dummy.predict(X_val)),
    }
)

cluster_hour_pred = predict_historical_average(
    train, validation, ["cluster_id", "hour"]
)
results.append(
    {
        "Model": "Historical average by cluster and hour",
        "Model type": "Baseline",
        **regression_metrics(y_val, cluster_hour_pred),
    }
)


# Candidate global models
candidates = {
    "Ridge regression": Pipeline(
        [
            ("preprocess", make_preprocessor(GLOBAL_CATEGORICAL, GLOBAL_NUMERIC)),
            ("model", Ridge(alpha=1.0)),
        ]
    ),
    "Random forest regressor": Pipeline(
        [
            ("preprocess", make_preprocessor(GLOBAL_CATEGORICAL, GLOBAL_NUMERIC)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# HistGradientBoosting tuned via GridSearchCV with TimeSeriesSplit (no future leakage).
hgb_pipeline = Pipeline(
    [
        ("preprocess", make_preprocessor(GLOBAL_CATEGORICAL, GLOBAL_NUMERIC)),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ]
)
grid = {
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_iter": [200, 300, 500],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__l2_regularization": [0.0, 0.01, 0.1],
}

print("Running GridSearchCV for HistGradientBoosting...")
search = GridSearchCV(
    estimator=hgb_pipeline,
    param_grid=grid,
    scoring="neg_mean_absolute_error",
    cv=TimeSeriesSplit(n_splits=4),
    n_jobs=-1,
    refit=True,
).fit(train.sort_values("date")[GLOBAL_FEATURES], train.sort_values("date")[TARGET])
print(f"Best CV MAE: {-search.best_score_:.3f}")
print(f"Best params: {search.best_params_}")
candidates["HistGradientBoosting regressor"] = search.best_estimator_


# Train candidates on the un-sorted training set and score on validation.
for name, model in candidates.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    results.append(
        {
            "Model": name,
            "Model type": "Global model",
            **regression_metrics(y_val, model.predict(X_val)),
        }
    )


# Per-cluster Ridge as a separate cluster-specific model family.
print("Training separate Ridge models per cluster...")
results.append(
    {
        "Model": "Separate Ridge regression per cluster",
        "Model type": "Separate cluster models",
        **regression_metrics(
            y_val,
            predict_per_cluster(fit_per_cluster_ridge(train), validation),
        ),
    }
)


# Validation comparison.
comparison = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
baseline_mae = comparison.loc[
    comparison["Model"] == "Historical average by cluster and hour", "MAE"
].iloc[0]
comparison["MAE improvement vs cluster-hour baseline (%)"] = (
    (baseline_mae - comparison["MAE"]) / baseline_mae * 100
)
comparison = comparison.round(
    {"MAE": 3, "RMSE": 3, "R2": 4, "MAE improvement vs cluster-hour baseline (%)": 2}
)
comparison.to_csv(TABLE_DIR / "table_8_validation_model_comparison.csv", index=False)
print("\nValidation model comparison:")
print(comparison.to_string(index=False))

best_name = comparison.iloc[0]["Model"]
best_type = comparison.iloc[0]["Model type"]
print(f"\nBest validation model: {best_name}")


# Refit on train + validation and score on test.
if best_name == "Global mean baseline":
    final = DummyRegressor(strategy="mean").fit(
        train_val[GLOBAL_FEATURES], train_val[TARGET]
    )
    test_pred = final.predict(test[GLOBAL_FEATURES])
elif best_name == "Historical average by cluster and hour":
    test_pred = predict_historical_average(train_val, test, ["cluster_id", "hour"])
elif best_name == "Separate Ridge regression per cluster":
    test_pred = predict_per_cluster(fit_per_cluster_ridge(train_val), test)
else:
    final = clone(candidates[best_name])
    final.fit(train_val[GLOBAL_FEATURES], train_val[TARGET])
    test_pred = final.predict(test[GLOBAL_FEATURES])

test_pred = np.maximum(np.asarray(test_pred), 0)
metrics = regression_metrics(test[TARGET], test_pred)
test_performance = pd.DataFrame(
    [
        {
            "Selected model": best_name,
            "Model type": best_type,
            "Test MAE": metrics["MAE"],
            "Test RMSE": metrics["RMSE"],
            "Test R2": metrics["R2"],
        }
    ]
).round({"Test MAE": 3, "Test RMSE": 3, "Test R2": 4})
test_performance.to_csv(TABLE_DIR / "table_9_final_test_performance.csv", index=False)
print("\nFinal test performance:")
print(test_performance.to_string(index=False))


# Save test predictions for the evaluation script.
predictions = test.copy()
predictions["actual_pickup_count"] = test[TARGET].to_numpy()
predictions["predicted_pickup_count"] = test_pred
predictions["error"] = (
    predictions["actual_pickup_count"] - predictions["predicted_pickup_count"]
)
predictions["absolute_error"] = predictions["error"].abs()
predictions.to_csv(PREDICTION_DIR / "test_predictions.csv", index=False)


# Permutation feature importance on validation. We deliberately use the
# train-only model in `candidates[best_name]` (not the train+val refit) so
# that the permutation runs on data the model has not seen during training.
if best_name in candidates:
    sample = validation.sample(min(5000, len(validation)), random_state=RANDOM_STATE)
    importance = permutation_importance(
        candidates[best_name],
        sample[GLOBAL_FEATURES],
        sample[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    fi = (
        pd.DataFrame(
            {
                "feature": GLOBAL_FEATURES,
                "importance_mean": importance.importances_mean.round(2),
                "importance_std": importance.importances_std.round(2),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    print("\nPermutation feature importance (validation, 5 repeats):")
    print(fi.to_string(index=False))

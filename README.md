# DLBDSME01 — Demand Forecasting Model for Public Transport (NYC Uber)

## 1. What this code does

The code is the analytical pipeline behind the DLBDSME01 Model Engineering case study, Task 2: a demand-forecasting model for a public-transport operator in New York City.

It takes the FiveThirtyEight 2014 NYC Uber pickup dataset, follows the CRISP-DM process end-to-end, and produces every table, figure, and metric referenced in the report:

- Loads the six monthly CSV files (April–September 2014, ~4.5M pickups), assesses data quality, and visualises temporal and spatial demand patterns.
- Cleans the data, partitions NYC into K = 10 spatial clusters via `MiniBatchKMeans`, and aggregates demand to the (cluster, date, hour) grid. Splits the panel into train (Apr–Jul), validation (Aug), and test (Sep).
- Computes Pearson and Spearman correlations between every feature and the target, and runs a Kolmogorov–Smirnov test on the train-vs-test feature distributions to diagnose drift.
- Trains and compares six candidate models (two baselines, Ridge, Random Forest, HistGradientBoosting, per-cluster Ridge). Tunes HistGradientBoosting via `GridSearchCV` with `TimeSeriesSplit(n_splits=4)` over an 81-combination grid. Computes permutation feature importance.
- Refits the selected model on train + validation, scores it on September test data, and breaks the error down by cluster, by hour, and by cluster × hour to support the underprediction diagnosis in the report.

## 2. Setup

### 2.1 Prerequisites

- **Python ≥ 3.13.** Install via `pyenv` or the official installer at <https://www.python.org/downloads/>.
- **uv.** Install once:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Or via Homebrew on macOS: `brew install uv`.
- **Dataset.** Download the FiveThirtyEight Uber NYC zip from <https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city> and place it at `data/Data-for-Task 2-20260524.zip`.

### 2.2 Install Python dependencies

From the repository root:

```bash
uv sync
```

## 3. How to run the code

Run the five scripts in order — each consumes outputs from the previous phase.

```bash
uv run python src/data_understanding.py       # ~30 s
uv run python src/data_preparation.py         # ~2 min
uv run python src/exploratory_analysis.py     # ~10 s
uv run python src/modeling.py                 # ~1 min  (GridSearchCV: 81 combos × 4 folds)
uv run python src/error_evaluation.py         # ~10 s
```

All outputs (tables, figures, predictions, the saved model) land under `artifacts/`, created on first run. `random_state=42` is used everywhere, so re-runs produce identical results.

If `modeling.py` is too slow on the target machine, swap `GridSearchCV` for `RandomizedSearchCV(n_iter=20)` over the same grid — the rest of the pipeline is unaffected.

## 4. Project files

- `src/data_understanding.py` — loads the six monthly CSVs from the dataset zip and prints a quality summary (record counts, missing values, duplicates, bounding-box outliers). Saves the monthly demand chart, the weekday/weekend hourly chart, and the spatial pickup density plot.
- `src/data_preparation.py` — cleans the data, fits MiniBatchKMeans with K=10 on the pickup coordinates, and saves the cluster centres. Aggregates the pickups to a (cluster, date, hour) panel, adds the time features I use later (`hour`, `weekday_number`, `is_weekend`, `month`, `day_of_month`), and writes out the train (Apr–Jul), validation (Aug), and test (Sep) splits.
- `src/exploratory_analysis.py` — computes Pearson and Spearman correlations against `pickup_count`, draws the correlation heatmap, and runs a Kolmogorov–Smirnov test between train and test for each feature so I can see where the distributions drift.
- `src/modeling.py` — defines the candidate models (two baselines, Ridge, Random Forest, HistGradientBoosting, per-cluster Ridge) and tunes HistGradientBoosting with `GridSearchCV` and `TimeSeriesSplit(n_splits=4)`. Picks the best model on validation, refits it on train + validation, scores it on the September test set, and saves permutation feature importance.
- `src/error_evaluation.py` — takes the modelling outputs and produces the evaluation tables: validation vs test, total-actual-vs-predicted demand, and a cluster × hour bias grid that shows where the model underpredicts.
- `src/helpers.py` — small shared module with two functions reused across the scripts above: `load_pickup_data` (reads the monthly CSVs out of the dataset zip) and `load_split` (reads the train/validation/test panels written by `data_preparation.py`).

The other files in the root are just project plumbing: `pyproject.toml` and `uv.lock` for dependencies, `.python-version` to pin the Python version, and `.gitignore` to keep noise out of the repo.

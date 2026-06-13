from pathlib import Path

import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from helpers import load_pickup_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "Data-for-Task 2-20260524.zip"
PROCESSED_DIR = PROJECT_ROOT / "artifacts" / "data_preparation" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 10
RANDOM_STATE = 42


df = load_pickup_data(ZIP_PATH)
print(f"Loaded {len(df):,} rows")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime", "lat", "lon", "base"])
df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)].copy()

df["date"] = df["datetime"].dt.floor("D")
df["hour"] = df["datetime"].dt.hour
df["weekday_number"] = df["datetime"].dt.dayofweek
df["weekday_name"] = df["datetime"].dt.day_name()
df["is_weekend"] = df["weekday_number"].isin([5, 6]).astype(int)
df["month"] = df["datetime"].dt.month
df["day_of_month"] = df["datetime"].dt.day

# Cluster directly on (lat, lon). At NYC's latitude (~40.7°N) the geographic
# distortion across a city-scale bounding box is small enough that a Cartesian
# projection isn't worth the extra complexity for K-Means.
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    batch_size=20_000,
    n_init="auto",
)
coords = df[["lat", "lon"]].to_numpy()
kmeans.fit(coords)
df["raw_cluster"] = kmeans.predict(coords)

# Re-rank clusters by demand volume so cluster_id=1 is the busiest cluster.
counts = df.groupby("raw_cluster").size().sort_values(ascending=False)
ranked = {int(raw): int(new) for new, raw in enumerate(counts.index, start=1)}
df["cluster_id"] = df["raw_cluster"].map(ranked).astype(int)

# Aggregate to (cluster, date, hour) demand and fill zero-demand cells.
hourly = (
    df.groupby(["date", "hour", "cluster_id"]).size().reset_index(name="pickup_count")
)
all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
grid = pd.MultiIndex.from_product(
    [all_dates, range(24), range(1, N_CLUSTERS + 1)],
    names=["date", "hour", "cluster_id"],
).to_frame(index=False)

panel = grid.merge(hourly, on=["date", "hour", "cluster_id"], how="left")
panel["pickup_count"] = panel["pickup_count"].fillna(0).astype(int)
panel["weekday_number"] = panel["date"].dt.dayofweek
panel["weekday_name"] = panel["date"].dt.day_name()
panel["is_weekend"] = panel["weekday_number"].isin([5, 6]).astype(int)
panel["month"] = panel["date"].dt.month
panel["day_of_month"] = panel["date"].dt.day
panel = panel[
    [
        "date",
        "hour",
        "cluster_id",
        "weekday_number",
        "weekday_name",
        "is_weekend",
        "month",
        "day_of_month",
        "pickup_count",
    ]
]

# Time-based split: April–July train, August validation, September test.
train = panel[panel["date"] < "2014-08-01"]
validation = panel[
    (panel["date"] >= "2014-08-01") & (panel["date"] < "2014-09-01")
]
test = panel[panel["date"] >= "2014-09-01"]

train.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
validation.to_csv(PROCESSED_DIR / "validation_data.csv", index=False)
test.to_csv(PROCESSED_DIR / "test_data.csv", index=False)

print(
    f"Train {len(train):,} / Validation {len(validation):,} / Test {len(test):,} rows"
)

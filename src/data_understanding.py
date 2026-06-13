from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from helpers import load_pickup_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "Data-for-Task 2-20260524.zip"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "data_understanding"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE = (6.5, 3.6)


df = load_pickup_data(ZIP_PATH)
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["date"] = df["datetime"].dt.date
df["month"] = df["datetime"].dt.to_period("M").astype(str)
df["hour"] = df["datetime"].dt.hour
df["weekday_number"] = df["datetime"].dt.dayofweek
df["is_weekend"] = df["weekday_number"].isin([5, 6])

print(f"Loaded {len(df):,} rows")


# Dataset overview table
overview = pd.DataFrame(
    {
        "aspect": [
            "Observation period",
            "Number of records",
            "Number of variables in raw data",
            "Raw variables",
            "Number of monthly files",
            "Number of unique base codes",
            "Latitude range",
            "Longitude range",
        ],
        "value": [
            f"{df['datetime'].min().date()} to {df['datetime'].max().date()}",
            f"{len(df):,}",
            "4",
            "Date/Time, Lat, Lon, Base",
            df["source_file"].nunique(),
            df["base"].nunique(),
            f"{df['lat'].min():.4f} to {df['lat'].max():.4f}",
            f"{df['lon'].min():.4f} to {df['lon'].max():.4f}",
        ],
    }
)
overview.to_csv(TABLE_DIR / "table_1_dataset_overview.csv", index=False)


# Data quality table
duplicate_count = int(df.duplicated(subset=["datetime", "lat", "lon", "base"]).sum())
quality = pd.DataFrame(
    {
        "check": [
            "Missing values in raw variables",
            "Invalid datetime values",
            "Exact duplicate rows",
            "Invalid latitude values",
            "Invalid longitude values",
        ],
        "result": [
            int(df[["datetime", "lat", "lon", "base"]].isna().sum().sum()),
            int(df["datetime"].isna().sum()),
            f"{duplicate_count:,}",
            int((~df["lat"].between(-90, 90)).sum()),
            int((~df["lon"].between(-180, 180)).sum()),
        ],
    }
)
quality.to_csv(TABLE_DIR / "table_2_data_quality_summary.csv", index=False)

print("\nTable 1: Dataset Overview")
print(overview.to_string(index=False))
print("\nTable 2: Data Quality Summary")
print(quality.to_string(index=False))


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# Figure 1: pickups per month
monthly = df.groupby("month").size().reset_index(name="pickup_count").sort_values("month")
plt.figure(figsize=FIGSIZE)
plt.bar(monthly["month"], monthly["pickup_count"])
plt.title("Pickup Records per Month")
plt.xlabel("Month")
plt.ylabel("Number of pickups")
plt.xticks(rotation=45)
save_figure("figure_1_pickups_per_month.png")


# Figure 2: weekday vs weekend hourly demand
hourly = df.groupby(["is_weekend", "hour"]).size().reset_index(name="pickup_count")
weekday = hourly[~hourly["is_weekend"]]
weekend = hourly[hourly["is_weekend"]]
plt.figure(figsize=FIGSIZE)
plt.plot(weekday["hour"], weekday["pickup_count"], marker="o", label="Weekday")
plt.plot(weekend["hour"], weekend["pickup_count"], marker="o", label="Weekend")
plt.title("Hourly Demand on Weekdays and Weekends")
plt.xlabel("Hour of day")
plt.ylabel("Number of pickups")
plt.xticks(range(0, 24, 2))
plt.legend()
save_figure("figure_2_hourly_weekday_weekend_demand.png")


# Figure 3: spatial pickup density
plt.figure(figsize=(5.8, 5.0))
plt.hexbin(df["lon"], df["lat"], gridsize=100, mincnt=1)
plt.title("Spatial Pickup Density")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label="Pickup density")
save_figure("figure_3_spatial_pickup_density.png")

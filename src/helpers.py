"""Shared loaders used across the pipeline scripts."""

import re
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PREP_DIR = PROJECT_ROOT / "artifacts" / "data_preparation" / "processed"


def extract_month(filename: str) -> str:
    match = re.search(r"data-([a-z]{3})14\.csv", filename.lower())
    return match.group(1).capitalize() + " 2014" if match else "Unknown"


def load_pickup_data(zip_path: Path) -> pd.DataFrame:
    """Concatenate the monthly pickup CSVs from the dataset zip and rename
    the columns to lowercase."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find zip file: {zip_path}")

    frames = []
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = sorted(
            name
            for name in z.namelist()
            if name.lower().endswith(".csv") and "data-" in Path(name).name.lower()
        )
        if not csv_files:
            raise ValueError("No monthly pickup CSVs found in the zip.")

        for file_name in csv_files:
            with z.open(file_name) as f:
                temp = pd.read_csv(f)
            temp.columns = [c.strip() for c in temp.columns]
            temp["source_file"] = Path(file_name).name
            temp["source_month"] = extract_month(Path(file_name).name)
            frames.append(temp)

    df = pd.concat(frames, ignore_index=True)
    return df.rename(
        columns={"Date/Time": "datetime", "Lat": "lat", "Lon": "lon", "Base": "base"}
    )


def load_split(name: str) -> pd.DataFrame:
    """Load one of the train/validation/test panels written by
    data_preparation.py."""
    path = DATA_PREP_DIR / f"{name}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, parse_dates=["date"])

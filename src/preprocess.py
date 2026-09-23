import pandas as pd
import numpy as np
import glob
import os

# ==========================================
# BRIDGE STRUCTURAL HEALTH MONITORING
# DATA PREPROCESSING
# ==========================================

DATA_FOLDER = "data"
OUTPUT_FILE = "results/clean_sensor_data.csv"

# Sensor channels
ALL_CHANNELS = [f"ch_{i}" for i in range(1, 31)]

# Channels identified as invalid / duplicate / metadata
REMOVE_CHANNELS = [
    "ch_12",   # abnormal scale
    "ch_23",   # constant invalid value
    "ch_24",   # constant invalid value
    "ch_26",   # duplicate of ch_17
    "ch_27"    # metadata/timestamp-like value
]

# Final sensor channels
SENSOR_CHANNELS = [
    c for c in ALL_CHANNELS
    if c not in REMOVE_CHANNELS
]


def clean_file(file_path):
    """Read and clean one bridge sensor CSV file."""

    print("Processing:", os.path.basename(file_path))

    # Read timestamp + selected sensor channels
    columns = ["ts"] + SENSOR_CHANNELS
    df = pd.read_csv(file_path, usecols=columns)

    # Convert timestamp
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # Convert sensor columns to numeric
    for col in SENSOR_CHANNELS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # Clean ch_22 extreme invalid values
    # --------------------------------------------------
    if "ch_22" in df.columns:
        bad_values = df["ch_22"].abs() > 1000
        bad_count = bad_values.sum()

        if bad_count > 0:
            print(f"  ch_22 invalid values replaced: {bad_count}")
            df.loc[bad_values, "ch_22"] = np.nan

            # Interpolate the short missing section
            df["ch_22"] = df["ch_22"].interpolate(
                method="linear",
                limit_direction="both"
            )

    # --------------------------------------------------
    # Remove rows with invalid timestamps
    # --------------------------------------------------
    df = df.dropna(subset=["ts"])

    # --------------------------------------------------
    # Handle remaining missing sensor values
    # --------------------------------------------------
    for col in SENSOR_CHANNELS:
        if df[col].isna().sum() > 0:
            df[col] = df[col].interpolate(
                method="linear",
                limit_direction="both"
            )

    return df


def main():

    # Find all bridge CSV files
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))

    if not files:
        print("ERROR: No CSV files found in data folder.")
        return

    print("=" * 60)
    print("BRIDGE SENSOR DATA PREPROCESSING")
    print("=" * 60)

    print("Files found:", len(files))
    print("Sensor channels used:", len(SENSOR_CHANNELS))
    print("Channels:", SENSOR_CHANNELS)
    print()

    cleaned_data = []

    # Process every CSV file
    for file_path in files:
        df = clean_file(file_path)
        cleaned_data.append(df)

    # Combine all files
    final_df = pd.concat(cleaned_data, ignore_index=True)

    # Sort chronologically
    final_df = final_df.sort_values("ts").reset_index(drop=True)

    # Remove duplicate timestamps if any
    final_df = final_df.drop_duplicates(subset=["ts"])

    # Create results folder
    os.makedirs("results", exist_ok=True)

    # Save cleaned dataset
    final_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)
    print("Output file:", OUTPUT_FILE)
    print("Final shape:", final_df.shape)
    print("Final columns:", list(final_df.columns))
    print("Missing values:", final_df.isna().sum().sum())
    print()
    print("First 5 rows:")
    print(final_df.head())


if __name__ == "__main__":
    main()
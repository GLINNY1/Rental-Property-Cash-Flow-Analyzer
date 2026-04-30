"""
Realtor Property Data Preprocessing
=====================================================================
Cleans the Kaggle Realtor dataset and saves a clean property table
ready to be joined with Zillow and Census features.

Input:
  inputData/realtor-data_zip.csv

Output:
  outputData/realtor_clean.csv

Run from project root:
    python preprocessing/realtorPreProcessing.py
=====================================================================
"""

import os
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, "inputData", "realtor-data_zip.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputData")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "realtor_clean.csv")

KEEP_COLS = [
    "zip_code",
    "city",
    "state",
    "price",
    "bed",
    "bath",
    "house_size",
    "prev_sold_date",
    "price_per_sqft",
    "bed_bath_ratio",
    "size_category",
    "bed_category",
]

# Print a formatted section header for major preprocessing stages.
def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# Print one aligned status line for step-by-step logging.
def print_step(label, value):
    print(f"  {label:<45} {value}")


# Load raw Realtor records and print initial dataset overview.
def load_data() -> pd.DataFrame:
    print_section("LOADING Realtor Property Data")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print_step("Raw rows loaded:", f"{len(df):,}")
    print_step("Raw columns:", df.columns.tolist())
    print()
    print("  Status breakdown:")
    for status, count in df["status"].value_counts().items():
        print(f"    {status:<20} {count:>10,}")
    return df


# Keep only sold records so prices reflect completed transactions.
def filter_sold_properties(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 1: Filter to Sold Properties Only")
    before = len(df)
    df = df[df["status"] == "sold"].copy()
    dropped = before - len(df)

    print_step("Rows before filter:", f"{before:,}")
    print_step("Rows kept (sold):", f"{len(df):,}")
    print_step("Rows dropped (for_sale, ready_to_build):", f"{dropped:,}")
    print()
    print("  [INFO] Keeping only 'sold' records - these have actual")
    print("    transaction prices, not asking prices.")
    return df


# Remove columns that are not needed for downstream modeling.
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 2: Dropping Unnecessary Columns")
    drop_cols = ["brokered_by", "street", "acre_lot", "status"]
    df = df.drop(columns=drop_cols, errors="ignore")
    print_step("Columns dropped:", drop_cols)
    print_step("Columns remaining:", df.columns.tolist())
    return df


# Normalize ZIP codes and drop rows with missing or invalid ZIPs.
def clean_zip_codes(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 3: ZIP Code Formatting")
    before = len(df)
    df = df.dropna(subset=["zip_code"])
    df["zip_code"] = (
        df["zip_code"].astype(float).astype(int).astype(str).str.zfill(5)
    )
    df = df[df["zip_code"].str.len() == 5]
    df = df[df["zip_code"] != "00000"]

    print_step("Rows before ZIP cleaning:", f"{before:,}")
    print_step("Rows after ZIP cleaning:", f"{len(df):,}")
    print_step("Dropped (missing/invalid ZIP):", f"{before - len(df):,}")
    print()

    zip_len = df["zip_code"].str.len().value_counts().sort_index()
    print("  ZIP length distribution after padding:")
    for length, count in zip_len.items():
        print(f"    {length} digits: {count:,}")
    return df


# Drop rows with missing sale price values.
def drop_missing_price(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 4: Dropping Rows Missing Price")
    before = len(df)
    df = df.dropna(subset=["price"])
    print_step("Rows dropped (no price):", f"{before - len(df):,}")
    print_step("Rows remaining:", f"{len(df):,}")
    return df


# Filter unrealistic price outliers outside the rental-focused range.
def remove_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 5: Removing Price Outliers")
    before = len(df)
    price_min = 10_000
    price_max = 5_000_000
    df = df[(df["price"] >= price_min) & (df["price"] <= price_max)]

    print_step(f"Price range kept (${price_min:,} - ${price_max:,}):", "")
    print_step("Rows dropped (price outliers):", f"{before - len(df):,}")
    print_step("Rows remaining:", f"{len(df):,}")
    print()
    print("  Price distribution after cleaning:")
    for pct in [0.05, 0.25, 0.50, 0.75, 0.95]:
        val = df["price"].quantile(pct)
        print(f"    {int(pct * 100):>3}th percentile: ${val:>12,.0f}")
    return df


# Remove outliers in bed, bath, and house-size fields when values are present.
def remove_bed_bath_size_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 6: Removing Bed / Bath / House Size Outliers")
    before = len(df)
    bed_min, bed_max = 1, 10
    bath_min, bath_max = 1, 15
    size_min, size_max = 200, 10_000

    bed_mask = df["bed"].isna() | df["bed"].between(bed_min, bed_max)
    bath_mask = df["bath"].isna() | df["bath"].between(bath_min, bath_max)
    size_mask = df["house_size"].isna() | df["house_size"].between(size_min, size_max)
    df = df[bed_mask & bath_mask & size_mask].copy()

    print_step(f"Beds kept ({bed_min}-{bed_max}):", "")
    print_step(f"Baths kept ({bath_min}-{bath_max}):", "")
    print_step(f"House size kept ({size_min}-{size_max} sqft):", "")
    print_step("Rows dropped (bed/bath/size outliers):", f"{before - len(df):,}")
    print_step("Rows remaining:", f"{len(df):,}")
    return df


# Drop rows only when all three core structural features are missing.
def drop_rows_missing_all_core_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 7: Dropping Rows Missing All Core Features")
    before = len(df)
    df = df.dropna(subset=["bed", "bath", "house_size"], how="all")
    print_step("Rows dropped (missing bed AND bath AND size):", f"{before - len(df):,}")
    print_step("Rows remaining:", f"{len(df):,}")
    return df


# Impute bed, bath, and size using ZIP, then state, then global medians.
def impute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 8: Imputing Missing Bed / Bath / House Size")

    print("  Missing before imputation:")
    for col in ["bed", "bath", "house_size"]:
        missing = df[col].isna().sum()
        pct = missing / len(df) * 100
        print(f"    {col:<15} {missing:>8,} missing ({pct:.1f}%)")

    for col in ["bed", "bath", "house_size"]:
        zip_median = df.groupby("zip_code")[col].transform("median")
        df[col] = df[col].fillna(zip_median)

        state_median = df.groupby("state")[col].transform("median")
        df[col] = df[col].fillna(state_median)

        overall_median = df[col].median()
        df[col] = df[col].fillna(overall_median)

    print()
    print("  Missing after imputation:")
    for col in ["bed", "bath", "house_size"]:
        missing = df[col].isna().sum()
        print(f"    {col:<15} {missing:>8,} missing")

    df["bed"] = df["bed"].round(0).astype(int)
    df["bath"] = df["bath"].round(1)
    df["house_size"] = df["house_size"].round(0).astype(int)
    return df


# Map house size to a readable size bucket for segmentation.
def size_category(sqft):
    if sqft < 1000:
        return "small"
    if sqft < 2000:
        return "medium"
    if sqft < 3500:
        return "large"
    return "xlarge"


# Map bedroom count to standardized bedroom-tier labels.
def bed_category(beds):
    if beds == 1:
        return "1br"
    if beds == 2:
        return "2br"
    if beds == 3:
        return "3br"
    if beds == 4:
        return "4br"
    return "5br_plus"


# Create engineered property-level features used by downstream models.
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 9: Feature Engineering")
    df["price_per_sqft"] = (df["price"] / df["house_size"]).round(2)
    df["bed_bath_ratio"] = (df["bed"] / df["bath"]).round(2)
    df["size_category"] = df["house_size"].apply(size_category)
    df["bed_category"] = df["bed"].apply(bed_category)

    print_step("price_per_sqft created:", f"{df['price_per_sqft'].notna().sum():,} rows")
    print_step("bed_bath_ratio created:", f"{df['bed_bath_ratio'].notna().sum():,} rows")
    print_step("size_category created:", df["size_category"].value_counts().to_dict())
    print_step("bed_category created:", df["bed_category"].value_counts().to_dict())
    return df


# Keep final columns and print missingness and coverage summaries.
def finalize_columns_and_summary(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 10: Final Column Selection")
    df = df[[c for c in KEEP_COLS if c in df.columns]]
    print_step("Final columns:", df.columns.tolist())
    print()

    print(f"  {'Column':<20} {'Non-Null':>10} {'Missing':>10} {'Missing %':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    for col in df.columns:
        non_null = df[col].notna().sum()
        missing = df[col].isna().sum()
        pct = missing / len(df) * 100
        print(f"  {col:<20} {non_null:>10,} {missing:>10,} {pct:>9.1f}%")

    print()
    print("  Price distribution (final):")
    for pct in [0.05, 0.25, 0.50, 0.75, 0.95]:
        val = df["price"].quantile(pct)
        print(f"    {int(pct * 100):>3}th percentile: ${val:>12,.0f}")

    print()
    print("  State coverage:")
    print(f"    {df['state'].nunique()} states | {df['zip_code'].nunique():,} unique ZIPs")
    print(f"    Top 5 states: {df['state'].value_counts().head(5).to_dict()}")
    return df


# Save cleaned Realtor data and print final completion summary.
def save_output(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 60)
    print("  REALTOR PREPROCESSING COMPLETE")
    print("=" * 60)
    print()
    print("  Output: outputData/realtor_clean.csv")
    print(f"    {df.shape[0]:,} properties × {df.shape[1]} columns")
    print()
    print("  Columns:")
    print("    Identifiers: zip_code, city, state")
    print("    Property:    price, bed, bath, house_size, prev_sold_date")
    print("    Engineered:  price_per_sqft, bed_bath_ratio,")
    print("                 size_category, bed_category")
    print()


# Run the full Realtor preprocessing pipeline in original step order.
def main() -> None:
    df = load_data()
    df = filter_sold_properties(df)
    df = drop_unnecessary_columns(df)
    df = clean_zip_codes(df)
    df = drop_missing_price(df)
    df = remove_price_outliers(df)
    df = remove_bed_bath_size_outliers(df)
    df = drop_rows_missing_all_core_features(df)
    df = impute_core_features(df)
    df = engineer_features(df)
    df = finalize_columns_and_summary(df)
    save_output(df)


if __name__ == "__main__":
    main()
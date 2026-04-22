"""
Census ACS Data Preprocessing
=====================================================================
Cleans and validates the Census ACS 5-Year ZIP-level data pulled
from the Census API, and saves a clean output table.

Input:
  inputData/censusData/census_acs_zip.csv   (from census_api_pull.py)

Output:
  outputData/census_zip_features.csv

Run from project root:
    python preprocessing/censusPreProcessing.py
=====================================================================
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, "inputData", "censusData", "census_acs_zip.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputData")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(label, value):
    print(f"  {label:<45} {value}")


# =============================================================================
# LOAD
# =============================================================================
print_section("LOADING Census ACS Data")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print_step("Raw rows loaded:", f"{len(df):,}")
print_step("Raw columns:", f"{df.columns.tolist()}")


# =============================================================================
# STEP 1 — ZIP CODE FORMATTING
# =============================================================================
print_section("STEP 1: ZIP Code Formatting")

before = len(df)
df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

# Drop any rows where zip_code is invalid (all zeros or too short)
df = df[df["zip_code"].str.len() == 5]
df = df[df["zip_code"] != "00000"]

print_step("ZIPs before cleaning:", f"{before:,}")
print_step("ZIPs after cleaning:", f"{len(df):,}")
print_step("Sample ZIP codes:", df["zip_code"].head(5).tolist())


# =============================================================================
# STEP 2 — HANDLE CENSUS SENTINEL VALUES
# =============================================================================
print_section("STEP 2: Handling Missing / Sentinel Values")

# Census uses -666666666 to mean "data not available"
# Should already be replaced from API pull but double-check
numeric_cols = [
    "median_gross_rent",
    "median_home_value",
    "median_household_income",
    "total_occupied_units",
    "owner_occupied",
    "renter_occupied",
    "total_housing_units",
    "vacant_units",
    "total_population",
]

for col in numeric_cols:
    if col in df.columns:
        sentinel_count = (df[col] == -666666666).sum()
        if sentinel_count > 0:
            print_step(f"  Replacing {sentinel_count} sentinel values in {col}:", "→ NaN")
            df[col] = df[col].replace(-666666666, np.nan)

        # Also replace any negative values (invalid for these fields)
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print_step(f"  Replacing {neg_count} negative values in {col}:", "→ NaN")
            df[col] = df[col].where(df[col] >= 0, np.nan)

print_step("Sentinel value check complete", "✓")


# =============================================================================
# STEP 3 — DATA TYPE CONVERSION
# =============================================================================
print_section("STEP 3: Data Type Conversion")

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Round population and unit counts to integers
int_cols = [
    "total_occupied_units", "owner_occupied", "renter_occupied",
    "total_housing_units", "vacant_units", "total_population"
]
for col in int_cols:
    if col in df.columns:
        df[col] = df[col].round(0).astype("Int64")   # Int64 supports NaN

print_step("Numeric columns converted:", f"{len(numeric_cols)} columns")


# =============================================================================
# STEP 4 — REMOVE OUTLIERS
# =============================================================================
print_section("STEP 4: Removing Outliers")

before = len(df)

# Median gross rent: $0 or over $10,000/month are clearly wrong
rent_outliers = df["median_gross_rent"].notna() & (
    (df["median_gross_rent"] <= 0) | (df["median_gross_rent"] > 10_000)
)
df.loc[rent_outliers, "median_gross_rent"] = np.nan
print_step("Rent outliers nulled (≤$0 or >$10k):", f"{rent_outliers.sum():,} ZIPs")

# Median home value: $0 or over $5M are clearly wrong
value_outliers = df["median_home_value"].notna() & (
    (df["median_home_value"] <= 0) | (df["median_home_value"] > 5_000_000)
)
df.loc[value_outliers, "median_home_value"] = np.nan
print_step("Home value outliers nulled (≤$0 or >$5M):", f"{value_outliers.sum():,} ZIPs")

# Median household income: $0 or over $500k are suspicious
income_outliers = df["median_household_income"].notna() & (
    (df["median_household_income"] <= 0) | (df["median_household_income"] > 500_000)
)
df.loc[income_outliers, "median_household_income"] = np.nan
print_step("Income outliers nulled (≤$0 or >$500k):", f"{income_outliers.sum():,} ZIPs")

print_step("Rows after outlier treatment:", f"{len(df):,} (no rows dropped, values nulled)")


# =============================================================================
# STEP 5 — RECOMPUTE DERIVED FEATURES
# =============================================================================
print_section("STEP 5: Recomputing Derived Features")

# Renter percentage: share of occupied units that are renter-occupied
df["renter_pct"] = (
    df["renter_occupied"] / df["total_occupied_units"]
).round(4)

# Clamp to [0, 1] — ratios outside this range are data errors
df.loc[df["renter_pct"] > 1, "renter_pct"] = np.nan
df.loc[df["renter_pct"] < 0, "renter_pct"] = np.nan

# Vacancy rate: share of total housing units that are vacant
df["vacancy_rate"] = (
    df["vacant_units"] / df["total_housing_units"]
).round(4)

df.loc[df["vacancy_rate"] > 1, "vacancy_rate"] = np.nan
df.loc[df["vacancy_rate"] < 0, "vacancy_rate"] = np.nan

# Income-to-rent ratio: annual income / annual rent
# Shows how many years of income = 1 year of rent (affordability signal)
df["income_to_rent_ratio"] = (
    df["median_household_income"] / (df["median_gross_rent"] * 12)
).round(4)

# Owner percentage: complement of renter_pct
df["owner_pct"] = (
    df["owner_occupied"] / df["total_occupied_units"]
).round(4)
df.loc[df["owner_pct"] > 1, "owner_pct"] = np.nan
df.loc[df["owner_pct"] < 0, "owner_pct"] = np.nan

print_step("renter_pct computed:", f"{df['renter_pct'].notna().sum():,} ZIPs")
print_step("vacancy_rate computed:", f"{df['vacancy_rate'].notna().sum():,} ZIPs")
print_step("income_to_rent_ratio computed:", f"{df['income_to_rent_ratio'].notna().sum():,} ZIPs")
print_step("owner_pct computed:", f"{df['owner_pct'].notna().sum():,} ZIPs")


# =============================================================================
# STEP 6 — MISSING VALUE SUMMARY
# =============================================================================
print_section("STEP 6: Missing Value Summary")

keep_cols = [
    "zip_code",
    "median_gross_rent",
    "median_home_value",
    "median_household_income",
    "total_occupied_units",
    "owner_occupied",
    "renter_occupied",
    "total_housing_units",
    "vacant_units",
    "total_population",
    "renter_pct",
    "owner_pct",
    "vacancy_rate",
    "income_to_rent_ratio",
]

df = df[[c for c in keep_cols if c in df.columns]]

print(f"  {'Column':<35} {'Non-Null':>10} {'Missing':>10} {'Missing %':>10}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
for col in df.columns:
    non_null = df[col].notna().sum()
    missing  = df[col].isna().sum()
    pct      = (missing / len(df) * 100)
    print(f"  {col:<35} {non_null:>10,} {missing:>10,} {pct:>9.1f}%")


# =============================================================================
# STEP 7 — FINAL VALIDATION
# =============================================================================
print_section("STEP 7: Final Validation")

# Check for duplicate ZIPs
dupes = df["zip_code"].duplicated().sum()
print_step("Duplicate ZIP codes:", f"{dupes:,}")
if dupes > 0:
    df = df.drop_duplicates(subset=["zip_code"], keep="first")
    print_step("Duplicates removed, keeping first:", "✓")

# Sanity check on renter_pct + owner_pct ≈ 1.0
if "renter_pct" in df.columns and "owner_pct" in df.columns:
    both = df[["renter_pct", "owner_pct"]].dropna()
    total = (both["renter_pct"] + both["owner_pct"])
    bad = ((total < 0.95) | (total > 1.05)).sum()
    print_step("ZIPs where renter+owner pct deviates >5%:", f"{bad:,}")

print_step("Final shape:", f"{df.shape[0]:,} rows × {df.shape[1]} columns")
print_step("Final columns:", df.columns.tolist())


# =============================================================================
# SAVE
# =============================================================================
out_path = os.path.join(OUTPUT_DIR, "census_zip_features.csv")
df.to_csv(out_path, index=False)

print()
print("=" * 60)
print("  CENSUS PREPROCESSING COMPLETE")
print("=" * 60)
print()
print(f"  Output: outputData/census_zip_features.csv")
print(f"    {df.shape[0]:,} ZIPs × {df.shape[1]} features")
print()
print("  Features:")
print("    Raw:     median_gross_rent, median_home_value,")
print("             median_household_income, total_population,")
print("             total_occupied_units, owner_occupied,")
print("             renter_occupied, total_housing_units, vacant_units")
print("    Derived: renter_pct, owner_pct, vacancy_rate,")
print("             income_to_rent_ratio")
print()
"""
Census API Data Pull
=====================================================================
Pulls ACS 5-Year Estimates at the ZIP Code Tabulation Area (ZCTA)
level and saves to inputData/censusData/census_acs_zip.csv
 
File lives in: preprocessing/census_api_pull.py
.env lives in: project root (one level up)
 
API key is loaded from .env file — never hardcoded here.
 
Run from project root:
    pip install python-dotenv census us pandas
    python preprocessing/census_api_pull.py
=====================================================================
"""
 
import os
import pandas as pd
from dotenv import load_dotenv
from census import Census
 
# ── Load API key from .env in project root ────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
API_KEY = os.getenv("CENSUS_API_KEY")
 
if not API_KEY:
    raise ValueError(
        "CENSUS_API_KEY not found. "
        "Make sure your .env file exists and contains: CENSUS_API_KEY=your_key_here"
    )
 
print("Census API key loaded ✓")
 
# ── ACS variables to pull ─────────────────────────────────────────────────────
# Format: { 'ACS_CODE': 'friendly_column_name' }
VARIABLES = {
    "B25064_001E": "median_gross_rent",           # Median gross rent ($)
    "B25077_001E": "median_home_value",           # Median home value ($)
    "B19013_001E": "median_household_income",     # Median household income ($)
    "B25003_001E": "total_occupied_units",        # Total occupied housing units
    "B25003_002E": "owner_occupied",              # Owner-occupied units
    "B25003_003E": "renter_occupied",             # Renter-occupied units
    "B25002_001E": "total_housing_units",         # Total housing units
    "B25002_003E": "vacant_units",               # Vacant housing units
    "B01003_001E": "total_population",            # Total population
}
 
# ── Pull from Census API ──────────────────────────────────────────────────────
print("Connecting to Census API...")
print("Pulling ACS 5-Year estimates at ZIP Code (ZCTA) level...")
print("This may take 30–60 seconds...\n")
 
c = Census(API_KEY)
 
# ACS 5-year is most reliable at ZIP level (smaller sample sizes need 5yr avg)
data = c.acs5.get(
    fields=["NAME"] + list(VARIABLES.keys()),
    geo={"for": "zip code tabulation area:*"},
)
 
print(f"  → Received {len(data):,} ZIP code records")
 
# ── Convert to DataFrame ──────────────────────────────────────────────────────
df = pd.DataFrame(data)
 
# Rename ACS codes to friendly names
df = df.rename(columns=VARIABLES)
df = df.rename(columns={"zip code tabulation area": "zip_code"})
 
# Drop the NAME column (redundant with zip_code)
df = df.drop(columns=["NAME"], errors="ignore")
 
# ── Clean up data types ───────────────────────────────────────────────────────
# Census API returns everything as strings — convert numeric columns
numeric_cols = list(VARIABLES.values())
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
# Replace Census sentinel value -666666666 (= data not available) with NaN
df[numeric_cols] = df[numeric_cols].replace(-666666666, pd.NA)
 
# Zero-pad ZIP codes to 5 digits
df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
 
# ── Derived features ──────────────────────────────────────────────────────────
# Renter percentage: share of occupied units that are renter-occupied
df["renter_pct"] = df["renter_occupied"] / df["total_occupied_units"]
 
# Vacancy rate: share of total housing units that are vacant
df["vacancy_rate"] = df["vacant_units"] / df["total_housing_units"]
 
# Income-to-rent ratio: how many months of income to cover annual rent
df["income_to_rent_ratio"] = (
    df["median_household_income"] / (df["median_gross_rent"] * 12)
)
 
# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("Dataset summary:")
print(f"  Total ZIPs pulled:     {len(df):,}")
print(f"  ZIPs with rent data:   {df['median_gross_rent'].notna().sum():,}")
print(f"  ZIPs with income data: {df['median_household_income'].notna().sum():,}")
print(f"  ZIPs with renter pct:  {df['renter_pct'].notna().sum():,}")
print()
print("Sample rows:")
print(df[["zip_code", "median_gross_rent", "median_household_income",
          "renter_pct", "vacancy_rate"]].head(5).to_string(index=False))
 
# ── Save output ───────────────────────────────────────────────────────────────
out_dir = os.path.join(ROOT_DIR, "inputData", "censusData")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "census_acs_zip.csv")
 
df.to_csv(out_path, index=False)
print(f"\n✓ Saved → {out_path}")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""
Master Preprocessing Pipeline
=====================================================================
Joins all preprocessed datasets into one master table and computes
cash flow investment metrics used as model features and labels.

Inputs (from outputData/):
  realtor_clean.csv           — 747,498 cleaned property records
  zillow_zip_features.csv     — ZIP-level Zillow features
  census_zip_features.csv     — ZIP-level Census ACS features
  zillow_metro_features.csv   — Metro-level Zillow features

Output:
  outputData/master.csv       — Final joined table ready for modeling

Join strategy:
  Property → ZIP features     (on zip_code)
  Property → Census features  (on zip_code)
  Property → Metro features   (on state abbreviation, best-effort)

Run from project root:
    python preprocessing/masterPreProcessing.py
=====================================================================
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputData")

def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(label, value):
    print(f"  {label:<45} {value}")

def zip_str(series):
    """Normalize ZIP codes to 5-digit zero-padded strings."""
    return series.astype(str).str.zfill(5)


# ── State full name → abbreviation mapping ────────────────────────────────────
STATE_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}


# =============================================================================
# LOAD ALL PREPROCESSED FILES
# =============================================================================
print_section("LOADING All Preprocessed Files")

realtor = pd.read_csv(os.path.join(OUTPUT_DIR, "realtor_clean.csv"), low_memory=False)
print_step("realtor_clean.csv:", f"{len(realtor):,} rows × {realtor.shape[1]} cols")

zillow_zip = pd.read_csv(os.path.join(OUTPUT_DIR, "zillow_zip_features.csv"), low_memory=False)
print_step("zillow_zip_features.csv:", f"{len(zillow_zip):,} rows × {zillow_zip.shape[1]} cols")

census_zip = pd.read_csv(os.path.join(OUTPUT_DIR, "census_zip_features.csv"), low_memory=False)
print_step("census_zip_features.csv:", f"{len(census_zip):,} rows × {census_zip.shape[1]} cols")

metro = pd.read_csv(os.path.join(OUTPUT_DIR, "zillow_metro_features.csv"), low_memory=False)
print_step("zillow_metro_features.csv:", f"{len(metro):,} rows × {metro.shape[1]} cols")


# =============================================================================
# STEP 1 — NORMALIZE ZIP CODES ACROSS ALL FILES
# =============================================================================
print_section("STEP 1: Normalizing ZIP Codes")

realtor["zip_code"]   = zip_str(realtor["zip_code"])
zillow_zip["zip_code"] = zip_str(zillow_zip["zip_code"])
census_zip["zip_code"] = zip_str(census_zip["zip_code"])

print_step("ZIP codes normalized to 5-digit strings:", "✓")
print_step("Realtor sample ZIPs:", realtor["zip_code"].head(3).tolist())
print_step("Zillow ZIP sample:", zillow_zip["zip_code"].head(3).tolist())
print_step("Census ZIP sample:", census_zip["zip_code"].head(3).tolist())


# =============================================================================
# STEP 2 — JOIN ZIP-LEVEL ZILLOW FEATURES
# =============================================================================
print_section("STEP 2: Joining ZIP-Level Zillow Features")

before = len(realtor)
master = realtor.merge(zillow_zip, on="zip_code", how="left")

# StateName from zillow_zip duplicates state col — drop it
if "StateName" in master.columns:
    master = master.drop(columns=["StateName"])

matched = master["zhvi_mid_tier"].notna().sum()
print_step("Properties with Zillow ZIP match:", f"{matched:,} ({matched/before*100:.1f}%)")
print_step("Properties without match (NaN):", f"{before - matched:,}")
print_step("Columns added:", [c for c in zillow_zip.columns if c != "zip_code"])


# =============================================================================
# STEP 3 — JOIN ZIP-LEVEL CENSUS FEATURES
# =============================================================================
print_section("STEP 3: Joining ZIP-Level Census Features")

master = master.merge(census_zip, on="zip_code", how="left")

matched = master["median_household_income"].notna().sum()
print_step("Properties with Census ZIP match:", f"{matched:,} ({matched/before*100:.1f}%)")
print_step("Properties without match (NaN):", f"{before - matched:,}")
print_step("Columns added:", [c for c in census_zip.columns if c != "zip_code"])


# =============================================================================
# STEP 4 — JOIN METRO-LEVEL FEATURES (via state abbreviation)
# =============================================================================
print_section("STEP 4: Joining Metro-Level Zillow Features")

# Convert full state name to abbreviation for metro join
master["state_abbr"] = master["state"].map(STATE_ABBR)

# For metro join: use the state-level median for each metro feature
# (best available join when we don't have city→metro mapping)
# Strategy: compute state-level median of each metro feature,
# then join to properties by state abbreviation
metro_cols = [c for c in metro.columns if c not in ["RegionName", "StateName"]]

print("  Computing state-level aggregates from metro table...")
state_metro = (metro
               .groupby("StateName")[metro_cols]
               .median()
               .reset_index()
               .rename(columns={"StateName": "state_abbr"}))

master = master.merge(state_metro, on="state_abbr", how="left")

matched = master["mortgage_pmt_20pct"].notna().sum()
print_step("Properties with metro state match:", f"{matched:,} ({matched/before*100:.1f}%)")
print_step("Metro columns added:", metro_cols)


# =============================================================================
# STEP 5 — SELECT BEST RENT ESTIMATE
# =============================================================================
print_section("STEP 5: Selecting Best Rent Estimate")

# Priority: ZIP-level ZORI > Metro-level ZORI > Census median gross rent
master["best_rent_estimate"] = (
    master["zori_rent"]
    .fillna(master["zori_metro_rent"])
    .fillna(master["median_gross_rent"])
)

zip_rent   = master["zori_rent"].notna().sum()
metro_rent = (master["zori_rent"].isna() & master["zori_metro_rent"].notna()).sum()
census_rent = (master["zori_rent"].isna() & master["zori_metro_rent"].isna() & master["median_gross_rent"].notna()).sum()
no_rent    = master["best_rent_estimate"].isna().sum()

print_step("Using ZIP ZORI (most granular):", f"{zip_rent:,} properties")
print_step("Using Metro ZORI (fallback):", f"{metro_rent:,} properties")
print_step("Using Census rent (last resort):", f"{census_rent:,} properties")
print_step("No rent estimate available:", f"{no_rent:,} properties")


# =============================================================================
# STEP 6 — SELECT BEST HOME VALUE ESTIMATE
# =============================================================================
print_section("STEP 6: Selecting Best Home Value Estimate")

# Priority: actual sale price > ZIP ZHVI > Metro ZHVI > Census home value
master["best_value_estimate"] = (
    master["price"]
    .fillna(master["zhvi_mid_tier"])
    .fillna(master["zhvi_metro_mid_tier"])
    .fillna(master["median_home_value"])
)

print_step("Using actual sale price:", f"{master['price'].notna().sum():,} properties")
print_step("Best value estimate coverage:", f"{master['best_value_estimate'].notna().sum():,} properties")


# =============================================================================
# STEP 7 — COMPUTE CASH FLOW METRICS
# =============================================================================
print_section("STEP 7: Computing Cash Flow Metrics")

# ── Constants ────────────────────────────────────────────────────────────────
VACANCY_RATE    = 0.05   # 5%  — standard vacancy allowance
MAINTENANCE_PCT = 0.01   # 1%  — annual maintenance as % of home value
MGMT_PCT        = 0.08   # 8%  — property management fee as % of rent

# ── Annual rent & income ──────────────────────────────────────────────────────
master["annual_gross_rent"] = master["best_rent_estimate"] * 12
master["vacancy_loss"]      = master["annual_gross_rent"] * VACANCY_RATE
master["effective_gross_income"] = (
    master["annual_gross_rent"] - master["vacancy_loss"]
)

# ── Operating expenses (excluding mortgage) ───────────────────────────────────
master["annual_maintenance"] = master["best_value_estimate"] * MAINTENANCE_PCT
master["annual_mgmt_fee"]    = master["annual_gross_rent"] * MGMT_PCT
master["total_operating_expenses"] = (
    master["annual_maintenance"] + master["annual_mgmt_fee"]
)

# ── Net Operating Income (NOI) ────────────────────────────────────────────────
# NOI = Effective Gross Income - Operating Expenses (before mortgage)
master["noi"] = (
    master["effective_gross_income"] - master["total_operating_expenses"]
)

# ── Cap Rate ──────────────────────────────────────────────────────────────────
# Cap Rate = NOI / Purchase Price (mortgage-independent return measure)
master["cap_rate"] = (master["noi"] / master["price"]).round(4)

# ── Cash Flow (3 down payment scenarios) ─────────────────────────────────────
for pct in [5, 10, 20]:
    col = f"total_pmt_{pct}pct"
    if col in master.columns:
        master[f"annual_total_pmt_{pct}pct"] = master[col] * 12
        master[f"annual_cash_flow_{pct}pct"] = (
            master["effective_gross_income"] - master[f"annual_total_pmt_{pct}pct"]
        )

# ── Cash-on-Cash Return (3 scenarios) ─────────────────────────────────────────
# CoC = Annual Cash Flow / Total Cash Invested (down payment)
for pct in [5, 10, 20]:
    cash_flow_col = f"annual_cash_flow_{pct}pct"
    if cash_flow_col in master.columns:
        down_payment = master["price"] * (pct / 100)
        master[f"coc_return_{pct}pct"] = (
            master[cash_flow_col] / down_payment
        ).round(4)

# ── Gross Rent Yield ──────────────────────────────────────────────────────────
master["gross_rent_yield_calc"] = (
    master["annual_gross_rent"] / master["price"]
).round(4)

print_step("Annual gross rent computed:", f"{master['annual_gross_rent'].notna().sum():,} properties")
print_step("NOI computed:", f"{master['noi'].notna().sum():,} properties")
print_step("Cap rate computed:", f"{master['cap_rate'].notna().sum():,} properties")
print_step("Cash flow (20% down) computed:", f"{master['annual_cash_flow_20pct'].notna().sum():,} properties")
print_step("CoC return (20% down) computed:", f"{master['coc_return_20pct'].notna().sum():,} properties")

print()
print("  Key metric distributions (properties with full data):")
full = master.dropna(subset=["cap_rate", "annual_cash_flow_20pct"])
print(f"  {'Metric':<30} {'Mean':>10} {'Median':>10} {'Std':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
for col, label in [
    ("cap_rate",               "Cap Rate"),
    ("coc_return_20pct",       "CoC Return (20% down)"),
    ("gross_rent_yield_calc",  "Gross Rent Yield"),
]:
    if col in full.columns:
        m = full[col].mean()
        med = full[col].median()
        s = full[col].std()
        print(f"  {label:<30} {m:>10.3f} {med:>10.3f} {s:>10.3f}")


# =============================================================================
# STEP 8 — CREATE CLASSIFICATION LABEL
# =============================================================================
print_section("STEP 8: Creating Classification Label")

# Primary label: cash flow positive at 20% down (most standard scenario)
master["is_cash_flow_positive"] = (
    master["annual_cash_flow_20pct"] > 0
).astype("Int8")   # Int8 supports NaN for rows where we can't compute

# Secondary labels for other down payment scenarios
for pct in [5, 10]:
    col = f"annual_cash_flow_{pct}pct"
    if col in master.columns:
        master[f"is_cash_flow_positive_{pct}pct"] = (
            master[col] > 0
        ).astype("Int8")

label_counts = master["is_cash_flow_positive"].value_counts()
total_labeled = master["is_cash_flow_positive"].notna().sum()

print_step("Total properties with label:", f"{total_labeled:,}")
print()
print("  Label distribution (20% down scenario):")
for val, count in label_counts.items():
    pct = count / total_labeled * 100
    label = "Cash Flow POSITIVE" if val == 1 else "Cash Flow NEGATIVE"
    print(f"    {label}: {count:,} ({pct:.1f}%)")


# =============================================================================
# STEP 9 — FINAL COLUMN ORGANIZATION
# =============================================================================
print_section("STEP 9: Final Column Organization")

# Organize columns into logical groups
id_cols = ["zip_code", "city", "state", "state_abbr"]

property_cols = [
    "price", "bed", "bath", "house_size", "prev_sold_date",
    "price_per_sqft", "bed_bath_ratio", "size_category", "bed_category"
]

zillow_zip_cols = [
    "zhvi_mid_tier", "zhvi_1br", "zhvi_2br", "zhvi_3br", "zhvi_4br",
    "zori_rent", "price_to_rent_ratio", "gross_rent_yield", "gross_rent_multiplier"
]

census_cols = [
    "median_gross_rent", "median_home_value", "median_household_income",
    "total_population", "renter_pct", "owner_pct", "vacancy_rate",
    "income_to_rent_ratio", "total_occupied_units", "owner_occupied",
    "renter_occupied", "total_housing_units", "vacant_units"
]

metro_cols_ordered = [
    "zhvi_metro_mid_tier", "zori_metro_rent",
    "mortgage_pmt_5pct", "mortgage_pmt_10pct", "mortgage_pmt_20pct",
    "total_pmt_5pct", "total_pmt_10pct", "total_pmt_20pct",
    "home_value_forecast_yoy_pct", "market_heat_index",
    "for_sale_inventory", "median_list_price", "new_listings",
    "median_sale_price", "pct_sold_above_list", "mean_days_pending",
    "share_price_cut", "sale_to_list_ratio"
]

derived_cols = [
    "best_rent_estimate", "best_value_estimate",
    "annual_gross_rent", "vacancy_loss", "effective_gross_income",
    "annual_maintenance", "annual_mgmt_fee", "total_operating_expenses",
    "noi", "cap_rate", "gross_rent_yield_calc",
    "annual_total_pmt_5pct", "annual_cash_flow_5pct", "coc_return_5pct",
    "annual_total_pmt_10pct", "annual_cash_flow_10pct", "coc_return_10pct",
    "annual_total_pmt_20pct", "annual_cash_flow_20pct", "coc_return_20pct",
]

label_cols = [
    "is_cash_flow_positive",
    "is_cash_flow_positive_5pct",
    "is_cash_flow_positive_10pct",
]

all_cols = id_cols + property_cols + zillow_zip_cols + census_cols + metro_cols_ordered + derived_cols + label_cols
final_cols = [c for c in all_cols if c in master.columns]
master = master[final_cols]

print_step("Final shape:", f"{master.shape[0]:,} rows × {master.shape[1]} columns")
print()
print(f"  {'Group':<25} {'Columns':>8}")
print(f"  {'-'*25} {'-'*8}")
print(f"  {'Identifiers':<25} {len([c for c in id_cols if c in master.columns]):>8}")
print(f"  {'Property features':<25} {len([c for c in property_cols if c in master.columns]):>8}")
print(f"  {'Zillow ZIP features':<25} {len([c for c in zillow_zip_cols if c in master.columns]):>8}")
print(f"  {'Census features':<25} {len([c for c in census_cols if c in master.columns]):>8}")
print(f"  {'Metro features':<25} {len([c for c in metro_cols_ordered if c in master.columns]):>8}")
print(f"  {'Derived / cash flow':<25} {len([c for c in derived_cols if c in master.columns]):>8}")
print(f"  {'Labels':<25} {len([c for c in label_cols if c in master.columns]):>8}")


# =============================================================================
# STEP 10 — MISSING VALUE SUMMARY
# =============================================================================
print_section("STEP 10: Missing Value Summary")

print(f"  {'Column':<35} {'Non-Null':>10} {'Missing %':>10}")
print(f"  {'-'*35} {'-'*10} {'-'*10}")

groups = {
    "-- PROPERTY --": property_cols,
    "-- ZILLOW ZIP --": zillow_zip_cols,
    "-- CENSUS --": census_cols[:6],
    "-- METRO --": ["mortgage_pmt_20pct", "total_pmt_20pct", "market_heat_index", "home_value_forecast_yoy_pct"],
    "-- DERIVED --": ["noi", "cap_rate", "annual_cash_flow_20pct", "coc_return_20pct"],
    "-- LABELS --": label_cols,
}

for group_name, cols in groups.items():
    print(f"\n  {group_name}")
    for col in cols:
        if col in master.columns:
            non_null = master[col].notna().sum()
            pct_missing = (master[col].isna().sum() / len(master) * 100)
            print(f"  {col:<35} {non_null:>10,} {pct_missing:>9.1f}%")


master["prop_price_to_rent"] = master["price"] / (master["best_rent_estimate"] * 12)
master["prop_gross_yield"]   = (master["best_rent_estimate"] * 12) / master["price"]
# =============================================================================
# SAVE
# =============================================================================
out_path = os.path.join(OUTPUT_DIR, "master.csv")
master.to_csv(out_path, index=False)

print()
print("=" * 60)
print("  MASTER PREPROCESSING COMPLETE")
print("=" * 60)
print()
print(f"  Output: outputData/master.csv")
print(f"    {master.shape[0]:,} properties × {master.shape[1]} columns")
print()
print("  Ready for modeling:")
print("    Features:  property + zillow ZIP + census + metro signals")
print("    Label:     is_cash_flow_positive (1 = positive, 0 = negative)")
print("    Models:    Logistic Regression, Decision Tree,")
print("               Neural Network, K-Means Clustering")
print()
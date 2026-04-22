"""
Zillow Data Preprocessing
=====================================================================
Processes all Zillow CSV files into two clean output tables:

  outputData/zillow_zip_features.csv    ZIP-level features
  outputData/zillow_metro_features.csv  Metro-level features

Input folder structure expected:
  inputData/
    zillowHomeValues/
      zhvi_by_ZIP_Code.csv
      zhvi_1BDR.csv, zhvi_2BDR.csv, zhvi_3BDR.csv, zhvi_4BDR.csv
      mortgage-5.csv, mortgage-10.csv, mortgage-20.csv
      total_monthly_payment_downpayment_5.csv
      total_monthly_payment_downpayment_10.csv
      total_monthly_payment_downpayment_20.csv
    zillowRentalValues/
      zori_zip.csv
      Metro_zori_uc_sfrcondomfr_sm_month.csv
    zillowZHVF/
      ZHVF-(Metro-Monthly).csv
    zillowSales/
      Median-Sale-Price-(Metro-Monthly).csv
      Percent-of-Homes-Sold-Above-List.csv
    zillowHeatMarketIndex/
      Market-Heat-Index-(Metro-Monthly).csv
    zillowForSaleListing/
      For-Sale-Inventory-(Metro-Monthly-Smoothed).csv
      Median-List-Price-(Metro-Monthly).csv
      New-Listings-(Metro-Monthly).csv
    zillowDaysOnMarket&PriceCuts/
      Mean-Days-to-Pending-(Metro-Monthly).csv
      Share-of-Listings-With-a-Price-Cut.csv

Run from project root:
    python preprocessing/zillowPreProcessing.py
=====================================================================
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Script lives at: projectRoot/preprocessing/zillowPreProcessing.py
# ROOT_DIR resolves to: projectRoot/
INPUT_DIR  = os.path.join(ROOT_DIR, "inputData")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputData")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Snapshot date: latest month available across all files
# Note: Pct Sold Above List ends Feb 2026, everything else March 2026
SNAPSHOT       = "2026-03-31"
SNAPSHOT_SALES = "2026-02-28"   # used only for Pct Sold Above List

# ── ID columns present in every Zillow file ───────────────────────────────────
ID_COLS = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_csv(path, encoding="utf-8"):
    """Read a Zillow CSV with fallback to latin1 encoding."""
    try:
        return pd.read_csv(path, low_memory=False, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def get_snapshot(df, value_name, snapshot_date=SNAPSHOT):
    """
    Extract a single snapshot month from a wide Zillow time-series.
    Returns one row per region with just the snapshot value.
    """
    if snapshot_date not in df.columns:
        # Fall back to the closest available date
        date_cols = sorted([
            c for c in df.columns
            if c not in ID_COLS + ["BaseDate"]
            and str(c)[:4].isdigit()
        ])
        snapshot_date = date_cols[-1]
        print(f"    ⚠ Snapshot {SNAPSHOT} not found — using {snapshot_date}")

    keep = [c for c in ID_COLS if c in df.columns] + [snapshot_date]
    snap = df[keep].copy()
    snap = snap.rename(columns={snapshot_date: value_name})
    return snap


def zip_pad(series):
    """Zero-pad ZIP codes to 5 digits."""
    return (series.fillna(-1)
                  .astype(float)
                  .astype(int)
                  .astype(str)
                  .str.zfill(5)
                  .replace("00000", pd.NA)
                  .replace("-0001", pd.NA))


def filter_msa(df):
    """Keep only MSA (metro) rows, drop country-level aggregate."""
    if "RegionType" in df.columns:
        return df[df["RegionType"] == "msa"].copy()
    return df


def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_dataset(name, df, geo_col="zip_code"):
    non_null = df[geo_col].notna().sum() if geo_col in df.columns else len(df)
    print(f"  ✓ {name:<45} {non_null:>7,} regions")


# =============================================================================
# SECTION 1 — ZIP-LEVEL FEATURES
# =============================================================================
print_section("SECTION 1: ZIP-Level Features")

zip_base = None   # will accumulate all ZIP features

# ── Dataset 1: ZHVI Mid-Tier (All Homes, ZIP) ─────────────────────────────────
print("\n  [Dataset 1/9] ZHVI Mid-Tier — All Homes (ZIP Level)")
path = os.path.join(INPUT_DIR, "zillowHomeValues", "zhvi_by_ZIP_Code.csv")
df = read_csv(path)
snap = get_snapshot(df, "zhvi_mid_tier")
snap["zip_code"] = zip_pad(snap["RegionName"])
snap = snap[["zip_code", "StateName", "zhvi_mid_tier"]].dropna(subset=["zip_code"])
zip_base = snap.copy()
print_dataset("ZHVI Mid-Tier (ZIP)", zip_base)

# ── Dataset 2: ZHVI by Bedroom Count (ZIP) ───────────────────────────────────
print("\n  [Dataset 2/9] ZHVI by Bedroom Count — 1BR, 2BR, 3BR, 4BR (ZIP Level)")
for br in [1, 2, 3, 4]:
    path = os.path.join(INPUT_DIR, "zillowHomeValues", f"zhvi_{br}BDR.csv")
    df   = read_csv(path)
    snap = get_snapshot(df, f"zhvi_{br}br")
    snap["zip_code"] = zip_pad(snap["RegionName"])
    snap = snap[["zip_code", f"zhvi_{br}br"]].dropna(subset=["zip_code"])
    zip_base = zip_base.merge(snap, on="zip_code", how="left")
    print_dataset(f"ZHVI {br}BR (ZIP)", snap)

# ── Dataset 3: ZORI — Observed Rent Index (ZIP) ───────────────────────────────
print("\n  [Dataset 3/9] ZORI — Zillow Observed Rent Index (ZIP Level)")
path = os.path.join(INPUT_DIR, "zillowRentalValues", "zori_zip.csv")
df   = read_csv(path)
snap = get_snapshot(df, "zori_rent")
snap["zip_code"] = zip_pad(snap["RegionName"])
snap = snap[["zip_code", "zori_rent"]].dropna(subset=["zip_code"])
zip_base = zip_base.merge(snap, on="zip_code", how="left")
print_dataset("ZORI Rent (ZIP)", snap)

# ── Dataset 4: Mortgage Payments — 3 Down Payment Scenarios (Metro → ZIP) ────
# Note: Mortgage files are Metro-level. They will be joined to ZIP via
# metro_features later. Kept here for reference in derived metrics.
print("\n  [Dataset 4/9] Mortgage Payments — 5%, 10%, 20% Down (Metro Level)")
print("  ℹ Mortgage data is Metro-level — stored in metro table, joined later")

# ── Dataset 5: Total Monthly Payment — 3 Scenarios (Metro → ZIP) ─────────────
print("\n  [Dataset 5/9] Total Monthly Payment — 5%, 10%, 20% Down (Metro Level)")
print("  ℹ Total payment data is Metro-level — stored in metro table, joined later")

# ── Derived ZIP Features ──────────────────────────────────────────────────────
print("\n  Computing derived ZIP-level features...")

# Price-to-rent ratio: home value / annual rent (higher = less affordable to rent vs buy)
zip_base["price_to_rent_ratio"] = (
    zip_base["zhvi_mid_tier"] / (zip_base["zori_rent"] * 12)
)

# Gross rent yield: annual rent / home value (higher = better raw income return)
zip_base["gross_rent_yield"] = (
    (zip_base["zori_rent"] * 12) / zip_base["zhvi_mid_tier"]
)

# Gross rent multiplier: years of rent to equal purchase price
zip_base["gross_rent_multiplier"] = zip_base["price_to_rent_ratio"]

print(f"  Price-to-rent ratio:  {zip_base['price_to_rent_ratio'].notna().sum():,} ZIPs")
print(f"  Gross rent yield:     {zip_base['gross_rent_yield'].notna().sum():,} ZIPs")

# ── Save ZIP features ─────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "zillow_zip_features.csv")
zip_base.to_csv(out_path, index=False)
print(f"\n  ✓ Saved → outputData/zillow_zip_features.csv")
print(f"    Shape: {zip_base.shape[0]:,} rows × {zip_base.shape[1]} columns")
print(f"    Columns: {zip_base.columns.tolist()}")


# =============================================================================
# SECTION 2 — METRO-LEVEL FEATURES
# =============================================================================
print_section("SECTION 2: Metro-Level Features")

metro_base = None   # will accumulate all metro features

# ── Dataset 6: ZHVI Mid-Tier (Metro) ─────────────────────────────────────────
print("\n  [Dataset 6/15] ZHVI Mid-Tier — All Homes (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowHomeValues",
                    "Metro_zhvi_uc_sfrcondo_tier_0_33_0_67_sm_sa_month.csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "zhvi_metro_mid_tier")
snap = snap[["RegionName", "StateName", "zhvi_metro_mid_tier"]].dropna(subset=["RegionName"])
metro_base = snap.copy()
print_dataset("ZHVI Mid-Tier (Metro)", metro_base, geo_col="RegionName")

# ── Dataset 7: ZORI — Metro (Smoothed, Seasonally Adjusted) ──────────────────
print("\n  [Dataset 7/15] ZORI — Observed Rent Index (Metro, Smoothed SA)")
path = os.path.join(INPUT_DIR, "zillowRentalValues",
                    "Metro_zori_uc_sfrcondomfr_sm_sa_month.csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "zori_metro_rent")
snap = snap[["RegionName", "zori_metro_rent"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="outer")
print_dataset("ZORI Rent (Metro)", snap, geo_col="RegionName")

# ── Dataset 8: Mortgage Payments — 5%, 10%, 20% Down ─────────────────────────
print("\n  [Dataset 8/15] Mortgage Payments — 5%, 10%, 20% Down (Metro Level)")
for pct in [5, 10, 20]:
    path = os.path.join(INPUT_DIR, "zillowHomeValues", f"mortgage-{pct}.csv")
    df   = read_csv(path)
    df   = filter_msa(df)
    snap = get_snapshot(df, f"mortgage_pmt_{pct}pct")
    snap = snap[["RegionName", f"mortgage_pmt_{pct}pct"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset(f"Mortgage Payment {pct}% Down (Metro)", snap, geo_col="RegionName")

# ── Dataset 9: Total Monthly Payment — 5%, 10%, 20% Down ─────────────────────
print("\n  [Dataset 9/15] Total Monthly Payment — 5%, 10%, 20% Down (Metro Level)")
for pct in [5, 10, 20]:
    path = os.path.join(INPUT_DIR, "zillowHomeValues",
                        f"total_monthly_payment_downpayment_{pct}.csv")
    df   = read_csv(path)
    df   = filter_msa(df)
    snap = get_snapshot(df, f"total_pmt_{pct}pct")
    snap = snap[["RegionName", f"total_pmt_{pct}pct"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset(f"Total Monthly Payment {pct}% Down (Metro)", snap, geo_col="RegionName")

# ── Dataset 10: ZHVF — Home Value Forecast, 1-Year Ahead ─────────────────────
print("\n  [Dataset 10/15] ZHVF — Home Value Forecast, 1-Year Ahead (Metro Level)")
path  = os.path.join(INPUT_DIR, "zillowZHVF", "ZHVF-(Metro-Monthly).csv")
df    = read_csv(path)
df    = filter_msa(df)
# ZHVF only has 3 forecast columns — take the 1-year ahead (2027-03-31)
yr_col = "2027-03-31"
snap  = df[["RegionName", yr_col]].copy()
snap  = snap.rename(columns={yr_col: "home_value_forecast_yoy_pct"})
snap  = snap.dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("ZHVF 1-Year Forecast % (Metro)", snap, geo_col="RegionName")

# ── Dataset 11: Market Heat Index ─────────────────────────────────────────────
print("\n  [Dataset 11/15] Market Heat Index (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowHeatMarketIndex",
                    "Market-Heat-Index-(Metro-Monthly).csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "market_heat_index")
snap = snap[["RegionName", "market_heat_index"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("Market Heat Index (Metro)", snap, geo_col="RegionName")

# ── Dataset 12: For-Sale Inventory ────────────────────────────────────────────
print("\n  [Dataset 12/15] For-Sale Inventory — Smoothed, All Homes (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowForSaleListing",
                    "For-Sale-Inventory-(Metro-Monthly-Smoothed).csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "for_sale_inventory")
snap = snap[["RegionName", "for_sale_inventory"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("For-Sale Inventory (Metro)", snap, geo_col="RegionName")

# ── Dataset 13: Median List Price ─────────────────────────────────────────────
print("\n  [Dataset 13/15] Median List Price — Smoothed, All Homes (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowForSaleListing",
                    "Median-List-Price-(Metro-Monthly).csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "median_list_price")
snap = snap[["RegionName", "median_list_price"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("Median List Price (Metro)", snap, geo_col="RegionName")

# ── Dataset 14: New Listings ───────────────────────────────────────────────────
print("\n  [Dataset 14/15] New Listings — Smoothed, All Homes (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowForSaleListing",
                    "New-Listings-(Metro-Monthly).csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "new_listings")
snap = snap[["RegionName", "new_listings"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("New Listings (Metro)", snap, geo_col="RegionName")

# ── Dataset 15: Median Sale Price ─────────────────────────────────────────────
print("\n  [Dataset 15/19] Median Sale Price — Nowcast, All Homes (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowSales",
                    "Median-Sale-Price-(Metro-Monthly).csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "median_sale_price")
snap = snap[["RegionName", "median_sale_price"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("Median Sale Price (Metro)", snap, geo_col="RegionName")

# ── Dataset 16: Percent of Homes Sold Above List ──────────────────────────────
print("\n  [Dataset 16/19] Pct of Homes Sold Above List — Smoothed (Metro Level)")
print("  ℹ Latest available month is Feb 2026 (not March) — using 2026-02-28")
path = os.path.join(INPUT_DIR, "zillowSales",
                    "Percent-of-Homes-Sold-Above-List.csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "pct_sold_above_list", snapshot_date=SNAPSHOT_SALES)
snap = snap[["RegionName", "pct_sold_above_list"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("Pct Sold Above List (Metro)", snap, geo_col="RegionName")

# ── Dataset 17: Mean Days to Pending ──────────────────────────────────────────
print("\n  [Dataset 17/19] Mean Days to Pending — Smoothed, All Homes (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowDaysOnMarket&PriceCuts",
                    "Mean-Days-to-Pending-(Metro-Monthly).csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "mean_days_pending")
snap = snap[["RegionName", "mean_days_pending"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("Mean Days to Pending (Metro)", snap, geo_col="RegionName")

# ── Dataset 18: Share of Listings With a Price Cut ────────────────────────────
print("\n  [Dataset 18/19] Share of Listings With a Price Cut — Smoothed (Metro Level)")
path = os.path.join(INPUT_DIR, "zillowDaysOnMarket&PriceCuts",
                    "Share-of-Listings-With-a-Price-Cut.csv")
df   = read_csv(path)
df   = filter_msa(df)
snap = get_snapshot(df, "share_price_cut")
snap = snap[["RegionName", "share_price_cut"]].dropna(subset=["RegionName"])
metro_base = metro_base.merge(snap, on="RegionName", how="left")
print_dataset("Share of Listings With Price Cut (Metro)", snap, geo_col="RegionName")

# ── Derived Metro Features ────────────────────────────────────────────────────
print("\n  Computing derived metro-level features...")

# Sale-to-list ratio proxy: median sale price / median list price
metro_base["sale_to_list_ratio"] = (
    metro_base["median_sale_price"] / metro_base["median_list_price"]
)

# Market competitiveness score proxy: high heat + high pct above list + low days pending
# Normalized components (raw values kept separately above)
print(f"  Sale-to-list ratio:   {metro_base['sale_to_list_ratio'].notna().sum():,} metros")

# ── Save Metro features ───────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "zillow_metro_features.csv")
metro_base.to_csv(out_path, index=False)
print(f"\n  ✓ Saved → outputData/zillow_metro_features.csv")
print(f"    Shape: {metro_base.shape[0]:,} rows × {metro_base.shape[1]} columns")
print(f"    Columns: {metro_base.columns.tolist()}")


# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  ZILLOW PREPROCESSING COMPLETE")
print("=" * 60)
print()
print("  Output files:")
print(f"    outputData/zillow_zip_features.csv")
print(f"      {zip_base.shape[0]:,} ZIPs × {zip_base.shape[1]} features")
print(f"      Includes: ZHVI mid-tier, ZHVI 1-4BR, ZORI rent,")
print(f"                price-to-rent ratio, gross rent yield, GRM")
print()
print(f"    outputData/zillow_metro_features.csv")
print(f"      {metro_base.shape[0]:,} metros × {metro_base.shape[1]} features")
print(f"      Includes: ZHVI metro, ZORI metro, mortgage payments (3 scenarios),")
print(f"                total monthly payments (3 scenarios), ZHVF forecast,")
print(f"                market heat, inventory, list price, new listings,")
print(f"                sale price, pct above list, days pending, price cuts,")
print(f"                sale-to-list ratio")
print()
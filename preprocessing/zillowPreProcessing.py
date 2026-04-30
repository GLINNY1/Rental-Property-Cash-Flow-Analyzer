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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Script lives at: projectRoot/preprocessing/zillowPreProcessing.py
# ROOT_DIR resolves to: projectRoot/
INPUT_DIR  = os.path.join(ROOT_DIR, "inputData")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputData")
ZIP_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "zillow_zip_features.csv")
METRO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "zillow_metro_features.csv")
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

# Read Zillow CSVs with encoding fallback for robust file loading.
def read_csv(path, encoding="utf-8"):
    """Read a Zillow CSV with fallback to latin1 encoding."""
    try:
        return pd.read_csv(path, low_memory=False, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


# Extract a single snapshot month into a compact region-level table.
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
        print(f"    WARNING: Snapshot {SNAPSHOT} not found - using {snapshot_date}")

    keep = [c for c in ID_COLS if c in df.columns] + [snapshot_date]
    snap = df[keep].copy()
    snap = snap.rename(columns={snapshot_date: value_name})
    return snap


# Normalize ZIP identifiers into 5-digit strings with invalid values nulled.
def zip_pad(series):
    """Zero-pad ZIP codes to 5 digits."""
    return (series.fillna(-1)
                  .astype(float)
                  .astype(int)
                  .astype(str)
                  .str.zfill(5)
                  .replace("00000", pd.NA)
                  .replace("-0001", pd.NA))


# Keep only metro-area rows when a RegionType filter is available.
def filter_msa(df):
    """Keep only MSA (metro) rows, drop country-level aggregate."""
    if "RegionType" in df.columns:
        return df[df["RegionType"] == "msa"].copy()
    return df


# Print a section header for progress logging.
def print_section(title):
    """Print a section header for progress logging."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# Print non-null region count for a loaded dataset.
def print_dataset(name, df, geo_col="zip_code"):
    """Print non-null region count for a loaded dataset."""
    non_null = df[geo_col].notna().sum() if geo_col in df.columns else len(df)
    print(f"  [OK] {name:<42} {non_null:>7,} regions")


# Safely divide two series while converting zero denominators to NaN.
def safe_divide(numerator, denominator):
    """Divide while mapping zero denominators to NaN."""
    return numerator.div(denominator.where(denominator != 0))


# Build ZIP-level Zillow features and save the ZIP output file.
def build_zip_features() -> pd.DataFrame:
    print_section("SECTION 1: ZIP-Level Features")
    zip_base = None

    print("\n  [Dataset 1/9] ZHVI Mid-Tier — All Homes (ZIP Level)")
    path = os.path.join(INPUT_DIR, "zillowHomeValues", "zhvi_by_ZIP_Code.csv")
    df = read_csv(path)
    snap = get_snapshot(df, "zhvi_mid_tier")
    snap["zip_code"] = zip_pad(snap["RegionName"])
    snap = snap[["zip_code", "StateName", "zhvi_mid_tier"]].dropna(subset=["zip_code"])
    zip_base = snap.copy()
    print_dataset("ZHVI Mid-Tier (ZIP)", zip_base)

    print("\n  [Dataset 2/9] ZHVI by Bedroom Count — 1BR, 2BR, 3BR, 4BR (ZIP Level)")
    for br in [1, 2, 3, 4]:
        path = os.path.join(INPUT_DIR, "zillowHomeValues", f"zhvi_{br}BDR.csv")
        df = read_csv(path)
        snap = get_snapshot(df, f"zhvi_{br}br")
        snap["zip_code"] = zip_pad(snap["RegionName"])
        snap = snap[["zip_code", f"zhvi_{br}br"]].dropna(subset=["zip_code"])
        zip_base = zip_base.merge(snap, on="zip_code", how="left")
        print_dataset(f"ZHVI {br}BR (ZIP)", snap)

    print("\n  [Dataset 3/9] ZORI — Zillow Observed Rent Index (ZIP Level)")
    path = os.path.join(INPUT_DIR, "zillowRentalValues", "zori_zip.csv")
    df = read_csv(path)
    snap = get_snapshot(df, "zori_rent")
    snap["zip_code"] = zip_pad(snap["RegionName"])
    snap = snap[["zip_code", "zori_rent"]].dropna(subset=["zip_code"])
    zip_base = zip_base.merge(snap, on="zip_code", how="left")
    print_dataset("ZORI Rent (ZIP)", snap)

    print("\n  [Dataset 4/9] Mortgage Payments — 5%, 10%, 20% Down (Metro Level)")
    print("  [INFO] Mortgage data is Metro-level - stored in metro table, joined later")

    print("\n  [Dataset 5/9] Total Monthly Payment — 5%, 10%, 20% Down (Metro Level)")
    print("  [INFO] Total payment data is Metro-level - stored in metro table, joined later")

    print("\n  Computing derived ZIP-level features...")
    zip_base["price_to_rent_ratio"] = safe_divide(zip_base["zhvi_mid_tier"], zip_base["zori_rent"] * 12)
    zip_base["gross_rent_yield"] = safe_divide(zip_base["zori_rent"] * 12, zip_base["zhvi_mid_tier"])
    zip_base["gross_rent_multiplier"] = zip_base["price_to_rent_ratio"]
    print(f"  Price-to-rent ratio:  {zip_base['price_to_rent_ratio'].notna().sum():,} ZIPs")
    print(f"  Gross rent yield:     {zip_base['gross_rent_yield'].notna().sum():,} ZIPs")

    zip_base.to_csv(ZIP_OUTPUT_PATH, index=False)
    print("\n  [OK] Saved -> outputData/zillow_zip_features.csv")
    print(f"    Shape: {zip_base.shape[0]:,} rows × {zip_base.shape[1]} columns")
    print(f"    Columns: {zip_base.columns.tolist()}")
    return zip_base


# Build metro-level Zillow features and save the metro output file.
def build_metro_features() -> pd.DataFrame:
    print_section("SECTION 2: Metro-Level Features")
    metro_base = None

    print("\n  [Dataset 6/15] ZHVI Mid-Tier — All Homes (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowHomeValues", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "zhvi_metro_mid_tier")
    snap = snap[["RegionName", "StateName", "zhvi_metro_mid_tier"]].dropna(subset=["RegionName"])
    metro_base = snap.copy()
    print_dataset("ZHVI Mid-Tier (Metro)", metro_base, geo_col="RegionName")

    print("\n  [Dataset 7/15] ZORI — Observed Rent Index (Metro, Smoothed SA)")
    path = os.path.join(INPUT_DIR, "zillowRentalValues", "Metro_zori_uc_sfrcondomfr_sm_sa_month.csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "zori_metro_rent")
    snap = snap[["RegionName", "zori_metro_rent"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="outer")
    print_dataset("ZORI Rent (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 8/15] Mortgage Payments — 5%, 10%, 20% Down (Metro Level)")
    for pct in [5, 10, 20]:
        path = os.path.join(INPUT_DIR, "zillowHomeValues", f"mortgage-{pct}.csv")
        df = read_csv(path)
        df = filter_msa(df)
        snap = get_snapshot(df, f"mortgage_pmt_{pct}pct")
        snap = snap[["RegionName", f"mortgage_pmt_{pct}pct"]].dropna(subset=["RegionName"])
        metro_base = metro_base.merge(snap, on="RegionName", how="left")
        print_dataset(f"Mortgage Payment {pct}% Down (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 9/15] Total Monthly Payment — 5%, 10%, 20% Down (Metro Level)")
    for pct in [5, 10, 20]:
        path = os.path.join(INPUT_DIR, "zillowHomeValues", f"total_monthly_payment_downpayment_{pct}.csv")
        df = read_csv(path)
        df = filter_msa(df)
        snap = get_snapshot(df, f"total_pmt_{pct}pct")
        snap = snap[["RegionName", f"total_pmt_{pct}pct"]].dropna(subset=["RegionName"])
        metro_base = metro_base.merge(snap, on="RegionName", how="left")
        print_dataset(f"Total Monthly Payment {pct}% Down (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 10/15] ZHVF — Home Value Forecast, 1-Year Ahead (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowZHVF", "ZHVF-(Metro-Monthly).csv")
    df = read_csv(path)
    df = filter_msa(df)
    yr_col = "2027-03-31"
    snap = df[["RegionName", yr_col]].copy()
    snap = snap.rename(columns={yr_col: "home_value_forecast_yoy_pct"})
    snap = snap.dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("ZHVF 1-Year Forecast % (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 11/15] Market Heat Index (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowHeatMarketIndex", "Market-Heat-Index-(Metro-Monthly).csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "market_heat_index")
    snap = snap[["RegionName", "market_heat_index"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("Market Heat Index (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 12/15] For-Sale Inventory — Smoothed, All Homes (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowForSaleListing", "For-Sale-Inventory-(Metro-Monthly-Smoothed).csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "for_sale_inventory")
    snap = snap[["RegionName", "for_sale_inventory"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("For-Sale Inventory (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 13/15] Median List Price — Smoothed, All Homes (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowForSaleListing", "Median-List-Price-(Metro-Monthly).csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "median_list_price")
    snap = snap[["RegionName", "median_list_price"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("Median List Price (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 14/15] New Listings — Smoothed, All Homes (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowForSaleListing", "New-Listings-(Metro-Monthly).csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "new_listings")
    snap = snap[["RegionName", "new_listings"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("New Listings (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 15/19] Median Sale Price — Nowcast, All Homes (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowSales", "Median-Sale-Price-(Metro-Monthly).csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "median_sale_price")
    snap = snap[["RegionName", "median_sale_price"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("Median Sale Price (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 16/19] Pct of Homes Sold Above List — Smoothed (Metro Level)")
    print("  [INFO] Latest available month is Feb 2026 (not March) - using 2026-02-28")
    path = os.path.join(INPUT_DIR, "zillowSales", "Percent-of-Homes-Sold-Above-List.csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "pct_sold_above_list", snapshot_date=SNAPSHOT_SALES)
    snap = snap[["RegionName", "pct_sold_above_list"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("Pct Sold Above List (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 17/19] Mean Days to Pending — Smoothed, All Homes (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowDaysOnMarket&PriceCuts", "Mean-Days-to-Pending-(Metro-Monthly).csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "mean_days_pending")
    snap = snap[["RegionName", "mean_days_pending"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("Mean Days to Pending (Metro)", snap, geo_col="RegionName")

    print("\n  [Dataset 18/19] Share of Listings With a Price Cut — Smoothed (Metro Level)")
    path = os.path.join(INPUT_DIR, "zillowDaysOnMarket&PriceCuts", "Share-of-Listings-With-a-Price-Cut.csv")
    df = read_csv(path)
    df = filter_msa(df)
    snap = get_snapshot(df, "share_price_cut")
    snap = snap[["RegionName", "share_price_cut"]].dropna(subset=["RegionName"])
    metro_base = metro_base.merge(snap, on="RegionName", how="left")
    print_dataset("Share of Listings With Price Cut (Metro)", snap, geo_col="RegionName")

    print("\n  Computing derived metro-level features...")
    metro_base["sale_to_list_ratio"] = safe_divide(metro_base["median_sale_price"], metro_base["median_list_price"])
    print(f"  Sale-to-list ratio:   {metro_base['sale_to_list_ratio'].notna().sum():,} metros")

    metro_base.to_csv(METRO_OUTPUT_PATH, index=False)
    print("\n  [OK] Saved -> outputData/zillow_metro_features.csv")
    print(f"    Shape: {metro_base.shape[0]:,} rows × {metro_base.shape[1]} columns")
    print(f"    Columns: {metro_base.columns.tolist()}")
    return metro_base


# Print final output summary for both generated Zillow feature tables.
def print_summary(zip_base: pd.DataFrame, metro_base: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("  ZILLOW PREPROCESSING COMPLETE")
    print("=" * 60)
    print()
    print("  Output files:")
    print("    outputData/zillow_zip_features.csv")
    print(f"      {zip_base.shape[0]:,} ZIPs × {zip_base.shape[1]} features")
    print("      Includes: ZHVI mid-tier, ZHVI 1-4BR, ZORI rent,")
    print("                price-to-rent ratio, gross rent yield, GRM")
    print()
    print("    outputData/zillow_metro_features.csv")
    print(f"      {metro_base.shape[0]:,} metros × {metro_base.shape[1]} features")
    print("      Includes: ZHVI metro, ZORI metro, mortgage payments (3 scenarios),")
    print("                total monthly payments (3 scenarios), ZHVF forecast,")
    print("                market heat, inventory, list price, new listings,")
    print("                sale price, pct above list, days pending, price cuts,")
    print("                sale-to-list ratio")
    print()


# Run the full Zillow preprocessing workflow and write both output datasets.
def main() -> None:
    zip_base = build_zip_features()
    metro_base = build_metro_features()
    print_summary(zip_base, metro_base)


if __name__ == "__main__":
    main()
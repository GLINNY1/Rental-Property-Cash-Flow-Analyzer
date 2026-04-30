"""
Rental Property Cash Flow Predictor — Improved Version
=====================================================================
Takes a property's details as input and predicts whether it will
generate positive cash flow as a rental investment.

Improvements over v1:
  - Bedroom-adjusted rent estimate using ZHVI bedroom tier ratios
  - Real ZIP-level vacancy rate from Census (not flat 5%)
  - State-level property tax rates (not flat 1.2%)
  - Zillow total monthly payment lookup (includes real mortgage rate)
  - User chooses down payment scenario (5%, 10%, or 20%)
  - All three down payment scenarios shown side by side

Run from project root:
    python src/predict.py
=====================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve project root reliably for script and notebook execution.
def _get_root_dir():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == "src":
            return os.path.dirname(script_dir)
        return script_dir
    except NameError:
        if "PROJECT_ROOT" in globals():
            return PROJECT_ROOT  # noqa: F821
        return os.getcwd()

ROOT_DIR   = _get_root_dir()
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputData")

# ── State-level effective property tax rates (%) ──────────────────────────────
# Source: Lincoln Institute of Land Policy / Tax Foundation averages
STATE_PROPERTY_TAX = {
    'AL': 0.0040, 'AK': 0.0100, 'AZ': 0.0060, 'AR': 0.0062, 'CA': 0.0076,
    'CO': 0.0051, 'CT': 0.0173, 'DE': 0.0057, 'DC': 0.0085, 'FL': 0.0089,
    'GA': 0.0092, 'HI': 0.0028, 'ID': 0.0063, 'IL': 0.0205, 'IN': 0.0085,
    'IA': 0.0153, 'KS': 0.0138, 'KY': 0.0086, 'LA': 0.0055, 'ME': 0.0109,
    'MD': 0.0099, 'MA': 0.0110, 'MI': 0.0154, 'MN': 0.0108, 'MS': 0.0065,
    'MO': 0.0099, 'MT': 0.0084, 'NE': 0.0162, 'NV': 0.0060, 'NH': 0.0186,
    'NJ': 0.0247, 'NM': 0.0079, 'NY': 0.0172, 'NC': 0.0080, 'ND': 0.0098,
    'OH': 0.0153, 'OK': 0.0090, 'OR': 0.0097, 'PA': 0.0153, 'PR': 0.0083,
    'RI': 0.0153, 'SC': 0.0057, 'SD': 0.0117, 'TN': 0.0067, 'TX': 0.0180,
    'UT': 0.0058, 'VT': 0.0183, 'VI': 0.0100, 'VA': 0.0082, 'WA': 0.0093,
    'WV': 0.0059, 'WI': 0.0185, 'WY': 0.0057,
}
DEFAULT_TAX_RATE = 0.0100  # 1.0% national average fallback

# ── Homeowner's insurance estimate by state (% of home value/year) ─────────────
STATE_INSURANCE = {
    'AL': 0.0081, 'AK': 0.0046, 'AZ': 0.0048, 'AR': 0.0086, 'CA': 0.0060,
    'CO': 0.0073, 'CT': 0.0059, 'DE': 0.0045, 'DC': 0.0051, 'FL': 0.0119,
    'GA': 0.0076, 'HI': 0.0031, 'ID': 0.0055, 'IL': 0.0061, 'IN': 0.0069,
    'IA': 0.0070, 'KS': 0.0107, 'KY': 0.0073, 'LA': 0.0109, 'ME': 0.0053,
    'MD': 0.0052, 'MA': 0.0059, 'MI': 0.0063, 'MN': 0.0075, 'MS': 0.0094,
    'MO': 0.0083, 'MT': 0.0066, 'NE': 0.0093, 'NV': 0.0049, 'NH': 0.0050,
    'NJ': 0.0063, 'NM': 0.0060, 'NY': 0.0056, 'NC': 0.0068, 'ND': 0.0080,
    'OH': 0.0063, 'OK': 0.0107, 'OR': 0.0051, 'PA': 0.0055, 'PR': 0.0060,
    'RI': 0.0071, 'SC': 0.0075, 'SD': 0.0082, 'TN': 0.0073, 'TX': 0.0101,
    'UT': 0.0049, 'VT': 0.0049, 'VI': 0.0060, 'VA': 0.0059, 'WA': 0.0051,
    'WV': 0.0056, 'WI': 0.0058, 'WY': 0.0060,
}
DEFAULT_INSURANCE_RATE = 0.0060  # 0.6% national average fallback

# Assumptions
MAINTENANCE_PCT = 0.01   # 1% of home value/year (standard rule of thumb)
MGMT_PCT        = 0.08   # 8% of rent/year for property management


# Load trained models, scaler, and expected feature list from disk.
def load_models():
    print("  Loading models...", end=" ", flush=True)
    lr       = joblib.load(os.path.join(OUTPUT_DIR, "model_logistic_regression.pkl"))
    dt       = joblib.load(os.path.join(OUTPUT_DIR, "model_decision_tree.pkl"))
    nn       = joblib.load(os.path.join(OUTPUT_DIR, "model_neural_network.pkl"))
    scaler   = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(OUTPUT_DIR, "feature_list.pkl"))
    print("done ✓")
    return lr, dt, nn, scaler, features


# Load Zillow/Census feature tables and normalize ZIP code formats.
def load_market_data():
    print("  Loading market data...", end=" ", flush=True)
    zip_df    = pd.read_csv(os.path.join(OUTPUT_DIR, "zillow_zip_features.csv"),
                            low_memory=False)
    metro_df  = pd.read_csv(os.path.join(OUTPUT_DIR, "zillow_metro_features.csv"),
                            low_memory=False)
    census_df = pd.read_csv(os.path.join(OUTPUT_DIR, "census_zip_features.csv"),
                            low_memory=False)
    zip_df["zip_code"]    = zip_df["zip_code"].astype(str).str.zfill(5)
    census_df["zip_code"] = census_df["zip_code"].astype(str).str.zfill(5)
    print("done ✓")
    return zip_df, metro_df, census_df


# Collect and validate property details entered by the user.
def get_user_input():
    print()
    print("=" * 58)
    print("  Enter Property Details")
    print("  (copy directly from Zillow or any listing)")
    print("=" * 58)

    while True:
        try:
            price = float(input("\n  Purchase price ($):           ")
                          .replace(",", "").replace("$", ""))
            if price <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a valid price.")

    while True:
        try:
            beds = int(input("  Bedrooms:                     "))
            if beds < 1 or beds > 10:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a number between 1 and 10.")

    while True:
        try:
            baths = float(input("  Bathrooms:                    "))
            if baths < 1 or baths > 15:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a number between 1 and 15.")

    while True:
        try:
            sqft = int(input("  Square footage:               ").replace(",", ""))
            if sqft < 200 or sqft > 10000:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a value between 200 and 10,000.")

    while True:
        zip_code = input("  ZIP code:                     ").strip().zfill(5)
        if len(zip_code) == 5 and zip_code.isdigit():
            break
        print("  Please enter a valid 5-digit ZIP code.")

    return price, beds, baths, sqft, zip_code


# Adjust rent estimate by bedroom tier using ZIP-level ZHVI ratios.
def adjust_rent_for_bedrooms(base_rent, beds, zip_row):
    """
    Scale base rent up/down based on bedroom count using ZHVI tier ratios.
    If a 3BR home is worth 1.5x the median, we expect rent to be ~1.5x too.
    """
    br_col = f"zhvi_{beds}br" if beds <= 4 else "zhvi_4br"
    mid    = zip_row.get("zhvi_mid_tier") if zip_row is not None else None
    br_val = zip_row.get(br_col) if zip_row is not None else None

    if mid and br_val and mid > 0 and not pd.isna(mid) and not pd.isna(br_val):
        ratio = br_val / mid
        # Cap ratio to reasonable range (0.5x to 2.5x) to avoid outlier skew
        ratio = max(0.5, min(2.5, ratio))
        return base_rent * ratio

    # Fallback: simple multipliers if no ZHVI tier data
    fallback = {1: 0.75, 2: 0.90, 3: 1.10, 4: 1.30}
    return base_rent * fallback.get(beds, 1.0)


# Retrieve ZIP/metro/census signals with median-based fallbacks.
def lookup_market_features(zip_code, zip_df, metro_df, census_df):
    """Look up all market features for a given ZIP code."""
    features = {}
    warnings = []
    state_abbr = None

    # ── ZIP-level Zillow ──────────────────────────────────────────────────────
    zip_row_df = zip_df[zip_df["zip_code"] == zip_code]
    if zip_row_df.empty:
        warnings.append(f"ZIP {zip_code} not in Zillow data — using national medians")
        zip_row = None
    else:
        zip_row = zip_row_df.iloc[0].to_dict()
        state_abbr = str(zip_row.get("StateName", "")) or None

    zillow_zip_cols = ["zhvi_mid_tier", "zhvi_1br", "zhvi_2br",
                       "zhvi_3br", "zhvi_4br", "zori_rent"]
    for col in zillow_zip_cols:
        val = zip_row.get(col) if zip_row else None
        features[col] = val if val is not None and not pd.isna(val) \
                        else zip_df[col].median()

    # ── Census ────────────────────────────────────────────────────────────────
    cen_row_df = census_df[census_df["zip_code"] == zip_code]
    if cen_row_df.empty:
        warnings.append(f"ZIP {zip_code} not in Census data — using national medians")
        cen_row = None
    else:
        cen_row = cen_row_df.iloc[0].to_dict()

    census_cols = ["median_gross_rent", "median_household_income",
                   "renter_pct", "vacancy_rate", "income_to_rent_ratio",
                   "total_population", "median_home_value"]
    for col in census_cols:
        val = cen_row.get(col) if cen_row else None
        features[col] = val if val is not None and not pd.isna(val) \
                        else census_df[col].median()

    # ── Metro-level (state median) ────────────────────────────────────────────
    metro_signal_cols = [
        "zhvi_metro_mid_tier", "zori_metro_rent", "market_heat_index",
        "median_list_price", "median_sale_price", "mean_days_pending",
        "share_price_cut", "pct_sold_above_list", "home_value_forecast_yoy_pct",
        "for_sale_inventory", "sale_to_list_ratio",
        "total_pmt_5pct", "total_pmt_10pct", "total_pmt_20pct",
        "mortgage_pmt_5pct", "mortgage_pmt_10pct", "mortgage_pmt_20pct",
    ]

    if state_abbr and state_abbr in metro_df["StateName"].values:
        state_metro = metro_df[metro_df["StateName"] == state_abbr]
        for col in metro_signal_cols:
            if col in state_metro.columns:
                val = state_metro[col].median()
                features[col] = val if not pd.isna(val) else metro_df[col].median()
    else:
        warnings.append("State not matched in metro data — using national medians")
        for col in metro_signal_cols:
            if col in metro_df.columns:
                features[col] = metro_df[col].median()

    features["state_abbr"] = state_abbr
    return features, warnings, zip_row


# Compute cash flow metrics for one down-payment scenario.
def compute_cash_flow(price, beds, monthly_rent, down_pct,
                      vacancy_rate, state_abbr, total_pmt_col, features):
    """
    Compute annual cash flow using real data wherever available.
    Uses Zillow total monthly payment if available, otherwise estimates.
    """
    down_payment   = price * down_pct
    annual_rent    = monthly_rent * 12

    # Use real ZIP vacancy rate (clamped to 3%–25% range)
    vac_rate       = max(0.03, min(0.25, vacancy_rate)) if not pd.isna(vacancy_rate) else 0.05
    egi            = annual_rent * (1 - vac_rate)

    # Operating expenses
    annual_maint   = price * MAINTENANCE_PCT
    annual_mgmt    = annual_rent * MGMT_PCT
    total_opex     = annual_maint + annual_mgmt

    noi            = egi - total_opex
    cap_rate       = noi / price if price > 0 else 0

    # ── Annual ownership cost ─────────────────────────────────────────────────
    # Prefer Zillow total payment (already includes mortgage + tax + insurance)
    # Scale it to the actual purchase price since Zillow's figure is for
    # the metro median home value
    zillow_total   = features.get(total_pmt_col)
    zillow_zhvi    = features.get("zhvi_metro_mid_tier")

    if zillow_total and zillow_zhvi and not pd.isna(zillow_total) \
            and not pd.isna(zillow_zhvi) and zillow_zhvi > 0:
        # Scale payment proportionally to actual price vs metro median
        scale_factor    = price / zillow_zhvi
        annual_total_pmt = zillow_total * 12 * scale_factor
        pmt_source      = "Zillow (scaled to purchase price)"
    else:
        # Fallback: compute manually
        rate_monthly   = 0.068 / 12
        n              = 360
        loan           = price * (1 - down_pct)
        monthly_mtg    = loan * (rate_monthly * (1 + rate_monthly)**n) / \
                         ((1 + rate_monthly)**n - 1) if loan > 0 else 0
        tax_rate       = STATE_PROPERTY_TAX.get(state_abbr, DEFAULT_TAX_RATE)
        ins_rate       = STATE_INSURANCE.get(state_abbr, DEFAULT_INSURANCE_RATE)
        annual_total_pmt = (monthly_mtg * 12) + (price * tax_rate) + (price * ins_rate)
        pmt_source      = "Estimated (6.8% rate + state tax/insurance)"

    annual_cf  = egi - annual_total_pmt
    monthly_cf = annual_cf / 12
    coc        = annual_cf / down_payment if down_payment > 0 else 0

    return {
        "monthly_rent":       monthly_rent,
        "annual_rent":        annual_rent,
        "vacancy_rate_used":  vac_rate,
        "egi":                egi,
        "noi":                noi,
        "cap_rate":           cap_rate,
        "down_payment":       down_payment,
        "annual_total_pmt":   annual_total_pmt,
        "monthly_total_pmt":  annual_total_pmt / 12,
        "annual_cf":          annual_cf,
        "monthly_cf":         monthly_cf,
        "coc":                coc,
        "pmt_source":         pmt_source,
    }


# Build a model-ready single-row feature frame in training column order.
def build_feature_row(price, beds, baths, sqft, monthly_rent, market_features, feature_list):
    """Build the feature vector the model expects."""
    annual_rent        = monthly_rent * 12
    prop_price_to_rent = price / annual_rent if annual_rent > 0 else 999
    prop_gross_yield   = annual_rent / price if price > 0 else 0

    row = {
        "bed":                  beds,
        "bath":                 baths,
        "house_size":           sqft,
        "price_per_sqft":       price / sqft if sqft > 0 else 0,
        "bed_bath_ratio":       beds / baths if baths > 0 else 0,
        "prop_price_to_rent":   prop_price_to_rent,
        "prop_gross_yield":     prop_gross_yield,
    }
    row.update(market_features)
    # Force property-specific values to win over any market-level values
    row["prop_price_to_rent"] = prop_price_to_rent
    row["prop_gross_yield"]   = prop_gross_yield
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in feature_list}])
    X = X.fillna(X.median())
    return X


# Run all trained models and return class predictions plus probabilities.
def run_models(price, beds, baths, sqft, monthly_rent, market_features,
               lr, dt, nn, scaler, feature_list):
    """Run all three models and return predictions."""
    X        = build_feature_row(price, beds, baths, sqft, monthly_rent, market_features, feature_list)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_list)

    results = []
    for name, model, X_in in [
        ("Logistic Regression", lr, X_scaled),
        ("Decision Tree",       dt, X),
        ("Neural Network",      nn, X_scaled),
    ]:
        pred  = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0][1]
        results.append((name, pred, proba))

    return results


# Print market context, cash flow scenarios, model outputs, and verdict.
def print_report(price, beds, baths, sqft, zip_code,
                 market_features, zip_row, warnings,
                 monthly_rent, model_results,
                 cf_5, cf_10, cf_20):
    """Print the full prediction report."""

    state_abbr   = market_features.get("state_abbr", "N/A")
    vacancy_used = cf_20["vacancy_rate_used"]

    # ── Market Data ───────────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print(f"  Market Data for ZIP {zip_code}  (State: {state_abbr})")
    print("=" * 58)

    rows = [
        ("Zillow home value (ZIP):",      market_features.get("zhvi_mid_tier"),      "$",    0),
        ("Zillow home value (Metro):",    market_features.get("zhvi_metro_mid_tier"), "$",   0),
        ("Estimated monthly rent:",       monthly_rent,                               "$",   0),
        ("Median household income:",      market_features.get("median_household_income"), "$", 0),
        ("Renter percentage:",            market_features.get("renter_pct"),          "%",   1),
        ("Vacancy rate (ZIP actual):",    market_features.get("vacancy_rate"),        "%",   1),
        ("Market heat index:",            market_features.get("market_heat_index"),   "",    2),
        ("Mean days to pending:",         market_features.get("mean_days_pending"),   " days", 1),
        ("1-yr value forecast:",          market_features.get("home_value_forecast_yoy_pct"), "%", 2),
        ("State property tax rate:",      STATE_PROPERTY_TAX.get(state_abbr, DEFAULT_TAX_RATE), "%", 2),
    ]

    for label, val, suffix, fmt in rows:
        if val is None or (isinstance(val, float) and (np.isnan(val) or abs(val) > 500)):
            print(f"  {label:<38} N/A")
        elif suffix == "$":
            print(f"  {label:<38} ${val:>12,.0f}")
        elif suffix == "%":
            print(f"  {label:<38} {val*100:>11.1f}%")
        elif suffix == " days":
            print(f"  {label:<38} {val:>10.1f} days")
        else:
            print(f"  {label:<38} {val:>12.2f}")

    if warnings:
        print()
        for w in warnings:
            print(f"  ⚠  {w}")

    # ── Rent Explanation ──────────────────────────────────────────────────────
    print()
    print(f"  Rent Note: Base Census rent adjusted for {beds}BR using")
    print(f"  Zillow home value tier ratios → ${monthly_rent:,.0f}/mo")
    print(f"  Vacancy used: {vacancy_used*100:.1f}% (ZIP actual, clamped 3–25%)")
    print(f"  Payment source: {cf_20['pmt_source']}")

    # ── Cash Flow (3 scenarios) ───────────────────────────────────────────────
    print()
    print("=" * 58)
    print("  Cash Flow Scenarios")
    print("=" * 58)
    print(f"  {'':30} {'5% Down':>8}  {'10% Down':>8}  {'20% Down':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Down payment':30} ${cf_5['down_payment']:>7,.0f}  ${cf_10['down_payment']:>7,.0f}  ${cf_20['down_payment']:>7,.0f}")
    print(f"  {'Monthly rent':30} ${monthly_rent:>7,.0f}  ${monthly_rent:>7,.0f}  ${monthly_rent:>7,.0f}")
    print(f"  {'Monthly total payment':30} ${cf_5['monthly_total_pmt']:>7,.0f}  ${cf_10['monthly_total_pmt']:>7,.0f}  ${cf_20['monthly_total_pmt']:>7,.0f}")

    for label, key, is_money in [
        ("Annual cash flow",    "annual_cf",  True),
        ("Monthly cash flow",   "monthly_cf", True),
        ("Cap rate",            "cap_rate",   False),
        ("Cash-on-cash return", "coc",        False),
    ]:
        v5  = cf_5[key]
        v10 = cf_10[key]
        v20 = cf_20[key]
        if is_money:
            s5  = f"${v5:>+8,.0f}" if v5 >= 0 else f"-${abs(v5):>7,.0f}"
            s10 = f"${v10:>+8,.0f}" if v10 >= 0 else f"-${abs(v10):>7,.0f}"
            s20 = f"${v20:>+8,.0f}" if v20 >= 0 else f"-${abs(v20):>7,.0f}"
            print(f"  {label:<30} {s5}  {s10}  {s20}")
        else:
            print(f"  {label:<30} {v5*100:>7.1f}%  {v10*100:>7.1f}%  {v20*100:>7.1f}%")

    print(f"  {'NOI (before mortgage)':30} ${cf_20['noi']:>+8,.0f}")

    # ── Model Predictions (20% down) ──────────────────────────────────────────
    print()
    print("=" * 58)
    print("  Model Predictions  (20% down scenario)")
    print("=" * 58)

    votes = 0
    for name, pred, proba in model_results:
        label   = "Cash Flow POSITIVE ✓" if pred == 1 else "Cash Flow NEGATIVE ✗"
        bar_len = int(proba * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {name:<22} {label}")
        print(f"  {'':22} Positive probability: {proba*100:.1f}%  [{bar}]")
        print()
        if pred == 1:
            votes += 1

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 58)
    verdicts = {
        3: "STRONG BUY  ✓✓✓  All 3 models predict POSITIVE cash flow",
        2: "LIKELY BUY  ✓✓✗  2 of 3 models predict POSITIVE cash flow",
        1: "LIKELY PASS ✗✗✓  2 of 3 models predict NEGATIVE cash flow",
        0: "STRONG PASS ✗✗✗  All 3 models predict NEGATIVE cash flow",
    }
    print(f"  VERDICT: {verdicts[votes]}")
    print("=" * 58)


# Orchestrate user input, feature lookup, inference, and reporting loop.
def main():
    print()
    print("=" * 58)
    print("  Rental Property Cash Flow Predictor  v2")
    print("  Logistic Regression + Decision Tree +")
    print("  Neural Network")
    print("=" * 58)
    print()
    print("  Initializing...")

    lr, dt, nn, scaler, feature_list = load_models()
    zip_df, metro_df, census_df      = load_market_data()

    while True:
        price, beds, baths, sqft, zip_code = get_user_input()

        market_features, warnings, zip_row = lookup_market_features(
            zip_code, zip_df, metro_df, census_df
        )

        state_abbr = market_features.get("state_abbr")

        # Bedroom-adjusted rent estimate
        base_rent    = market_features.get("median_gross_rent", census_df["median_gross_rent"].median())
        monthly_rent = adjust_rent_for_bedrooms(base_rent, beds, zip_row)

        # Vacancy rate from Census ZIP data
        vacancy_rate = market_features.get("vacancy_rate", 0.05)

        # Compute cash flow for all 3 down payment scenarios
        cf_5  = compute_cash_flow(price, beds, monthly_rent, 0.05,
                                   vacancy_rate, state_abbr,
                                   "total_pmt_5pct", market_features)
        cf_10 = compute_cash_flow(price, beds, monthly_rent, 0.10,
                                   vacancy_rate, state_abbr,
                                   "total_pmt_10pct", market_features)
        cf_20 = compute_cash_flow(price, beds, monthly_rent, 0.20,
                                   vacancy_rate, state_abbr,
                                   "total_pmt_20pct", market_features)

        # Run models (always on 20% down scenario features)
        model_results = run_models(
            price, beds, baths, sqft, monthly_rent, market_features,
            lr, dt, nn, scaler, feature_list
        )

        # Print full report
        print_report(
            price, beds, baths, sqft, zip_code,
            market_features, zip_row, warnings,
            monthly_rent, model_results,
            cf_5, cf_10, cf_20
        )

        print()
        again = input("  Predict another property? (y/n): ").strip().lower()
        if again != "y":
            print()
            print("  Thanks for using the Rental Property Cash Flow Predictor.")
            print()
            break


if __name__ == "__main__":
    main()
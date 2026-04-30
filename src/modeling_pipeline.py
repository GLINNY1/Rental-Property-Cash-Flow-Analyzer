from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("outputData/master.csv")
OUTPUT_PATH = Path("outputData")
FIGURES_PATH = Path("outputData/figures")
SCORES_PATH = OUTPUT_PATH / "investor_scores.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
TARGET_COL = "is_cash_flow_positive"

LEAKAGE_KEYWORDS = ("cash_flow", "coc_return", "annual", "monthly_payment")
IMBALANCE_THRESHOLD = 0.70

# ── Feature Groups ────────────────────────────────────────────────────────────
# LEAKAGE NOTE: The following columns are intentionally excluded because they
# are derived directly from the label formula (cash flow = rent - expenses).
# Including them would let the model "cheat" by seeing the answer embedded
# in the inputs, producing unrealistically high AUC scores (>0.99).
#
# EXCLUDED (leaky):
#   best_rent_estimate    — used directly to compute annual_cash_flow label
#   best_value_estimate   — used directly in expense calculation for label
#   noi                   — Net Operating Income, intermediate step in label
#   cap_rate              — derived from NOI, leaks label
#   gross_rent_yield_calc — derived from rent/price, leaks label
#   zori_rent             — same signal as best_rent_estimate
#   acre_lot              — dropped during realtor preprocessing, not in data
#
# KEPT: only features a real investor would observe BEFORE buying the property.

PROPERTY_FEATURES = [
    "bed",                  # number of bedrooms
    "bath",                 # number of bathrooms
    "house_size",           # square footage
    "price_per_sqft",       # purchase price efficiency
    "bed_bath_ratio",       # layout signal
    "prop_price_to_rent",   # property-specific price-to-rent ratio
    "prop_gross_yield",     # property-specific gross rent yield
]

MARKET_FEATURES = [
    "zhvi_mid_tier",                  # market home value benchmark (ZIP)
    "zhvi_metro_mid_tier",            # metro-level home value benchmark
    "market_heat_index",              # supply/demand balance score
    "median_list_price",              # what homes are listed for
    "median_sale_price",              # what homes actually sell for
    "mean_days_pending",              # how fast homes go under contract
    "share_price_cut",                # market softness indicator
    "pct_sold_above_list",            # market competitiveness
    "home_value_forecast_yoy_pct",    # 1-year appreciation forecast
    "for_sale_inventory",             # supply level
    "sale_to_list_ratio",             # sale vs ask price ratio
]

NEIGHBORHOOD_FEATURES = [
    "median_household_income",   # ZIP income level
    "median_gross_rent",         # Census rent benchmark (not Zillow)
    "renter_pct",                # share of renters in ZIP
    "vacancy_rate",              # rental market health
    "income_to_rent_ratio",      # affordability of rent for locals
    "total_population",          # market size
]

ALL_FEATURES = (
    PROPERTY_FEATURES
    + MARKET_FEATURES
    + NEIGHBORHOOD_FEATURES
)

# Clustering uses market + neighborhood signals only (no property-level features)
# Also excludes leaky columns
CLUSTER_FEATURES = [
    "zhvi_mid_tier",
    "median_gross_rent",
    "median_household_income",
    "vacancy_rate",
    "market_heat_index",
    "renter_pct",
]


def load_master_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, low_memory=False)


def run_stage_0(master: pd.DataFrame) -> None:
    print("STAGE 0 - Orientation & Leakage Audit")
    print(f"Loading data from: {DATA_PATH}")

    print("\n[Shape]")
    print(f"Rows: {master.shape[0]:,}")
    print(f"Columns: {master.shape[1]:,}")

    print("\n[Dtypes]")
    print(master.dtypes.to_string())

    print("\n[Null Counts - All Columns]")
    print(master.isna().sum().sort_values(ascending=False).to_string())

    print(f"\n[Class Distribution - {TARGET_COL}]")
    if TARGET_COL not in master.columns:
        raise KeyError(f"Target column not found: {TARGET_COL}")

    target_series = master[TARGET_COL].dropna()
    class_counts = target_series.value_counts(dropna=False).sort_index()
    class_ratios = target_series.value_counts(normalize=True).sort_index()

    print("Counts:")
    print(class_counts.to_string())
    print("Ratios:")
    print(class_ratios.to_string(float_format=lambda x: f"{x:.4f}"))

    majority_ratio = class_ratios.max() if not class_ratios.empty else 0.0
    imbalance_flagged = majority_ratio > IMBALANCE_THRESHOLD
    if imbalance_flagged:
        print(
            f"IMBALANCE FLAG: majority class ratio {majority_ratio:.4f} "
            f"exceeds {IMBALANCE_THRESHOLD:.2f}."
        )
    else:
        print(
            f"Class balance OK: majority class ratio {majority_ratio:.4f} "
            f"does not exceed {IMBALANCE_THRESHOLD:.2f}."
        )

    print("\n[Leakage Candidates]")
    leakage_columns = [
        col for col in master.columns if any(keyword in col.lower() for keyword in LEAKAGE_KEYWORDS)
    ]
    if leakage_columns:
        for col in leakage_columns:
            print(col)
    else:
        print("No leakage candidates found by keyword rules.")

    print("\n--- STAGE 0 COMPLETE ---")
    print("Awaiting confirmation before Stage 1.")


def is_class_imbalance_flagged(y: pd.Series) -> bool:
    ratios = y.value_counts(normalize=True)
    majority_ratio = ratios.max() if not ratios.empty else 0.0
    return majority_ratio > IMBALANCE_THRESHOLD


def run_stage_1(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("\nSTAGE 1 - Feature Set Construction")

    available_features = []
    missing_features = []

    for feature in ALL_FEATURES:
        if feature in master.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)

    if missing_features:
        print("[Warnings - Missing Features Skipped]")
        for feature in missing_features:
            print(f"WARNING: Missing feature skipped -> {feature}")

    if not available_features:
        raise ValueError("No requested features found in dataset after filtering.")

    rows_before_target_drop = len(master)
    model_df = master.dropna(subset=[TARGET_COL]).copy()
    rows_after_target_drop = len(model_df)
    dropped_target_null_rows = rows_before_target_drop - rows_after_target_drop

    X = model_df[available_features].copy()
    y = model_df[TARGET_COL].copy()

    nulls_before_impute = int(X.isna().sum().sum())
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    nulls_after_impute = int(X.isna().sum().sum())

    print("[Selected Features]")
    for feature in available_features:
        print(feature)

    print("\n[Stage 1 Summary]")
    print(f"Rows dropped where {TARGET_COL} is null: {dropped_target_null_rows:,}")
    print(f"Final feature matrix shape: {X.shape[0]:,} rows x {X.shape[1]:,} columns")
    print(f"Total null values before median imputation: {nulls_before_impute:,}")
    print(f"Total null values after median imputation: {nulls_after_impute:,}")

    print("\n--- STAGE 1 COMPLETE ---")
    print("Awaiting confirmation before Stage 2.")
    return X, y


def _print_split_distribution(name: str, y_split: pd.Series) -> None:
    counts = y_split.value_counts().sort_index()
    ratios = y_split.value_counts(normalize=True).sort_index()
    print(f"{name} count: {len(y_split):,}")
    print("  Class counts:")
    print("  " + counts.to_string().replace("\n", "\n  "))
    print("  Class ratios:")
    print("  " + ratios.to_string(float_format=lambda x: f"{x:.4f}").replace("\n", "\n  "))


def run_stage_2(
    X: pd.DataFrame, y: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    print("\nSTAGE 2 - Train / Validation / Test Split")

    # First split off test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split remaining 85% into train/val so val is 15% of full dataset.
    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print("[Split Sizes]")
    print(f"Train: {len(X_train):,} ({len(X_train)/len(X):.2%})")
    print(f"Validation: {len(X_val):,} ({len(X_val)/len(X):.2%})")
    print(f"Test: {len(X_test):,} ({len(X_test)/len(X):.2%})")

    print("\n[Class Balance By Split]")
    _print_split_distribution("Train", y_train)
    _print_split_distribution("Validation", y_val)
    _print_split_distribution("Test", y_test)

    print("\n[Feature Matrix Versions]")
    print("Raw matrices prepared: X_train, X_val, X_test (for Decision Tree).")
    print("Scaled matrices prepared: X_train_scaled, X_val_scaled, X_test_scaled (for LR/NN).")

    print("\n--- STAGE 2 COMPLETE ---")
    print("Awaiting confirmation before Stage 3.")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    )


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_stage_3(
    X_train_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    feature_names: list[str],
    results: dict,
) -> LogisticRegression:
    print("\nSTAGE 3 - Logistic Regression")

    use_balanced = is_class_imbalance_flagged(y_train)
    class_weight = "balanced" if use_balanced else None
    if use_balanced:
        print("Class imbalance detected -> using class_weight='balanced'.")
    else:
        print("Class imbalance not flagged -> using default class weights.")

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    lr_model.fit(X_train_scaled, y_train)

    y_val_pred = lr_model.predict(X_val_scaled)
    y_val_proba = lr_model.predict_proba(X_val_scaled)[:, 1]

    lr_metrics = compute_binary_metrics(y_val, y_val_pred, y_val_proba)
    results["logistic_regression"] = lr_metrics

    print("[Validation Metrics]")
    print(f"Accuracy : {lr_metrics['accuracy']:.4f}")
    print(f"Precision: {lr_metrics['precision']:.4f}")
    print(f"Recall   : {lr_metrics['recall']:.4f}")
    print(f"F1       : {lr_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {lr_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(lr_metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_string())

    coef = lr_model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=True)

    fig_height = max(6, int(len(coef_df) * 0.35))
    plt.figure(figsize=(10, fig_height))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.axvline(x=0, color="black", linewidth=1)
    plt.title("Logistic Regression Coefficients (Ranked by |Magnitude|)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    lr_plot_path = FIGURES_PATH / "lr_coefficients.png"
    plt.savefig(lr_plot_path, dpi=150)
    plt.close()
    print(f"Saved coefficient plot -> {lr_plot_path}")

    print("\n--- STAGE 3 COMPLETE ---")
    print("Awaiting confirmation before Stage 4.")
    return lr_model


def run_stage_4(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    feature_names: list[str],
    results: dict,
) -> DecisionTreeClassifier:
    print("\nSTAGE 4 - Decision Tree")

    use_balanced = is_class_imbalance_flagged(y_train)
    class_weight = "balanced" if use_balanced else None
    if use_balanced:
        print("Class imbalance detected -> using class_weight='balanced' for DT.")
    else:
        print("Class imbalance not flagged -> using default class weights for DT.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    depth_scores: dict[int, float] = {}

    for depth in range(3, 13):
        fold_scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_valid = y_train.iloc[valid_idx]

            dt_fold = DecisionTreeClassifier(
                max_depth=depth,
                random_state=RANDOM_STATE,
                class_weight=class_weight,
            )
            dt_fold.fit(X_fold_train, y_fold_train)
            y_fold_pred = dt_fold.predict(X_fold_valid)
            fold_scores.append(f1_score(y_fold_valid, y_fold_pred, zero_division=0))

        depth_scores[depth] = float(sum(fold_scores) / len(fold_scores))

    best_depth = max(depth_scores, key=depth_scores.get)
    print("[CV Results - Mean F1 by max_depth]")
    for depth, score in depth_scores.items():
        print(f"max_depth={depth}: F1={score:.4f}")
    print(f"Best max_depth selected: {best_depth}")

    dt_model = DecisionTreeClassifier(
        max_depth=best_depth,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    dt_model.fit(X_train, y_train)

    y_val_pred = dt_model.predict(X_val)
    y_val_proba = dt_model.predict_proba(X_val)[:, 1]

    dt_metrics = compute_binary_metrics(y_val, y_val_pred, y_val_proba)
    dt_metrics["best_max_depth"] = best_depth
    results["decision_tree"] = dt_metrics

    print("[Validation Metrics]")
    print(f"Accuracy : {dt_metrics['accuracy']:.4f}")
    print(f"Precision: {dt_metrics['precision']:.4f}")
    print(f"Recall   : {dt_metrics['recall']:.4f}")
    print(f"F1       : {dt_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {dt_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(dt_metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_string())

    plt.figure(figsize=(24, 12))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=["0", "1"],
        filled=True,
        max_depth=4,
        fontsize=8,
    )
    plt.title("Decision Tree (Visual Depth Capped at 4)")
    plt.tight_layout()
    tree_path = FIGURES_PATH / "dt_tree.png"
    plt.savefig(tree_path, dpi=150)
    plt.close()
    print(f"Saved tree plot -> {tree_path}")

    importances_df = pd.DataFrame(
        {"feature": feature_names, "importance": dt_model.feature_importances_}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(importances_df["feature"], importances_df["importance"])
    plt.title("Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    imp_path = FIGURES_PATH / "dt_importances.png"
    plt.savefig(imp_path, dpi=150)
    plt.close()
    print(f"Saved importance plot -> {imp_path}")

    print("\n--- STAGE 4 COMPLETE ---")
    print("Awaiting confirmation before Stage 5.")
    return dt_model


def run_stage_5(
    X_train_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    results: dict,
) -> MLPClassifier:
    print("\nSTAGE 5 - Neural Network")

    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.15,
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    nn_model.fit(X_train_scaled, y_train)

    y_val_pred = nn_model.predict(X_val_scaled)
    y_val_proba = nn_model.predict_proba(X_val_scaled)[:, 1]

    nn_metrics = compute_binary_metrics(y_val, y_val_pred, y_val_proba)
    results["neural_network"] = nn_metrics

    print("[Validation Metrics]")
    print(f"Accuracy : {nn_metrics['accuracy']:.4f}")
    print(f"Precision: {nn_metrics['precision']:.4f}")
    print(f"Recall   : {nn_metrics['recall']:.4f}")
    print(f"F1       : {nn_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {nn_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(nn_metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_string())

    loss_curve = nn_model.loss_curve_
    val_scores = nn_model.validation_scores_
    val_losses = [1 - score for score in val_scores]
    epochs = range(1, len(loss_curve) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_curve, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss (1 - score)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Network Loss Curve")
    plt.legend()
    plt.tight_layout()
    nn_plot_path = FIGURES_PATH / "nn_loss_curve.png"
    plt.savefig(nn_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss curve plot -> {nn_plot_path}")

    print("\n--- STAGE 5 COMPLETE ---")

    print("Awaiting confirmation before Stage 6.")
    return nn_model


def _choose_k_from_elbow(inertias: list[float], k_values: list[int]) -> int:
    if len(inertias) < 3:
        return k_values[0]
    second_diff = np.diff(inertias, n=2)
    elbow_idx = int(np.argmax(second_diff))
    # second difference is aligned to k_values[1:-1]
    return k_values[elbow_idx + 1]


def run_stage_6(master: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    print("\nSTAGE 6 - K-Means Market Segmentation")

    missing_cluster_features = [col for col in CLUSTER_FEATURES if col not in master.columns]
    if missing_cluster_features:
        raise KeyError(
            f"Missing required clustering features: {missing_cluster_features}"
        )
    if "zip_code" not in master.columns:
        raise KeyError("zip_code is required for Stage 6 clustering.")

    zip_level = (
        master.groupby("zip_code", as_index=False)[CLUSTER_FEATURES]
        .median()
        .copy()
    )

    zip_level[CLUSTER_FEATURES] = zip_level[CLUSTER_FEATURES].apply(
        lambda col: col.fillna(col.median())
    )

    scaler = StandardScaler()
    X_zip_scaled = scaler.fit_transform(zip_level[CLUSTER_FEATURES])

    k_values = list(range(2, 11))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_zip_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xticks(k_values)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Curve (ZIP-Level)")
    plt.tight_layout()
    elbow_path = FIGURES_PATH / "kmeans_elbow.png"
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"Saved elbow plot -> {elbow_path}")

    chosen_k = _choose_k_from_elbow(inertias, k_values)
    print(f"Chosen k from elbow heuristic: {chosen_k}")

    final_kmeans = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    zip_level["cluster_label"] = final_kmeans.fit_predict(X_zip_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_points = pca.fit_transform(X_zip_scaled)
    pca_df = pd.DataFrame(
        {
            "pc1": pca_points[:, 0],
            "pc2": pca_points[:, 1],
            "cluster_label": zip_level["cluster_label"],
        }
    )

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        pca_df["pc1"],
        pca_df["pc2"],
        c=pca_df["cluster_label"],
        cmap="tab10",
        s=18,
        alpha=0.75,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("ZIP-Level K-Means Clusters (PCA Projection)")
    plt.colorbar(scatter, label="Cluster Label")
    plt.tight_layout()
    pca_path = FIGURES_PATH / "kmeans_pca.png"
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"Saved PCA cluster plot -> {pca_path}")

    master_with_clusters = master.merge(
        zip_level[["zip_code", "cluster_label"]],
        on="zip_code",
        how="left",
    )

    cluster_profile = (
        master_with_clusters.groupby("cluster_label")[CLUSTER_FEATURES]
        .mean()
        .sort_values("median_gross_rent", ascending=False)
    )
    print("\n[Cluster Profile Table - Mean Features by Cluster (Sorted by cap_rate)]")
    print(cluster_profile.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\n--- STAGE 6 COMPLETE ---")
    print("Awaiting confirmation before Stage 7.")
    return master_with_clusters, chosen_k


def run_stage_7(
    lr_model: LogisticRegression,
    dt_model: DecisionTreeClassifier,
    nn_model: MLPClassifier,
    X_test: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    results: dict,
) -> None:
    print("\nSTAGE 7 - Evaluation Report")

    model_specs = [
        ("logistic_regression", "Logistic Regression", lr_model, X_test_scaled),
        ("decision_tree", "Decision Tree", dt_model, X_test),
        ("neural_network", "Neural Network", nn_model, X_test_scaled),
    ]

    report_rows = []
    roc_payload = {}
    confusion_payload = {}

    for key, label, model, X_eval in model_specs:
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1]
        metrics = compute_binary_metrics(y_test, y_pred, y_proba)
        results[f"{key}_test"] = metrics
        report_rows.append(
            {
                "Model": label,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            }
        )
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_payload[label] = (fpr, tpr, metrics["roc_auc"])
        confusion_payload[label] = metrics["confusion_matrix"]

    metrics_table = pd.DataFrame(report_rows).sort_values("ROC-AUC", ascending=False)
    print("[Test Set Metrics Comparison]")
    print(
        metrics_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    plt.figure(figsize=(8, 6))
    for label, (fpr, tpr, auc_val) in roc_payload.items():
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = FIGURES_PATH / "roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC curves plot -> {roc_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, cm) in zip(axes, confusion_payload.items()):
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_arr[i, j]}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    plt.tight_layout()
    cm_path = FIGURES_PATH / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrices plot -> {cm_path}")

    print("\n--- STAGE 7 COMPLETE ---")
    print("Awaiting confirmation before Stage 8.")


def main() -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    master = load_master_data()
    results: dict = {}

    run_stage_0(master)
    X, y = run_stage_1(master)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    ) = run_stage_2(X, y)
    _ = (X_train, X_val, X_test, y_test, X_test_scaled)
    lr_model = run_stage_3(
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        y_train=y_train,
        y_val=y_val,
        feature_names=list(X.columns),
        results=results,
    )
    dt_model = run_stage_4(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_names=list(X.columns),
        results=results,
    )
    nn_model = run_stage_5(
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        y_train=y_train,
        y_val=y_val,
        results=results,
    )
    # Save all trained models and scaler to disk
    joblib.dump(lr_model,     "outputData/model_logistic_regression.pkl")
    joblib.dump(dt_model,     "outputData/model_decision_tree.pkl")
    joblib.dump(nn_model,     "outputData/model_neural_network.pkl")
    joblib.dump(scaler,       "outputData/scaler.pkl")
    joblib.dump(ALL_FEATURES, "outputData/feature_list.pkl")
    print("Models saved to outputData/")

    run_stage_6(master)
    run_stage_7(
        lr_model=lr_model,
        dt_model=dt_model,
        nn_model=nn_model,
        X_test=X_test,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        results=results,
    )


if __name__ == "__main__":
    main()
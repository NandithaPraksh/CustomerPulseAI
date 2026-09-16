"""
CustomerPulse AI — End-to-end automated customer churn prediction.
All application logic is contained in this single file.
Run with:  streamlit run app.py
"""

import io
import warnings
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_DATE = pd.Timestamp("2011-09-30")
NON_PRODUCT_STOCK_CODES = {
    "D", "M", "B", "POST", "DOT", "AMAZONFEE", "BANK CHARGES",
    "CRUK", "PADS", "S", "TEST001", "TEST002",
}
REQUIRED_COLUMNS = {
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Country",
}
RANDOM_STATE = 42
TEST_SIZE = 0.20

RISK_LOW_MAX    = 0.33
RISK_MEDIUM_MAX = 0.66

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _show_fig(fig: plt.Figure) -> None:
    st.image(f"data:image/png;base64,{_fig_to_b64(fig)}", use_container_width=True)
    plt.close(fig)


def _risk_label(prob: float) -> str:
    if prob >= RISK_MEDIUM_MAX:
        return "HIGH"
    if prob >= RISK_LOW_MAX:
        return "MEDIUM"
    return "LOW"


def _risk_colour(label: str) -> str:
    return {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#27ae60"}[label]


def _metric_row(cols, labels, values, deltas=None):
    for i, col in enumerate(cols):
        delta = deltas[i] if deltas else None
        col.metric(labels[i], values[i], delta=delta)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Load CSV from uploaded bytes into a DataFrame."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", low_memory=False)
        return df
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return pd.DataFrame()


def validate_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check that the DataFrame has the required columns for this pipeline."""
    if df.empty:
        return False, ["The uploaded file is empty or could not be parsed."]
    issues = []
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        issues.append(
            f"Missing required columns: {', '.join(sorted(missing_cols))}. "
            "Please upload an Online Retail-compatible CSV."
        )
    if len(df) < 10:
        issues.append("Dataset has fewer than 10 rows — too small to analyse.")
    return (len(issues) == 0), issues


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> dict:
    """
    Full automatic cleaning pipeline for Online Retail-style data.
    Returns a dict of named DataFrames and a cleaning-summary dict.
    """
    raw_df = df.copy()
    n_raw = len(raw_df)

    # Step 1 – drop exact duplicates
    df_work = raw_df.drop_duplicates()
    n_dupes = n_raw - len(df_work)

    # Step 2 – drop rows without CustomerID
    df_work = df_work.dropna(subset=["CustomerID"])
    df_work["CustomerID"] = df_work["CustomerID"].astype(int).astype(str)
    n_no_cid = n_raw - n_dupes - len(df_work)

    # Step 3 – parse InvoiceDate; drop rows where it fails
    df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")
    n_bad_date = df_work["InvoiceDate"].isna().sum()
    df_work = df_work.dropna(subset=["InvoiceDate"])

    # Step 4 – cast Quantity and UnitPrice to numeric; drop unparseable
    df_work["Quantity"]  = pd.to_numeric(df_work["Quantity"],  errors="coerce")
    df_work["UnitPrice"] = pd.to_numeric(df_work["UnitPrice"], errors="coerce")
    df_work = df_work.dropna(subset=["Quantity", "UnitPrice"])

    # Step 5 – flag invoice types
    df_work["_inv_prefix"] = df_work["InvoiceNo"].astype(str).str[0].str.upper()
    df_work["_is_cancel"]  = df_work["_inv_prefix"] == "C"
    df_work["_is_adjust"]  = df_work["_inv_prefix"] == "A"

    # Step 6 – drop bad-debt adjustment rows (A-prefix; no real customer)
    df_work = df_work[~df_work["_is_adjust"]].copy()

    # Step 7 – separate returns/cancellations
    returns_df = df_work[df_work["_is_cancel"]].copy()
    n_cancel = len(returns_df)

    # Step 8 – work only on non-cancel rows from here
    df_purch = df_work[~df_work["_is_cancel"]].copy()

    # Step 9 – remove non-product StockCodes
    df_purch["_stock_upper"] = df_purch["StockCode"].astype(str).str.strip().str.upper()
    df_purch = df_purch[~df_purch["_stock_upper"].isin(NON_PRODUCT_STOCK_CODES)]

    # Step 10 – remove non-positive UnitPrice from purchases (zero or negative)
    df_purch = df_purch[df_purch["UnitPrice"] > 0]

    # Step 11 – remove non-positive Quantity from purchases (stock-correction rows)
    df_purch = df_purch[df_purch["Quantity"] > 0]

    # Step 12 – compute line revenue
    df_purch["LineRevenue"] = df_purch["Quantity"] * df_purch["UnitPrice"]

    # Step 13 – clean returns similarly
    returns_df["LineRevenue"] = returns_df["Quantity"].abs() * returns_df["UnitPrice"].abs()

    # Drop helper columns
    for col in ["_inv_prefix", "_is_cancel", "_is_adjust", "_stock_upper"]:
        if col in df_purch.columns:
            df_purch.drop(columns=[col], inplace=True)
        if col in returns_df.columns:
            returns_df.drop(columns=[col], inplace=True)

    n_final_purch = len(df_purch)
    n_removed_other = (n_raw - n_dupes - n_no_cid - n_bad_date
                       - n_cancel - n_final_purch)

    summary = {
        "original_rows":          n_raw,
        "duplicates_removed":     n_dupes,
        "missing_cid_removed":    n_no_cid,
        "bad_dates_removed":      n_bad_date,
        "cancellations_flagged":  n_cancel,
        "non_product_or_invalid": max(0, n_removed_other),
        "final_purchase_rows":    n_final_purch,
        "unique_customers":       df_purch["CustomerID"].nunique(),
        "date_min":               df_purch["InvoiceDate"].min(),
        "date_max":               df_purch["InvoiceDate"].max(),
    }

    return {
        "raw_df":      raw_df,
        "cleaned_df":  df_work,
        "purchases_df": df_purch,
        "returns_df":   returns_df,
        "summary":      summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def engineer_customer_features(
    purchases_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """
    Aggregate transaction rows into one row per customer.
    Uses only transactions on or before snapshot date to prevent leakage.
    """
    # ── restrict to pre-snapshot window ──────────────────────────────────────
    p = purchases_df[purchases_df["InvoiceDate"] <= snapshot].copy()
    r = returns_df[returns_df["InvoiceDate"] <= snapshot].copy()

    if p.empty:
        return pd.DataFrame()

    # ── purchase-side aggregations ────────────────────────────────────────────
    grp = p.groupby("CustomerID")

    agg = grp.agg(
        first_purchase   = ("InvoiceDate", "min"),
        last_purchase    = ("InvoiceDate", "max"),
        distinct_invoices= ("InvoiceNo",   "nunique"),
        total_items      = ("Quantity",    "sum"),
        monetary_total   = ("LineRevenue", "sum"),
        distinct_products= ("StockCode",   "nunique"),
    ).reset_index()

    # recency
    agg["recency_days"] = (snapshot - agg["last_purchase"]).dt.days
    # frequency  = number of distinct purchase dates
    freq = grp["InvoiceDate"].apply(lambda x: x.dt.normalize().nunique()).reset_index()
    freq.columns = ["CustomerID", "frequency"]
    agg = agg.merge(freq, on="CustomerID", how="left")

    # avg order value
    agg["avg_order_value"] = (
        agg["monetary_total"] / agg["distinct_invoices"].replace(0, np.nan)
    )

    # tenure
    agg["tenure_days"] = (snapshot - agg["first_purchase"]).dt.days

    # purchase span
    agg["purchase_span_days"] = (
        (agg["last_purchase"] - agg["first_purchase"]).dt.days
    )

    # inter-purchase gap (mean and std)
    def _gap_stats(dates: pd.Series):
        sorted_d = dates.dt.normalize().drop_duplicates().sort_values()
        if len(sorted_d) < 2:
            return pd.Series({"inter_purchase_gap_mean": np.nan,
                              "inter_purchase_gap_std":  np.nan})
        gaps = sorted_d.diff().dt.days.dropna()
        return pd.Series({"inter_purchase_gap_mean": gaps.mean(),
                          "inter_purchase_gap_std":  gaps.std()})

    gap_df = grp["InvoiceDate"].apply(_gap_stats).reset_index()
    # pivot inter_purchase columns if they came back stacked
    if "level_1" in gap_df.columns:
        gap_df = gap_df.pivot(index="CustomerID", columns="level_1",
                              values="InvoiceDate").reset_index()
        gap_df.columns.name = None

    agg = agg.merge(gap_df, on="CustomerID", how="left")

    # last 90 days behaviour (relative to snapshot)
    cutoff_90 = snapshot - pd.Timedelta(days=90)
    p90 = p[p["InvoiceDate"] >= cutoff_90]
    g90 = p90.groupby("CustomerID").agg(
        purchases_last_90d = ("InvoiceNo",    "nunique"),
        revenue_last_90d   = ("LineRevenue",  "sum"),
    ).reset_index()
    agg = agg.merge(g90, on="CustomerID", how="left")
    agg["purchases_last_90d"] = agg["purchases_last_90d"].fillna(0)
    agg["revenue_last_90d"]   = agg["revenue_last_90d"].fillna(0)

    # country (mode per customer)
    country_mode = (
        p.groupby("CustomerID")["Country"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        .reset_index()
        .rename(columns={"Country": "country"})
    )
    agg = agg.merge(country_mode, on="CustomerID", how="left")
    agg["is_uk"] = (agg["country"] == "United Kingdom").astype(int)

    # ── return-side aggregations ───────────────────────────────────────────────
    if not r.empty:
        ret_agg = r.groupby("CustomerID").agg(
            n_returns        = ("InvoiceNo",    "nunique"),
            cancelled_amount = ("LineRevenue",  "sum"),
        ).reset_index()
        agg = agg.merge(ret_agg, on="CustomerID", how="left")
    else:
        agg["n_returns"]        = 0
        agg["cancelled_amount"] = 0.0

    agg["n_returns"]        = agg["n_returns"].fillna(0)
    agg["cancelled_amount"] = agg["cancelled_amount"].fillna(0.0)
    agg["return_rate"]      = (
        agg["n_returns"] / agg["distinct_invoices"].replace(0, np.nan)
    ).fillna(0)

    # ── RFM quintile score ────────────────────────────────────────────────────
    # label names: rfm_recency_q, rfm_frequency_q, rfm_monetary_q
    for col, ascending in [("recency_days", False),
                            ("frequency",   True),
                            ("monetary_total", True)]:
        label = "rfm_" + col.split("_")[0] + "_q"
        agg[label] = pd.qcut(
            agg[col].rank(method="first", ascending=ascending),
            q=5, labels=[1, 2, 3, 4, 5]
        ).astype(float)

    agg["rfm_score"] = (
        agg["rfm_recency_q"] +
        agg["rfm_frequency_q"] +
        agg["rfm_monetary_q"]
    )

    # drop helper datetime columns
    agg.drop(columns=["first_purchase", "last_purchase"], inplace=True)

    return agg.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHURN LABEL CREATION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def create_churn_labels(
    purchases_df: pd.DataFrame,
    customer_features: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """
    churn = 1  if a customer made zero qualifying purchases after snapshot.
    churn = 0  if a customer made at least one purchase after snapshot.
    Only customers who exist in customer_features (pre-snapshot history) are labelled.
    """
    post = purchases_df[purchases_df["InvoiceDate"] > snapshot]
    active_after = set(post["CustomerID"].unique())

    df = customer_features.copy()
    df["churn"] = df["CustomerID"].apply(
        lambda cid: 0 if cid in active_after else 1
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODELLING DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary_total", "avg_order_value",
    "total_items", "distinct_products", "distinct_invoices",
    "return_rate", "cancelled_amount", "tenure_days",
    "inter_purchase_gap_mean", "inter_purchase_gap_std",
    "purchase_span_days", "purchases_last_90d", "revenue_last_90d",
    "n_returns", "rfm_score", "is_uk",
]
CATEGORICAL_FEATURES: list[str] = []   # country excluded: too sparse; is_uk used instead


def prepare_modeling_data(labelled_df: pd.DataFrame):
    """
    Returns X_train, X_test, y_train, y_test, feature_names.
    CustomerID is excluded from features.
    """
    available_num = [c for c in NUMERIC_FEATURES if c in labelled_df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in labelled_df.columns]
    feature_cols  = available_num + available_cat

    X = labelled_df[feature_cols].copy()
    y = labelled_df["churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline(clf, numeric_features: list[str]) -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        [("num", num_pipe, numeric_features)],
        remainder="drop",
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


@st.cache_data(show_spinner=False)
def train_models(
    _X_train: pd.DataFrame,
    _X_test:  pd.DataFrame,
    y_train:  np.ndarray,
    y_test:   np.ndarray,
    feature_cols: list[str],
) -> dict:
    """
    Train LR, RF, HistGB, KNN. Select best by ROC-AUC.
    Returns results dict with metrics, fitted pipelines, and the best model.
    """
    numeric_features = [c for c in feature_cols if c in NUMERIC_FEATURES]

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(
            n_neighbors=7, metric="manhattan",
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in classifiers.items():
        pipe = _build_pipeline(clf, numeric_features)
        try:
            # 5-fold CV
            cv_res = cross_validate(
                pipe, _X_train, y_train, cv=cv,
                scoring=["roc_auc", "average_precision"],
                return_train_score=False,
            )
            # Fit on full training set
            pipe.fit(_X_train, y_train)

            # Test-set predictions
            y_pred  = pipe.predict(_X_test)
            y_proba = pipe.predict_proba(_X_test)[:, 1]

            results[name] = {
                "pipeline":       pipe,
                "y_pred":         y_pred,
                "y_proba":        y_proba,
                "cv_roc_auc":     cv_res["test_roc_auc"].mean(),
                "cv_pr_auc":      cv_res["test_average_precision"].mean(),
                "accuracy":       accuracy_score(y_test, y_pred),
                "precision":      precision_score(y_test, y_pred, zero_division=0),
                "recall":         recall_score(y_test, y_pred, zero_division=0),
                "f1":             f1_score(y_test, y_pred, zero_division=0),
                "roc_auc":        roc_auc_score(y_test, y_proba),
                "pr_auc":         average_precision_score(y_test, y_proba),
                "conf_matrix":    confusion_matrix(y_test, y_pred),
                "roc_curve":      roc_curve(y_test, y_proba),
                "pr_curve":       precision_recall_curve(y_test, y_proba),
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}

    # Best model by CV ROC-AUC
    valid = {k: v for k, v in results.items() if "error" not in v}
    best_name = max(valid, key=lambda k: valid[k]["cv_roc_auc"])
    results["_best_name"] = best_name

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. PREDICTIONS & RISK CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_predictions(
    _pipeline,
    labelled_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply fitted pipeline to ALL customers; return predictions table."""
    X_all = labelled_df[feature_cols].copy()
    proba = _pipeline.predict_proba(X_all)[:, 1]
    out   = labelled_df[["CustomerID"]].copy()
    out["churn_probability"] = proba
    out["risk_level"]        = out["churn_probability"].apply(_risk_label)
    out["actual_churn"]      = labelled_df["churn"].values
    # Attach key behavioural features
    for col in ["recency_days", "frequency", "monetary_total",
                "return_rate", "tenure_days", "rfm_score"]:
        if col in labelled_df.columns:
            out[col] = labelled_df[col].values
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. INDIVIDUAL EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

def explain_prediction(
    pipeline,
    customer_row: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, plt.Figure | None]:
    """
    Generate model-derived contributing factors.
    Uses SHAP where compatible; falls back to permutation-style magnitude
    for unsupported models.
    Returns (importance_df, fig).
    """
    X_single = customer_row[feature_cols].copy()
    # Transform through the preprocessor
    prep  = pipeline.named_steps["prep"]
    clf   = pipeline.named_steps["clf"]
    X_t   = prep.transform(X_single)

    try:
        import shap
        if hasattr(clf, "estimators_") or hasattr(clf, "_raw_predict"):
            # Tree-based
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_t)
            if isinstance(shap_values, list):          # RF returns list[2]
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]
        else:
            explainer   = shap.LinearExplainer(clf, X_t)
            shap_values = explainer.shap_values(X_t)
            sv = shap_values[0]

        importance_df = pd.DataFrame({
            "feature":    feature_cols,
            "shap_value": sv,
        }).sort_values("shap_value", key=abs, ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(7, 4))
        colours = ["#c0392b" if v > 0 else "#2980b9"
                   for v in importance_df["shap_value"]]
        ax.barh(importance_df["feature"][::-1],
                importance_df["shap_value"][::-1],
                color=colours[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (positive → higher churn risk)")
        ax.set_title("Model-derived contributing factors (SHAP)")
        fig.tight_layout()
        return importance_df, fig

    except Exception:
        # Fallback: use |coefficient| or feature_importances_
        if hasattr(clf, "coef_"):
            imps = np.abs(clf.coef_[0])
        elif hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
        else:
            return pd.DataFrame(), None

        importance_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": imps,
        }).sort_values("importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(importance_df["feature"][::-1],
                importance_df["importance"][::-1],
                color="#3b82d4")
        ax.set_xlabel("Feature importance (model-derived)")
        ax.set_title("Model-derived contributing factors")
        fig.tight_layout()
        return importance_df, fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. EDA CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _eda_dataset_overview(purchases_df, returns_df, summary):
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Purchase rows (cleaned)", f"{summary['final_purchase_rows']:,}")
    c2.metric("Unique customers", f"{summary['unique_customers']:,}")
    c3.metric("Unique products",
              f"{purchases_df['StockCode'].nunique():,}")
    c4.metric("Countries",
              f"{purchases_df['Country'].nunique():,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Date from",
              summary["date_min"].strftime("%Y-%m-%d") if pd.notna(summary["date_min"]) else "–")
    c6.metric("Date to",
              summary["date_max"].strftime("%Y-%m-%d") if pd.notna(summary["date_max"]) else "–")
    c7.metric("Return invoices", f"{len(returns_df):,}")


def _eda_cleaning_summary(summary):
    st.subheader("Data Cleaning Summary")
    rows = [
        ("Original rows",                  summary["original_rows"]),
        ("Duplicates removed",             summary["duplicates_removed"]),
        ("Missing CustomerID removed",     summary["missing_cid_removed"]),
        ("Invalid dates removed",          summary["bad_dates_removed"]),
        ("Cancellation rows identified",   summary["cancellations_flagged"]),
        ("Non-product / invalid rows",     summary["non_product_or_invalid"]),
        ("Final usable purchase rows",     summary["final_purchase_rows"]),
    ]
    cs = pd.DataFrame(rows, columns=["Step", "Count"])
    st.dataframe(cs, use_container_width=True, hide_index=True)


def _eda_monthly_volume(purchases_df):
    st.subheader("Monthly Transaction Volume")
    tmp = purchases_df.copy()
    tmp["YearMonth"] = tmp["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = (
        tmp.groupby("YearMonth")
           .agg(invoices=("InvoiceNo", "nunique"),
                revenue=("LineRevenue", "sum"))
           .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(monthly["YearMonth"], monthly["invoices"], color="#3b82d4")
    axes[0].set_title("Distinct invoices per month")
    axes[0].set_xlabel("Month"); axes[0].set_ylabel("Invoice count")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(monthly["YearMonth"], monthly["revenue"], color="#22c55e")
    axes[1].set_title("Revenue per month (£)")
    axes[1].set_xlabel("Month"); axes[1].set_ylabel("Revenue (£)")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _show_fig(fig)
    st.caption(
        "November 2011 shows the highest invoice volume (pre-Christmas peak). "
        "December 2011 is a partial month (data ends 9 Dec) and appears lower."
    )


def _eda_top_countries(purchases_df):
    st.subheader("Revenue by Country (Top 10)")
    rev_by_country = (
        purchases_df.groupby("Country")["LineRevenue"]
        .sum().sort_values(ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    rev_by_country[::-1].plot(kind="barh", ax=ax, color="#3b82d4")
    ax.set_xlabel("Total revenue (£)")
    ax.set_title("Top 10 countries by revenue")
    fig.tight_layout()
    _show_fig(fig)


def _eda_rfm_distributions(feat_df):
    st.subheader("RFM Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title, colour in zip(
        axes,
        ["recency_days", "frequency", "monetary_total"],
        ["Recency (days since last purchase)",
         "Frequency (distinct purchase dates)",
         "Monetary total (£)"],
        ["#3b82d4", "#22c55e", "#e67e22"],
    ):
        vals = feat_df[col].dropna()
        ax.hist(vals, bins=40, color=colour, edgecolor="white", linewidth=0.4)
        ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("Customer count")
    fig.tight_layout()
    _show_fig(fig)
    st.caption(
        "Recency and monetary distributions are right-skewed — typical for retail data. "
        "Most customers have low frequency, indicating infrequent purchasing behaviour."
    )


def _eda_churn_balance(feat_df):
    st.subheader("Churn Class Balance")
    counts = feat_df["churn"].value_counts().rename({0: "Retained", 1: "Churned"})
    total  = counts.sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Retained",  f"{counts.get('Retained', 0):,}")
    c2.metric("Churned",   f"{counts.get('Churned',  0):,}")
    c3.metric("Churn rate", f"{100*counts.get('Churned',0)/total:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values,
           color=["#22c55e", "#c0392b"])
    ax.set_title("Churned vs Retained customers")
    ax.set_ylabel("Customer count")
    for i, (idx, val) in enumerate(counts.items()):
        ax.text(i, val + total * 0.005, str(val), ha="center", fontsize=10)
    fig.tight_layout()
    _show_fig(fig)

    if counts.get("Churned", 0) > counts.get("Retained", 0):
        st.info(
            "The dataset is majority-churn. "
            "Models trained on imbalanced data may over-predict churn. "
            "All classifiers use class_weight='balanced' to compensate."
        )


def _eda_boxplots_by_churn(feat_df):
    st.subheader("Feature Distributions by Churn Label")
    plot_cols = ["recency_days", "frequency", "monetary_total",
                 "avg_order_value", "tenure_days", "return_rate"]
    available = [c for c in plot_cols if c in feat_df.columns]
    n = len(available)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes_flat = axes.flatten()
    for i, col in enumerate(available):
        ax = axes_flat[i]
        data = [
            feat_df.loc[feat_df["churn"] == 0, col].dropna(),
            feat_df.loc[feat_df["churn"] == 1, col].dropna(),
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        labels=["Retained", "Churned"])
        for patch, colour in zip(bp["boxes"], ["#22c55e", "#c0392b"]):
            patch.set_facecolor(colour)
        ax.set_title(col); ax.set_ylabel(col)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Boxplots of key features by churn label", y=1.01)
    fig.tight_layout()
    _show_fig(fig)
    st.caption(
        "Churned customers typically show higher recency (longer since last purchase), "
        "lower frequency, and lower monetary total — consistent with expected churn behaviour."
    )


def _eda_correlation_heatmap(feat_df):
    st.subheader("Feature Correlation Heatmap")
    num_cols = [c for c in NUMERIC_FEATURES if c in feat_df.columns]
    corr = feat_df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Pearson correlation — customer-level features")
    fig.tight_layout()
    _show_fig(fig)


def _eda_rfm_scatter(feat_df):
    st.subheader("RFM Scatter: Recency vs Frequency (coloured by churn)")
    sample = feat_df.sample(min(1000, len(feat_df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = sample["churn"].map({0: "#22c55e", 1: "#c0392b"})
    ax.scatter(sample["recency_days"], sample["frequency"],
               c=colours, alpha=0.55, edgecolors="white", linewidth=0.3, s=40)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#22c55e", label="Retained"),
                  Patch(facecolor="#c0392b", label="Churned")]
    ax.legend(handles=legend_els)
    ax.set_xlabel("Recency (days)"); ax.set_ylabel("Frequency")
    ax.set_title("Recency vs Frequency — up to 1,000 customers")
    fig.tight_layout()
    _show_fig(fig)


def perform_eda(purchases_df, returns_df, summary, feat_df):
    """Render the full EDA section."""
    _eda_dataset_overview(purchases_df, returns_df, summary)
    st.divider()
    _eda_cleaning_summary(summary)
    st.divider()
    _eda_monthly_volume(purchases_df)
    st.divider()
    _eda_top_countries(purchases_df)
    st.divider()
    _eda_rfm_distributions(feat_df)
    st.divider()
    _eda_churn_balance(feat_df)
    st.divider()
    _eda_boxplots_by_churn(feat_df)
    st.divider()
    _eda_correlation_heatmap(feat_df)
    st.divider()
    _eda_rfm_scatter(feat_df)


# ─────────────────────────────────────────────────────────────────────────────
# 10. BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_business_insights(
    labelled_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_results: dict,
    best_name: str,
    feature_cols: list[str],
):
    """Render business-insights page from actual calculated results."""
    st.header("Business Insights")

    # ── Historical observations ───────────────────────────────────────────────
    st.subheader("Historical Observations (from labelled dataset)")
    total_cust  = len(labelled_df)
    hist_churn  = labelled_df["churn"].sum()
    hist_rate   = hist_churn / total_cust if total_cust else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total customers (modelling dataset)", f"{total_cust:,}")
    c2.metric("Historically churned",               f"{hist_churn:,}")
    c3.metric("Historical churn rate",              f"{hist_rate:.1%}")

    st.info(
        "**Historical churn rate** is calculated from the post-snapshot observation window "
        f"(2011-10-01 to {labelled_df.get('last_purchase_date', pd.Timestamp('2011-12-09')).strftime('%Y-%m-%d') if False else '2011-12-09'}). "
        "This is not a perpetual churn rate — it represents the proportion of customers "
        "who did not return during the ~70-day window available after the snapshot date."
    )

    st.divider()

    # ── Model predictions ─────────────────────────────────────────────────────
    st.subheader("Model Predictions")
    st.caption(
        "Churn probabilities are model-estimated. They are not guarantees. "
        "Use them to prioritise retention efforts, not as definitive outcomes."
    )

    risk_counts = predictions_df["risk_level"].value_counts()
    high_n   = int(risk_counts.get("HIGH",   0))
    medium_n = int(risk_counts.get("MEDIUM", 0))
    low_n    = int(risk_counts.get("LOW",    0))
    avg_prob = predictions_df["churn_probability"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-risk customers",    f"{high_n:,}")
    c2.metric("Medium-risk customers",  f"{medium_n:,}")
    c3.metric("Low-risk customers",     f"{low_n:,}")
    c4.metric("Avg predicted churn prob", f"{avg_prob:.1%}")

    # Risk distribution pie
    fig, ax = plt.subplots(figsize=(5, 5))
    labels, sizes, colours = [], [], []
    for lbl, clr in [("HIGH","#c0392b"),("MEDIUM","#e67e22"),("LOW","#27ae60")]:
        n = int(risk_counts.get(lbl, 0))
        if n > 0:
            labels.append(f"{lbl} ({n:,})")
            sizes.append(n)
            colours.append(clr)
    ax.pie(sizes, labels=labels, colors=colours,
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Predicted risk distribution")
    _show_fig(fig)

    st.divider()

    # ── Model-derived contributing factors ────────────────────────────────────
    st.subheader("Model-Derived Contributing Factors")
    st.caption(
        "The chart below shows which features the best model assigns the "
        "greatest weight to across all customers. This does **not** imply "
        "causation — it reflects patterns the model detected in the training data."
    )
    best_pipe = model_results[best_name]["pipeline"]
    clf = best_pipe.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0])
    else:
        imps = None

    if imps is not None:
        fi_df = (
            pd.DataFrame({"feature": feature_cols, "importance": imps})
            .sort_values("importance", ascending=False)
            .head(12)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
                color="#3b82d4")
        ax.set_xlabel("Importance score")
        ax.set_title(f"Top feature importances — {best_name}")
        fig.tight_layout()
        _show_fig(fig)
    else:
        st.info("Feature importance not available for the selected model type.")

    st.divider()

    # ── Behavioural observations ──────────────────────────────────────────────
    st.subheader("Customer Behaviour Observations")
    med_rec_churn = labelled_df.loc[labelled_df["churn"]==1, "recency_days"].median()
    med_rec_ret   = labelled_df.loc[labelled_df["churn"]==0, "recency_days"].median()
    med_freq_churn= labelled_df.loc[labelled_df["churn"]==1, "frequency"].median()
    med_freq_ret  = labelled_df.loc[labelled_df["churn"]==0, "frequency"].median()
    med_mon_churn = labelled_df.loc[labelled_df["churn"]==1, "monetary_total"].median()
    med_mon_ret   = labelled_df.loc[labelled_df["churn"]==0, "monetary_total"].median()

    obs = [
        ("Median recency — churned",  f"{med_rec_churn:.0f} days"),
        ("Median recency — retained", f"{med_rec_ret:.0f} days"),
        ("Median frequency — churned",  f"{med_freq_churn:.1f} visits"),
        ("Median frequency — retained", f"{med_freq_ret:.1f} visits"),
        ("Median revenue — churned",  f"£{med_mon_churn:,.0f}"),
        ("Median revenue — retained", f"£{med_mon_ret:,.0f}"),
    ]
    obs_df = pd.DataFrame(obs, columns=["Observation", "Value"])
    st.dataframe(obs_df, use_container_width=True, hide_index=True)
    st.caption(
        "These are observed differences in the dataset — not causal claims. "
        "Churned customers had longer gaps since their last purchase and transacted less frequently, "
        "which is consistent with their churn label by construction."
    )

    st.divider()

    # ── Limitations ───────────────────────────────────────────────────────────
    st.subheader("Limitations & Disclaimers")
    st.warning(
        "**Churn label construction:** The post-snapshot observation window is approximately "
        "70 days (Oct 1 – Dec 9, 2011), not a full 90-day window. The partial December month "
        "may reduce the observed churn rate relative to a full 90-day period.\n\n"
        "**Missing CustomerID:** ~25% of the original transaction rows had no CustomerID and "
        "were excluded. These transactions are unattributed and cannot contribute to customer profiles.\n\n"
        "**Model estimates:** Churn probabilities are statistical estimates based on observed "
        "historical patterns. They do not account for external factors such as competitor activity, "
        "economic changes, or product launches after the observation window.\n\n"
        "**Small post-snapshot class:** If the churn rate is very high (as may result from the "
        "short post-snapshot window), the model may reflect the recency cut-off more than true "
        "long-term churn behaviour."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def page_home():
    st.title("CustomerPulse AI")
    st.markdown("### Automated Customer Retention & Churn Prediction")

    st.markdown(
        """
        CustomerPulse AI is an end-to-end automated analytics platform that helps businesses
        understand customer behaviour, identify customers at risk of churning, and support
        data-driven retention decisions.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### What this application does")
        st.markdown(
            """
            - **Automatically cleans** raw transaction data — removes duplicates, invalid records,
              non-product entries, and unattributable rows
            - **Engineers customer-level features** from transaction history: recency, frequency,
              monetary value, return behaviour, purchase gaps, and more
            - **Creates a churn label** from the available historical data using a defensible
              temporal snapshot approach
            - **Performs full EDA** including distributions, class balance, monthly trends,
              country analysis, and feature correlations
            - **Trains and compares four ML models**: Logistic Regression, Random Forest,
              HistGradientBoosting, and K-Nearest Neighbours
            - **Selects the best model** automatically based on cross-validated ROC-AUC
            - **Generates churn probabilities** for every customer
            - **Classifies customers** into risk tiers: High, Medium, Low
            - **Explains individual predictions** using model-derived contributing factors
            - **Provides business insights** synthesised from actual calculated results
            """
        )
    with col2:
        st.markdown("#### Pipeline")
        steps = [
            ("📤", "Upload",     "Upload your customer transaction CSV"),
            ("🔍", "Validate",   "Automatic schema and data-quality checks"),
            ("🧹", "Clean",      "Remove duplicates, invalid rows, non-products"),
            ("⚙️", "Features",   "Aggregate transactions into customer profiles"),
            ("🏷️", "Label",      "Define churn from post-snapshot purchase history"),
            ("📊", "Analyse",    "Automated exploratory data analysis"),
            ("🤖", "Train",      "Train and compare classification models"),
            ("🎯", "Predict",    "Generate churn probabilities for all customers"),
            ("💡", "Explain",    "Model-derived contributing factors per customer"),
            ("📈", "Insights",   "Synthesised business observations"),
        ]
        for icon, label, desc in steps:
            st.markdown(f"**{icon} {label}** — {desc}")

    st.divider()
    st.markdown("#### How to get started")
    st.info(
        "Navigate to **Upload & Data Quality** in the sidebar and upload your "
        "transaction CSV. The complete pipeline runs automatically."
    )

    st.markdown("#### Important Notice")
    st.warning(
        "Churn probabilities produced by this application are **model-estimated statistical risks** "
        "based on observed historical transaction patterns. They are **not guarantees** of future "
        "customer behaviour. Retention decisions should combine these estimates with additional "
        "business context and human judgement."
    )

    st.divider()
    st.caption(
        "CustomerPulse AI · Built with Streamlit and scikit-learn · "
        "Predictions are statistical estimates, not guarantees."
    )


def page_upload(state: dict):
    st.title("Upload & Data Quality")
    st.markdown(
        "Upload a customer transaction CSV. "
        "The application expects an **Online Retail**-compatible file with the columns: "
        "`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, "
        "`UnitPrice`, `CustomerID`, `Country`."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is None:
        st.info("Awaiting file upload. You can use the provided `data.csv` file.")
        return

    file_bytes = uploaded.read()
    if len(file_bytes) == 0:
        st.error("The uploaded file is empty. Please upload a valid CSV.")
        return

    with st.spinner("Loading data…"):
        raw_df = load_data(file_bytes)

    if raw_df.empty:
        st.error("Could not parse the uploaded file. Please check it is a valid CSV.")
        return

    valid, issues = validate_data(raw_df)
    if not valid:
        for issue in issues:
            st.error(issue)
        return

    st.success(f"File loaded: **{uploaded.name}** — {len(raw_df):,} rows × {len(raw_df.columns)} columns")

    with st.expander("Raw data preview (first 200 rows)", expanded=False):
        st.dataframe(raw_df.head(200), use_container_width=True)

    with st.expander("Column information", expanded=False):
        info_rows = []
        for col in raw_df.columns:
            n_miss = int(raw_df[col].isna().sum())
            info_rows.append({
                "Column":   col,
                "Dtype":    str(raw_df[col].dtype),
                "Non-null": int(raw_df[col].notna().sum()),
                "Missing":  n_miss,
                "Missing%": f"{100*n_miss/len(raw_df):.2f}%",
                "Unique":   int(raw_df[col].nunique()),
            })
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    # ── Run cleaning pipeline ─────────────────────────────────────────────────
    with st.spinner("Cleaning data…"):
        clean_result = clean_data(raw_df)

    summary      = clean_result["summary"]
    purchases_df = clean_result["purchases_df"]
    returns_df   = clean_result["returns_df"]

    st.divider()
    st.subheader("Cleaning Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original rows",         f"{summary['original_rows']:,}")
    c2.metric("Final purchase rows",   f"{summary['final_purchase_rows']:,}")
    c3.metric("Unique customers",      f"{summary['unique_customers']:,}")
    c4.metric("Cancellation rows",     f"{summary['cancellations_flagged']:,}")

    cleaning_table = [
        ("Original rows",                  summary["original_rows"]),
        ("Duplicates removed",             summary["duplicates_removed"]),
        ("Missing CustomerID removed",     summary["missing_cid_removed"]),
        ("Invalid dates removed",          summary["bad_dates_removed"]),
        ("Cancellation rows identified",   summary["cancellations_flagged"]),
        ("Non-product / zero-price rows",  summary["non_product_or_invalid"]),
        ("Final usable purchase rows",     summary["final_purchase_rows"]),
    ]
    st.dataframe(
        pd.DataFrame(cleaning_table, columns=["Step", "Row Count"]),
        use_container_width=True, hide_index=True,
    )

    # ── Feature engineering ───────────────────────────────────────────────────
    with st.spinner("Engineering customer features…"):
        feat_df = engineer_customer_features(purchases_df, returns_df, SNAPSHOT_DATE)

    if feat_df.empty:
        st.error(
            "No customers with pre-snapshot transaction history found. "
            "Check that the dataset covers dates on or before 2011-09-30."
        )
        return

    # ── Churn labels ──────────────────────────────────────────────────────────
    with st.spinner("Creating churn labels…"):
        labelled_df = create_churn_labels(purchases_df, feat_df, SNAPSHOT_DATE)

    st.warning(
        "**Churn label notice:** The churn label is constructed from the available "
        "historical observation period. The dataset ends on 2011-12-09, so the "
        "post-snapshot label window is approximately 70 days — shorter than a "
        "standard 90-day window. This is not a true 90-day churn label."
    )

    # ── Model training ────────────────────────────────────────────────────────
    with st.spinner("Preparing modelling data…"):
        X_train, X_test, y_train, y_test, feature_cols = prepare_modeling_data(labelled_df)

    with st.spinner("Training models (this may take a minute)…"):
        model_results = train_models(X_train, X_test, y_train, y_test, feature_cols)

    best_name = model_results["_best_name"]

    with st.spinner("Generating predictions…"):
        predictions_df = generate_predictions(
            model_results[best_name]["pipeline"],
            labelled_df,
            feature_cols,
        )

    # ── Persist to session state ──────────────────────────────────────────────
    state["raw_df"]        = raw_df
    state["purchases_df"]  = purchases_df
    state["returns_df"]    = returns_df
    state["summary"]       = summary
    state["feat_df"]       = feat_df
    state["labelled_df"]   = labelled_df
    state["model_results"] = model_results
    state["best_name"]     = best_name
    state["feature_cols"]  = feature_cols
    state["predictions_df"]= predictions_df
    state["X_train"]       = X_train
    state["X_test"]        = X_test
    state["y_train"]       = y_train
    state["y_test"]        = y_test
    state["pipeline_ready"]= True

    st.success(
        f"Pipeline complete! Best model: **{best_name}** "
        f"(CV ROC-AUC = {model_results[best_name]['cv_roc_auc']:.3f}). "
        "Navigate to the other sections in the sidebar to explore results."
    )


def page_eda(state: dict):
    st.title("Exploratory Data Analysis")
    if not state.get("pipeline_ready"):
        st.warning("Please upload a dataset on the **Upload & Data Quality** page first.")
        return
    perform_eda(
        state["purchases_df"],
        state["returns_df"],
        state["summary"],
        state["labelled_df"],
    )


def page_model_performance(state: dict):
    st.title("Model Performance")
    if not state.get("pipeline_ready"):
        st.warning("Please upload a dataset on the **Upload & Data Quality** page first.")
        return

    model_results = state["model_results"]
    best_name     = state["best_name"]
    y_test        = state["y_test"]
    feature_cols  = state["feature_cols"]
    labelled_df   = state["labelled_df"]
    X_train       = state["X_train"]
    X_test        = state["X_test"]

    st.markdown(
        f"""
        **Modelling methodology:**
        - Binary classification (churn = 1 / retained = 0)
        - Feature period: transactions on or before **2011-09-30**
        - Label period: transactions **2011-10-01 to 2011-12-09**
        - Train / test split: **{100*(1-TEST_SIZE):.0f}% / {100*TEST_SIZE:.0f}%** (stratified)
        - Number of features: **{len(feature_cols)}**
        - Training customers: **{len(X_train):,}** · Test customers: **{len(X_test):,}**
        - Model selection criterion: **5-fold cross-validated ROC-AUC**
        - Best model: **{best_name}**
        """
    )

    # Class balance notice
    churn_rate = labelled_df["churn"].mean()
    if churn_rate > 0.6 or churn_rate < 0.3:
        st.info(
            f"The dataset has a churn rate of **{churn_rate:.1%}**. "
            "With imbalanced classes, accuracy alone can be misleading — "
            "a model predicting the majority class always would achieve that accuracy without learning. "
            "ROC-AUC and PR-AUC are more informative metrics here."
        )

    st.divider()

    # ── Model comparison table ────────────────────────────────────────────────
    st.subheader("Model Comparison")
    comp_rows = []
    for name, res in model_results.items():
        if name.startswith("_") or "error" in res:
            continue
        comp_rows.append({
            "Model":         name,
            "CV ROC-AUC":    f"{res['cv_roc_auc']:.4f}",
            "CV PR-AUC":     f"{res['cv_pr_auc']:.4f}",
            "Test Accuracy": f"{res['accuracy']:.4f}",
            "Test Precision":f"{res['precision']:.4f}",
            "Test Recall":   f"{res['recall']:.4f}",
            "Test F1":       f"{res['f1']:.4f}",
            "Test ROC-AUC":  f"{res['roc_auc']:.4f}",
            "Test PR-AUC":   f"{res['pr_auc']:.4f}",
            "Best":          "✅" if name == best_name else "",
        })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.caption(
        "CV = 5-fold Stratified cross-validation on training data. "
        "Test metrics are evaluated on the held-out 20% test set. "
        "Best model selected by CV ROC-AUC."
    )

    # ── Per-model detail ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Detailed Results by Model")

    model_names = [n for n in model_results if not n.startswith("_") and "error" not in model_results[n]]
    selected_model = st.selectbox("Select a model to inspect", model_names,
                                   index=model_names.index(best_name) if best_name in model_names else 0)
    res = model_results[selected_model]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{res['accuracy']:.4f}")
    col2.metric("Precision", f"{res['precision']:.4f}")
    col3.metric("Recall",    f"{res['recall']:.4f}")
    col4.metric("F1",        f"{res['f1']:.4f}")
    col5, col6 = st.columns(2)
    col5.metric("ROC-AUC",   f"{res['roc_auc']:.4f}")
    col6.metric("PR-AUC",    f"{res['pr_auc']:.4f}")

    # Metric explanations
    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
            | Metric | Meaning |
            |--------|---------|
            | **Accuracy** | Fraction of all predictions that are correct. Can be misleading with imbalanced classes. |
            | **Precision** | Of customers predicted to churn, what fraction actually churned? |
            | **Recall** | Of customers who actually churned, what fraction did the model identify? |
            | **F1** | Harmonic mean of precision and recall. Balances both. |
            | **ROC-AUC** | Area under the ROC curve. Measures the model's ability to rank churners above non-churners. 1.0 = perfect, 0.5 = random. |
            | **PR-AUC** | Area under the Precision-Recall curve. More informative than ROC-AUC under class imbalance. |
            """
        )

    # Confusion matrix
    cm = res["conf_matrix"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Confusion matrix heatmap
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
    )
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix — {selected_model}")

    # ROC curve
    fpr, tpr, _ = res["roc_curve"]
    axes[1].plot(fpr, tpr, color="#3b82d4", lw=2,
                 label=f"ROC-AUC = {res['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve"); axes[1].legend(loc="lower right")

    # PR curve
    prec, rec, _ = res["pr_curve"]
    axes[2].plot(rec, prec, color="#e67e22", lw=2,
                 label=f"PR-AUC = {res['pr_auc']:.3f}")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve"); axes[2].legend(loc="upper right")

    fig.tight_layout()
    _show_fig(fig)


def page_predictions(state: dict):
    st.title("Churn Predictions")
    if not state.get("pipeline_ready"):
        st.warning("Please upload a dataset on the **Upload & Data Quality** page first.")
        return

    predictions_df = state["predictions_df"]

    st.caption(
        "⚠️ Churn probability is a **model-estimated risk score**, not a guarantee. "
        "Use it to prioritise retention efforts alongside business judgement."
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    risk_counts = predictions_df["risk_level"].value_counts()
    high_n   = int(risk_counts.get("HIGH",   0))
    medium_n = int(risk_counts.get("MEDIUM", 0))
    low_n    = int(risk_counts.get("LOW",    0))
    avg_prob = predictions_df["churn_probability"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total customers",         f"{len(predictions_df):,}")
    c2.metric("🔴 High risk  (≥66%)",     f"{high_n:,}")
    c3.metric("🟡 Medium risk (33–65%)",  f"{medium_n:,}")
    c4.metric("🟢 Low risk  (<33%)",      f"{low_n:,}")
    c5.metric("Avg churn probability",   f"{avg_prob:.1%}")

    st.divider()

    # ── Risk band definitions ─────────────────────────────────────────────────
    with st.expander("Risk band definitions"):
        st.markdown(
            """
            | Risk Level | Churn Probability | Suggested Action |
            |------------|-------------------|------------------|
            | 🔴 **HIGH**   | ≥ 66%  | Immediate personal outreach, priority retention offer |
            | 🟡 **MEDIUM** | 33–65% | Targeted re-engagement email campaign |
            | 🟢 **LOW**    | < 33%  | Standard communications, loyalty reward |
            """
        )

    # ── Filter and table ──────────────────────────────────────────────────────
    risk_filter = st.multiselect(
        "Filter by risk level",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

    display_df = predictions_df[predictions_df["risk_level"].isin(risk_filter)].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map("{:.4f}".format)

    st.dataframe(
        display_df.rename(columns={
            "CustomerID":         "Customer ID",
            "churn_probability":  "Churn Probability",
            "risk_level":         "Risk Level",
            "actual_churn":       "Actual Churn (historical)",
            "recency_days":       "Recency (days)",
            "frequency":          "Frequency",
            "monetary_total":     "Monetary Total (£)",
            "return_rate":        "Return Rate",
            "tenure_days":        "Tenure (days)",
            "rfm_score":          "RFM Score",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── CSV download ──────────────────────────────────────────────────────────
    csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv_bytes,
        file_name="customerpulse_predictions.csv",
        mime="text/csv",
    )


def page_customer_explorer(state: dict):
    st.title("Customer Risk Explorer")
    if not state.get("pipeline_ready"):
        st.warning("Please upload a dataset on the **Upload & Data Quality** page first.")
        return

    predictions_df = state["predictions_df"]
    labelled_df    = state["labelled_df"]
    model_results  = state["model_results"]
    best_name      = state["best_name"]
    feature_cols   = state["feature_cols"]

    # ── Customer selector ─────────────────────────────────────────────────────
    # Sort by churn probability descending so high-risk appear first
    sorted_ids = (
        predictions_df.sort_values("churn_probability", ascending=False)["CustomerID"].tolist()
    )
    selected_id = st.selectbox("Select a Customer ID", sorted_ids)

    pred_row = predictions_df[predictions_df["CustomerID"] == selected_id].iloc[0]
    feat_row = labelled_df[labelled_df["CustomerID"] == selected_id]

    if feat_row.empty:
        st.error("Customer not found in the modelling dataset.")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    prob  = float(pred_row["churn_probability"])
    risk  = pred_row["risk_level"]
    clr   = _risk_colour(risk)

    st.markdown(
        f"### Customer {selected_id} &nbsp;—&nbsp; "
        f'<span style="color:{clr};font-weight:700">{risk} RISK</span>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Probability",  f"{prob:.1%}")
    c2.metric("Risk Level",          risk)
    c3.metric("Actual Churn",        "Churned" if pred_row["actual_churn"] == 1 else "Retained")
    c4.metric("RFM Score",           f"{pred_row.get('rfm_score', '–'):.1f}" if pd.notna(pred_row.get("rfm_score")) else "–")

    st.divider()

    # ── Customer profile ──────────────────────────────────────────────────────
    st.subheader("Customer Profile")
    profile_cols = [
        ("recency_days",          "Recency (days since last purchase)"),
        ("frequency",             "Frequency (distinct purchase dates)"),
        ("monetary_total",        "Monetary Total (£)"),
        ("avg_order_value",       "Avg Order Value (£)"),
        ("total_items",           "Total Items Purchased"),
        ("distinct_products",     "Distinct Products"),
        ("distinct_invoices",     "Distinct Invoices"),
        ("tenure_days",           "Tenure (days)"),
        ("purchase_span_days",    "Purchase Span (days)"),
        ("inter_purchase_gap_mean","Avg Inter-Purchase Gap (days)"),
        ("inter_purchase_gap_std", "Std Inter-Purchase Gap (days)"),
        ("purchases_last_90d",    "Purchases (last 90 days pre-snapshot)"),
        ("revenue_last_90d",      "Revenue (last 90 days pre-snapshot, £)"),
        ("n_returns",             "Number of Return Invoices"),
        ("return_rate",           "Return Rate"),
        ("cancelled_amount",      "Cancelled Amount (£)"),
        ("rfm_score",             "RFM Score"),
        ("is_uk",                 "Is UK Customer"),
    ]
    row_data = feat_row.iloc[0]
    profile_table = []
    for col, label in profile_cols:
        if col in row_data.index and pd.notna(row_data[col]):
            val = row_data[col]
            if isinstance(val, float):
                profile_table.append({"Feature": label, "Value": f"{val:,.2f}"})
            else:
                profile_table.append({"Feature": label, "Value": str(val)})
    st.dataframe(
        pd.DataFrame(profile_table),
        use_container_width=True, hide_index=True,
    )

    # ── Explainability ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model-Derived Contributing Factors")
    st.caption(
        "The factors below reflect which features influenced this customer's "
        "churn probability score according to the selected model. "
        "**These do not imply causation** — they describe patterns the model "
        "detected in the data relative to other customers."
    )

    best_pipe = model_results[best_name]["pipeline"]
    try:
        imp_df, fig = explain_prediction(
            best_pipe, feat_row, feature_cols, best_name
        )
        if fig is not None:
            _show_fig(fig)
        if not imp_df.empty:
            st.dataframe(imp_df.reset_index(drop=True),
                         use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not generate explanation for this customer: {exc}")


def page_business_insights(state: dict):
    if not state.get("pipeline_ready"):
        st.warning("Please upload a dataset on the **Upload & Data Quality** page first.")
        return
    generate_business_insights(
        state["labelled_df"],
        state["predictions_df"],
        state["model_results"],
        state["best_name"],
        state["feature_cols"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="CustomerPulse AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Shared session state ──────────────────────────────────────────────────
    if "pipeline_ready" not in st.session_state:
        st.session_state["pipeline_ready"] = False

    state = st.session_state

    # ── Sidebar navigation ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📊 CustomerPulse AI")
        st.divider()

        page = st.radio(
            "Navigation",
            options=[
                "🏠 Home",
                "📤 Upload & Data Quality",
                "📊 Exploratory Data Analysis",
                "🤖 Model Performance",
                "🎯 Churn Predictions",
                "🔍 Customer Risk Explorer",
                "💼 Business Insights",
            ],
            label_visibility="collapsed",
        )

        if state.get("pipeline_ready"):
            st.divider()
            best = state.get("best_name", "–")
            res  = state["model_results"].get(best, {})
            st.markdown("**Pipeline status**")
            st.success("✅ Pipeline ready")
            st.caption(f"Best model: {best}")
            if "cv_roc_auc" in res:
                st.caption(f"CV ROC-AUC: {res['cv_roc_auc']:.3f}")
            n_cust = len(state.get("predictions_df", []))
            st.caption(f"Customers scored: {n_cust:,}")
        else:
            st.divider()
            st.info("Upload a dataset to begin.")

        st.divider()
        st.caption(
            "Predictions are model-estimated statistical risks, not guarantees."
        )

    # ── Route to selected page ────────────────────────────────────────────────
    if page == "🏠 Home":
        page_home()
    elif page == "📤 Upload & Data Quality":
        page_upload(state)
    elif page == "📊 Exploratory Data Analysis":
        page_eda(state)
    elif page == "🤖 Model Performance":
        page_model_performance(state)
    elif page == "🎯 Churn Predictions":
        page_predictions(state)
    elif page == "🔍 Customer Risk Explorer":
        page_customer_explorer(state)
    elif page == "💼 Business Insights":
        page_business_insights(state)


if __name__ == "__main__":
    main()

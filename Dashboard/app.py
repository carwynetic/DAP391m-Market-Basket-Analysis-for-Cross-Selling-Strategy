import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import ast

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

import time
import statsmodels.formula.api as smf
# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Market Basket Analysis Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS
st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .kpi-title {
        color: #A0A0A0;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
    }
    .insight-box {
        background-color: rgba(244, 164, 96, 0.1);
        border-left: 5px solid #f4a460;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        color: #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING (SAFE LOAD)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")

@st.cache_data
def safe_load_csv(filename, required=False):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        if required:
            st.sidebar.warning(f"Missing file: {filename}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        # Clean column names: remove BOM and extra spaces
        df.columns = (
            df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        return df

    except Exception as e:
        st.sidebar.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()

def clean_frozenset_str(val):
    if pd.isna(val): return str(val)
    val = str(val)
    return val.replace("frozenset({", "").replace("})", "").replace("'", "").replace('"', "")
def extract_stockcodes(value):
    """
    Extract StockCode list from values like:
    frozenset({'22748', '22745'})
    22748, 22745
    ['22748', '22745']
    """
    if pd.isna(value):
        return []

    s = str(value).strip()

    if s.startswith("frozenset("):
        s = s.replace("frozenset(", "", 1)
        if s.endswith(")"):
            s = s[:-1]

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (set, list, tuple, frozenset)):
            return [str(x).strip() for x in parsed]
        return [str(parsed).strip()]
    except Exception:
        pass

    s = s.replace("{", "").replace("}", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace("'", "").replace('"', "")

    return [p.strip() for p in s.split(",") if p.strip()]


def build_product_lookup(df_lookup, df_items):
    """
    Build StockCode -> Description dictionary.
    Priority:
    1. product_lookup.csv
    2. online_retail_ii_basket_items.csv
    """
    if not df_lookup.empty and {"StockCode", "Description"}.issubset(df_lookup.columns):
        temp = df_lookup.copy()
    elif not df_items.empty and {"StockCode", "Description"}.issubset(df_items.columns):
        temp = df_items[["StockCode", "Description"]].copy()
    else:
        return {}

    temp["StockCode"] = temp["StockCode"].astype(str).str.strip()
    temp["Description"] = temp["Description"].astype(str).str.strip()

    temp = temp.dropna(subset=["StockCode", "Description"])
    temp = temp[temp["Description"] != ""]

    product_map = (
        temp
        .groupby("StockCode")["Description"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )

    return product_map


def codes_to_product_names(codes, product_map, keep_code=True):
    """
    Convert StockCodes to readable product names.
    """
    output = []

    for code in codes:
        code = str(code).strip()
        desc = product_map.get(code)

        if desc is None:
            desc = "Unknown Product"

        if keep_code:
            output.append(f"{code} - {desc}")
        else:
            output.append(desc if desc != "Unknown Product" else code)

    return " + ".join(output)


def enrich_rules_with_description(df, product_map):
    """
    Add readable product description columns to association rules dataframe.
    """
    if df.empty:
        return df

    df = df.copy()

    df["antecedent_codes"] = df["antecedents"].apply(extract_stockcodes)
    df["consequent_codes"] = df["consequents"].apply(extract_stockcodes)

    df["antecedents_desc"] = df["antecedent_codes"].apply(
        lambda codes: codes_to_product_names(codes, product_map, keep_code=False)
    )

    df["consequents_desc"] = df["consequent_codes"].apply(
        lambda codes: codes_to_product_names(codes, product_map, keep_code=False)
    )

    df["antecedents_display"] = df["antecedent_codes"].apply(
        lambda codes: codes_to_product_names(codes, product_map, keep_code=True)
    )

    df["consequents_display"] = df["consequent_codes"].apply(
        lambda codes: codes_to_product_names(codes, product_map, keep_code=True)
    )

    df["rule_desc"] = df["antecedents_desc"] + " → " + df["consequents_desc"]
    df["rule_display"] = df["antecedents_display"] + " → " + df["consequents_display"]

    return df
# Load Core Data
# Load Core Data
df_rules = safe_load_csv("association_rules_strong.csv", required=True)
df_top20 = safe_load_csv("top_20_association_rules.csv", required=True)
df_baskets = safe_load_csv("online_retail_ii_basket_df.csv", required=True)

# Large local-only file, not uploaded to GitHub
df_items = safe_load_csv("online_retail_ii_basket_items.csv", required=False)

# Small lookup / summary files
df_lookup = safe_load_csv("product_lookup.csv", required=True)
df_top_products = safe_load_csv("top_product_frequency.csv", required=True)

# Load Optional Data
df_sim = safe_load_csv("add_to_cart_lift_simulation.csv", required=False)
df_model = safe_load_csv("final_causal_impact_summary.csv", required=True)
df_alg_runtime = safe_load_csv("algorithm_runtime_summary.csv", required=False)
df_apr_freq = safe_load_csv("apriori_frequent_itemsets.csv", required=False)
df_fp_freq = safe_load_csv("fpgrowth_frequent_itemsets.csv", required=False)

# Clean frozenset strings in rules if antecedents_str doesn't exist
if not df_rules.empty:
    if 'antecedents_str' not in df_rules.columns:
        df_rules['antecedents_str'] = df_rules['antecedents'].apply(clean_frozenset_str)
        df_rules['consequents_str'] = df_rules['consequents'].apply(clean_frozenset_str)
        df_rules['rule'] = df_rules['antecedents_str'] + " → " + df_rules['consequents_str']

if not df_top20.empty and 'rule' not in df_top20.columns:
    df_top20['antecedents_str'] = df_top20['antecedents'].apply(clean_frozenset_str)
    df_top20['consequents_str'] = df_top20['consequents'].apply(clean_frozenset_str)
    df_top20['rule'] = df_top20['antecedents_str'] + " → " + df_top20['consequents_str']
elif df_top20.empty and not df_rules.empty:
    # Fallback if top_20 file is missing
    df_top20 = df_rules.sort_values('lift', ascending=False).head(20).copy()
# Map StockCode to real product Description for dashboard display
product_map = build_product_lookup(df_lookup, df_items)

df_rules = enrich_rules_with_description(df_rules, product_map)
df_top20 = enrich_rules_with_description(df_top20, product_map)

# ==========================================
# UPLOADED DATASET VALIDATION
# ==========================================

REQUIRED_UPLOAD_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country"
]

def validate_uploaded_mba_dataset(df):
    errors = []
    warnings = []

    # Clean column names
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # Check required columns
    missing_cols = [col for col in REQUIRED_UPLOAD_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        return df, errors, warnings

    # Basic invalid row checks
    if df["InvoiceNo"].isna().sum() > 0:
        errors.append("InvoiceNo contains missing values.")

    if df["StockCode"].isna().sum() > 0:
        errors.append("StockCode contains missing values.")

    if df["Description"].isna().sum() > 0:
        errors.append("Description contains missing values.")

    # Numeric validation
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    if df["Quantity"].isna().sum() > 0:
        errors.append("Quantity contains non-numeric values.")

    if df["UnitPrice"].isna().sum() > 0:
        errors.append("UnitPrice contains non-numeric values.")

    if (df["Quantity"] <= 0).sum() > 0:
        errors.append("Quantity contains values <= 0.")

    if (df["UnitPrice"] <= 0).sum() > 0:
        errors.append("UnitPrice contains values <= 0.")

    # Cancelled invoice check
    cancelled_count = df["InvoiceNo"].astype(str).str.startswith("C").sum()
    if cancelled_count > 0:
        errors.append(f"Dataset contains {cancelled_count} cancelled invoices starting with 'C'.")

    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        warnings.append(f"Dataset contains {duplicate_count} duplicated rows.")

    return df, errors, warnings


def summarize_uploaded_mba_dataset(df):
    total_rows = len(df)
    total_baskets = df["InvoiceNo"].nunique()
    total_products = df["StockCode"].nunique()
    total_countries = df["Country"].nunique()

    basket_sizes = (
        df.groupby("InvoiceNo")["StockCode"]
        .nunique()
        .reset_index(name="BasketSize")
    )

    avg_basket_size = basket_sizes["BasketSize"].mean()
    min_basket_size = basket_sizes["BasketSize"].min()
    max_basket_size = basket_sizes["BasketSize"].max()

    return {
        "total_rows": total_rows,
        "total_baskets": total_baskets,
        "total_products": total_products,
        "total_countries": total_countries,
        "avg_basket_size": avg_basket_size,
        "min_basket_size": min_basket_size,
        "max_basket_size": max_basket_size
    }
# ==========================================
# UPLOADED DATASET BASKET CONSTRUCTION
# ==========================================

def build_uploaded_baskets(df):
    df = df.copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    basket_df = (
        df.groupby("InvoiceNo")["StockCode"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="Items")
    )

    basket_df["BasketSize"] = basket_df["Items"].apply(len)

    total_baskets_before = len(basket_df)

    basket_df_filtered = basket_df[basket_df["BasketSize"] >= 2].copy()

    total_baskets_after = len(basket_df_filtered)
    removed_single_item_baskets = total_baskets_before - total_baskets_after

    return basket_df, basket_df_filtered, {
        "total_baskets_before": total_baskets_before,
        "total_baskets_after": total_baskets_after,
        "removed_single_item_baskets": removed_single_item_baskets
    }


def encode_uploaded_transactions(basket_df_filtered):
    transactions = basket_df_filtered["Items"].tolist()

    te = TransactionEncoder()
    encoded_array = te.fit(transactions).transform(transactions)

    transaction_matrix = pd.DataFrame(
        encoded_array,
        columns=te.columns_,
        index=basket_df_filtered["InvoiceNo"]
    )

    transaction_matrix.index.name = "InvoiceNo"

    return transaction_matrix
# ==========================================
# UPLOADED DATASET FREQUENT ITEMSET MINING
# ==========================================

def run_uploaded_frequent_itemset_mining(transaction_matrix, min_support=0.01, max_len=3):
    results = {}

    # Apriori
    apriori_start = time.perf_counter()
    apriori_itemsets = apriori(
        transaction_matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )
    apriori_runtime = time.perf_counter() - apriori_start

    # FP-Growth
    fpgrowth_start = time.perf_counter()
    fpgrowth_itemsets = fpgrowth(
        transaction_matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )
    fpgrowth_runtime = time.perf_counter() - fpgrowth_start

    # Format itemsets for display
    for df in [apriori_itemsets, fpgrowth_itemsets]:
        if not df.empty:
            df["itemsets_str"] = df["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))
            df["itemset_size"] = df["itemsets"].apply(lambda x: len(x))
            df.sort_values(["support", "itemset_size"], ascending=[False, False], inplace=True)

    runtime_summary = pd.DataFrame([
        {
            "Algorithm": "Apriori",
            "Runtime_Seconds": apriori_runtime,
            "Frequent_Itemsets": len(apriori_itemsets)
        },
        {
            "Algorithm": "FP-Growth",
            "Runtime_Seconds": fpgrowth_runtime,
            "Frequent_Itemsets": len(fpgrowth_itemsets)
        }
    ])

    results["apriori_itemsets"] = apriori_itemsets
    results["fpgrowth_itemsets"] = fpgrowth_itemsets
    results["runtime_summary"] = runtime_summary

    return results
# ==========================================
# UPLOADED DATASET ASSOCIATION RULE GENERATION
# ==========================================

def build_uploaded_product_map(df):
    temp = df[["StockCode", "Description"]].copy()
    temp["StockCode"] = temp["StockCode"].astype(str).str.strip()
    temp["Description"] = temp["Description"].astype(str).str.strip()

    temp = temp.dropna(subset=["StockCode", "Description"])
    temp = temp[temp["Description"] != ""]

    return (
        temp.groupby("StockCode")["Description"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )


def uploaded_itemset_to_text(itemset, uploaded_product_map=None, keep_code=True):
    items = sorted([str(x).strip() for x in list(itemset)])
    output = []

    for code in items:
        desc = None
        if uploaded_product_map is not None:
            desc = uploaded_product_map.get(code)

        if desc:
            output.append(f"{code} - {desc}" if keep_code else desc)
        else:
            output.append(code)

    return " + ".join(output)


def generate_uploaded_association_rules(
    frequent_itemsets,
    uploaded_df,
    min_confidence=0.2,
    strong_min_support=0.01,
    strong_min_confidence=0.4,
    strong_min_lift=2.0
):
    if frequent_itemsets.empty:
        return pd.DataFrame(), pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return rules, rules

    uploaded_product_map = build_uploaded_product_map(uploaded_df)

    rules = rules.copy()

    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: uploaded_itemset_to_text(x, uploaded_product_map, keep_code=True)
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: uploaded_itemset_to_text(x, uploaded_product_map, keep_code=True)
    )

    rules["antecedents_desc"] = rules["antecedents"].apply(
        lambda x: uploaded_itemset_to_text(x, uploaded_product_map, keep_code=False)
    )
    rules["consequents_desc"] = rules["consequents"].apply(
        lambda x: uploaded_itemset_to_text(x, uploaded_product_map, keep_code=False)
    )

    rules["rule_desc"] = rules["antecedents_desc"] + " → " + rules["consequents_desc"]
    rules["rule_display"] = rules["antecedents_str"] + " → " + rules["consequents_str"]

    strong_rules = rules[
        (rules["support"] >= strong_min_support) &
        (rules["confidence"] >= strong_min_confidence) &
        (rules["lift"] >= strong_min_lift)
    ].copy()

    strong_rules = strong_rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False]
    )

    return rules, strong_rules
# ==========================================
# UPLOADED DATASET OUTPUT EXPORT HELPERS
# ==========================================

def prepare_rules_for_download(df):
    if df.empty:
        return df

    export_df = df.copy()

    for col in ["antecedents", "consequents"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: ", ".join(sorted([str(i) for i in list(x)]))
            )

    return export_df


def convert_df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ==========================================
# UPLOADED REGRESSION DATASET VALIDATION
# ==========================================

REQUIRED_REGRESSION_COLUMNS = [
    "InvoiceNo",
    "Items",
    "BasketSize",
    "ProductRevenue",
    "TotalQuantity",
    "AvgUnitPrice",
    "Country"
]

OPTIONAL_REGRESSION_COLUMNS = [
    "CustomerID"
]


def parse_items_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value]

    s = str(value).strip()

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    s = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [x.strip() for x in s.split(",") if x.strip()]


def validate_uploaded_regression_dataset(df):
    errors = []
    warnings = []

    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    missing_cols = [col for col in REQUIRED_REGRESSION_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return df, errors, warnings

    if df["InvoiceNo"].isna().sum() > 0:
        errors.append("InvoiceNo contains missing values.")

    if df["Items"].isna().sum() > 0:
        errors.append("Items contains missing values.")

    if df["Country"].isna().sum() > 0:
        errors.append("Country contains missing values.")

    numeric_cols = [
        "BasketSize",
        "ProductRevenue",
        "TotalQuantity",
        "AvgUnitPrice"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().sum() > 0:
            errors.append(f"{col} contains non-numeric or missing values.")

        if (df[col] <= 0).sum() > 0:
            errors.append(f"{col} contains values <= 0.")

    duplicate_invoice_count = df["InvoiceNo"].duplicated().sum()
    if duplicate_invoice_count > 0:
        warnings.append(
            f"InvoiceNo contains {duplicate_invoice_count} duplicated values. "
            "Regression dataset should normally be one row per basket."
        )

    parsed_items = df["Items"].apply(parse_items_list)
    empty_items_count = parsed_items.apply(len).eq(0).sum()

    if empty_items_count > 0:
        errors.append(f"Items contains {empty_items_count} empty baskets.")

    if "BasketSize" in df.columns:
        parsed_item_counts = parsed_items.apply(len)
        mismatch_count = (parsed_item_counts != df["BasketSize"]).sum()

        if mismatch_count > 0:
            warnings.append(
                f"{mismatch_count} rows have BasketSize different from the number of parsed Items."
            )

    if "CustomerID" not in df.columns:
        warnings.append("CustomerID column is missing. This is acceptable because it is not required for regression.")

    return df, errors, warnings


def summarize_uploaded_regression_dataset(df):
    parsed_items = df["Items"].apply(parse_items_list)

    return {
        "rows": len(df),
        "unique_invoices": df["InvoiceNo"].nunique(),
        "countries": df["Country"].nunique(),
        "avg_basket_size": df["BasketSize"].mean(),
        "avg_revenue": df["ProductRevenue"].mean(),
        "median_revenue": df["ProductRevenue"].median(),
        "avg_quantity": df["TotalQuantity"].mean(),
        "avg_unit_price": df["AvgUnitPrice"].mean(),
        "unique_products_estimated": len(set(item for items in parsed_items for item in items))
    }
# ==========================================
# REGRESSION RULE_APPLIED FEATURE ENGINEERING
# ==========================================

def parse_rule_code_input(value):
    if value is None:
        return []

    s = str(value).strip()

    if not s:
        return []

    s = s.replace("[", "").replace("]", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("'", "").replace('"', "")

    return [x.strip() for x in s.split(",") if x.strip()]


def create_rule_applied_feature(df, antecedent_codes, consequent_codes):
    df = df.copy()

    antecedent_codes = [str(x).strip() for x in antecedent_codes if str(x).strip()]
    consequent_codes = [str(x).strip() for x in consequent_codes if str(x).strip()]

    rule_items = sorted(set(antecedent_codes + consequent_codes))

    if not rule_items:
        return df, {
            "error": "No rule items were provided."
        }

    df["parsed_items"] = df["Items"].apply(parse_items_list)

    df["rule_applied"] = df["parsed_items"].apply(
        lambda items: int(set(rule_items).issubset(set(items)))
    )

    df["antecedent_present"] = df["parsed_items"].apply(
        lambda items: int(set(antecedent_codes).issubset(set(items)))
    )

    df["consequent_present"] = df["parsed_items"].apply(
        lambda items: int(set(consequent_codes).issubset(set(items)))
    )

    applied_count = int(df["rule_applied"].sum())
    not_applied_count = int(len(df) - applied_count)
    applied_rate = applied_count / len(df) if len(df) > 0 else 0

    applied_avg_revenue = (
        df.loc[df["rule_applied"] == 1, "ProductRevenue"].mean()
        if applied_count > 0 else 0
    )

    not_applied_avg_revenue = (
        df.loc[df["rule_applied"] == 0, "ProductRevenue"].mean()
        if not_applied_count > 0 else 0
    )

    stats = {
        "antecedent_codes": antecedent_codes,
        "consequent_codes": consequent_codes,
        "rule_items": rule_items,
        "applied_count": applied_count,
        "not_applied_count": not_applied_count,
        "applied_rate": applied_rate,
        "applied_avg_revenue": applied_avg_revenue,
        "not_applied_avg_revenue": not_applied_avg_revenue
    }

    return df, stats



# ==========================================
# REGRESSION OUTLIER TREATMENT
# ==========================================

def remove_regression_outliers(df, selected_cols=None, upper_quantile=0.995):
    df = df.copy()

    if selected_cols is None:
        selected_cols = [
            "ProductRevenue",
            "BasketSize",
            "TotalQuantity",
            "AvgUnitPrice"
        ]

    available_cols = [col for col in selected_cols if col in df.columns]

    if not available_cols:
        return df, {
            "error": "No valid numeric columns found for outlier treatment."
        }

    rows_before = len(df)
    keep_mask = pd.Series(True, index=df.index)
    threshold_records = []

    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        upper_threshold = df[col].quantile(upper_quantile)
        removed_by_feature = int((df[col] > upper_threshold).sum())

        keep_mask = keep_mask & (df[col] <= upper_threshold)

        threshold_records.append({
            "Feature": col,
            "Upper Quantile": upper_quantile,
            "Upper Threshold": upper_threshold,
            "Rows Above Threshold": removed_by_feature
        })

    cleaned_df = df[keep_mask].copy()

    rows_after = len(cleaned_df)
    removed_rows = rows_before - rows_after
    removed_rate = removed_rows / rows_before if rows_before > 0 else 0

    stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "removed_rows": removed_rows,
        "removed_rate": removed_rate,
        "thresholds": pd.DataFrame(threshold_records)
    }

    return cleaned_df, stats

# ==========================================
# REGRESSION MODEL RUNNER
# ==========================================

def run_uploaded_ols_regression_models(df):
    df = df.copy()

    required_cols = [
        "ProductRevenue",
        "rule_applied",
        "BasketSize",
        "AvgUnitPrice",
        "TotalQuantity",
        "Country"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return pd.DataFrame(), [f"Missing required columns for regression: {missing_cols}"]

    numeric_cols = [
        "ProductRevenue",
        "rule_applied",
        "BasketSize",
        "AvgUnitPrice",
        "TotalQuantity"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Country"] = df["Country"].astype(str).str.strip()

    df = df.dropna(subset=required_cols).copy()

    if df.empty:
        return pd.DataFrame(), ["Regression dataset is empty after dropping missing values."]

    if df["rule_applied"].nunique() < 2:
        return pd.DataFrame(), ["rule_applied must contain both 0 and 1 before running regression."]

    model_specs = [
        {
            "Model": "Model 1A: Baseline",
            "Formula": "ProductRevenue ~ rule_applied"
        },
        {
            "Model": "Model 2A: + BasketSize",
            "Formula": "ProductRevenue ~ rule_applied + BasketSize"
        },
        {
            "Model": "Model 3A: + AvgUnitPrice",
            "Formula": "ProductRevenue ~ rule_applied + BasketSize + AvgUnitPrice"
        },
        {
            "Model": "Model 4A: + TotalQuantity",
            "Formula": "ProductRevenue ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity"
        },
        {
            "Model": "Model 5A: + Country Controls",
            "Formula": "ProductRevenue ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity + C(Country)"
        }
    ]

    results = []
    errors = []

    for spec in model_specs:
        try:
            fitted_model = smf.ols(
                formula=spec["Formula"],
                data=df
            ).fit(cov_type="HC3")

            rule_coef = fitted_model.params.get("rule_applied", np.nan)
            rule_pvalue = fitted_model.pvalues.get("rule_applied", np.nan)
            rule_std_error = fitted_model.bse.get("rule_applied", np.nan)
            rule_tvalue = fitted_model.tvalues.get("rule_applied", np.nan)

            results.append({
                "Model": spec["Model"],
                "Formula": spec["Formula"],
                "N_Observations": int(fitted_model.nobs),
                "Rule_Coefficient": rule_coef,
                "Rule_Std_Error": rule_std_error,
                "Rule_T_Value": rule_tvalue,
                "Rule_P_Value": rule_pvalue,
                "R_Squared": fitted_model.rsquared,
                "Adjusted_R_Squared": fitted_model.rsquared_adj
            })

        except Exception as e:
            errors.append(f"{spec['Model']} failed: {e}")

    return pd.DataFrame(results), errors



# ==========================================
# REGRESSION OUTPUT EXPORT HELPERS
# ==========================================

def prepare_regression_dataset_for_download(df):
    if df.empty:
        return df

    export_df = df.copy()

    if "parsed_items" in export_df.columns:
        export_df["parsed_items"] = export_df["parsed_items"].apply(
            lambda x: ", ".join([str(i) for i in x]) if isinstance(x, list) else str(x)
        )

    return export_df






# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.title("🛒 Parameters")
st.sidebar.markdown("Use the tabs in the main panel to explore different aspects of the Market Basket Analysis.")

country_filter = "All"
if not df_baskets.empty and 'Country' in df_baskets.columns:
    countries = ["All"] + list(df_baskets['Country'].dropna().unique())
    country_filter = st.sidebar.selectbox("Filter Country (Overview Tab)", countries)

# ==========================================
# 4. TABS SETUP
# ==========================================
tabs = st.tabs([
    "📊 1. Executive Overview", 
    "🔍 2. Rules Explorer", 
    "🎁 3. Bundle Recommendation", 
    "⚙️ 4. Algorithm Results", 
    "🧮 5. Add-to-Cart Simulator", 
    "📈 6. Model Results", 
    "💡 7. Final Conclusion",
    "🧪 8. Run MBA on New Dataset",
    "📈 9. Run Regression on New Dataset"
])
# ------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ------------------------------------------
with tabs[0]:
    st.header("Executive Overview")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    total_baskets, total_revenue, avg_basket_size, total_unique_prods = 0, 0, 0, 0
    df_b_filtered = df_baskets.copy()
    if country_filter != "All" and not df_b_filtered.empty:
        df_b_filtered = df_b_filtered[df_b_filtered['Country'] == country_filter]

    if not df_b_filtered.empty:
        total_baskets = len(df_b_filtered)
        total_revenue = df_b_filtered['ProductRevenue'].sum() if 'ProductRevenue' in df_b_filtered.columns else 0
        avg_basket_size = df_b_filtered['BasketSize'].mean() if 'BasketSize' in df_b_filtered.columns else 0
        
    if not df_lookup.empty and "StockCode" in df_lookup.columns:
        total_unique_prods = df_lookup["StockCode"].nunique()

    n_rules = len(df_rules)
    max_lift = df_rules['lift'].max() if not df_rules.empty else 0

    def kpi_card(title, value):
        return f'<div class="glass-card"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div></div>'

    col1.markdown(kpi_card("Total Baskets", f"{total_baskets:,.0f}"), unsafe_allow_html=True)
    col2.markdown(kpi_card("Total Revenue", f"£{total_revenue:,.2f}"), unsafe_allow_html=True)
    col3.markdown(kpi_card("Avg Basket Size", f"{avg_basket_size:.2f} items"), unsafe_allow_html=True)
    col4.markdown(kpi_card("Max Lift (Strong Rules)", f"{max_lift:.2f}"), unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        if not df_b_filtered.empty and 'BasketSize' in df_b_filtered.columns:
            fig1 = px.histogram(df_b_filtered, x='BasketSize', nbins=50, title="Basket Size Distribution", color_discrete_sequence=['#f4a460'])
            fig1.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Basket data not available.")

    with c2:
        if not df_top_products.empty and {"Product", "Frequency"}.issubset(df_top_products.columns):
            top_freq = (
                df_top_products
                .sort_values("Frequency", ascending=False)
                .head(10)
            )
    
            fig2 = px.bar(
                top_freq,
                x="Frequency",
                y="Product",
                orientation="h",
                title="Top 10 Products by Frequency",
                color_discrete_sequence=["#1f77b4"]
            )
    
            fig2.update_layout(
                yaxis={"categoryorder": "total ascending"},
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
    
            st.plotly_chart(fig2, use_container_width=True)
    
        else:
            st.info("Top product frequency data not available.")

# ------------------------------------------
# TAB 2: RULES EXPLORER
# ------------------------------------------
with tabs[1]:
    st.header("Rules Explorer")
    st.markdown("**(RQ1)** Which basket association rules have the highest lift values while still meeting minimum support and confidence thresholds?")
    
    if df_rules.empty:
        st.warning("Association rules dataset not found.")
    else:
        rc1, rc2, rc3 = st.columns(3)
        min_supp = rc1.slider("Min Support", float(df_rules['support'].min()), float(df_rules['support'].max()), float(df_rules['support'].min()))
        min_conf = rc2.slider("Min Confidence", float(df_rules['confidence'].min()), float(df_rules['confidence'].max()), float(df_rules['confidence'].min()))
        min_lift = rc3.slider("Min Lift", float(df_rules['lift'].min()), float(df_rules['lift'].max()), float(df_rules['lift'].min()))
        
        antecedent_filter = st.text_input("Filter by Antecedent Product / StockCode (leave blank for all):")
        
        filtered_rules = df_rules[
            (df_rules['support'] >= min_supp) & 
            (df_rules['confidence'] >= min_conf) & 
            (df_rules['lift'] >= min_lift)
        ]
        
        if antecedent_filter:
            filtered_rules = filtered_rules[
                filtered_rules["antecedents_display"].str.contains(antecedent_filter, case=False, na=False) |
                filtered_rules["antecedents_desc"].str.contains(antecedent_filter, case=False, na=False) |
                filtered_rules["antecedents_str"].str.contains(antecedent_filter, case=False, na=False)
            ]

        if filtered_rules.empty:
            st.warning("No rules match the current filters.")
        else:
            fig_scatter = px.scatter(
                    filtered_rules,
                    x='support',
                    y='confidence',
                    color='lift',
                    hover_data=['rule_desc', 'rule_display', 'support', 'confidence', 'lift'],
                    title="Support vs Confidence (Colored by Lift)",
                    color_continuous_scale='sunsetdark'
                )
            fig_scatter.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            display_cols = [
                "rule_desc",
                "rule_display",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction"
            ]

            available_cols = [c for c in display_cols if c in filtered_rules.columns]

            st.dataframe(
                filtered_rules[available_cols]
                .sort_values('lift', ascending=False)
                .head(50),
                use_container_width=True
            )
            best_rule = filtered_rules.sort_values('lift', ascending=False).iloc[0]
            st.markdown(f"""
            <div class="insight-box">
                <b>💡 Auto-Insight:</b> The strongest rule currently filtered is <b>{best_rule['rule_desc']}</b>.<br>
                This rule has a lift of <b>{best_rule['lift']:.2f}</b>, meaning the consequent is {best_rule['lift']:.2f} times more likely to be bought when the antecedent is in the basket.
                The confidence is <b>{best_rule['confidence']:.1%}</b> (percentage of antecedent buyers who also buy the consequent).
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: BUNDLE RECOMMENDATION
# ------------------------------------------
with tabs[2]:
    st.header("Bundle Recommendation for Product Team")
    st.markdown("Translate strong association rules into actionable business strategies.")
    
    if df_top20.empty:
        st.warning("Top 20 rules dataset not found.")
    else:
        # Just display as nice cards or a stylized table
        for idx, row in df_top20.head(10).iterrows():
            action = "Create Bundle Promotion" if row['lift'] > 10 else "Add to 'Frequently Bought Together'"
            if row['confidence'] > 0.7:
                action = "Recommend consequent at Checkout"

            st.markdown(f"""
            <div class="glass-card" style="padding: 15px;">
                <h4 style="color:#f4a460; margin-bottom:5px;">Bundle Idea {idx+1}: {row['rule_desc']}</h4>
                <b>Antecedent:</b> {row['antecedents_display']} <br>
                <b>Recommended (Consequent):</b> {row['consequents_display']} <br>
                <span style="color:#A0A0A0;">Support: {row['support']:.4f} | Confidence: {row['confidence']:.2%} | Lift: {row['lift']:.2f}</span><br>
                <b style="color:#50C878;">Suggested Action:</b> {action}
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: ALGORITHM RESULTS
# ------------------------------------------
with tabs[3]:
    st.header("Algorithm Results & Comparison")
    
    has_algo_files = not df_alg_runtime.empty or (not df_apr_freq.empty and not df_fp_freq.empty)
    
    if has_algo_files:
        st.success("Optional algorithm outputs found. Displaying Apriori vs FP-Growth comparison.")
        if not df_alg_runtime.empty:
            fig_rt = px.bar(df_alg_runtime, x='Algorithm', y='Runtime_Seconds', title="Runtime Comparison", color='Algorithm')
            fig_rt.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rt, use_container_width=True)
                            # Show speed-up metric
            if {"Algorithm", "Runtime_Seconds"}.issubset(df_alg_runtime.columns):
                apr_time = df_alg_runtime.loc[
                    df_alg_runtime["Algorithm"] == "Apriori", 
                    "Runtime_Seconds"
                ]

                fp_time = df_alg_runtime.loc[
                    df_alg_runtime["Algorithm"] == "FP-Growth", 
                    "Runtime_Seconds"
                ]

                if not apr_time.empty and not fp_time.empty and fp_time.iloc[0] > 0:
                    speedup = apr_time.iloc[0] / fp_time.iloc[0]
                    st.metric(
                        "FP-Growth Speed-up vs Apriori",
                        f"{speedup:.1f}x faster"
                    )
            
        c1, c2 = st.columns(2)
        with c1:
            if not df_apr_freq.empty:
                st.metric("Apriori Frequent Itemsets", len(df_apr_freq))
        with c2:
            if not df_fp_freq.empty:
                st.metric("FP-Growth Frequent Itemsets", len(df_fp_freq))
    else:
        st.info("Separate Apriori/FP-Growth outputs not found. Only final association rules are displayed.")
        if not df_rules.empty:
            st.markdown("### Final Rule Mining Setup")
            st.markdown("- **Algorithm:** Apriori / FP-Growth based association rule mining")
            st.markdown(f"- **Total Strong Rules:** {len(df_rules)}")
            st.markdown(f"- **Max Lift:** {df_rules['lift'].max():.2f}")
            st.markdown(f"- **Average Lift:** {df_rules['lift'].mean():.2f}")

# ------------------------------------------
# TAB 5: AOV / ADD-TO-CART SIMULATOR
# ------------------------------------------
with tabs[4]:
    st.header("Add-to-Cart / Revenue Simulator")
    st.markdown("*\"This simulator is scenario-based estimation, not causal proof.\"*")
    
    if df_rules.empty:
        st.warning("Rules data required for simulation.")
    else:
        s1, s2 = st.columns([1, 2])
        with s1:
            st.markdown("### Parameters")
            rule_options = df_rules.sort_values("lift", ascending=False).head(50).copy()

            selected_rule_idx = st.selectbox(
                "Select Rule to Simulate",
                options=rule_options.index,
                format_func=lambda i: rule_options.loc[i, "rule_desc"]
            )
            target_customers = st.number_input("Target Customers (Antecedent buyers)", min_value=100, max_value=100000, value=1000, step=100)
            conversion_rate = st.slider("Expected Conversion Rate", 0.01, 1.0, 0.05, 0.01)
            expected_aov = st.number_input("Expected AOV of Consequent (£)", min_value=1.0, max_value=500.0, value=15.0)
            discount_rate = st.slider("Discount Rate Applied", 0.0, 0.5, 0.1, 0.05)
            campaign_cost = st.number_input("Campaign Setup Cost (£)", value=100.0)
            
        with s2:
            # Logic
            rule_data = rule_options.loc[selected_rule_idx]
            conf = rule_data['confidence']
            
            est_add_to_cart = target_customers * conf
            est_converted = round(est_add_to_cart * conversion_rate)
            gross_revenue = est_converted * expected_aov
            discount_cost = gross_revenue * discount_rate
            net_revenue = gross_revenue - discount_cost - campaign_cost
            
            st.markdown("### Simulation Results")
            st.markdown(f"""
            <div class="glass-card">
                <p>Rule Confidence: <b>{conf:.2%}</b></p>
                <p>Estimated Add-to-Carts: <b>{est_add_to_cart:.0f}</b></p>
                <p>Estimated Converted Orders: <b>{est_converted:.0f}</b></p>
                <hr>
                <p>Gross Revenue: <b style='color:#50C878;'>£{gross_revenue:,.2f}</b></p>
                <p>Discount Cost: <b style='color:#FF6347;'>-£{discount_cost:,.2f}</b></p>
                <p>Campaign Cost: <b style='color:#FF6347;'>-£{campaign_cost:,.2f}</b></p>
                <h3>Estimated Net Revenue: <span style='color: {"#50C878" if net_revenue > 0 else "#FF6347"};'>£{net_revenue:,.2f}</span></h3>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 6: MODEL RESULTS / REGRESSION IMPACT
# ------------------------------------------
with tabs[5]:
    st.header("Regression-Based Robustness Check")

    if df_model.empty:
        st.warning("Regression summary file not found. Model result tab is disabled.")
    else:
        df_model = df_model.copy()

        # Clean column names: remove whitespace and BOM
        df_model.columns = (
            df_model.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        st.dataframe(df_model, use_container_width=True)

        required_model_col = "Model"

        if required_model_col not in df_model.columns:
            st.error(
                "Column 'Model' not found in final_causal_impact_summary.csv. "
                f"Current columns: {list(df_model.columns)}"
            )
            st.stop()

        pval_col = [c for c in df_model.columns if "p" in c.lower() and "val" in c.lower()]
        coef_col = [c for c in df_model.columns if "coef" in c.lower()]
        r2_col = [c for c in df_model.columns if "r" in c.lower() and "sq" in c.lower()]

        c1, c2 = st.columns(2)

        if coef_col:
            fig_c = px.bar(
                df_model,
                x="Model",
                y=coef_col[0],
                title="Rule Coefficient by Model",
                color_discrete_sequence=["#f4a460"]
            )
            fig_c.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            c1.plotly_chart(fig_c, use_container_width=True)

        if r2_col:
            fig_r = px.bar(
                df_model,
                x="Model",
                y=r2_col[0],
                title="R-Squared by Model",
                color_discrete_sequence=["#1f77b4"]
            )
            fig_r.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            c2.plotly_chart(fig_r, use_container_width=True)

        if not pval_col:
            st.info("No p-value column found. Statistical conclusion is not displayed.")
        else:
            p_col = pval_col[0]

            final_aov_model = df_model[
                df_model["Model"].astype(str).str.contains("Model 5A", case=False, na=False)
            ]

            final_log_model = df_model[
                df_model["Model"].astype(str).str.contains("Model 5B", case=False, na=False)
            ]

            final_aov_p = final_aov_model[p_col].iloc[0] if not final_aov_model.empty else None
            final_log_p = final_log_model[p_col].iloc[0] if not final_log_model.empty else None

            if final_aov_p is None or final_log_p is None:
                st.info("Model 5A or Model 5B was not found. Final controlled-model conclusion is not displayed.")
            elif final_aov_p >= 0.05 and final_log_p >= 0.05:
                st.markdown("""
                <div class='insight-box'>
                    <b>Conclusion:</b> Baseline models show a positive association, but after controlling for 
                    BasketSize, AvgUnitPrice, TotalQuantity, and Country, the selected rule is 
                    <b>not statistically significant</b>. Therefore, there is no strong evidence of causal impact after controls.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='insight-box'>
                    <b>Conclusion:</b> The final controlled model shows statistical significance. 
                    However, since the data is observational, this should be interpreted as association rather than strict causality.
                </div>
                """, unsafe_allow_html=True)
# ------------------------------------------
# TAB 7: FINAL BUSINESS CONCLUSION
# ------------------------------------------
with tabs[6]:
    st.header("Final Business Conclusion")
    
    if df_top20.empty:
        st.warning("Needs top 20 rules data.")
    else:
        st.markdown("### Top Rules Summarized")
        
        top_lift = df_top20.sort_values('lift', ascending=False).head(5)
        top_conf = df_top20.sort_values('confidence', ascending=False).head(5)
        
        c1, c2 = st.columns(2)
        c1.markdown("**Top 5 by Lift**")
        for i, r in top_lift.iterrows():
            c1.markdown(f"- {r['rule_desc']} (Lift: {r['lift']:.2f})")
            
        c2.markdown("**Top 5 by Confidence**")
        for i, r in top_conf.iterrows():
            c2.markdown(f"- {r['rule_desc']} (Conf: {r['confidence']:.1%})")
            
        st.markdown("---")
        st.markdown("### Executive Recommendation")
        
        best = top_lift.iloc[0]
        ant = best['antecedents_desc']
        con = best['consequents_desc']
        
        st.markdown(f"""
        > **Top rule:** {ant} → {con}  
        >
        > Customers who bought **{ant}** are more likely to buy **{con}**.  
        >
        > This rule has support = **{best['support']:.4f}**, confidence = **{best['confidence']:.1%}**, and lift = **{best['lift']:.2f}**.  
        >
        > **Action:** Product team can test this as a bundle offer or checkout recommendation to support cross-selling and improve Average Order Value.
        >
        > **Note:** This recommendation is based on association rule mining. It should be tested through campaign experiments before making a causal claim.
        """)
# ------------------------------------------
# TAB 8: RUN MBA ON NEW DATASET - UPLOAD + VALIDATION ONLY
# ------------------------------------------
with tabs[7]:
    st.header("Run MBA on New Dataset")
    st.markdown(
        "Upload a clean transaction-level CSV dataset. "
        "This stage only validates the uploaded file before running Apriori / FP-Growth."
    )

    uploaded_file = st.file_uploader(
        "Upload generated MBA test dataset",
        type=["csv"],
        help="Required columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country"
    )

    if uploaded_file is None:
        st.info("Upload one generated CSV file to start validation.")
    else:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            uploaded_df.columns = (
                uploaded_df.columns
                .astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
            )

            st.subheader("Uploaded Dataset Preview")
            st.dataframe(uploaded_df.head(20), use_container_width=True)

            validated_df, validation_errors, validation_warnings = validate_uploaded_mba_dataset(uploaded_df)

            st.subheader("Validation Result")

            if validation_errors:
                st.error("Uploaded dataset is not valid for Apriori / FP-Growth.")
                for err in validation_errors:
                    st.markdown(f"- {err}")
            else:
                st.success("Uploaded dataset is valid for Apriori / FP-Growth.")

                summary = summarize_uploaded_mba_dataset(validated_df)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{summary['total_rows']:,}")
                c2.metric("Baskets", f"{summary['total_baskets']:,}")
                c3.metric("Unique Products", f"{summary['total_products']:,}")
                c4.metric("Countries", f"{summary['total_countries']:,}")

                c5, c6, c7 = st.columns(3)
                c5.metric("Avg Basket Size", f"{summary['avg_basket_size']:.2f}")
                c6.metric("Min Basket Size", f"{summary['min_basket_size']:.0f}")
                c7.metric("Max Basket Size", f"{summary['max_basket_size']:.0f}")

                if validation_warnings:
                    st.warning("Dataset is valid but has warnings:")
                    for warn in validation_warnings:
                        st.markdown(f"- {warn}")

                st.subheader("Column Check")
                column_check = pd.DataFrame({
                    "Required Column": REQUIRED_UPLOAD_COLUMNS,
                    "Exists": [col in validated_df.columns for col in REQUIRED_UPLOAD_COLUMNS]
                })
                st.dataframe(column_check, use_container_width=True)

                st.subheader("Dataset Info")
                st.markdown(f"""
                <div class="insight-box">
                    <b>Status:</b> This uploaded file passed schema and basic data validation.<br>
                    <b>Next stage:</b> build baskets from InvoiceNo and StockCode, then encode transactions for Apriori / FP-Growth.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("Basket Construction")

                basket_df, basket_df_filtered, basket_stats = build_uploaded_baskets(validated_df)

                b1, b2, b3 = st.columns(3)
                b1.metric("Baskets Before Filtering", f"{basket_stats['total_baskets_before']:,}")
                b2.metric("Baskets After Filtering", f"{basket_stats['total_baskets_after']:,}")
                b3.metric("Removed Single-Item Baskets", f"{basket_stats['removed_single_item_baskets']:,}")

                st.markdown("### Basket Preview")
                basket_preview = basket_df_filtered.copy()
                basket_preview["Items"] = basket_preview["Items"].apply(lambda x: ", ".join(x))
                st.dataframe(basket_preview.head(20), use_container_width=True)

                if basket_df_filtered.empty:
                    st.error("No valid baskets remain after filtering. Apriori / FP-Growth cannot run.")
                else:
                    st.subheader("Transaction Encoding")

                    transaction_matrix = encode_uploaded_transactions(basket_df_filtered)

                    e1, e2, e3 = st.columns(3)
                    e1.metric("Transaction Matrix Rows", f"{transaction_matrix.shape[0]:,}")
                    e2.metric("Transaction Matrix Columns", f"{transaction_matrix.shape[1]:,}")
                    e3.metric("Unique Products Encoded", f"{transaction_matrix.shape[1]:,}")

                    st.markdown("### Encoded Matrix Preview")
                    encoded_preview = transaction_matrix.iloc[:10, :20].reset_index()
                    st.dataframe(encoded_preview, use_container_width=True)

                    st.markdown(f"""
                    <div class="insight-box">
                        <b>Status:</b> Basket construction and transaction encoding completed successfully.<br>
                        <b>Matrix shape:</b> {transaction_matrix.shape[0]:,} baskets × {transaction_matrix.shape[1]:,} products.<br>
                        <b>Next stage:</b> run Apriori and FP-Growth on this transaction matrix.
                    </div>
                    """, unsafe_allow_html=True)
                    # ==========================================
                    # STAGE 4: APRIORI + FP-GROWTH FREQUENT ITEMSET MINING
                    # ==========================================

                    st.markdown("---")
                    st.subheader("Frequent Itemset Mining")

                    m1, m2 = st.columns(2)

                    with m1:
                        uploaded_min_support = st.slider(
                            "Min Support for Uploaded Dataset",
                            min_value=0.001,
                            max_value=0.100,
                            value=0.010,
                            step=0.001,
                            format="%.3f"
                        )

                    with m2:
                        uploaded_max_len = st.slider(
                            "Max Itemset Length",
                            min_value=1,
                            max_value=5,
                            value=3,
                            step=1
                        )

                    run_mining = st.button("Run Apriori and FP-Growth on Uploaded Dataset")

                    if run_mining:
                        with st.spinner("Running Apriori and FP-Growth..."):
                            mining_results = run_uploaded_frequent_itemset_mining(
                                transaction_matrix=transaction_matrix,
                                min_support=uploaded_min_support,
                                max_len=uploaded_max_len
                            )

                        st.session_state["apriori_itemsets_uploaded"] = mining_results["apriori_itemsets"]
                        st.session_state["fpgrowth_itemsets_uploaded"] = mining_results["fpgrowth_itemsets"]
                        st.session_state["runtime_summary_uploaded"] = mining_results["runtime_summary"]
                        st.session_state["validated_uploaded_df"] = validated_df
                        st.session_state["uploaded_min_support"] = uploaded_min_support

                    if "fpgrowth_itemsets_uploaded" in st.session_state:
                        apriori_itemsets_uploaded = st.session_state["apriori_itemsets_uploaded"]
                        fpgrowth_itemsets_uploaded = st.session_state["fpgrowth_itemsets_uploaded"]
                        runtime_summary_uploaded = st.session_state["runtime_summary_uploaded"]

                        st.success("Apriori and FP-Growth completed successfully.")

                        st.subheader("Algorithm Runtime Summary")
                        st.dataframe(runtime_summary_uploaded, use_container_width=True)

                        r1, r2, r3 = st.columns(3)

                        apriori_runtime = runtime_summary_uploaded.loc[
                            runtime_summary_uploaded["Algorithm"] == "Apriori",
                            "Runtime_Seconds"
                        ].iloc[0]

                        fpgrowth_runtime = runtime_summary_uploaded.loc[
                            runtime_summary_uploaded["Algorithm"] == "FP-Growth",
                            "Runtime_Seconds"
                        ].iloc[0]

                        r1.metric("Apriori Frequent Itemsets", f"{len(apriori_itemsets_uploaded):,}")
                        r2.metric("FP-Growth Frequent Itemsets", f"{len(fpgrowth_itemsets_uploaded):,}")

                        if fpgrowth_runtime > 0:
                            speedup = apriori_runtime / fpgrowth_runtime
                            r3.metric("FP-Growth Speed-up", f"{speedup:.2f}x")
                        else:
                            r3.metric("FP-Growth Speed-up", "N/A")

                        if apriori_itemsets_uploaded.empty and fpgrowth_itemsets_uploaded.empty:
                            st.warning(
                                "No frequent itemsets were found. Try lowering min_support or using a dataset with stronger repeated basket patterns."
                            )
                        else:
                            c_apr, c_fp = st.columns(2)

                            with c_apr:
                                st.markdown("### Apriori Frequent Itemsets")
                                st.dataframe(
                                    apriori_itemsets_uploaded[
                                        ["itemsets_str", "itemset_size", "support"]
                                    ].head(50),
                                    use_container_width=True
                                )

                            with c_fp:
                                st.markdown("### FP-Growth Frequent Itemsets")
                                st.dataframe(
                                    fpgrowth_itemsets_uploaded[
                                        ["itemsets_str", "itemset_size", "support"]
                                    ].head(50),
                                    use_container_width=True
                                )

                            fig_runtime_uploaded = px.bar(
                                runtime_summary_uploaded,
                                x="Algorithm",
                                y="Runtime_Seconds",
                                title="Uploaded Dataset Runtime: Apriori vs FP-Growth",
                                color="Algorithm"
                            )
                            fig_runtime_uploaded.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)"
                            )
                            st.plotly_chart(fig_runtime_uploaded, use_container_width=True)

                            st.markdown(f"""
                            <div class="insight-box">
                                <b>Status:</b> Frequent itemset mining completed.<br>
                                <b>Apriori itemsets:</b> {len(apriori_itemsets_uploaded):,}<br>
                                <b>FP-Growth itemsets:</b> {len(fpgrowth_itemsets_uploaded):,}<br>
                                <b>Next stage:</b> generate association rules from these frequent itemsets.
                            </div>
                            """, unsafe_allow_html=True)

                    # ==========================================
                    # STAGE 5: ASSOCIATION RULE GENERATION
                    # ==========================================

                    if "fpgrowth_itemsets_uploaded" in st.session_state:
                        st.markdown("---")
                        st.subheader("Association Rule Generation")

                        g1, g2, g3 = st.columns(3)

                        with g1:
                            uploaded_rule_min_confidence = st.slider(
                                "Rule Generation Min Confidence",
                                min_value=0.05,
                                max_value=1.00,
                                value=0.20,
                                step=0.05,
                                format="%.2f"
                            )

                        with g2:
                            uploaded_strong_confidence = st.slider(
                                "Strong Rule Min Confidence",
                                min_value=0.05,
                                max_value=1.00,
                                value=0.40,
                                step=0.05,
                                format="%.2f"
                            )

                        with g3:
                            uploaded_strong_lift = st.slider(
                                "Strong Rule Min Lift",
                                min_value=1.00,
                                max_value=10.00,
                                value=2.00,
                                step=0.50,
                                format="%.2f"
                            )

                        generate_rules_button = st.button("Generate Association Rules from Uploaded Dataset")

                        if generate_rules_button:
                            fpgrowth_itemsets_uploaded = st.session_state["fpgrowth_itemsets_uploaded"]
                            validated_df_for_rules = st.session_state["validated_uploaded_df"]
                            uploaded_min_support_for_rules = st.session_state["uploaded_min_support"]

                            if fpgrowth_itemsets_uploaded.empty:
                                st.error("FP-Growth frequent itemsets are empty. Cannot generate association rules.")
                            else:
                                uploaded_rules, uploaded_strong_rules = generate_uploaded_association_rules(
                                    frequent_itemsets=fpgrowth_itemsets_uploaded,
                                    uploaded_df=validated_df_for_rules,
                                    min_confidence=uploaded_rule_min_confidence,
                                    strong_min_support=uploaded_min_support_for_rules,
                                    strong_min_confidence=uploaded_strong_confidence,
                                    strong_min_lift=uploaded_strong_lift
                                )

                                st.session_state["uploaded_rules"] = uploaded_rules
                                st.session_state["uploaded_strong_rules"] = uploaded_strong_rules

                        if "uploaded_rules" in st.session_state:
                            uploaded_rules = st.session_state["uploaded_rules"]
                            uploaded_strong_rules = st.session_state["uploaded_strong_rules"]

                            if uploaded_rules.empty:
                                st.warning("No association rules were generated. Try lowering min confidence or min support.")
                            else:
                                st.success("Association rules generated successfully.")

                                rule_kpi1, rule_kpi2, rule_kpi3, rule_kpi4 = st.columns(4)

                                rule_kpi1.metric("Generated Rules", f"{len(uploaded_rules):,}")
                                rule_kpi2.metric("Strong Rules", f"{len(uploaded_strong_rules):,}")
                                rule_kpi3.metric("Max Lift", f"{uploaded_rules['lift'].max():.2f}")
                                rule_kpi4.metric("Avg Confidence", f"{uploaded_rules['confidence'].mean():.2%}")

                                display_rule_cols = [
                                    "rule_desc",
                                    "rule_display",
                                    "support",
                                    "confidence",
                                    "lift",
                                    "leverage",
                                    "conviction"
                                ]

                                available_rule_cols = [
                                    col for col in display_rule_cols
                                    if col in uploaded_rules.columns
                                ]

                                st.markdown("### Top 20 Association Rules by Lift")

                                if uploaded_strong_rules.empty:
                                    st.warning("No strong rules match the current support, confidence, and lift thresholds.")

                                    fallback_top_rules = (
                                        uploaded_rules
                                        .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
                                        .head(20)
                                    )

                                    st.markdown("### Top 20 Generated Rules Without Strong-Rule Filter")
                                    st.dataframe(
                                        fallback_top_rules[available_rule_cols],
                                        use_container_width=True
                                    )

                                else:
                                    top_20_uploaded_rules = uploaded_strong_rules.head(20)

                                    st.dataframe(
                                        top_20_uploaded_rules[available_rule_cols],
                                        use_container_width=True
                                    )

                                    best_uploaded_rule = top_20_uploaded_rules.iloc[0]

                                    st.markdown(f"""
                                    <div class="insight-box">
                                        <b>Top uploaded-dataset rule:</b> {best_uploaded_rule["rule_desc"]}<br>
                                        <b>Support:</b> {best_uploaded_rule["support"]:.4f}<br>
                                        <b>Confidence:</b> {best_uploaded_rule["confidence"]:.2%}<br>
                                        <b>Lift:</b> {best_uploaded_rule["lift"]:.2f}<br>
                                        <b>Interpretation:</b> This rule is a candidate cross-selling pattern from the uploaded dataset, not causal proof.
                                    </div>
                                    """, unsafe_allow_html=True)

                                st.markdown("### All Generated Rules Preview")

                                all_rules_preview = (
                                    uploaded_rules
                                    .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
                                    .head(50)
                                )

                                st.dataframe(
                                    all_rules_preview[available_rule_cols],
                                    use_container_width=True
                                )

                                st.markdown(f"""
                                <div class="insight-box">
                                    <b>Status:</b> Association rule generation completed.<br>
                                    <b>Total generated rules:</b> {len(uploaded_rules):,}<br>
                                    <b>Total strong rules:</b> {len(uploaded_strong_rules):,}<br>
                                    <b>Next stage:</b> add visual charts and downloadable CSV outputs.
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # ==========================================
                    # ==========================================
                    # STAGE 6: VISUAL CHARTS + DOWNLOAD OUTPUTS
                    # ==========================================

                    if "uploaded_rules" in st.session_state:
                        uploaded_rules = st.session_state["uploaded_rules"]
                        uploaded_strong_rules = st.session_state["uploaded_strong_rules"]

                        st.markdown("---")
                        st.subheader("Uploaded Dataset Rule Visualizations")

                        rules_for_viz = uploaded_strong_rules.copy()

                        if rules_for_viz.empty:
                            rules_for_viz = uploaded_rules.copy()

                        if rules_for_viz.empty:
                            st.warning("No rules available for visualization.")
                        else:
                            rules_for_viz = rules_for_viz.sort_values(
                                ["lift", "confidence", "support"],
                                ascending=[False, False, False]
                            )

                            top_rules_viz = rules_for_viz.head(20).copy()

                            top_rules_viz["rule_short"] = top_rules_viz["rule_desc"].apply(
                                lambda x: x[:90] + "..." if len(str(x)) > 90 else str(x)
                            )

                            chart_col1, chart_col2 = st.columns(2)

                            with chart_col1:
                                fig_top_lift_uploaded = px.bar(
                                    top_rules_viz.sort_values("lift", ascending=True),
                                    x="lift",
                                    y="rule_short",
                                    orientation="h",
                                    title="Top 20 Uploaded Rules by Lift",
                                    hover_data=["rule_desc", "support", "confidence", "lift"]
                                )

                                fig_top_lift_uploaded.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    yaxis_title="Rule",
                                    xaxis_title="Lift",
                                    height=700
                                )

                                st.plotly_chart(fig_top_lift_uploaded, use_container_width=True)

                            with chart_col2:
                                fig_scatter_uploaded = px.scatter(
                                    rules_for_viz,
                                    x="support",
                                    y="confidence",
                                    color="lift",
                                    size="lift",
                                    hover_data=["rule_desc", "support", "confidence", "lift"],
                                    title="Uploaded Rules: Support vs Confidence Colored by Lift",
                                    color_continuous_scale="sunsetdark"
                                )

                                fig_scatter_uploaded.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    xaxis_title="Support",
                                    yaxis_title="Confidence"
                                )

                                st.plotly_chart(fig_scatter_uploaded, use_container_width=True)

                            fig_conf_uploaded = px.histogram(
                                rules_for_viz,
                                x="confidence",
                                nbins=20,
                                title="Uploaded Rules Confidence Distribution"
                            )

                            fig_conf_uploaded.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                xaxis_title="Confidence",
                                yaxis_title="Number of Rules"
                            )

                            st.plotly_chart(fig_conf_uploaded, use_container_width=True)


                        st.markdown("---")
                        st.subheader("Download Uploaded Dataset Outputs")

                        uploaded_rules_export = prepare_rules_for_download(uploaded_rules)
                        uploaded_strong_rules_export = prepare_rules_for_download(uploaded_strong_rules)

                        if uploaded_strong_rules.empty:
                            uploaded_top20_export = (
                                uploaded_rules_export
                                .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
                                .head(20)
                            )
                        else:
                            uploaded_top20_export = (
                                uploaded_strong_rules_export
                                .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
                                .head(20)
                            )

                        download_col1, download_col2, download_col3 = st.columns(3)

                        with download_col1:
                            st.download_button(
                                label="Download All Uploaded Rules CSV",
                                data=convert_df_to_csv_bytes(uploaded_rules_export),
                                file_name="uploaded_association_rules_all.csv",
                                mime="text/csv"
                            )

                        with download_col2:
                            st.download_button(
                                label="Download Strong Uploaded Rules CSV",
                                data=convert_df_to_csv_bytes(uploaded_strong_rules_export),
                                file_name="uploaded_association_rules_strong.csv",
                                mime="text/csv"
                            )

                        with download_col3:
                            st.download_button(
                                label="Download Top 20 Uploaded Rules CSV",
                                data=convert_df_to_csv_bytes(uploaded_top20_export),
                                file_name="uploaded_top_20_association_rules.csv",
                                mime="text/csv"
                            )

                        if "runtime_summary_uploaded" in st.session_state:
                            runtime_summary_uploaded_export = st.session_state["runtime_summary_uploaded"]

                            st.download_button(
                                label="Download Uploaded Algorithm Runtime CSV",
                                data=convert_df_to_csv_bytes(runtime_summary_uploaded_export),
                                file_name="uploaded_algorithm_runtime_summary.csv",
                                mime="text/csv"
                            )

                        st.markdown(f"""
                        <div class="insight-box">
                            <b>Status:</b> Visual charts and downloadable outputs completed.<br>
                            <b>Downloadable files:</b> all rules, strong rules, top 20 rules, and runtime summary.<br>
                            <b>Stage 6 completed.</b>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ==========================================
                        # STAGE 7: BUSINESS RECOMMENDATION CARDS
                        # ==========================================

                        st.markdown("---")
                        st.subheader("Uploaded Dataset Business Recommendations")

                        recommendation_rules = uploaded_strong_rules.copy()

                        if recommendation_rules.empty:
                            recommendation_rules = uploaded_rules.copy()

                        if recommendation_rules.empty:
                            st.warning("No rules available for business recommendation.")
                        else:
                            recommendation_rules = (
                                recommendation_rules
                                .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
                                .head(10)
                                .reset_index(drop=True)
                            )

                            st.markdown(
                                "These recommendation cards translate uploaded-dataset association rules into candidate cross-selling actions."
                            )

                            for rec_no, row in recommendation_rules.iterrows():
                                support = row["support"]
                                confidence = row["confidence"]
                                lift = row["lift"]

                                if confidence >= 0.80 and lift >= 5:
                                    action = "Recommend consequent at checkout"
                                    priority = "High Priority"
                                elif lift >= 5:
                                    action = "Create bundle promotion"
                                    priority = "Medium Priority"
                                elif confidence >= 0.60:
                                    action = "Add to Frequently Bought Together"
                                    priority = "Medium Priority"
                                else:
                                    action = "Monitor as weak recommendation candidate"
                                    priority = "Low Priority"

                                with st.container(border=True):
                                    st.markdown(f"### Recommendation {rec_no + 1}: {priority}")

                                    st.markdown(f"**Rule:** {row['rule_desc']}")
                                    st.markdown(f"**Full Rule Display:** {row['rule_display']}")

                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Support", f"{support:.4f}")
                                    c2.metric("Confidence", f"{confidence:.2%}")
                                    c3.metric("Lift", f"{lift:.2f}")

                                    st.markdown(f"**Suggested Action:** {action}")
                                    st.caption(
                                        "This recommendation is based on association rule mining from the uploaded dataset. "
                                        "It is a cross-selling hypothesis, not causal proof."
                                    )

                            st.markdown(f"""
                            <div class="insight-box">
                                <b>Status:</b> Business recommendation cards completed.<br>
                                <b>Displayed recommendations:</b> {len(recommendation_rules):,}<br>
                                <b>Step 1 completed:</b> uploaded dataset can now run Association Rule Mining and return interpretable outputs.
                            </div>
                            """, unsafe_allow_html=True)
                        

                    else:
                        st.info("Generate association rules first to enable charts, downloads, and business recommendations.")                                                                     
        except Exception as e:
            st.error(f"Could not read uploaded CSV file: {e}")

# ------------------------------------------
# TAB 9: RUN REGRESSION ON NEW DATASET - UPLOAD + VALIDATION ONLY
# ------------------------------------------
with tabs[8]:
    st.header("Run Regression on New Dataset")

    st.markdown(
        "Upload a basket-level regression-ready CSV dataset. "
        "This stage only validates the uploaded file before creating rule_applied and running OLS regression."
    )

    uploaded_regression_file = st.file_uploader(
        "Upload generated regression test dataset",
        type=["csv"],
        key="regression_dataset_uploader",
        help="Required columns: InvoiceNo, Items, BasketSize, ProductRevenue, TotalQuantity, AvgUnitPrice, Country"
    )

    if uploaded_regression_file is None:
        st.info("Upload one generated regression CSV file to start validation.")
    else:
        try:
            uploaded_regression_df = pd.read_csv(uploaded_regression_file)

            uploaded_regression_df.columns = (
                uploaded_regression_df.columns
                .astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
            )

            st.subheader("Uploaded Regression Dataset Preview")
            st.dataframe(uploaded_regression_df.head(20), use_container_width=True)

            validated_regression_df, regression_errors, regression_warnings = validate_uploaded_regression_dataset(
                uploaded_regression_df
            )

            st.subheader("Regression Dataset Validation Result")

            if regression_errors:
                st.error("Uploaded regression dataset is not valid.")
                for err in regression_errors:
                    st.markdown(f"- {err}")
            else:
                st.success("Uploaded regression dataset is valid.")

                regression_summary = summarize_uploaded_regression_dataset(validated_regression_df)

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Rows", f"{regression_summary['rows']:,}")
                r2.metric("Unique Invoices", f"{regression_summary['unique_invoices']:,}")
                r3.metric("Countries", f"{regression_summary['countries']:,}")
                r4.metric("Estimated Products", f"{regression_summary['unique_products_estimated']:,}")

                r5, r6, r7, r8 = st.columns(4)
                r5.metric("Avg Basket Size", f"{regression_summary['avg_basket_size']:.2f}")
                r6.metric("Avg Revenue", f"£{regression_summary['avg_revenue']:.2f}")
                r7.metric("Avg Quantity", f"{regression_summary['avg_quantity']:.2f}")
                r8.metric("Avg Unit Price", f"£{regression_summary['avg_unit_price']:.2f}")

                if regression_warnings:
                    st.warning("Dataset is valid but has warnings:")
                    for warn in regression_warnings:
                        st.markdown(f"- {warn}")

                st.subheader("Column Check")

                regression_column_check = pd.DataFrame({
                    "Required Column": REQUIRED_REGRESSION_COLUMNS,
                    "Exists": [col in validated_regression_df.columns for col in REQUIRED_REGRESSION_COLUMNS]
                })

                st.dataframe(regression_column_check, use_container_width=True)

                st.subheader("Numeric Summary")

                numeric_summary_cols = [
                    "BasketSize",
                    "ProductRevenue",
                    "TotalQuantity",
                    "AvgUnitPrice"
                ]

                numeric_summary = (
                    validated_regression_df[numeric_summary_cols]
                    .describe()
                    .T
                    .reset_index()
                    .rename(columns={"index": "Feature"})
                )

                st.dataframe(numeric_summary, use_container_width=True)

                st.subheader("Country Distribution")

                country_summary = (
                    validated_regression_df["Country"]
                    .value_counts()
                    .reset_index()
                )
                country_summary.columns = ["Country", "Basket_Count"]

                st.dataframe(country_summary.head(20), use_container_width=True)

                fig_country_regression = px.bar(
                    country_summary.head(10),
                    x="Basket_Count",
                    y="Country",
                    orientation="h",
                    title="Top Countries in Uploaded Regression Dataset"
                )

                fig_country_regression.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(fig_country_regression, use_container_width=True)

                st.session_state["validated_regression_df"] = validated_regression_df

                st.markdown("""
                <div class="insight-box">
                    <b>Status:</b> Regression dataset passed schema and basic data validation.<br>
                    <b>Next stage:</b> select a representative association rule and create the rule_applied variable.
                </div>
                """, unsafe_allow_html=True)
                # ==========================================
                # STAGE 4: CREATE rule_applied VARIABLE
                # ==========================================

                st.markdown("---")
                st.subheader("Create rule_applied Variable")

                st.markdown(
                    "Select a representative rule. The dashboard will create `rule_applied = 1` "
                    "when a basket contains all selected antecedent and consequent items."
                )

                rule_source = st.radio(
                    "Rule Selection Method",
                    options=[
                        "Use generated representative rule",
                        "Manual input"
                    ],
                    horizontal=True
                )

                if rule_source == "Use generated representative rule":
                    antecedent_input = "22748, 22745"
                    consequent_input = "22746"

                    st.info(
                        "Using generated representative rule: "
                        "22748 + 22745 → 22746"
                    )
                else:
                    rule_col1, rule_col2 = st.columns(2)

                    with rule_col1:
                        antecedent_input = st.text_input(
                            "Antecedent StockCodes",
                            value="22748, 22745",
                            help="Example: 22748, 22745"
                        )

                    with rule_col2:
                        consequent_input = st.text_input(
                            "Consequent StockCodes",
                            value="22746",
                            help="Example: 22746"
                        )

                create_rule_button = st.button("Create rule_applied Feature")

                if create_rule_button:
                    antecedent_codes = parse_rule_code_input(antecedent_input)
                    consequent_codes = parse_rule_code_input(consequent_input)

                    rule_applied_df, rule_applied_stats = create_rule_applied_feature(
                        validated_regression_df,
                        antecedent_codes,
                        consequent_codes
                    )

                    if "error" in rule_applied_stats:
                        st.error(rule_applied_stats["error"])
                    else:
                        st.session_state["regression_rule_applied_df"] = rule_applied_df
                        st.session_state["regression_rule_applied_stats"] = rule_applied_stats

                if "regression_rule_applied_df" in st.session_state:
                    rule_applied_df = st.session_state["regression_rule_applied_df"]
                    rule_applied_stats = st.session_state["regression_rule_applied_stats"]

                    st.success("rule_applied feature created successfully.")

                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Rule Applied Baskets", f"{rule_applied_stats['applied_count']:,}")
                    a2.metric("Rule Not Applied Baskets", f"{rule_applied_stats['not_applied_count']:,}")
                    a3.metric("Applied Rate", f"{rule_applied_stats['applied_rate']:.2%}")
                    a4.metric("Rule Items", f"{len(rule_applied_stats['rule_items'])}")

                    b1, b2 = st.columns(2)
                    b1.metric("Avg Revenue - Applied", f"£{rule_applied_stats['applied_avg_revenue']:.2f}")
                    b2.metric("Avg Revenue - Not Applied", f"£{rule_applied_stats['not_applied_avg_revenue']:.2f}")

                    st.markdown("### Selected Rule")

                    selected_rule_df = pd.DataFrame({
                        "Part": ["Antecedent", "Consequent", "All Rule Items"],
                        "StockCodes": [
                            ", ".join(rule_applied_stats["antecedent_codes"]),
                            ", ".join(rule_applied_stats["consequent_codes"]),
                            ", ".join(rule_applied_stats["rule_items"])
                        ]
                    })

                    st.dataframe(selected_rule_df, use_container_width=True)

                    st.markdown("### rule_applied Distribution")

                    rule_applied_summary = (
                        rule_applied_df["rule_applied"]
                        .value_counts()
                        .reset_index()
                    )
                    rule_applied_summary.columns = ["rule_applied", "Basket_Count"]

                    st.dataframe(rule_applied_summary, use_container_width=True)

                    fig_rule_applied = px.bar(
                        rule_applied_summary,
                        x="rule_applied",
                        y="Basket_Count",
                        title="rule_applied Distribution"
                    )

                    fig_rule_applied.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="rule_applied",
                        yaxis_title="Basket Count"
                    )

                    st.plotly_chart(fig_rule_applied, use_container_width=True)

                    st.markdown("### Preview Rows with rule_applied")

                    preview_cols = [
                        "InvoiceNo",
                        "Items",
                        "BasketSize",
                        "ProductRevenue",
                        "TotalQuantity",
                        "AvgUnitPrice",
                        "Country",
                        "rule_applied",
                        "antecedent_present",
                        "consequent_present"
                    ]

                    available_preview_cols = [
                        col for col in preview_cols
                        if col in rule_applied_df.columns
                    ]

                    st.dataframe(
                        rule_applied_df[available_preview_cols].head(50),
                        use_container_width=True
                    )

                    if rule_applied_stats["applied_count"] < 10:
                        st.warning(
                            "Very few baskets have rule_applied = 1. Regression may be unstable."
                        )

                    st.markdown("""
                    <div class="insight-box">
                        <b>Status:</b> rule_applied feature created successfully.<br>
                        <b>Next stage:</b> remove outliers before running OLS regression models.
                    </div>
                    """, unsafe_allow_html=True)
                # ==========================================
                # STAGE 5: OUTLIER TREATMENT
                # ==========================================

                st.markdown("---")
                st.subheader("Outlier Treatment for Regression")

                if "regression_rule_applied_df" not in st.session_state:
                    st.info("Create rule_applied first before removing outliers.")
                else:
                    rule_applied_df = st.session_state["regression_rule_applied_df"]

                    st.markdown(
                        "Remove extreme upper-tail observations before running OLS regression. "
                        "This helps reduce distortion from unusually large basket revenue, quantity, or price values."
                    )

                    outlier_cols = st.multiselect(
                        "Select columns for outlier treatment",
                        options=[
                            "ProductRevenue",
                            "BasketSize",
                            "TotalQuantity",
                            "AvgUnitPrice"
                        ],
                        default=[
                            "ProductRevenue",
                            "BasketSize",
                            "TotalQuantity",
                            "AvgUnitPrice"
                        ],
                        key="regression_outlier_columns"
                    )

                    upper_quantile = st.slider(
                        "Upper quantile threshold",
                        min_value=0.950,
                        max_value=0.999,
                        value=0.995,
                        step=0.001,
                        format="%.3f",
                        key="regression_upper_quantile"
                    )

                    apply_outlier_button = st.button(
                        "Apply Outlier Treatment",
                        key="apply_regression_outlier_treatment"
                    )

                    if apply_outlier_button:
                        regression_cleaned_df, outlier_stats = remove_regression_outliers(
                            rule_applied_df,
                            selected_cols=outlier_cols,
                            upper_quantile=upper_quantile
                        )

                        if "error" in outlier_stats:
                            st.error(outlier_stats["error"])
                        else:
                            st.session_state["regression_cleaned_df"] = regression_cleaned_df
                            st.session_state["regression_outlier_stats"] = outlier_stats

                    if "regression_cleaned_df" in st.session_state:
                        regression_cleaned_df = st.session_state["regression_cleaned_df"]
                        outlier_stats = st.session_state["regression_outlier_stats"]

                        st.success("Outlier treatment completed successfully.")

                        o1, o2, o3, o4 = st.columns(4)
                        o1.metric("Rows Before", f"{outlier_stats['rows_before']:,}")
                        o2.metric("Rows After", f"{outlier_stats['rows_after']:,}")
                        o3.metric("Removed Rows", f"{outlier_stats['removed_rows']:,}")
                        o4.metric("Removed Rate", f"{outlier_stats['removed_rate']:.2%}")

                        st.markdown("### Outlier Thresholds")

                        threshold_df = outlier_stats["thresholds"].copy()
                        st.dataframe(threshold_df, use_container_width=True)

                        st.markdown("### Numeric Summary After Outlier Treatment")

                        numeric_cols_after = [
                            "BasketSize",
                            "ProductRevenue",
                            "TotalQuantity",
                            "AvgUnitPrice",
                            "rule_applied"
                        ]

                        available_numeric_cols_after = [
                            col for col in numeric_cols_after
                            if col in regression_cleaned_df.columns
                        ]

                        numeric_summary_after = (
                            regression_cleaned_df[available_numeric_cols_after]
                            .describe()
                            .T
                            .reset_index()
                            .rename(columns={"index": "Feature"})
                        )

                        st.dataframe(numeric_summary_after, use_container_width=True)

                        st.markdown("### rule_applied After Outlier Treatment")

                        cleaned_rule_summary = (
                            regression_cleaned_df["rule_applied"]
                            .value_counts()
                            .reset_index()
                        )
                        cleaned_rule_summary.columns = ["rule_applied", "Basket_Count"]

                        st.dataframe(cleaned_rule_summary, use_container_width=True)

                        fig_cleaned_rule = px.bar(
                            cleaned_rule_summary,
                            x="rule_applied",
                            y="Basket_Count",
                            title="rule_applied Distribution After Outlier Treatment"
                        )

                        fig_cleaned_rule.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="rule_applied",
                            yaxis_title="Basket Count"
                        )

                        st.plotly_chart(fig_cleaned_rule, use_container_width=True)

                        st.markdown("### Cleaned Regression Dataset Preview")

                        preview_clean_cols = [
                            "InvoiceNo",
                            "BasketSize",
                            "ProductRevenue",
                            "TotalQuantity",
                            "AvgUnitPrice",
                            "Country",
                            "rule_applied"
                        ]

                        available_preview_clean_cols = [
                            col for col in preview_clean_cols
                            if col in regression_cleaned_df.columns
                        ]

                        st.dataframe(
                            regression_cleaned_df[available_preview_clean_cols].head(50),
                            use_container_width=True
                        )

                        st.markdown("""
                        <div class="insight-box">
                            <b>Status:</b> Outlier treatment completed successfully.<br>
                            <b>Next stage:</b> run OLS regression models using the cleaned regression dataset.
                        </div>
                        """, unsafe_allow_html=True)
                # ==========================================
                # STAGE 6: RUN OLS REGRESSION MODELS
                # ==========================================

                st.markdown("---")
                st.subheader("OLS Regression Models")

                if "regression_cleaned_df" not in st.session_state:
                    st.info("Apply outlier treatment first before running regression models.")
                else:
                    regression_cleaned_df = st.session_state["regression_cleaned_df"]

                    st.markdown(
                        "Run OLS regression models to test whether `rule_applied` is associated with "
                        "`ProductRevenue` after adding basket-level controls."
                    )

                    run_regression_button = st.button(
                        "Run OLS Regression Models",
                        key="run_uploaded_ols_regression_models"
                    )

                    if run_regression_button:
                        regression_results_df, regression_errors = run_uploaded_ols_regression_models(
                            regression_cleaned_df
                        )

                        st.session_state["uploaded_regression_results_df"] = regression_results_df
                        st.session_state["uploaded_regression_errors"] = regression_errors

                    if "uploaded_regression_results_df" in st.session_state:
                        regression_results_df = st.session_state["uploaded_regression_results_df"]
                        regression_errors = st.session_state["uploaded_regression_errors"]

                        if regression_errors:
                            st.warning("Some regression models returned warnings or errors:")
                            for err in regression_errors:
                                st.markdown(f"- {err}")

                        if regression_results_df.empty:
                            st.error("No regression results were generated.")
                        else:
                            st.success("OLS regression models completed successfully.")

                            st.markdown("### Regression Results Summary")

                            display_regression_df = regression_results_df.copy()

                            numeric_display_cols = [
                                "Rule_Coefficient",
                                "Rule_Std_Error",
                                "Rule_T_Value",
                                "Rule_P_Value",
                                "R_Squared",
                                "Adjusted_R_Squared"
                            ]

                            for col in numeric_display_cols:
                                if col in display_regression_df.columns:
                                    display_regression_df[col] = display_regression_df[col].round(4)

                            st.dataframe(display_regression_df, use_container_width=True)

                            final_model_row = regression_results_df[
                                regression_results_df["Model"].str.contains("Model 5A", case=False, na=False)
                            ]

                            if not final_model_row.empty:
                                final_model = final_model_row.iloc[0]

                                k1, k2, k3, k4 = st.columns(4)
                                k1.metric("Final Model Rule Coef", f"{final_model['Rule_Coefficient']:.4f}")
                                k2.metric("Final Model p-value", f"{final_model['Rule_P_Value']:.4f}")
                                k3.metric("Final Model R²", f"{final_model['R_Squared']:.4f}")
                                k4.metric("Observations", f"{int(final_model['N_Observations']):,}")

                            chart_col1, chart_col2 = st.columns(2)

                            with chart_col1:
                                fig_coef = px.bar(
                                    regression_results_df,
                                    x="Model",
                                    y="Rule_Coefficient",
                                    title="rule_applied Coefficient by Regression Model"
                                )

                                fig_coef.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    xaxis_title="Model",
                                    yaxis_title="Coefficient"
                                )

                                st.plotly_chart(fig_coef, use_container_width=True)

                            with chart_col2:
                                fig_pvalue = px.bar(
                                    regression_results_df,
                                    x="Model",
                                    y="Rule_P_Value",
                                    title="rule_applied p-value by Regression Model"
                                )

                                fig_pvalue.add_hline(
                                    y=0.05,
                                    line_dash="dash",
                                    annotation_text="0.05 threshold"
                                )

                                fig_pvalue.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    xaxis_title="Model",
                                    yaxis_title="p-value"
                                )

                                st.plotly_chart(fig_pvalue, use_container_width=True)

                            fig_r2 = px.bar(
                                regression_results_df,
                                x="Model",
                                y="R_Squared",
                                title="R-Squared by Regression Model"
                            )

                            fig_r2.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                xaxis_title="Model",
                                yaxis_title="R-Squared"
                            )

                            st.plotly_chart(fig_r2, use_container_width=True)

                            st.markdown("### Regression Interpretation")

                            if not final_model_row.empty:
                                final_coef = final_model["Rule_Coefficient"]
                                final_pvalue = final_model["Rule_P_Value"]
                                final_r2 = final_model["R_Squared"]

                                if final_pvalue < 0.05:
                                    final_conclusion = (
                                        "In the final controlled model, rule_applied remains statistically significant. "
                                        "This suggests that baskets containing the selected rule are still associated with different ProductRevenue "
                                        "after controlling for BasketSize, AvgUnitPrice, TotalQuantity, and Country."
                                    )
                                else:
                                    final_conclusion = (
                                        "In the final controlled model, rule_applied is not statistically significant. "
                                        "This suggests that the raw revenue difference is likely explained by basket-level controls."
                                    )

                                st.markdown(f"""
                                <div class="insight-box">
                                    <b>Final Controlled Model:</b> Model 5A<br>
                                    <b>rule_applied coefficient:</b> {final_coef:.4f}<br>
                                    <b>p-value:</b> {final_pvalue:.4f}<br>
                                    <b>R-squared:</b> {final_r2:.4f}<br><br>
                                    <b>Conclusion:</b> {final_conclusion}<br><br>
                                    <b>Important note:</b> This is an observational robustness check, not causal proof.
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("""
                            <div class="insight-box">
                                <b>Status:</b> OLS regression models completed successfully.<br>
                                <b>Next stage:</b> add regression output downloads and final interpretation card.
                            </div>
                            """, unsafe_allow_html=True)
                # ==========================================
                # STAGE 7: DOWNLOAD REGRESSION OUTPUTS + FINAL CARD
                # ==========================================

                st.markdown("---")
                st.subheader("Download Regression Outputs")

                if "uploaded_regression_results_df" not in st.session_state:
                    st.info("Run OLS regression models first before downloading regression outputs.")
                else:
                    regression_results_export = st.session_state["uploaded_regression_results_df"].copy()
                    regression_cleaned_export = prepare_regression_dataset_for_download(
                        st.session_state["regression_cleaned_df"]
                    )
                    regression_rule_applied_export = prepare_regression_dataset_for_download(
                        st.session_state["regression_rule_applied_df"]
                    )

                    d1, d2, d3 = st.columns(3)

                    with d1:
                        st.download_button(
                            label="Download Regression Results CSV",
                            data=convert_df_to_csv_bytes(regression_results_export),
                            file_name="uploaded_regression_results.csv",
                            mime="text/csv"
                        )

                    with d2:
                        st.download_button(
                            label="Download Cleaned Regression Dataset CSV",
                            data=convert_df_to_csv_bytes(regression_cleaned_export),
                            file_name="uploaded_regression_cleaned_dataset.csv",
                            mime="text/csv"
                        )

                    with d3:
                        st.download_button(
                            label="Download Rule Applied Dataset CSV",
                            data=convert_df_to_csv_bytes(regression_rule_applied_export),
                            file_name="uploaded_rule_applied_dataset.csv",
                            mime="text/csv"
                        )

                    st.markdown("---")
                    st.subheader("Final Regression Interpretation Card")

                    rule_applied_stats = st.session_state["regression_rule_applied_stats"]

                    baseline_model = regression_results_export[
                        regression_results_export["Model"].str.contains("Model 1A", case=False, na=False)
                    ]

                    final_model = regression_results_export[
                        regression_results_export["Model"].str.contains("Model 5A", case=False, na=False)
                    ]

                    if baseline_model.empty or final_model.empty:
                        st.warning("Baseline model or final controlled model is missing.")
                    else:
                        baseline_row = baseline_model.iloc[0]
                        final_row = final_model.iloc[0]

                        baseline_coef = baseline_row["Rule_Coefficient"]
                        baseline_pvalue = baseline_row["Rule_P_Value"]

                        final_coef = final_row["Rule_Coefficient"]
                        final_pvalue = final_row["Rule_P_Value"]
                        final_r2 = final_row["R_Squared"]
                        final_n = int(final_row["N_Observations"])

                        if baseline_pvalue < 0.05 and final_pvalue >= 0.05:
                            final_interpretation = (
                                "The selected rule shows a statistically significant raw association in the baseline model, "
                                "but the effect disappears after adding basket-level controls. "
                                "This means the observed revenue difference is likely explained by BasketSize, AvgUnitPrice, "
                                "TotalQuantity, and Country rather than the rule itself."
                            )
                        elif final_pvalue < 0.05:
                            final_interpretation = (
                                "The selected rule remains statistically significant after controls. "
                                "This suggests the rule is still associated with ProductRevenue after accounting for basket-level factors."
                            )
                        else:
                            final_interpretation = (
                                "The selected rule is not statistically significant in the final controlled model. "
                                "There is not enough evidence that rule_applied has an independent association with ProductRevenue after controls."
                            )

                        
                        with st.container(border=True):
                            st.markdown("### Final Regression Result")

                            st.markdown(
                                f"**Selected Rule:** {', '.join(rule_applied_stats['antecedent_codes'])} "
                                f"→ {', '.join(rule_applied_stats['consequent_codes'])}"
                            )

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Rule Applied Baskets", f"{rule_applied_stats['applied_count']:,}")
                            c2.metric("Rule Not Applied Baskets", f"{rule_applied_stats['not_applied_count']:,}")
                            c3.metric("Applied Rate", f"{rule_applied_stats['applied_rate']:.2%}")

                            st.markdown("#### Baseline Model")
                            b1, b2 = st.columns(2)
                            b1.metric("Baseline Coefficient", f"{baseline_coef:.4f}")
                            b2.metric("Baseline p-value", f"{baseline_pvalue:.4f}")

                            st.markdown("#### Final Controlled Model")
                            f1, f2, f3, f4 = st.columns(4)
                            f1.metric("Final Coefficient", f"{final_coef:.4f}")
                            f2.metric("Final p-value", f"{final_pvalue:.4f}")
                            f3.metric("Final R²", f"{final_r2:.4f}")
                            f4.metric("Observations", f"{final_n:,}")

                            st.markdown("#### Interpretation")
                            st.markdown(final_interpretation)

                            st.caption(
                                "Important note: This regression is an observational robustness check, not causal proof."
                            )

                        st.markdown("""
                        <div class="insight-box">
                            <b>Status:</b> Regression output downloads and final interpretation card completed.<br>
                            <b>Step 2 completed:</b> uploaded regression-ready dataset can now run rule_applied creation, outlier treatment, OLS models, interpretation, and downloadable outputs.
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not read uploaded regression CSV file: {e}")

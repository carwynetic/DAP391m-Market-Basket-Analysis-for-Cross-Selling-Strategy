import streamlit as st
import pandas as pd
import numpy as np
try:
    from streamlit_float import float_init, float_parent
    FLOAT_CHATBOX_AVAILABLE = True
except Exception:
    FLOAT_CHATBOX_AVAILABLE = False
    
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
# COUNTRY-SPECIFIC DATA SOURCE HELPERS
# ==========================================

def parse_country_basket_items(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    s = str(value).strip()

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    s = s.replace("[", "").replace("]", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("'", "").replace('"', "")

    return [x.strip() for x in s.split(",") if x.strip()]


def filter_by_selected_country(df, selected_country):
    if df.empty:
        return df.copy()

    if selected_country == "All":
        return df.copy()

    if "Country" not in df.columns:
        return df.copy()

    return df[df["Country"].astype(str).str.strip() == selected_country].copy()


def build_country_basket_source(df_baskets, selected_country):
    df_country = filter_by_selected_country(df_baskets, selected_country)

    if df_country.empty:
        return df_country, {
            "selected_country": selected_country,
            "basket_rows": 0,
            "usable_baskets": 0,
            "unique_products": 0,
            "avg_basket_size": 0,
            "total_revenue": 0
        }

    df_country = df_country.copy()

    if "Items" in df_country.columns:
        df_country["ItemsParsed"] = df_country["Items"].apply(parse_country_basket_items)
        df_country["ParsedBasketSize"] = df_country["ItemsParsed"].apply(len)
    else:
        df_country["ItemsParsed"] = [[] for _ in range(len(df_country))]
        df_country["ParsedBasketSize"] = 0

    df_country_usable = df_country[df_country["ParsedBasketSize"] >= 2].copy()

    all_items = set()
    for items in df_country_usable["ItemsParsed"]:
        all_items.update(items)

    summary = {
        "selected_country": selected_country,
        "basket_rows": len(df_country),
        "usable_baskets": len(df_country_usable),
        "unique_products": len(all_items),
        "avg_basket_size": df_country["BasketSize"].mean() if "BasketSize" in df_country.columns else 0,
        "total_revenue": df_country["ProductRevenue"].sum() if "ProductRevenue" in df_country.columns else 0
    }

    return df_country_usable, summary
# ==========================================
# COUNTRY-SPECIFIC MBA PIPELINE
# ==========================================

COUNTRY_ARM_MIN_BASKETS = 100
COUNTRY_ARM_MIN_SUPPORT = 0.01
COUNTRY_ARM_RULE_MIN_CONFIDENCE = 0.10
COUNTRY_ARM_STRONG_MIN_CONFIDENCE = 0.40
COUNTRY_ARM_STRONG_MIN_LIFT = 3.00
COUNTRY_ARM_MAX_LEN = 3
COUNTRY_APRIORI_MAX_BASKETS = 8000


def itemset_to_text(itemset):
    if pd.isna(itemset):
        return ""

    try:
        return " + ".join(sorted([str(x) for x in list(itemset)]))
    except Exception:
        return str(itemset)


def add_rule_display_columns(rules_df):
    if rules_df.empty:
        return rules_df

    rules_df = rules_df.copy()

    rules_df["antecedents_str"] = rules_df["antecedents"].apply(itemset_to_text)
    rules_df["consequents_str"] = rules_df["consequents"].apply(itemset_to_text)

    rules_df["rule_desc"] = (
        rules_df["antecedents_str"]
        + " → "
        + rules_df["consequents_str"]
    )

    rules_df["rule_display"] = rules_df["rule_desc"]

    return rules_df


def prepare_country_transactions(df_country):
    if df_country.empty:
        return []

    if "ItemsParsed" not in df_country.columns:
        df_country = df_country.copy()
        df_country["ItemsParsed"] = df_country["Items"].apply(parse_country_basket_items)

    transactions = []

    for items in df_country["ItemsParsed"]:
        clean_items = sorted(set([str(x).strip() for x in items if str(x).strip()]))

        if len(clean_items) >= 2:
            transactions.append(clean_items)

    return transactions


@st.cache_data(show_spinner=False)
def run_country_mba_pipeline_cached(
    selected_country,
    df_country,
    min_support=COUNTRY_ARM_MIN_SUPPORT,
    rule_min_confidence=COUNTRY_ARM_RULE_MIN_CONFIDENCE,
    strong_min_confidence=COUNTRY_ARM_STRONG_MIN_CONFIDENCE,
    strong_min_lift=COUNTRY_ARM_STRONG_MIN_LIFT,
    max_len=COUNTRY_ARM_MAX_LEN
):
    transactions = prepare_country_transactions(df_country)

    output = {
        "selected_country": selected_country,
        "status": "not_started",
        "message": "",
        "transactions": len(transactions),
        "transaction_matrix_shape": (0, 0),
        "apriori_itemsets": pd.DataFrame(),
        "fpgrowth_itemsets": pd.DataFrame(),
        "rules": pd.DataFrame(),
        "strong_rules": pd.DataFrame(),
        "top_20_rules": pd.DataFrame(),
        "runtime_summary": pd.DataFrame()
    }

    if len(transactions) < COUNTRY_ARM_MIN_BASKETS:
        output["status"] = "not_enough_baskets"
        output["message"] = (
            f"Only {len(transactions):,} usable baskets found. "
            f"Minimum required for country-specific mining is {COUNTRY_ARM_MIN_BASKETS:,}."
        )
        return output

    te = TransactionEncoder()
    encoded_array = te.fit(transactions).transform(transactions)

    transaction_matrix = pd.DataFrame(
        encoded_array,
        columns=te.columns_
    )

    output["transaction_matrix_shape"] = transaction_matrix.shape

    runtime_records = []

    apriori_itemsets = pd.DataFrame()

    if len(transactions) <= COUNTRY_APRIORI_MAX_BASKETS:
        start_time = time.time()

        apriori_itemsets = apriori(
            transaction_matrix,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len
        )

        apriori_runtime = time.time() - start_time

        if not apriori_itemsets.empty:
            apriori_itemsets["itemset_size"] = apriori_itemsets["itemsets"].apply(len)
            apriori_itemsets["itemsets_str"] = apriori_itemsets["itemsets"].apply(itemset_to_text)

        runtime_records.append({
            "Algorithm": "Apriori",
            "Runtime_Seconds": round(apriori_runtime, 4),
            "Frequent_Itemsets": len(apriori_itemsets),
            "Status": "Completed"
        })
    else:
        runtime_records.append({
            "Algorithm": "Apriori",
            "Runtime_Seconds": np.nan,
            "Frequent_Itemsets": 0,
            "Status": f"Skipped: baskets > {COUNTRY_APRIORI_MAX_BASKETS:,}"
        })

    start_time = time.time()

    fpgrowth_itemsets = fpgrowth(
        transaction_matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    fpgrowth_runtime = time.time() - start_time

    if not fpgrowth_itemsets.empty:
        fpgrowth_itemsets["itemset_size"] = fpgrowth_itemsets["itemsets"].apply(len)
        fpgrowth_itemsets["itemsets_str"] = fpgrowth_itemsets["itemsets"].apply(itemset_to_text)

    runtime_records.append({
        "Algorithm": "FP-Growth",
        "Runtime_Seconds": round(fpgrowth_runtime, 4),
        "Frequent_Itemsets": len(fpgrowth_itemsets),
        "Status": "Completed"
    })

    runtime_summary = pd.DataFrame(runtime_records)

    if fpgrowth_itemsets.empty:
        output["status"] = "no_frequent_itemsets"
        output["message"] = "No frequent itemsets found for this country and support threshold."
        output["apriori_itemsets"] = apriori_itemsets
        output["fpgrowth_itemsets"] = fpgrowth_itemsets
        output["runtime_summary"] = runtime_summary
        return output

    try:
        rules = association_rules(
            fpgrowth_itemsets,
            metric="confidence",
            min_threshold=rule_min_confidence
        )
    except Exception as e:
        output["status"] = "rule_generation_failed"
        output["message"] = f"Association rule generation failed: {e}"
        output["apriori_itemsets"] = apriori_itemsets
        output["fpgrowth_itemsets"] = fpgrowth_itemsets
        output["runtime_summary"] = runtime_summary
        return output

    if rules.empty:
        output["status"] = "no_rules"
        output["message"] = "Frequent itemsets were found, but no association rules met the confidence threshold."
        output["apriori_itemsets"] = apriori_itemsets
        output["fpgrowth_itemsets"] = fpgrowth_itemsets
        output["runtime_summary"] = runtime_summary
        return output

    rules = add_rule_display_columns(rules)

    rules = rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    strong_rules = rules[
        (rules["confidence"] >= strong_min_confidence)
        & (rules["lift"] >= strong_min_lift)
    ].copy()

    strong_rules = strong_rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    top_20_rules = strong_rules.head(20).copy()

    output["status"] = "completed"
    output["message"] = "Country-specific MBA pipeline completed successfully."
    output["apriori_itemsets"] = apriori_itemsets
    output["fpgrowth_itemsets"] = fpgrowth_itemsets
    output["rules"] = rules
    output["strong_rules"] = strong_rules
    output["top_20_rules"] = top_20_rules
    output["runtime_summary"] = runtime_summary

    return output

# ==========================================
# COUNTRY-SPECIFIC REGRESSION MODEL HELPERS
# ==========================================

COUNTRY_REGRESSION_MIN_ROWS = 100
COUNTRY_REGRESSION_MIN_APPLIED = 5


def normalize_rule_itemset(value):
    if isinstance(value, (set, frozenset, list, tuple)):
        return set([str(x).strip() for x in value if str(x).strip()])

    if pd.isna(value):
        return set()

    s = str(value).strip()

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (set, frozenset, list, tuple)):
            return set([str(x).strip() for x in parsed if str(x).strip()])
    except Exception:
        pass

    s = (
        s.replace("frozenset", "")
        .replace("set", "")
        .replace("(", "")
        .replace(")", "")
        .replace("{", "")
        .replace("}", "")
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    return set([x.strip() for x in s.split(",") if x.strip()])


def get_rule_items_from_row(rule_row):
    antecedents = normalize_rule_itemset(rule_row.get("antecedents", set()))
    consequents = normalize_rule_itemset(rule_row.get("consequents", set()))
    all_items = antecedents.union(consequents)

    rule_desc = rule_row.get("rule_desc", "")
    if not rule_desc:
        rule_desc = itemset_to_text(antecedents) + " → " + itemset_to_text(consequents)

    return antecedents, consequents, all_items, rule_desc


def create_country_rule_applied_dataset(df_country, rule_row):
    df_model = df_country.copy()

    if "ItemsParsed" not in df_model.columns:
        df_model["ItemsParsed"] = df_model["Items"].apply(parse_country_basket_items)

    antecedents, consequents, all_rule_items, rule_desc = get_rule_items_from_row(rule_row)

    df_model["rule_applied"] = df_model["ItemsParsed"].apply(
        lambda items: int(all_rule_items.issubset(set([str(x).strip() for x in items])))
    )

    if "AvgUnitPrice" not in df_model.columns:
        if "ProductRevenue" in df_model.columns and "TotalQuantity" in df_model.columns:
            df_model["AvgUnitPrice"] = np.where(
                df_model["TotalQuantity"] > 0,
                df_model["ProductRevenue"] / df_model["TotalQuantity"],
                np.nan
            )
        else:
            df_model["AvgUnitPrice"] = np.nan

    needed_cols = [
        "InvoiceNo",
        "BasketSize",
        "ProductRevenue",
        "TotalQuantity",
        "AvgUnitPrice",
        "Country",
        "rule_applied"
    ]

    existing_cols = [col for col in needed_cols if col in df_model.columns]
    df_model = df_model[existing_cols].copy()

    numeric_cols = ["BasketSize", "ProductRevenue", "TotalQuantity", "AvgUnitPrice"]

    for col in numeric_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    df_model = df_model.dropna(subset=["BasketSize", "ProductRevenue", "TotalQuantity", "AvgUnitPrice", "rule_applied"])
    df_model = df_model[df_model["ProductRevenue"] > 0].copy()

    for col in numeric_cols:
        upper_threshold = df_model[col].quantile(0.995)
        df_model = df_model[df_model[col] <= upper_threshold].copy()

    return df_model, {
        "antecedents": antecedents,
        "consequents": consequents,
        "all_rule_items": all_rule_items,
        "rule_desc": rule_desc,
        "applied_count": int(df_model["rule_applied"].sum()) if not df_model.empty else 0,
        "not_applied_count": int((df_model["rule_applied"] == 0).sum()) if not df_model.empty else 0
    }


def fit_country_regression_model(df_model, model_name, formula, is_log_model=False):
    model = smf.ols(formula=formula, data=df_model).fit()

    coef = model.params.get("rule_applied", np.nan)
    p_value = model.pvalues.get("rule_applied", np.nan)
    t_value = model.tvalues.get("rule_applied", np.nan)
    std_error = model.bse.get("rule_applied", np.nan)

    try:
        ci_lower, ci_upper = model.conf_int().loc["rule_applied"].tolist()
    except Exception:
        ci_lower, ci_upper = np.nan, np.nan

    approx_effect = None
    if is_log_model and pd.notna(coef):
        approx_effect = (np.exp(coef) - 1) * 100

    if pd.notna(p_value) and p_value < 0.05 and coef > 0:
        interpretation = "Positive and statistically significant"
    elif pd.notna(p_value) and p_value < 0.05 and coef < 0:
        interpretation = "Negative and statistically significant"
    else:
        interpretation = "Not statistically significant"

    return {
        "Model": model_name,
        "Formula": formula,
        "N_Observations": int(model.nobs),
        "Rule_Coefficient": round(coef, 4) if pd.notna(coef) else np.nan,
        "Rule_Std_Error": round(std_error, 4) if pd.notna(std_error) else np.nan,
        "Rule_T_Value": round(t_value, 4) if pd.notna(t_value) else np.nan,
        "P_Value": round(p_value, 4) if pd.notna(p_value) else np.nan,
        "CI_Lower": round(ci_lower, 4) if pd.notna(ci_lower) else np.nan,
        "CI_Upper": round(ci_upper, 4) if pd.notna(ci_upper) else np.nan,
        "R_Squared": round(model.rsquared, 4),
        "Adj_R_Squared": round(model.rsquared_adj, 4),
        "Approx_Percentage_Effect_Log_Models": round(approx_effect, 4) if approx_effect is not None else None,
        "Interpretation": interpretation
    }


def run_country_regression_pipeline(df_country, top_rule_row, selected_country):
    output = {
        "selected_country": selected_country,
        "status": "not_started",
        "message": "",
        "rule_metadata": {},
        "model_dataset": pd.DataFrame(),
        "results": pd.DataFrame()
    }

    if df_country.empty:
        output["status"] = "empty_country_dataset"
        output["message"] = "No basket-level data available for this country."
        return output

    df_model, rule_metadata = create_country_rule_applied_dataset(df_country, top_rule_row)

    output["rule_metadata"] = rule_metadata
    output["model_dataset"] = df_model

    if len(df_model) < COUNTRY_REGRESSION_MIN_ROWS:
        output["status"] = "not_enough_rows"
        output["message"] = (
            f"Only {len(df_model):,} usable rows after cleaning. "
            f"Minimum required for country-specific regression is {COUNTRY_REGRESSION_MIN_ROWS:,}."
        )
        return output

    if rule_metadata["applied_count"] < COUNTRY_REGRESSION_MIN_APPLIED:
        output["status"] = "not_enough_rule_applied"
        output["message"] = (
            f"Only {rule_metadata['applied_count']:,} baskets contain the selected rule. "
            f"Minimum required is {COUNTRY_REGRESSION_MIN_APPLIED:,}."
        )
        return output

    df_model = df_model.copy()
    df_model["log_ProductRevenue"] = np.log(df_model["ProductRevenue"])

    formulas = [
        ("Model 1A: Baseline Revenue", "ProductRevenue ~ rule_applied", False),
        ("Model 2A: + BasketSize", "ProductRevenue ~ rule_applied + BasketSize", False),
        ("Model 3A: + AvgUnitPrice", "ProductRevenue ~ rule_applied + BasketSize + AvgUnitPrice", False),
        ("Model 4A: + TotalQuantity", "ProductRevenue ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity", False),
        ("Model 1B: Baseline log Revenue", "log_ProductRevenue ~ rule_applied", True),
        ("Model 2B: log Revenue + BasketSize", "log_ProductRevenue ~ rule_applied + BasketSize", True),
        ("Model 3B: log Revenue + AvgUnitPrice", "log_ProductRevenue ~ rule_applied + BasketSize + AvgUnitPrice", True),
        ("Model 4B: log Revenue + TotalQuantity", "log_ProductRevenue ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity", True),
    ]

    records = []

    for model_name, formula, is_log_model in formulas:
        try:
            records.append(
                fit_country_regression_model(
                    df_model=df_model,
                    model_name=model_name,
                    formula=formula,
                    is_log_model=is_log_model
                )
            )
        except Exception as e:
            records.append({
                "Model": model_name,
                "Formula": formula,
                "N_Observations": len(df_model),
                "Rule_Coefficient": np.nan,
                "Rule_Std_Error": np.nan,
                "Rule_T_Value": np.nan,
                "P_Value": np.nan,
                "CI_Lower": np.nan,
                "CI_Upper": np.nan,
                "R_Squared": np.nan,
                "Adj_R_Squared": np.nan,
                "Approx_Percentage_Effect_Log_Models": None,
                "Interpretation": f"Model failed: {e}"
            })

    results_df = pd.DataFrame(records)

    output["status"] = "completed"
    output["message"] = "Country-specific regression models completed successfully."
    output["results"] = results_df
    output["model_dataset"] = df_model

    return output

# =========================================================
# FLOATING PROJECT ASSISTANT CHATBOX
# =========================================================

PROJECT_KNOWLEDGE = {
    "project_name": "Market Basket Analysis for Cross-Selling Strategy using Online Retail II Dataset",
    "dataset": "Online Retail II basket-level dataset after cleaning and transaction construction.",
    "purpose": (
        "This dashboard helps analyze product co-occurrence patterns, discover association rules, "
        "compare Apriori and FP-Growth, simulate cross-selling actions, and check regression-based robustness."
    ),
    "research_questions": [
        "RQ1: Which basket association rules have the highest lift values while still meeting minimum support and confidence thresholds?",
        "RQ2: Which product categories or product groups show strong cross-selling potential based on co-occurrence and lift?"
    ],
    "team_members": [
        "Không đủ dữ liệu để xác minh. Điền tên thành viên nhóm vào PROJECT_KNOWLEDGE['team_members']."
    ]
}


def _safe_dict_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _format_number(value, decimals=2):
    try:
        if value is None:
            return "N/A"
        if isinstance(value, int):
            return f"{value:,}"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def build_dashboard_context_text(
    selected_country,
    country_filter_audit=None,
    country_mba_outputs=None,
    country_model_outputs=None,
    current_tab=None
):
    audit = country_filter_audit if isinstance(country_filter_audit, dict) else {}
    mba_outputs = country_mba_outputs if isinstance(country_mba_outputs, dict) else {}
    model_outputs = country_model_outputs if isinstance(country_model_outputs, dict) else {}

    country = selected_country if selected_country else "All"
    tab_context = current_tab or st.session_state.get("project_ai_tab_context", "General")
    basket_rows = audit.get("basket_rows", "N/A")
    usable_baskets = audit.get("usable_baskets", "N/A")
    unique_products = audit.get("unique_products", "N/A")
    avg_basket_size = audit.get("avg_basket_size", "N/A")
    total_revenue = audit.get("total_revenue", "N/A")

    status = mba_outputs.get("status", audit.get("status", "N/A"))
    message = mba_outputs.get("message", audit.get("message", "N/A"))

    rules_df = mba_outputs.get("rules")
    strong_rules_df = mba_outputs.get("strong_rules")

    if strong_rules_df is None:
        strong_rules_df = mba_outputs.get("top20")

    runtime_df = mba_outputs.get("runtime_summary")

    if runtime_df is None:
        runtime_df = mba_outputs.get("runtime")

    generated_rules = len(rules_df) if hasattr(rules_df, "__len__") and rules_df is not None else audit.get("generated_rules", "N/A")
    strong_rules = len(strong_rules_df) if hasattr(strong_rules_df, "__len__") and strong_rules_df is not None else audit.get("strong_rules", "N/A")

    top_rule_text = "N/A"
    top_support = "N/A"
    top_confidence = "N/A"
    top_lift = "N/A"

    try:
        if strong_rules_df is not None and len(strong_rules_df) > 0:
            top_rule = strong_rules_df.sort_values(
                by=["lift", "confidence", "support"],
                ascending=False
            ).iloc[0]

            top_rule_text = str(top_rule.get("rule_desc", top_rule.get("rule", "N/A")))
            top_support = _format_number(top_rule.get("support"), 4)
            top_confidence = f"{float(top_rule.get('confidence')) * 100:.2f}%"
            top_lift = _format_number(top_rule.get("lift"), 2)
    except Exception:
        pass

    apriori_runtime = "N/A"
    fpgrowth_runtime = "N/A"
    speedup = "N/A"

    try:
        if runtime_df is not None and len(runtime_df) > 0:
            apriori_row = runtime_df[runtime_df["Algorithm"].astype(str).str.contains("Apriori", case=False, na=False)]
            fpgrowth_row = runtime_df[runtime_df["Algorithm"].astype(str).str.contains("FP", case=False, na=False)]

            if not apriori_row.empty:
                apriori_runtime = float(apriori_row.iloc[0]["Runtime_Seconds"])
            if not fpgrowth_row.empty:
                fpgrowth_runtime = float(fpgrowth_row.iloc[0]["Runtime_Seconds"])
            if apriori_runtime != "N/A" and fpgrowth_runtime != "N/A" and fpgrowth_runtime > 0:
                speedup = apriori_runtime / fpgrowth_runtime
    except Exception:
        pass

    regression_summary = "No regression context available."

    try:
        if isinstance(model_outputs, dict) and model_outputs.get("status") == "completed":
            results_df = model_outputs.get("results")

            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                final_candidates = results_df[
                    results_df["Model"].astype(str).str.contains("Model 5|Model 4A", case=False, na=False)
                ]

                if final_candidates.empty:
                    final_candidates = results_df.tail(1)

                final_row = final_candidates.iloc[-1]

                final_model = final_row.get("Model", "N/A")
                coef = final_row.get("Rule_Coefficient", "N/A")
                pvalue = final_row.get("P_Value", final_row.get("Rule_P_Value", "N/A"))
                r2 = final_row.get("R_Squared", "N/A")

                regression_summary = (
                    f"Final model: {final_model}\n"
                    f"Rule coefficient: {_format_number(coef, 4)}\n"
                    f"p-value: {_format_number(pvalue, 4)}\n"
                    f"R-squared: {_format_number(r2, 4)}"
                )
    except Exception:
        pass

    return {
        "country": country,
        "current_tab": tab_context,
        "basket_rows": basket_rows,
        "usable_baskets": usable_baskets,
        "unique_products": unique_products,
        "avg_basket_size": avg_basket_size,
        "total_revenue": total_revenue,
        "status": status,
        "message": message,
        "generated_rules": generated_rules,
        "strong_rules": strong_rules,
        "top_rule": top_rule_text,
        "top_support": top_support,
        "top_confidence": top_confidence,
        "top_lift": top_lift,
        "apriori_runtime": apriori_runtime,
        "fpgrowth_runtime": fpgrowth_runtime,
        "speedup": speedup,
        "regression_summary": regression_summary,
    }

def explain_current_tab(context):
    tab = context.get("current_tab", "General")
    country = context.get("country", "All")
    status = context.get("status", "N/A")

    if "Executive Overview" in tab:
        return (
            f"Tab hiện tại: Executive Overview.\n\n"
            f"Tab này dùng để xem tổng quan dataset theo country đang chọn: {country}.\n"
            f"Nó hiển thị tổng baskets, total revenue, average basket size, max lift, "
            f"basket size distribution, và top products by frequency."
        )

    if "Rules Explorer" in tab:
        return (
            f"Tab hiện tại: Rules Explorer.\n\n"
            f"Tab này trả lời RQ1 bằng cách hiển thị association rules theo support, confidence, và lift.\n"
            f"Country đang chọn: {country}\n"
            f"Pipeline status: {status}\n\n"
            f"Top rule hiện tại: {context.get('top_rule', 'N/A')}\n"
            f"Support: {context.get('top_support', 'N/A')}\n"
            f"Confidence: {context.get('top_confidence', 'N/A')}\n"
            f"Lift: {context.get('top_lift', 'N/A')}\n\n"
            f"Lưu ý: association rule là pattern đồng xuất hiện, không phải bằng chứng nhân quả."
        )

    if "Bundle Recommendation" in tab:
        return (
            f"Tab hiện tại: Bundle Recommendation.\n\n"
            f"Tab này chuyển strong association rules thành gợi ý hành động cho product team, "
            f"ví dụ checkout recommendation, bundle promotion, hoặc frequently bought together.\n"
            f"Country đang chọn: {country}\n"
            f"Top rule dùng làm gợi ý: {context.get('top_rule', 'N/A')}"
        )

    if "Algorithm Results" in tab:
        speedup_text = "N/A"
        try:
            if context.get("speedup") != "N/A":
                speedup_text = f"{float(context['speedup']):.2f}x"
        except Exception:
            pass

        return (
            f"Tab hiện tại: Algorithm Results.\n\n"
            f"Tab này so sánh Apriori và FP-Growth theo runtime và số frequent itemsets.\n"
            f"Country đang chọn: {country}\n"
            f"Apriori runtime: {context.get('apriori_runtime', 'N/A')}\n"
            f"FP-Growth runtime: {context.get('fpgrowth_runtime', 'N/A')}\n"
            f"FP-Growth speed-up: {speedup_text}"
        )

    if "Add-to-Cart Simulator" in tab:
        return (
            f"Tab hiện tại: Add-to-Cart / Revenue Simulator.\n\n"
            f"Tab này mô phỏng doanh thu từ một kịch bản cross-selling dựa trên rule confidence, "
            f"target customers, conversion rate, expected AOV, discount rate, và campaign cost.\n"
            f"Country đang chọn: {country}\n\n"
            f"Lưu ý: đây là scenario-based estimation, không phải causal proof."
        )

    if "Model Results" in tab:
        return (
            f"Tab hiện tại: Model Results.\n\n"
            f"Tab này dùng regression để kiểm tra robustness của selected rule sau khi thêm basket-level controls.\n"
            f"Country đang chọn: {country}\n\n"
            f"{context.get('regression_summary', 'No regression context available.')}\n\n"
            f"Lưu ý: regression ở đây là observational robustness check, không phải causal proof."
        )

    if "Final Conclusion" in tab:
        return (
            f"Tab hiện tại: Final Conclusion.\n\n"
            f"Tab này tổng hợp kết quả rule mining, algorithm comparison, regression robustness check, "
            f"và business recommendation.\n"
            f"Country đang chọn: {country}\n"
            f"Top rule: {context.get('top_rule', 'N/A')}\n"
            f"Lift: {context.get('top_lift', 'N/A')}"
        )

    if "Run MBA on New Dataset" in tab:
        return (
            "Tab hiện tại: Run MBA on New Dataset.\n\n"
            "Tab này dùng để upload transaction-level CSV mới, validate schema, build baskets, "
            "encode transaction matrix, chạy Apriori/FP-Growth, generate association rules, "
            "và xuất kết quả rule mining."
        )

    if "Run Regression on New Dataset" in tab:
        return (
            "Tab hiện tại: Run Regression on New Dataset.\n\n"
            "Tab này dùng để upload basket-level regression-ready CSV, validate dữ liệu, "
            "tạo biến rule_applied, xử lý outlier, chạy OLS regression, và xuất regression outputs."
        )

    return (
        "Tab context hiện tại chưa được chọn rõ. "
        "Hãy chọn Tab context trong Project Assistant trước khi hỏi về tab hiện tại."
    )

def answer_project_assistant(user_question, context):
    q = user_question.lower().strip()

    if not q:
        return "Nhập câu hỏi về dashboard, dataset, rule mining, regression, hoặc mục tiêu project."
    if any(phrase in q for phrase in [
        "tab này",
        "trang này",
        "màn hình này",
        "đang xem gì",
        "giải thích tab",
        "giải thích trang",
        "kết luận tab",
        "kết luận trang"
    ]):
        return explain_current_tab(context)

    if any(word in q for word in ["mục đích", "purpose", "web này", "dashboard này", "project này", "làm gì"]):
        return (
            f"Project: {PROJECT_KNOWLEDGE['project_name']}\n\n"
            f"Mục đích: {PROJECT_KNOWLEDGE['purpose']}"
        )

    if any(word in q for word in ["dataset", "data", "dữ liệu", "online retail"]):
        return (
            f"Dataset: {PROJECT_KNOWLEDGE['dataset']}\n\n"
            f"Country đang chọn: {context['country']}\n"
            f"Basket rows: {context['basket_rows']}\n"
            f"Usable baskets: {context['usable_baskets']}\n"
            f"Unique products: {context['unique_products']}\n"
            f"Avg basket size: {context['avg_basket_size']}\n"
            f"Total revenue: {context['total_revenue']}"
        )

    if any(word in q for word in ["thành viên", "member", "team", "nhóm"]):
        members = PROJECT_KNOWLEDGE.get("team_members", [])
        return "Thành viên nhóm:\n" + "\n".join([f"- {m}" for m in members])

    if any(word in q for word in ["rq", "research question", "câu hỏi nghiên cứu"]):
        return "Research Questions:\n" + "\n".join([f"- {rq}" for rq in PROJECT_KNOWLEDGE["research_questions"]])

    if any(word in q for word in ["country", "quốc gia", "đang chọn", "selected"]):
        return (
            f"Country đang chọn: {context['country']}\n"
            f"Pipeline status: {context['status']}\n"
            f"Message: {context['message']}"
        )

    if any(word in q for word in ["rule", "association", "lift", "confidence", "support", "luật"]):
        return (
            f"Rule mining summary cho {context['country']}:\n\n"
            f"Generated rules: {context['generated_rules']}\n"
            f"Strong rules: {context['strong_rules']}\n\n"
            f"Top rule: {context['top_rule']}\n"
            f"Support: {context['top_support']}\n"
            f"Confidence: {context['top_confidence']}\n"
            f"Lift: {context['top_lift']}\n\n"
            f"Lưu ý: Association rule là pattern đồng xuất hiện, không phải bằng chứng nhân quả."
        )

    if any(word in q for word in ["apriori", "fp-growth", "fpgrowth", "algorithm", "thuật toán", "runtime"]):
        speedup_text = "N/A"
        try:
            if context["speedup"] != "N/A":
                speedup_text = f"{float(context['speedup']):.2f}x"
        except Exception:
            pass

        return (
            f"Algorithm comparison cho {context['country']}:\n\n"
            f"Apriori runtime: {context['apriori_runtime']}\n"
            f"FP-Growth runtime: {context['fpgrowth_runtime']}\n"
            f"FP-Growth speed-up: {speedup_text}\n\n"
            f"Nếu hai thuật toán trả về cùng số frequent itemsets, khác biệt chính nằm ở thời gian chạy."
        )

    if any(word in q for word in ["regression", "model", "ols", "p-value", "r-squared", "hồi quy"]):
        return (
            f"Regression summary cho {context['country']}:\n\n"
            f"{context['regression_summary']}\n\n"
            f"Lưu ý: Regression section là robustness check quan sát, không phải causal proof."
        )

    if any(word in q for word in ["tab", "chức năng", "hướng dẫn", "dùng sao"]):
        return (
            "Các tab chính:\n"
            "- Executive Overview: tổng quan basket, revenue, product frequency.\n"
            "- Rules Explorer: lọc và xem association rules theo support, confidence, lift.\n"
            "- Bundle Recommendation: chuyển strong rules thành gợi ý cross-selling/bundle.\n"
            "- Algorithm Results: so sánh Apriori và FP-Growth.\n"
            "- Add-to-Cart Simulator: mô phỏng doanh thu theo kịch bản cross-sell.\n"
            "- Model Results: kiểm tra robustness bằng regression.\n"
            "- Final Conclusion: tổng hợp kết quả cuối cùng."
        )

    return (
        "Không nhận diện được ý hỏi.\n\n"
        "Các nhóm câu hỏi đang hỗ trợ:\n"
        "- Dataset / dữ liệu\n"
        "- Top rule / support / confidence / lift\n"
        "- Quốc gia đang chọn\n"
        "- Apriori / FP-Growth / runtime\n"
        "- Regression / OLS / p-value / R-squared\n"
        "- Mục đích dashboard\n"
        "- Thành viên nhóm\n"
        "- Chức năng từng tab"
    )


def render_floating_project_assistant(
    selected_country,
    country_filter_audit=None,
    country_mba_outputs=None,
    country_model_outputs=None
):
    if FLOAT_CHATBOX_AVAILABLE:
        float_init()

    if "project_ai_open" not in st.session_state:
        st.session_state.project_ai_open = False

    if "project_ai_messages" not in st.session_state:
        st.session_state.project_ai_messages = [
            {
                "role": "assistant",
                "content": "Tôi có thể trả lời nhanh về dataset, rules, model, country filter, mục tiêu web, và thành viên nhóm."
            }
        ]
    ASSISTANT_TAB_OPTIONS = [
        "📊 1. Executive Overview",
        "🔍 2. Rules Explorer",
        "🎁 3. Bundle Recommendation",
        "⚙️ 4. Algorithm Results",
        "🧮 5. Add-to-Cart Simulator",
        "📈 6. Model Results",
        "💡 7. Final Conclusion",
        "🧪 8. Run MBA on New Dataset",
        "📈 9. Run Regression on New Dataset"
    ]

    if "project_ai_tab_context" not in st.session_state:
        st.session_state.project_ai_tab_context = ASSISTANT_TAB_OPTIONS[0]
    context = build_dashboard_context_text(
        selected_country=selected_country,
        country_filter_audit=country_filter_audit,
        country_mba_outputs=country_mba_outputs,
        country_model_outputs=country_model_outputs,
        current_tab=st.session_state.get("project_ai_tab_context", ASSISTANT_TAB_OPTIONS[0])
    )

    # ==========================
    # CLOSED MODE: only small icon
    # ==========================
    if not st.session_state.project_ai_open:
        with st.container():
            st.markdown(
                """
                <style>
                div[data-testid="stPopover"] > button {
                    width: 56px !important;
                    height: 56px !important;
                    min-width: 56px !important;
                    min-height: 56px !important;
                    border-radius: 999px !important;
                    padding: 0 !important;

                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;

                    font-size: 24px !important;
                    line-height: 1 !important;

                    background-color: #111827 !important;
                    border: 1px solid rgba(255,255,255,0.24) !important;
                    box-shadow: 0 12px 32px rgba(0,0,0,0.45) !important;
                }

                div[data-testid="stPopover"] > button p {
                    margin: 0 !important;
                    padding: 0 !important;
                    line-height: 1 !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            with st.popover("🤖", help="Open Project Assistant"):
                st.markdown("### 🤖 Project Assistant")
                st.caption(f"Current context: {context['country']}")
                st.selectbox(
                    "Tab context",
                    options=ASSISTANT_TAB_OPTIONS,
                    key="project_ai_tab_context"
                )
                chat_history_box = st.container(height=260)

                with chat_history_box:
                    for msg in st.session_state.project_ai_messages[-8:]:
                        if msg["role"] == "user":
                            st.markdown(f"**You:** {msg['content']}")
                        else:
                            st.markdown(f"**Assistant:** {msg['content']}")

                with st.form("project_ai_form", clear_on_submit=True):
                    user_question = st.text_input(
                        "Ask about this dashboard",
                        placeholder="Ví dụ: top rule của quốc gia này là gì?",
                        key="project_ai_input"
                    )

                    submitted = st.form_submit_button("Send")

                if submitted and user_question.strip():
                    answer = answer_project_assistant(user_question, context)

                    st.session_state.project_ai_messages.append({
                        "role": "user",
                        "content": user_question.strip()
                    })

                    st.session_state.project_ai_messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.rerun()

                if st.button("Clear chat", key="project_ai_clear"):
                    st.session_state.project_ai_messages = [
                        {
                            "role": "assistant",
                            "content": "Chat đã được reset. Hỏi lại về dashboard, dataset, rule mining hoặc regression."
                        }
                    ]
                    st.rerun()

            if FLOAT_CHATBOX_AVAILABLE:
                float_parent(
                    css="""
                    position: fixed;
                    bottom: 86px;
                    right: 24px;
                    width: 56px;
                    height: 56px;
                    z-index: 999999;
                    background: transparent;
                    border: none;
                    padding: 0;
                    box-shadow: none;
                    overflow: visible;
                    """
                )

    # ==========================
    # OPEN MODE: full chat panel
    # ==========================
    else:
        with st.container():
            if st.button(
                "✕ Close AI Assistant",
                key="project_ai_close_button"
            ):
                st.session_state.project_ai_open = False
                st.rerun()
            ##################################################
            st.markdown("### 🤖 Project Assistant")
            st.caption(f"Current context: {context['country']}")
            st.selectbox(
                "Tab context",
                options=ASSISTANT_TAB_OPTIONS,
                key="project_ai_tab_context"
            )
            quick_questions = [
                ("📦 Dataset", "tóm tắt dataset"),
                ("🔝 Top Rule", "top rule là gì"),
                ("⚙️ Algorithm", "so sánh Apriori và FP-Growth"),
                ("📈 Regression", "kết quả regression"),
                ("💡 Recommendation", "gợi ý business recommendation"),
                ("🧭 Current Tab", "Giải thích tab này")
            ]

            quick_cols = st.columns(3)

            for idx, (label, prompt_text) in enumerate(quick_questions):
                with quick_cols[idx % 3]:
                    if st.button(label, key=f"project_ai_quick_{idx}"):
                        answer = answer_project_assistant(prompt_text, context)

                        st.session_state.project_ai_messages.append({
                            "role": "user",
                            "content": prompt_text
                        })

                        st.session_state.project_ai_messages.append({
                            "role": "assistant",
                            "content": answer
                        })

                        st.rerun()

            chat_history_box = st.container(height=260)
            ##################################################
            with chat_history_box:
                for msg in st.session_state.project_ai_messages[-8:]:
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Assistant:** {msg['content']}")

            with st.form("project_ai_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Ask about this dashboard",
                    placeholder="Ví dụ: top rule của quốc gia này là gì?"
                )
                submitted = st.form_submit_button("Send")

            if submitted and user_question.strip():
                answer = answer_project_assistant(user_question, context)

                st.session_state.project_ai_messages.append({
                    "role": "user",
                    "content": user_question.strip()
                })

                st.session_state.project_ai_messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.rerun()

            if st.button("Clear chat", key="project_ai_clear"):
                st.session_state.project_ai_messages = [
                    {
                        "role": "assistant",
                        "content": "Chat đã được reset. Hỏi lại về dashboard, dataset, rule mining hoặc regression."
                    }
                ]
                st.rerun()

            if FLOAT_CHATBOX_AVAILABLE:
                float_parent(
                    css="""
                    position: fixed;
                    bottom: 18px;
                    right: 18px;
                    width: 420px;
                    height: 640px;
                    max-height: calc(100vh - 36px);
                    z-index: 999999;
                    background: #111827;
                    border: 1px solid rgba(255,255,255,0.18);
                    border-radius: 14px;
                    padding: 12px;
                    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
                    overflow-y: auto;
                    """
                )
# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.title("🛒 Parameters")
st.sidebar.markdown("Use the tabs in the main panel to explore different aspects of the Market Basket Analysis.")

country_filter = "All"

if not df_baskets.empty and "Country" in df_baskets.columns:
    countries = (
        df_baskets["Country"]
        .dropna()
        .astype(str)
        .str.strip()
        .sort_values()
        .unique()
        .tolist()
    )

    countries = ["All"] + countries

    country_filter = st.sidebar.selectbox(
        "Filter Country",
        countries,
        key="global_country_filter"
    )
else:
    st.sidebar.warning("Country column not found in basket dataset.")

# ==========================================
# COUNTRY-SCOPED DATA USED BY DASHBOARD
# ==========================================

selected_country = country_filter

df_baskets_country, country_filter_audit = build_country_basket_source(
    df_baskets,
    selected_country
)

with st.sidebar.expander("Country Filter Audit"):
    st.markdown(f"**Selected country:** {country_filter_audit['selected_country']}")
    st.markdown(f"**Basket rows:** {country_filter_audit['basket_rows']:,}")
    st.markdown(f"**Usable baskets:** {country_filter_audit['usable_baskets']:,}")
    st.markdown(f"**Unique products:** {country_filter_audit['unique_products']:,}")
    st.markdown(f"**Avg basket size:** {country_filter_audit['avg_basket_size']:.2f}")
    st.markdown(f"**Total revenue:** £{country_filter_audit['total_revenue']:,.2f}")
# ==========================================
# COUNTRY-SPECIFIC MBA OUTPUTS
# ==========================================

country_mba_outputs = None

if selected_country != "All":
    with st.spinner(f"Running country-specific MBA pipeline for {selected_country}..."):
        country_mba_outputs = run_country_mba_pipeline_cached(
            selected_country=selected_country,
            df_country=df_baskets_country
        )

    st.session_state["country_mba_outputs"] = country_mba_outputs
else:
    st.session_state["country_mba_outputs"] = None

with st.sidebar.expander("Country ARM Pipeline Audit"):
    if selected_country == "All":
        st.markdown("**Mode:** Global precomputed outputs")
        st.markdown("Country-specific ARM is only triggered when a country is selected.")
    else:
        country_mba_outputs = st.session_state.get("country_mba_outputs")

        if country_mba_outputs is None:
            st.markdown("No country-specific ARM output available.")
        else:
            st.markdown(f"**Country:** {country_mba_outputs['selected_country']}")
            st.markdown(f"**Status:** {country_mba_outputs['status']}")
            st.markdown(f"**Message:** {country_mba_outputs['message']}")
            st.markdown(f"**Transactions:** {country_mba_outputs['transactions']:,}")

            matrix_rows, matrix_cols = country_mba_outputs["transaction_matrix_shape"]
            st.markdown(f"**Transaction matrix:** {matrix_rows:,} × {matrix_cols:,}")

            st.markdown(f"**Apriori itemsets:** {len(country_mba_outputs['apriori_itemsets']):,}")
            st.markdown(f"**FP-Growth itemsets:** {len(country_mba_outputs['fpgrowth_itemsets']):,}")
            st.markdown(f"**Generated rules:** {len(country_mba_outputs['rules']):,}")
            st.markdown(f"**Strong rules:** {len(country_mba_outputs['strong_rules']):,}")

# ==========================================
# ACTIVE DASHBOARD OUTPUT SELECTOR
# ==========================================

def get_active_dashboard_outputs(
    selected_country,
    df_rules,
    df_top20,
    df_alg_runtime,
    df_apr_freq,
    df_fp_freq,
    product_map
):
    if selected_country == "All":
        return {
            "mode": "Global precomputed outputs",
            "status": "completed",
            "message": "Using global precomputed dashboard outputs.",
            "rules": df_rules.copy(),
            "top20": df_top20.copy(),
            "runtime": df_alg_runtime.copy(),
            "apriori_itemsets": df_apr_freq.copy(),
            "fpgrowth_itemsets": df_fp_freq.copy()
        }

    country_outputs = st.session_state.get("country_mba_outputs")

    if country_outputs is None:
        return {
            "mode": "Country-specific outputs",
            "status": "not_available",
            "message": "Country-specific MBA output is not available.",
            "rules": pd.DataFrame(),
            "top20": pd.DataFrame(),
            "runtime": pd.DataFrame(),
            "apriori_itemsets": pd.DataFrame(),
            "fpgrowth_itemsets": pd.DataFrame()
        }

    if country_outputs.get("status") != "completed":
        return {
            "mode": "Country-specific outputs",
            "status": country_outputs.get("status", "unknown"),
            "message": country_outputs.get("message", "Country-specific pipeline did not complete."),
            "rules": pd.DataFrame(),
            "top20": pd.DataFrame(),
            "runtime": country_outputs.get("runtime_summary", pd.DataFrame()).copy(),
            "apriori_itemsets": country_outputs.get("apriori_itemsets", pd.DataFrame()).copy(),
            "fpgrowth_itemsets": country_outputs.get("fpgrowth_itemsets", pd.DataFrame()).copy()
        }

    country_rules = country_outputs.get("strong_rules", pd.DataFrame()).copy()

    if country_rules.empty:
        country_rules = country_outputs.get("rules", pd.DataFrame()).copy()

    if not country_rules.empty:
        country_rules = enrich_rules_with_description(country_rules, product_map)
        country_rules = country_rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    country_top20 = country_rules.head(20).copy()

    return {
        "mode": "Country-specific outputs",
        "status": "completed",
        "message": country_outputs.get("message", "Country-specific MBA pipeline completed successfully."),
        "rules": country_rules,
        "top20": country_top20,
        "runtime": country_outputs.get("runtime_summary", pd.DataFrame()).copy(),
        "apriori_itemsets": country_outputs.get("apriori_itemsets", pd.DataFrame()).copy(),
        "fpgrowth_itemsets": country_outputs.get("fpgrowth_itemsets", pd.DataFrame()).copy()
    }


active_outputs = get_active_dashboard_outputs(
    selected_country=selected_country,
    df_rules=df_rules,
    df_top20=df_top20,
    df_alg_runtime=df_alg_runtime,
    df_apr_freq=df_apr_freq,
    df_fp_freq=df_fp_freq,
    product_map=product_map
)

active_rules = active_outputs["rules"]
active_top20 = active_outputs["top20"]
active_alg_runtime = active_outputs["runtime"]
active_apr_freq = active_outputs["apriori_itemsets"]
active_fp_freq = active_outputs["fpgrowth_itemsets"]


# ==========================================
# ACTIVE MODEL RESULTS OUTPUT SELECTOR
# ==========================================

global_model_results = pd.DataFrame()

for candidate_name in [
    "df_model",
    "df_causal",
    "df_model_results",
    "df_final_causal",
    "df_final_causal_impact",
    "final_causal_impact_summary"
]:
    if candidate_name in globals():
        candidate_df = globals()[candidate_name]
        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
            global_model_results = candidate_df.copy()
            break

country_model_outputs = None

if selected_country != "All":
    if active_outputs["status"] == "completed" and not active_top20.empty:
        country_model_outputs = run_country_regression_pipeline(
            df_country=df_baskets_country,
            top_rule_row=active_top20.iloc[0],
            selected_country=selected_country
        )
    else:
        country_model_outputs = {
            "selected_country": selected_country,
            "status": active_outputs["status"],
            "message": active_outputs["message"],
            "rule_metadata": {},
            "model_dataset": pd.DataFrame(),
            "results": pd.DataFrame()
        }

st.session_state["country_model_outputs"] = country_model_outputs

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
# ------------------------------------------
# TAB 2: RULES EXPLORER
# ------------------------------------------
with tabs[1]:
    st.header("Rules Explorer")
    st.markdown("**(RQ1)** Which basket association rules have the highest lift values while still meeting minimum support and confidence thresholds?")

    st.caption(
        f"Current output mode: {active_outputs['mode']} | "
        f"Selected country: {selected_country} | "
        f"Status: {active_outputs['status']}"
    )

    if active_rules.empty:
        st.warning(active_outputs["message"])
    else:
        rc1, rc2, rc3 = st.columns(3)

        supp_min = float(active_rules["support"].min())
        supp_max = float(active_rules["support"].max())
        conf_min = float(active_rules["confidence"].min())
        conf_max = float(active_rules["confidence"].max())
        lift_min = float(active_rules["lift"].min())
        lift_max = float(active_rules["lift"].max())

        if supp_min == supp_max:
            min_supp = supp_min
            rc1.metric("Min Support", f"{min_supp:.4f}")
        else:
            min_supp = rc1.slider(
                "Min Support",
                supp_min,
                supp_max,
                supp_min,
                key=f"rules_min_support_{selected_country}"
            )

        if conf_min == conf_max:
            min_conf = conf_min
            rc2.metric("Min Confidence", f"{min_conf:.2f}")
        else:
            min_conf = rc2.slider(
                "Min Confidence",
                conf_min,
                conf_max,
                conf_min,
                key=f"rules_min_confidence_{selected_country}"
            )

        if lift_min == lift_max:
            min_lift = lift_min
            rc3.metric("Min Lift", f"{min_lift:.2f}")
        else:
            min_lift = rc3.slider(
                "Min Lift",
                lift_min,
                lift_max,
                lift_min,
                key=f"rules_min_lift_{selected_country}"
            )

        antecedent_filter = st.text_input(
            "Filter by Antecedent Product / StockCode (leave blank for all):",
            key=f"antecedent_filter_{selected_country}"
        )

        filtered_rules = active_rules[
            (active_rules["support"] >= min_supp)
            & (active_rules["confidence"] >= min_conf)
            & (active_rules["lift"] >= min_lift)
        ].copy()

        if antecedent_filter:
            filter_cols = [
                col for col in [
                    "antecedents_display",
                    "antecedents_desc",
                    "antecedents_str"
                ]
                if col in filtered_rules.columns
            ]

            if filter_cols:
                mask = False
                for col in filter_cols:
                    mask = mask | filtered_rules[col].astype(str).str.contains(
                        antecedent_filter,
                        case=False,
                        na=False
                    )
                filtered_rules = filtered_rules[mask]

        if filtered_rules.empty:
            st.warning("No rules match the current filters.")
        else:
            hover_cols = [
                col for col in [
                    "rule_desc",
                    "rule_display",
                    "support",
                    "confidence",
                    "lift"
                ]
                if col in filtered_rules.columns
            ]

            fig_scatter = px.scatter(
                filtered_rules,
                x="support",
                y="confidence",
                color="lift",
                hover_data=hover_cols,
                title=f"Support vs Confidence Colored by Lift - {selected_country}"
            )

            fig_scatter.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

            available_cols = [
                col for col in [
                    "rule_desc",
                    "rule_display",
                    "support",
                    "confidence",
                    "lift",
                    "leverage",
                    "conviction"
                ]
                if col in filtered_rules.columns
            ]

            st.dataframe(
                filtered_rules[available_cols]
                .sort_values("lift", ascending=False)
                .head(50),
                use_container_width=True
            )

            best_rule = filtered_rules.sort_values("lift", ascending=False).iloc[0]

            st.markdown(f"""
            <div class="insight-box">
                <b>Auto-Insight:</b> The strongest rule for <b>{selected_country}</b> is 
                <b>{best_rule['rule_desc']}</b>.<br>
                Lift = <b>{best_rule['lift']:.2f}</b>, 
                confidence = <b>{best_rule['confidence']:.1%}</b>, 
                support = <b>{best_rule['support']:.4f}</b>.<br>
                This is an association pattern, not causal proof.
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: BUNDLE RECOMMENDATION
# ------------------------------------------
# ------------------------------------------
# TAB 3: BUNDLE RECOMMENDATION
# ------------------------------------------
with tabs[2]:
    st.header("Bundle Recommendation for Product Team")
    st.markdown("Translate strong association rules into actionable business strategies.")

    st.caption(
        f"Current output mode: {active_outputs['mode']} | "
        f"Selected country: {selected_country} | "
        f"Status: {active_outputs['status']}"
    )

    if active_top20.empty:
        st.warning(active_outputs["message"])
    else:
        top_bundle_rules = (
            active_top20
            .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
            .head(10)
            .reset_index(drop=True)
        )

        for idx, row in top_bundle_rules.iterrows():
            if row["confidence"] >= 0.70:
                action = "Recommend consequent at Checkout"
            elif row["lift"] > 10:
                action = "Create Bundle Promotion"
            else:
                action = "Add to Frequently Bought Together"

            antecedent_text = row.get("antecedents_display", row.get("antecedents_str", ""))
            consequent_text = row.get("consequents_display", row.get("consequents_str", ""))

            st.markdown(f"""
            <div class="glass-card" style="padding: 15px;">
                <h4 style="color:#f4a460; margin-bottom:5px;">
                    Bundle Idea {idx + 1}: {row['rule_desc']}
                </h4>
                <b>Antecedent:</b> {antecedent_text}<br>
                <b>Recommended Consequent:</b> {consequent_text}<br>
                <span style="color:#A0A0A0;">
                    Support: {row['support']:.4f} | 
                    Confidence: {row['confidence']:.2%} | 
                    Lift: {row['lift']:.2f}
                </span><br>
                <b style="color:#50C878;">Suggested Action:</b> {action}
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: ALGORITHM RESULTS
# ------------------------------------------
# ------------------------------------------
# TAB 4: ALGORITHM RESULTS
# ------------------------------------------
with tabs[3]:
    st.header("Algorithm Results & Comparison")

    st.caption(
        f"Current output mode: {active_outputs['mode']} | "
        f"Selected country: {selected_country} | "
        f"Status: {active_outputs['status']}"
    )

    has_algo_outputs = (
        not active_alg_runtime.empty
        or not active_apr_freq.empty
        or not active_fp_freq.empty
    )

    if not has_algo_outputs:
        st.warning(active_outputs["message"])
    else:
        st.success("Displaying Apriori vs FP-Growth comparison for current selection.")

        if not active_alg_runtime.empty:
            st.markdown("### Runtime Summary")
            st.dataframe(active_alg_runtime, use_container_width=True)

            if {"Algorithm", "Runtime_Seconds"}.issubset(active_alg_runtime.columns):
                runtime_for_chart = active_alg_runtime.dropna(subset=["Runtime_Seconds"]).copy()

                if not runtime_for_chart.empty:
                    fig_rt = px.bar(
                        runtime_for_chart,
                        x="Algorithm",
                        y="Runtime_Seconds",
                        title=f"Runtime Comparison - {selected_country}",
                        color="Algorithm"
                    )

                    fig_rt.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )

                    st.plotly_chart(fig_rt, use_container_width=True)

                apr_time = active_alg_runtime.loc[
                    active_alg_runtime["Algorithm"] == "Apriori",
                    "Runtime_Seconds"
                ]

                fp_time = active_alg_runtime.loc[
                    active_alg_runtime["Algorithm"] == "FP-Growth",
                    "Runtime_Seconds"
                ]

                if (
                    not apr_time.empty
                    and not fp_time.empty
                    and pd.notna(apr_time.iloc[0])
                    and pd.notna(fp_time.iloc[0])
                    and fp_time.iloc[0] > 0
                ):
                    speedup = apr_time.iloc[0] / fp_time.iloc[0]
                    st.metric("FP-Growth Speed-up vs Apriori", f"{speedup:.1f}x faster")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Apriori Frequent Itemsets", f"{len(active_apr_freq):,}")

            if not active_apr_freq.empty:
                display_apr = active_apr_freq.copy()

                if "itemsets_str" not in display_apr.columns and "itemsets" in display_apr.columns:
                    display_apr["itemsets_str"] = display_apr["itemsets"].apply(itemset_to_text)

                cols = [col for col in ["itemsets_str", "itemset_size", "support"] if col in display_apr.columns]
                st.dataframe(
                    display_apr.sort_values("support", ascending=False)[cols].head(10),
                    use_container_width=True
                )

        with c2:
            st.metric("FP-Growth Frequent Itemsets", f"{len(active_fp_freq):,}")

            if not active_fp_freq.empty:
                display_fp = active_fp_freq.copy()

                if "itemsets_str" not in display_fp.columns and "itemsets" in display_fp.columns:
                    display_fp["itemsets_str"] = display_fp["itemsets"].apply(itemset_to_text)

                cols = [col for col in ["itemsets_str", "itemset_size", "support"] if col in display_fp.columns]
                st.dataframe(
                    display_fp.sort_values("support", ascending=False)[cols].head(10),
                    use_container_width=True
                )
# ------------------------------------------
# TAB 5: AOV / ADD-TO-CART SIMULATOR
# ------------------------------------------
# ------------------------------------------
# TAB 5: AOV / ADD-TO-CART SIMULATOR
# ------------------------------------------
with tabs[4]:
    st.header("Add-to-Cart / Revenue Simulator")
    st.markdown("*This simulator is scenario-based estimation, not causal proof.*")

    st.caption(
        f"Current output mode: {active_outputs['mode']} | "
        f"Selected country: {selected_country} | "
        f"Status: {active_outputs['status']}"
    )

    if active_rules.empty:
        st.warning(active_outputs["message"])
    else:
        s1, s2 = st.columns([1, 2])

        with s1:
            st.markdown("### Parameters")

            rule_options = (
                active_rules
                .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
                .head(50)
                .copy()
            )

            selected_rule_idx = st.selectbox(
                "Select Rule to Simulate",
                options=rule_options.index,
                format_func=lambda i: rule_options.loc[i, "rule_desc"],
                key=f"sim_rule_select_{selected_country}"
            )

            target_customers = st.number_input(
                "Target Customers (Antecedent buyers)",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                key=f"sim_target_customers_{selected_country}"
            )

            conversion_rate = st.slider(
                "Expected Conversion Rate",
                0.01,
                1.0,
                0.05,
                0.01,
                key=f"sim_conversion_rate_{selected_country}"
            )

            expected_aov = st.number_input(
                "Expected AOV of Consequent (£)",
                min_value=1.0,
                max_value=500.0,
                value=15.0,
                key=f"sim_expected_aov_{selected_country}"
            )

            discount_rate = st.slider(
                "Discount Rate Applied",
                0.0,
                0.5,
                0.1,
                0.05,
                key=f"sim_discount_rate_{selected_country}"
            )

            campaign_cost = st.number_input(
                "Campaign Setup Cost (£)",
                value=100.0,
                key=f"sim_campaign_cost_{selected_country}"
            )

        with s2:
            rule_data = rule_options.loc[selected_rule_idx]
            conf = rule_data["confidence"]

            est_add_to_cart = target_customers * conf
            est_converted = round(est_add_to_cart * conversion_rate)
            gross_revenue = est_converted * expected_aov
            discount_cost = gross_revenue * discount_rate
            net_revenue = gross_revenue - discount_cost - campaign_cost

            st.markdown("### Simulation Results")
            st.markdown(f"""
            <div class="glass-card">
                <p><b>Selected country:</b> {selected_country}</p>
                <p><b>Selected rule:</b> {rule_data['rule_desc']}</p>
                <p>Rule Confidence: <b>{conf:.2%}</b></p>
                <p>Estimated Add-to-Carts: <b>{est_add_to_cart:.0f}</b></p>
                <p>Estimated Converted Orders: <b>{est_converted:.0f}</b></p>
                <hr>
                <p>Gross Revenue: <b style='color:#50C878;'>£{gross_revenue:,.2f}</b></p>
                <p>Discount Cost: <b style='color:#FF6347;'>-£{discount_cost:,.2f}</b></p>
                <p>Campaign Cost: <b style='color:#FF6347;'>-£{campaign_cost:,.2f}</b></p>
                <h3>
                    Estimated Net Revenue: 
                    <span style='color: {"#50C878" if net_revenue > 0 else "#FF6347"};'>
                        £{net_revenue:,.2f}
                    </span>
                </h3>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 6: MODEL RESULTS
# ------------------------------------------
with tabs[5]:
    st.header("Regression-Based Robustness Check")

    st.caption(
        f"Current selection: {selected_country} | "
        f"{'Global precomputed model results' if selected_country == 'All' else 'Country-specific regression'}"
    )

    if selected_country == "All":
        model_results_to_show = global_model_results.copy()

        if model_results_to_show.empty:
            st.warning("Global regression result file is not available.")
        else:
            st.dataframe(model_results_to_show, use_container_width=True)

            c1, c2 = st.columns(2)

            with c1:
                if {"Model", "Rule_Coefficient"}.issubset(model_results_to_show.columns):
                    fig_coef = px.bar(
                        model_results_to_show,
                        x="Model",
                        y="Rule_Coefficient",
                        title="Rule Coefficient by Model",
                        color="Interpretation" if "Interpretation" in model_results_to_show.columns else None
                    )

                    fig_coef.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )

                    st.plotly_chart(fig_coef, use_container_width=True)

            with c2:
                if {"Model", "R_Squared"}.issubset(model_results_to_show.columns):
                    fig_r2 = px.bar(
                        model_results_to_show,
                        x="Model",
                        y="R_Squared",
                        title="R-Squared by Model"
                    )

                    fig_r2.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )

                    st.plotly_chart(fig_r2, use_container_width=True)

            final_candidates = model_results_to_show[
                model_results_to_show["Model"].astype(str).str.contains("Model 5", case=False, na=False)
            ]

            if final_candidates.empty:
                final_candidates = model_results_to_show.tail(1)

            final_row = final_candidates.iloc[-1]

            st.markdown(f"""
            <div class="insight-box">
                <b>Conclusion for All Countries:</b><br>
                Final model: <b>{final_row.get("Model", "N/A")}</b><br>
                Rule coefficient: <b>{final_row.get("Rule_Coefficient", np.nan)}</b><br>
                p-value: <b>{final_row.get("P_Value", np.nan)}</b><br>
                R-squared: <b>{final_row.get("R_Squared", np.nan)}</b><br><br>
                This is an observational robustness check, not causal proof.
            </div>
            """, unsafe_allow_html=True)

    else:
        country_model_outputs = st.session_state.get("country_model_outputs")

        if country_model_outputs is None:
            st.warning("Country-specific regression output is not available.")
        elif country_model_outputs["status"] != "completed":
            st.warning(country_model_outputs["message"])
        else:
            country_results = country_model_outputs["results"].copy()
            country_model_df = country_model_outputs["model_dataset"].copy()
            rule_metadata = country_model_outputs["rule_metadata"]

            st.success(country_model_outputs["message"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Regression Rows", f"{len(country_model_df):,}")
            m2.metric("Rule Applied Baskets", f"{rule_metadata['applied_count']:,}")
            m3.metric("Rule Not Applied Baskets", f"{rule_metadata['not_applied_count']:,}")
            m4.metric(
                "Applied Rate",
                f"{rule_metadata['applied_count'] / len(country_model_df):.2%}"
                if len(country_model_df) > 0 else "0.00%"
            )

            st.markdown(f"""
            <div class="insight-box">
                <b>Selected country:</b> {selected_country}<br>
                <b>Selected rule for regression:</b> {rule_metadata["rule_desc"]}<br>
                <b>Note:</b> Country-specific regression does not include Country as a control because the selected dataset contains only one country.
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Country-Specific Regression Results")
            st.dataframe(country_results, use_container_width=True)

            c1, c2 = st.columns(2)

            with c1:
                fig_coef = px.bar(
                    country_results,
                    x="Model",
                    y="Rule_Coefficient",
                    title=f"Rule Coefficient by Model - {selected_country}",
                    color="Interpretation"
                )

                fig_coef.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(fig_coef, use_container_width=True)

            with c2:
                fig_r2 = px.bar(
                    country_results,
                    x="Model",
                    y="R_Squared",
                    title=f"R-Squared by Model - {selected_country}"
                )

                fig_r2.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(fig_r2, use_container_width=True)

            final_controlled = country_results[
                country_results["Model"].astype(str).str.contains("Model 4A", case=False, na=False)
            ]

            if final_controlled.empty:
                final_controlled = country_results.tail(1)

            final_row = final_controlled.iloc[0]

            coef = final_row["Rule_Coefficient"]
            p_value = final_row["P_Value"]
            r_squared = final_row["R_Squared"]

            if pd.notna(p_value) and p_value < 0.05:
                conclusion_text = (
                    f"For {selected_country}, the selected rule remains statistically significant "
                    f"after controlling for BasketSize, AvgUnitPrice, and TotalQuantity."
                )
            else:
                conclusion_text = (
                    f"For {selected_country}, the selected rule is not statistically significant "
                    f"after controlling for BasketSize, AvgUnitPrice, and TotalQuantity."
                )

            st.markdown(f"""
            <div class="insight-box">
                <b>Country-Specific Regression Conclusion:</b><br>
                Final controlled model: <b>{final_row["Model"]}</b><br>
                Rule coefficient: <b>{coef}</b><br>
                p-value: <b>{p_value}</b><br>
                R-squared: <b>{r_squared}</b><br><br>
                {conclusion_text}<br><br>
                <b>Important note:</b> This is an observational robustness check, not causal proof.
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                label=f"Download {selected_country} Regression Results CSV",
                data=convert_df_to_csv_bytes(country_results),
                file_name=f"{selected_country.lower().replace(' ', '_')}_regression_results.csv",
                mime="text/csv"
            )
# ------------------------------------------
# TAB 7: FINAL CONCLUSION
# ------------------------------------------
with tabs[6]:
    st.header("Final Conclusion")

    import ast

    def _as_df(obj):
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()

    def _pick_dict(d, keys, default=None):
        if not isinstance(d, dict):
            return default
        for key in keys:
            if key in d:
                val = d.get(key)
                if val is not None and val != "":
                    return val
        return default

    def _row_value(row, keys, default=None):
        for key in keys:
            if hasattr(row, "index") and key in row.index:
                val = row.get(key)
                if val is not None and val != "":
                    return val
        return default

    def _fmt_int(value):
        try:
            if value is None or pd.isna(value):
                return "N/A"
            return f"{int(float(value)):,}"
        except Exception:
            return "N/A"

    def _fmt_num(value, digits=4):
        try:
            if value is None or pd.isna(value):
                return "N/A"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "N/A"

    def _fmt_pct(value):
        try:
            if value is None or pd.isna(value):
                return "N/A"
            value = float(value)
            if value <= 1:
                return f"{value:.2%}"
            return f"{value:.2f}%"
        except Exception:
            return "N/A"

    def _extract_items(value):
        if isinstance(value, (list, tuple, set)):
            return [str(x).strip() for x in value if str(x).strip()]

        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        return [x.strip() for x in text.split(",") if x.strip()]

    def _find_basket_source_df():
        preferred_names = [
            "active_basket_df",
            "filtered_basket_df",
            "df_basket_filtered",
            "df_basket",
            "basket_df",
            "online_retail_ii_basket_df"
        ]

        for name in preferred_names:
            obj = globals().get(name)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                return obj.copy()

        for _, obj in list(globals().items()):
            if (
                isinstance(obj, pd.DataFrame)
                and not obj.empty
                and {"InvoiceNo", "Items", "BasketSize"}.issubset(set(obj.columns))
            ):
                return obj.copy()

        return pd.DataFrame()

    def _filter_selected_baskets(source_df, selected_country_value):
        if source_df.empty:
            return source_df

        if selected_country_value != "All" and "Country" in source_df.columns:
            return source_df[
                source_df["Country"].astype(str).str.strip()
                == str(selected_country_value).strip()
            ].copy()

        return source_df.copy()

    def _unique_item_count(basket_df_value):
        if basket_df_value.empty:
            return 0

        if "StockCode" in basket_df_value.columns:
            return basket_df_value["StockCode"].astype(str).nunique()

        if "Items" in basket_df_value.columns:
            unique_items = set()
            for items_value in basket_df_value["Items"].dropna():
                unique_items.update(_extract_items(items_value))
            return len(unique_items)

        return 0

    active_outputs_safe = globals().get("active_outputs", {})
    active_rules_safe = _as_df(globals().get("active_rules", pd.DataFrame()))
    active_top20_safe = _as_df(globals().get("active_top20", pd.DataFrame()))
    active_alg_runtime_safe = _as_df(globals().get("active_alg_runtime", pd.DataFrame()))

    global_model_results_safe = pd.DataFrame()
    for candidate_name in [
        "global_model_results",
        "df_model",
        "df_causal",
        "df_model_results",
        "df_final_causal",
        "df_final_causal_impact"
    ]:
        candidate_df = globals().get(candidate_name)
        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
            global_model_results_safe = candidate_df.copy()
            break

    country_min_transactions = globals().get("COUNTRY_ARM_MIN_BASKETS", 100)

    country_filter_audit_safe = globals().get("country_filter_audit", {}) or {}

    if selected_country == "All":
        country_mba_outputs_safe = None
    else:
        country_mba_outputs_safe = st.session_state.get("country_mba_outputs")

    basket_rows = int(country_filter_audit_safe.get("basket_rows", 0))
    usable_baskets = int(country_filter_audit_safe.get("usable_baskets", 0))
    unique_products = int(country_filter_audit_safe.get("unique_products", 0))

    if selected_country == "All":
        generated_rules_count = len(df_rules) if isinstance(df_rules, pd.DataFrame) else len(active_rules_safe)
        strong_rules_count = len(df_rules) if isinstance(df_rules, pd.DataFrame) else len(active_rules_safe)

    else:
        if isinstance(country_mba_outputs_safe, dict):
            generated_rules_count = len(_as_df(country_mba_outputs_safe.get("rules", pd.DataFrame())))
            strong_rules_count = len(_as_df(country_mba_outputs_safe.get("strong_rules", pd.DataFrame())))

            matrix_shape = country_mba_outputs_safe.get("transaction_matrix_shape", (0, 0))
            if (
                isinstance(matrix_shape, tuple)
                and len(matrix_shape) == 2
                and int(matrix_shape[1]) > 0
            ):
                unique_products = int(matrix_shape[1])

            usable_baskets = int(country_mba_outputs_safe.get("transactions", usable_baskets))
        else:
            generated_rules_count = 0
            strong_rules_count = 0

    output_status = _pick_dict(active_outputs_safe, ["status"], "global")
    output_message = _pick_dict(active_outputs_safe, ["message"], "")
############################################################
    st.caption(
        f"Current selection: {selected_country} | "
        f"{'Global conclusion' if selected_country == 'All' else 'Country-specific conclusion'}"
    )

    if selected_country == "All":
        st.subheader("Overall Market Basket Analysis Conclusion")

        with st.container(border=True):
            st.subheader("Global Association Rule Mining Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total baskets", _fmt_int(basket_rows))
            c2.metric("Usable baskets", _fmt_int(usable_baskets))
            c3.metric("Unique products", _fmt_int(unique_products))
            c4.metric("Generated rules", _fmt_int(generated_rules_count))

            if active_top20_safe.empty:
                st.warning("No global top association rules are available.")
            else:
                top_rule = active_top20_safe.iloc[0]

                top_rule_desc = _row_value(
                    top_rule,
                    ["rule_desc", "rule", "rule_display"],
                    "N/A"
                )
                top_support = _row_value(top_rule, ["support"], None)
                top_confidence = _row_value(top_rule, ["confidence"], None)
                top_lift = _row_value(top_rule, ["lift"], None)

                st.write(f"**Top global rule:** {top_rule_desc}")

                r1, r2, r3 = st.columns(3)
                r1.metric("Support", _fmt_num(top_support, 4))
                r2.metric("Confidence", _fmt_pct(top_confidence))
                r3.metric("Lift", _fmt_num(top_lift, 2))

                st.write(
                    "The global dataset contains strong product co-occurrence patterns "
                    "that can support cross-selling and bundle recommendation decisions."
                )
                st.caption("Association rules are pattern-based signals, not causal proof.")

        if not global_model_results_safe.empty:
            final_model = global_model_results_safe.tail(1).iloc[0]

            with st.container(border=True):
                st.subheader("Global Regression Robustness Check")

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Rule coefficient",
                    _fmt_num(_row_value(final_model, ["Rule_Coefficient", "rule_coefficient"], None), 4)
                )
                m2.metric(
                    "p-value",
                    _fmt_num(_row_value(final_model, ["P_Value", "p_value", "Rule_P_Value"], None), 4)
                )
                m3.metric(
                    "R-squared",
                    _fmt_num(_row_value(final_model, ["R_Squared", "r_squared"], None), 4)
                )

                st.write(f"**Final model:** {_row_value(final_model, ['Model'], 'N/A')}")
                st.write(
                    "The regression section is used as an observational robustness check. "
                    "It should not be interpreted as causal evidence."
                )

        with st.container(border=True):
            st.subheader("Final Business Recommendation")
            st.write(
                "Use high-lift and high-confidence rules as candidates for checkout recommendations, "
                "bundle promotions, product placement, and add-to-cart suggestions."
            )
            st.caption("Final implementation should be validated with real business experiments before deployment.")

    else:
        st.subheader(f"Country-Specific Conclusion: {selected_country}")

        if output_status != "completed":
            st.warning(output_message)

            with st.container(border=True):
                st.subheader(f"Conclusion for {selected_country}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Basket rows", _fmt_int(basket_rows))
                c2.metric("Usable baskets", _fmt_int(usable_baskets))
                c3.metric("Minimum required", _fmt_int(country_min_transactions))

                st.write(
                    "Country-specific association rule mining was not executed because "
                    "the selected country does not have enough usable baskets."
                )
                st.write(
                    "Do not generate rule-based bundle or simulator recommendations for this country. "
                    "Use the global output only as general reference, not as a country-specific result."
                )

        elif active_top20_safe.empty:
            st.warning("Country-specific mining completed, but no strong rules were found.")

            with st.container(border=True):
                st.subheader(f"Conclusion for {selected_country}")
                st.write(
                    "The selected country has enough baskets for mining, but no strong association rules "
                    "passed the current support, confidence, and lift thresholds."
                )
                st.write(
                    "Lower thresholds carefully or collect more transaction data before making "
                    "country-specific cross-selling recommendations."
                )

        else:
            top_rule = active_top20_safe.iloc[0]

            top_rule_desc = _row_value(
                top_rule,
                ["rule_desc", "rule", "rule_display"],
                "N/A"
            )
            top_support = _row_value(top_rule, ["support"], None)
            top_confidence = _row_value(top_rule, ["confidence"], None)
            top_lift = _row_value(top_rule, ["lift"], None)

            with st.container(border=True):
                st.subheader(f"Association Rule Mining Result for {selected_country}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Usable baskets", _fmt_int(usable_baskets))
                c2.metric("Unique products", _fmt_int(unique_products))
                c3.metric("Generated rules", _fmt_int(generated_rules_count))
                c4.metric("Strong rules", _fmt_int(strong_rules_count))

                st.write(f"**Top country-specific rule:** {top_rule_desc}")

                r1, r2, r3 = st.columns(3)
                r1.metric("Support", _fmt_num(top_support, 4))
                r2.metric("Confidence", _fmt_pct(top_confidence))
                r3.metric("Lift", _fmt_num(top_lift, 2))

                st.write(
                    f"{selected_country} has country-specific basket patterns that differ from the global output. "
                    "These rules should be used for localized cross-selling recommendations."
                )
                st.caption("Association rules are pattern-based signals, not causal proof.")

            if not active_alg_runtime_safe.empty and {
                "Algorithm",
                "Runtime_Seconds"
            }.issubset(active_alg_runtime_safe.columns):
                runtime_df = active_alg_runtime_safe.copy()

                apriori_runtime = runtime_df.loc[
                    runtime_df["Algorithm"].astype(str).str.lower().eq("apriori"),
                    "Runtime_Seconds"
                ]

                fpgrowth_runtime = runtime_df.loc[
                    runtime_df["Algorithm"].astype(str).str.lower().isin(["fp-growth", "fpgrowth"]),
                    "Runtime_Seconds"
                ]

                apriori_time = float(apriori_runtime.iloc[0]) if not apriori_runtime.empty else None
                fpgrowth_time = float(fpgrowth_runtime.iloc[0]) if not fpgrowth_runtime.empty else None

                with st.container(border=True):
                    st.subheader(f"Algorithm Comparison for {selected_country}")

                    a1, a2, a3 = st.columns(3)
                    a1.metric("Apriori runtime", _fmt_num(apriori_time, 4))
                    a2.metric("FP-Growth runtime", _fmt_num(fpgrowth_time, 4))

                    if apriori_time is not None and fpgrowth_time is not None and fpgrowth_time > 0:
                        speedup = apriori_time / fpgrowth_time
                        a3.metric("FP-Growth speed-up", f"{speedup:.2f}x")
                        st.write(f"FP-Growth was {speedup:.2f}x faster than Apriori for {selected_country}.")
                    else:
                        a3.metric("FP-Growth speed-up", "N/A")
                        st.write("Runtime comparison is available, but speed-up cannot be calculated.")

            country_model_outputs = st.session_state.get("country_model_outputs", {})
            country_results = _as_df(country_model_outputs.get("results", pd.DataFrame()))
            country_model_status = country_model_outputs.get("status", "")

            if country_model_status == "completed" and not country_results.empty:
                final_controlled = country_results[
                    country_results["Model"].astype(str).str.contains("Model 4A", case=False, na=False)
                ]

                if final_controlled.empty:
                    final_controlled = country_results.tail(1)

                final_row = final_controlled.iloc[0]

                coef = _row_value(final_row, ["Rule_Coefficient", "rule_coefficient"], None)
                p_value = _row_value(final_row, ["P_Value", "p_value", "Rule_P_Value"], None)
                r_squared = _row_value(final_row, ["R_Squared", "r_squared"], None)

                try:
                    p_value_float = float(p_value)
                except Exception:
                    p_value_float = None

                if p_value_float is not None and p_value_float < 0.05:
                    regression_conclusion = (
                        "The selected rule remains statistically significant after basket-level controls."
                    )
                else:
                    regression_conclusion = (
                        "The selected rule is not statistically significant after basket-level controls."
                    )

                with st.container(border=True):
                    st.subheader(f"Regression Robustness Check for {selected_country}")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rule coefficient", _fmt_num(coef, 4))
                    m2.metric("p-value", _fmt_num(p_value, 4))
                    m3.metric("R-squared", _fmt_num(r_squared, 4))

                    st.write(f"**Final controlled model:** {_row_value(final_row, ['Model'], 'N/A')}")
                    st.write(regression_conclusion)
                    st.caption("This regression result is observational and should not be interpreted as causal proof.")

            else:
                with st.container(border=True):
                    st.subheader(f"Regression Robustness Check for {selected_country}")
                    st.write(
                        "Country-specific regression output is not available or was not completed. "
                        "The final recommendation should rely only on association-rule evidence for this country."
                    )

            with st.container(border=True):
                st.subheader(f"Final Business Recommendation for {selected_country}")
                st.write(
                    "Prioritize the strongest country-specific rules for localized cross-selling actions. "
                    "Recommended actions include checkout recommendations, bundle promotion tests, "
                    "and comparison against the global rule set."
                )
                st.caption("These recommendations are candidate business actions and require real-world validation.")
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
assistant_model_context = country_model_outputs

if selected_country == "All":
    assistant_model_context = {
        "status": "completed",
        "results": global_model_results
    }

render_floating_project_assistant(
    selected_country=selected_country,
    country_filter_audit=country_filter_audit,
    country_mba_outputs=active_outputs,
    country_model_outputs=assistant_model_context
)


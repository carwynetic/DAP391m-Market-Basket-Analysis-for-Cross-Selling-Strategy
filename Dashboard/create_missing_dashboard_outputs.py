import os
import ast
import time
import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

import statsmodels.formula.api as smf


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data"

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.2

RUN_ALGORITHM_OUTPUTS = True
RUN_ADD_TO_CART_SIMULATION = True
RUN_MODEL_OUTPUTS = True


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def read_csv_safe(filename):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return pd.DataFrame()

    print(f"[LOAD] {path}")
    return pd.read_csv(path)


def save_csv(df, filename):
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} | shape = {df.shape}")


def parse_item_collection(x):
    """
    Parse item collection from:
    - "['21232', '21523']"
    - "frozenset({'22748', '22745'})"
    - "{'22748', '22745'}"
    - "22748, 22745"
    """
    if pd.isna(x):
        return []

    if isinstance(x, (list, set, tuple, frozenset)):
        return [str(i).strip() for i in x]

    s = str(x).strip()

    if s.startswith("frozenset(") and s.endswith(")"):
        s = s[len("frozenset("):-1]

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, set, tuple, frozenset)):
            return [str(i).strip() for i in parsed]
        return [str(parsed).strip()]
    except Exception:
        pass

    s = s.replace("{", "").replace("}", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace("'", "").replace('"', "")

    return [i.strip() for i in s.split(",") if i.strip()]


def itemset_to_string(itemset):
    return ", ".join(sorted([str(i) for i in itemset]))


def add_rule_string_columns(rules_df):
    if rules_df.empty:
        return rules_df

    rules_df = rules_df.copy()

    rules_df["antecedents_str"] = rules_df["antecedents"].apply(itemset_to_string)
    rules_df["consequents_str"] = rules_df["consequents"].apply(itemset_to_string)
    rules_df["rule"] = rules_df["antecedents_str"] + " → " + rules_df["consequents_str"]

    return rules_df


# =========================================================
# 1. CREATE ADD-TO-CART SIMULATION FILE
# =========================================================
def create_add_to_cart_simulation():
    print("\n========== CREATE add_to_cart_lift_simulation.csv ==========")

    top20 = read_csv_safe("top_20_association_rules.csv")

    if top20.empty:
        print("[SKIP] top_20_association_rules.csv not found.")
        return

    if "rule" not in top20.columns:
        if "antecedents_str" not in top20.columns:
            top20["antecedents_str"] = top20["antecedents"].apply(lambda x: ", ".join(parse_item_collection(x)))
        if "consequents_str" not in top20.columns:
            top20["consequents_str"] = top20["consequents"].apply(lambda x: ", ".join(parse_item_collection(x)))
        top20["rule"] = top20["antecedents_str"] + " → " + top20["consequents_str"]

    assumed_customers = 1000

    sim_df = top20[["rule", "confidence", "lift"]].copy()
    sim_df["assumed_customers"] = assumed_customers
    sim_df["estimated_add_to_cart"] = (
        sim_df["confidence"] * assumed_customers
    ).round(0).astype(int)

    save_csv(sim_df, "add_to_cart_lift_simulation.csv")


# =========================================================
# 2. CREATE ALGORITHM OUTPUTS
# =========================================================
def create_algorithm_outputs():
    print("\n========== CREATE ALGORITHM OUTPUTS ==========")

    basket_df = read_csv_safe("online_retail_ii_basket_df.csv")

    if basket_df.empty:
        print("[SKIP] online_retail_ii_basket_df.csv not found.")
        return

    if "Items" not in basket_df.columns:
        print("[ERROR] online_retail_ii_basket_df.csv must contain Items column.")
        return

    if "BasketSize" not in basket_df.columns:
        basket_df["BasketSize"] = basket_df["Items"].apply(lambda x: len(parse_item_collection(x)))

    # Match preprocessing logic:
    # 1. remove single-item baskets
    # 2. remove outlier basket size by 99th percentile
    baskets_for_rules = basket_df[basket_df["BasketSize"] >= 2].copy()

    basket_size_threshold = baskets_for_rules["BasketSize"].quantile(0.99)
    baskets_for_rules = baskets_for_rules[
        baskets_for_rules["BasketSize"] <= basket_size_threshold
    ].copy()

    print(f"Basket size threshold: {basket_size_threshold}")
    print(f"Baskets for rule mining: {baskets_for_rules.shape[0]}")

    transactions = baskets_for_rules["Items"].apply(parse_item_collection).tolist()

    print("Encoding transaction matrix with sparse=True...")
    te = TransactionEncoder()
    transaction_sparse = te.fit(transactions).transform(transactions, sparse=True)

    transaction_matrix = pd.DataFrame.sparse.from_spmatrix(
        transaction_sparse,
        columns=te.columns_
    )

    print("Transaction matrix shape:", transaction_matrix.shape)

    # -------------------------
    # APRIORI
    # -------------------------
    print("\nRunning Apriori...")
    start_time = time.time()

    frequent_itemsets_apriori = apriori(
        transaction_matrix,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    apriori_time = time.time() - start_time
    print("Apriori itemsets:", len(frequent_itemsets_apriori))
    print("Apriori time:", round(apriori_time, 4), "seconds")

    frequent_itemsets_apriori["itemsets_str"] = frequent_itemsets_apriori["itemsets"].apply(itemset_to_string)
    frequent_itemsets_apriori["itemset_size"] = frequent_itemsets_apriori["itemsets"].apply(len)

    save_csv(frequent_itemsets_apriori, "apriori_frequent_itemsets.csv")

    # Apriori rules
    apriori_rules = association_rules(
        frequent_itemsets_apriori,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    apriori_rules = add_rule_string_columns(apriori_rules)
    save_csv(apriori_rules, "apriori_rules.csv")

    # -------------------------
    # FP-GROWTH
    # -------------------------
    print("\nRunning FP-Growth...")
    start_time = time.time()

    frequent_itemsets_fpgrowth = fpgrowth(
        transaction_matrix,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    fpgrowth_time = time.time() - start_time
    print("FP-Growth itemsets:", len(frequent_itemsets_fpgrowth))
    print("FP-Growth time:", round(fpgrowth_time, 4), "seconds")

    frequent_itemsets_fpgrowth["itemsets_str"] = frequent_itemsets_fpgrowth["itemsets"].apply(itemset_to_string)
    frequent_itemsets_fpgrowth["itemset_size"] = frequent_itemsets_fpgrowth["itemsets"].apply(len)

    save_csv(frequent_itemsets_fpgrowth, "fpgrowth_frequent_itemsets.csv")

    # FP-Growth rules
    fpgrowth_rules = association_rules(
        frequent_itemsets_fpgrowth,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    fpgrowth_rules = add_rule_string_columns(fpgrowth_rules)
    save_csv(fpgrowth_rules, "fpgrowth_rules.csv")

    # -------------------------
    # RUNTIME SUMMARY
    # -------------------------
    algorithm_runtime_summary = pd.DataFrame({
        "Algorithm": ["Apriori", "FP-Growth"],
        "Runtime_Seconds": [round(apriori_time, 4), round(fpgrowth_time, 4)],
        "Number_of_Frequent_Itemsets": [
            len(frequent_itemsets_apriori),
            len(frequent_itemsets_fpgrowth)
        ],
        "Number_of_Rules": [
            len(apriori_rules),
            len(fpgrowth_rules)
        ],
        "Average_Lift": [
            apriori_rules["lift"].mean() if not apriori_rules.empty else np.nan,
            fpgrowth_rules["lift"].mean() if not fpgrowth_rules.empty else np.nan
        ],
        "Max_Lift": [
            apriori_rules["lift"].max() if not apriori_rules.empty else np.nan,
            fpgrowth_rules["lift"].max() if not fpgrowth_rules.empty else np.nan
        ]
    })

    save_csv(algorithm_runtime_summary, "algorithm_runtime_summary.csv")


# =========================================================
# 3. CREATE MODEL / REGRESSION SUMMARY
# =========================================================
def create_model_outputs():
    print("\n========== CREATE final_causal_impact_summary.csv ==========")

    basket_df = read_csv_safe("online_retail_ii_basket_df.csv")
    top20 = read_csv_safe("top_20_association_rules.csv")

    if basket_df.empty or top20.empty:
        print("[SKIP] Need online_retail_ii_basket_df.csv and top_20_association_rules.csv.")
        return

    required_cols = ["InvoiceNo", "Items", "BasketSize", "ProductRevenue", "TotalQuantity", "AvgUnitPrice", "Country"]

    missing_cols = [c for c in required_cols if c not in basket_df.columns]
    if missing_cols:
        print("[ERROR] Missing columns in basket_df:", missing_cols)
        return

    # Select top rule
    selected_rule = top20.sort_values(["lift", "confidence", "support"], ascending=False).iloc[0]

    antecedents = set(parse_item_collection(selected_rule["antecedents"]))
    consequents = set(parse_item_collection(selected_rule["consequents"]))
    all_rule_items = antecedents.union(consequents)

    print("Selected rule:", selected_rule.get("rule", "No rule column"))
    print("Antecedents:", antecedents)
    print("Consequents:", consequents)

    model_df = basket_df.copy()
    model_df["Items_set"] = model_df["Items"].apply(lambda x: set(parse_item_collection(x)))

    model_df["antecedent_present"] = model_df["Items_set"].apply(lambda items: antecedents.issubset(items))
    model_df["consequent_present"] = model_df["Items_set"].apply(lambda items: consequents.issubset(items))

    model_df["rule_applied"] = model_df["Items_set"].apply(
        lambda items: all_rule_items.issubset(items)
    ).astype(int)

    model_df["recommendation_candidate"] = (
        model_df["antecedent_present"] &
        ~model_df["consequent_present"]
    ).astype(int)

    model_df["AOV"] = model_df["ProductRevenue"]

    model_df = model_df[
        [
            "InvoiceNo",
            "AOV",
            "rule_applied",
            "recommendation_candidate",
            "BasketSize",
            "TotalQuantity",
            "AvgUnitPrice",
            "CustomerID",
            "Country",
            "Items_set"
        ]
    ].copy()

    # log AOV
    model_df = model_df[model_df["AOV"] > 0].copy()
    model_df["log_AOV"] = np.log(model_df["AOV"])

    # Outlier handling: 99.5% threshold
    outlier_vars = ["AOV", "BasketSize", "TotalQuantity", "AvgUnitPrice"]
    upper_limits = model_df[outlier_vars].quantile(0.995)

    model_df["is_outlier"] = False

    for col in outlier_vars:
        model_df[f"{col}_outlier"] = model_df[col] > upper_limits[col]
        model_df["is_outlier"] = model_df["is_outlier"] | model_df[f"{col}_outlier"]

    model_reg_clean = model_df[model_df["is_outlier"] == False].copy()

    print("Original regression rows:", model_df.shape[0])
    print("Cleaned regression rows:", model_reg_clean.shape[0])
    print("rule_applied count:", model_reg_clean["rule_applied"].sum())

    formulas = {
        "Model 1A - Baseline AOV": "AOV ~ rule_applied",
        "Model 2A - + BasketSize": "AOV ~ rule_applied + BasketSize",
        "Model 3A - + BasketSize + AvgUnitPrice": "AOV ~ rule_applied + BasketSize + AvgUnitPrice",
        "Model 4A - + BasketSize + AvgUnitPrice + TotalQuantity": "AOV ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity",
        "Model 5A - + BasketSize + AvgUnitPrice + TotalQuantity + Country": "AOV ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity + C(Country)",

        "Model 1B - Baseline log_AOV": "log_AOV ~ rule_applied",
        "Model 2B - log_AOV + BasketSize": "log_AOV ~ rule_applied + BasketSize",
        "Model 3B - log_AOV + BasketSize + AvgUnitPrice": "log_AOV ~ rule_applied + BasketSize + AvgUnitPrice",
        "Model 4B - log_AOV + BasketSize + AvgUnitPrice + TotalQuantity": "log_AOV ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity",
        "Model 5B - log_AOV + BasketSize + AvgUnitPrice + TotalQuantity + Country": "log_AOV ~ rule_applied + BasketSize + AvgUnitPrice + TotalQuantity + C(Country)"
    }

    rows = []

    for model_name, formula in formulas.items():
        print("Running:", model_name)

        model = smf.ols(
            formula=formula,
            data=model_reg_clean
        ).fit(cov_type="HC3")

        coef = model.params.get("rule_applied", np.nan)
        pval = model.pvalues.get("rule_applied", np.nan)

        ci = model.conf_int().loc["rule_applied"] if "rule_applied" in model.params.index else [np.nan, np.nan]

        approx_pct = np.nan
        if "log_AOV" in formula:
            approx_pct = (np.exp(coef) - 1) * 100

        if pd.notna(pval) and pval < 0.05 and coef > 0:
            interpretation = "Positive and statistically significant"
        else:
            interpretation = "Not statistically significant"

        rows.append({
            "Model": model_name,
            "Rule_Coefficient": round(coef, 4),
            "P_Value": round(pval, 4),
            "CI_Lower": round(ci[0], 4),
            "CI_Upper": round(ci[1], 4),
            "R_Squared": round(model.rsquared, 4),
            "Adj_R_Squared": round(model.rsquared_adj, 4),
            "Approx_Percentage_Effect_Log_Models": round(approx_pct, 4) if pd.notna(approx_pct) else np.nan,
            "Interpretation": interpretation
        })

    final_summary = pd.DataFrame(rows)

    save_csv(final_summary, "final_causal_impact_summary.csv")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if RUN_ADD_TO_CART_SIMULATION:
        create_add_to_cart_simulation()

    if RUN_ALGORITHM_OUTPUTS:
        create_algorithm_outputs()

    if RUN_MODEL_OUTPUTS:
        create_model_outputs()

    print("\nDONE. Missing dashboard CSV files have been created.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import ast

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
DATA_DIR = "data"

@st.cache_data
def safe_load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.sidebar.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()
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
df_rules = safe_load_csv("association_rules_strong.csv")
df_top20 = safe_load_csv("top_20_association_rules.csv")
df_baskets = safe_load_csv("online_retail_ii_basket_df.csv")
df_items = safe_load_csv("online_retail_ii_basket_items.csv")
df_lookup = safe_load_csv("product_lookup.csv")
# Load Optional Data
df_sim = safe_load_csv("add_to_cart_lift_simulation.csv")
df_model = safe_load_csv("final_causal_impact_summary.csv")
df_alg_runtime = safe_load_csv("algorithm_runtime_summary.csv")
df_apr_freq = safe_load_csv("apriori_frequent_itemsets.csv")
df_fp_freq = safe_load_csv("fpgrowth_frequent_itemsets.csv")

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
    "💡 7. Final Conclusion"
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
        
    if not df_items.empty:
        total_unique_prods = df_items['StockCode'].nunique() if 'StockCode' in df_items.columns else 0

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
        if not df_items.empty and 'Description' in df_items.columns and 'InvoiceNo' in df_items.columns:
            top_freq = df_items.groupby('Description')['InvoiceNo'].nunique().sort_values(ascending=False).head(10).reset_index()
            top_freq.columns = ['Product', 'Frequency']
            fig2 = px.bar(top_freq, x='Frequency', y='Product', orientation='h', title="Top 10 Products by Frequency", color_discrete_sequence=['#1f77b4'])
            fig2.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Item data not available.")

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
    st.header("Causal Impact / Regression Results")
    
    if df_model.empty:
        st.warning("Regression summary file not found. Model result tab is disabled.")
    else:
        st.dataframe(df_model, use_container_width=True)
        
        # Display dynamically based on what's available
        # Assuming columns might be like 'Model', 'Coefficient', 'P_Value', 'R_Squared'
        cols = df_model.columns.str.lower()
        pval_col = [c for c in df_model.columns if 'p' in c.lower() and 'val' in c.lower()]
        coef_col = [c for c in df_model.columns if 'coef' in c.lower()]
        r2_col = [c for c in df_model.columns if 'r' in c.lower() and 'sq' in c.lower()]
        
        c1, c2 = st.columns(2)
        if coef_col and 'Model' in df_model.columns:
            fig_c = px.bar(df_model, x='Model', y=coef_col[0], title="Rule Coefficient by Model", color_discrete_sequence=['#f4a460'])
            fig_c.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            c1.plotly_chart(fig_c, use_container_width=True)
            
        if r2_col and 'Model' in df_model.columns:
            fig_r = px.bar(df_model, x='Model', y=r2_col[0], title="R-Squared by Model", color_discrete_sequence=['#1f77b4'])
            fig_r.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            c2.plotly_chart(fig_r, use_container_width=True)
        
        if pval_col and "Model" in df_model.columns:
            p_col = pval_col[0]

    final_aov_model = df_model[df_model["Model"].astype(str).str.contains("Model 5A", na=False)]
    final_log_model = df_model[df_model["Model"].astype(str).str.contains("Model 5B", na=False)]

    final_aov_p = final_aov_model[p_col].iloc[0] if not final_aov_model.empty else None
    final_log_p = final_log_model[p_col].iloc[0] if not final_log_model.empty else None

    if final_aov_p is not None and final_log_p is not None:
        if final_aov_p >= 0.05 and final_log_p >= 0.05:
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
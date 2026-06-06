import pandas as pd
import os

DATA_DIR = "data"

input_path = os.path.join(DATA_DIR, "online_retail_ii_basket_items.csv")
output_path = os.path.join(DATA_DIR, "product_lookup.csv")

items = pd.read_csv(
    input_path,
    usecols=["StockCode", "Description"]
)

items["StockCode"] = items["StockCode"].astype(str).str.strip()
items["Description"] = items["Description"].astype(str).str.strip()

product_lookup = (
    items
    .dropna(subset=["StockCode", "Description"])
    .groupby("StockCode")["Description"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    .reset_index()
)

product_lookup.to_csv(output_path, index=False)

print("Saved:", output_path)
print(product_lookup.head())
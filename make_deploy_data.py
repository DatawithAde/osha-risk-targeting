import pandas as pd
import os

BASE     = r"C:\Users\ayori\Documents\Capstone Project\Near Miss"
DATA_DIR = os.path.join(BASE, "data")

print("Creating slim deployment dataset for the Data Insights tab...")

# The deployed app's Data Insights tab only needs these 6 columns
KEEP = ['naics_2digit', 'year_int', 'high_risk',
        'serious_violations', 'log_employees', 'is_comprehensive']

df = pd.read_csv(os.path.join(DATA_DIR, "processed_osha.csv"),
                 usecols=KEEP, low_memory=False)

out = os.path.join(DATA_DIR, "insights_data.csv")
df.to_csv(out, index=False)

size_mb = os.path.getsize(out) / 1024 / 1024
print(f"Saved {out}")
print(f"Rows: {len(df):,}  Columns: {list(df.columns)}")
print(f"Size: {size_mb:.1f} MB  (was ~133 MB)")
print("\nThis slim file is what gets committed to GitHub.")
print("DONE.")

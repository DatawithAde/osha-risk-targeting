import pandas as pd
import numpy as np
import os

BASE     = r"C:\Users\ayori\Documents\Capstone Project\Near Miss"
DATA_DIR = os.path.join(BASE, "data")

print("=" * 60)
print("STEP 1: LOADING INSPECTION DATA")
print("=" * 60)

insp = pd.read_csv(
    os.path.join(DATA_DIR, "osha_inspection_combined.csv"),
    low_memory=False,
    usecols=[
        'ACTIVITY_NR', 'ESTAB_NAME', 'SITE_STATE', 'NAICS_CODE',
        'SIC_CODE', 'OPEN_DATE', 'INSP_TYPE', 'INSP_SCOPE',
        'UNION_STATUS', 'NR_IN_ESTAB', 'OWNER_TYPE',
        'SAFETY_MANUF', 'SAFETY_CONST', 'SAFETY_MARIT'
    ]
)

print(f"Inspection raw shape: {insp.shape}")

# Parse date and filter 2016-2023
insp['OPEN_DATE'] = pd.to_datetime(insp['OPEN_DATE'], errors='coerce')
insp = insp[insp['OPEN_DATE'].dt.year.between(2016, 2023)]
insp['year']  = insp['OPEN_DATE'].dt.year
insp['month'] = insp['OPEN_DATE'].dt.month
print(f"Inspection after date filter (2016-2023): {insp.shape}")

# Encode union status
insp['is_union'] = (insp['UNION_STATUS'] == 'A').astype(int)

# Industry sector flags
insp['is_construction'] = (insp['SAFETY_CONST'].notna() & (insp['SAFETY_CONST'] == 'X')).astype(int)
insp['is_manufacturing'] = (insp['SAFETY_MANUF'].notna() & (insp['SAFETY_MANUF'] == 'X')).astype(int)
insp['is_maritime']     = (insp['SAFETY_MARIT'].notna() & (insp['SAFETY_MARIT'] == 'X')).astype(int)

# NAICS 2-digit sector
insp['NAICS_CODE']   = insp['NAICS_CODE'].astype(str).str.strip().str[:6]
insp['naics_2digit'] = insp['NAICS_CODE'].str[:2]

# Employee count (log-transform)
insp['NR_IN_ESTAB']   = pd.to_numeric(insp['NR_IN_ESTAB'], errors='coerce')
insp['log_employees'] = np.log1p(insp['NR_IN_ESTAB'].fillna(0))

# COVID year flag
insp['covid_year'] = (insp['year'] == 2020).astype(int)

# ── Pre-inspection signals: INSP_TYPE and INSP_SCOPE ─────────────────────────
# INSP_TYPE codes (impetus for the inspection — known before inspection begins):
#   A = Accident/Fatality/Catastrophe   B = Complaint   C = Referral
#   D = Monitoring  E = Variance  F = FollowUp  G = Unprog Related
#   H = Programmed Planned  I = Prog Related  J = Other  (M = Unprog, etc.)
# We flag the high-signal unprogrammed reactive types.
insp['INSP_TYPE'] = insp['INSP_TYPE'].astype(str).str.strip().str.upper()
insp['is_fatality_insp']  = (insp['INSP_TYPE'] == 'A').astype(int)
insp['is_complaint_insp'] = (insp['INSP_TYPE'] == 'B').astype(int)
insp['is_referral_insp']  = (insp['INSP_TYPE'] == 'C').astype(int)
insp['is_followup_insp']  = (insp['INSP_TYPE'] == 'F').astype(int)
# Unprogrammed reactive = fatality OR complaint OR referral (employer didn't choose this)
insp['is_unprogrammed']   = insp['INSP_TYPE'].isin(['A', 'B', 'C', 'G']).astype(int)

# INSP_SCOPE codes:  A = Comprehensive  B = Partial  C = Records  D = Other/NoInsp
insp['INSP_SCOPE'] = insp['INSP_SCOPE'].astype(str).str.strip().str.upper()
insp['is_comprehensive'] = (insp['INSP_SCOPE'] == 'A').astype(int)
insp['is_partial']       = (insp['INSP_SCOPE'] == 'B').astype(int)

print(f"Inspection processed shape: {insp.shape}")
print(f"INSP_TYPE distribution:\n{insp['INSP_TYPE'].value_counts().head(10)}")
print(f"INSP_SCOPE distribution:\n{insp['INSP_SCOPE'].value_counts().head(6)}\n")

# ── STEP 2: VIOLATIONS ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: LOADING VIOLATION DATA")
print("=" * 60)

viol = pd.read_csv(
    os.path.join(DATA_DIR, "osha_violation_combined.csv"),
    low_memory=False,
    usecols=['ACTIVITY_NR', 'CITATION_ID', 'HIST_PENALTY', 'HIST_VTYPE']
)
print(f"Violation raw shape: {viol.shape}")
print(f"HIST_VTYPE unique values: {viol['HIST_VTYPE'].dropna().unique()[:20]}")

# Aggregate per inspection
viol['HIST_PENALTY'] = pd.to_numeric(viol['HIST_PENALTY'], errors='coerce').fillna(0)
viol['HIST_VTYPE']   = viol['HIST_VTYPE'].astype(str).str.strip().str.upper()

viol_agg = viol.groupby('ACTIVITY_NR').agg(
    total_violations   = ('CITATION_ID',  'count'),
    serious_violations = ('HIST_VTYPE',   lambda x: (x == 'S').sum()),
    willful_violations = ('HIST_VTYPE',   lambda x: (x == 'W').sum()),
    repeat_violations  = ('HIST_VTYPE',   lambda x: (x == 'R').sum()),
    other_violations   = ('HIST_VTYPE',   lambda x: (x == 'O').sum()),
    total_penalty      = ('HIST_PENALTY', 'sum'),
    max_penalty        = ('HIST_PENALTY', 'max'),
).reset_index()

print(f"Violation aggregated shape: {viol_agg.shape}\n")

# ── STEP 3: MERGE INSPECTION + VIOLATIONS ─────────────────────────────────────
print("=" * 60)
print("STEP 3: MERGING INSPECTION + VIOLATIONS")
print("=" * 60)

df = insp.merge(viol_agg, on='ACTIVITY_NR', how='left')

# Fill NaN for establishments with no violations on record
fill_cols = ['total_violations', 'serious_violations', 'willful_violations',
             'repeat_violations', 'other_violations', 'total_penalty', 'max_penalty']
df[fill_cols] = df[fill_cols].fillna(0)

df['log_penalty']     = np.log1p(df['total_penalty'])
df['log_max_penalty'] = np.log1p(df['max_penalty'])

print(f"Merged shape: {df.shape}\n")

# ── STEP 4: ITA 300A FEATURES ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: LOADING ITA 300A DATA")
print("=" * 60)

ita_frames = []
for year in range(2016, 2024):
    fpath = os.path.join(DATA_DIR, f"ITA_300A_{year}.csv")
    if os.path.exists(fpath):
        try:
            chunk = pd.read_csv(fpath, low_memory=False)
            chunk.columns = chunk.columns.str.lower().str.strip()
            chunk['data_year'] = year
            ita_frames.append(chunk)
            print(f"  Loaded ITA {year}: {chunk.shape}")
        except Exception as e:
            print(f"  SKIPPED ITA {year}: {e}")

ita = pd.concat(ita_frames, ignore_index=True)
print(f"ITA combined shape: {ita.shape}")

# Compute TRIR from raw numbers: (cases / hours) * 200,000
ita['total_cases'] = (
    ita['total_dafw_cases'].fillna(0) +
    ita['total_djtr_cases'].fillna(0) +
    ita['total_other_cases'].fillna(0)
)
ita['total_hours_worked'] = pd.to_numeric(ita['total_hours_worked'], errors='coerce')
ita['computed_trir'] = np.where(
    ita['total_hours_worked'] > 0,
    (ita['total_cases'] / ita['total_hours_worked']) * 200000,
    np.nan
)

# DART rate: (dafw + djtr) / hours * 200,000
ita['dart_cases'] = ita['total_dafw_cases'].fillna(0) + ita['total_djtr_cases'].fillna(0)
ita['computed_dart'] = np.where(
    ita['total_hours_worked'] > 0,
    (ita['dart_cases'] / ita['total_hours_worked']) * 200000,
    np.nan
)

# Aggregate ITA to NAICS + year level
ita['naics_code'] = ita['naics_code'].astype(str).str.strip().str[:6]
ita_naics = ita.groupby(['naics_code', 'data_year']).agg(
    avg_trir         = ('computed_trir',         'mean'),
    avg_dart         = ('computed_dart',          'mean'),
    avg_deaths       = ('total_deaths',           'mean'),
    avg_employees    = ('annual_average_employees','mean'),
    total_dafw_cases = ('total_dafw_cases',        'sum'),
).reset_index().rename(columns={'data_year': 'year'})

# Cap extreme TRIR/DART outliers (data entry errors exist)
ita_naics['avg_trir'] = ita_naics['avg_trir'].clip(upper=100)
ita_naics['avg_dart'] = ita_naics['avg_dart'].clip(upper=50)

print(f"ITA aggregated by NAICS+year: {ita_naics.shape}\n")

# Join ITA to main df on NAICS + year
df['year_int'] = df['year'].astype(int)
ita_naics['year'] = ita_naics['year'].astype(int)
df = df.merge(
    ita_naics,
    left_on=['NAICS_CODE', 'year_int'],
    right_on=['naics_code', 'year'],
    how='left'
)

# Fill ITA NaN with industry medians
for col in ['avg_trir', 'avg_dart', 'avg_deaths']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

print(f"Shape after ITA join: {df.shape}\n")

# ── STEP 5: SEVERE INJURY REPORTS ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5: LOADING SEVERE INJURY REPORTS")
print("=" * 60)

sir = pd.read_csv(
    os.path.join(DATA_DIR, "severe_injury_reports.csv"),
    low_memory=False
)
sir.columns = sir.columns.str.strip()
print(f"SIR shape: {sir.shape}")

# Aggregate SIR to NAICS level (use Primary NAICS)
sir['naics_sir'] = sir['Primary NAICS'].astype(str).str.strip().str[:6]
sir['EventDate'] = pd.to_datetime(sir['EventDate'], errors='coerce')
sir['sir_year']  = sir['EventDate'].dt.year

sir_agg = sir.groupby(['naics_sir', 'sir_year']).agg(
    sir_count        = ('ID',          'count'),
    amputation_count = ('Amputation',  'sum'),
    hospitalization_count = ('Hospitalized', 'sum'),
).reset_index().rename(columns={'naics_sir': 'naics_code', 'sir_year': 'year'})

sir_agg['naics_code'] = sir_agg['naics_code'].astype(str)
sir_agg['year']       = sir_agg['year'].astype(float)

df['NAICS_CODE_str'] = df['NAICS_CODE'].astype(str)
df['year_float']     = df['year_int'].astype(float)

df = df.merge(
    sir_agg,
    left_on=['NAICS_CODE_str', 'year_float'],
    right_on=['naics_code', 'year'],
    how='left'
)
df['sir_count']             = df['sir_count'].fillna(0)
df['amputation_count']      = df['amputation_count'].fillna(0)
df['hospitalization_count'] = df['hospitalization_count'].fillna(0)

print(f"Shape after SIR join: {df.shape}\n")

# ── STEP 6: TARGET VARIABLE ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: BUILDING TARGET VARIABLE")
print("=" * 60)

# HIGH RISK (inspection-level outcome) =
#   any willful violation OR any repeat violation
#   OR serious violations in the top 10%
#   OR single max penalty >= $15,000
# SIR removed: it is NAICS-level and floods the positive class.
# NOTE: these outcome fields are used ONLY to build the label. They are
#       NOT used as model features (see train_model.py) to avoid leakage.
serious_thresh = df['serious_violations'].quantile(0.90)

df['high_risk'] = (
    (df['willful_violations'] > 0) |
    (df['repeat_violations']  > 0) |
    (df['serious_violations'] >= serious_thresh) |
    (df['max_penalty']        >= 15000)
).astype(int)

print(f"Serious-violation 90th pct threshold: {serious_thresh}")
print(f"High-risk rate: {df['high_risk'].mean():.1%}")
print(f"High-risk count: {df['high_risk'].sum():,}")
print(f"Low-risk count: {(df['high_risk']==0).sum():,}\n")

# ── STEP 7: FINAL FEATURE SET + SAVE ─────────────────────────────────────────
print("=" * 60)
print("STEP 7: SAVING PROCESSED DATA")
print("=" * 60)

KEEP_COLS = [
    # Identifiers (not used in training but useful for app)
    'ACTIVITY_NR', 'ESTAB_NAME', 'SITE_STATE', 'NAICS_CODE',
    'naics_2digit', 'year_int', 'month',
    # Features
    'log_employees', 'is_union', 'is_construction',
    'is_manufacturing', 'is_maritime', 'covid_year',
    'is_fatality_insp', 'is_complaint_insp', 'is_referral_insp',
    'is_followup_insp', 'is_unprogrammed',
    'is_comprehensive', 'is_partial',
    'total_violations', 'serious_violations', 'willful_violations',
    'repeat_violations', 'other_violations',
    'log_penalty', 'log_max_penalty',
    'avg_trir', 'avg_dart', 'avg_deaths',
    'sir_count', 'amputation_count', 'hospitalization_count',
    # Target
    'high_risk'
]

df_final = df[KEEP_COLS].copy()
df_final = df_final.dropna(subset=['high_risk'])

out_path = os.path.join(DATA_DIR, "processed_osha.csv")
df_final.to_csv(out_path, index=False)

print(f"Final dataset shape: {df_final.shape}")
print(f"Columns: {list(df_final.columns)}")
print(f"Saved → {out_path}")
print("\nDONE. Paste this output back to Claude.")

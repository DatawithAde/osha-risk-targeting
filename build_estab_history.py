import pandas as pd
import numpy as np
import os

BASE     = r"C:\Users\ayori\Documents\Capstone Project\Near Miss"
DATA_DIR = os.path.join(BASE, "data")

print("=" * 60)
print("POINT-IN-TIME ESTABLISHMENT HISTORY BUILDER")
print("=" * 60)
print("Strictly backward-looking: each inspection sees ONLY")
print("prior inspections at the same establishment. No leakage.\n")

# ── 1. Load raw inspection (key fields + date) ────────────────────────────────
print("Loading inspection data...")
insp = pd.read_csv(
    os.path.join(DATA_DIR, "osha_inspection_combined.csv"),
    low_memory=False,
    usecols=['ACTIVITY_NR', 'ESTAB_NAME', 'SITE_STATE', 'OPEN_DATE']
)
insp['OPEN_DATE'] = pd.to_datetime(insp['OPEN_DATE'], errors='coerce')
insp = insp[insp['OPEN_DATE'].dt.year.between(2016, 2023)].copy()

# Build establishment key: NAME + STATE (HOST_EST_KEY is unusable — single value)
insp['name_state'] = (
    insp['ESTAB_NAME'].astype(str).str.upper().str.strip() + '||' +
    insp['SITE_STATE'].astype(str).str.upper().str.strip()
)
print(f"Inspections loaded: {len(insp):,}")

# ── 2. Load violation aggregates (the OUTCOME of each past inspection) ─────────
print("Loading violation aggregates...")
viol = pd.read_csv(
    os.path.join(DATA_DIR, "osha_violation_combined.csv"),
    low_memory=False,
    usecols=['ACTIVITY_NR', 'CITATION_ID', 'HIST_PENALTY', 'HIST_VTYPE']
)
viol['HIST_PENALTY'] = pd.to_numeric(viol['HIST_PENALTY'], errors='coerce').fillna(0)
viol['HIST_VTYPE']   = viol['HIST_VTYPE'].astype(str).str.strip().str.upper()

viol_agg = viol.groupby('ACTIVITY_NR').agg(
    v_serious  = ('HIST_VTYPE', lambda x: (x == 'S').sum()),
    v_willful  = ('HIST_VTYPE', lambda x: (x == 'W').sum()),
    v_repeat   = ('HIST_VTYPE', lambda x: (x == 'R').sum()),
    v_penalty  = ('HIST_PENALTY', 'sum'),
).reset_index()
viol_agg['v_willful_repeat'] = viol_agg['v_willful'] + viol_agg['v_repeat']

# Attach each inspection's own outcome (used to build OTHERS' history, never its own)
insp = insp.merge(viol_agg, on='ACTIVITY_NR', how='left')
for c in ['v_serious', 'v_willful', 'v_repeat', 'v_penalty', 'v_willful_repeat']:
    insp[c] = insp[c].fillna(0)

# ── 3. Sort by establishment, then by date (ties broken by ACTIVITY_NR) ───────
print("Sorting by establishment + date...")
insp = insp.sort_values(['name_state', 'OPEN_DATE', 'ACTIVITY_NR']).reset_index(drop=True)

grp = insp.groupby('name_state', sort=False)

# ── 4. Backward-only cumulative features via shift() ──────────────────────────
# cumsum() then shift(1) => running total EXCLUDING the current row.
print("Computing point-in-time history (shift excludes current row)...")

insp['prior_inspections'] = grp.cumcount()  # 0 for first ever, 1 for second, ...

insp['prior_serious_total'] = (
    grp['v_serious'].cumsum() - insp['v_serious']
)
insp['prior_willful_repeat'] = (
    grp['v_willful_repeat'].cumsum() - insp['v_willful_repeat']
)
insp['prior_penalty_total'] = (
    grp['v_penalty'].cumsum() - insp['v_penalty']
)

# Days since previous inspection at this establishment
insp['prev_open_date'] = grp['OPEN_DATE'].shift(1)
insp['days_since_last_insp'] = (
    insp['OPEN_DATE'] - insp['prev_open_date']
).dt.days

# Binary: has this establishment been inspected before?
insp['had_prior_inspection'] = (insp['prior_inspections'] > 0).astype(int)

# First-ever inspections: no prior info. Fill sensibly.
# days_since_last_insp: use -1 sentinel (model learns "no prior")
insp['days_since_last_insp'] = insp['days_since_last_insp'].fillna(-1)
insp['log_days_since_last'] = np.where(
    insp['days_since_last_insp'] >= 0,
    np.log1p(insp['days_since_last_insp']),
    -1
)
insp['log_prior_penalty'] = np.log1p(insp['prior_penalty_total'].clip(lower=0))

# ── 5. Sanity checks ──────────────────────────────────────────────────────────
print("\n── Sanity checks ──")
print(f"First-ever inspections (prior=0): {(insp['prior_inspections']==0).sum():,} "
      f"({(insp['prior_inspections']==0).mean()*100:.1f}%)")
print(f"With prior history (prior>=1)   : {(insp['prior_inspections']>=1).sum():,} "
      f"({(insp['prior_inspections']>=1).mean()*100:.1f}%)")
print(f"Max prior_inspections           : {insp['prior_inspections'].max():,}")
print(f"Mean prior_serious_total        : {insp['prior_serious_total'].mean():.2f}")
print(f"Max prior_serious_total         : {insp['prior_serious_total'].max():,.0f}")

# Leakage guard: a first-ever inspection MUST have zero prior everything
first = insp[insp['prior_inspections'] == 0]
assert (first['prior_serious_total'] == 0).all(),   "LEAK: prior_serious on first insp"
assert (first['prior_willful_repeat'] == 0).all(),  "LEAK: prior_wr on first insp"
assert (first['prior_penalty_total'] == 0).all(),   "LEAK: prior_penalty on first insp"
print(">>> Leakage guard PASSED: first-ever inspections carry zero prior history.\n")

# ── 6. Save history keyed by ACTIVITY_NR ──────────────────────────────────────
hist_cols = [
    'ACTIVITY_NR',
    'prior_inspections',
    'prior_serious_total',
    'prior_willful_repeat',
    'log_prior_penalty',
    'log_days_since_last',
    'had_prior_inspection',
]
history = insp[hist_cols].copy()
out_hist = os.path.join(DATA_DIR, "estab_history.csv")
history.to_csv(out_hist, index=False)
print(f"Saved history → {out_hist}  shape={history.shape}")

# ── 7. Merge into processed_osha.csv ──────────────────────────────────────────
print("\nMerging history into processed_osha.csv...")
proc_path = os.path.join(DATA_DIR, "processed_osha.csv")
df = pd.read_csv(proc_path, low_memory=False)

# Drop any old history cols if rerun
df = df.drop(columns=[c for c in hist_cols if c != 'ACTIVITY_NR'], errors='ignore')

df = df.merge(history, on='ACTIVITY_NR', how='left')
# Any unmatched (shouldn't happen) -> treat as no history
for c in hist_cols:
    if c == 'ACTIVITY_NR':
        continue
    if c == 'log_days_since_last':
        df[c] = df[c].fillna(-1)
    else:
        df[c] = df[c].fillna(0)

df.to_csv(proc_path, index=False)
print(f"Final shape: {df.shape}")
print(f"New columns added: {[c for c in hist_cols if c!='ACTIVITY_NR']}")
print(f"Saved → {proc_path}")
print("\nDONE. Paste this output back to Claude.")

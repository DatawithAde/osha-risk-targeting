import pandas as pd
import glob
import os
import shutil

BASE      = r"C:\Users\ayori\Documents\Capstone Project\Near Miss"
DATA_OUT  = os.path.join(BASE, "data")
os.makedirs(DATA_OUT, exist_ok=True)

# ── 1. Combine Inspection chunks ──────────────────────────────────────────────
print("=" * 60)
print("COMBINING INSPECTION CHUNKS")
print("=" * 60)

insp_files = sorted(glob.glob(os.path.join(BASE, "OSHA_Inspection", "*.csv")))
print(f"Found {len(insp_files)} inspection files\n")

insp_frames = []
for i, f in enumerate(insp_files):
    try:
        chunk = pd.read_csv(f, low_memory=False)
        insp_frames.append(chunk)
        print(f"  [{i+1}/{len(insp_files)}] {os.path.basename(f)}  shape={chunk.shape}")
    except Exception as e:
        print(f"  [{i+1}/{len(insp_files)}] SKIPPED {os.path.basename(f)} — {e}")

insp_combined = pd.concat(insp_frames, ignore_index=True)
out_insp = os.path.join(DATA_OUT, "osha_inspection_combined.csv")
insp_combined.to_csv(out_insp, index=False)
print(f"\nInspection combined shape : {insp_combined.shape}")
print(f"Columns: {list(insp_combined.columns)}")
print(f"Saved → {out_insp}\n")

# ── 2. Combine Violation chunks ───────────────────────────────────────────────
print("=" * 60)
print("COMBINING VIOLATION CHUNKS")
print("=" * 60)

viol_files = sorted(glob.glob(os.path.join(BASE, "OSHA_Violation", "*.csv")))
print(f"Found {len(viol_files)} violation files\n")

viol_frames = []
for i, f in enumerate(viol_files):
    try:
        chunk = pd.read_csv(f, low_memory=False)
        viol_frames.append(chunk)
        print(f"  [{i+1}/{len(viol_files)}] {os.path.basename(f)}  shape={chunk.shape}")
    except Exception as e:
        print(f"  [{i+1}/{len(viol_files)}] SKIPPED {os.path.basename(f)} — {e}")

viol_combined = pd.concat(viol_frames, ignore_index=True)
out_viol = os.path.join(DATA_OUT, "osha_violation_combined.csv")
viol_combined.to_csv(out_viol, index=False)
print(f"\nViolation combined shape  : {viol_combined.shape}")
print(f"Columns: {list(viol_combined.columns)}")
print(f"Saved → {out_viol}\n")

# ── 3. Copy + rename ITA files into data/ ────────────────────────────────────
print("=" * 60)
print("COPYING ITA FILES TO data/ FOLDER")
print("=" * 60)

ita_map = {
    "ITA Data CY 2016.csv": "ITA_300A_2016.csv",
    "ITA Data CY 2017.csv": "ITA_300A_2017.csv",
    "ITA Data CY 2018.csv": "ITA_300A_2018.csv",
    "ITA Data CY 2019.csv": "ITA_300A_2019.csv",
    "ITA Data CY 2020.csv": "ITA_300A_2020.csv",
    "ITA Data CY 2021.csv": "ITA_300A_2021.csv",
    "ITA Data CY 2022.csv": "ITA_300A_2022.csv",
    "ITA Data CY 2023.csv": "ITA_300A_2023.csv",
}

for src_name, dst_name in ita_map.items():
    src = os.path.join(BASE, src_name)
    dst = os.path.join(DATA_OUT, dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        df_check = pd.read_csv(dst, low_memory=False, nrows=2)
        print(f"  Copied {src_name} → {dst_name}")
        print(f"    Columns: {list(df_check.columns)}\n")
    else:
        print(f"  NOT FOUND: {src_name} — check filename exactly\n")

# ── 4. Copy severe injury reports ─────────────────────────────────────────────
print("=" * 60)
print("COPYING SEVERE INJURY REPORTS")
print("=" * 60)

sir_src = os.path.join(BASE, "severe_injury_reports.csv")
sir_dst = os.path.join(DATA_OUT, "severe_injury_reports.csv")
if os.path.exists(sir_src):
    shutil.copy2(sir_src, sir_dst)
    sir_check = pd.read_csv(sir_dst, low_memory=False, nrows=2)
    print(f"  Copied severe_injury_reports.csv")
    print(f"  Columns: {list(sir_check.columns)}\n")
else:
    print("  NOT FOUND: severe_injury_reports.csv — check filename exactly\n")

# ── 5. Final summary ──────────────────────────────────────────────────────────
print("=" * 60)
print("FINAL data/ FOLDER CONTENTS")
print("=" * 60)
for f in sorted(os.listdir(DATA_OUT)):
    size_mb = os.path.getsize(os.path.join(DATA_OUT, f)) / 1024 / 1024
    print(f"  {f:50s}  {size_mb:.1f} MB")

print("\nDONE. Paste this entire output back to Claude.")

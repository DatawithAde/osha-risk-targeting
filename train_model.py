import pandas as pd
import numpy as np
import pickle
import os
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve
)

BASE     = r"C:\Users\ayori\Documents\Capstone Project\Near Miss"
DATA_DIR = os.path.join(BASE, "data")
ART_DIR  = os.path.join(BASE, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)
df = pd.read_csv(os.path.join(DATA_DIR, "processed_osha.csv"), low_memory=False)
print(f"Loaded shape: {df.shape}")
print(f"High-risk rate: {df['high_risk'].mean():.1%}\n")

# ── STEP 2: FEATURE SET (OPTION A — PRE-INSPECTION ONLY, NO LEAKAGE) ─────────
print("=" * 60)
print("STEP 2: BUILDING LEAKAGE-FREE FEATURE SET")
print("=" * 60)

# Features knowable BEFORE the inspection outcome.
# Deliberately EXCLUDED (these build the label = leakage):
#   total_violations, serious_violations, willful_violations,
#   repeat_violations, other_violations, log_penalty, log_max_penalty
NUMERIC_FEATURES = [
    'log_employees',
    'is_union',
    'is_construction',
    'is_manufacturing',
    'is_maritime',
    'covid_year',
    'is_fatality_insp',
    'is_complaint_insp',
    'is_referral_insp',
    'is_followup_insp',
    'is_unprogrammed',
    'is_comprehensive',
    'is_partial',
    'avg_trir',
    'avg_dart',
    'avg_deaths',
    'sir_count',
    'amputation_count',
    'hospitalization_count',
    'month',
    # Point-in-time establishment history (backward-looking, leakage-guarded)
    'prior_inspections',
    'prior_serious_total',
    'prior_willful_repeat',
    'log_prior_penalty',
    'log_days_since_last',
    'had_prior_inspection',
]

# One-hot encode NAICS 2-digit sector (no fake ordinal ranking)
df['naics_2digit'] = df['naics_2digit'].fillna('00').astype(str).str.zfill(2)
naics_dummies = pd.get_dummies(df['naics_2digit'], prefix='naics')
print(f"NAICS one-hot columns: {naics_dummies.shape[1]} sectors")

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X = pd.concat([df[NUMERIC_FEATURES], naics_dummies], axis=1)
y = df['high_risk']
FEATURES = list(X.columns)

print(f"Total features: {len(FEATURES)} (no violation/penalty columns)")
print(f"Feature matrix: {X.shape}\n")

# ── STEP 3: TEMPORAL SPLIT (train 2016-2021, test 2022-2023) ─────────────────
print("=" * 60)
print("STEP 3: TEMPORAL SPLIT")
print("=" * 60)

train_mask = df['year_int'] <= 2021
test_mask  = df['year_int'] >= 2022

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train (2016-2021): {X_train.shape}  high-risk={y_train.mean():.1%}")
print(f"Test  (2022-2023): {X_test.shape}  high-risk={y_test.mean():.1%}")
print("Temporal split proves the model predicts FORWARD in time.\n")

# ── STEP 4: TRAIN ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: TRAINING XGBOOST")
print("=" * 60)

SPW = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {SPW:.2f}\n")

model = XGBClassifier(
    n_estimators          = 400,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    gamma                 = 1,
    scale_pos_weight      = SPW,
    eval_metric           = 'aucpr',   # PR-AUC: right metric for imbalanced data
    early_stopping_rounds = 25,
    random_state          = 42,
    n_jobs                = -1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
print(f"\nBest iteration: {model.best_iteration}")

# ── STEP 5: EVALUATE ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: EVALUATION (pre-inspection targeting model)")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

roc_auc  = roc_auc_score(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)
baseline = y_test.mean()  # PR-AUC baseline = positive rate
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"PR-AUC (AvgPrec) : {avg_prec:.4f}   (random baseline = {baseline:.4f})")
print(f"Lift over base   : {avg_prec / baseline:.2f}x")

# ── STEP 6: PLOTS ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: SAVING PLOTS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Low Risk', 'High Risk']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Pre-Inspection Risk Targeting')
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ── STEP 7: SHAP ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: SHAP")
print("=" * 60)

shap_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
print(f"Computing SHAP on {len(shap_sample):,} rows...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(shap_sample)

fig3, ax3 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES, show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "shap_summary.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")

fig4, ax4 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES,
                  plot_type='bar', show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "shap_bar.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_bar.png")

# ── STEP 8: SAVE ARTIFACTS ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: SAVING ARTIFACTS")
print("=" * 60)

with open(os.path.join(ART_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)
np.save(os.path.join(ART_DIR, "shap_values.npy"), shap_values)
shap_sample.to_csv(os.path.join(ART_DIR, "X_test_sample.csv"), index=False)

with open(os.path.join(ART_DIR, "feature_names.json"), "w") as f:
    json.dump(FEATURES, f, indent=2)

metrics = {
    'model_type': 'Pre-inspection risk targeting (Option A, leakage-free)',
    'split': 'Temporal: train 2016-2021, test 2022-2023',
    'roc_auc':  round(roc_auc, 4),
    'pr_auc':   round(avg_prec, 4),
    'pr_baseline': round(float(baseline), 4),
    'lift': round(float(avg_prec / baseline), 2),
    'n_train': int(len(X_train)),
    'n_test':  int(len(X_test)),
    'n_features': len(FEATURES),
    'high_risk_rate': round(float(y.mean()), 4),
    'best_iteration': int(model.best_iteration),
    'scale_pos_weight': round(float(SPW), 2),
}
with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved: model.pkl, shap_values.npy, X_test_sample.csv, feature_names.json, metrics.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {avg_prec:.4f}  ({avg_prec/baseline:.2f}x over baseline)")
print("=" * 60)
print("\nDONE. Paste this output back to Claude.")

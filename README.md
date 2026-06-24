# OSHA Pre-Inspection Risk Targeting Model
Live Demo https://osha-risk-targeting-br9dy9sglrhnwdmtpqpm9g.streamlit.app/

An XGBoost classifier that predicts which establishments are likely to have
**serious, willful, or repeat OSHA violations** — using only information knowable
*before* an inspection takes place. Built as a decision-support tool for
risk-based inspection targeting, mirroring OSHA's real Site-Specific Targeting program.

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.70 |
| PR-AUC | 0.36 (random baseline 0.18) |
| Lift over random targeting | **1.99×** |
| High-risk recall | 68% |

Trained on **585,200 inspections (2016–2023)** with a **temporal split**
(train ≤2021, test ≥2022) to prove forward prediction.

## Why this is methodologically sound

- **No target leakage.** Violation counts and penalties define the training label
  but are never used as features. The model predicts outcomes it cannot see.
- **Point-in-time establishment history.** Prior-inspection features are computed
  strictly backward in time, with an assertion guard ensuring first-ever
  inspections carry zero history. No future information leaks into any prediction.
- **Honest metrics.** PR-AUC and lift are reported alongside ROC-AUC because
  accuracy is misleading on imbalanced data (19% positive class).
- **SHAP explainability.** Every top feature aligns with real EHS enforcement
  logic: establishment size, inspection scope, sector, trigger type, and
  prior serious-violation history.

## Data sources (all public)

- OSHA Enforcement: inspections + violations (enforcedata.dol.gov)
- OSHA ITA 300A injury/illness summary data, 2016–2023 (osha.gov)
- OSHA Severe Injury Reports (osha.gov)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline

1. `combine_chunks.py` — merge raw OSHA data chunks
2. `data_prep.py` — feature engineering + target variable
3. `build_estab_history.py` — point-in-time history features
4. `train_model.py` — XGBoost training, SHAP, artifacts
5. `app.py` — Streamlit dashboard

## Disclaimer

Research and portfolio demonstration only. Not an official OSHA tool and not
affiliated with the U.S. Department of Labor.

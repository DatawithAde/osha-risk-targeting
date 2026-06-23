"""
OSHA Pre-Inspection Risk Targeting Dashboard
XGBoost classifier + SHAP explainability
Predicts high-risk establishments BEFORE inspection using leakage-free features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ART_DIR  = "artifacts"
DATA_DIR = "data"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OSHA Risk Targeting",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(ART_DIR, "model.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    with open(os.path.join(ART_DIR, "feature_names.json")) as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    with open(os.path.join(ART_DIR, "metrics.json")) as f:
        return json.load(f)

@st.cache_data
def load_sample():
    return pd.read_csv(os.path.join(ART_DIR, "X_test_sample.csv"))

model     = load_model()
FEATURES  = load_features()
METRICS   = load_metrics()

# NAICS sector reference
NAICS_SECTORS = {
    '11': 'Agriculture, Forestry, Fishing',
    '21': 'Mining, Quarrying, Oil & Gas',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing (Food, Textile, Apparel)',
    '32': 'Manufacturing (Wood, Paper, Chemical)',
    '33': 'Manufacturing (Metal, Machinery, Transport)',
    '42': 'Wholesale Trade',
    '44': 'Retail Trade',
    '45': 'Retail Trade',
    '48': 'Transportation & Warehousing',
    '49': 'Transportation & Warehousing',
    '51': 'Information',
    '52': 'Finance & Insurance',
    '53': 'Real Estate',
    '54': 'Professional & Technical Services',
    '56': 'Administrative & Waste Services',
    '61': 'Educational Services',
    '62': 'Health Care & Social Assistance',
    '71': 'Arts, Entertainment, Recreation',
    '72': 'Accommodation & Food Services',
    '81': 'Other Services',
    '92': 'Public Administration',
}

def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Construct a single-row DataFrame matching FEATURES exactly, zeros elsewhere."""
    row = {f: 0 for f in FEATURES}
    for k, v in inputs.items():
        if k in row:
            row[k] = v
    # Set the one-hot NAICS column
    naics_col = f"naics_{inputs['_naics2']}"
    if naics_col in row:
        row[naics_col] = 1
    return pd.DataFrame([row])[FEATURES]

def risk_band(prob):
    if prob >= 0.60: return "HIGH RISK", "#c0392b"
    if prob >= 0.35: return "MODERATE RISK", "#e67e22"
    return "LOWER RISK", "#27ae60"

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("OSHA Pre-Inspection Risk Targeting")
st.markdown(
    "Predicts which establishments are likely to have **serious, willful, or repeat "
    "violations** — using only information knowable *before* an inspection. "
    "Built on 585,200 OSHA inspections (2016–2023)."
)

# Top metric strip
m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
m2.metric("PR-AUC", f"{METRICS['pr_auc']:.3f}", help="Precision-Recall AUC (imbalanced-data metric)")
m3.metric("Lift over random", f"{METRICS['lift']:.2f}×",
          help="How much better than random inspection targeting")
m4.metric("High-risk base rate", f"{METRICS['high_risk_rate']*100:.0f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk Predictor", "📂 Batch Scoring", "🔍 Explainability", "📊 Data Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MANUAL RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Score a single establishment")
    st.caption("Set the characteristics OSHA would know before scheduling an inspection.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Establishment**")
        naics2 = st.selectbox(
            "Industry sector (NAICS)",
            options=list(dict.fromkeys(NAICS_SECTORS.keys())),
            index=list(dict.fromkeys(NAICS_SECTORS.keys())).index('23'),
            format_func=lambda x: f"{x} — {NAICS_SECTORS.get(x, 'Other')}"
        )
        n_emp = st.number_input("Number of employees", 1, 50000, 75)
        union = st.checkbox("Union workplace")

    with c2:
        st.markdown("**Inspection context**")
        scope = st.radio("Inspection scope", ["Comprehensive", "Partial"], horizontal=True)
        itype = st.selectbox("Inspection trigger", [
            "Programmed (planned)", "Complaint", "Referral",
            "Fatality / Catastrophe", "Follow-up"
        ])
        avg_trir = st.slider("Industry avg TRIR", 0.0, 20.0, 4.0, 0.1,
                             help="Total Recordable Incident Rate for this sector")

    with c3:
        st.markdown("**Establishment history**")
        had_prior = st.checkbox("Has been inspected before", value=True)
        prior_n = st.number_input("Prior inspections", 0, 360, 2,
                                  disabled=not had_prior)
        prior_serious = st.number_input("Prior serious violations", 0, 410, 1,
                                        disabled=not had_prior)
        prior_wr = st.number_input("Prior willful/repeat violations", 0, 100, 0,
                                   disabled=not had_prior)
        prior_pen = st.number_input("Prior total penalties ($)", 0, 1000000, 5000,
                                    disabled=not had_prior)

    if st.button("Predict risk", type="primary"):
        inputs = {
            '_naics2': naics2,
            'log_employees': np.log1p(n_emp),
            'is_union': int(union),
            'is_construction': int(naics2 == '23'),
            'is_manufacturing': int(naics2 in ['31', '32', '33']),
            'is_maritime': 0,
            'covid_year': 0,
            'is_comprehensive': int(scope == "Comprehensive"),
            'is_partial': int(scope == "Partial"),
            'is_complaint_insp': int(itype == "Complaint"),
            'is_referral_insp': int(itype == "Referral"),
            'is_fatality_insp': int(itype == "Fatality / Catastrophe"),
            'is_followup_insp': int(itype == "Follow-up"),
            'is_unprogrammed': int(itype in ["Complaint", "Referral", "Fatality / Catastrophe"]),
            'avg_trir': avg_trir,
            'avg_dart': avg_trir * 0.5,
            'avg_deaths': 0.003,
            'month': 6,
            'prior_inspections': prior_n if had_prior else 0,
            'prior_serious_total': prior_serious if had_prior else 0,
            'prior_willful_repeat': prior_wr if had_prior else 0,
            'log_prior_penalty': np.log1p(prior_pen) if had_prior else 0,
            'log_days_since_last': np.log1p(365) if had_prior else -1,
            'had_prior_inspection': int(had_prior),
        }
        X_one = build_feature_row(inputs)
        prob = float(model.predict_proba(X_one)[0][1])
        label, color = risk_band(prob)

        st.divider()
        g1, g2 = st.columns([1, 1.4])

        with g1:
            st.markdown(f"<h2 style='color:{color};margin-bottom:0'>{label}</h2>",
                        unsafe_allow_html=True)
            st.metric("Predicted probability of serious/willful/repeat violation",
                      f"{prob:.1%}")

        with g2:
            fig = plt.figure(figsize=(5, 2.6))
            ax = fig.add_subplot(111)
            # Horizontal risk bar with zones
            ax.barh([0], [100], color='#eee', height=0.5)
            ax.barh([0], [35], color='#eafaf1', height=0.5)
            ax.barh([0], [25], left=35, color='#fef5e7', height=0.5)
            ax.barh([0], [40], left=60, color='#fdedec', height=0.5)
            # Marker for the predicted value
            ax.axvline(prob * 100, color=color, linewidth=4)
            ax.text(prob * 100, 0.45, f"{prob:.0%}", ha='center',
                    fontsize=16, fontweight='bold', color=color)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.7)
            ax.set_yticks([])
            ax.set_xticks([0, 35, 60, 100])
            ax.set_xlabel("Risk probability (%)")
            for s in ['top', 'right', 'left']:
                ax.spines[s].set_visible(False)
            st.pyplot(fig, clear_figure=True)
            plt.close('all')

        # Per-prediction SHAP waterfall
        st.markdown("**Why this score?**")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_one)
        expl = shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value,
            data=X_one.iloc[0],
            feature_names=FEATURES
        )
        fig_w, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(expl, max_display=12, show=False)
        st.pyplot(fig_w, clear_figure=True)
        plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH SCORING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Score many establishments at once")
    st.caption("Upload a CSV with the model's feature columns to get a ranked risk list. "
               "Use the sample below to see the expected format.")

    sample = load_sample()
    st.download_button(
        "Download sample input (held-out 2022–2023 establishments)",
        data=sample.to_csv(index=False).encode(),
        file_name="sample_establishments.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload establishments CSV", type=["csv"])

    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            # Align columns to FEATURES (fill missing with 0, drop extras)
            for f in FEATURES:
                if f not in batch.columns:
                    batch[f] = 0
            X_batch = batch[FEATURES].fillna(0)

            probs = model.predict_proba(X_batch)[:, 1]
            out = batch.copy()
            out['risk_probability'] = probs
            out['risk_band'] = pd.cut(
                probs, bins=[-0.01, 0.35, 0.60, 1.01],
                labels=['Lower', 'Moderate', 'High']
            )
            out = out.sort_values('risk_probability', ascending=False).reset_index(drop=True)

            st.success(f"Scored {len(out):,} establishments.")

            b1, b2, b3 = st.columns(3)
            b1.metric("High risk", f"{(out['risk_band']=='High').sum():,}")
            b2.metric("Moderate", f"{(out['risk_band']=='Moderate').sum():,}")
            b3.metric("Lower", f"{(out['risk_band']=='Lower').sum():,}")

            display_cols = ['risk_probability', 'risk_band'] + \
                           [c for c in ['prior_serious_total','prior_inspections',
                                        'log_employees','is_comprehensive'] if c in out.columns]
            st.dataframe(
                out[display_cols].head(100).style.format({'risk_probability': '{:.1%}'}),
                use_container_width=True, height=400
            )

            st.download_button(
                "Download full ranked results",
                data=out.to_csv(index=False).encode(),
                file_name="scored_establishments.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Could not score file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How the model makes decisions")

    e1, e2 = st.columns([1, 1])
    with e1:
        st.markdown("**Methodology**")
        st.markdown(f"""
        - **Model:** {METRICS.get('model_type', 'XGBoost classifier')}
        - **Validation:** {METRICS.get('split', 'Temporal split')}
        - **Training rows:** {METRICS['n_train']:,}
        - **Test rows:** {METRICS['n_test']:,}
        - **Features:** {METRICS['n_features']} (no violation/penalty leakage)
        - **Class imbalance handling:** scale_pos_weight = {METRICS['scale_pos_weight']}
        """)
        st.info("Violation and penalty outcomes are used **only** to build the "
                "training label — never as model inputs. Establishment history is "
                "computed strictly from prior inspections, point-in-time, so no "
                "future information leaks into any prediction.")

    with e2:
        st.markdown("**Performance**")
        st.markdown(f"""
        - **ROC-AUC:** {METRICS['roc_auc']:.3f}
        - **PR-AUC:** {METRICS['pr_auc']:.3f} (random baseline {METRICS['pr_baseline']:.3f})
        - **Lift over random targeting:** {METRICS['lift']:.2f}×
        """)
        st.caption("On imbalanced data, PR-AUC and lift are more honest than raw "
                   "accuracy. A model predicting 'high risk' for everyone would score "
                   "81% accuracy while being useless — these metrics avoid that trap.")

    st.divider()
    st.markdown("**Global feature importance (SHAP)**")
    shap_img = os.path.join(ART_DIR, "shap_summary.png")
    if os.path.exists(shap_img):
        st.image(shap_img, use_column_width=True)
        st.caption("Each dot is an establishment. Red = high feature value, blue = low. "
                   "Position shows how that feature pushed the prediction toward (right) "
                   "or away from (left) high-risk.")
    else:
        st.warning("SHAP summary image not found. Run train_model.py to generate it.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("OSHA enforcement patterns in the data")

    proc_path = os.path.join(DATA_DIR, "insights_data.csv")
    if not os.path.exists(proc_path):
        # Fall back to the full processed file if slim version not present
        proc_path = os.path.join(DATA_DIR, "processed_osha.csv")
    if not os.path.exists(proc_path):
        st.warning("insights_data.csv not found in data/. This tab needs it for charts.")
    else:
        @st.cache_data
        def load_insights():
            cols = ['naics_2digit', 'year_int', 'high_risk', 'serious_violations',
                    'log_employees', 'is_comprehensive']
            d = pd.read_csv(proc_path, usecols=lambda c: c in cols, low_memory=False)
            return d

        d = load_insights()

        i1, i2 = st.columns(2)
        with i1:
            st.markdown("**High-risk rate by sector**")
            sector = d.groupby('naics_2digit')['high_risk'].mean().reset_index()
            sector['Sector'] = sector['naics_2digit'].astype(str).str.zfill(2).map(
                lambda x: f"{x} {NAICS_SECTORS.get(x, '')[:20]}")
            sector = sector.sort_values('high_risk', ascending=False).head(12)
            sector = sector.set_index('Sector')[['high_risk']].rename(
                columns={'high_risk': 'High-risk rate'})
            st.bar_chart(sector, horizontal=True, color='#c0392b', height=420)

        with i2:
            st.markdown("**High-risk rate over time**")
            yearly = d.groupby('year_int')['high_risk'].mean().reset_index()
            yearly = yearly.set_index('year_int').rename(
                columns={'high_risk': 'High-risk rate'})
            st.line_chart(yearly, color='#c0392b', height=420)

        st.caption(f"Based on {len(d):,} inspections, 2016–2023. Heavy manufacturing "
                   "(NAICS 31–33) and construction (23) consistently show the highest "
                   "high-risk rates — matching known OSHA enforcement priorities.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data: OSHA Enforcement (inspections + violations), ITA 300A injury data, "
           "Severe Injury Reports — all public via dol.gov & osha.gov. "
           "Model for research/portfolio demonstration; not an official OSHA tool.")

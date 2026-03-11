import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Pediatric BMT Outcome Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #0a2342 0%, #1a4a7a 100%);
    padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
    border-left: 5px solid #00b4d8;
}
.main-header h1 { color: #fff; font-size: 1.9rem; font-weight: 600; margin: 0 0 0.3rem 0; }
.main-header p { color: #90caf9; font-size: 0.9rem; margin: 0 0 0.7rem 0; }
.badge {
    display: inline-block; background: rgba(0,180,216,0.15); color: #00b4d8;
    border: 1px solid #00b4d8; padding: 0.15rem 0.6rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500; margin-right: 0.4rem;
}
.info-card {
    background: #f8faff; border: 1px solid #e3eaf5; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem; border-left: 4px solid #1a4a7a;
}
.info-card h4 { color: #0a2342; font-size: 0.8rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 0.4rem 0; }
.info-card p { color: #444; font-size: 0.88rem; margin: 0; line-height: 1.6; }
.metric-card {
    background: white; border: 1px solid #e3eaf5; border-radius: 10px;
    padding: 0.9rem; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 0.7rem;
}
.metric-card .value { font-size: 1.7rem; font-weight: 600; color: #1a4a7a; font-family: 'IBM Plex Mono', monospace; }
.metric-card .label { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
.result-success { background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border: 2px solid #4caf50; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
.result-risk { background: linear-gradient(135deg, #fce4ec, #fff3e0); border: 2px solid #f44336; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
.result-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.4rem; }
.result-prob { font-size: 2.8rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.section-title { color: #0a2342; font-size: 1rem; font-weight: 600;
    border-bottom: 2px solid #00b4d8; padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem 0; }
.tab-header { font-size: 0.78rem; font-weight: 600; color: #0a2342;
    text-transform: uppercase; letter-spacing: 0.5px; margin: 0.8rem 0 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Feature name mapping (technical → clinical) ───────────────────────────
FEATURE_LABELS = {
    "Recipientgender": "Recipient Gender (0=Female, 1=Male)",
    "Stemcellsource": "Stem Cell Source (0=Peripheral, 1=Bone Marrow)",
    "Donorage": "Donor Age (years)",
    "Donorage35": "Donor Age > 35 (0=No, 1=Yes)",
    "IIIV": "CMV Serostatus Donor/Recipient",
    "Gendermatch": "Donor-Recipient Gender Match",
    "DonorABO": "Donor ABO Blood Group",
    "RecipientABO": "Recipient ABO Blood Group",
    "RecipientRh": "Recipient Rh Factor",
    "ABOmatch": "ABO Compatibility",
    "CMVstatus": "CMV Status",
    "DonorCMV": "Donor CMV Status",
    "RecipientCMV": "Recipient CMV Status",
    "Disease": "Underlying Disease",
    "Riskgroup": "Disease Risk Group",
    "Txpostrelapse": "Transplant Post-Relapse",
    "Diseasegroup": "Disease Group",
    "HLAmatch": "HLA Match Level",
    "HLAmismatch": "HLA Mismatch",
    "Antigen": "Antigen Mismatch",
    "Alel": "Allele Mismatch",
    "HLAgrI": "HLA Group I",
    "Recipientage": "Recipient Age (years)",
    "Recipientage10": "Recipient Age > 10 (0=No, 1=Yes)",
    "Recipientageint": "Recipient Age Interval",
    "Relapse": "Disease Relapse (0=No, 1=Yes)",
    "aGvHDIIIIV": "Acute GvHD Grade III-IV",
    "extcGvHD": "Extensive Chronic GvHD",
    "CD34kgx10d6": "CD34+ Cells (×10⁶/kg)",
    "CD3dkgx10d8": "CD3+ Cells (×10⁸/kg)",
    "CD3dCD34": "CD3/CD34 Cell Ratio",
    "Rbodymass": "Recipient Body Mass (kg)",
    "ANCrecovery": "ANC Recovery Day",
    "PLTrecovery": "Platelet Recovery Day",
    "time_to_aGvHD_III_IV": "Time to Acute GvHD Grade III-IV (days)",
    "survival_time": "Survival Time (days)",
    "survival_status": "Survival Status",
}

# ── Feature groups for tabs ───────────────────────────────────────────────
DONOR_FEATURES = [
    "Donorage", "Donorage35", "DonorABO", "DonorCMV",
    "Stemcellsource", "CD34kgx10d6", "CD3dkgx10d8", "CD3dCD34"
]
RECIPIENT_FEATURES = [
    "Recipientgender", "Recipientage", "Recipientage10", "Recipientageint",
    "RecipientABO", "RecipientRh", "RecipientCMV", "Rbodymass"
]
CLINICAL_FEATURES = [
    "Disease", "Riskgroup", "Diseasegroup", "Txpostrelapse",
    "HLAmatch", "HLAmismatch", "Antigen", "Alel", "HLAgrI",
    "IIIV", "Gendermatch", "ABOmatch", "CMVstatus",
    "Relapse", "aGvHDIIIIV", "extcGvHD",
    "ANCrecovery", "PLTrecovery", "time_to_aGvHD_III_IV", "survival_time"
]

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
    return model, scaler, feature_names, model_name

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧬 Pediatric Bone Marrow Transplant Outcome Predictor</h1>
    <p>AI-powered clinical decision support for hematopoietic stem cell transplantation in children</p>
    <span class="badge">XGBoost · 97.2% ROC-AUC</span>
    <span class="badge">187 Pediatric Patients</span>
    <span class="badge">37 Clinical Features</span>
    <span class="badge">SHAP Explainability</span>
</div>
""", unsafe_allow_html=True)

with st.expander("📋 About this Clinical Tool", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-card"><h4>🎯 Clinical Objective</h4>
        <p>Predict post-transplant survival in pediatric patients with hematologic malignancies and non-malignant disorders requiring allogeneic bone marrow transplantation.</p></div>
        <div class="info-card"><h4>👶 Patient Population</h4>
        <p>187 pediatric patients · Age range 0–18 years · Diagnoses include leukemia, aplastic anemia, thalassemia, and other hematologic conditions.</p></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card"><h4>🤖 Machine Learning Pipeline</h4>
        <p>4 models trained: XGBoost (best · 97.2% AUC) · LightGBM · Random Forest · SVM · SMOTE balancing · SHAP explainability.</p></div>
        <div class="info-card"><h4>⚠️ Clinical Disclaimer</h4>
        <p>This tool is for research purposes only. Clinical decisions must be made by qualified medical professionals in conjunction with complete patient assessment.</p></div>
        """, unsafe_allow_html=True)

try:
    model, scaler, feature_names, model_name = load_artifacts()
    st.success(f"✅ Model loaded: **{model_name}** · ROC-AUC: 97.2% · Accuracy: 92.1%")
except Exception as e:
    st.error(f"⚠️ Model not found. Run `python -m src.train_model` first.\n{e}")
    st.stop()

# ── Sidebar with 3 tabs ───────────────────────────────────────────────────
st.sidebar.markdown("## 🩺 Patient Data Entry")
tab1, tab2, tab3 = st.sidebar.tabs(["👤 Donor", "👶 Recipient", "🏥 Clinical"])

inputs = {feat: 0.0 for feat in feature_names}

with tab1:
    st.markdown('<div class="tab-header">Donor Information</div>', unsafe_allow_html=True)
    for feat in DONOR_FEATURES:
        if feat in feature_names:
            label = FEATURE_LABELS.get(feat, feat)
            inputs[feat] = st.number_input(label, value=0.0, key=f"d_{feat}", format="%.2f")

with tab2:
    st.markdown('<div class="tab-header">Recipient (Child) Information</div>', unsafe_allow_html=True)
    for feat in RECIPIENT_FEATURES:
        if feat in feature_names:
            label = FEATURE_LABELS.get(feat, feat)
            inputs[feat] = st.number_input(label, value=0.0, key=f"r_{feat}", format="%.2f")

with tab3:
    st.markdown('<div class="tab-header">Clinical & Transplant Data</div>', unsafe_allow_html=True)
    for feat in CLINICAL_FEATURES:
        if feat in feature_names:
            label = FEATURE_LABELS.get(feat, feat)
            inputs[feat] = st.number_input(label, value=0.0, key=f"c_{feat}", format="%.2f")

# remaining features not in any group
for feat in feature_names:
    if feat not in DONOR_FEATURES + RECIPIENT_FEATURES + CLINICAL_FEATURES:
        inputs[feat] = 0.0

# ── Main content ──────────────────────────────────────────────────────────
col_pred, col_info = st.columns([2, 1])

with col_pred:
    st.markdown('<div class="section-title">🔮 Transplant Outcome Prediction</div>', unsafe_allow_html=True)
    if st.button("🚀 Predict Transplant Outcome", type="primary", use_container_width=True):
        input_df = pd.DataFrame([inputs])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        prediction = model.predict(input_scaled_df)[0]
        probability = model.predict_proba(input_scaled_df)[0][1]

        if prediction == 1:
            st.markdown(f"""<div class="result-success">
                <div class="result-title">✅ Transplant likely to SUCCEED</div>
                <div class="result-prob" style="color:#2e7d32">{probability*100:.1f}%</div>
                <div style="color:#555;font-size:0.85rem;margin-top:0.5rem">Predicted Survival Probability</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-risk">
                <div class="result-title">⚠️ High Risk of Transplant Failure</div>
                <div class="result-prob" style="color:#c62828">{(1-probability)*100:.1f}%</div>
                <div style="color:#555;font-size:0.85rem;margin-top:0.5rem">Predicted Failure Probability</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        fig_g, ax = plt.subplots(figsize=(8, 1.4))
        ax.barh(0, 1, color='#f0f0f0', height=0.5)
        ax.barh(0, probability, color='#4caf50' if prediction == 1 else '#f44336', height=0.5)
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.axvline(x=0.5, color='#888', linestyle='--', linewidth=1, label='Decision threshold')
        ax.set_title(f"Survival Probability: {probability*100:.1f}%", fontweight='bold', color='#0a2342')
        ax.legend(fontsize=7, loc='upper left')
        plt.tight_layout(); st.pyplot(fig_g); plt.close()

        # SHAP
        st.markdown('<div class="section-title">🔍 Clinical Feature Impact (SHAP)</div>', unsafe_allow_html=True)
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled_df)
            sv = shap_values[0] if not isinstance(shap_values, list) else shap_values[1][0]
            idx = np.argsort(np.abs(sv))[::-1][:12]
            top_feat_raw = [feature_names[i] for i in idx]
            top_feat_label = [FEATURE_LABELS.get(f, f) for f in top_feat_raw]
            top_val = [sv[i] for i in idx]
            colors = ['#c62828' if v > 0 else '#1565c0' for v in top_val]

            fig_shap, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top_feat_label[::-1], top_val[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
            ax.axvline(x=0, color='#333', linewidth=1)
            ax.set_title("Top 12 Clinical Features Driving Prediction", fontweight='bold', color='#0a2342', fontsize=11)
            ax.set_xlabel("SHAP Value (impact on predicted survival)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            red_p = mpatches.Patch(color='#c62828', label='Increases survival probability')
            blue_p = mpatches.Patch(color='#1565c0', label='Decreases survival probability')
            ax.legend(handles=[red_p, blue_p], fontsize=8)
            plt.tight_layout(); st.pyplot(fig_shap); plt.close()
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

with col_info:
    st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)
    for val, label in [("97.2%","ROC-AUC"),("92.1%","Accuracy"),("93.8%","Precision"),("88.2%","Recall")]:
        st.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Best Model</div>', unsafe_allow_html=True)
    st.info(f"**{model_name}**\n\nAutomatically selected based on highest ROC-AUC among 4 trained classifiers.")
    st.markdown('<div class="section-title">📌 Key Clinical Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card"><p>
    🔴 <b>Survival Time</b> — strongest predictor<br>
    🔴 <b>CD34+ Cell Dose</b> — stem cell engraftment<br>
    🔴 <b>ANC Recovery Day</b> — immune reconstitution<br>
    🔵 <b>Donor Age</b> — graft quality<br>
    🔵 <b>Recipient Body Mass</b> — conditioning tolerance
    </p></div>
    """, unsafe_allow_html=True)

# ── Model Comparison ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)
if os.path.exists("models/model_comparison.csv"):
    df_res = pd.read_csv("models/model_comparison.csv", index_col=0)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df_res.style.highlight_max(axis=0, color='#e3f2fd').format("{:.4f}"), use_container_width=True)
    with c2:
        fig_c, ax = plt.subplots(figsize=(6, 4))
        ml = df_res.index.tolist()
        x = np.arange(len(ml)); w = 0.25
        for i, (met, col) in enumerate(zip(['ROC-AUC','Accuracy','F1-Score'],['#1a4a7a','#00b4d8','#4caf50'])):
            if met in df_res.columns:
                ax.bar(x+i*w, df_res[met], w, label=met, color=col, alpha=0.85)
        ax.set_xticks(x+w); ax.set_xticklabels(ml, fontsize=8)
        ax.set_ylim(0.7, 1.0)
        ax.set_title("Classifier Performance Comparison", fontweight='bold', color='#0a2342')
        ax.legend(fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_c); plt.close()

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.78rem;padding:0.8rem'>
    🎓 <b>Coding Week 2026</b> · École Centrale Casablanca · Project 4 — Medical Machine Learning<br>
    Dataset: UCI ML Repository · Bone Marrow Transplant in Children (ID: 565) · 187 patients · 37 features<br>
    ⚠️ For research purposes only — not a substitute for clinical judgment
</div>
""", unsafe_allow_html=True)
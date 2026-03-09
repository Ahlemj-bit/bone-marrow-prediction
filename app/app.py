import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Bone Marrow Transplant Predictor",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
    return model, scaler, feature_names, model_name

st.title("🧬 Bone Marrow Transplant Success Predictor")
st.markdown("**Clinical Decision Support Tool — Pediatric Patients**")

try:
    model, scaler, feature_names, model_name = load_artifacts()
    st.success(f"✅ Model loaded: **{model_name}**")
except Exception as e:
    st.error(f"⚠️ Model not found. Run `python -m src.train_model` first.\n{e}")
    st.stop()

st.divider()
st.sidebar.header("🩺 Patient Data")

inputs = {}
for feat in feature_names:
    inputs[feat] = st.sidebar.number_input(feat, value=0.0)

st.subheader("🔮 Prediction")

if st.button("🚀 Predict", type="primary", use_container_width=True):
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    prediction = model.predict(input_scaled_df)[0]
    probability = model.predict_proba(input_scaled_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Transplant likely to SUCCEED — {probability*100:.1f}%")
    else:
        st.error(f"⚠️ Risk of FAILURE — {(1-probability)*100:.1f}%")

    st.metric("Success Probability", f"{probability*100:.1f}%")

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled_df)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        idx = np.argsort(np.abs(sv))[::-1][:10]
        top_feat = [feature_names[i] for i in idx]
        top_val = [sv[i] for i in idx]
        colors = ["#e53935" if v > 0 else "#1e88e5" for v in top_val]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_feat[::-1], top_val[::-1], color=colors[::-1])
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_title("SHAP — Feature Contributions")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP unavailable: {e}")

st.divider()
if os.path.exists("models/model_comparison.csv"):
    st.subheader("📊 Model Comparison")
    df_res = pd.read_csv("models/model_comparison.csv", index_col=0)
    st.dataframe(df_res, use_container_width=True)

st.caption("Coding Week 2026 — École Centrale Casablanca 🎓")
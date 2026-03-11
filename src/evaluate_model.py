"""
src/evaluate_model.py
Évaluation du modèle et génération des visualisations SHAP.

Usage :
    python src/evaluate_model.py
"""

import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SRC_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from data_processing import preprocess_pipeline

MODEL_PATH = ROOT_DIR / "data" / "best_model.pkl"
FEAT_PATH  = ROOT_DIR / "data" / "feature_names.pkl"
DATA_PATH  = ROOT_DIR / "data" / "bone_marrow.csv"
OUTPUT_DIR = ROOT_DIR / "data"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_feature_names():
    with open(FEAT_PATH, "rb") as f:
        return pickle.load(f)


def evaluate(model, X_test, y_test, feature_names):
    """Calcule et affiche toutes les métriques de performance."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1-score":  f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }

    print("\n── Métriques de performance ──")
    for k, v in metrics.items():
        print(f"  {k:12s} : {v:.4f}")
    print("\n" + classification_report(
        y_test, y_pred, target_names=["Non-survie", "Survie"]
    ))
    return metrics


def compute_shap(model, X, feature_names, max_samples: int = 100):
    """
    Calcule les valeurs SHAP et génère le summary plot.
    Retourne (explainer, shap_values).
    """
    try:
        import shap
    except ImportError:
        print("⚠  shap non installé. Lancez : pip install shap")
        return None, None

    X_sample = X[:max_samples]

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer   = shap.KernelExplainer(model.predict_proba, X_sample[:30])
        shap_values = explainer.shap_values(X_sample)

    # Pour les modèles binaires shap_values peut être [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✔  SHAP summary plot → {out_path}")

    return explainer, shap_values


def get_shap_for_instance(explainer, instance: np.ndarray, feature_names):
    """
    Calcule les valeurs SHAP pour une instance unique.
    Retourne un dict {feature: shap_value}.
    """
    try:
        import shap
        sv = explainer.shap_values(instance)
        sv = sv[1] if isinstance(sv, list) else sv
        return dict(zip(feature_names, sv.flatten()))
    except Exception:
        return {}


if __name__ == "__main__":
    model        = load_model()
    feature_names = load_feature_names()

    X, y, _, _ = preprocess_pipeline(str(DATA_PATH), use_smote=False)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    evaluate(model, X_test, y_test, feature_names)
    compute_shap(model, X_test, feature_names)

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.data_processing import load_data, preprocess

MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0),
    "LightGBM": LGBMClassifier(n_estimators=200, random_state=42, class_weight="balanced", verbose=-1),
    "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

def train_all_models(X_train, X_test, y_train, y_test):
    results = {}
    trained = {}
    for name, model in MODELS.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        print(f"   {metrics}")
    results_df = pd.DataFrame(results).T
    print("\n=== Model Comparison ===")
    print(results_df.sort_values("ROC-AUC", ascending=False))
    return results_df, trained

def select_best_model(results_df, trained_models):
    best_name = results_df["ROC-AUC"].idxmax()
    best_model = trained_models[best_name]
    print(f"\n🏆 Best model: {best_name}")
    return best_name, best_model

def save_model(model, name, path="models/best_model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    joblib.dump(name, "models/best_model_name.pkl")
    print(f"✅ Model saved → {path}")

def main():
    data_path = "data/bone_marrow.csv"
    if not os.path.exists(data_path):
        print("⚠️ Downloading dataset from UCI...")
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=565)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            os.makedirs("data", exist_ok=True)
            df.to_csv(data_path, index=False)
            print(f"✅ Dataset saved to {data_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return
    else:
        df = load_data(data_path)

    X_train, X_test, y_train, y_test, _ = preprocess(df)
    results_df, trained_models = train_all_models(X_train, X_test, y_train, y_test)
    results_df.to_csv("models/model_comparison.csv")
    best_name, best_model = select_best_model(results_df, trained_models)
    save_model(best_model, best_name)
    y_pred = best_model.predict(X_test)
    print(f"\n📋 Report — {best_name}")
    print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

if __name__ == "__main__":
    main()
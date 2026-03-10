import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

model = joblib.load("models/best_model.pkl")

df = pd.read_csv("data/bone_marrow.csv")

target = "target"
X = df.drop(columns=[target])
y = df[target]

pred_proba = model.predict_proba(X)[:,1]

auc = roc_auc_score(y, pred_proba)

print("ROC AUC:", auc)

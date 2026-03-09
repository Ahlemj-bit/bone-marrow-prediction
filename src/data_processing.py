"""
data_processing.py
Core data loading, cleaning, and preprocessing pipeline.
Bone Marrow Transplant — Pediatric Patients
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


# Target column name in the bone marrow dataset
TARGET_COL = "survival_status"


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset from given path."""
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    missing = df.isnull().sum()
    total = missing.sum()
    if total > 0:
        print(f"⚠️  Missing values found: {total} total")
        print(missing[missing > 0])
    else:
        print("✅ No missing values found.")
    return missing


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - Numeric columns  → median (robust to outliers)
    - Categorical/object columns → mode
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    print("✅ Missing values handled.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object/categorical columns."""
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    print("✅ Categorical columns encoded.")
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    float64 -> float32  |  int64 -> int32
    """
    before = df.memory_usage(deep=True).sum() / 1024
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    after = df.memory_usage(deep=True).sum() / 1024
    saved_pct = (before - after) / before * 100 if before > 0 else 0
    print(f"✅ Memory: {before:.2f} KB → {after:.2f} KB "
          f"(saved {saved_pct:.1f}%)")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove outliers using the IQR method for specified columns."""
    original_len = len(df)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[
            (df[col] >= Q1 - 1.5 * IQR) &
            (df[col] <= Q3 + 1.5 * IQR)
        ]
    print(f"✅ Outliers removed: {original_len - len(df)} rows dropped.")
    return df.reset_index(drop=True)


def handle_imbalance(X: pd.DataFrame, y: pd.Series):
    """
    Apply SMOTE oversampling to handle class imbalance.
    Applied ONLY on training data to prevent data leakage.
    """
    print(f"Before SMOTE → {dict(pd.Series(y).value_counts())}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"After  SMOTE → {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def preprocess(df: pd.DataFrame,
               target_col: str = TARGET_COL,
               test_size: float = 0.2,
               apply_smote: bool = True):
    """
    Full preprocessing pipeline:
      1. Handle missing values
      2. Encode categorical features
      3. Optimize memory
      4. Split features / target
      5. Train / test split (stratified)
      6. Scale features (StandardScaler)
      7. Handle class imbalance (SMOTE on train only)

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    df = handle_missing_values(df.copy())
    df = encode_categoricals(df)
    df = optimize_memory(df)

    # Detect target column (flexible naming)
    if target_col not in df.columns:
        # Try common alternative names
        for candidate in ["survival_status", "survival", "event",
                          "Status", "status", "label", "target"]:
            if candidate in df.columns:
                target_col = candidate
                break
        else:
            target_col = df.columns[-1]
            print(f"⚠️  Target not found, using last column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    if apply_smote:
        X_train_sc, y_train = handle_imbalance(X_train_sc, y_train)
        y_train = pd.Series(y_train).reset_index(drop=True)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    joblib.dump(target_col, "models/target_col.pkl")
    print("✅ Scaler and feature names saved.")

    return X_train_sc, X_test_sc, y_train, y_test, scaler
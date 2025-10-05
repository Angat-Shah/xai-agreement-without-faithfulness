"""
XAI Agreement and Faithfulness Study
Complete Experiment Pipeline

This script:
1. Loads and preprocesses the UCI Adult Income dataset
2. Trains an XGBoost classifier
3. Generates explanations via SHAP, LIME, and Permutation Sensitivity (PS)
4. Evaluates explanation agreement (RQ1)
5. Evaluates faithfulness via comprehensiveness and sufficiency (RQ2)
6. Analyzes agreement vs faithfulness relationship (RQ3)
7. Evaluates explanation stability (RQ4)
8. Saves all results as machine-readable files
"""

import os
import sys
import json
import time
import warnings
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb
import shap
import lime
import lime.lime_tabular

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

np.random.seed(RANDOM_SEED)


# ============================================================================
# PHASE 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_adult_dataset():
    """Load UCI Adult Income dataset from sklearn/OpenML."""
    print("=" * 70)
    print("PHASE 1: Loading and preprocessing data")
    print("=" * 70)

    import ssl
    import urllib.request

    # Fix SSL certificate verification issue on macOS
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    from sklearn.datasets import fetch_openml
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = adult.frame

    print(f"Raw dataset shape: {df.shape}")

    # Detect target column (OpenML v2 uses 'class')
    target_col = "class" if "class" in df.columns else "income"
    print(f"Target column: {target_col}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")

    # Clean the dataset
    df = df.dropna().reset_index(drop=True)
    print(f"After dropping NaN: {df.shape}")

    # Identify feature types
    feature_cols = [c for c in df.columns if c != target_col]

    categorical_cols = df[feature_cols].select_dtypes(include=["category", "object"]).columns.tolist()
    continuous_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

    print(f"Continuous features ({len(continuous_cols)}): {continuous_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Encode target
    y = (df[target_col].astype(str).str.strip().str.replace(".", "", regex=False).isin([">50K", ">50K."])).astype(int)

    # Ordinal encode categoricals for XGBoost
    X = df[feature_cols].copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Convert all to float
    X = X.astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")

    dataset_info = {
        "name": DATASET_NAME,
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "n_continuous": len(continuous_cols),
        "n_categorical": len(categorical_cols),
        "feature_names": feature_cols,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "target_distribution": y.value_counts().to_dict(),
    }

    return X_train, X_test, y_train, y_test, dataset_info, label_encoders


# ============================================================================
# PHASE 2: MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier and evaluate."""
    print("\n" + "=" * 70)
    print("PHASE 2: Training XGBoost model")
    print("=" * 70)

    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    return model, metrics



# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("XAI AGREEMENT AND FAITHFULNESS STUDY")
    print("=" * 70)
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # Phase 1: Data
    X_train, X_test, y_train, y_test, dataset_info, label_encoders = load_adult_dataset()

    # Phase 2: Model
    model, model_metrics = train_model(X_train, y_train, X_test, y_test)
    print(f"Baseline XGBoost Accuracy: {model_metrics['accuracy']:.4f}, AUC: {model_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()

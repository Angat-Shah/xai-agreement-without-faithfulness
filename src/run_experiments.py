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
# PHASE 3: SELECT EXPLANATION SAMPLES
# ============================================================================

def select_explanation_samples(model, X_test, y_test, n_samples=N_EXPLAIN_SAMPLES):
    """Select stratified sample of test instances for explanation."""
    print(f"\nSelecting {n_samples} instances for XAI evaluation...")

    probs = model.predict_proba(X_test)[:, 1]
    confidence = np.abs(probs - 0.5) * 2  # 0 = uncertain, 1 = confident

    # Stratify by confidence terciles and predicted class
    preds = model.predict(X_test)
    conf_terciles = pd.qcut(confidence, q=3, labels=["low", "medium", "high"])

    strata = pd.Series(
        [f"{c}_{p}" for c, p in zip(conf_terciles, preds)],
        index=X_test.index
    )

    # Proportional sampling from each stratum
    sampled_idx = []
    for stratum in strata.unique():
        stratum_idx = strata[strata == stratum].index
        n_from_stratum = max(1, int(n_samples * len(stratum_idx) / len(strata)))
        n_from_stratum = min(n_from_stratum, len(stratum_idx))
        sampled = np.random.choice(stratum_idx, size=n_from_stratum, replace=False)
        sampled_idx.extend(sampled)

    # Trim or pad to exact n_samples
    sampled_idx = np.array(sampled_idx)
    if len(sampled_idx) > n_samples:
        sampled_idx = np.random.choice(sampled_idx, size=n_samples, replace=False)
    elif len(sampled_idx) < n_samples:
        remaining = np.setdiff1d(X_test.index, sampled_idx)
        extra = np.random.choice(remaining, size=n_samples - len(sampled_idx), replace=False)
        sampled_idx = np.concatenate([sampled_idx, extra])

    X_sample = X_test.loc[sampled_idx]
    y_sample = y_test.loc[sampled_idx]

    print(f"Selected {len(X_sample)} instances")
    print(f"  Predicted class distribution: {pd.Series(model.predict(X_sample)).value_counts().to_dict()}")
    sample_probs = model.predict_proba(X_sample)[:, 1]
    sample_conf = np.abs(sample_probs - 0.5) * 2
    print(f"  Confidence: mean={sample_conf.mean():.3f}, std={sample_conf.std():.3f}")

    return X_sample, y_sample


# ============================================================================
# PHASE 4: GENERATE EXPLANATIONS
# ============================================================================

def generate_shap_explanations(model, X_sample, X_train):
    """Generate TreeSHAP explanations."""
    print("\n" + "-" * 50)
    print("Generating SHAP (TreeSHAP) explanations...")
    t0 = time.time()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values shape: (n_samples, n_features)
    # For binary classification, TreeExplainer returns values for positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class

    elapsed = time.time() - t0
    print(f"  SHAP explanations generated in {elapsed:.1f}s")
    print(f"  Shape: {shap_values.shape}")

    return shap_values


def generate_lime_explanations(model, X_sample, X_train, dataset_info):
    """Generate LIME explanations."""
    print("\n" + "-" * 50)
    print("Generating LIME explanations...")
    t0 = time.time()

    feature_names = dataset_info["feature_names"]
    categorical_indices = [feature_names.index(c) for c in dataset_info["categorical_cols"]]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        categorical_features=categorical_indices,
        class_names=["<=50K", ">50K"],
        mode="classification",
        random_state=RANDOM_SEED,
    )

    lime_values = np.zeros((len(X_sample), len(feature_names)))

    for i, (idx, row) in enumerate(X_sample.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  LIME: {i+1}/{len(X_sample)}")

        exp = explainer.explain_instance(
            row.values,
            model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_NUM_SAMPLES,
        )

        # Extract coefficients for positive class (label=1)
        exp_map = dict(exp.as_map().get(1, exp.as_map().get(0, [])))
        for feat_idx, weight in exp_map.items():
            lime_values[i, feat_idx] = weight

    elapsed = time.time() - t0
    print(f"  LIME explanations generated in {elapsed:.1f}s")
    print(f"  Shape: {lime_values.shape}")

    return lime_values


def generate_permutation_importance(model, X_sample, dataset_info):
    """Generate per-instance permutation-based sensitivity (PS)."""
    print("\n" + "-" * 50)
    print("Generating Permutation Sensitivity (PS) explanations...")
    t0 = time.time()

    n_samples, n_features = X_sample.shape
    perm_values = np.zeros((n_samples, n_features))

    original_probs = model.predict_proba(X_sample.values)[:, 1]

    n_repeats = 20  # Number of permutation repeats per feature

    for j in range(n_features):
        if (j + 1) % 5 == 0:
            print(f"  Permutation: feature {j+1}/{n_features}")

        importance_scores = np.zeros(n_samples)

        for _ in range(n_repeats):
            X_permuted = X_sample.values.copy()
            # Shuffle this feature across the sample
            perm_idx = np.random.permutation(n_samples)
            X_permuted[:, j] = X_permuted[perm_idx, j]

            permuted_probs = model.predict_proba(X_permuted)[:, 1]
            # Importance = absolute change in predicted probability
            importance_scores += np.abs(original_probs - permuted_probs)

        perm_values[:, j] = importance_scores / n_repeats

    elapsed = time.time() - t0
    print(f"  Permutation sensitivity generated in {elapsed:.1f}s")
    print(f"  Shape: {perm_values.shape}")

    return perm_values


# ============================================================================
# PHASE 5: AGREEMENT ANALYSIS (RQ1)
# ============================================================================

def compute_agreement(shap_vals, lime_vals, perm_vals, feature_names):
    """Compute pairwise agreement between explanation methods."""
    print("\n" + "=" * 70)
    print("PHASE 5: Computing explanation agreement (RQ1)")
    print("=" * 70)

    n_samples = shap_vals.shape[0]
    n_features = shap_vals.shape[1]

    methods = {
        "SHAP": np.abs(shap_vals),  # Use absolute values for ranking
        "LIME": np.abs(lime_vals),
        "PermImp": perm_vals,  # Already absolute
    }

    method_names = list(methods.keys())
    pairs = [(method_names[i], method_names[j])
             for i in range(len(method_names))
             for j in range(i+1, len(method_names))]

    results = {
        "spearman": {},
        "kendall": {},
        "topk_3": {},
        "topk_5": {},
    }

    for m1, m2 in pairs:
        pair_key = f"{m1}_vs_{m2}"
        spearman_rhos = []
        kendall_taus = []
        topk3_overlaps = []
        topk5_overlaps = []

        for i in range(n_samples):
            v1 = methods[m1][i]
            v2 = methods[m2][i]

            # Skip if either is all zeros
            if np.std(v1) == 0 or np.std(v2) == 0:
                continue

            # Spearman
            rho, _ = stats.spearmanr(v1, v2)
            if not np.isnan(rho):
                spearman_rhos.append(rho)

            # Kendall
            tau, _ = stats.kendalltau(v1, v2)
            if not np.isnan(tau):
                kendall_taus.append(tau)

            # Top-k overlap
            rank1 = np.argsort(-v1)
            rank2 = np.argsort(-v2)

            topk3 = len(set(rank1[:3]) & set(rank2[:3])) / 3.0
            topk3_overlaps.append(topk3)

            topk5 = len(set(rank1[:5]) & set(rank2[:5])) / 5.0
            topk5_overlaps.append(topk5)

        results["spearman"][pair_key] = {
            "mean": float(np.mean(spearman_rhos)),
            "std": float(np.std(spearman_rhos)),
            "median": float(np.median(spearman_rhos)),
            "values": [float(v) for v in spearman_rhos],
        }
        results["kendall"][pair_key] = {
            "mean": float(np.mean(kendall_taus)),
            "std": float(np.std(kendall_taus)),
            "median": float(np.median(kendall_taus)),
            "values": [float(v) for v in kendall_taus],
        }
        results["topk_3"][pair_key] = {
            "mean": float(np.mean(topk3_overlaps)),
            "std": float(np.std(topk3_overlaps)),
            "values": [float(v) for v in topk3_overlaps],
        }
        results["topk_5"][pair_key] = {
            "mean": float(np.mean(topk5_overlaps)),
            "std": float(np.std(topk5_overlaps)),
            "values": [float(v) for v in topk5_overlaps],
        }

        print(f"\n{pair_key}:")
        print(f"  Spearman ρ: {results['spearman'][pair_key]['mean']:.3f} ± {results['spearman'][pair_key]['std']:.3f}")
        print(f"  Kendall τ:  {results['kendall'][pair_key]['mean']:.3f} ± {results['kendall'][pair_key]['std']:.3f}")
        print(f"  Top-3 overlap: {results['topk_3'][pair_key]['mean']:.3f} ± {results['topk_3'][pair_key]['std']:.3f}")
        print(f"  Top-5 overlap: {results['topk_5'][pair_key]['mean']:.3f} ± {results['topk_5'][pair_key]['std']:.3f}")

    return results


# ============================================================================
# PHASE 6: FAITHFULNESS EVALUATION (RQ2)
# ============================================================================

def compute_masking_values(X_train, dataset_info):
    """Compute training-set median/mode for feature masking."""
    mask_values = {}
    for col in dataset_info["feature_names"]:
        if col in dataset_info["continuous_cols"]:
            mask_values[col] = float(X_train[col].median())
        else:
            mask_values[col] = float(X_train[col].mode().iloc[0])
    return mask_values


def compute_faithfulness(model, X_sample, shap_vals, lime_vals, perm_vals,
                         dataset_info, mask_values):
    """Compute comprehensiveness and sufficiency for each method."""
    print("\n" + "=" * 70)
    print("PHASE 6: Computing faithfulness metrics (RQ2)")
    print("=" * 70)

    feature_names = dataset_info["feature_names"]
    n_features = len(feature_names)

    methods = {
        "SHAP": np.abs(shap_vals),
        "LIME": np.abs(lime_vals),
        "PermImp": perm_vals,
    }

    # Also add random baseline
    rng = np.random.RandomState(RANDOM_SEED + 1)
    methods["Random"] = rng.rand(*shap_vals.shape)

    mask_array = np.array([mask_values[f] for f in feature_names])

    # Original predictions
    original_probs = model.predict_proba(X_sample.values)[:, 1]
    original_preds = model.predict(X_sample.values)

    results = {"comprehensiveness": {}, "sufficiency": {}}

    for method_name, importance_vals in methods.items():
        print(f"\n  Method: {method_name}")
        comp_scores = {k: [] for k in FAITHFULNESS_K_VALUES}
        suff_scores = {k: [] for k in FAITHFULNESS_K_VALUES}

        for i in range(len(X_sample)):
            x = X_sample.values[i].copy()
            imp = importance_vals[i]
            ranked_features = np.argsort(-imp)  # Descending importance
            pred_class = original_preds[i]
            orig_prob = original_probs[i] if pred_class == 1 else (1 - original_probs[i])

            for k in FAITHFULNESS_K_VALUES:
                if k > n_features:
                    continue

                # Comprehensiveness: remove top-k features
                x_comp = x.copy()
                for fi in ranked_features[:k]:
                    x_comp[fi] = mask_array[fi]
                comp_prob = model.predict_proba(x_comp.reshape(1, -1))[0]
                comp_prob_class = comp_prob[pred_class]
                comp_scores[k].append(orig_prob - comp_prob_class)

                # Sufficiency: keep only top-k features
                x_suff = mask_array.copy()
                for fi in ranked_features[:k]:
                    x_suff[fi] = x[fi]
                suff_prob = model.predict_proba(x_suff.reshape(1, -1))[0]
                suff_prob_class = suff_prob[pred_class]
                suff_scores[k].append(orig_prob - suff_prob_class)

        results["comprehensiveness"][method_name] = {}
        results["sufficiency"][method_name] = {}

        for k in FAITHFULNESS_K_VALUES:
            results["comprehensiveness"][method_name][f"k={k}"] = {
                "mean": float(np.mean(comp_scores[k])),
                "std": float(np.std(comp_scores[k])),
                "median": float(np.median(comp_scores[k])),
                "values": [float(v) for v in comp_scores[k]],
            }
            results["sufficiency"][method_name][f"k={k}"] = {
                "mean": float(np.mean(suff_scores[k])),
                "std": float(np.std(suff_scores[k])),
                "median": float(np.median(suff_scores[k])),
                "values": [float(v) for v in suff_scores[k]],
            }

            print(f"    k={k}: Comp={np.mean(comp_scores[k]):.4f}±{np.std(comp_scores[k]):.4f}, "
                  f"Suff={np.mean(suff_scores[k]):.4f}±{np.std(suff_scores[k]):.4f}")

    return results



# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("XAI AGREEMENT AND FAITHFULNESS STUDY")
    print("=" * 70)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"N explanation samples: {N_EXPLAIN_SAMPLES}")
    print()

    # Phase 1: Data
    X_train, X_test, y_train, y_test, dataset_info, label_encoders = load_adult_dataset()

    # Phase 2: Model
    model, model_metrics = train_model(X_train, y_train, X_test, y_test)

    # Phase 3: Select samples
    X_sample, y_sample = select_explanation_samples(model, X_test, y_test)

    # Phase 4: Generate explanations
    shap_vals = generate_shap_explanations(model, X_sample, X_train)
    lime_vals = generate_lime_explanations(model, X_sample, X_train, dataset_info)
    perm_vals = generate_permutation_importance(model, X_sample, dataset_info)

    # Save raw explanations sample
    X_sample.to_csv(os.path.join(RAW_DIR, "X_sample.csv"), index=False)
    y_sample.to_csv(os.path.join(RAW_DIR, "y_sample.csv"), index=False)

    # Phase 5: Agreement
    agreement_results = compute_agreement(
        shap_vals, lime_vals, perm_vals, dataset_info["feature_names"]
    )

    # Phase 6: Faithfulness
    mask_values = compute_masking_values(X_train, dataset_info)
    faithfulness_results = compute_faithfulness(
        model, X_sample, shap_vals, lime_vals, perm_vals, dataset_info, mask_values
    )

if __name__ == "__main__":
    main()

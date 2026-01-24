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
# PHASE 7: AGREEMENT VS FAITHFULNESS (RQ3)
# ============================================================================

def compute_agreement_vs_faithfulness(agreement_results, faithfulness_results,
                                       shap_vals, lime_vals, perm_vals, model,
                                       X_sample, dataset_info, mask_values):
    """Analyze whether inter-method agreement predicts faithfulness."""
    print("\n" + "=" * 70)
    print("PHASE 7: Agreement vs. Faithfulness analysis (RQ3)")
    print("=" * 70)

    feature_names = dataset_info["feature_names"]
    n_features = len(feature_names)
    mask_array = np.array([mask_values[f] for f in feature_names])

    methods = {
        "SHAP": np.abs(shap_vals),
        "LIME": np.abs(lime_vals),
        "PermImp": perm_vals,
    }

    # For each instance, compute agreement between SHAP and LIME
    # and the average faithfulness (comprehensiveness) of both
    n_samples = len(X_sample)
    original_probs = model.predict_proba(X_sample.values)[:, 1]
    original_preds = model.predict(X_sample.values)

    pair_analyses = {}

    method_names = list(methods.keys())
    pairs = [(method_names[i], method_names[j])
             for i in range(len(method_names))
             for j in range(i+1, len(method_names))]

    for m1, m2 in pairs:
        pair_key = f"{m1}_vs_{m2}"
        agreements = []
        comp_a_k3 = []
        comp_b_k3 = []
        avg_comp_k3 = []

        for i in range(n_samples):
            v1 = methods[m1][i]
            v2 = methods[m2][i]

            if np.std(v1) == 0 or np.std(v2) == 0:
                continue

            rho, _ = stats.spearmanr(v1, v2)
            if np.isnan(rho):
                continue
            agreements.append(rho)

            # Average comprehensiveness at k=3 for both methods
            x = X_sample.values[i].copy()
            pred_class = original_preds[i]
            orig_prob = original_probs[i] if pred_class == 1 else (1 - original_probs[i])

            comp_vals = []
            for vals in [v1, v2]:
                ranked = np.argsort(-vals)
                x_comp = x.copy()
                for fi in ranked[:3]:
                    x_comp[fi] = mask_array[fi]
                comp_prob = model.predict_proba(x_comp.reshape(1, -1))[0][pred_class]
                comp_vals.append(orig_prob - comp_prob)

            comp_a_k3.append(comp_vals[0])
            comp_b_k3.append(comp_vals[1])
            avg_comp_k3.append(np.mean(comp_vals))

        agreements = np.array(agreements)
        avg_comp_k3 = np.array(avg_comp_k3)
        comp_a_k3 = np.array(comp_a_k3)
        comp_b_k3 = np.array(comp_b_k3)

        # Correlation between agreement and faithfulness
        corr_pearson, p_pearson = stats.pearsonr(agreements, avg_comp_k3)
        corr_spearman, p_spearman = stats.spearmanr(agreements, avg_comp_k3)

        # RQ3 Sensitivity Analysis: correlate agreement with individual methods
        r_a, p_a = stats.spearmanr(agreements, comp_a_k3)
        r_b, p_b = stats.spearmanr(agreements, comp_b_k3)

        # Split into high/low agreement groups
        median_agreement = np.median(agreements)
        high_agree_mask = agreements >= median_agreement
        low_agree_mask = agreements < median_agreement

        high_agree_faith = avg_comp_k3[high_agree_mask]
        low_agree_faith = avg_comp_k3[low_agree_mask]

        # Wilcoxon test
        if len(high_agree_faith) > 0 and len(low_agree_faith) > 0:
            u_stat, u_p = stats.mannwhitneyu(high_agree_faith, low_agree_faith, alternative='two-sided')
        else:
            u_stat, u_p = np.nan, np.nan

        pair_analyses[pair_key] = {
            "pearson_r": float(corr_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(corr_spearman),
            "spearman_p": float(p_spearman),
            "high_agreement_faithfulness_mean": float(np.mean(high_agree_faith)),
            "low_agreement_faithfulness_mean": float(np.mean(low_agree_faith)),
            "mannwhitney_U": float(u_stat) if not np.isnan(u_stat) else None,
            "mannwhitney_p": float(u_p) if not np.isnan(u_p) else None,
            "n_high": int(np.sum(high_agree_mask)),
            "n_low": int(np.sum(low_agree_mask)),
            "sensitivity_analysis": {
                "method_a": m1,
                "spearman_r_a": float(r_a),
                "spearman_p_a": float(p_a),
                "method_b": m2,
                "spearman_r_b": float(r_b),
                "spearman_p_b": float(p_b),
                "spearman_r_avg": float(corr_spearman),
                "spearman_p_avg": float(p_spearman),
            },
            "agreement_values": [float(v) for v in agreements],
            "faithfulness_values": [float(v) for v in avg_comp_k3],
        }

        print(f"\n{pair_key}:")
        print(f"  Pearson r(agreement, faithfulness) = {corr_pearson:.3f} (p={p_pearson:.4f})")
        print(f"  Spearman r(agreement, faithfulness) = {corr_spearman:.3f} (p={p_spearman:.4f})")
        print(f"  High-agreement faithfulness: {np.mean(high_agree_faith):.4f}")
        print(f"  Low-agreement faithfulness:  {np.mean(low_agree_faith):.4f}")
        if not np.isnan(u_p):
            print(f"  Mann-Whitney U p-value: {u_p:.4f}")

    return pair_analyses


# ============================================================================
# PHASE 8: STABILITY ANALYSIS (RQ4)
# ============================================================================

def compute_stability(model, X_sample, X_train, dataset_info):
    """Evaluate explanation stability under input perturbation."""
    print("\n" + "=" * 70)
    print("PHASE 8: Computing explanation stability (RQ4)")
    print("=" * 70)

    feature_names = dataset_info["feature_names"]
    continuous_cols = dataset_info["continuous_cols"]
    continuous_indices = [feature_names.index(c) for c in continuous_cols]
    categorical_indices = [feature_names.index(c) for c in dataset_info["categorical_cols"]]

    # Compute feature ranges for noise scaling
    feature_ranges = {}
    for col in continuous_cols:
        col_range = X_train[col].max() - X_train[col].min()
        feature_ranges[col] = col_range

    # Subsample for stability (computationally expensive)
    n_stability = min(100, len(X_sample))
    stability_idx = np.random.choice(len(X_sample), size=n_stability, replace=False)
    X_stability = X_sample.iloc[stability_idx]

    print(f"Computing stability for {n_stability} instances, {STABILITY_N_PERTURBATIONS} perturbations each")

    # SHAP explainer
    shap_explainer = shap.TreeExplainer(model)

    # LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        categorical_features=categorical_indices,
        class_names=["<=50K", ">50K"],
        mode="classification",
        random_state=RANDOM_SEED,
    )

    shap_stabilities = []
    lime_stabilities = []

    for idx_i, (idx, row) in enumerate(X_stability.iterrows()):
        if (idx_i + 1) % 20 == 0:
            print(f"  Stability: instance {idx_i+1}/{n_stability}")

        x_orig = row.values.copy()

        # Original explanations
        shap_orig = shap_explainer.shap_values(x_orig.reshape(1, -1))
        if isinstance(shap_orig, list):
            shap_orig = shap_orig[1]
        shap_orig = np.abs(shap_orig.flatten())

        lime_exp_orig = lime_explainer.explain_instance(
            x_orig, model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_NUM_SAMPLES,
        )
        lime_orig = np.zeros(len(feature_names))
        exp_map = dict(lime_exp_orig.as_map().get(1, lime_exp_orig.as_map().get(0, [])))
        for fi, w in exp_map.items():
            lime_orig[fi] = abs(w)

        shap_perturbed_rhos = []
        lime_perturbed_rhos = []

        for p in range(STABILITY_N_PERTURBATIONS):
            x_pert = x_orig.copy()
            # Perturb continuous features with Gaussian noise
            for ci, col in zip(continuous_indices, continuous_cols):
                noise_std = STABILITY_NOISE_SCALE * feature_ranges[col]
                x_pert[ci] += np.random.normal(0, noise_std)

            # SHAP on perturbed
            shap_pert = shap_explainer.shap_values(x_pert.reshape(1, -1))
            if isinstance(shap_pert, list):
                shap_pert = shap_pert[1]
            shap_pert = np.abs(shap_pert.flatten())

            if np.std(shap_orig) > 0 and np.std(shap_pert) > 0:
                rho, _ = stats.spearmanr(shap_orig, shap_pert)
                if not np.isnan(rho):
                    shap_perturbed_rhos.append(rho)

            # LIME on perturbed
            lime_exp_pert = lime_explainer.explain_instance(
                x_pert, model.predict_proba,
                num_features=len(feature_names),
                num_samples=LIME_NUM_SAMPLES,
            )
            lime_pert = np.zeros(len(feature_names))
            exp_map_pert = dict(lime_exp_pert.as_map().get(1, lime_exp_pert.as_map().get(0, [])))
            for fi, w in exp_map_pert.items():
                lime_pert[fi] = abs(w)

            if np.std(lime_orig) > 0 and np.std(lime_pert) > 0:
                rho, _ = stats.spearmanr(lime_orig, lime_pert)
                if not np.isnan(rho):
                    lime_perturbed_rhos.append(rho)

        if shap_perturbed_rhos:
            shap_stabilities.append(np.mean(shap_perturbed_rhos))
        if lime_perturbed_rhos:
            lime_stabilities.append(np.mean(lime_perturbed_rhos))

    results = {
        "SHAP": {
            "mean": float(np.mean(shap_stabilities)),
            "std": float(np.std(shap_stabilities)),
            "median": float(np.median(shap_stabilities)),
            "values": [float(v) for v in shap_stabilities],
        },
        "LIME": {
            "mean": float(np.mean(lime_stabilities)),
            "std": float(np.std(lime_stabilities)),
            "median": float(np.median(lime_stabilities)),
            "values": [float(v) for v in lime_stabilities],
        },
    }

    # Statistical comparison
    if len(shap_stabilities) > 0 and len(lime_stabilities) > 0:
        stat, p_val = stats.wilcoxon(
            shap_stabilities[:min(len(shap_stabilities), len(lime_stabilities))],
            lime_stabilities[:min(len(shap_stabilities), len(lime_stabilities))]
        )
        diffs = np.array(shap_stabilities) - np.array(lime_stabilities)
        non_zero = diffs[diffs != 0]
        n_nz = len(non_zero)
        ranks = stats.rankdata(np.abs(non_zero))
        w_plus = float(np.sum(ranks[non_zero > 0]))
        w_minus = float(np.sum(ranks[non_zero < 0]))
        total_ranks = n_nz * (n_nz + 1) / 2.0
        r_rb = float((w_plus - w_minus) / total_ranks) if total_ranks > 0 else 0.0

        results["wilcoxon_stat"] = float(stat)
        results["wilcoxon_p"] = float(p_val)
        results["w_plus"] = w_plus
        results["w_minus"] = w_minus
        results["matched_pairs_rank_biserial_r"] = r_rb

    print(f"\nStability Results:")
    print(f"  SHAP: {results['SHAP']['mean']:.3f} ± {results['SHAP']['std']:.3f}")
    print(f"  LIME: {results['LIME']['mean']:.3f} ± {results['LIME']['std']:.3f}")
    if "wilcoxon_p" in results:
        print(f"  Wilcoxon statistic W: {results['wilcoxon_stat']:.1f}")
        print(f"  Wilcoxon p-value: {results['wilcoxon_p']:.4f}")
        print(f"  Matched-pairs rank-biserial r: {results['matched_pairs_rank_biserial_r']:.3f}")

    return results


# ============================================================================
# PHASE 9: METRIC AGREEMENT ANALYSIS
# ============================================================================

def compute_metric_agreement(faithfulness_results):
    """Analyze whether different metrics agree on method rankings."""
    print("\n" + "=" * 70)
    print("PHASE 9: Metric agreement analysis")
    print("=" * 70)

    methods = ["SHAP", "LIME", "PermImp", "Random"]

    # For each k, rank methods by comprehensiveness and sufficiency
    rankings = {"comprehensiveness": {}, "sufficiency": {}}

    for k_str in [f"k={k}" for k in FAITHFULNESS_K_VALUES]:
        # Comprehensiveness: higher is better
        comp_means = {m: faithfulness_results["comprehensiveness"][m][k_str]["mean"]
                      for m in methods}
        comp_ranking = sorted(comp_means.keys(), key=lambda m: -comp_means[m])

        # Sufficiency: lower is better (less probability drop when keeping top features)
        suff_means = {m: faithfulness_results["sufficiency"][m][k_str]["mean"]
                      for m in methods}
        suff_ranking = sorted(suff_means.keys(), key=lambda m: suff_means[m])

        rankings["comprehensiveness"][k_str] = comp_ranking
        rankings["sufficiency"][k_str] = suff_ranking

        print(f"\n{k_str}:")
        print(f"  Comprehensiveness ranking: {' > '.join(comp_ranking)}")
        print(f"  Sufficiency ranking:       {' > '.join(suff_ranking)}")

        # Check if rankings agree
        agreement = comp_ranking == suff_ranking
        print(f"  Rankings agree: {agreement}")

    # Compute rank correlation between metrics across k values
    all_comp_ranks = []
    all_suff_ranks = []
    for k_str in [f"k={k}" for k in FAITHFULNESS_K_VALUES]:
        comp_r = rankings["comprehensiveness"][k_str]
        suff_r = rankings["sufficiency"][k_str]
        all_comp_ranks.append([comp_r.index(m) for m in methods])
        all_suff_ranks.append([suff_r.index(m) for m in methods])

    results = {
        "rankings": rankings,
        "methods": methods,
    }

    # Per-instance metric agreement: for each instance, do comp and suff
    # agree on which method is best?
    per_instance_agreement = {}
    for k_str in [f"k={k}" for k in FAITHFULNESS_K_VALUES]:
        agrees = 0
        total = 0
        real_methods = ["SHAP", "LIME", "PermImp"]

        comp_data = faithfulness_results["comprehensiveness"]
        suff_data = faithfulness_results["sufficiency"]

        n_instances = len(comp_data[real_methods[0]][k_str]["values"])

        for i in range(n_instances):
            # For each instance, which method has highest comprehensiveness?
            comp_best = max(real_methods,
                           key=lambda m: comp_data[m][k_str]["values"][i])
            # Which has lowest sufficiency score?
            suff_best = min(real_methods,
                           key=lambda m: suff_data[m][k_str]["values"][i])
            if comp_best == suff_best:
                agrees += 1
            total += 1

        agreement_rate = agrees / total if total > 0 else 0
        per_instance_agreement[k_str] = {
            "agreement_rate": float(agreement_rate),
            "n_agree": agrees,
            "n_total": total,
        }
        print(f"\n  Per-instance metric agreement at {k_str}: {agreement_rate:.3f} ({agrees}/{total})")

    results["per_instance_agreement"] = per_instance_agreement

    return results


# ============================================================================
# PHASE 8B: LIME SEED SENSITIVITY (STOCHASTICITY CHECK)
# ============================================================================

def evaluate_lime_seed_sensitivity(model, X_sample, X_train, dataset_info, n_instances=25):
    """Evaluate LIME stochastic sampling sensitivity across independent random seeds."""
    print("\n" + "=" * 70)
    print("PHASE 8B: LIME Seed Sensitivity Analysis (Fixed Inputs, Varied Seeds)")
    print("=" * 70)

    feature_names = dataset_info["feature_names"]
    categorical_indices = [feature_names.index(c) for c in dataset_info["categorical_cols"]]
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    n_eval = min(n_instances, len(X_sample))
    instance_rhos = []

    for idx in range(n_eval):
        row_vals = X_sample.iloc[idx].values
        seed_exps = []
        for s in seeds:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=feature_names,
                categorical_features=categorical_indices,
                class_names=["<=50K", ">50K"],
                mode="classification",
                random_state=s,
            )
            exp_res = explainer.explain_instance(
                row_vals,
                model.predict_proba,
                num_features=len(feature_names),
                num_samples=LIME_NUM_SAMPLES,
            )
            w = np.zeros(len(feature_names))
            exp_map = dict(exp_res.as_map().get(1, exp_res.as_map().get(0, [])))
            for fi, weight in exp_map.items():
                w[fi] = abs(weight)
            seed_exps.append(w)

        pair_rhos = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                r, _ = stats.spearmanr(seed_exps[i], seed_exps[j])
                if not np.isnan(r):
                    pair_rhos.append(r)
        if pair_rhos:
            instance_rhos.append(float(np.mean(pair_rhos)))

    results = {
        "n_instances": n_eval,
        "seeds": seeds,
        "num_samples": LIME_NUM_SAMPLES,
        "mean_stability": float(np.mean(instance_rhos)),
        "std_stability": float(np.std(instance_rhos)),
        "median_stability": float(np.median(instance_rhos)),
        "values": instance_rhos,
    }
    print(f"  LIME Seed Stability across {len(seeds)} seeds on {n_eval} instances:")
    print(f"  Mean Spearman rho: {results['mean_stability']:.3f} ± {results['std_stability']:.3f} (median {results['median_stability']:.3f})")
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

    all_results = {}
    t_start = time.time()

    # Phase 1: Data
    X_train, X_test, y_train, y_test, dataset_info, label_encoders = load_adult_dataset()
    all_results["dataset_info"] = dataset_info

    # Phase 2: Model
    model, model_metrics = train_model(X_train, y_train, X_test, y_test)
    all_results["model_metrics"] = model_metrics

    # Phase 3: Select samples
    X_sample, y_sample = select_explanation_samples(model, X_test, y_test)

    # Phase 4: Generate explanations
    shap_vals = generate_shap_explanations(model, X_sample, X_train)
    lime_vals = generate_lime_explanations(model, X_sample, X_train, dataset_info)
    perm_vals = generate_permutation_importance(model, X_sample, dataset_info)

    # Save raw explanations
    np.save(os.path.join(RAW_DIR, "shap_values.npy"), shap_vals)
    np.save(os.path.join(RAW_DIR, "lime_values.npy"), lime_vals)
    np.save(os.path.join(RAW_DIR, "perm_values.npy"), perm_vals)
    X_sample.to_csv(os.path.join(RAW_DIR, "X_sample.csv"), index=False)
    y_sample.to_csv(os.path.join(RAW_DIR, "y_sample.csv"), index=False)

    # Phase 5: Agreement
    agreement_results = compute_agreement(
        shap_vals, lime_vals, perm_vals, dataset_info["feature_names"]
    )
    all_results["agreement"] = {
        metric: {pair: {k: v for k, v in data.items() if k != "values"}
                 for pair, data in pairs.items()}
        for metric, pairs in agreement_results.items()
    }

    # Phase 6: Faithfulness
    mask_values = compute_masking_values(X_train, dataset_info)
    faithfulness_results = compute_faithfulness(
        model, X_sample, shap_vals, lime_vals, perm_vals, dataset_info, mask_values
    )
    all_results["faithfulness"] = {
        metric_type: {
            method: {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                     for k, v in method_data.items()}
            for method, method_data in type_data.items()
        }
        for metric_type, type_data in faithfulness_results.items()
    }

    # Phase 7: Agreement vs Faithfulness
    agree_faith_results = compute_agreement_vs_faithfulness(
        agreement_results, faithfulness_results,
        shap_vals, lime_vals, perm_vals, model, X_sample, dataset_info, mask_values
    )
    all_results["agreement_vs_faithfulness"] = {
        pair: {k: v for k, v in data.items()
               if k not in ["agreement_values", "faithfulness_values", "sensitivity_analysis"]}
        for pair, data in agree_faith_results.items()
    }
    all_results["rq3_sensitivity"] = {
        pair: data.get("sensitivity_analysis", {})
        for pair, data in agree_faith_results.items()
    }

    # Phase 8: Stability (Input Perturbation)
    stability_results = compute_stability(model, X_sample, X_train, dataset_info)
    all_results["stability"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "values"}
        if isinstance(v, dict) else v
        for k, v in stability_results.items()
    }

    # Phase 8B: LIME Seed Sensitivity Analysis
    lime_seed_results = evaluate_lime_seed_sensitivity(
        model, X_sample, X_train, dataset_info, n_instances=25
    )
    all_results["lime_seed_sensitivity"] = {
        k: v for k, v in lime_seed_results.items() if k != "values"
    }

    # Phase 9: Metric agreement
    metric_agree_results = compute_metric_agreement(faithfulness_results)
    all_results["metric_agreement"] = metric_agree_results

    # Save all results
    with open(os.path.join(PROCESSED_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save raw detailed results with per-instance values
    raw_detailed = {
        "agreement": agreement_results,
        "faithfulness": faithfulness_results,
        "agreement_vs_faithfulness": agree_faith_results,
        "rq3_sensitivity": all_results["rq3_sensitivity"],
        "stability": stability_results,
        "lime_seed_sensitivity": lime_seed_results,
    }
    with open(os.path.join(RAW_DIR, "detailed_results.json"), "w") as f:
        json.dump(raw_detailed, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETED in {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {RESULTS_DIR}")

    return all_results, raw_detailed, agreement_results, faithfulness_results, \
           agree_faith_results, stability_results, lime_seed_results


if __name__ == "__main__":
    main()

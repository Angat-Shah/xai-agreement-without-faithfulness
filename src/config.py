"""
XAI Agreement and Faithfulness Study - Configuration
"""

import os

# Random seed for reproducibility
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
PROCESSED_DIR = os.path.join(RESULTS_DIR, "processed")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Dataset
DATASET_NAME = "adult"
TEST_SIZE = 0.2

# XAI evaluation
N_EXPLAIN_SAMPLES = 500  # Number of test instances to explain
LIME_NUM_SAMPLES = 5000  # LIME perturbation samples
LIME_NUM_FEATURES = None  # Use all features

# Faithfulness evaluation
FAITHFULNESS_K_VALUES = [1, 3, 5]  # Top-k features to mask/keep

# Stability evaluation
STABILITY_NOISE_SCALE = 0.01  # σ as fraction of feature range
STABILITY_N_PERTURBATIONS = 10  # Number of perturbations per instance

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# Create directories
for d in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

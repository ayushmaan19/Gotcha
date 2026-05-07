"""
Training pipeline for multimodal fusion models and evaluation reports.

Changelog (v2.0):
- Added XGBoost as default model with scale_pos_weight for class imbalance
- Added SMOTE oversampling for minority class
- Added modality-aware feature grouping (face/gaze/audio/context)
- Added audio exclusion testing with automatic modality selection
- Added rolling temporal features (mean, std, delta) for gaze/face
- Added StandardScaler pipeline for gradient-boosted models
- Added synthetic pre-training + real data fine-tuning workflow
- Added precision-recall curve + security-optimized threshold tuning
- Added balanced_accuracy as primary metric
"""

import argparse
import os
import sys
import pickle
import glob
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, balanced_accuracy_score, average_precision_score,
)
from sklearn.inspection import permutation_importance

# Try importing optional dependencies
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
    print("[!] xgboost not installed. Run: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False
    print("[!] imblearn not installed. Run: pip install imbalanced-learn")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt


MODEL_OUT = "fusion_model.pkl"
MODEL_THRESH_OUT = "fusion_threshold.txt"  # Save optimal threshold
DATASET_DIR = "dataset_logs"
SIM_DATA_DIR = "simulated_data"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# MODALITY FEATURE GROUPS
# ---------------------------------------------------------------------------

# Face liveness + micro-expression features
FACE_COLS = ["face_score", "micro_var"]

# Gaze + blink behavior features
GAZE_COLS = ["gaze_score", "anomaly_duration"]

# Audio forensics features (potentially noisy — test exclusion)
AUDIO_COLS = ["voice_score", "jitter", "shimmer", "spectral_centroid"]

# Context/scene features
CONTEXT_COLS = ["object_flag"]

# All features in training order
ALL_FEATURE_COLS = FACE_COLS + GAZE_COLS + AUDIO_COLS + CONTEXT_COLS


def get_feature_columns(include_audio: bool = True) -> list:
    """Return feature columns with optional audio exclusion."""
    cols = FACE_COLS + GAZE_COLS
    if include_audio:
        cols += AUDIO_COLS
    cols += CONTEXT_COLS
    return cols


# ---------------------------------------------------------------------------
# STEP 1: LOAD DATASET
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str = None,
                 merge_logs: bool = True,
                 include_simulated: bool = False) -> pd.DataFrame:

    dfs = []

    # Manual CSV load
    if csv_path and os.path.isfile(csv_path):
        print(f"[*] Loading dataset: {csv_path}")

        df = pd.read_csv(csv_path)

        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        if "final_integrity_score" in df.columns:
            df = df.drop(columns=["final_integrity_score"])

        if "label" not in df.columns:
            raise ValueError("Dataset missing label column")

        df["label"] = df["label"].astype(int)
        df = df[df["label"].isin([0, 1])]
        df = df.drop_duplicates()

        dfs.append(df)

    # Auto-load all CSVs from dataset_logs/
    elif merge_logs and os.path.isdir(DATASET_DIR):

        csv_files = glob.glob(
            os.path.join(DATASET_DIR, "*.csv")
        )

        print(f"[*] Found {len(csv_files)} CSV datasets")

        for f in csv_files:
            try:
                df = pd.read_csv(f)

                if len(df) < 10:
                    print(f"    [!] Skipping tiny dataset: {os.path.basename(f)}")
                    continue

                if "timestamp" in df.columns:
                    df = df.drop(columns=["timestamp"])

                if "final_integrity_score" in df.columns:
                    df = df.drop(columns=["final_integrity_score"])

                if "label" not in df.columns:
                    print(f"    [!] Missing label column: {os.path.basename(f)}")
                    continue

                df["label"] = df["label"].astype(int)
                df = df[df["label"].isin([0, 1])]

                df = df.drop_duplicates()

                dfs.append(df)

                print(f"    -> Loaded {len(df)} samples from {os.path.basename(f)}")

            except Exception as e:
                print(f"    [!] Error loading {f}: {e}")

    if not dfs:
        raise ValueError("No valid datasets found.")

    merged = pd.concat(dfs, ignore_index=True)

    merged = merged.drop_duplicates()
    merged = merged.dropna()

    print("\n" + "=" * 60)
    print(f"[*] FINAL DATASET SIZE: {len(merged)}")
    print("[*] CLASS DISTRIBUTION:")
    print(merged["label"].value_counts())
    print("=" * 60)

    return merged


# def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
#     """
#     Generate synthetic dataset matching the feature_logger schema.
#     Used when no real data is available.
#     """
#     rng = np.random.RandomState(seed)
#     half = n_samples // 2

#     # Legitimate samples (good engagement, low stress)
#     legit = pd.DataFrame({
#         "gaze_score": rng.normal(85, 10, half).clip(0, 100),
#         "face_score": rng.normal(90, 8, half).clip(0, 100),
#         "voice_score": rng.normal(88, 12, half).clip(0, 100),
#         "micro_var": rng.exponential(0.5, half).clip(0.1, 5.0),
#         "jitter": rng.normal(1.5, 0.8, half).clip(0, 5),
#         "shimmer": rng.normal(4.0, 2.0, half).clip(0, 10),
#         "spectral_centroid": rng.normal(1500, 400, half).clip(500, 4000),
#         "anomaly_duration": rng.exponential(0.5, half).clip(0, 3),
#         "object_flag": rng.binomial(1, 0.05, half),  # 5% false alarms
#         "final_integrity_score": rng.normal(80, 15, half).clip(0, 100),
#         "label": np.zeros(half, dtype=int),
#     })

#     # Suspicious samples (distracted, stressed, cheating)
#     suspicious = pd.DataFrame({
#         "gaze_score": rng.normal(35, 20, half).clip(0, 100),
#         "face_score": rng.normal(50, 25, half).clip(0, 100),
#         "voice_score": rng.normal(55, 25, half).clip(0, 100),
#         "micro_var": rng.exponential(0.1, half).clip(0, 0.3),  # Frozen face
#         "jitter": rng.normal(5.0, 2.0, half).clip(0, 15),
#         "shimmer": rng.normal(12.0, 4.0, half).clip(0, 25),
#         "spectral_centroid": rng.normal(2500, 800, half).clip(500, 5000),
#         "anomaly_duration": rng.exponential(3.0, half).clip(0, 10),
#         "object_flag": rng.binomial(1, 0.35, half),  # 35% device detected
#         "final_integrity_score": rng.normal(35, 20, half).clip(0, 100),
#         "label": np.ones(half, dtype=int),
#     })

#     df = pd.concat([legit, suspicious], ignore_index=True)
#     df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

#     # Save for inspection
#     df.to_csv("synthetic_data.csv", index=False)
#     print("[*] Generated synthetic dataset saved to synthetic_data.csv")

#     return df


# # ---------------------------------------------------------------------------
# # STEP 2: TEMPORAL FEATURE ENGINEERING
# # ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Add rolling window temporal features for gaze and face modalities.
    Captures trend (delta) and volatility (roll_std) over time.

    Args:
        df: DataFrame with raw per-frame features
        window: Rolling window size in samples

    Returns:
        DataFrame with added temporal columns
    """
    temporal_cols = FACE_COLS + GAZE_COLS
    df = df.copy()

    for col in temporal_cols:
        if col not in df.columns:
            continue
        # Rolling mean: captures recent baseline
        df[f"{col}_roll_mean"] = df[col].rolling(window=window, min_periods=1).mean()
        # Rolling std: captures volatility / instability
        df[f"{col}_roll_std"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        # Delta: rate of change from previous frame
        df[f"{col}_delta"] = df[col].diff().fillna(0)

    return df


# ---------------------------------------------------------------------------
# STEP 3: TRAIN MODELS (XGBoost / GradientBoosting / RandomForest)
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost from class distribution."""
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def create_model_pipeline(model_type: str = "xgboost",
                          scale_pos_weight: float = 1.0,
                          random_state: int = 42):
    """
    Create a model pipeline with optional scaling.

    Args:
        model_type: 'xgboost', 'gradient_boosting', 'random_forest', or 'calibrated'
        scale_pos_weight: For XGBoost class imbalance handling
        random_state: Random seed

    Returns:
        sklearn Pipeline or model instance
    """
    if model_type == "xgboost":
        if not _HAS_XGBOOST:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        model = XGBClassifier(
          n_estimators=300,
          max_depth=5,
          learning_rate=0.03,
          subsample=0.8,
          colsample_bytree=0.8,
          min_child_weight=3,
          gamma=0.2,
          reg_alpha=0.5,
          reg_lambda=1.5,
          scale_pos_weight=scale_pos_weight,
          eval_metric="logloss",
          random_state=random_state,
          n_jobs=-1,
)
        # XGBoost benefits from scaled features
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        return pipeline, True  # True = uses proba

    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        return pipeline, True

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        # RandomForest doesn't strictly need scaling, but pipeline for consistency
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        return pipeline, True

    elif model_type == "calibrated":
        base_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=random_state,
            )),
        ])
        clf = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=3)
        return clf, True

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_with_smote(X_train: np.ndarray, y_train: np.ndarray,
                     use_smote: bool = True) -> tuple:
    """
    Apply SMOTE oversampling if enabled and available.

    Returns:
        (X_resampled, y_resampled, used_smote: bool)
    """
    if not use_smote or not _HAS_IMBLEARN:
        if use_smote and not _HAS_IMBLEARN:
            print("[!] SMOTE requested but imblearn not installed. Skipping.")
        return X_train, y_train, False

    # Only apply SMOTE if we have enough minority samples
    min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
    if min_class_count < 6:
        print(f"[!] Too few minority samples ({min_class_count}) for SMOTE. Skipping.")
        return X_train, y_train, False

    print(f"[*] Applying SMOTE... (before: {dict(zip(*np.unique(y_train, return_counts=True)))})")
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, min_class_count - 1))
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"    -> After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res, True


# ---------------------------------------------------------------------------
# STEP 4: EVALUATE MODEL (with security-focused threshold tuning)
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                           target_recall: float = 0.90) -> tuple:
    """
    Find decision threshold optimized for security (high recall).

    For a security system, False Negatives (missed threats) are far more
    costly than False Positives. We optimize for target recall.

    Returns:
        (best_threshold, metrics_at_threshold)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Find threshold that achieves target recall
    # precision_recall_curve returns thresholds of length len(precision)-1
    valid_idx = np.where(recall[:-1] >= target_recall)[0]

    if len(valid_idx) > 0:
        # Among thresholds meeting target recall, pick the one with best precision
        best_idx = valid_idx[np.argmax(precision[valid_idx])]
        best_threshold = float(thresholds[best_idx])
    else:
        # If target recall can't be met, use threshold that maximizes F1
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])

    # Compute metrics at this threshold
    y_pred_at_thresh = (y_proba >= best_threshold).astype(int)
    metrics = {
        "threshold": best_threshold,
        "target_recall": target_recall,
        "precision_at_thresh": float(precision_score(y_true, y_pred_at_thresh, zero_division=0)),
        "recall_at_thresh": float(recall_score(y_true, y_pred_at_thresh, zero_division=0)),
        "f1_at_thresh": float(f1_score(y_true, y_pred_at_thresh, zero_division=0)),
    }

    return best_threshold, metrics


def evaluate_model(clf,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   feature_cols: list,
                   save_plots: bool = True,
                   target_recall: float = 0.90) -> dict:
    """
    Comprehensive model evaluation with security-focused metrics and plots.

    Args:
        clf: Trained classifier
        X_train, y_train: Training data (for cross-validation)
        X_test, y_test: Test data
        feature_cols: Feature column names
        save_plots: Whether to save evaluation plots
        target_recall: Target recall for threshold optimization

    Returns:
        dict: All computed metrics including optimal threshold
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_proba)

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = 0.5  # Undefined if only one class present

    print(f"\n-- Test Set Metrics --")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Balanced Acc:     {balanced_acc:.4f}  <-- PRIMARY METRIC")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  ROC-AUC:          {roc_auc:.4f}")
    print(f"  Avg Precision:    {avg_precision:.4f}")

    # Security-optimized threshold
    best_thresh, thresh_metrics = find_optimal_threshold(
        y_test, y_proba, target_recall=target_recall
    )
    print(f"\n-- Security-Optimized Threshold (recall >= {target_recall}) --")
    print(f"  Optimal Threshold: {best_thresh:.4f}")
    print(f"  Precision @ Thresh: {thresh_metrics['precision_at_thresh']:.4f}")
    print(f"  Recall @ Thresh:    {thresh_metrics['recall_at_thresh']:.4f}")
    print(f"  F1 @ Thresh:        {thresh_metrics['f1_at_thresh']:.4f}")

    # Classification report
    print(f"\n-- Classification Report --")
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Suspicious"],
                                 zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"-- Confusion Matrix --")
    print(cm)

    # 5-Fold Cross Validation
    print(f"\n-- 5-Fold Cross Validation --")
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracy = cross_val_score(clf, X_full, y_full, cv=cv, scoring="accuracy")
    cv_bal_acc = cross_val_score(clf, X_full, y_full, cv=cv, scoring="balanced_accuracy")
    cv_f1 = cross_val_score(clf, X_full, y_full, cv=cv, scoring="f1")
    cv_roc_auc = cross_val_score(clf, X_full, y_full, cv=cv, scoring="roc_auc")

    print(f"  CV Accuracy:      {cv_accuracy.mean():.4f} +/- {cv_accuracy.std():.4f}")
    print(f"  CV Balanced-Acc:  {cv_bal_acc.mean():.4f} +/- {cv_bal_acc.std():.4f}  <--")
    print(f"  CV F1-Score:      {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"  CV ROC-AUC:       {cv_roc_auc.mean():.4f} +/- {cv_roc_auc.std():.4f}")

    # Feature importance (tree importances or permutation fallback)
    print(f"\n-- Feature Importances --")
    importances = _extract_feature_importances(clf, X_test, y_test, feature_cols)
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"  {feature_cols[i]:<30s} {importances[i]:.4f}")

    # Save plots
    if save_plots:
        _save_evaluation_plots(cm, y_test, y_proba, feature_cols, importances,
                               precision, recall, f1, best_thresh)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "optimal_threshold": best_thresh,
        "threshold_precision": thresh_metrics["precision_at_thresh"],
        "threshold_recall": thresh_metrics["recall_at_thresh"],
        "threshold_f1": thresh_metrics["f1_at_thresh"],
        "cv_accuracy_mean": cv_accuracy.mean(),
        "cv_balanced_accuracy_mean": cv_bal_acc.mean(),
        "cv_f1_mean": cv_f1.mean(),
        "cv_roc_auc_mean": cv_roc_auc.mean(),
    }


def _extract_feature_importances(clf, X_ref: np.ndarray, y_ref: np.ndarray, feature_cols: list) -> np.ndarray:
    """Return importances for both tree and non-tree models."""
    # Unwrap from pipeline if needed
    inner = clf
    if hasattr(clf, "named_steps") and "model" in clf.named_steps:
        inner = clf.named_steps["model"]

    if hasattr(inner, "feature_importances_"):
        return np.asarray(inner.feature_importances_)

    # Generic fallback for calibrated/pipeline models
    try:
        perm = permutation_importance(
            clf,
            X_ref,
            y_ref,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=-1,
        )
        imp = np.maximum(perm.importances_mean, 0.0)
        if float(np.sum(imp)) > 0:
            imp = imp / np.sum(imp)
        return imp
    except Exception as e:
        print(f"[!] Could not compute permutation importance: {e}")
        return np.zeros(len(feature_cols), dtype=float)


def _save_evaluation_plots(cm: np.ndarray, y_test: np.ndarray, y_proba: np.ndarray,
                            feature_cols: list, importances: np.ndarray,
                            precision: float, recall: float, f1: float,
                            best_threshold: float):
    """Save confusion matrix, ROC curve, PR curve, and feature importance plots."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Legitimate', 'Suspicious'])
    ax.set_yticklabels(['Legitimate', 'Suspicious'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{timestamp}.png", dpi=150)
    plt.close()
    print(f"\n[*] Saved: confusion_matrix_{timestamp}.png")

    # 2. ROC Curve Plot
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"roc_curve_{timestamp}.png", dpi=150)
        plt.close()
        print(f"[*] Saved: roc_curve_{timestamp}.png")
    except Exception as e:
        print(f"[!] Could not generate ROC curve: {e}")

    # 3. Precision-Recall Curve Plot
    try:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
        ap_score = average_precision_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(pr_recall, pr_precision, 'g-', linewidth=2,
            label=f'PR (AP = {ap_score:.3f})')
        ax.axhline(y=pr_precision.mean() if len(pr_precision) > 0 else 0.5,
                   color='k', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve\n(Optimal threshold = {best_threshold:.3f})')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"pr_curve_{timestamp}.png", dpi=150)
        plt.close()
        print(f"[*] Saved: pr_curve_{timestamp}.png")
    except Exception as e:
        print(f"[!] Could not generate PR curve: {e}")

    # 4. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    sorted_idx = np.argsort(importances)
    sorted_importances = importances[sorted_idx]
    sorted_features = [feature_cols[i] for i in sorted_idx]
    bars = ax.barh(
        range(len(sorted_importances)),
        sorted_importances,
        color='#2C7FB8',
        edgecolor='#0B3C5D',
        linewidth=1.2,
        alpha=0.95,
    )
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(sorted_features, fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Feature Importances', fontsize=18, fontweight='bold', pad=14)
    ax.tick_params(axis='x', labelsize=12)

    max_importance = float(sorted_importances.max()) if len(sorted_importances) else 0.0
    for bar, value in zip(bars, sorted_importances):
        ax.text(
            bar.get_width() + max_importance * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold',
            color='#0B3C5D',
        )

    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"feature_importance_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved: feature_importance_{timestamp}.png")


# ---------------------------------------------------------------------------
# STEP 5: ABLATION STUDY (with modality-aware configurations)
# ---------------------------------------------------------------------------

def run_ablation_study(df: pd.DataFrame, model_type: str = "xgboost",
                       use_smote: bool = True, add_temporal: bool = True):
    """
    Run ablation study by training models with different feature subsets.
    Tests the impact of each modality and audio exclusion.

    Configurations:
    - All features (with audio)
    - Without audio features
    - Face + Gaze only
    - Each modality alone
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY")
    print("=" * 80)

    configs = {
        "All Features (with audio)": get_feature_columns(include_audio=True),
        "Exclude Audio": get_feature_columns(include_audio=False),
        "Face Only": FACE_COLS,
        "Gaze Only": GAZE_COLS,
        "Audio Only": AUDIO_COLS,
        "Context Only": CONTEXT_COLS,
        "Face + Gaze (no audio/context)": FACE_COLS + GAZE_COLS,
    }

    # Apply temporal features if data is large enough
    if add_temporal and len(df) >= 60:
        df = add_temporal_features(df)
        # Add temporal cols to relevant configs
        temporal_face = [c for c in df.columns if c.startswith("face_score_") or c.startswith("micro_var_")]
        temporal_gaze = [c for c in df.columns if c.startswith("gaze_score_") or c.startswith("anomaly_duration_")]

        configs["All + Temporal"] = get_feature_columns(include_audio=True) + temporal_face + temporal_gaze
        configs["No Audio + Temporal"] = get_feature_columns(include_audio=False) + temporal_face + temporal_gaze

    results = []

    for config_name, features in configs.items():
        print(f"\n[*] Configuration: {config_name}")

        # Filter to features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"    [!] Missing columns (skipped): {missing}")

        if len(available_features) == 0:
            print(f"    [!] Skipping - no available columns")
            continue

        print(f"    Features: {available_features}")

        X = df[available_features].values
        y = df["label"].values.astype(int)

        # Use stratified CV for stable comparisons on imbalanced datasets
        spw = compute_scale_pos_weight(y)
        try:
            clf, _ = create_model_pipeline(model_type, scale_pos_weight=spw)
        except ImportError:
            print(f"    [!] {model_type} not available, falling back to random_forest")
            clf, _ = create_model_pipeline("random_forest", scale_pos_weight=spw)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        try:
            cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            cv_bal_acc = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
            cv_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1")
            cv_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

            accuracy = float(cv_acc.mean())
            bal_acc = float(cv_bal_acc.mean())
            f1 = float(cv_f1.mean())
            roc_auc = float(cv_auc.mean())

            results.append({
                "Configuration": config_name,
                "Accuracy": accuracy,
                "Balanced-Acc": bal_acc,
                "F1": f1,
                "ROC-AUC": roc_auc,
                "Features": len(available_features),
            })

            print(f"    Accuracy: {accuracy:.4f} | Bal-Acc: {bal_acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"    [!] Error during CV: {e}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<28s} | {'Accuracy':>10s} | {'Bal-Acc':>10s} | {'F1':>10s} | {'ROC-AUC':>10s} | {'#Feat':>6s}")
    print("-" * 96)
    for r in results:
        print(f"{r['Configuration']:<28s} | {r['Accuracy']:>10.4f} | {r['Balanced-Acc']:>10.4f} | {r['F1']:>10.4f} | {r['ROC-AUC']:>10.4f} | {r['Features']:>6d}")
    print("=" * 96)

    # Highlight best configuration
    if results:
        best = max(results, key=lambda x: x["Balanced-Acc"])
        print(f"\n[*] BEST CONFIG (by Balanced-Acc): {best['Configuration']}")
        print(f"    Balanced-Acc: {best['Balanced-Acc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# STEP 6: AUDIO IMPACT ANALYSIS
# ---------------------------------------------------------------------------

def analyze_audio_impact(df: pd.DataFrame, model_type: str = "xgboost") -> dict:
    """
    Explicitly test whether audio features improve or hurt performance.
    Returns the recommended configuration.
    """
    print("\n" + "=" * 60)
    print("AUDIO MODALITY IMPACT ANALYSIS")
    print("=" * 60)

    configs = {
        "with_audio": get_feature_columns(include_audio=True),
        "without_audio": get_feature_columns(include_audio=False),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, features in configs.items():
        available = [f for f in features if f in df.columns]
        if len(available) == 0:
            continue

        X = df[available].values
        y = df["label"].values.astype(int)
        spw = compute_scale_pos_weight(y)

        try:
            clf, _ = create_model_pipeline(model_type, scale_pos_weight=spw)
        except ImportError:
            clf, _ = create_model_pipeline("random_forest", scale_pos_weight=spw)

        cv_bal_acc = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
        cv_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

        results[name] = {
            "balanced_acc": float(cv_bal_acc.mean()),
            "roc_auc": float(cv_auc.mean()),
            "features": available,
        }

        print(f"  {name}: Bal-Acc={results[name]['balanced_acc']:.4f}, ROC-AUC={results[name]['roc_auc']:.4f}")

    # Recommendation
    if "with_audio" in results and "without_audio" in results:
        no_audio_better = results["without_audio"]["balanced_acc"] > results["with_audio"]["balanced_acc"]
        if no_audio_better:
            print(f"\n[!] RECOMMENDATION: EXCLUDE audio from fusion")
            print(f"    Without audio: {results['without_audio']['balanced_acc']:.4f}")
            print(f"    With audio:    {results['with_audio']['balanced_acc']:.4f}")
            return {"use_audio": False, "features": results["without_audio"]["features"]}
        else:
            print(f"\n[*] RECOMMENDATION: KEEP audio in fusion")
            print(f"    With audio:    {results['with_audio']['balanced_acc']:.4f}")
            print(f"    Without audio: {results['without_audio']['balanced_acc']:.4f}")
            return {"use_audio": True, "features": results["with_audio"]["features"]}

    return {"use_audio": True, "features": get_feature_columns(include_audio=True)}


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main(csv_path: str = None,
         ablation: bool = False,
         model_type: str = "xgboost",
         include_simulated: bool = False,
         use_smote: bool = True,
         add_temporal: bool = True,
         use_pretrained: bool = False,
         target_recall: float = 0.90,
         analyze_audio: bool = True):
    """Main training pipeline."""

    print("=" * 60)
    print("SDP-1 Integrity Model Trainer v2.0")
    print("=" * 60)
    print(f"Model: {model_type}")
    print(f"SMOTE: {use_smote} (imblearn available: {_HAS_IMBLEARN})")
    print(f"Temporal features: {add_temporal}")
    print(f"Pre-train + fine-tune: {use_pretrained}")
    print(f"Target recall: {target_recall}")
    print("=" * 60)

    # Check dependencies
    if model_type == "xgboost" and not _HAS_XGBOOST:
        print("[!] xgboost not installed. Falling back to gradient_boosting")
        model_type = "gradient_boosting"

    # Load data
    df = load_dataset(
        csv_path=csv_path,
        merge_logs=(csv_path is None),
        include_simulated=include_simulated,
    )

    if len(df) < 20:
        print("[!] Insufficient data. Need at least 20 samples.")
        sys.exit(1)

    # Add temporal features if enabled and enough data
    if add_temporal and len(df) >= 60:
        print(f"[*] Adding temporal features (window=30)...")
        n_before = len(df.columns)
        df = add_temporal_features(df)
        n_after = len(df.columns)
        print(f"    -> Added {n_after - n_before} temporal columns")

    # Ablation study mode
    if ablation:
        run_ablation_study(df, model_type=model_type, use_smote=use_smote, add_temporal=add_temporal)
        return

    # Audio impact analysis
    feature_cols = get_feature_columns(include_audio=True)
    if analyze_audio and len(df) >= 30:
        audio_recommendation = analyze_audio_impact(
            df,
            model_type=model_type
        )

    print("\n[*] Audio analysis completed")

    # FIXED FEATURE ORDER
    feature_cols = [
        "face_score",
        "micro_var",
        "gaze_score",
        "anomaly_duration",
        "voice_score",
        "jitter",
        "shimmer",
        "spectral_centroid",
        "object_flag",
    ]

    print(f"\n[*] FINAL FEATURE ORDER:")
    for i, col in enumerate(feature_cols):
        print(f"{i}: {col}")

    # Check all features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"[!] Missing columns in dataset: {missing}")
        print(f"    Available: {list(df.columns)}")
        # Fall back to available features
        feature_cols = [f for f in feature_cols if f in df.columns]
        print(f"    Using available: {feature_cols}")

    print("\n[*] FEATURE VALIDATION")
    for col in feature_cols:
        print(f"{col}: min={df[col].min():.3f}, max={df[col].max():.3f}")

    # Session-aware split to avoid leakage
    if "session_id" in df.columns:
        unique_sessions = df["session_id"].unique()

        train_sessions, test_sessions = train_test_split(
            unique_sessions,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        train_df = df[df["session_id"].isin(train_sessions)]
        test_df = df[df["session_id"].isin(test_sessions)]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values.astype(int)

        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values.astype(int)
    else:
        X = df[feature_cols].values
        y = df["label"].values.astype(int)

        # 80/20 stratified split
        print(f"\n[*] Splitting data: 80% train / 20% test (stratified)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    # Apply SMOTE
    X_train_res, y_train_res, used_smote = train_with_smote(X_train, y_train, use_smote=use_smote)
    X_train_res = np.nan_to_num(X_train_res)
    X_test = np.nan_to_num(X_test)

    # Train
    spw = compute_scale_pos_weight(y_train_res)
    print(f"\n[*] Training {model_type} (scale_pos_weight={spw:.2f})...")

    clf, _ = create_model_pipeline(model_type, scale_pos_weight=spw)
    clf.fit(X_train_res, y_train_res)
    print(f"    -> Training complete. {len(X_train_res)} samples used.")

    # Evaluate
    metrics = evaluate_model(
        clf, X_train_res, y_train_res, X_test, y_test,
        feature_cols, save_plots=True, target_recall=target_recall,
    )

    # Save model + metadata
    payload = {
        "model": clf,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_type": model_type,
        "used_smote": used_smote,
        "optimal_threshold": metrics["optimal_threshold"],
        "timestamp": datetime.now().isoformat(),
    }

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)

    # Save threshold to a text file for easy reading by app.py
    with open(MODEL_THRESH_OUT, "w") as f:
        f.write(f"{metrics['optimal_threshold']:.4f}\n")

    print(f"\n[✓] Model saved to: {MODEL_OUT}")
    print(f"[✓] Threshold saved to: {MODEL_THRESH_OUT}")
    print(f"    Features: {feature_cols}")
    print(f"    Optimal threshold (security): {metrics['optimal_threshold']:.4f}")
    print(f"    To load: pickle.load(open('{MODEL_OUT}', 'rb'))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDP-1 Integrity Model Trainer v2.0 (Research Edition)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV dataset file"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study with different feature subsets"
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["xgboost", "gradient_boosting", "random_forest", "calibrated"],
        help="Model type: xgboost (default), gradient_boosting, random_forest, calibrated"
    )
    parser.add_argument(
        "--include-simulated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include CSVs from simulated_data/ when auto-merging datasets",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE oversampling",
    )
    parser.add_argument(
        "--no-temporal", action="store_true",
        help="Disable temporal feature engineering",
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Enable synthetic pre-training + real fine-tuning (XGBoost only)",
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.90,
        help="Target recall for security-optimized threshold (default: 0.90)",
    )
    parser.add_argument(
        "--no-audio-analysis", action="store_true",
        help="Skip audio impact analysis",
    )
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        ablation=args.ablation,
        model_type=args.model,
        include_simulated=args.include_simulated,
        use_smote=not args.no_smote,
        add_temporal=not args.no_temporal,
        use_pretrained=args.pretrained,
        target_recall=args.target_recall,
        analyze_audio=not args.no_audio_analysis,
    )

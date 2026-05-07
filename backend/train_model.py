"""Training pipeline for multimodal fusion models and evaluation reports."""

import argparse
import os
import sys
import pickle
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt


MODEL_OUT = "fusion_model.pkl"
DATASET_DIR = "dataset_logs"
SIM_DATA_DIR = "simulated_data"
RANDOM_STATE = 42


#  STEP 1: LOAD DATASET

def load_dataset(csv_path: str = None,
                 merge_logs: bool = True,
                 include_simulated: bool = True) -> pd.DataFrame:
    """
    Load dataset from CSV file(s).
    
    Args:
        csv_path: Path to specific CSV file (optional)
        merge_logs: If True, merge all CSVs from dataset_logs/ directory
        include_simulated: If True, also merge CSVs from simulated_data/
    
    Returns:
        pd.DataFrame with features and labels
    """
    dfs = []
    
    # Load specific CSV if provided
    if csv_path and os.path.isfile(csv_path):
        print(f"[*] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        dfs.append(df)
    
    # Merge all logs from dataset_logs/ if enabled
    if merge_logs and os.path.isdir(DATASET_DIR):
        log_files = glob.glob(os.path.join(DATASET_DIR, "session_*.csv"))
        print(f"[*] Found {len(log_files)} session logs in {DATASET_DIR}/")
        for f in log_files:
            try:
                df = pd.read_csv(f)
                # Skip near-empty logs (often interrupted sessions)
                if len(df) > 1:
                    dfs.append(df)
                    print(f"    → Loaded {len(df)} samples from {os.path.basename(f)}")
            except Exception as e:
                print(f"    [!] Error loading {f}: {e}")

    # Merge simulated datasets if enabled
    if merge_logs and include_simulated and os.path.isdir(SIM_DATA_DIR):
        sim_files = glob.glob(os.path.join(SIM_DATA_DIR, "*.csv"))
        print(f"[*] Found {len(sim_files)} simulated CSVs in {SIM_DATA_DIR}/")
        for f in sim_files:
            try:
                df = pd.read_csv(f)
                if len(df) > 1:
                    dfs.append(df)
                    print(f"    → Loaded {len(df)} samples from {os.path.basename(f)}")
            except Exception as e:
                print(f"    [!] Error loading {f}: {e}")
    
    if not dfs:
        print("[!] No datasets found. Generating synthetic data...")
        return generate_synthetic_dataset()
    
    # Merge all dataframes
    merged = pd.concat(dfs, ignore_index=True)
    
    # Clean up: drop rows with missing labels
    if 'label' in merged.columns:
        merged = merged.dropna(subset=['label'])
        merged['label'] = merged['label'].astype(int)
    
    print(f"[*] Total dataset size: {len(merged)} samples")
    print(f"    Class distribution:\n{merged['label'].value_counts().to_string()}\n")
    
    return merged


def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset matching the feature_logger schema.
    Used when no real data is available.
    """
    rng = np.random.RandomState(seed)
    half = n_samples // 2
    
    # Legitimate samples (good engagement, low stress)
    legit = pd.DataFrame({
        "gaze_score": rng.normal(85, 10, half).clip(0, 100),
        "face_score": rng.normal(90, 8, half).clip(0, 100),
        "voice_score": rng.normal(88, 12, half).clip(0, 100),
        "micro_var": rng.exponential(0.5, half).clip(0.1, 5.0),
        "jitter": rng.normal(1.5, 0.8, half).clip(0, 5),
        "shimmer": rng.normal(4.0, 2.0, half).clip(0, 10),
        "spectral_centroid": rng.normal(1500, 400, half).clip(500, 4000),
        "anomaly_duration": rng.exponential(0.5, half).clip(0, 3),
        "object_flag": rng.binomial(1, 0.05, half),  # 5% false alarms
        "final_integrity_score": rng.normal(80, 15, half).clip(0, 100),
        "label": np.zeros(half, dtype=int),
    })
    
    # Suspicious samples (distracted, stressed, cheating)
    suspicious = pd.DataFrame({
        "gaze_score": rng.normal(35, 20, half).clip(0, 100),
        "face_score": rng.normal(50, 25, half).clip(0, 100),
        "voice_score": rng.normal(55, 25, half).clip(0, 100),
        "micro_var": rng.exponential(0.1, half).clip(0, 0.3),  # Frozen face
        "jitter": rng.normal(5.0, 2.0, half).clip(0, 15),
        "shimmer": rng.normal(12.0, 4.0, half).clip(0, 25),
        "spectral_centroid": rng.normal(2500, 800, half).clip(500, 5000),
        "anomaly_duration": rng.exponential(3.0, half).clip(0, 10),
        "object_flag": rng.binomial(1, 0.35, half),  # 35% device detected
        "final_integrity_score": rng.normal(35, 20, half).clip(0, 100),
        "label": np.ones(half, dtype=int),
    })
    
    df = pd.concat([legit, suspicious], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save for inspection
    df.to_csv("synthetic_data.csv", index=False)
    print("[*] Generated synthetic dataset saved to synthetic_data.csv")
    
    return df


def get_feature_columns() -> list:
    """Return the feature columns used for training."""
    return [
        "gaze_score",
        "face_score",
        "voice_score",
        "micro_var",
        "jitter",
        "shimmer",
        "spectral_centroid",
        "anomaly_duration",
        "object_flag",
        "final_integrity_score",
    ]


#  STEP 2: TRAIN RANDOM FOREST

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 200,
                        max_depth: int = 12) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
    
    Returns:
        Trained RandomForestClassifier
    """
    print(f"[*] Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    
    clf.fit(X_train, y_train)
    print(f"    → Training complete. {len(y_train)} samples used.")
    
    return clf


def train_calibrated_logistic_formula(X_train: np.ndarray,
                                      y_train: np.ndarray):
    """
    Train a calibrated logistic fusion model.

    Formula class:
      p(y=1) = sigmoid(beta0 + beta^T x + interaction terms)

    Implemented as a pipeline with pairwise interactions + scaling,
    wrapped in probability calibration for better confidence quality.
    """
    print("[*] Training calibrated logistic fusion formula (with interactions)...")

    base_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])

    clf = CalibratedClassifierCV(
        estimator=base_pipe,
        method="sigmoid",
        cv=3,
    )
    clf.fit(X_train, y_train)
    print(f"    → Training complete. {len(y_train)} samples used.")
    return clf


#  STEP 3: EVALUATE MODEL

def evaluate_model(clf,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   feature_cols: list,
                   save_plots: bool = True) -> dict:
    """
    Comprehensive model evaluation with metrics and plots.
    
    Args:
        clf: Trained classifier
        X_train, y_train: Training data (for cross-validation)
        X_test, y_test: Test data
        feature_cols: Feature column names
        save_plots: Whether to save evaluation plots
    
    Returns:
        dict: All computed metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = 0.5  # Undefined if only one class present
    
    print(f"\n── Test Set Metrics ──")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Classification report
    print(f"\n── Classification Report ──")
    print(classification_report(y_test, y_pred, 
                                 target_names=["Legitimate", "Suspicious"],
                                 zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"── Confusion Matrix ──")
    print(cm)
    
    # 5-Fold Cross Validation
    print(f"\n── 5-Fold Cross Validation ──")
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracy = cross_val_score(clf, X_full, y_full, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(clf, X_full, y_full, cv=cv, scoring="f1")
    cv_roc_auc = cross_val_score(clf, X_full, y_full, cv=cv, scoring="roc_auc")
    
    print(f"  CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  CV F1-Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  CV ROC-AUC:  {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")
    
    # Feature importance (tree importances or permutation fallback)
    print(f"\n── Feature Importances ──")
    importances = _extract_feature_importances(clf, X_test, y_test, feature_cols)
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"  {feature_cols[i]:<25s} {importances[i]:.4f}")
    
    # Save plots
    if save_plots:
        _save_evaluation_plots(cm, y_test, y_proba, feature_cols, importances)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "cv_accuracy_mean": cv_accuracy.mean(),
        "cv_f1_mean": cv_f1.mean(),
        "cv_roc_auc_mean": cv_roc_auc.mean(),
    }


def _extract_feature_importances(clf, X_ref: np.ndarray, y_ref: np.ndarray, feature_cols: list) -> np.ndarray:
    """Return importances for both tree and non-tree models."""
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_)

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
                            feature_cols: list, importances: np.ndarray):
    """Save confusion matrix, ROC curve, and feature importance plots."""
    
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
    
    # Add text annotations
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
    
    # 3. Feature Importance Plot
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
    ax.set_yticklabels(sorted_features, fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_title('Feature Importances (Random Forest)', fontsize=24, fontweight='bold', pad=14)
    ax.tick_params(axis='x', labelsize=14)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight('bold')

    # Add values on each bar to improve readability in paper figures.
    max_importance = float(sorted_importances.max()) if len(sorted_importances) else 0.0
    for bar, value in zip(bars, sorted_importances):
        ax.text(
            bar.get_width() + max_importance * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va='center',
            ha='left',
            fontsize=12,
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


#  STEP 4: ABLATION STUDY

def run_ablation_study(df: pd.DataFrame):
    """
    Run ablation study by training models with different feature subsets.
    
    Configurations:
    - All features
    - Without gaze features
    - Without audio features
    - Without face features
    """
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    
    all_features = get_feature_columns()
    
    # Define ablation configurations
    # Fair comparison: multimodal core excludes final_integrity_score because it is
    # already a fused heuristic score and can mask single-modality differences.
    multimodal_core = [
        "gaze_score",
        "face_score",
        "voice_score",
        "micro_var",
        "jitter",
        "shimmer",
        "spectral_centroid",
        "anomaly_duration",
        "object_flag",
    ]

    configs = {
        "Multimodal (Fair)": multimodal_core,
        "Gaze Only": ["gaze_score", "anomaly_duration"],
        "Face Only": ["face_score", "micro_var"],
        "Audio Only": ["voice_score", "jitter", "shimmer", "spectral_centroid"],
        "With Pre-Fused Score": all_features,
    }
    
    results = []
    
    for config_name, features in configs.items():
        print(f"\n[*] Configuration: {config_name}")
        print(f"    Features: {features}")
        
        # Check all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"    [!] Skipping - missing columns: {missing}")
            continue
        
        X = df[features].values
        y = df["label"].values.astype(int)

        # Use CV means for stable comparisons on imbalanced datasets
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1")
        cv_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        cv_bal_acc = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")

        accuracy = float(cv_acc.mean())
        f1 = float(cv_f1.mean())
        roc_auc = float(cv_auc.mean())
        bal_acc = float(cv_bal_acc.mean())
        
        results.append({
            "Configuration": config_name,
            "Accuracy": accuracy,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "Balanced-Acc": bal_acc,
            "Features": len(features),
        })
        
        print(f"    Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | Bal-Acc: {bal_acc:.4f}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<24s} | {'Accuracy':>10s} | {'F1':>10s} | {'ROC-AUC':>10s} | {'Bal-Acc':>10s} | {'#Feat':>6s}")
    print("-"*92)
    for r in results:
        print(f"{r['Configuration']:<24s} | {r['Accuracy']:>10.4f} | {r['F1']:>10.4f} | {r['ROC-AUC']:>10.4f} | {r['Balanced-Acc']:>10.4f} | {r['Features']:>6d}")
    print("="*92)
    
    return results


#  MAIN ENTRY POINT

def main(csv_path: str = None,
         ablation: bool = False,
         model_type: str = "calibrated",
         include_simulated: bool = True):
    """Main training pipeline."""
    
    # Load data
    df = load_dataset(
        csv_path=csv_path,
        merge_logs=(csv_path is None),
        include_simulated=include_simulated,
    )
    
    if len(df) < 20:
        print("[!] Insufficient data. Need at least 20 samples.")
        sys.exit(1)
    
    # Ablation study mode
    if ablation:
        run_ablation_study(df)
        return
    
    # Standard training
    feature_cols = get_feature_columns()
    
    # Check all features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"[!] Missing columns in dataset: {missing}")
        print(f"    Available: {list(df.columns)}")
        sys.exit(1)
    
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    
    # 80/20 split
    print(f"\n[*] Splitting data: 80% train / 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train
    if model_type == "rf":
        clf = train_random_forest(X_train, y_train)
        model_name = "random_forest"
    else:
        clf = train_calibrated_logistic_formula(X_train, y_train)
        model_name = "calibrated_logistic_formula"
    
    # Evaluate
    metrics = evaluate_model(clf, X_train, y_train, X_test, y_test, feature_cols)
    
    # Save model
    payload = {
        "model": clf,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_type": model_name,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)
    
    print(f"\n[✓] Model saved to: {MODEL_OUT}")
    print(f"    Features: {feature_cols}")
    print(f"    To load: pickle.load(open('{MODEL_OUT}', 'rb'))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDP-1 Integrity Model Trainer (Research Edition)"
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
        "--model", type=str, default="calibrated", choices=["calibrated", "rf"],
        help="Model formula to train: calibrated (default) or rf"
    )
    parser.add_argument(
        "--include-simulated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include CSVs from simulated_data/ when auto-merging datasets",
    )
    args = parser.parse_args()
    
    main(
        csv_path=args.csv,
        ablation=args.ablation,
        model_type=args.model,
        include_simulated=args.include_simulated,
    )

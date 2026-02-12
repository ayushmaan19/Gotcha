"""
SDP-1: Integrity Model Trainer
================================
Trains a RandomForestClassifier on biometric features and saves the model
as `integrity_model.pkl`.

Expected CSV schema
───────────────────
  yaw       : float  — Head yaw angle in degrees (0 = centre)
  ear       : float  — Eye Aspect Ratio (≈0.25 open, <0.20 blink)
  volume    : float  — RMS audio energy (0.0–1.0)
  label     : int    — 0 = legitimate,  1 = suspicious

Usage
─────
  # 1. Put your dataset next to this file (or pass a path):
  python train_model.py                         # uses data.csv in cwd
  python train_model.py --csv /path/to/data.csv

  # 2. The trained model is saved as integrity_model.pkl
"""

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CSV = "data.csv"
MODEL_OUT   = "integrity_model.pkl"

# Feature engineering thresholds (mirror app.py logic)
YAW_THRESH    = 20.0   # degrees — looking-away boundary
EAR_THRESH    = 0.20   # blink / eye-closure boundary
VOL_HIGH      = 0.60   # "shouting" energy boundary
VOL_LOW       = 0.01   # "muted / silent" energy boundary


def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset when no real CSV is available.
    This lets you train a baseline model immediately.
    
    Class 0 — Legitimate (centred gaze, normal blink, moderate voice)
    Class 1 — Suspicious  (off-axis gaze, abnormal blink, stressed voice)
    """
    rng = np.random.RandomState(seed)
    half = n_samples // 2

    # ── Legitimate samples ───────────────────────────────────────────────
    yaw_legit  = rng.normal(0.0, 5.0, half)          # mostly centred
    ear_legit  = rng.normal(0.28, 0.03, half)         # natural blink range
    vol_legit  = rng.normal(0.25, 0.08, half)         # calm speaking
    labels_legit = np.zeros(half, dtype=int)

    # ── Suspicious samples ───────────────────────────────────────────────
    yaw_sus  = rng.normal(35.0, 12.0, half)           # looking off to the side
    yaw_sus  = np.abs(yaw_sus)                        # always positive deviation
    ear_sus  = rng.normal(0.15, 0.05, half)           # squinting / forced
    # Half muted, half shouting
    vol_quiet = rng.normal(0.02, 0.01, half // 2)
    vol_loud  = rng.normal(0.75, 0.10, half - half // 2)
    vol_sus   = np.concatenate([vol_quiet, vol_loud])
    rng.shuffle(vol_sus)
    labels_sus = np.ones(half, dtype=int)

    df = pd.DataFrame({
        "yaw":    np.concatenate([yaw_legit, yaw_sus]),
        "ear":    np.concatenate([ear_legit, ear_sus]).clip(0.0, 0.5),
        "volume": np.concatenate([vol_legit, vol_sus]).clip(0.0, 1.0),
        "label":  np.concatenate([labels_legit, labels_sus]),
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive higher-order features that match the fusion logic in app.py.
    These give the tree-based model better decision boundaries.
    """
    df = df.copy()
    df["yaw_abs"]       = df["yaw"].abs()
    df["looking_away"]  = (df["yaw_abs"] > YAW_THRESH).astype(int)
    df["blink_risk"]    = (df["ear"] < EAR_THRESH).astype(int)
    df["vol_silent"]    = (df["volume"] < VOL_LOW).astype(int)
    df["vol_shout"]     = (df["volume"] > VOL_HIGH).astype(int)

    # Interaction features (cross-modal cues)
    df["away_and_quiet"]  = df["looking_away"] * df["vol_silent"]
    df["away_and_loud"]   = df["looking_away"] * df["vol_shout"]

    # Basic sub-scores (simplified version of app.py fusion)
    df["gaze_score"]  = 100.0 - df["yaw_abs"].clip(0, 50) * 2.0
    df["face_score"]  = np.where(df["ear"] < 0.10, 0.0, 100.0)
    df["voice_score"] = 100.0 - np.where(
        df["vol_silent"] | df["vol_shout"], 30.0, 0.0
    )
    df["fusion_score"] = (
        0.4 * df["gaze_score"]
        + 0.3 * df["voice_score"]
        + 0.3 * df["face_score"]
    )
    return df


def train(csv_path: str | None = None):
    """Load data, engineer features, train model, evaluate, and save."""

    # ── 1. Load or generate data ─────────────────────────────────────────
    if csv_path and os.path.isfile(csv_path):
        print(f"[*] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        required = {"yaw", "ear", "volume", "label"}
        missing = required - set(df.columns)
        if missing:
            print(f"[!] CSV is missing columns: {missing}")
            sys.exit(1)
    else:
        if csv_path:
            print(f"[!] File not found: {csv_path}")
        print("[*] Generating synthetic dataset (2 000 samples) ...")
        df = generate_synthetic_dataset()
        # Save so the user can inspect / extend it
        df.to_csv("data.csv", index=False)
        print(f"    → Saved synthetic data to data.csv")

    print(f"[*] Dataset shape: {df.shape}")
    print(f"    Class distribution:\n{df['label'].value_counts().to_string()}\n")

    # ── 2. Feature engineering ───────────────────────────────────────────
    df = add_engineered_features(df)

    feature_cols = [
        "yaw_abs", "ear", "volume",
        "looking_away", "blink_risk",
        "vol_silent", "vol_shout",
        "away_and_quiet", "away_and_loud",
        "gaze_score", "face_score", "voice_score", "fusion_score",
    ]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    # ── 3. Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 4. Train Random Forest ───────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Suspicious"]))
    print("── Confusion Matrix ──")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
    print(f"\n[*] 5-Fold CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importances
    print("\n── Feature Importances ──")
    for name, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {name:<20s} {imp:.4f}")

    # ── 6. Save ──────────────────────────────────────────────────────────
    payload = {
        "model": clf,
        "feature_cols": feature_cols,
        "thresholds": {
            "yaw": YAW_THRESH,
            "ear": EAR_THRESH,
            "vol_high": VOL_HIGH,
            "vol_low": VOL_LOW,
        },
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[✓] Model saved to {MODEL_OUT}")
    print(f"    Features: {feature_cols}")
    print(f"    To load:  pickle.load(open('{MODEL_OUT}', 'rb'))")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the SDP-1 Integrity classifier"
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help="Path to CSV with columns [yaw, ear, volume, label]"
    )
    args = parser.parse_args()
    train(csv_path=args.csv)

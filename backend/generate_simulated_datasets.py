"""
Generate realistic simulated multimodal datasets for quick experimentation.

Important:
- These datasets are simulated and should be reported as synthetic/simulated in papers.
- The generator bootstraps class-conditional statistics from existing session logs
  when available; otherwise it falls back to safe defaults.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


FEATURES = [
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


@dataclass
class ClassStats:
    mean: np.ndarray
    std: np.ndarray


def _default_stats() -> dict[int, ClassStats]:
    legit_mean = np.array([88, 90, 87, 0.65, 1.6, 4.5, 1500, 0.5, 0.04, 84], dtype=float)
    legit_std = np.array([9, 8, 10, 0.25, 0.8, 1.7, 350, 0.6, 0.10, 10], dtype=float)

    suspicious_mean = np.array([38, 48, 56, 0.14, 4.8, 11.0, 2500, 2.8, 0.35, 36], dtype=float)
    suspicious_std = np.array([20, 20, 22, 0.08, 1.9, 3.5, 700, 2.2, 0.25, 18], dtype=float)

    return {
        0: ClassStats(legit_mean, legit_std),
        1: ClassStats(suspicious_mean, suspicious_std),
    }


def _load_real_logs(dataset_logs_dir: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(dataset_logs_dir, "session_*.csv")))
    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path)
            # Skip aborted near-empty logs.
            if len(df) > 1 and "label" in df.columns:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)
    return merged


def _class_stats_from_df(df: pd.DataFrame) -> dict[int, ClassStats]:
    stats = _default_stats()
    for label in (0, 1):
        sub = df[df["label"] == label]
        if len(sub) < 20:
            continue

        # Use robust quantile clipping before estimating stats.
        clipped = sub.copy()
        for col in FEATURES:
            if col not in clipped.columns:
                continue
            lo, hi = clipped[col].quantile(0.01), clipped[col].quantile(0.99)
            clipped[col] = clipped[col].clip(lo, hi)

        mean = np.array([clipped[c].mean() for c in FEATURES], dtype=float)
        std = np.array([max(clipped[c].std(ddof=0), 1e-6) for c in FEATURES], dtype=float)
        stats[label] = ClassStats(mean=mean, std=std)
    return stats


def _sample_class(rng: np.random.Generator, stats: ClassStats, n: int, label: int) -> pd.DataFrame:
    # Correlated latent factors to produce realistic feature coupling.
    z_behavior = rng.normal(0.0, 1.0, n)
    z_stress = rng.normal(0.0, 1.0, n)
    z_env = rng.normal(0.0, 1.0, n)
    z_noise = rng.normal(0.0, 1.0, (n, len(FEATURES)))

    # Base independent draw around class mean/std.
    x = stats.mean + stats.std * z_noise

    # Inject structure.
    # behavior factor: gaze, face, final integrity
    x[:, 0] += (8.0 if label == 0 else 10.0) * z_behavior
    x[:, 1] += (6.0 if label == 0 else 8.0) * z_behavior
    x[:, 9] += (10.0 if label == 0 else 12.0) * z_behavior

    # stress factor: voice/jitter/shimmer/centroid
    x[:, 2] += (-4.0 if label == 0 else -6.0) * z_stress
    x[:, 4] += (0.6 if label == 0 else 1.1) * z_stress
    x[:, 5] += (1.0 if label == 0 else 1.8) * z_stress
    x[:, 6] += (120.0 if label == 0 else 220.0) * z_stress

    # env factor: object flag and anomaly duration and integrity
    x[:, 7] += (0.4 if label == 0 else 1.0) * z_env
    x[:, 9] += (-2.0 if label == 0 else -5.0) * z_env

    df = pd.DataFrame(x, columns=FEATURES)

    # Clip to physical ranges.
    df["gaze_score"] = df["gaze_score"].clip(0, 100)
    df["face_score"] = df["face_score"].clip(0, 100)
    df["voice_score"] = df["voice_score"].clip(0, 100)
    df["micro_var"] = df["micro_var"].clip(0.01, 5.0)
    df["jitter"] = df["jitter"].clip(0.0, 15.0)
    df["shimmer"] = df["shimmer"].clip(0.0, 25.0)
    df["spectral_centroid"] = df["spectral_centroid"].clip(400.0, 5000.0)
    df["anomaly_duration"] = df["anomaly_duration"].clip(0.0, 12.0)
    df["final_integrity_score"] = df["final_integrity_score"].clip(0.0, 100.0)

    # Bernoulli object flag using class-specific priors and env latent.
    base_p = 0.05 if label == 0 else 0.35
    p = np.clip(base_p + 0.08 * np.tanh(z_env), 0.0, 1.0)
    df["object_flag"] = rng.binomial(1, p).astype(int)

    # Coherence adjustment for final integrity score.
    fused = 0.4 * df["gaze_score"] + 0.3 * df["voice_score"] + 0.3 * df["face_score"]
    penalty = 12.0 * df["object_flag"] + 2.2 * df["anomaly_duration"]
    if label == 1:
        penalty += 4.0
    df["final_integrity_score"] = np.clip(0.75 * fused + 0.25 * df["final_integrity_score"] - penalty, 0, 100)

    df["label"] = int(label)
    return df


def make_dataset(stats: dict[int, ClassStats], n_rows: int, suspicious_ratio: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_suspicious = int(round(n_rows * suspicious_ratio))
    n_legit = n_rows - n_suspicious

    legit = _sample_class(rng, stats[0], n_legit, label=0)
    suspicious = _sample_class(rng, stats[1], n_suspicious, label=1)

    df = pd.concat([legit, suspicious], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Add timestamp-like progression to mimic session windows.
    df.insert(0, "timestamp", np.arange(len(df), dtype=float))
    return df


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "dataset_logs")
    out_dir = os.path.join(base_dir, "simulated_data")
    os.makedirs(out_dir, exist_ok=True)

    real_df = _load_real_logs(logs_dir)
    if real_df is not None:
        stats = _class_stats_from_df(real_df)
        print(f"[*] Using real logs for calibration: {len(real_df)} rows")
    else:
        stats = _default_stats()
        print("[*] No usable logs found. Using default synthetic priors")

    specs = [
        (3000, 0.62, 20260317),
        (8000, 0.60, 20260318),
        (15000, 0.58, 20260319),
    ]

    created = []
    for n_rows, ratio, seed in specs:
        df = make_dataset(stats, n_rows=n_rows, suspicious_ratio=ratio, seed=seed)
        name = f"simulated_{n_rows}_r{int(ratio*100)}.csv"
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        created.append((name, len(df), int(df["label"].sum()), int((df["label"] == 0).sum())))
        print(f"[+] Wrote {name}: {len(df)} rows")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(out_dir, f"manifest_{stamp}.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Simulated datasets (must be reported as simulated)\n")
        f.write("name,total_rows,suspicious_rows,legitimate_rows\n")
        for row in created:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

    print(f"[+] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

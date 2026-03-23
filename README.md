# Real-Time Multimodal Biometric Integrity Analysis (RMBIA)

A lightweight, CPU-first multimodal integrity scoring system for remote communication.
The project combines face, gaze, audio, and context features in real time and uses a trained fusion model for suspicious-behavior detection.

## Overview

RMBIA is designed for continuous behavioral verification during a video session.
Instead of one-time login authentication, it continuously tracks behavioral and physiological signals to estimate an integrity/risk score.

Primary use case: detecting social engineering indicators such as coached responses, abnormal gaze patterns, liveness inconsistency, and audio stress anomalies.

## What Is Implemented

- Real-time Streamlit + WebRTC pipeline.
- Face-based features (liveness variance and expression stability).
- Gaze/blink features (attention drift, blink behavior).
- Audio forensics features (jitter, shimmer proxies, spectral descriptors).
- Optional object-context signal via EfficientDet Lite0.
- ML fusion with RandomForest and a live dashboard score.
- Dataset logger for supervised model training.
- Training and ablation pipeline with plots and reports.
- Synthetic dataset generator for rapid scale-up experiments.

## Latest Updates (March 2026)

- Fixed audio mono conversion robustness in the real-time stream path to handle channel layout differences.
- Restored object detection flag propagation into logged features for true multimodal capture.
- Updated ablation pipeline for fair multimodal vs single-modality comparison.
- Added Balanced Accuracy to handle class imbalance more honestly.
- Fixed training behavior so custom CSV runs do not auto-merge session logs unintentionally.
- Added synthetic benchmark generation script for 10k-60k controlled experiments.
- Expanded ignore rules to keep generated artifacts and binaries out of normal source commits.

## Verified Results Snapshot

### Real logged-session dataset (732 rows)

- Test Accuracy: 0.7755
- Test F1: 0.8619
- Test ROC-AUC: 0.6194
- 5-Fold CV Accuracy: 0.7844 +- 0.0138

### Fair ablation (5-fold CV, logged-session dataset)

- Multimodal (fair): Accuracy 0.7869, Balanced Accuracy 0.6066
- Gaze only: Accuracy 0.7855, Balanced Accuracy 0.5603
- Face only: Accuracy 0.7828, Balanced Accuracy 0.6018
- Audio only: Accuracy 0.3336, Balanced Accuracy 0.5000

### Synthetic large-scale run (10,000 rows)

- Test Accuracy: 0.9550
- Test F1: 0.9617
- Test ROC-AUC: 0.9907
- CV Accuracy: 0.9514

### Lightweight runtime evidence (Apple M1, CPU)

- Fusion model size: ~197 KB
- Detector model size: ~13 MB
- Fusion inference: ~13.37 ms/sample
- Detector inference (640x480): ~36.47 ms/frame (~27.4 FPS)

## Repository Structure

```text
.
├── README.md
├── project_whitepaper.md
├── backend/
│   ├── app.py
│   ├── collect_data.py
│   ├── feature_logger.py
│   ├── train_model.py
│   ├── generate_simulated_datasets.py
│   ├── requirements.txt
│   ├── efficientdet_lite0.tflite
│   ├── dataset_logs/
│   ├── dataset_raw/
│   └── simulated_data/
└── frontend/
```

## Setup

### 1) Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3) Run the app

```bash
streamlit run backend/app.py
```

## Data Collection and Training

### Collect session data

```bash
python backend/collect_data.py
```

### Train on merged logged data

```bash
python backend/train_model.py
```

### Train on a specific CSV only

```bash
python backend/train_model.py --csv backend/simulated_data/simulated_10000_r58.csv
```

### Run fair ablation

```bash
python backend/train_model.py --ablation
```

### Generate synthetic benchmark datasets

```bash
python backend/generate_simulated_datasets.py
```

## Notes for Submission

- Main implementation is in backend.
- frontend is currently a placeholder directory.
- Generated reports, plots, synthetic outputs, and model binaries are treated as artifacts and excluded from normal source tracking.

## Author

Ayushmaan Kumar Yadav


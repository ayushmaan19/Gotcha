<h1 align="center">Real-Time Multimodal Biometric Integrity Analysis (RMBIA)</h1>

<p align="center">
	<b>Zero-trust behavioral security for remote communication</b><br/>
	Face + Gaze + Audio + Context fused into a lightweight real-time integrity score.
</p>

<p align="center">
	<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
	<img src="https://img.shields.io/badge/Streamlit-Realtime-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
	<img src="https://img.shields.io/badge/Model-XGBoost-2E7D32?style=for-the-badge" />
	<img src="https://img.shields.io/badge/Inference-CPU%20Optimized-1565C0?style=for-the-badge" />
	<!-- <img src="https://img.shields.io/badge/Status-Submission%20Ready-6A1B9A?style=for-the-badge" /> -->
</p>

---

## Why This Project

Most systems validate identity once at login. RMBIA keeps validating behavior during the session.

It focuses on social engineering and impersonation cues such as:
- suspicious gaze behavior
- liveness/expression inconsistency
- audio stress anomalies
- contextual scene/object signals

---

## Core Pipeline

1. Capture live video/audio through WebRTC.
2. Extract multimodal features (face, gaze, audio, context).
3. Fuse signals with a trained XGBoost model.
4. Update real-time integrity risk on dashboard.

---

## What Is Implemented

- Real-time Streamlit + WebRTC app
- Face liveness and micro-expression variance features
- Gaze and blink behavior tracking
- Audio forensics-style features (jitter/shimmer proxies, spectral stats)
- Optional context signal using EfficientDet Lite0
- Fusion model with supervised training pipeline
- Session-level feature logger
- Training + ablation + evaluation plots

---

## Latest Updates (May 2026, V2)

- XGBoost with SMOTE for class imbalance and stable fusion training.
- Security-focused threshold tuning and Average Precision reporting.
- Temporal feature engineering for gaze/face trend stability.
- Audio impact analysis with fixed, production-safe feature order.
- Session-aware split to prevent leakage in evaluation.
- ML fusion calibration toggle with deterministic raw mode.

---

## Results Snapshot

### V2 Real-life Dataset Merge (120,136 rows)

| Metric | Value |
|---|---:|
| Test Accuracy | 0.8656 |
| Test F1 | 0.8707 |
| Test ROC-AUC | 0.9471 |
| Avg Precision | 0.9634 |
| 5-Fold CV Balanced Acc | 0.8725 +- 0.0019 |

### Fair Ablation (5-Fold CV)

| Configuration | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Multimodal (with audio) | 0.8691 | 0.8691 |
| Multimodal (without audio) | 0.8481 | 0.8481 |

### Large-Scale Run (10,000+ rows)

| Metric | Value |
|---|---:|
| Test Accuracy | 0.9550 |
| Test F1 | 0.9617 |
| Test ROC-AUC | 0.9907 |
| CV Accuracy | 0.9514 |

### Lightweight Runtime (Apple M1, CPU)

| Component | Value |
|---|---:|
| Fusion model size | ~197 KB |
| Detector model size | ~13 MB |
| Fusion inference | ~13.37 ms/sample |
| Detector inference (640x480) | ~36.47 ms/frame (~27.4 FPS) |

---

## Future Prospects and Improvements

- Expand session-aware evaluation to grouped cross-validation with per-user folds.
- Add uncertainty calibration (temperature scaling) for smoother risk probabilities.
- Introduce adaptive thresholds by environment context and device class.
- Add model drift detection and periodic retraining triggers.
- Optimize latency with batched inference and lightweight quantization.
- Improve UX with transparent reason codes and confidence bands.

---

## Project Structure

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
│ 
└── frontend/
```

---

## Quick Start

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run real-time app

```bash
streamlit run backend/app.py
```

---

## Data and Training Commands

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

---

## Submission Notes

- Backend contains the full working implementation.
- Frontend folder is currently a placeholder.
- Generated reports/plots/models are treated as build artifacts and excluded from normal source tracking.

---

## Author

**Ayushmaan Kumar Yadav**


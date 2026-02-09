# Real-Time Multimodal Biometric Integrity Analysis (RMBIA)
### A "Zero-Trust" Behavioral Security Framework for Remote Communication

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Computer Vision](https://img.shields.io/badge/Library-MediaPipe%20%7C%20OpenCV-green)
![Status](https://img.shields.io/badge/Status-Prototype%20(MVP)-orange)

---

## 📌 1. Project Abstract
**RMBIA** is a real-time security framework designed to detect **Social Engineering** and **Identity Impersonation** attacks in video conferencing. Unlike traditional authentication that verifies identity only at login (Entry-Point Security), this system employs **Continuous Behavioral Authentication**.

By fusing **Ocular Kinematics** (Gaze Tracking), **Facial Geometry** (Liveness Detection), and **Vocal Prosody** (Stress Analysis), the system calculates a dynamic **Integrity Score** to flag suspicious behavior—such as coerced speech, script reading, or deepfake usage—in real-time. This effectively creates a "Zero-Trust" security layer inside the video call session.

---

## 🚀 2. Key Features & Modules

### A. Visual Liveness & Integrity (Face Module)
- **Technology:** Google MediaPipe Face Mesh (468 Landmarks).
- **Function:** Detects 3D facial geometry to distinguish between live humans and 2D spoofing attacks.
- **Micro-Expression Analysis:** Monitors facial variance to detect "frozen" deepfake expressions or unnatural emotional consistency.

### B. Ocular Attention Monitoring (Gaze Module)
- **Technology:** OpenCV & Eye Aspect Ratio (EAR).
- **Function:** Tracks eye gaze vectors and blink rates.
- **Social Engineering Detection:** Flags **Gaze Aversion** (looking off-screen constantly), which is a high-confidence indicator of reading a script or forced coaching (Cognitive Load).

### C. Audio Forensics (Voice Module) *[Planned]*
- **Technology:** Librosa (Spectral Analysis).
- **Function:** Extracts **Jitter** (Pitch Instability) and **Shimmer** (Loudness Instability) to detect stress biomarkers often present in coerced victims or synthetic voice artifacts.

---

## 🧠 3. System Architecture

The system follows a **Multimodal Late-Fusion Pipeline**:

1.  **Input Layer:** Captures live Video/Audio stream via WebRTC for low-latency (<300ms) transmission.
2.  **Inference Layer:** Parallel processing of Visual and Auditory streams using lightweight CPU-optimized models (Edge Computing).
3.  **Fusion Layer:** A weighted algorithm combines individual probability scores into a single metric.
4.  **Decision Layer:** Updates the "Integrity Score" dashboard in real-time.

### The Fusion Algorithm (Novelty)
The core innovation is the **Weighted Integrity Metric**, which prioritizes behavioral signals over static visual checks:

$$I_{score} = (w_1 \cdot S_{gaze}) + (w_2 \cdot S_{voice}) + (w_3 \cdot S_{face})$$

*Where $w$ represents the adaptive weight of each modality based on signal quality (e.g., Gaze=0.4, Voice=0.3, Face=0.3).*

---

## 🛠️ 4. Installation & Setup Guide

### Prerequisites
- Python 3.8 or higher
- A working webcam

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your-username/biometric-integrity-analysis.git](https://github.com/your-username/biometric-integrity-analysis.git)
cd biometric-integrity-analysis
```
---
## Step 2: Create a Virtual Environment (Recommended)
```
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
---
## Step 3: Install Dependencies
```
pip install streamlit streamlit-webrtc mediapipe opencv-python-headless numpy av
```
---
## Step 4: Run the Application
```
streamlit run app.py
```
---
## 📂 5. Project Structure
```
├── app.py                 # Main Streamlit Application (MVP Logic)
├── modules/
│   ├── face_mesh.py       # Facial Landmark Extraction Logic
│   ├── gaze_tracker.py    # Eye Aspect Ratio & Head Pose Logic
│   └── audio_analysis.py  # (Future) Jitter/Shimmer Logic
├── utils/
│   └── helper.py          # Frame processing utilities
├── requirements.txt       # Python dependencies list
└── README.md              # Project Documentation
```
---
## Future Roadmap
[x] Phase 1: Video-Only MVP (Face Mesh + Head Pose).

[ ] Phase 2: Integration of Audio Stress Analysis (Librosa).

[ ] Phase 3: Creation of a custom validation dataset (Truth vs. Deception).

[ ] Phase 4: Validation Testing (Ablation Study) & Publication in IEEE Regional Conference.

---
## License
Distributed under the MIT License. See LICENSE for more information. 

---
## 💻 Author
**Ayushmaan Kumar Yadav** | [Know More &#8599;](https://www.ayushmaanport.dev)


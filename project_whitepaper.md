# Project Whitepaper: Real-Time Multimodal Biometric Integrity Analysis

---

## 1. Introduction: The "Why" Behind the Project

### 1.1 The Problem: The "Login Paradox"

Imagine a bank vault. To get in, you need a high-security key (Password + 2FA). This is **Entry-Point Authentication**. But once the door is open, the security system turns off. If a thief knocks out the guard and walks in after the door is unlocked, the system doesn't know.

This is exactly how **Zoom, Teams, and Google Meet** work today.

- **Step 1:** You log in (Secure).
- **Step 2:** The video call starts (Unsecure).

### 1.2 The Threat Landscape

Cybercriminals are exploiting this gap using **Social Engineering**:

1. **Deepfakes:** A hacker uses AI to swap their face with your boss's face in real-time using GANs/Diffusion models.
2. **Coercion:** A real employee is on the call, but a criminal is standing behind the camera with a weapon, forcing them to approve a transaction.
3. **Session Hijacking:** You log in, walk away to get coffee, and an unauthorized actor sits down at your unlocked laptop.

### 1.3 The Solution: "Continuous" Authentication

We are building a **"Digital Security Guard"** that sits inside the video call. It doesn't just check *who you are*; it checks *how you are acting*. It monitors you **every second** (Continuous) using **four different senses** (Multimodal) to ensure you are behaving normally (Integrity).

---

## 2. System Architecture (The "How")

The system follows a **Pipeline Architecture**. Data flows in a straight line, like a factory assembly line.

```
┌─────────────┐     ┌─────────────────────────────────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │         INFERENCE LAYER                  │     │   FUSION     │     │  DASHBOARD  │
│   WebRTC    │────▶│  A. Face Liveness (FaceMesh 478)        │────▶│   LAYER      │────▶│  Integrity  │
│   Video +   │     │  B. Ocular Attention (Gaze+EAR)         │     │              │     │   Score %   │
│   Audio     │     │  C. Audio Forensics (Jitter/Shimmer)    │     │  Weighted    │     │  + Status   │
│             │     │  D. Object Detection (EfficientDet)     │     │  Sum Rule    │     │  + Alerts   │
└─────────────┘     └─────────────────────────────────────────┘     └─────────────┘     └─────────────┘
```

### The 4 Stages of the Pipeline

| Stage | Technology | Purpose |
|-------|-----------|---------|
| **Input** | WebRTC (Web Real-Time Communication) | Streams video + audio from browser with near-zero latency |
| **Inference** | MediaPipe FaceMesh, Librosa, EfficientDet-Lite0 | Four AI models analyze Face, Eyes, Voice, and Environment |
| **Fusion** | Weighted Sum Rule (configurable) | Combines sub-scores into a single trust metric |
| **Application** | Streamlit Dashboard | Displays real-time Integrity Score with visual HUD overlay |

---

## 3. Technical Deep Dive: The Four Modules

### Module A: Visual Liveness (The Face)

**Goal:** Prove the person is a 3D human, not a 2D photo or a deepfake.

**Technology:** Google MediaPipe Face Mesh (478 landmarks with iris refinement).

**What is it?** Imagine drawing a "connect-the-dots" spiderweb over your face. MediaPipe draws 478 dots (landmarks) on your face in real-time, including 10 iris landmarks for sub-millimeter eye tracking.

**How we use it:**

1. **3D Head Pose Geometry:** We compute the relative position of the Nose Tip (landmark 1) vs the Ear Tragions (landmarks 234, 454). If the ratio deviates beyond a configurable threshold, the head is turned — flagging distraction or an imposter swapping in from the side.

2. **Micro-Expression Variance (Novel):** We continuously sample 12 key facial landmarks (mouth corners, cheeks, brows, chin) into a rolling buffer of 30 frames. We compute the **positional variance** across this window:
   - **High variance** → alive human (natural micro-muscle movements)
   - **Near-zero variance** → static photo, frozen deepfake mask, or "numb" AI-generated face
   - Threshold: `variance < 0.08` triggers "FROZEN (DEEPFAKE?)" alert

3. **Liveness Timer:** If no blink or gaze micro-saccade occurs for 60 seconds, the system flags "STILL (FAKE?)" — catching photos held to the camera.

### Module B: Ocular Attention (The Eyes)

**Goal:** Detect script reading (Social Engineering), dead staring (Deepfake), and lying (Cognitive Load).

**Technology:** Eye Aspect Ratio (EAR) + Iris Gaze Vector + Lie Detection Heuristic.

#### B1. Blink Detection (EAR)

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

- **EAR ≈ 0.3** → Eye Open
- **EAR < Threshold (default 0.25)** → Blink detected
- Humans blink 15–20 times/minute. Deepfakes often blink too little (staring) or too much (glitching).

#### B2. Gaze Tracking

**Horizontal:** Calculates iris position (landmarks 468/473) relative to eye corners (362/263 for left, 33/133 for right). Result: 0.0 = far left, 0.5 = center, 1.0 = far right.

**Vertical (Ethnicity-Agnostic):** Uses **linear algebra** to measure iris offset from the line connecting stable eye corners (bone structure), rather than eyelid positions. This makes the system work accurately across all face shapes — including monolid eyes.

$$\text{VerticalRatio} = 0.5 + \frac{iris_y - eyeCenter_y}{||corner_1 - corner_2||}$$

**Timed Distraction (Whitepaper Rule):** If gaze deviates beyond the sensitivity threshold:
- **0–3 seconds:** Grace period (small −1 pt nudge)
- **> 3 seconds:** Aggressive penalty (−10 pts/second) — catches sustained script reading

#### B3. Lie Detection (Cognitive Load)

Based on NLP eye-access cue theory: **Looking Up-Right** (subject's right) correlates with *visual construction* (fabrication/lying), as opposed to visual *recall* (memory).

When both conditions are met simultaneously:
- Horizontal gaze > `LIE_HORIZ_THRESH` (looking right)
- Vertical gaze < `LIE_VERT_THRESH` (looking up)

→ Lie confidence ramps up (+3/frame). When the cluster breaks → confidence decays (−1/frame). At confidence > 50%, the system reports "LIE DETECTED" and drags the Gaze sub-score down by −5 pts/frame.

### Module C: Audio Forensics (The Voice)

**Goal:** Detect nervousness (Coercion) and robotic artifacts (AI Voice Cloning).

**Technology:** Librosa (Python Spectral Analysis), running in a dedicated `AudioProcessorBase` thread.

**Architecture:** Audio frames arrive via WebRTC at 48 kHz. They are buffered into 1-second chunks, then analyzed:

| Biomarker | What It Measures | Stress Signal |
|-----------|-----------------|---------------|
| **Jitter** | Pitch micro-tremors (`pYIN` F0 extraction → period perturbation) | Vocal cord tension under fear/stress |
| **Shimmer** | Amplitude micro-tremors (peak envelope perturbation) | Breath control loss under duress |
| **Spectral Centroid** | Average frequency "brightness" | Tense/metallic voice = high centroid |
| **MFCC Flatness** | Ratio of geometric to arithmetic mean of MFCC variance | Flat = synthetic/AI voice; Rich = biological |

**Voice Sub-Score Derivation:**
```
score = 100
if jitter > JITTER_THRESH:  score -= min(30, (jitter - thresh) * 10)
if shimmer > SHIMMER_THRESH: score -= min(30, (shimmer - thresh) * 5)
if spectral_centroid > 3500: score -= 10
if mfcc_flatness > 0.6:     score -= 15
```

### Module D: Environmental Context (Object Detection)

**Goal:** Prevent malpractice — detect secondary communication devices.

**Technology:** EfficientDet-Lite0 (quantized TFLite, CPU-optimized). Runs every 10 frames (~3 Hz) for balanced speed/accuracy.

**Banned Objects:** `cell phone`, `mobile phone`, `laptop`, `computer`, `tv`, `monitor`, `remote`

**Logic:** Case-insensitive keyword matching + configurable confidence threshold (default 65%). On detection: red bounding box overlay + immediate −30 pt penalty to fused score.

---

## 4. The "Secret Sauce": Weighted Fusion Algorithm

This is the core research contribution. We don't just "guess" — we use a **Weighted Sum Rule** with configurable weights, multimodal synergy penalties, and timed logic.

### The Formula

$$\text{Integrity Score} = \frac{w_g \cdot \text{GazeScore} + w_v \cdot \text{VoiceScore} + w_f \cdot \text{FaceScore}}{w_g + w_v + w_f}$$

### Default Weights

| Signal | Weight | Rationale |
|--------|--------|-----------|
| **Gaze** (40%) | 0.4 | Strongest indicator of social engineering (script reading, distraction) |
| **Voice** (30%) | 0.3 | High jitter = physiological stress; flat MFCCs = AI clone |
| **Face** (30%) | 0.3 | Low variance = deepfake; absence = session hijacking |

### Multimodal Synergy Penalties

This is where **1 + 1 = 3**. Individual signals may be ambiguous, but *combinations* are definitive:

| Combination | Penalty | Interpretation |
|-------------|---------|---------------|
| Distracted gaze **AND** voice jitter | **−40 pts** | 🔴 **Coercion Alert** — user is looking at captor + voice trembling |
| Banned device detected | **−30 pts** | 🚨 **Cheating** — phone/laptop in frame |
| Frozen landmarks (var < 0.08) | Face score drain (−3/frame) | 🧊 **Deepfake** — no micro-expressions |
| No activity for 60s | Face score drain (−5/frame) | ⏸️ **Static image** — possible photo attack |

### The Logic Flow

```
Start at 100 Points (each sub-score).

PER FRAME:
  Gaze away for < 3s    → GazeScore -= 1   (grace period)
  Gaze away for > 3s    → GazeScore -= 10  (sustained distraction)
  Gaze centered          → GazeScore += 3   (recovery)
  Lie cluster active     → GazeScore -= 5
  
  Face missing           → FaceScore -= 15
  Frozen micro-expr      → FaceScore -= 3
  Normal movement        → FaceScore += 2   (recovery)
  
  Voice jitter high      → VoiceScore -= up to 30
  Voice shimmer high     → VoiceScore -= up to 30
  Voice normal           → VoiceScore = 100
  
THEN FUSE:
  fused = weighted_average(Gaze, Voice, Face)
  
THEN SYNERGY:
  if distracted AND jittery → fused -= 40  (COERCION)
  if device detected        → fused -= 30  (CHEATING)
  
CLAMP: max(0, min(100, fused))
```

---

## 5. Novelty: What Makes This New?

### 5.1 Behavioral vs. Digital Forensics

- **Current State:** Most research focuses on *Digital Forensics* (finding bad pixels, compression artifacts). This is an "Arms Race" — as Deepfakes improve, these detectors fail.
- **Our Approach:** **Behavioral Forensics**. A Deepfake might have perfect pixels, but if the "user" doesn't blink for 2 minutes, or reads a script without moving their eyes, our system catches the *behavior*. You can fake a face, but it is incredibly hard to fake natural human micro-behavior in real-time.

### 5.2 Multimodal Synergy (1 + 1 = 3)

**Example: Identifying a "Fake Smile"**
- **Visual Only:** "Mouth is curved up. It's a smile." → *Fooled.*
- **Our System:** "Mouth is curved up (Visual), BUT eyes are not crinkling (Ocular) AND voice pitch is flat (Audio). This is a fake smile." → *Detected.*

**Example: Identifying Coercion**
- **Visual Only:** "Face is present, user looks normal." → *Fooled.*
- **Our System:** "Face is present, BUT eyes keep darting right (Gaze) AND voice has high jitter (Audio). User is under duress." → *Detected. −40 pt penalty.*

### 5.3 Lightweight Edge AI

Unlike server-side solutions requiring NVIDIA A100 GPUs, our system runs entirely on **CPU** using:
- MediaPipe (Google's edge-optimized framework)
- TFLite quantized models (EfficientDet-Lite0)
- Librosa (pure Python spectral analysis)

This preserves **privacy** (video/audio never leaves the device) and **democratizes security** (runs on any laptop).

### 5.4 Ethnicity-Agnostic Eye Tracking

Our vertical gaze computation uses linear algebra relative to **eye corners** (bone structure) rather than eyelid positions. This makes gaze tracking accurate for all face shapes — including monolid (East Asian) eyes — where eyelid-based methods systematically fail.

---

## 6. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI Framework | Streamlit | Rapid dashboard prototyping |
| Video Streaming | streamlit-webrtc (WebRTC) | Zero-latency browser video/audio capture |
| Face Analysis | MediaPipe FaceMesh (0.10.9) | 478-landmark facial geometry |
| Object Detection | EfficientDet-Lite0 (TFLite) | CPU-optimized banned-device scanning |
| Audio Analysis | Librosa + NumPy | Spectral features, pitch tracking (pYIN) |
| Computer Vision | OpenCV | Frame manipulation, HUD overlay rendering |
| Concurrency | Python threading (Lock) | Thread-safe Audio→Video state communication |

---

## 7. Glossary of Terms

| Term | Definition |
|------|-----------|
| **Biometrics** | Measuring biological traits (Face, Fingerprint, Voice) for identification |
| **Inference** | The exact moment the AI model makes a prediction on input data |
| **Latency** | End-to-end delay between reality and the computer seeing it (target: < 50ms) |
| **Zero-Trust** | Security model: "Never Trust, Always Verify" — applied continuously, not just at login |
| **Prosody** | The rhythm, stress, and intonation of speech (the "music" of your voice) |
| **Jitter** | Micro-tremors in vocal pitch caused by involuntary vocal cord tension (stress biomarker) |
| **Shimmer** | Micro-tremors in vocal loudness caused by breath control loss |
| **MFCCs** | Mel-Frequency Cepstral Coefficients — numerical "texture" of voice timbre |
| **EAR** | Eye Aspect Ratio — geometric ratio detecting open vs. closed eyes |
| **Micro-expressions** | Involuntary facial muscle movements (1/25th of a second) revealing true emotion |
| **Multimodal** | Combining multiple disjoint data types (Audio + Video + Geometry) to improve accuracy |
| **Weighted Fusion** | Combining signals with different importance weights into a single decision score |
| **pYIN** | Probabilistic YIN — pitch detection algorithm used for fundamental frequency (F0) extraction |
| **Spectral Centroid** | The "center of mass" of a sound spectrum — higher = brighter/tenser voice |

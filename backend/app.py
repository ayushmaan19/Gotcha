"""Real-time multimodal integrity monitoring dashboard."""

import sys
import streamlit as st


try:
    import mediapipe as mp
    _probe = mp.solutions.face_mesh         
except (AttributeError, ImportError) as _err:
    st.set_page_config(page_title="SDP-1: Environment Error", layout="centered")
    st.error("## Incompatible MediaPipe version")
    st.markdown(
        f"""
        Your current Python is **{sys.version.split()[0]}** (`{sys.executable}`).

        MediaPipe **{getattr(mp, '__version__', '?')}** installed here does **not**
        include the `solutions` module required by this app.

        **Fix — run with the project venv instead:**
        ```bash
        cd backend
        /Users/ayushmaankumaryadav/Desktop/gotcha/.venv/bin/python -m streamlit run app.py
        ```
        Or create a shortcut:
        ```bash
        alias gotcha='/Users/ayushmaankumaryadav/Desktop/gotcha/.venv/bin/python -m streamlit run'
        gotcha app.py
        ```
        """
    )
    st.stop()

import cv2
import av
import numpy as np
import xgboost
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import time
import threading
import collections
import os
from datetime import datetime
import pickle

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, RTCConfiguration
import librosa
import plotly.graph_objects as go

# Feature logging for dataset collection (modular addition)
from feature_logger import FeatureLogger, extract_features_from_state

st.set_page_config(
    page_title="SDP-1: Threat Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🔧 System Controls")

st.sidebar.markdown("### 🎯 Sensitivity")
HEAD_TURN_SENSITIVITY = st.sidebar.slider(
    "Head Turn Sensitivity", 0.20, 0.50, 0.40,
    help="Higher = stricter (0.5 = any tilt flagged)")
GAZE_SENSITIVITY = st.sidebar.slider(
    "Eye Gaze Sensitivity", 0.02, 0.15, 0.05,
    help="Lower = stricter (detects smaller eye movements)")
SIDE_GAZE_THRESHOLD = st.sidebar.slider(
    "Side Gaze Deviation", 0.08, 0.30, 0.14,
    help="Lower = more sensitive to left/right diversion",
)
SIDE_GAZE_HOLD = st.sidebar.slider(
    "Side Gaze Hold (s)", 0.20, 1.20, 0.45,
    help="How long side diversion must persist before DISTRACTED is triggered",
)
BLINK_THRESHOLD = st.sidebar.slider(
    "Blink Threshold (EAR)", 0.10, 0.40, 0.25)

st.sidebar.markdown("### 🧠 Lie Detection")
LIE_HORIZ_THRESH = st.sidebar.slider(
    "Lie Gaze Horizontal (>)", 0.50, 0.80, 0.55,
    help="Lower = more sensitive to rightward gaze")
LIE_VERT_THRESH = st.sidebar.slider(
    "Lie Gaze Vertical (<)", 0.20, 0.60, 0.45,
    help="Higher = more sensitive to upward gaze")

st.sidebar.markdown("### 🕒 Anti-Spoof Timing")
STILL_FAKE_SECONDS = st.sidebar.slider(
    "Stillness Before Fake Label (s)", 120, 180, 150,
    help="Minimum continuous stillness before showing STILL/DEEPFAKE labels",
)

st.sidebar.markdown("### 📱 Object Detection")
OBJ_CONFIDENCE = st.sidebar.slider(
    "Device Confidence", 0.40, 0.90, 0.65,
    help="Increase to reduce false positives")
CONTEXT_INTERVAL = st.sidebar.slider(
    "Object Detection Interval (frames)", 1, 30, 10,
    help="Run EfficientDet every N frames. Higher = less CPU usage")

st.sidebar.markdown("### 🎙️ Audio Forensics")
ENABLE_WEBRTC = st.sidebar.checkbox(
    "Enable Camera/WebRTC",
    value=True,
    help="Disable only if your browser throws media device component errors.",
)
ENABLE_WEBRTC_AUDIO = st.sidebar.checkbox(
    "Enable WebRTC Audio Capture",
    value=True,
    help="Disable if browser shows media device errors (ondevicechange).",
)
JITTER_THRESH = st.sidebar.slider(
    "Jitter Threshold (%)", 1.0, 8.0, 3.0,
    help="Voice pitch wobble limit — lower = stricter")
SHIMMER_THRESH = st.sidebar.slider(
    "Shimmer Threshold (%)", 3.0, 15.0, 8.0,
    help="Volume fluctuation limit — lower = stricter")

st.sidebar.markdown("### ⚖️ Fusion Weights")
W_GAZE = st.sidebar.slider("Gaze Weight", 0.1, 0.7, 0.4)
W_VOICE = st.sidebar.slider("Voice Weight", 0.1, 0.5, 0.3)
W_FACE = st.sidebar.slider("Face Weight", 0.1, 0.5, 0.3)

st.sidebar.markdown("### 🧮 Fusion Mode")
FUSION_METHOD = st.sidebar.selectbox(
    "Fusion Formula",
    options=["Adaptive Harmonic", "Adaptive Gaussian"],
    index=0,
    help="Use robust nonlinear fusion (harmonic or gaussian), not plain weighted average",
)
GAUSSIAN_SIGMA = st.sidebar.slider(
    "Gaussian Sigma", 0.15, 0.60, 0.35,
    help="Lower = stricter penalty on weak modality scores",
)

st.sidebar.markdown("### 📊 Dataset Collection")
ENABLE_LOGGING = st.sidebar.checkbox("Enable Dataset Logging", value=False,
    help="Log features to CSV for training")
LOGGING_LABEL = st.sidebar.radio(
    "Current Label",
    options=["Legitimate", "Suspicious"],
    horizontal=True,
    help="Label for logged samples"
) if ENABLE_LOGGING else "Legitimate"
LOGGING_LABEL_INT = 0 if LOGGING_LABEL == "Legitimate" else 1

st.sidebar.markdown("### 🤖 ML Fusion (Optional)")
USE_ML_FUSION = st.sidebar.checkbox("Use Trained ML Fusion", value=False,
    help="Use fusion_model.pkl instead of harmonic fusion")

st.sidebar.markdown("---")
st.sidebar.info("System Status: **ONLINE**")


class IntegrityState:
    """Communicates analysis results across threads + to dashboard."""
    def __init__(self):
        self.lock = threading.Lock()
        # Raw metrics
        self.audio_msg = "SILENT"
        self.jitter = 0.0
        self.shimmer = 0.0
        self.spectral_centroid = 0.0
        self.mfcc_flatness = 0.0
        # Derived sub-score (0-100, 100 = normal)
        self.voice_score = 100.0
        # Dashboard-facing values (written by BiometricAnalyzer)
        self.integrity_score = 100.0
        self.gaze_score = 100.0
        self.face_score = 100.0
        self.status_msg = "INITIALISING"
        self.anomaly_count = 0
        self.score_history = collections.deque(maxlen=120)  # ~60s at 2Hz
        self.time_history = collections.deque(maxlen=120)
        self.session_start = time.time()
        self.last_history_update = 0.0  # track when history was last updated
        # Seed initial data point
        self.score_history.append(100.0)
        self.time_history.append(0.0)
        
        # Calibration Baselines (for personalized detection)
        self.is_calibrated = False
        self.is_calibrating = False
        self.base_yaw = 0.0
        self.base_ear = 0.3
        self.base_pitch = 0.0
        
        # Adaptive Fusion State (tracks anomaly duration)
        self.anomaly_duration = 0.0
        
        # Micro-expression variance (copied from BiometricAnalyzer for logging)
        self.micro_var = 1.0

        # Object detection flag (copied from BiometricAnalyzer for logging)
        self.object_flag = False
        
        # Score smoothing buffer (rolling window for less jittery display)
        self.score_buffer = collections.deque(maxlen=15)  # 0.5 sec at 30fps
        self.integrity_ema = None
        self.integrity_ema_alpha = 0.2
        
        # Thread-safe fusion settings 
        self.fusion_method = "Adaptive Harmonic"
        self.use_ml_fusion = False
        self.w_gaze = 0.4
        self.w_voice = 0.3
        self.w_face = 0.3
        self.gaussian_sigma = 0.35

global_state = IntegrityState()

# Sync sidebar settings to thread-safe state
with global_state.lock:
    global_state.fusion_method = FUSION_METHOD
    global_state.use_ml_fusion = USE_ML_FUSION
    global_state.w_gaze = W_GAZE
    global_state.w_voice = W_VOICE
    global_state.w_face = W_FACE
    global_state.gaussian_sigma = GAUSSIAN_SIGMA

if "prev_ml_mode" not in st.session_state:
    st.session_state.prev_ml_mode = USE_ML_FUSION

mode_changed = (st.session_state.prev_ml_mode != USE_ML_FUSION)
if mode_changed:
    st.session_state.prev_ml_mode = USE_ML_FUSION
    st.session_state.prev_integrity_score = None
    st.session_state.ema_score = None
    st.session_state.last_ml_prob = None

    with global_state.lock:
        global_state.integrity_ema = None
        global_state.score_buffer.clear()
        global_state.integrity_score = 100.0

    print("[*] ML fusion mode switched -> reset smoothing state")


_ml_model = None
_ml_feature_cols = None

def load_ml_fusion_model():
    """Load the trained fusion model if available."""
    global _ml_model, _ml_feature_cols
    
    if _ml_model is not None:
        return _ml_model, _ml_feature_cols
    
    model_path = os.path.join(os.path.dirname(__file__), "fusion_model.pkl")
    
    if not os.path.exists(model_path):
        return None, None
    
    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        _ml_model = payload["model"]
        _ml_feature_cols = payload["feature_cols"]
        print(f"[ML Fusion] Loaded model from {model_path}")
        return _ml_model, _ml_feature_cols
    except Exception as e:
        print(f"[ML Fusion] Error loading model: {e}")
        return None, None


def load_optimal_threshold() -> float:
    """Load the security-optimized decision threshold from file."""
    thresh_path = os.path.join(os.path.dirname(__file__), "fusion_threshold.txt")
    if os.path.exists(thresh_path):
        try:
            with open(thresh_path, "r") as f:
                return float(f.read().strip())
        except Exception:
            pass
    return 0.50  # Default threshold


def ml_fusion_predict(gaze_score, face_score, voice_score, micro_var,
                      jitter, shimmer, spectral_centroid, anomaly_duration,
                      object_flag, final_integrity_score) -> float:
    """
    Get integrity score from trained ML model.
    Returns probability of legitimate * 100.
    Uses security-optimized threshold if available.
    """
    model, feature_cols = load_ml_fusion_model()
    
    if model is None:
        return None  # Fallback signal
    
    # Build feature vector in correct order
    feature_map = {
        "gaze_score": gaze_score,
        "face_score": face_score,
        "voice_score": voice_score,
        "micro_var": micro_var,
        "jitter": jitter,
        "shimmer": shimmer,
        "spectral_centroid": spectral_centroid,
        "anomaly_duration": anomaly_duration,
        "object_flag": int(object_flag),
        "final_integrity_score": final_integrity_score,
    }
    
    try:
        # Wrap it in np.array() so XGBoost accepts it
        X = np.array([[feature_map.get(col, 0.0) for col in feature_cols]])
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        # Keep integrity high when probability of class 0 (legitimate) is high.
        if classes is not None and 0 in classes:
            legit_idx = int(np.where(classes == 0)[0][0])
        else:
            legit_idx = 0
        
        ml_prob = float(proba[legit_idx])
        ml_prob = (0.85 * ml_prob) + 0.075
        raw_score = float(ml_prob * 100.0)
        
        # Apply security-optimized threshold for flagging
        # This only affects the binary decision, not the continuous score
        return raw_score
    except Exception as e:
        print(f"[ML Fusion] Prediction error: {e}")
        return None


#  HELPER FUNCTIONS

def get_aspect_ratio(top, bottom, right, left, w, h):
    """Eye Aspect Ratio (EAR) = vertical / horizontal distance."""
    t = np.array([top.x * w, top.y * h])
    b = np.array([bottom.x * w, bottom.y * h])
    r = np.array([right.x * w, right.y * h])
    l = np.array([left.x * w, left.y * h])
    vert = np.linalg.norm(t - b)
    horiz = np.linalg.norm(r - l)
    return vert / horiz if horiz > 0 else 0.0


def get_gaze_ratio(eye_inner, eye_outer, iris_center, w, h):
    """Horizontal gaze: 0 = left, 0.5 = center, 1 = right."""
    cx = iris_center.x * w
    ix = eye_inner.x * w
    ox = eye_outer.x * w
    lo, hi = min(ix, ox), max(ix, ox)
    span = hi - lo
    if span == 0:
        return 0.5
    cx = max(lo, min(hi, cx))
    return (cx - lo) / span


def get_vertical_gaze_ratio_linear(eye_inner, eye_outer, iris_center, w, h):
    """
    Vertical gaze using linear algebra relative to eye corners.
    Ethnicity-agnostic (independent of eyelid shape).
    < 0.5 = up,  0.5 = center,  > 0.5 = down
    """
    fy = h
    p1 = np.array([eye_inner.x, eye_inner.y * fy])
    p2 = np.array([eye_outer.x, eye_outer.y * fy])
    iris = np.array([iris_center.x, iris_center.y * fy])
    center_y = (p1[1] + p2[1]) / 2.0
    eye_w = np.sqrt(((p1[0] - p2[0]) * w) ** 2 + (p1[1] - p2[1]) ** 2)
    if eye_w == 0:
        return 0.5
    return 0.5 + (iris[1] - center_y) / eye_w


def compute_landmark_variance(history):
    """
    Compute average landmark position variance over a rolling window.
    """
    if len(history) < 2:
        return 1.0  # assume alive until enough data
    
    # Stack all frames: shape (num_frames, num_landmarks, 2)
    arr = np.array(history)
    # Variance across frames for each landmark coordinate
    var = np.var(arr, axis=0)
    # Average variance across all landmarks and x/y
    mean_var = np.mean(var)
    return float(mean_var)


def adaptive_harmonic_fusion(gaze_score, voice_score, face_score, anomaly_time,
                              w_gaze=0.4, w_voice=0.3, w_face=0.3):
    """Fuse modality scores with weighted harmonic mean and anomaly decay."""
    # Normalize inputs with a floor to avoid instability.
    g = max(15.0, gaze_score) / 100.0
    v = max(15.0, voice_score) / 100.0
    f = max(15.0, face_score) / 100.0
    
    # Increase weight on weak channels so low scores are reflected in fusion.
    w_g = w_gaze + (0.1 if g < 0.5 else 0)
    w_v = w_voice + (0.1 if v < 0.5 else 0)
    w_f = w_face + (0.1 if f < 0.5 else 0)
    total_w = w_g + w_v + w_f
    
    # Weighted harmonic mean.
    try:
        harmonic_mean = total_w / ((w_g / g) + (w_v / v) + (w_f / f))
    except ZeroDivisionError:
        harmonic_mean = 0.0
        
    # 4. Temporal Decay (apply during active anomaly windows)
    decay_factor = max(0.0, 1.0 - (0.15 * anomaly_time))
    
    # 5. Final Adaptive Harmonic Fusion: I(t) = 100 · H(t) · D(t)
    final_score = harmonic_mean * decay_factor * 100.0
    return int(max(0, min(100, final_score)))


def adaptive_gaussian_fusion(gaze_score, voice_score, face_score, anomaly_time,
                             w_gaze=0.4, w_voice=0.3, w_face=0.3,
                             sigma=0.35):
    """
    Adaptive Gaussian Fusion (AGF)

    Maps each normalized modality score to a gaussian reliability around "healthy"
    state (x=1.0), then combines reliabilities using weighted geometric mean.
    This penalizes low single-modality values more strongly than arithmetic mean.
    """
    g = np.clip(gaze_score / 100.0, 0.0, 1.0)
    v = np.clip(voice_score / 100.0, 0.0, 1.0)
    f = np.clip(face_score / 100.0, 0.0, 1.0)

    w_g = w_gaze + (0.1 if g < 0.5 else 0.0)
    w_v = w_voice + (0.1 if v < 0.5 else 0.0)
    w_f = w_face + (0.1 if f < 0.5 else 0.0)
    total_w = max(1e-9, w_g + w_v + w_f)

    sigma = max(1e-3, float(sigma))
    r_g = np.exp(-((1.0 - g) ** 2) / (2.0 * sigma * sigma))
    r_v = np.exp(-((1.0 - v) ** 2) / (2.0 * sigma * sigma))
    r_f = np.exp(-((1.0 - f) ** 2) / (2.0 * sigma * sigma))

    geometric_rel = np.exp(
        (w_g * np.log(r_g + 1e-12) + w_v * np.log(r_v + 1e-12) + w_f * np.log(r_f + 1e-12))
        / total_w
    )

    # Apply temporal decay only during active anomaly windows.
    decay_factor = max(0.0, 1.0 - (0.15 * anomaly_time))
    final_score = geometric_rel * decay_factor * 100.0
    return int(max(0, min(100, final_score)))


#  MODULE C — AUDIO FORENSICS

class AudioAnalyzer(AudioProcessorBase):
    """
    Real-time voice stress analysis.
    Biomarkers: Jitter  |  Shimmer  |  Spectral Centroid  |  MFCC flatness
    """

    def __init__(self):
        self.rate = 48000           # WebRTC typically 48 kHz
        self.chunk_buffer = np.array([], dtype=np.float32)
        self.buffer_target = 48000  # 1 second of audio

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        raw = frame.to_ndarray()
        self.rate = frame.sample_rate or self.rate
        self.buffer_target = self.rate  # keep 1 s

        # Convert to mono robustly for both layouts:
        # - planar/channel-first: (channels, samples)
        # - interleaved/sample-first: (samples, channels)
        if raw.ndim == 1:
            mono = raw.astype(np.float32, copy=False)
        elif raw.ndim == 2:
            # Heuristic: very small first dimension usually means channel-first.
            if raw.shape[0] <= 8 and raw.shape[1] > raw.shape[0]:
                mono = np.mean(raw, axis=0)
            else:
                mono = np.mean(raw, axis=1)
        else:
            mono = raw.reshape(-1)

        if mono.dtype != np.float32:
            mono = mono.astype(np.float32) / 32768.0

        self.chunk_buffer = np.concatenate((self.chunk_buffer, mono))

        if len(self.chunk_buffer) >= self.buffer_target:
            chunk = self.chunk_buffer[: self.buffer_target]
            self.chunk_buffer = self.chunk_buffer[self.buffer_target:]
            try:
                self._analyze(chunk)
            except Exception as e:
                print(f"[AudioAnalyzer] {e}")
        return frame

    def _analyze(self, y: np.ndarray):
        global global_state
        sr = self.rate

        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < 0.005:
            with global_state.lock:
                global_state.audio_msg = "SILENT"
                global_state.voice_score = 100.0
                global_state.jitter = 0.0
                global_state.shimmer = 0.0
            return

        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=60, fmax=500, sr=sr, frame_length=2048
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        jitter_pct = 0.0
        if len(f0_clean) > 2:
            # Relative jitter = mean |delta_period| / mean_period  x 100
            periods = 1.0 / f0_clean
            diffs = np.abs(np.diff(periods))
            jitter_pct = float(np.mean(diffs) / np.mean(periods) * 100.0)

        hop = 512
        frames_amp = librosa.util.frame(y, frame_length=2048, hop_length=hop)
        peak_amps = np.max(np.abs(frames_amp), axis=0)
        shimmer_pct = 0.0
        if len(peak_amps) > 2:
            diffs_a = np.abs(np.diff(peak_amps))
            shimmer_pct = float(np.mean(diffs_a) / (np.mean(peak_amps) + 1e-9) * 100.0)

        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        avg_cent = float(np.mean(cent))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # "Flatness" = ratio of geometric to arithmetic mean of MFCC variance
        mfcc_var = np.var(mfccs, axis=1) + 1e-12
        geo = np.exp(np.mean(np.log(mfcc_var)))
        ari = np.mean(mfcc_var)
        mfcc_flatness = float(geo / ari)  # closer to 1 = flatter = more synthetic

        score = 100.0
        msg_parts = []

        if jitter_pct > JITTER_THRESH:
            penalty = min(30.0, (jitter_pct - JITTER_THRESH) * 10.0)
            score -= penalty
            msg_parts.append(f"JITTER {jitter_pct:.1f}%")

        if shimmer_pct > SHIMMER_THRESH:
            penalty = min(20.0, (shimmer_pct - SHIMMER_THRESH) * 2.0)
            score -= penalty
            msg_parts.append(f"SHIMMER {shimmer_pct:.1f}%")

        if avg_cent > 3500:
            score -= 10.0
            msg_parts.append(f"HI-FREQ {int(avg_cent)}Hz")

        if mfcc_flatness > 0.6:
            score -= 15.0
            msg_parts.append("FLAT-VOICE")

        score = max(0.0, min(100.0, score))
        msg = " | ".join(msg_parts) if msg_parts else f"NORMAL ({int(avg_cent)}Hz)"

        with global_state.lock:
            global_state.voice_score = score
            global_state.jitter = jitter_pct
            global_state.shimmer = shimmer_pct
            global_state.spectral_centroid = avg_cent
            global_state.mfcc_flatness = mfcc_flatness
            global_state.audio_msg = msg


#  CORE ENGINE — VIDEO + FUSION

class BiometricAnalyzer(VideoProcessorBase):
    def __init__(self):
        # MediaPipe FaceMesh (478 landmarks with iris refinement)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.integrity_score = 100.0
        self.last_activity_time = time.time()
        self.blink_count = 0
        self.prev_gaze_pos = 0.5
        self.status_msg = "INITIALISING"
        self.status_color = (200, 200, 200)
        self.lie_confidence = 0.0

        # Sub-scores
        self.gaze_score = 100.0
        self.face_score = 100.0

        # Micro-expression variance buffer (rolling 30 snapshots)
        self.landmark_history = collections.deque(maxlen=30)
        self.micro_var = 1.0   # alive until proven otherwise

        # Timed gaze tracking
        self.gaze_away_start = None
        self.side_away_time = 0.0
        self.gaze_lateral_ema = 0.0

        try:
            # Resolve model path relative to this script
            _dir = os.path.dirname(os.path.abspath(__file__))
            _model = os.path.join(_dir, "efficientdet_lite0.tflite")
            base_opts = python.BaseOptions(
                model_asset_path=_model
            )
            opts = vision.ObjectDetectorOptions(
                base_options=base_opts, score_threshold=0.4
            )
            self.detector = vision.ObjectDetector.create_from_options(opts)
            self.detector_ready = True
        except Exception as e:
            print(f"[ObjectDetector] {e}")
            self.detector_ready = False

        self.frame_count = 0
        self.detected_objects = []
        self.banned_objects = [
            "cell phone", "mobile phone", "laptop", "computer",
            "tv", "monitor", "remote",
        ]

        # Calibration
        self.calibration_frames = 0
        self.base_vert_ratio = 0.5
        
        # Frame timing for accurate anomaly duration (not FPS-dependent)
        self._last_frame_time = time.time()

    #  recv — runs every frame (~30 Hz)
    def recv(self, frame):
        # Calculate real time delta between frames
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        suspicious_object_found = False
        forbidden_label = ""

        if self.detector_ready:
            # Configurable frame-skip for EfficientDet (default every 10 frames)
            if self.frame_count % CONTEXT_INTERVAL == 0:
                img_rgb_c = np.ascontiguousarray(img_rgb)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=img_rgb_c
                )
                self.detected_objects = self.detector.detect(mp_image).detections
            self.frame_count += 1

            for det in self.detected_objects:
                cat = det.categories[0]
                label, score = cat.category_name, cat.score
                bb = det.bounding_box
                pt1 = (bb.origin_x, bb.origin_y)
                pt2 = (bb.origin_x + bb.width, bb.origin_y + bb.height)

                if any(b in label.lower() for b in self.banned_objects) and score > OBJ_CONFIDENCE:
                    suspicious_object_found = True
                    forbidden_label = label.upper()
                    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)
                    cv2.putText(
                        img, f"BAN: {label} ({int(score*100)}%)",
                        (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )

        results = self.face_mesh.process(img_rgb)
        face_detected = False
        looking_away = False
        is_blinking = False

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark

            nose_x = lm[1].x
            left_ear_x, right_ear_x = lm[234].x, lm[454].x
            fw = right_ear_x - left_ear_x
            if fw > 0:
                rel_nose = (nose_x - left_ear_x) / fw
                if rel_nose < HEAD_TURN_SENSITIVITY or rel_nose > (1 - HEAD_TURN_SENSITIVITY):
                    looking_away = True

            # Sample key landmarks around cheeks, mouth, brows
            key_ids = [
                61, 291, 0, 17,       # mouth corners + top/bottom lip
                50, 280,              # cheeks
                70, 300,              # outer brows
                105, 334,             # inner brows
                10, 152,              # forehead / chin
            ]
            snapshot = [(lm[i].x * w, lm[i].y * h) for i in key_ids]
            self.landmark_history.append(snapshot)
            self.micro_var = compute_landmark_variance(self.landmark_history)

            l_ear = get_aspect_ratio(lm[386], lm[374], lm[263], lm[362], w, h)
            r_ear = get_aspect_ratio(lm[159], lm[145], lm[33], lm[133], w, h)
            avg_ear = (l_ear + r_ear) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                is_blinking = True
                self.blink_count += 1
                self.last_activity_time = time.time()

            l_gz = get_gaze_ratio(lm[362], lm[263], lm[468], w, h)
            r_gz = get_gaze_ratio(lm[33], lm[133], lm[473], w, h)
            avg_gz = (l_gz + r_gz) / 2.0

            l_vr = get_vertical_gaze_ratio_linear(lm[362], lm[263], lm[468], w, h)
            r_vr = get_vertical_gaze_ratio_linear(lm[33], lm[133], lm[473], w, h)
            avg_vr = (l_vr + r_vr) / 2.0

            # Auto-calibrate first 30 frames
            if self.calibration_frames < 30:
                self.base_vert_ratio = avg_vr
                self.calibration_frames += 1
            norm_vr = 0.5 + (avg_vr - self.base_vert_ratio)

            # Gaze deviation checks
            if abs(avg_gz - 0.5) > GAZE_SENSITIVITY:
                looking_away = True
            if abs(norm_vr - 0.5) > GAZE_SENSITIVITY:
                looking_away = True

            # Side-diversion detector: smooth horizontal drift + hold-time gate.
            # This is more reliable than a single-frame threshold.
            lateral_dev = abs(avg_gz - 0.5)
            self.gaze_lateral_ema = (0.8 * self.gaze_lateral_ema) + (0.2 * lateral_dev)
            if self.gaze_lateral_ema > SIDE_GAZE_THRESHOLD:
                self.side_away_time += dt
            else:
                self.side_away_time = max(0.0, self.side_away_time - (2.0 * dt))

            if self.side_away_time >= SIDE_GAZE_HOLD:
                looking_away = True

            # Immediate trigger for extreme lateral glance.
            if lateral_dev > (SIDE_GAZE_THRESHOLD + 0.07):
                looking_away = True

            # Track micro-saccades (proof of life)
            if abs(avg_gz - self.prev_gaze_pos) > 0.005:
                self.last_activity_time = time.time()
            self.prev_gaze_pos = avg_gz

            if avg_gz > LIE_HORIZ_THRESH and norm_vr < LIE_VERT_THRESH:
                self.lie_confidence = min(100.0, self.lie_confidence + 3.0)
            else:
                self.lie_confidence = max(0.0, self.lie_confidence - 1.0)

            mp_drawing.draw_landmarks(
                img, results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_TESSELATION, None,
                mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                img, results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_IRISES, None,
                mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        if not face_detected:
            self.gaze_score = 0.0
        elif looking_away:
            # Timed penalty for sustained diversion
            if self.gaze_away_start is None:
                self.gaze_away_start = time.time()
            away_duration = time.time() - self.gaze_away_start
            if away_duration > 3.0:
                # -10 per second beyond 3 s, down to 0
                self.gaze_score = max(0.0, self.gaze_score - 10.0)
            else:
                # Small nudge during grace period
                self.gaze_score = max(0.0, self.gaze_score - 1.0)
        else:
            self.gaze_away_start = None
            self.gaze_score = min(100.0, self.gaze_score + 3.0)  # recovery

        # Lie confidence drags gaze score
        if self.lie_confidence > 50:
            self.gaze_score = max(0.0, self.gaze_score - 5.0)

        if not face_detected:
            self.face_score = max(0.0, self.face_score - 8.0)
        elif (time.time() - self.last_activity_time) > STILL_FAKE_SECONDS:
            self.face_score = max(0.0, self.face_score - 3.0)   # stillness
        elif self.micro_var < 0.08:
            # Very low landmark variance can indicate spoofed/static face.
            self.face_score = max(0.0, self.face_score - 2.0)
        else:
            self.face_score = min(100.0, self.face_score + 2.0)

        with global_state.lock:
            voice_score = global_state.voice_score
            audio_msg = global_state.audio_msg
            jitter_val = global_state.jitter
            shimmer_val = global_state.shimmer

        
        # Track anomaly duration for temporal decay
        is_anomaly = (self.gaze_score < 50) or (voice_score < 50) or (self.face_score < 50)
        
        with global_state.lock:
            if is_anomaly:
                # Increase timer using real time delta (frame-rate independent)
                global_state.anomaly_duration += dt
            else:
                # Recovery (cool down at 3x rate of accumulation)
                global_state.anomaly_duration = max(0.0, global_state.anomaly_duration - (dt * 3))
            anomaly_time = global_state.anomaly_duration

        # Do not carry temporal penalty into normal frames.
        effective_anomaly_time = anomaly_time if is_anomaly else 0.0
        
        # Choose fusion method based on thread-safe state
        with global_state.lock:
            fusion_method = global_state.fusion_method
            use_ml_fusion = global_state.use_ml_fusion
            w_gaze = global_state.w_gaze
            w_voice = global_state.w_voice
            w_face = global_state.w_face
            gaussian_sigma = global_state.gaussian_sigma
            spectral_centroid = global_state.spectral_centroid
        
        # ML Fusion (if enabled and model available)
        if use_ml_fusion:
            ml_score = ml_fusion_predict(
                gaze_score=self.gaze_score,
                face_score=self.face_score,
                voice_score=voice_score,
                micro_var=self.micro_var,
                jitter=jitter_val,
                shimmer=shimmer_val,
                spectral_centroid=spectral_centroid,
                anomaly_duration=anomaly_time,
                object_flag=suspicious_object_found,
                final_integrity_score=self.integrity_score,
            )
            if ml_score is not None:
                fused = ml_score
            else:
                # Fallback to selected robust nonlinear fusion
                if fusion_method == "Adaptive Gaussian":
                    fused = adaptive_gaussian_fusion(
                        self.gaze_score, voice_score, self.face_score,
                        effective_anomaly_time, w_gaze, w_voice, w_face, gaussian_sigma
                    )
                else:
                    fused = adaptive_harmonic_fusion(
                        self.gaze_score, voice_score, self.face_score,
                        effective_anomaly_time, w_gaze, w_voice, w_face
                    )
        else:
            if fusion_method == "Adaptive Gaussian":
                fused = adaptive_gaussian_fusion(
                    self.gaze_score, voice_score, self.face_score,
                    effective_anomaly_time, w_gaze, w_voice, w_face, gaussian_sigma
                )
            else:
                fused = adaptive_harmonic_fusion(
                    self.gaze_score, voice_score, self.face_score,
                    effective_anomaly_time, w_gaze, w_voice, w_face
                )

        # "Looking away AND voice trembling -> -40 (Coercion Red Alert)"
        # ── Multimodal synergy penalties ──────────────────────────────────
        # Additional penalty for simultaneous gaze diversion and voice stress.
        if looking_away and jitter_val > JITTER_THRESH:
            fused -= 20.0

        # Banned object override
        if suspicious_object_found:
            object_penalty = 18.0
            fused -= object_penalty

        raw_fused = max(0.0, min(100.0, fused))

        if use_ml_fusion:
            if raw_fused > 96.0:
                raw_fused = 96.0 + (raw_fused - 96.0) * 0.15

            # Apply rolling buffer smoothing for less jittery display
            # Score buffer averages over last N frames for smoother UI
            with global_state.lock:
                global_state.score_buffer.append(raw_fused)
                if global_state.integrity_ema is None:
                    global_state.integrity_ema = raw_fused
                else:
                    alpha = global_state.integrity_ema_alpha
                    global_state.integrity_ema = (alpha * raw_fused) + ((1.0 - alpha) * global_state.integrity_ema)
                self.integrity_score = float(global_state.integrity_ema)
        else:
            self.integrity_score = float(raw_fused)
        
        with global_state.lock:
            global_state.integrity_score = self.integrity_score
            global_state.gaze_score = self.gaze_score
            global_state.face_score = self.face_score
            global_state.micro_var = self.micro_var  # For feature logging
            global_state.object_flag = suspicious_object_found  # For feature logging
            if self.frame_count % 15 == 0:  # ~2 Hz sample rate for graph
                elapsed = time.time() - global_state.session_start
                global_state.score_history.append(self.integrity_score)
                global_state.time_history.append(elapsed)
                global_state.last_history_update = elapsed
            if self.integrity_score < 50:
                global_state.anomaly_count += 1

        if suspicious_object_found:
            self.status_msg = "DEVICE: " + forbidden_label
            self.status_color = (0, 0, 255)
        elif not face_detected:
            self.status_msg = "NO SUBJECT"
            self.status_color = (0, 0, 255)
        elif looking_away and jitter_val > JITTER_THRESH:
            self.status_msg = "COERCION ALERT"
            self.status_color = (0, 0, 200)
        elif self.lie_confidence > 50:
            self.status_msg = "LIE DETECTED"
            self.status_color = (0, 0, 180)
        elif self.side_away_time >= SIDE_GAZE_HOLD:
            self.status_msg = f"SIDE DIVERSION ({self.side_away_time:.1f}s)"
            self.status_color = (0, 140, 255)
        elif looking_away:
            away_t = (time.time() - self.gaze_away_start) if self.gaze_away_start else 0
            self.status_msg = f"DISTRACTED ({away_t:.0f}s)"
            self.status_color = (0, 165, 255)
        elif self.micro_var < 0.08 and (time.time() - self.last_activity_time) > STILL_FAKE_SECONDS:
            self.status_msg = "FROZEN (DEEPFAKE?)"
            self.status_color = (255, 0, 255)
        elif (time.time() - self.last_activity_time) > STILL_FAKE_SECONDS:
            self.status_msg = "STILL (FAKE?)"
            self.status_color = (255, 0, 255)
        elif voice_score < 60:
            self.status_msg = "VOICE STRESS"
            self.status_color = (0, 100, 255)
        else:
            self.status_msg = "VERIFIED"
            self.status_color = (0, 255, 0)

        # Sync status to dashboard
        with global_state.lock:
            global_state.status_msg = self.status_msg

        #  HUD OVERLAY
        bar_h = 70
        cv2.rectangle(img, (0, 0), (w, bar_h), (20, 20, 20), -1)

        # Progress bar
        bar_w = int((self.integrity_score / 100.0) * (w - 40))
        bar_color = (
            (0, 255, 0) if self.integrity_score > 70
            else (0, 165, 255) if self.integrity_score > 40
            else (0, 0, 255)
        )
        cv2.rectangle(img, (20, 55), (20 + bar_w, 65), bar_color, -1)
        cv2.rectangle(img, (20, 55), (w - 20, 65), (80, 80, 80), 1)

        # Main text
        cv2.putText(
            img, f"INTEGRITY: {int(self.integrity_score)}%",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            img, f"STATUS: {self.status_msg}",
            (w - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.status_color, 2,
        )
        # Sub-scores (small)
        sub = f"G:{int(self.gaze_score)}  F:{int(self.face_score)}  V:{int(voice_score)}  | {audio_msg}"
        cv2.putText(
            img, sub, (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


#  RTC CONFIGURATION (Cloud Deployment)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


#  CUSTOM CSS — Dark Cybersecurity Theme

DARK_CSS = """
<style>
/* ── Global dark overrides ─────────────────────────────────────────────── */
.stApp {
    background-color: #0a0e17;
    color: #c5cdd9;
}
section[data-testid="stSidebar"] {
    background-color: #0d1220 !important;
    border-right: 1px solid #1a2744;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4fc3f7;
}

/* ── KPI metric cards ──────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 0 20px rgba(0,150,255,0.08);
}
div[data-testid="stMetric"] label {
    color: #7eb8da !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700;
}

/* ── Title styling ─────────────────────────────────────────────────────── */
.dashboard-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e0e6ed;
    letter-spacing: 2px;
    border-bottom: 2px solid #1e3a5f;
    padding-bottom: 10px;
    margin-bottom: 6px;
}
.dashboard-subtitle {
    color: #5a7994;
    font-size: 0.85rem;
    margin-bottom: 18px;
}

/* ── Flashing alert for SUSPICIOUS status ──────────────────────────────── */
@keyframes threat-pulse {
    0%   { opacity: 1; }
    50%  { opacity: 0.3; }
    100% { opacity: 1; }
}
.threat-flash {
    animation: threat-pulse 0.8s ease-in-out infinite;
    color: #ff1744 !important;
    font-weight: 900;
    font-size: 1.4rem;
    text-shadow: 0 0 16px rgba(255,23,68,0.6);
}
.status-secure {
    color: #00e676;
    font-weight: 700;
    font-size: 1.4rem;
    text-shadow: 0 0 12px rgba(0,230,118,0.4);
}

/* ── Module status pills ───────────────────────────────────────────────── */
.module-pill {
    display: inline-block;
    background: #12202f;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 0.78rem;
    color: #90caf9;
    width: 100%;
}
.module-pill .dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00e676;
    margin-right: 8px;
    box-shadow: 0 0 6px #00e676;
}

/* ── Plotly chart container ────────────────────────────────────────────── */
.stPlotlyChart {
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    overflow: hidden;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


#  DASHBOARD RENDERING

def build_live_chart(times, scores):
    """Build a live timeline chart with moving cursor and fail-touch markers."""
    fig = go.Figure()

    # Score trace
    fig.add_trace(go.Scatter(
        x=list(times), y=list(scores),
        mode="lines",
        line=dict(color="#00e5ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.07)",
        name="Integrity",
    ))

    if times and scores:
        now_x = times[-1]
        fig.add_vline(
            x=now_x,
            line_dash="dot",
            line_color="#00e676",
            line_width=1.6,
        )

        fig.add_trace(go.Scatter(
            x=[now_x],
            y=[scores[-1]],
            mode="markers",
            marker=dict(color="#00e676", size=8, line=dict(color="#0d1b2a", width=1)),
            name="Now",
            hovertemplate="Now: %{y:.1f}<extra></extra>",
        ))

        fail_x = [t for t, s in zip(times, scores) if s <= 50]
        fail_y = [s for s in scores if s <= 50]
        if fail_x:
            pulse_size = 8 if int(time.time() * 2) % 2 == 0 else 11
            fig.add_trace(go.Scatter(
                x=fail_x,
                y=fail_y,
                mode="markers",
                marker=dict(color="#ff1744", size=pulse_size, symbol="circle"),
                name="Threshold Touch",
                hovertemplate="Threshold touch: %{y:.1f}<extra></extra>",
            ))

    # Failure threshold line
    if times:
        fig.add_hline(
            y=50, line_dash="dash", line_color="#ff1744", line_width=1.5,
            annotation_text="FAIL THRESHOLD",
            annotation_position="top left",
            annotation_font_color="#ff1744",
            annotation_font_size=10,
        )

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        font=dict(color="#7eb8da", size=11),
        xaxis=dict(
            title="Session Time (s)",
            gridcolor="#1e3a5f",
            zerolinecolor="#1e3a5f",
        ),
        yaxis=dict(
            title="Score", range=[0, 105],
            gridcolor="#1e3a5f",
            zerolinecolor="#1e3a5f",
        ),
        showlegend=False,
    )
    return fig


def draw_dashboard(score, status, anomaly_count, gaze, face, voice,
                   audio_msg, times, scores,
                   kpi_ph, chart_ph, subscore_ph, frame=0):
    """Update only the live dashboard placeholders (called in a loop)."""

    with kpi_ph.container():
        k1, k2, k3 = st.columns(3)
        score_delta = f"{score - 100:+.0f}" if score < 100 else "Nominal"
        with k1:
            st.metric("INTEGRITY SCORE", f"{int(score)}%", delta=score_delta,
                      delta_color="inverse")
        with k2:
            st.metric("ANOMALY COUNT", str(anomaly_count),
                      delta=f"{anomaly_count} events", delta_color="off")
        with k3:
            is_secure = score >= 50
            status_label = status if status != "INITIALISING" else "STANDBY"
            if is_secure:
                st.markdown(f'<p style="margin:0;color:#7eb8da;font-size:0.85rem;'
                            f'text-transform:uppercase;letter-spacing:1.2px;">'
                            f'SESSION STATUS</p>'
                            f'<p class="status-secure">{status_label}</p>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="margin:0;color:#7eb8da;font-size:0.85rem;'
                            f'text-transform:uppercase;letter-spacing:1.2px;">'
                            f'SESSION STATUS</p>'
                            f'<p class="threat-flash">⚠ {status_label}</p>',
                            unsafe_allow_html=True)

    chart_ph.empty()
    with chart_ph.container():
        st.markdown("##### 📈 Integrity Timeline")
        st.plotly_chart(
            build_live_chart(times, scores),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"integrity_chart_{frame}",
        )

    with subscore_ph.container():
        st.markdown("##### 🔬 Sub-Score Breakdown")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Gaze", f"{int(gaze)}")
        sc2.metric("Face", f"{int(face)}")
        sc3.metric("Voice", f"{int(voice)}")
        st.caption(f"Audio: {audio_msg}")


#  STREAMLIT LAYOUT — static elements + live-updating placeholders

st.markdown('<div class="dashboard-title">🛡️ SDP-1 &mdash; THREAT COMMAND CENTER</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Real-Time Multimodal Biometric Integrity Analysis &bull; Face Liveness &bull; Gaze Attention &bull; Voice Forensics &bull; Object Detection</div>', unsafe_allow_html=True)

kpi_placeholder = st.empty()

st.markdown("")

vid_col, chart_col = st.columns([2, 1])

with vid_col:
    if ENABLE_WEBRTC:
        media_constraints = {"video": True, "audio": bool(ENABLE_WEBRTC_AUDIO)}
        audio_factory = AudioAnalyzer if ENABLE_WEBRTC_AUDIO else None

        webrtc_ctx = webrtc_streamer(
            key="integrity-engine",
            video_processor_factory=BiometricAnalyzer,
            audio_processor_factory=audio_factory,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints=media_constraints,
            async_processing=True,
        )

        if not ENABLE_WEBRTC_AUDIO:
            st.info("WebRTC audio capture is OFF (compatibility mode). Enable it from sidebar if your browser supports media devices.")
    else:
        webrtc_ctx = None
        st.warning("Camera/WebRTC is currently OFF.")
        st.caption("If you saw a media device error earlier, open this app on localhost or HTTPS, then enable 'Camera/WebRTC' again.")
        st.code("http://localhost:8501")

with chart_col:
    chart_placeholder = st.empty()
    subscore_placeholder = st.empty()

st.markdown("---")
b1, b2, b3 = st.columns([2, 2, 1])

with b1:
    st.markdown("##### Active Modules")
    modules = [
        ("A", "Visual Liveness", "Face Mesh 478 + Micro-expression"),
        ("B", "Ocular Attention", "EAR Blink + Gaze Vector + Lie"),
        ("C", "Audio Forensics", "Jitter + Shimmer + MFCC + Spectral"),
        ("D", "Object Detection", "EfficientDet-Lite0 Banned Devices"),
    ]
    for code, name, desc in modules:
        st.markdown(
            f'<div class="module-pill"><span class="dot"></span>'
            f'<strong>{code}</strong> &mdash; {name} '
            f'<span style="color:#5a7994">({desc})</span></div>',
            unsafe_allow_html=True,
        )

with b2:
    st.markdown("##### Synergy Alert Rules")
    st.error("**Coercion**: Distracted gaze + Voice jitter → **-40 pts**")
    st.warning("**Deepfake**: Frozen landmarks + Low blink → **Face drain**")
    st.info("**Cheating**: Banned device detected → **-30 pts**")

with b3:
    st.markdown("##### Fusion")
    st.latex(r"\small I = \frac{w_g G + w_v V + w_f F}{\Sigma w}")
    st.caption(f"w_g={W_GAZE}  w_v={W_VOICE}  w_f={W_FACE}")


# Initialize feature logger if enabled (session-scoped)
if "feature_logger" not in st.session_state:
    st.session_state.feature_logger = None

if ENABLE_LOGGING and st.session_state.feature_logger is None:
    st.session_state.feature_logger = FeatureLogger()
    st.sidebar.success(f"💾 Logging to: {st.session_state.feature_logger.csv_path}")
elif not ENABLE_LOGGING and st.session_state.feature_logger is not None:
    st.session_state.feature_logger.close()
    st.session_state.feature_logger = None

_frame_counter = 0
while True:
    with global_state.lock:
        _score     = global_state.integrity_score
        _status    = global_state.status_msg
        _anomalies = global_state.anomaly_count
        _gaze      = global_state.gaze_score
        _face      = global_state.face_score
        _voice     = global_state.voice_score
        _audio_msg = global_state.audio_msg
        _object_flag = global_state.object_flag
        
        # Add data point if video processor hasn't updated recently (keeps graph alive)
        elapsed = time.time() - global_state.session_start
        if elapsed - global_state.last_history_update > 0.5:
            global_state.score_history.append(_score)
            global_state.time_history.append(elapsed)
            global_state.last_history_update = elapsed
        
        _times     = list(global_state.time_history)
        _scores    = list(global_state.score_history)

    # Dataset logging (modular, non-blocking)
    if st.session_state.feature_logger is not None:
        features = extract_features_from_state(global_state, object_detected=_object_flag)
        st.session_state.feature_logger.log_features(features, LOGGING_LABEL_INT)

    draw_dashboard(
        score=_score,
        status=_status,
        anomaly_count=_anomalies,
        gaze=_gaze,
        face=_face,
        voice=_voice,
        audio_msg=_audio_msg,
        times=_times,
        scores=_scores,
        kpi_ph=kpi_placeholder,
        chart_ph=chart_placeholder,
        subscore_ph=subscore_placeholder,
        frame=_frame_counter,
    )

    _frame_counter += 1
    time.sleep(1)

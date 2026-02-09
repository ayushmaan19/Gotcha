"""
SDP-1: Real-Time Multimodal Biometric Integrity Analysis
=========================================================
Whitepaper-compliant implementation.
Modules:
  A. Visual Liveness  (Face Mesh 468 landmarks, 3D geometry, micro-expression variance)
  B. Ocular Attention (EAR blink, horizontal + vertical gaze, lie detection)
  C. Audio Forensics  (Jitter, Shimmer, Spectral Centroid, MFCCs)
  D. Object Detection (EfficientDet-Lite0 banned-device scan)
Fusion: Weighted Sum Rule  0.4*Gaze + 0.3*Voice + 0.3*Face
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import cv2
import av
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import streamlit as st
import time
import threading
import collections

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase
import librosa

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SDP-1: Biometric Integrity",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Controls ────────────────────────────────────────────────────────
st.sidebar.title("🔧 System Controls")

st.sidebar.markdown("### 🎯 Sensitivity")
HEAD_TURN_SENSITIVITY = st.sidebar.slider(
    "Head Turn Sensitivity", 0.20, 0.50, 0.40,
    help="Higher = stricter (0.5 = any tilt flagged)")
GAZE_SENSITIVITY = st.sidebar.slider(
    "Eye Gaze Sensitivity", 0.02, 0.15, 0.05,
    help="Lower = stricter (detects smaller eye movements)")
BLINK_THRESHOLD = st.sidebar.slider(
    "Blink Threshold (EAR)", 0.10, 0.40, 0.25)

st.sidebar.markdown("### 🧠 Lie Detection")
LIE_HORIZ_THRESH = st.sidebar.slider(
    "Lie Gaze Horizontal (>)", 0.50, 0.80, 0.55,
    help="Lower = more sensitive to rightward gaze")
LIE_VERT_THRESH = st.sidebar.slider(
    "Lie Gaze Vertical (<)", 0.20, 0.60, 0.45,
    help="Higher = more sensitive to upward gaze")

st.sidebar.markdown("### 📱 Object Detection")
OBJ_CONFIDENCE = st.sidebar.slider(
    "Device Confidence", 0.40, 0.90, 0.65,
    help="Increase to reduce false positives")

st.sidebar.markdown("### 🎙️ Audio Forensics")
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

st.sidebar.markdown("---")
st.sidebar.info("System Status: **ONLINE**")


# ── Shared Thread-Safe State ────────────────────────────────────────────────
class IntegrityState:
    """Communicates audio analysis results to the video thread."""
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

global_state = IntegrityState()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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
    Micro-expression detector.
    Measures the average per-landmark positional variance over a rolling window.
    High variance = alive human.  Near-zero = photo / frozen deepfake.
    """
    if len(history) < 2:
        return 1.0  # assume alive until enough data
    arr = np.array(history)            # shape (T, N, 2)
    var = np.var(arr, axis=0).mean()   # average variance across landmarks & axes
    return float(var)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE C — AUDIO FORENSICS
# ══════════════════════════════════════════════════════════════════════════════

class AudioAnalyzer(AudioProcessorBase):
    """
    Real-time voice stress analysis.
    Biomarkers: Jitter  |  Shimmer  |  Spectral Centroid  |  MFCC flatness
    """

    def __init__(self):
        self.rate = 48000           # WebRTC typically 48 kHz
        self.chunk_buffer = np.array([], dtype=np.float32)
        self.buffer_target = 48000  # 1 second of audio

    # ── WebRTC callback ──────────────────────────────────────────────────
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        raw = frame.to_ndarray()
        self.rate = frame.sample_rate or self.rate
        self.buffer_target = self.rate  # keep 1 s

        # stereo -> mono
        mono = np.mean(raw, axis=1) if raw.ndim > 1 and raw.shape[1] > 1 else raw.flatten()
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

    # ── Core analysis ────────────────────────────────────────────────────
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

        # ── 1. Pitch (F0) via pYIN -> Jitter ─────────────────────────────
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

        # ── 2. Shimmer (amplitude perturbation) ─────────────────────────
        hop = 512
        frames_amp = librosa.util.frame(y, frame_length=2048, hop_length=hop)
        peak_amps = np.max(np.abs(frames_amp), axis=0)
        shimmer_pct = 0.0
        if len(peak_amps) > 2:
            diffs_a = np.abs(np.diff(peak_amps))
            shimmer_pct = float(np.mean(diffs_a) / (np.mean(peak_amps) + 1e-9) * 100.0)

        # ── 3. Spectral Centroid ─────────────────────────────────────────
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        avg_cent = float(np.mean(cent))

        # ── 4. MFCCs -> flatness (AI voice proxy) ────────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # "Flatness" = ratio of geometric to arithmetic mean of MFCC variance
        mfcc_var = np.var(mfccs, axis=1) + 1e-12
        geo = np.exp(np.mean(np.log(mfcc_var)))
        ari = np.mean(mfcc_var)
        mfcc_flatness = float(geo / ari)  # closer to 1 = flatter = more synthetic

        # ── Derive voice sub-score (100 = normal) ───────────────────────
        score = 100.0
        msg_parts = []

        if jitter_pct > JITTER_THRESH:
            penalty = min(30.0, (jitter_pct - JITTER_THRESH) * 10.0)
            score -= penalty
            msg_parts.append(f"JITTER {jitter_pct:.1f}%")

        if shimmer_pct > SHIMMER_THRESH:
            penalty = min(30.0, (shimmer_pct - SHIMMER_THRESH) * 5.0)
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


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ENGINE — VIDEO + FUSION
# ══════════════════════════════════════════════════════════════════════════════

class BiometricAnalyzer(VideoProcessorBase):
    def __init__(self):
        # MediaPipe FaceMesh (478 landmarks with iris refinement)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── State ────────────────────────────────────────────────────────
        self.integrity_score = 100.0
        self.last_activity_time = time.time()
        self.blink_count = 0
        self.prev_gaze_pos = 0.5
        self.status_msg = "INITIALISING"
        self.status_color = (200, 200, 200)
        self.lie_confidence = 0.0

        # Sub-scores (whitepaper fusion)
        self.gaze_score = 100.0
        self.face_score = 100.0

        # Micro-expression variance buffer (rolling 30 snapshots)
        self.landmark_history = collections.deque(maxlen=30)
        self.micro_var = 1.0   # alive until proven otherwise

        # Timed gaze tracking (whitepaper: > 3 s -> penalty)
        self.gaze_away_start = None

        # ── Object Detector ──────────────────────────────────────────────
        try:
            base_opts = python.BaseOptions(
                model_asset_path="backend/efficientdet_lite0.tflite"
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

    # ══════════════════════════════════════════════════════════════════════
    #  recv — runs every frame (~30 Hz)
    # ══════════════════════════════════════════════════════════════════════
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── OBJECT DETECTION (every 10 frames) ──────────────────────────
        suspicious_object_found = False
        forbidden_label = ""

        if self.detector_ready:
            if self.frame_count % 10 == 0:
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

        # ── FACE MESH ────────────────────────────────────────────────────
        results = self.face_mesh.process(img_rgb)
        face_detected = False
        looking_away = False
        is_blinking = False

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark

            # ── A1. HEAD POSE (3D geometry) ──────────────────────────────
            nose_x = lm[1].x
            left_ear_x, right_ear_x = lm[234].x, lm[454].x
            fw = right_ear_x - left_ear_x
            if fw > 0:
                rel_nose = (nose_x - left_ear_x) / fw
                if rel_nose < HEAD_TURN_SENSITIVITY or rel_nose > (1 - HEAD_TURN_SENSITIVITY):
                    looking_away = True

            # ── A2. MICRO-EXPRESSION VARIANCE ────────────────────────────
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

            # ── B1. BLINK (EAR) ──────────────────────────────────────────
            l_ear = get_aspect_ratio(lm[386], lm[374], lm[263], lm[362], w, h)
            r_ear = get_aspect_ratio(lm[159], lm[145], lm[33], lm[133], w, h)
            avg_ear = (l_ear + r_ear) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                is_blinking = True
                self.blink_count += 1
                self.last_activity_time = time.time()

            # ── B2. GAZE (horizontal + vertical) ─────────────────────────
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

            # Strict deviation check
            if abs(avg_gz - 0.5) > GAZE_SENSITIVITY:
                looking_away = True
            if abs(norm_vr - 0.5) > GAZE_SENSITIVITY:
                looking_away = True
            if avg_gz < 0.4 or avg_gz > 0.6:
                looking_away = True

            # Track micro-saccades (proof of life)
            if abs(avg_gz - self.prev_gaze_pos) > 0.005:
                self.last_activity_time = time.time()
            self.prev_gaze_pos = avg_gz

            # ── B3. LIE DETECTION (Up-Right cluster) ─────────────────────
            if avg_gz > LIE_HORIZ_THRESH and norm_vr < LIE_VERT_THRESH:
                self.lie_confidence = min(100.0, self.lie_confidence + 3.0)
            else:
                self.lie_confidence = max(0.0, self.lie_confidence - 1.0)

            # ── Draw mesh + irises ───────────────────────────────────────
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

        # ══════════════════════════════════════════════════════════════════
        #  WEIGHTED FUSION ALGORITHM  (Whitepaper Section 4)
        #  IntegrityScore = W_GAZE * GazeScore + W_VOICE * VoiceScore
        #                 + W_FACE * FaceScore
        # ══════════════════════════════════════════════════════════════════

        # ── Gaze sub-score (0-100) ───────────────────────────────────────
        if not face_detected:
            self.gaze_score = 0.0
        elif looking_away:
            # Timed penalty: whitepaper says > 3 s -> big penalty
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

        # ── Face sub-score (0-100) ───────────────────────────────────────
        if not face_detected:
            self.face_score = max(0.0, self.face_score - 15.0)
        elif (time.time() - self.last_activity_time) > 60.0:
            self.face_score = max(0.0, self.face_score - 5.0)   # stillness
        elif self.micro_var < 0.08:
            # Very low landmark variance -> possibly photo / frozen deepfake
            self.face_score = max(0.0, self.face_score - 3.0)
        else:
            self.face_score = min(100.0, self.face_score + 2.0)

        # ── Voice sub-score (from AudioAnalyzer) ─────────────────────────
        with global_state.lock:
            voice_score = global_state.voice_score
            audio_msg = global_state.audio_msg
            jitter_val = global_state.jitter
            shimmer_val = global_state.shimmer

        # ── FUSED INTEGRITY SCORE ────────────────────────────────────────
        total_w = W_GAZE + W_VOICE + W_FACE
        fused = (
            (W_GAZE * self.gaze_score)
            + (W_VOICE * voice_score)
            + (W_FACE * self.face_score)
        ) / total_w

        # ── Multimodal synergy penalties (whitepaper Section 4) ──────────
        # "Looking away AND voice trembling -> -40 (Coercion Red Alert)"
        if looking_away and jitter_val > JITTER_THRESH:
            fused -= 40.0

        # Banned object override
        if suspicious_object_found:
            fused -= 30.0

        self.integrity_score = max(0.0, min(100.0, fused))

        # ── Status message ───────────────────────────────────────────────
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
        elif looking_away:
            away_t = (time.time() - self.gaze_away_start) if self.gaze_away_start else 0
            self.status_msg = f"DISTRACTED ({away_t:.0f}s)"
            self.status_color = (0, 165, 255)
        elif self.micro_var < 0.08:
            self.status_msg = "FROZEN (DEEPFAKE?)"
            self.status_color = (255, 0, 255)
        elif (time.time() - self.last_activity_time) > 60.0:
            self.status_msg = "STILL (FAKE?)"
            self.status_color = (255, 0, 255)
        elif voice_score < 60:
            self.status_msg = "VOICE STRESS"
            self.status_color = (0, 100, 255)
        else:
            self.status_msg = "VERIFIED"
            self.status_color = (0, 255, 0)

        # ══════════════════════════════════════════════════════════════════
        #  HUD OVERLAY
        # ══════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("SDP-1: Real-Time Multimodal Biometric Integrity")
st.caption("Continuous authentication via Face Liveness - Gaze Attention - Voice Forensics - Object Detection")

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_streamer(
        key="integrity-engine",
        video_processor_factory=BiometricAnalyzer,
        audio_processor_factory=AudioAnalyzer,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

with col2:
    st.markdown("#### Module Status")
    st.success("A — Visual Liveness (Face Mesh 478 + Micro-expression Variance)")
    st.success("B — Ocular Attention (EAR Blink + Gaze Vector + Lie Detection)")
    st.success("C — Audio Forensics (Jitter + Shimmer + MFCC + Spectral Centroid)")
    st.success("D — Object Detection (EfficientDet-Lite0 Banned Devices)")

    st.markdown("---")
    st.markdown("#### Fusion Formula")
    st.latex(r"\text{Score} = \frac{w_g \cdot G + w_v \cdot V + w_f \cdot F}{w_g + w_v + w_f}")
    st.caption(f"Weights — Gaze: {W_GAZE} | Voice: {W_VOICE} | Face: {W_FACE}")

    st.markdown("---")
    st.markdown("#### Synergy Alerts")
    st.error("**Coercion**: Distracted gaze + Voice jitter = **-40 pts**")
    st.warning("**Deepfake**: Frozen landmarks + Low blink = **Face score drain**")
    st.info("**Cheating**: Device detected = **-30 pts immediate**")

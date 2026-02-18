"""
SDP-1: Real Face Data Collector
================================
Records YOUR face data for training a personalized integrity model.

Usage:
  python collect_data.py

Controls:
  Hold '0' → Record LEGITIMATE behavior (looking at screen, natural)
  Hold '1' → Record SUSPICIOUS behavior (looking away, phone, etc.)
  Press 'q' → Quit and save data
"""

import csv
import cv2
import mediapipe as mp
import numpy as np
import os
import re

# ── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks for EAR calculation
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio for blink detection."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * h_dist) if h_dist > 0 else 0.3


def calculate_yaw(landmarks):
    """Calculate head yaw angle (looking left/right)."""
    nose = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    
    # Nose position relative to ear midpoint
    ear_midpoint = (left_ear.x + right_ear.x) / 2
    ear_width = abs(right_ear.x - left_ear.x)
    
    if ear_width > 0.001:
        # Deviation from center (0 = looking straight)
        deviation = (nose.x - ear_midpoint) / ear_width
        yaw = deviation * 90  # Scale to degrees
        return abs(yaw)
    return 0.0


# ── CSV Setup (Auto-Increment Session Numbering) ────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_raw")
os.makedirs(DATASET_DIR, exist_ok=True)

# Find next available session number
existing_nums = []
if os.path.isdir(DATASET_DIR):
    for fname in os.listdir(DATASET_DIR):
        match = re.match(r"person_(\d+)\.csv$", fname)
        if match:
            existing_nums.append(int(match.group(1)))

next_num = max(existing_nums, default=0) + 1
csv_filename = os.path.join(DATASET_DIR, f"person_{next_num}.csv")

print(f"Saving data to: dataset_raw/person_{next_num}.csv")

csv_file = open(csv_filename, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["yaw", "ear", "volume", "label"])

print("=" * 60)
print("        SDP-1: FACE DATA COLLECTOR")
print("=" * 60)
print("\nControls:")
print("  Hold '0' → Record LEGITIMATE (looking at screen)")
print("  Hold '1' → Record SUSPICIOUS (looking away/phone)")
print("  Press 'q' → Save and quit")
print("\nTip: Record ~200+ samples of each class for best results")
print("=" * 60)

# ── Camera Loop ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

legit_count = 0
sus_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera error!")
        break
    
    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process face
    results = face_mesh.process(rgb_frame)
    
    yaw = 0.0
    ear = 0.3  # Default safe value
    status = "No Face Detected"
    color = (0, 0, 255)  # Red
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate metrics
        yaw = calculate_yaw(landmarks)
        ear_left = calculate_ear(landmarks, LEFT_EYE, w, h)
        ear_right = calculate_ear(landmarks, RIGHT_EYE, w, h)
        ear = (ear_left + ear_right) / 2
        
        status = f"Yaw: {yaw:.1f}° | EAR: {ear:.3f}"
        color = (0, 255, 0)  # Green
        
        # Draw key landmarks
        for idx in [1, 234, 454]:  # Nose, left ear, right ear
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 5, (255, 0, 255), -1)
    
    # ── UI Overlay ───────────────────────────────────────────────────────
    # Background panel
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    
    # Status
    cv2.putText(frame, status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Counts
    cv2.putText(frame, f"Legitimate: {legit_count} | Suspicious: {sus_count}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Hold [0]=REAL  [1]=FAKE  [Q]=Quit", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.imshow('SDP-1 Data Collector', frame)
    
    # ── Key Handling ─────────────────────────────────────────────────────
    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('0') and results.multi_face_landmarks:
        # Record legitimate sample
        writer.writerow([yaw, ear, 0.15, 0])
        legit_count += 1
        print(f"[LEGIT #{legit_count}] Yaw={yaw:.2f}, EAR={ear:.3f}")
    elif key == ord('1') and results.multi_face_landmarks:
        # Record suspicious sample
        writer.writerow([yaw, ear, 0.15, 1])
        sus_count += 1
        print(f"[SUSPI #{sus_count}] Yaw={yaw:.2f}, EAR={ear:.3f}")

# ── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
csv_file.close()

print("\n" + "=" * 60)
print(f"Data saved to: dataset_raw/person_{next_num}.csv")
print(f"Total samples: {legit_count + sus_count}")
print(f"  Legitimate: {legit_count}")
print(f"  Suspicious: {sus_count}")
print("=" * 60)
print(f"\nNext step: python train_model.py --csv dataset_raw/person_{next_num}.csv")

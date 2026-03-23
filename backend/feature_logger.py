"""
SDP-1: Feature Logger
=====================
Non-blocking feature aggregation and logging for dataset collection.
Writes 1-second windowed samples to CSV for training/evaluation.

Usage:
    from feature_logger import FeatureLogger
    
    logger = FeatureLogger()
    logger.log_features(features_dict, label)  # Call every frame
    logger.close()  # On shutdown
"""

import os
import csv
import time
import threading
from datetime import datetime
from collections import deque


class FeatureLogger:
    """
    Thread-safe, non-blocking feature logger.
    Aggregates features over 1-second windows and writes to CSV.
    """
    
    # CSV columns matching global_state fields
    COLUMNS = [
        "timestamp",
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
        "label"
    ]
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the feature logger.
        
        Args:
            output_dir: Directory for CSV files. Defaults to backend/dataset_logs/
        """
        if output_dir is None:
            # Default to dataset_logs in same directory as this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(module_dir, "dataset_logs")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create session file with timestamp
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"session_{session_ts}.csv")
        
        # Non-blocking buffer for 1-second aggregation
        self._buffer = deque(maxlen=100)  # ~3 seconds at 30fps
        self._lock = threading.Lock()
        self._last_write_time = time.time()
        self._window_duration = 1.0  # 1-second windows
        
        # Track session stats
        self._sample_count = 0
        self._is_open = True
        
        # Write CSV header
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._file.flush()
        
        print(f"[FeatureLogger] Logging to: {self.csv_path}")
    
    def log_features(self, features: dict, label: int) -> None:
        """
        Add features to the buffer. Called every frame (non-blocking).
        
        Args:
            features: Dict with keys matching COLUMNS (except timestamp/label)
            label: 0 = legitimate, 1 = suspicious
        """
        if not self._is_open:
            return
        
        # Add timestamp and label
        sample = {
            "timestamp": time.time(),
            "label": label,
            **features
        }
        
        with self._lock:
            self._buffer.append(sample)
        
        # Check if 1-second window has passed
        self._maybe_write_window()
    
    def _maybe_write_window(self) -> None:
        """Write aggregated window if 1 second has passed."""
        now = time.time()
        if now - self._last_write_time < self._window_duration:
            return
        
        self._last_write_time = now
        
        with self._lock:
            if len(self._buffer) == 0:
                return
            
            # Aggregate buffer (use most recent values, they're most accurate)
            samples = list(self._buffer)
            self._buffer.clear()
        
        # Compute window aggregates (mean for continuous, mode for discrete)
        aggregated = self._aggregate_window(samples)
        
        # Write to CSV
        self._writer.writerow(aggregated)
        self._file.flush()
        self._sample_count += 1
    
    def _aggregate_window(self, samples: list) -> dict:
        """
        Aggregate a window of samples into a single row.
        Uses mean for numeric fields, last value for label.
        """
        if not samples:
            return {}
        
        # Use the last sample's timestamp and label
        result = {
            "timestamp": samples[-1]["timestamp"],
            "label": samples[-1]["label"],
        }
        
        # Average the numeric columns
        numeric_cols = [
            "gaze_score", "face_score", "voice_score",
            "micro_var", "jitter", "shimmer", "spectral_centroid",
            "anomaly_duration", "final_integrity_score"
        ]
        
        for col in numeric_cols:
            values = [s.get(col, 0.0) for s in samples if col in s]
            if values:
                result[col] = round(sum(values) / len(values), 4)
            else:
                result[col] = 0.0
        
        # Object flag: 1 if ANY sample had suspicious object
        result["object_flag"] = int(any(s.get("object_flag", 0) for s in samples))
        
        return result
    
    def write_window(self) -> None:
        """Force write current buffer as a window. Called externally."""
        now = time.time()
        self._last_write_time = now - self._window_duration  # Force write
        self._maybe_write_window()
    
    def get_sample_count(self) -> int:
        """Return number of samples written."""
        return self._sample_count
    
    def close(self) -> None:
        """Flush remaining buffer and close file."""
        if not self._is_open:
            return
        
        self._is_open = False
        
        # Write any remaining samples
        with self._lock:
            if len(self._buffer) > 0:
                samples = list(self._buffer)
                self._buffer.clear()
                aggregated = self._aggregate_window(samples)
                if aggregated:
                    self._writer.writerow(aggregated)
        
        self._file.close()
        print(f"[FeatureLogger] Closed. Total samples: {self._sample_count}")
    
    def __del__(self):
        """Ensure file is closed on garbage collection."""
        try:
            self.close()
        except:
            pass


# Convenience function for extracting features from global_state
def extract_features_from_state(global_state, object_detected: bool = False) -> dict:
    """
    Extract loggable features from IntegrityState object.
    
    Args:
        global_state: The IntegrityState instance
        object_detected: Whether a banned object was detected this frame
    
    Returns:
        dict: Features ready for logging
    """
    with global_state.lock:
        return {
            "gaze_score": global_state.gaze_score,
            "face_score": global_state.face_score,
            "voice_score": global_state.voice_score,
            "micro_var": getattr(global_state, "micro_var", 0.0),
            "jitter": global_state.jitter,
            "shimmer": global_state.shimmer,
            "spectral_centroid": global_state.spectral_centroid,
            "anomaly_duration": global_state.anomaly_duration,
            "object_flag": int(object_detected),
            "final_integrity_score": global_state.integrity_score,
        }

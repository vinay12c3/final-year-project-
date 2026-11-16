"""Image pre-processing and hand landmark detection with MediaPipe HandLandmarker."""
from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")          # hide MediaPipe info logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from .config import HAND_MODEL  # noqa: E402
from .features import NUM_POINTS, empty_frame  # noqa: E402

# Pairs of landmark indices that form the hand skeleton (for drawing).
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def enhance(bgr: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Light pre-processing: limit size, and brighten dark / low-contrast frames.

    CLAHE (contrast-limited adaptive histogram equalisation) on the lightness
    channel helps MediaPipe in dim classrooms and hospital corridors.
    """
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[..., 0]
    if l_chan.mean() < 70:                      # only clearly under-exposed frames
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


class HandTracker:
    """Wraps MediaPipe HandLandmarker.

    mode="video"  - for webcam streams: uses tracking between frames (faster, steadier)
    mode="image"  - for independent photos
    """

    def __init__(self, mode: str = "video", num_hands: int = 2,
                 min_detection: float = 0.5, min_tracking: float = 0.5):
        if not HAND_MODEL.exists():
            raise FileNotFoundError(f"MediaPipe hand model not found at {HAND_MODEL}")
        self.mode = mode
        running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=running,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, bgr: np.ndarray, timestamp_ms: int | None = None):
        """Detect hands in a BGR frame.

        Returns (raw, hands):
          raw   - (2, 21, 3) float32 landmarks in project format (see features.py)
          hands - list of {"side", "score", "points": [[x, y], ...]} in 0..1 image
                  coordinates, used by the frontend to draw the skeleton
        """
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.mode == "video":
            ts = int(timestamp_ms) if timestamp_ms is not None else self._last_ts + 33
            ts = max(ts, self._last_ts + 1)          # MediaPipe needs increasing timestamps
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        return self._to_raw(result, w / h)

    @staticmethod
    def _to_raw(result, aspect: float):
        raw = empty_frame()
        hands = []
        detected = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            side = handed[0].category_name if handed else "Left"
            score = float(handed[0].score) if handed else 0.0
            detected.append((side, score, pts))

        # Slot 0 = MediaPipe "Left", slot 1 = "Right". If both hands get the same
        # label (it happens with crossed hands), fall back to image x-order.
        sides = [d[0] for d in detected]
        if len(detected) == 2 and sides[0] == sides[1]:
            detected.sort(key=lambda d: d[2][:, 0].mean())
            slots = [0, 1]
        else:
            slots = [0 if d[0] == "Left" else 1 for d in detected]

        for slot, (side, score, pts) in zip(slots, detected):
            centred = pts.copy()
            centred[:, 0] = (pts[:, 0] - 0.5) * aspect
            centred[:, 1] = pts[:, 1] - 0.5
            centred[:, 2] = pts[:, 2] * aspect
            raw[slot] = centred[:NUM_POINTS]
            hands.append({
                "side": side,
                "score": round(score, 3),
                "points": [[round(float(x), 4), round(float(y), 4)] for x, y in pts[:, :2]],
            })
        return raw, hands


def detect_image(tracker: HandTracker, bgr: np.ndarray):
    """Single photo: try the frame as-is, then fallbacks if no hand was found.

    Close-up crops (hand touching the image border) are detected far more
    reliably after adding a border and upscaling. Landmarks from a padded
    image are mapped back to the original image coordinates.
    """
    raw, hands = tracker.process(bgr)
    if hands:
        return raw, hands
    h, w = bgr.shape[:2]
    p = int(max(h, w) * 0.25)
    padded = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded = cv2.resize(padded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    raw_p, hands_p = tracker.process(padded)
    if hands_p:
        pw, ph = w + 2 * p, h + 2 * p
        for hand in hands_p:
            hand["points"] = [[round((x * pw - p) / w, 4), round((y * ph - p) / h, 4)] for x, y in hand["points"]]
        return raw_p, hands_p          # features are translation/scale invariant, raw is fine as-is
    return tracker.process(enhance(bgr))

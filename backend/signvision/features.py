"""Turn raw hand landmarks into model features, plus training-time augmentation.

Raw landmark format used everywhere in the project
--------------------------------------------------
One frame  -> float32 array of shape (2, 21, 3)
              slot 0 / slot 1 = the two hands, 21 MediaPipe points, (x, y, z).
              x and y are centred on the image and corrected for aspect ratio:
                  x = (x_norm - 0.5) * (width / height),  y = y_norm - 0.5
              A missing hand is filled with NaN.
A clip     -> (T, 2, 21, 3)

Storing raw points (not features) lets us change feature engineering or
augmentation later without re-running MediaPipe on the datasets.
"""
from __future__ import annotations

import numpy as np

NUM_POINTS = 21
HAND_DIM = NUM_POINTS * 3                       # 63 values per hand
STATIC_DIM = 2 * HAND_DIM + 2 + 3               # both hands + presence flags + hand-to-hand vector = 131
DYNAMIC_DIM = STATIC_DIM + 4                    # + wrist positions in the frame (x, y per hand) = 135
SENTENCE_DIM = STATIC_DIM + 8                   # + signer-relative wrist positions and their velocities = 139


def empty_frame() -> np.ndarray:
    return np.full((2, NUM_POINTS, 3), np.nan, dtype=np.float32)


# ------------------------------------------------------------------ features
def static_features(raw: np.ndarray) -> np.ndarray:
    """Hand-shape features that ignore where the hands are in the frame.

    Each hand is translated so the wrist is the origin and scaled so the
    farthest point is at distance 1, which makes the features independent
    of the signer's distance from the camera and position in the image.
    Accepts (2, 21, 3) or (N, 2, 21, 3); returns (131,) or (N, 131).
    """
    raw = np.asarray(raw, dtype=np.float32)
    single = raw.ndim == 3
    if single:
        raw = raw[None]
    n = raw.shape[0]

    present = ~np.isnan(raw[:, :, 0, 0])                         # (N, 2)
    pts = np.nan_to_num(raw, nan=0.0)
    rel = pts - pts[:, :, :1, :]                                  # wrist -> origin
    scale = np.linalg.norm(rel[..., :2], axis=-1).max(axis=-1)    # (N, 2)
    scale = np.where(scale > 1e-6, scale, 1.0)
    rel = rel / scale[..., None, None] * present[..., None, None]

    both = present.all(axis=1)
    pair_scale = scale.mean(axis=1)
    inter = (pts[:, 1, 0, :] - pts[:, 0, 0, :]) / pair_scale[:, None] * both[:, None]

    feats = np.concatenate(
        [rel.reshape(n, 2 * HAND_DIM), present.astype(np.float32), inter], axis=1
    ).astype(np.float32)
    return feats[0] if single else feats


def dynamic_features(raw: np.ndarray) -> np.ndarray:
    """Per-frame features for moving signs: hand shape + where the wrists are.

    Accepts (..., 2, 21, 3) and returns (..., 135).
    """
    raw = np.asarray(raw, dtype=np.float32)
    lead = raw.shape[:-3]
    flat = raw.reshape(-1, 2, NUM_POINTS, 3)
    stat = static_features(flat)
    present = ~np.isnan(flat[:, :, 0, 0])
    wrists = np.nan_to_num(flat[:, :, 0, :2], nan=0.0) * present[..., None]
    feats = np.concatenate([stat, wrists.reshape(-1, 4)], axis=1)
    return feats.reshape(*lead, DYNAMIC_DIM)


def sentence_features(clips: np.ndarray) -> np.ndarray:
    """Per-frame features for whole sentences, independent of where the signer sits.

    Different people sit at different places and distances from the camera, so
    wrist positions are taken relative to the clip's own average hand position
    and divided by the signer's typical hand size. Wrist velocities describe
    the movement itself. Accepts (T, 2, 21, 3) or (N, T, 2, 21, 3).
    """
    clips = np.asarray(clips, dtype=np.float32)
    single = clips.ndim == 4
    if single:
        clips = clips[None]
    n, t = clips.shape[:2]
    stat = static_features(clips.reshape(-1, 2, NUM_POINTS, 3)).reshape(n, t, STATIC_DIM)
    present = ~np.isnan(clips[:, :, :, 0, 0])                           # (N, T, 2)
    wrist = clips[:, :, :, 0, :2]                                        # (N, T, 2, 2)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(wrist.reshape(n, -1, 2), axis=1)             # (N, 2)
        size = np.nanmedian(np.linalg.norm(clips[..., 9, :2] - clips[..., 0, :2], axis=-1).reshape(n, -1), axis=1)
    centre = np.nan_to_num(centre, nan=0.0)
    size = np.where(np.isfinite(size) & (size > 1e-4), size, 0.1)        # wrist -> middle-finger base
    pos = (wrist - centre[:, None, None, :]) / size[:, None, None, None] * 0.1
    pos = np.nan_to_num(pos, nan=0.0) * present[..., None]
    vel = np.zeros_like(pos)
    vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * (present[:, 1:] & present[:, :-1])[..., None] * 5.0
    feats = np.concatenate([stat, pos.reshape(n, t, 4), vel.reshape(n, t, 4)], axis=-1).astype(np.float32)
    return feats[0] if single else feats


def make_noise_clips(raw: np.ndarray, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Movements that are NOT a sign, for a "not a sign" training class.

    - fragments: a short random piece of a real sign (what the camera sees when
      someone starts, pauses or stops half-way)
    - fidgets: a hand pose that drifts and wobbles (scratching, adjusting
      glasses, resting hands, waving to get attention)
    """
    out = np.empty((n, length) + raw.shape[2:], np.float32)
    for i in range(n):
        src = raw[rng.integers(len(raw))]
        if rng.random() < 0.6:
            t = len(src)
            k = int(t * rng.uniform(0.12, 0.35))
            a = rng.integers(0, max(1, t - k))
            out[i] = resample(src[a:a + k], length)
        else:
            pose = src[rng.integers(len(src))].copy()
            drift = rng.normal(0, 0.06, size=(2, 2))
            ts = np.linspace(0, 1, length)[:, None, None, None]
            clip = np.repeat(pose[None], length, axis=0)
            wobble = 0.015 * np.sin(ts * rng.uniform(3, 15) + rng.uniform(0, 6)) * rng.normal(size=(1, 2, 1, 3))
            clip[..., :2] += ts * drift[None, :, None, :] * rng.random() + wobble[..., :2]
            out[i] = clip
    return out


# ------------------------------------------------------------------ sequences
def resample(seq: np.ndarray, n: int) -> np.ndarray:
    """Pick n frames evenly spread over a clip of any length."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return np.stack([empty_frame()] * n)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return seq[idx]


def random_time_warp(clips: np.ndarray, n: int, rng: np.random.Generator,
                     min_keep: float = 0.8) -> np.ndarray:
    """Crop a random 80-100 % window of each clip and resample it to n frames.

    Simulates people signing faster/slower and starting at different moments.
    clips: (N, T, 2, 21, 3) -> (N, n, 2, 21, 3)
    """
    total = clips.shape[1]
    out = np.empty((clips.shape[0], n) + clips.shape[2:], dtype=np.float32)
    for i in range(clips.shape[0]):
        keep = int(round(total * rng.uniform(min_keep, 1.0)))
        start = rng.integers(0, total - keep + 1)
        out[i] = resample(clips[i, start:start + keep], n)
    return out


# ------------------------------------------------------------------ augmentation
def augment(raw: np.ndarray, rng: np.random.Generator, mirror_p: float = 0.5,
            max_rot_deg: float = 15.0, scale_range=(0.9, 1.1), shift: float = 0.05,
            noise: float = 0.004) -> np.ndarray:
    """Random geometric changes applied to a batch of samples.

    raw: (N, 2, 21, 3) for single frames or (N, T, 2, 21, 3) for clips.
    The same transform is applied to every frame of a clip.
      * mirror     - swaps left/right hands, so left-handed signers are covered
      * rotation   - small in-plane camera tilt
      * scale/shift- distance to the camera and position in the frame
      * noise      - landmark jitter
    """
    out = np.array(raw, dtype=np.float32, copy=True)
    n = out.shape[0]
    flat = out.reshape(n, -1, 3)                        # view over every point of a sample

    mirror = rng.random(n) < mirror_p
    if mirror.any():
        out[mirror] = np.flip(out[mirror], axis=-3)     # swap hand slots
        flat = out.reshape(n, -1, 3)
        flat[mirror, :, 0] *= -1.0

    with np.errstate(all="ignore"):
        centre = np.nanmean(flat[..., :2], axis=1)                  # (N, 2)
    centre = np.nan_to_num(centre, nan=0.0)

    theta = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg, n))
    cos, sin = np.cos(theta), np.sin(theta)
    s = rng.uniform(*scale_range, n)
    xy = flat[..., :2] - centre[:, None, :]
    x = (xy[..., 0] * cos[:, None] - xy[..., 1] * sin[:, None]) * s[:, None]
    y = (xy[..., 0] * sin[:, None] + xy[..., 1] * cos[:, None]) * s[:, None]
    offset = rng.uniform(-shift, shift, (n, 2))
    flat[..., 0] = x + centre[:, 0:1] + offset[:, 0:1]
    flat[..., 1] = y + centre[:, 1:2] + offset[:, 1:2]
    flat[..., 2] *= s[:, None]
    flat += rng.normal(0.0, noise, flat.shape).astype(np.float32)   # NaN stays NaN
    return flat.reshape(out.shape)

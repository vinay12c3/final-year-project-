"""Loading, saving and splitting landmark datasets.

On-disk layout (all landmark arrays use the raw format described in features.py)

    data/own/static/<LABEL>/<capture_id>.npy      (N, 2, 21, 3)  frames of one held sign
    data/own/dynamic/<label>/<capture_id>.npy     (T, 2, 21, 3)  one recorded clip
    data/processed/static/<source>.npz            raw (N,2,21,3), labels (N,), groups (N,)
    data/processed/dynamic/<source>.npz           raw (N,60,2,21,3), labels (N,), groups (N,)

"groups" identify samples that come from the same recording / video so that a
recording never ends up in both the training and the test set.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import json

from .config import OWN_DIR, PROCESSED_DIR, store_len
from .features import resample

_SAFE = re.compile(r"[^A-Za-z0-9_]")


def clean_label(label: str, kind: str) -> str:
    """Normalise a label: letters/numbers upper-case, words lower_snake_case."""
    label = label.strip()
    if kind == "sentence":
        from .sentences import label_for
        return label if label.startswith("s_") and _SAFE.sub("", label) == label else label_for(label)
    background = label.startswith("_")
    if kind == "dynamic":
        label = re.sub(r"^\d+\s*[.)]\s*", "", label)        # INCLUDE style "48. Hello" -> "Hello"
    label = re.sub(r"_+", "_", _SAFE.sub("_", label.replace(" ", "_"))).strip("_")
    if background:
        label = "_" + label
    return label.upper() if kind == "static" else label.lower()


@dataclass
class Dataset:
    raw: np.ndarray          # static: (N, 2, 21, 3)  dynamic: (N, STORE_LEN, 2, 21, 3)
    labels: np.ndarray       # (N,) str
    groups: np.ndarray       # (N,) str
    sources: np.ndarray      # (N,) str  e.g. "own" or the public dataset name

    def __len__(self):
        return len(self.labels)

    def subset(self, idx):
        return Dataset(self.raw[idx], self.labels[idx], self.groups[idx], self.sources[idx])


# ------------------------------------------------------------------ recordings
def save_recording(kind: str, label: str, frames: np.ndarray, who: str = "") -> Path:
    if kind == "sentence" and not label.startswith("s_"):
        remember_sentence_text(label)
    label = clean_label(label, kind)
    folder = OWN_DIR / kind / label
    folder.mkdir(parents=True, exist_ok=True)
    who = _SAFE.sub("", who)[:20]
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    path = folder / f"{name}{'-' + who if who else ''}.npy"
    np.save(path, np.asarray(frames, dtype=np.float32))
    return path


def recording_counts(kind: str) -> dict[str, int]:
    root = OWN_DIR / kind
    return {d.name: len(list(d.glob("*.npy"))) for d in sorted(root.iterdir()) if d.is_dir()} \
        if root.exists() else {}


def delete_last_recording(kind: str, label: str) -> bool:
    folder = OWN_DIR / kind / clean_label(label, kind)
    files = sorted(folder.glob("*.npy")) if folder.exists() else []
    if not files:
        return False
    files[-1].unlink()
    return True


def processed_counts(kind: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in sorted((PROCESSED_DIR / kind).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            for lab in z["labels"]:
                counts[str(lab)] = counts.get(str(lab), 0) + 1
    return counts


# ------------------------------------------------------------------ sentence texts
OWN_TEXTS = OWN_DIR / "sentence" / "texts.json"


def remember_sentence_text(text: str) -> str:
    """Stores the readable text of a sentence recorded by the user; returns its label."""
    from .sentences import label_for, text_for
    label = label_for(text)
    texts = json.loads(OWN_TEXTS.read_text()) if OWN_TEXTS.exists() else {}
    texts.setdefault(label, text_for(text))
    OWN_TEXTS.parent.mkdir(parents=True, exist_ok=True)
    OWN_TEXTS.write_text(json.dumps(texts, indent=1))
    return label


def sentence_texts() -> dict[str, str]:
    """label -> readable sentence, from the public corpus files and the user's own recordings."""
    texts: dict[str, str] = {}
    for f in sorted((PROCESSED_DIR / "sentence").glob("*.json")):
        texts.update(json.loads(f.read_text()))
    if OWN_TEXTS.exists():
        texts.update(json.loads(OWN_TEXTS.read_text()))
    return texts


# ------------------------------------------------------------------ loading
def load_dataset(kind: str, use_own: bool = True, use_public: bool = True) -> Dataset:
    raws, labels, groups, sources = [], [], [], []

    if use_public:
        for f in sorted((PROCESSED_DIR / kind).glob("*.npz")):
            with np.load(f, allow_pickle=False) as z:
                raws.append(z["raw"].astype(np.float32))
                labels += [str(x) for x in z["labels"]]
                groups += [f"{f.stem}/{g}" for g in z["groups"]]
                sources += [f.stem] * len(z["labels"])

    if use_own:
        root = OWN_DIR / kind
        for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(label_dir.glob("*.npy")):
                arr = np.load(f).astype(np.float32)
                if kind == "static":
                    arr = arr[~np.isnan(arr[:, :, 0, 0]).all(axis=1)]      # keep frames with a hand
                    if len(arr) == 0:
                        continue
                    raws.append(arr)
                    n = len(arr)
                else:
                    raws.append(resample(arr, store_len(kind))[None])
                    n = 1
                labels += [label_dir.name] * n
                groups += [f"own/{label_dir.name}/{f.stem}"] * n
                sources += ["own"] * n

    if not raws:
        shape = (0, 2, 21, 3) if kind == "static" else (0, store_len(kind), 2, 21, 3)
        return Dataset(np.zeros(shape, np.float32), np.array([], str), np.array([], str), np.array([], str))
    return Dataset(np.concatenate(raws), np.array(labels), np.array(groups), np.array(sources))


# ------------------------------------------------------------------ splitting
def group_split(ds: Dataset, test_frac: float = 0.2, seed: int = 42):
    """Stratified split by recording group: per label, ~test_frac of its groups go to test.

    Labels with a single group are split by sample instead (and a warning is
    returned) because otherwise they could not be evaluated at all.
    """
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(ds), dtype=bool)
    warnings = []
    for lab in np.unique(ds.labels):
        idx = np.where(ds.labels == lab)[0]
        uniq = np.unique(ds.groups[idx])
        if len(uniq) >= 2:
            k = max(1, int(round(len(uniq) * test_frac)))
            chosen = set(rng.choice(uniq, size=k, replace=False))
            test_mask[idx[np.isin(ds.groups[idx], list(chosen))]] = True
        elif len(idx) >= 2:
            warnings.append(lab)
            k = max(1, int(round(len(idx) * test_frac)))
            test_mask[rng.choice(idx, size=k, replace=False)] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0], warnings

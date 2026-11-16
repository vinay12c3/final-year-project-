"""Convert a folder of sign images into landmarks for the letters model.

Expected layout (any image dataset sorted into one folder per sign):

    <src>/
        A/  img001.jpg  img002.jpg ...
        B/  ...
        1/  ...

Example (Kaggle "Indian Sign Language (ISL)" dataset, unzipped to D:\\datasets\\isl):
    python -m scripts.prepare_images --src "D:\\datasets\\isl\\Indian" --name isl_kaggle

Writes data/processed/static/<name>.npz. Images where no hand is found are skipped.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signvision.config import PROCESSED_DIR
from signvision.data import clean_label
from signvision.trainer import log

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_tracker = None


def _init():
    global _tracker
    from signvision.hands import HandTracker
    _tracker = HandTracker(mode="image")


def _work(path: str):
    from signvision.hands import detect_image
    img = cv2.imread(path)
    if img is None:
        return None
    raw, hands = detect_image(_tracker, img)
    return raw if hands else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with one sub-folder per sign")
    ap.add_argument("--name", required=True, help="short dataset name, e.g. isl_kaggle")
    ap.add_argument("--per-class", type=int, default=600, help="max images per sign (0 = all)")
    ap.add_argument("--block", type=int, default=100,
                    help="consecutive images per split group; public sets are often frames cut from "
                         "videos, so neighbours are near-duplicates and must stay on the same side "
                         "of the train/test split (1 = every image independent)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}")
    rng = np.random.default_rng(0)
    jobs = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        files = sorted(str(f) for f in d.rglob("*") if f.suffix.lower() in EXTS)
        if args.per_class and len(files) > args.per_class:
            files = sorted(rng.choice(files, args.per_class, replace=False).tolist())
        lab = clean_label(d.name, "static")
        jobs += [(lab, f, f"{lab}-{k // max(1, args.block)}") for k, f in enumerate(files)]
    if not jobs:
        raise SystemExit("No images found. Each sign must be its own sub-folder of --src.")
    log(f"{len(jobs)} images in {len({j[0] for j in jobs})} classes, {args.workers} workers")

    raws, labels, groups, missed = [], [], [], 0
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for i, ((lab, f, grp), raw) in enumerate(zip(jobs, pool.map(_work, [j[1] for j in jobs], chunksize=16))):
            if raw is None:
                missed += 1
            else:
                raws.append(raw)
                labels.append(lab)
                groups.append(grp)
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(jobs)} done, hands found in {len(raws)}")

    out = PROCESSED_DIR / "static" / f"{args.name}.npz"
    np.savez_compressed(out, raw=np.stack(raws), labels=np.array(labels), groups=np.array(groups))
    log(f"saved {len(raws)} samples to {out}  (no hand found in {missed} images)")


if __name__ == "__main__":
    main()

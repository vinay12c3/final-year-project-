"""Convert a folder of sign videos into landmark clips for the words model.

The word label is taken from the folder that contains each video, so any
dataset sorted into one folder per word works, including INCLUDE (AI4Bharat):

    <src>/Greetings/48. Hello/MVI_0001.MOV  ->  label "hello"
    <src>/thank_you/clip1.mp4                ->  label "thank_you"

Sentence datasets (one folder per sentence, e.g. ISL-CSLTR) use --kind sentence:

    python -m scripts.prepare_videos --kind sentence --src data/raw/isl_csltr/Videos_Sentence_Level --name isl_csltr

Examples:
    python -m scripts.prepare_videos --src "D:\\datasets\\INCLUDE" --name include
    python -m scripts.prepare_videos --src "D:\\datasets\\INCLUDE" --name include --only hello thank_you sorry

Writes data/processed/<kind>/<name>.npz
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signvision.config import PROCESSED_DIR, store_len
from signvision.data import clean_label
from signvision.trainer import log

EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _work(job: tuple[str, int, int]):
    path, length, max_side = job
    import cv2

    from signvision.features import resample
    from signvision.hands import HandTracker, enhance

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / 15)))
    frames, i = [], 0
    with HandTracker(mode="video") as tracker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                raw, _ = tracker.process(enhance(bgr, max_side), int(i / fps * 1000))
                frames.append(raw)
            i += 1
    cap.release()
    if len(frames) < 5:
        return None
    clip = np.stack(frames)
    has = ~np.isnan(clip[:, :, 0, 0]).all(axis=1)
    if has.mean() < 0.3:
        return None
    first, last = np.argmax(has), len(has) - np.argmax(has[::-1])      # trim hand-less ends
    return resample(clip[first:last], length)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", choices=["dynamic", "sentence"], default="dynamic")
    ap.add_argument("--only", nargs="*", help="keep only these words (labels after cleaning)")
    ap.add_argument("--max-side", type=int, default=640,
                    help="resize frames to this size before hand detection; use 1280 for full-body HD videos")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}")
    only = {clean_label(x, args.kind) for x in args.only} if args.only else None
    jobs, texts = [], {}
    for f in sorted(src.rglob("*")):
        if f.suffix.lower() in EXTS:
            lab = clean_label(f.parent.name, args.kind)
            if only is None or lab in only:
                jobs.append((lab, str(f)))
                if args.kind == "sentence":
                    from signvision.sentences import text_for
                    texts[lab] = text_for(f.parent.name)
    if not jobs:
        raise SystemExit("No matching videos found.")
    log(f"{len(jobs)} videos in {len({l for l, _ in jobs})} classes, {args.workers} workers")
    length = store_len(args.kind)

    raws, labels, groups, missed = [], [], [], 0
    with ProcessPoolExecutor(args.workers) as pool:
        for i, ((lab, f), clip) in enumerate(zip(jobs, pool.map(_work, [(f, length, args.max_side) for _, f in jobs]))):
            if clip is None:
                missed += 1
            else:
                raws.append(clip)
                labels.append(lab)
                groups.append(f"{Path(f).parent.name}/{Path(f).stem}")
            if (i + 1) % 50 == 0:
                log(f"  {i + 1}/{len(jobs)} videos done")

    out = PROCESSED_DIR / args.kind / f"{args.name}.npz"
    if texts:
        out.with_suffix(".json").write_text(json.dumps({k: texts[k] for k in sorted(set(labels))}, indent=1))
    np.savez_compressed(out, raw=np.stack(raws), labels=np.array(labels), groups=np.array(groups))
    log(f"saved {len(raws)} clips to {out}  (skipped {missed} without enough hand frames)")


if __name__ == "__main__":
    main()

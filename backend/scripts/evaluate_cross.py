"""Cross-dataset test: train on the public dataset only, test on your own recordings.

This answers the question reviewers always ask: "does it work on new people,
cameras and rooms?". Only signs present in both sets are used. Results go to
reports/<kind>/cross_dataset.json and cross_confusion_matrix.png. The deployed
model is NOT changed.

    python -m scripts.evaluate_cross --kind static
    python -m scripts.evaluate_cross --kind dynamic
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from signvision import config as C
from signvision.data import group_split, load_dataset
from signvision.features import DYNAMIC_DIM, STATIC_DIM, augment, dynamic_features, random_time_warp, static_features
from signvision.models import SequenceClassifier, StaticMLP
from signvision.trainer import fit, log, predict, scores, write_reports


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["static", "dynamic"], default="static")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    kind = args.kind

    public = load_dataset(kind, use_own=False, use_public=True)
    own = load_dataset(kind, use_own=True, use_public=False)
    shared = sorted(set(public.labels) & set(own.labels))
    if len(shared) < 2:
        raise SystemExit("Need at least 2 signs that exist in both the public data and your recordings.")
    public = public.subset(np.isin(public.labels, shared))
    own = own.subset(np.isin(own.labels, shared))
    log(f"{len(shared)} shared signs; train on {len(public)} public samples, test on {len(own)} own samples")

    lab2id = {l: i for i, l in enumerate(shared)}
    enc = lambda d: np.array([lab2id[l] for l in d.labels], dtype=np.int64)  # noqa: E731
    tr_idx, va_idx, _ = group_split(public, 0.1)
    train, val = public.subset(tr_idx), public.subset(va_idx)
    y_tr = enc(train)

    if kind == "static":
        feat = static_features
        model = StaticMLP(STATIC_DIM, len(shared))
        make = lambda idx, rng: (static_features(augment(train.raw[idx], rng)), y_tr[idx])  # noqa: E731
        batch = 512
    else:
        from scripts.train_dynamic import plain_clips
        feat = lambda raw: dynamic_features(plain_clips(raw))  # noqa: E731
        model = SequenceClassifier(DYNAMIC_DIM, len(shared))
        make = lambda idx, rng: (dynamic_features(random_time_warp(augment(train.raw[idx], rng), C.SEQ_LEN, rng)),  # noqa: E731
                                 y_tr[idx])
        batch = 32

    hist, _ = fit(model, make, len(train), feat(val.raw), enc(val), epochs=args.epochs, batch_size=batch)
    y_te = enc(own)
    y_pred = predict(model, feat(own.raw)).argmax(1)
    res = scores(y_te, y_pred)
    log(f"CROSS-DATASET accuracy {res['accuracy']:.4f}  macro-F1 {res['f1']:.4f}")

    out = C.REPORTS_DIR / kind / "cross"
    write_reports(out, "Cross-dataset (public → own)", shared, y_te, y_pred, hist,
                  {"Trained on public only": res}, {}, {"n_train": len(train), "n_test": len(own)})
    (C.REPORTS_DIR / kind / "cross_dataset.json").write_text(json.dumps(
        {"shared_labels": shared, "n_train_public": len(train), "n_test_own": len(own), **res}, indent=2))
    log(f"saved to {out}")


if __name__ == "__main__":
    main()

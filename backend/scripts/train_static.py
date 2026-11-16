"""Train the letters/numbers model (single-frame hand shape -> sign).

Run from the backend folder:
    python -m scripts.train_static
    python -m scripts.train_static --epochs 80 --no-public      # only your own recordings
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from signvision import config as C
from signvision.data import group_split, load_dataset
from signvision.features import STATIC_DIM, augment, static_features
from signvision.models import StaticMLP, save_checkpoint
from signvision.trainer import (counts_by_source, fit, latency_ms, log, predict, run_baselines,
                                scores, write_reports)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-own", action="store_true", help="ignore recordings made in the app")
    ap.add_argument("--no-public", action="store_true", help="ignore prepared public datasets")
    ap.add_argument("--no-baselines", action="store_true", help="skip SVM/RF/k-NN comparison")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("static", use_own=not args.no_own, use_public=not args.no_public)
    labels = sorted(np.unique(ds.labels).tolist())
    log(f"loaded {len(ds)} samples, {len(labels)} signs: {' '.join(labels)}")
    if len(labels) < 2:
        raise SystemExit("Need at least 2 different signs to train. Record more signs on the "
                         "Record page or prepare a public dataset (see README).")

    tr_idx, te_idx, warn = group_split(ds, 0.2, args.seed)
    if warn:
        log(f"note: these signs have a single recording, so their test samples come from it: {warn}")
    train_full, test = ds.subset(tr_idx), ds.subset(te_idx)
    t_idx, v_idx, _ = group_split(train_full, 0.12, args.seed + 1)
    train, val = train_full.subset(t_idx), train_full.subset(v_idx)
    log(f"split: train {len(train)}  validation {len(val)}  test {len(test)}")

    lab2id = {l: i for i, l in enumerate(labels)}
    y_tr = np.array([lab2id[l] for l in train.labels], dtype=np.int64)
    y_va = np.array([lab2id[l] for l in val.labels], dtype=np.int64)
    y_te = np.array([lab2id[l] for l in test.labels], dtype=np.int64)
    x_va, x_te = static_features(val.raw), static_features(test.raw)

    def make_batch(idx, rng):
        return static_features(augment(train.raw[idx], rng)), y_tr[idx]

    model = StaticMLP(STATIC_DIM, len(labels))
    t0 = time.time()
    hist, best_val = fit(model, make_batch, len(train), x_va, y_va, epochs=args.epochs,
                         batch_size=args.batch, lr=args.lr, seed=args.seed)
    train_s = time.time() - t0

    y_pred = predict(model, x_te).argmax(1)
    ours = scores(y_te, y_pred)
    ours["train_seconds"] = round(train_s, 1)
    log(f"TEST  accuracy {ours['accuracy']:.4f}  macro-F1 {ours['f1']:.4f}")

    comparison = {"MLP on landmarks (ours)": ours}
    if not args.no_baselines:
        x_tr_plain = static_features(train_full.raw)
        y_tr_plain = np.array([lab2id[l] for l in train_full.labels], dtype=np.int64)
        comparison.update(run_baselines(x_tr_plain, y_tr_plain, x_te, y_te, seed=args.seed))

    lat = latency_ms(model, x_te[0] if len(x_te) else np.zeros(STATIC_DIM, np.float32))
    meta = {"trained_at": time.strftime("%Y-%m-%d %H:%M"), "test_accuracy": ours["accuracy"],
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "best_val_accuracy": best_val, "model_latency_ms": lat}
    save_checkpoint(C.STATIC_MODEL, model, "static", labels, meta)
    write_reports(C.REPORTS_DIR / "static", "Letters & numbers", labels, y_te, y_pred, hist, comparison,
                  counts_by_source(ds.labels, ds.sources),
                  {"meta": meta, "params": sum(p.numel() for p in model.parameters())})
    log(f"saved model to {C.STATIC_MODEL}")
    log(f"reports in {C.REPORTS_DIR / 'static'}  (model latency {lat:.2f} ms per frame)")


if __name__ == "__main__":
    main()

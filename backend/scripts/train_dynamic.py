"""Train the words model (sequence of landmark frames -> word).

Trains both a GRU and an LSTM version, keeps the one with the better
validation accuracy, and reports both (useful for the paper).

Run from the backend folder:
    python -m scripts.train_dynamic
    python -m scripts.train_dynamic --kind sentence      # the sentences model
    python -m scripts.train_dynamic --arch gru            # train only one
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from signvision import config as C
from signvision.data import group_split, load_dataset
from signvision.features import (DYNAMIC_DIM, SENTENCE_DIM, augment, dynamic_features, random_time_warp, resample,
                                 make_noise_clips, sentence_features)
from signvision.models import SequenceClassifier, save_checkpoint
from signvision.trainer import (counts_by_source, fit, latency_ms, log, predict, run_baselines,
                                scores, write_reports)


NOISE = "_none"          # background class: starts with "_" so the app never types it


def plain_clips(raw: np.ndarray, seq_len: int) -> np.ndarray:
    return np.stack([resample(c, seq_len) for c in raw]) if len(raw) else \
        np.zeros((0, seq_len, 2, 21, 3), np.float32)


def summary_features(feats: np.ndarray) -> np.ndarray:
    """Mean, std and first-last difference over time: fixed-size input for classic ML baselines."""
    return np.concatenate([feats.mean(1), feats.std(1), feats[:, -1] - feats[:, 0]], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["dynamic", "sentence"], default="dynamic")
    ap.add_argument("--arch", choices=["both", "gru", "lstm"], default="both")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=40, help="stop after this many epochs without improvement")
    ap.add_argument("--no-own", action="store_true")
    ap.add_argument("--no-public", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--labels", nargs="*", help="train only these classes (labels as stored, e.g. s_help_me)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-noise", action="store_true", help="sentences: do not add the 'not a sign' class")
    args = ap.parse_args()

    kind, seq = args.kind, C.seq_len(args.kind)
    feat_fn, feat_dim = (sentence_features, SENTENCE_DIM) if kind == "sentence" else (dynamic_features, DYNAMIC_DIM)
    what = "sentences" if kind == "sentence" else "words"
    ds = load_dataset(kind, use_own=not args.no_own, use_public=not args.no_public)
    if args.labels:
        ds = ds.subset(np.where(np.isin(ds.labels, args.labels))[0])
    if kind == "sentence" and not args.no_noise:
        # a "not a sign" class, so random movements are not forced into a sentence
        from signvision.data import Dataset
        rng0 = np.random.default_rng(args.seed + 7)
        n_noise = max(40, int(len(ds) * 0.15))
        noise = make_noise_clips(ds.raw, n_noise, ds.raw.shape[1], rng0)
        ds = Dataset(np.concatenate([ds.raw, noise]), np.concatenate([ds.labels, [NOISE] * n_noise]),
                     np.concatenate([ds.groups, [f"noise/{i}" for i in range(n_noise)]]),
                     np.concatenate([ds.sources, ["generated"] * n_noise]))
    labels = sorted(np.unique(ds.labels).tolist())
    log(f"loaded {len(ds)} clips, {len(labels)} {what}" + ("" if kind == "sentence" else f": {' '.join(labels)}"))
    if len(labels) < 2:
        raise SystemExit("Need at least 2 different words to train. Record more words on the "
                         "Record page or prepare a public dataset (see README).")

    tr_idx, te_idx, warn = group_split(ds, 0.2, args.seed)
    if warn:
        log(f"note: only one clip for {warn}; they cannot be tested fairly - record more")
    train_full, test = ds.subset(tr_idx), ds.subset(te_idx)
    t_idx, v_idx, _ = group_split(train_full, 0.15, args.seed + 1)
    train, val = train_full.subset(t_idx), train_full.subset(v_idx)
    log(f"split: train {len(train)}  validation {len(val)}  test {len(test)}")

    lab2id = {l: i for i, l in enumerate(labels)}
    enc = lambda d: np.array([lab2id[l] for l in d.labels], dtype=np.int64)  # noqa: E731
    y_tr, y_va, y_te = enc(train), enc(val), enc(test)
    x_va = feat_fn(plain_clips(val.raw, seq))
    x_te = feat_fn(plain_clips(test.raw, seq))

    def make_batch(idx, rng):
        clips = random_time_warp(augment(train.raw[idx], rng), seq, rng)
        drop = rng.random(clips.shape[:2]) < 0.05               # simulate missed detections
        clips[drop] = np.nan
        return feat_fn(clips), y_tr[idx]

    cells = ["gru", "lstm"] if args.arch == "both" else [args.arch]
    results = {}
    for cell in cells:
        log(f"---- training {cell.upper()} ----")
        model = SequenceClassifier(feat_dim, len(labels), cell=cell, hidden=args.hidden, dropout=args.dropout)
        t0 = time.time()
        hist, best_val = fit(model, make_batch, len(train), x_va, y_va, epochs=args.epochs,
                             batch_size=args.batch, lr=args.lr, patience=args.patience, seed=args.seed)
        probs_te = predict(model, x_te)
        y_pred = probs_te.argmax(1)
        res = scores(y_te, y_pred)
        top3 = np.argsort(probs_te, axis=1)[:, -3:]
        res["top3_accuracy"] = float(np.mean([y in t for y, t in zip(y_te, top3)])) if len(y_te) else 0.0
        res["train_seconds"] = round(time.time() - t0, 1)
        log(f"{cell.upper()} TEST accuracy {res['accuracy']:.4f}  top-3 {res['top3_accuracy']:.4f}  macro-F1 {res['f1']:.4f}")
        results[cell] = dict(model=model, hist=hist, val=best_val, pred=y_pred, scores=res)

    best_cell = max(results, key=lambda c: (results[c]["val"], results[c]["scores"]["accuracy"]))
    best = results[best_cell]
    log(f"selected {best_cell.upper()} (validation accuracy {best['val']:.4f})")

    comparison = {f"Bi-{c.upper()} + attention" + (" (ours)" if c == best_cell else ""): r["scores"]
                  for c, r in results.items()}
    if not args.no_baselines:
        x_tr_plain = summary_features(feat_fn(plain_clips(train_full.raw, seq)))
        comparison.update(run_baselines(x_tr_plain, enc(train_full), summary_features(x_te), y_te,
                                        seed=args.seed))

    sample = x_te[0] if len(x_te) else np.zeros((seq, feat_dim), np.float32)
    lat = latency_ms(best["model"], sample, runs=100)
    meta = {"trained_at": time.strftime("%Y-%m-%d %H:%M"), "test_accuracy": best["scores"]["accuracy"], "test_top3_accuracy": best["scores"]["top3_accuracy"],
            "cell": best_cell, "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "best_val_accuracy": best["val"], "model_latency_ms": lat, "seq_len": seq}
    if kind == "sentence":
        from signvision.data import sentence_texts
        texts = sentence_texts()
        meta["texts"] = {l: ("(not a sign)" if l == NOISE else texts.get(l, l)) for l in labels}
    path = C.MODEL_PATHS[kind]
    save_checkpoint(path, best["model"], kind, labels, meta)
    write_reports(C.REPORTS_DIR / kind, what.capitalize(), labels, y_te, best["pred"], best["hist"],
                  comparison, counts_by_source(ds.labels, ds.sources),
                  {"meta": meta, "params": sum(p.numel() for p in best["model"].parameters())},
                  names=[meta["texts"][l] for l in labels] if kind == "sentence" else None)
    log(f"saved model to {path}")
    log(f"reports in {C.REPORTS_DIR / kind}  (model latency {lat:.2f} ms per window)")


if __name__ == "__main__":
    main()

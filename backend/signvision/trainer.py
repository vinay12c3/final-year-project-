"""Shared training loop, evaluation and report figures for both models."""
from __future__ import annotations

import copy
import json
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)
from torch import nn  # noqa: E402

# Small models train fastest on a few threads; many threads just fight each other.
torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))

INK, ACCENT, TEAL, MUTED = "#101828", "#1D4ED8", "#1D4ED8", "#98A2B3"   # text, primary, good, neutral
WEAK = "#F79009"
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ training
def fit(model: nn.Module, make_batch, n_train: int, x_val: np.ndarray, y_val: np.ndarray,
        epochs: int = 60, batch_size: int = 256, lr: float = 2e-3, patience: int = 12,
        seed: int = 42):
    """Train with AdamW + cosine schedule; keep the weights with the best validation accuracy.

    make_batch(idx, rng) -> (features float32, labels int64) builds an augmented batch.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, int(np.ceil(n_train / batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        tot_loss = tot_ok = seen = 0
        for s in range(steps):
            idx = order[s * batch_size:(s + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb, yb = make_batch(idx, rng)
            xb, yb = torch.from_numpy(xb), torch.from_numpy(yb)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_ok += (out.argmax(1) == yb).sum().item()
            seen += len(idx)

        model.eval()
        with torch.inference_mode():
            out = model(xv) if len(xv) else torch.zeros(0, 1)
            vl = loss_fn(out, yv).item() if len(xv) else 0.0
            va = (out.argmax(1) == yv).float().mean().item() if len(xv) else tot_ok / max(seen, 1)
        hist["train_loss"].append(tot_loss / max(seen, 1))
        hist["train_acc"].append(tot_ok / max(seen, 1))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        log(f"epoch {ep:3d}/{epochs}  loss {hist['train_loss'][-1]:.4f}  "
            f"train acc {hist['train_acc'][-1]:.3f}  val acc {va:.3f}")

        if va > best_acc + 1e-4:
            best_acc, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop: no improvement for {patience} epochs")
                break
    model.load_state_dict(best_state)
    model.eval()
    return hist, best_acc


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = [torch.softmax(model(torch.from_numpy(x[i:i + batch])), 1).numpy()
            for i in range(0, len(x), batch)]
    return np.concatenate(outs) if outs else np.zeros((0, 0))


@torch.inference_mode()
def latency_ms(model: nn.Module, sample: np.ndarray, runs: int = 200) -> float:
    x = torch.from_numpy(sample[None])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                       # measure like a single live request
    for _ in range(10):
        model(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(threads)
    return float(np.median(times))


def scores(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f)}


def run_baselines(x_tr, y_tr, x_te, y_te, max_train: int = 15000, seed: int = 42) -> dict:
    """Classic ML baselines for comparison tables in the paper."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if len(x_tr) > max_train:
        idx = np.random.default_rng(seed).choice(len(x_tr), max_train, replace=False)
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "Random Forest": RandomForestClassifier(300, n_jobs=-1, random_state=seed),
    }
    out = {}
    for name, m in models.items():
        t = time.perf_counter()
        m.fit(x_tr, y_tr)
        fit_s = time.perf_counter() - t
        res = scores(y_te, m.predict(x_te))
        res["fit_seconds"] = round(fit_s, 2)
        out[name] = res
        log(f"baseline {name:20s} acc {res['accuracy']:.4f}  f1 {res['f1']:.4f}")
    return out


# ------------------------------------------------------------------ reports
def _rotate(ax, names):
    """Angle tick labels when they are long or numerous so they never overlap."""
    if len(names) > 20 or max((len(n) for n in names), default=0) > 3:
        plt.setp(ax.get_xticklabels(), rotation=45 if len(names) <= 20 else 90, ha="right", rotation_mode="anchor")


def write_reports(out_dir: Path, title: str, labels: list[str], y_true, y_pred, hist: dict,
                  comparison: dict, dataset_counts: dict, extra: dict, names: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = names or [l.replace("_", " ") for l in labels]
    many = len(labels) > 40            # e.g. 100 sentences: label axes by number, not by text
    present = sorted(set(y_true) | set(y_pred))

    # confusion matrix (row-normalised)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    side = 10 if many else max(5, 0.32 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    ticks = 10 if many else names
    sns.heatmap(cmn, ax=ax, cmap=sns.light_palette(ACCENT, as_cmap=True), vmin=0, vmax=1,
                xticklabels=ticks, yticklabels=ticks, square=True, cbar_kws={"shrink": 0.6},
                annot=len(labels) <= 15, fmt=".2f", linewidths=0 if many else 0.3, linecolor="#FFFFFF")
    ax.set_xlabel("Predicted sign" + (" (class number)" if many else ""))
    ax.set_ylabel("True sign" + (" (class number)" if many else ""))
    ax.set_title(f"{title}: confusion matrix (test set)")
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ep = range(1, len(hist["train_loss"]) + 1)
    axes[0].plot(ep, hist["train_loss"], color=INK, label="train")
    axes[0].plot(ep, hist["val_loss"], color=ACCENT, label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(ep, hist["train_acc"], color=INK, label="train")
    axes[1].plot(ep, hist["val_acc"], color=ACCENT, label="validation")
    axes[1].set_title("Accuracy")
    for a in axes:
        a.set_xlabel("Epoch")
        a.legend(frameon=False)
    fig.savefig(out_dir / "training_curves.png")
    plt.close(fig)

    # model comparison
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(comparison) + 1))
    items = sorted(comparison.items(), key=lambda kv: kv[1]["accuracy"])
    colors = [ACCENT if "(ours)" in k else MUTED for k, _ in items]
    ax.barh([k for k, _ in items], [v["accuracy"] * 100 for _, v in items], color=colors)
    for i, (_, v) in enumerate(items):
        ax.text(v["accuracy"] * 100 + 0.5, i, f"{v['accuracy'] * 100:.1f}%", va="center")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title(f"{title}: model comparison")
    fig.savefig(out_dir / "model_comparison.png")
    plt.close(fig)

    # per-class F1
    rep = classification_report(y_true, y_pred, labels=present,
                                target_names=[names[i] for i in present], zero_division=0,
                                output_dict=True)
    f1s = [rep[names[i]]["f1-score"] for i in present]
    if many:
        # too many classes to name: show the sorted F1 of every class
        order = np.argsort(f1s)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.bar(range(len(order)), [f1s[i] for i in order],
               color=[ACCENT if f1s[i] >= 0.9 else WEAK for i in order], width=0.9)
        ax.set_xlabel(f"{title} sorted by F1 ({len(order)} classes)")
        ax.set_xticks([])
    else:
        fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(present) + 2), 3))
        ax.bar([names[i] for i in present], f1s, color=[ACCENT if f >= 0.9 else WEAK for f in f1s])
        _rotate(ax, [names[i] for i in present])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title(f"{title}: F1 score per sign")
    fig.savefig(out_dir / "per_class_f1.png")
    plt.close(fig)

    # dataset distribution
    fig, ax = plt.subplots(figsize=(10 if many else max(6, 0.28 * len(labels) + 2), 3))
    bottom = np.zeros(len(labels))
    palette = [ACCENT, "#7CA2F4", INK, MUTED, "#F79009"]
    for k, (src, counter) in enumerate(dataset_counts.items()):
        vals = np.array([counter.get(l, 0) for l in labels])
        ax.bar(range(len(labels)) if many else names, vals, bottom=bottom, label=src,
               color=palette[k % len(palette)])
        bottom += vals
    ax.set_ylabel("Samples")
    ax.set_title(f"{title}: samples per sign")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))   # outside the bars
    if many:
        ax.set_xlabel("Class number")
    else:
        _rotate(ax, names)
    fig.savefig(out_dir / "dataset_distribution.png")
    plt.close(fig)

    text_rep = classification_report(y_true, y_pred, labels=present,
                                     target_names=[names[i] for i in present], zero_division=0, digits=4)
    (out_dir / "classification_report.txt").write_text(text_rep)

    metrics = {
        "title": title, "labels": labels, "test": scores(y_true, y_pred),
        "comparison": comparison, "history": hist,
        "per_class": {names[i]: {"precision": rep[names[i]]["precision"], "recall": rep[names[i]]["recall"],
                                 "f1": rep[names[i]]["f1-score"], "support": rep[names[i]]["support"]}
                      for i in present},
        **extra,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def counts_by_source(labels: np.ndarray, sources: np.ndarray) -> dict:
    return {src: Counter(labels[sources == src].tolist()) for src in np.unique(sources)}

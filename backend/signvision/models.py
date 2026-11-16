"""Neural networks for SignVision and helpers to save/load them."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class StaticMLP(nn.Module):
    """Classifies one frame of hand landmarks into a letter / number."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.hparams = dict(in_dim=in_dim, num_classes=num_classes, hidden=hidden, dropout=dropout)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):            # x: (B, in_dim)
        return self.net(x)


class SequenceClassifier(nn.Module):
    """Classifies a clip of landmark frames into a word.

    Linear projection -> 2-layer bidirectional GRU or LSTM -> attention pooling
    over time -> linear classifier. Attention lets the model focus on the
    frames where the sign actually happens.
    """

    def __init__(self, in_dim: int, num_classes: int, cell: str = "gru", hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hparams = dict(in_dim=in_dim, num_classes=num_classes, cell=cell, hidden=hidden,
                            layers=layers, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(hidden, hidden, num_layers=layers, batch_first=True,
                       bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def forward(self, x):            # x: (B, T, in_dim)
        h, _ = self.rnn(self.proj(x))
        w = torch.softmax(self.attn(h), dim=1)          # (B, T, 1)
        return self.head((w * h).sum(dim=1))


def save_checkpoint(path: Path, model: nn.Module, kind: str, labels: list[str],
                    meta: dict | None = None) -> None:
    torch.save({
        "kind": kind,                       # "static" or "dynamic"
        "arch": type(model).__name__,
        "hparams": model.hparams,
        "state_dict": model.state_dict(),
        "labels": labels,
        "meta": meta or {},
    }, path)


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cls = {"StaticMLP": StaticMLP, "SequenceClassifier": SequenceClassifier}[ckpt["arch"]]
    model = cls(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["labels"], ckpt.get("meta", {})

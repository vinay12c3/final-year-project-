"""Offline speech-to-text for the hearing side of a conversation.

Uses faster-whisper (an open-source CTranslate2 port of OpenAI Whisper, MIT
licence) with the English "base.en" model, quantised to int8 so it runs on a
laptop CPU. The model (~145 MB) is downloaded once into backend/assets and
then works without internet.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from . import config as C

MODEL_NAME = "base.en"
MODEL_REPO = "Systran/faster-whisper-base.en"
MODEL_DIR = C.ASSETS_DIR / "whisper-base.en"
# Words the conversation uses: nudges Whisper toward the right spellings.
PROMPT = "Hello, how are you? Good morning. Thank you. What is your name? My name is Ravi. Please, sorry, yes, no."

_model = None
_lock = threading.Lock()


def is_downloaded() -> bool:
    return (MODEL_DIR / "model.bin").exists()


def download() -> Path:
    """Fetches the model files once (needs internet only this one time)."""
    if not is_downloaded():
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_REPO, local_dir=str(MODEL_DIR))
    return MODEL_DIR


def _load():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            download()
            _model = WhisperModel(str(MODEL_DIR), device="cpu", compute_type="int8", cpu_threads=4)
    return _model


def transcribe(audio_path: str | Path) -> dict:
    """Returns {"text", "seconds", "audio_seconds"} for a recorded clip (webm/ogg/wav/mp3...)."""
    model = _load()
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,                  # skip silence before/after speaking
        initial_prompt=PROMPT,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "seconds": round(time.perf_counter() - t0, 2),
            "audio_seconds": round(float(info.duration), 2)}

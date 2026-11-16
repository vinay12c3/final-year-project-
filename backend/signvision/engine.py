"""Real-time translation: frames in, recognised signs and a growing sentence out."""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np
import torch

from . import config as C
from .features import dynamic_features, resample, sentence_features, static_features
from .hands import HandTracker, detect_image, enhance
from .models import load_checkpoint

DEBUG = os.environ.get("SIGNVISION_DEBUG") == "1"   # log what the model sees once per second

torch.set_num_threads(1)   # one live request = one small model call


class Translator:
    """Holds the trained models. One instance is shared by every connection."""

    def __init__(self):
        self._lock = threading.Lock()
        self.static = self.dynamic = self.sentence = None
        self.static_labels: list[str] = []
        self.dynamic_labels: list[str] = []
        self.sentence_labels: list[str] = []
        self.static_meta: dict = {}
        self.dynamic_meta: dict = {}
        self.sentence_meta: dict = {}
        self.sentence_texts: dict[str, str] = {}
        self._mtimes: dict = {}
        self.reload()

    @staticmethod
    def _mtime(path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def reload(self) -> None:
        with self._lock:
            if C.STATIC_MODEL.exists():
                self.static, self.static_labels, self.static_meta = load_checkpoint(C.STATIC_MODEL)
            if C.DYNAMIC_MODEL.exists():
                self.dynamic, self.dynamic_labels, self.dynamic_meta = load_checkpoint(C.DYNAMIC_MODEL)
            if C.SENTENCE_MODEL.exists():
                self.sentence, self.sentence_labels, self.sentence_meta = load_checkpoint(C.SENTENCE_MODEL)
                self.sentence_texts = self.sentence_meta.get("texts", {})
            self._mtimes = {p: self._mtime(p) for p in C.MODEL_PATHS.values()}

    def refresh(self) -> None:
        """Reload if a model file changed on disk (e.g. trained with train.bat while the app runs)."""
        if any(self._mtime(p) != t for p, t in self._mtimes.items()):
            try:
                self.reload()
            except Exception:      # file still being written; try again next time
                pass

    def status(self) -> dict:
        self.refresh()
        return {
            "letters": {"ready": self.static is not None, "labels": self.static_labels,
                        "trained_at": self.static_meta.get("trained_at"),
                        "test_accuracy": self.static_meta.get("test_accuracy")},
            "words": {"ready": self.dynamic is not None, "labels": self.dynamic_labels,
                      "trained_at": self.dynamic_meta.get("trained_at"),
                      "test_accuracy": self.dynamic_meta.get("test_accuracy")},
            "sentences": {"ready": self.sentence is not None, "labels": self.sentence_labels,
                          "texts": self.sentence_texts,
                          "trained_at": self.sentence_meta.get("trained_at"),
                          "test_accuracy": self.sentence_meta.get("test_accuracy")},
        }

    @torch.inference_mode()
    def predict_static(self, raw: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(static_features(raw)[None])
        return torch.softmax(self.static(x), dim=1)[0].numpy()

    @torch.inference_mode()
    def predict_dynamic(self, clip: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(dynamic_features(resample(clip, C.SEQ_LEN))[None])
        return torch.softmax(self.dynamic(x), dim=1)[0].numpy()

    @torch.inference_mode()
    def predict_sentence(self, clip: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(sentence_features(resample(clip, C.SENT_SEQ_LEN))[None])
        return torch.softmax(self.sentence(x), dim=1)[0].numpy()


def _top(probs: np.ndarray, labels: list[str], k: int = 3, names: dict | None = None) -> list[dict]:
    order = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "display": (names or {}).get(labels[i]) or C.display_label(labels[i]),
             "confidence": round(float(probs[i]), 4)} for i in order]


def _sentence_top(probs: np.ndarray, tr, k: int = 3) -> list[dict]:
    """Best real sentences (the "not a sign" class is never offered)."""
    keep = [i for i, l in enumerate(tr.sentence_labels) if not C.is_background(l)]
    sub = probs[keep]
    return _top(sub, [tr.sentence_labels[i] for i in keep], k=k, names=tr.sentence_texts)


def _travel(clip: np.ndarray) -> float:
    """How far the wrists moved in total (largest spread of either wrist, x or y)."""
    w = clip[:, :, 0, :2]
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        spread = np.nanmax(w, axis=0) - np.nanmin(w, axis=0)
    spread = spread[np.isfinite(spread)]
    return float(spread.max()) if spread.size else 0.0


class SentenceBuilder:
    """Keeps the translated text and supports manual editing commands."""

    def __init__(self):
        self.text = ""

    def add_letter(self, letter: str) -> None:
        self.text += letter

    def add_word(self, word: str) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self.text += word + " "

    def space(self) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""


class LiveSession:
    """State for one webcam connection.

    Letters mode: a letter is typed after it has been recognised confidently
    for LETTER_HOLD_SECONDS. Lowering the hands for SPACE_AFTER_SECONDS adds
    a space. The same letter twice in a row needs the hand to change in between.

    Words mode: the last WORD_WINDOW_SECONDS of frames are classified several
    times a second. A word is typed when WORD_AGREE predictions in a row agree
    with enough confidence. The same word is not typed twice in a row unless
    the hands drop or a different sign is seen in between.

    Sentences mode: the signer raises the hands, signs the whole sentence and
    lowers them. When the hands have been out of view for SENTENCE_END_PAUSE,
    the whole movement is classified once and the sentence is written.
    """

    def __init__(self, translator: Translator, mode: str = "letters"):
        translator.refresh()
        self.translator = translator
        self.tracker = HandTracker(mode="video")
        self.sentence = SentenceBuilder()
        self.mode = mode
        self._reset_state()

    def _reset_state(self):
        self.hold_label, self.hold_since = None, 0.0
        self.last_typed, self.released = None, True
        self.last_hand_time = time.monotonic()
        self.typed_since_gap = False
        self.window: deque = deque()
        self.last_word_pred = 0.0
        self.streak_label, self.streak = None, 0
        self.word_released = True
        self.last_word_time = 0.0
        self.smooth = None            # averaged letter probabilities over recent frames
        self._last_log = 0.0
        self.segment: list = []        # sentences: (time, raw) since the hands came up
        self.idle_after_still = False
        self.last_preview = 0.0

    def close(self):
        self.tracker.close()

    def set_mode(self, mode: str):
        if mode in ("letters", "words", "sentences") and mode != self.mode:
            self.mode = mode
            self._reset_state()

    # ------------------------------------------------------------------
    def process(self, bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        now = time.monotonic()
        frame = enhance(bgr)
        raw, hands = self.tracker.process(frame, int(now * 1000))
        t_detect = time.perf_counter()

        out = {"type": "frame", "mode": self.mode, "hands": hands, "current": None,
               "top": [], "progress": 0.0, "typed": None, "model_ready": True}
        if self.mode == "letters":
            self._letters(raw, bool(hands), now, out)
        elif self.mode == "sentences":
            self._sentences(raw, bool(hands), now, out)
        else:
            self._words(raw, bool(hands), now, out)

        out["text"] = self.sentence.text
        out["timing"] = {"detect_ms": round((t_detect - t0) * 1000, 1),
                         "model_ms": round((time.perf_counter() - t_detect) * 1000, 1),
                         "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if DEBUG and now - self._last_log > 1.0:
            self._last_log = now
            guesses = ", ".join(f"{g['label']} {g['confidence']:.0%}" for g in out["top"]) or "-"
            print(f"[live] {self.mode} {bgr.shape[1]}x{bgr.shape[0]} hands={len(hands)} "
                  f"guess=[{guesses}] hold={out['progress']:.0%} text={self.sentence.text!r}", flush=True)
            import cv2
            cv2.imwrite(str(C.REPORTS_DIR / "debug_last_frame.jpg"), bgr)
        return out

    def _letters(self, raw, has_hand, now, out):
        tr = self.translator
        if tr.static is None:
            out["model_ready"] = False
            return
        if not has_hand:
            self.hold_label, self.released = None, True
            self.smooth = None
            if self.typed_since_gap and now - self.last_hand_time > C.SPACE_AFTER_SECONDS:
                self.sentence.space()
                self.typed_since_gap = False
                out["typed"] = " "
            return
        self.last_hand_time = now

        probs = tr.predict_static(raw)
        # Average over recent frames so a single wobbly frame doesn't reset the hold.
        self.smooth = probs if self.smooth is None else (1 - C.LETTER_SMOOTHING) * self.smooth + C.LETTER_SMOOTHING * probs
        top = _top(self.smooth, tr.static_labels)
        out["top"], out["current"] = top, top[0]
        label, conf = top[0]["label"], top[0]["confidence"]

        if conf < C.LETTER_MIN_CONFIDENCE or C.is_background(label):
            self.hold_label = None
            return
        if label != self.hold_label:
            self.hold_label, self.hold_since = label, now
            if label != self.last_typed:
                self.released = True
        held = now - self.hold_since
        out["progress"] = round(min(1.0, held / C.LETTER_HOLD_SECONDS), 3)
        if held >= C.LETTER_HOLD_SECONDS and self.released:
            self.sentence.add_letter(C.display_label(label))
            out["typed"] = label
            self.last_typed, self.released = label, False
            self.typed_since_gap = True

    def _words(self, raw, has_hand, now, out):
        tr = self.translator
        if tr.dynamic is None:
            out["model_ready"] = False
            return
        if not has_hand and now - self.last_hand_time > 0.5:
            self.word_released = True                 # hands dropped: the same word may be typed again
        if has_hand:
            self.last_hand_time = now
        self.window.append((now, raw, has_hand))
        while self.window and now - self.window[0][0] > C.WORD_WINDOW_SECONDS:
            self.window.popleft()
        span = now - self.window[0][0]
        out["progress"] = round(min(1.0, span / C.WORD_WINDOW_SECONDS), 3)

        # Only judge a (nearly) full window in which hands are visible most of the time;
        # half-finished signs at the start of a movement are the main source of wrong words.
        hand_frames = sum(1 for _, _, h in self.window if h)
        if span < 0.7 * C.WORD_WINDOW_SECONDS or hand_frames < 8 or hand_frames < 0.6 * len(self.window):
            self.streak_label, self.streak = None, 0
            return
        if now - self.last_word_pred < C.WORD_PREDICT_EVERY:
            return
        self.last_word_pred = now

        clip = np.stack([r for _, r, _ in self.window])
        probs = tr.predict_dynamic(clip)
        top = _top(probs, tr.dynamic_labels)
        out["top"], out["current"] = top, top[0]
        label, conf = top[0]["label"], top[0]["confidence"]

        confident = conf >= C.WORD_MIN_CONFIDENCE and not C.is_background(label)
        if confident and label == self.streak_label:
            self.streak += 1
        else:
            self.streak_label, self.streak = (label, 1) if confident else (None, 0)
        if confident and label != self.last_typed:
            self.word_released = True                 # a different sign was seen in between

        if (self.streak >= C.WORD_AGREE and now - self.last_word_time >= C.WORD_COOLDOWN_SECONDS
                and (label != self.last_typed or self.word_released)):
            self.sentence.add_word(C.display_label(label))
            out["typed"] = label
            self.last_typed, self.word_released = label, False
            self.last_word_time = now
            self.streak_label, self.streak = None, 0
            self.window.clear()


    def _sentences(self, raw, has_hand, now, out):
        tr = self.translator
        if tr.sentence is None:
            out["model_ready"] = False
            return
        if has_hand:
            self.last_hand_time = now
            self.segment.append((now, raw))
            while now - self.segment[0][0] > C.SENTENCE_MAX_SECONDS:
                self.segment.pop(0)
            out["signing"] = round(now - self.segment[0][0], 2)
            # Hands kept in view but no longer moving also ends the sentence.
            still_since = self._still_since()
            if still_since is not None and not self._moved(still_since):
                # hands just resting in view: nothing has been signed yet
                self.segment = [(t, r) for t, r in self.segment if t > still_since]
                out["signing"] = 0.0
                return
            if still_since is not None and still_since - self.segment[0][0] >= C.SENTENCE_MIN_SECONDS:
                frames = [(t, r) for t, r in self.segment if t <= still_since]
                self.segment = [(t, r) for t, r in self.segment if t > still_since]
                self.idle_after_still = True
                self._finish_sentence(frames, out)
                return
            if self.idle_after_still:
                # wait for movement again before starting the next sentence
                if self._still_since() is not None or len(self.segment) < 3:
                    self.segment = self.segment[-int(C.SENTENCE_STILL_SECONDS * 15):]
                    return
                self.idle_after_still = False
            # a live preview of the best guess so far, twice a second
            if now - self.last_preview > 0.5 and now - self.segment[0][0] >= C.SENTENCE_MIN_SECONDS:
                self.last_preview = now
                top = _sentence_top(tr.predict_sentence(np.stack([r for _, r in self.segment])), tr)
                out["top"], out["current"] = top, None
            return
        if not self.segment:
            return
        pause = now - self.last_hand_time
        out["progress"] = round(min(1.0, pause / C.SENTENCE_END_PAUSE), 3)
        if self.segment and pause < C.SENTENCE_END_PAUSE:
            self.segment.append((now, raw))              # brief detection gaps stay part of the sentence
            return
        frames = [(t, r) for t, r in self.segment if t <= self.last_hand_time]
        self.segment = []
        self.idle_after_still = False
        self._finish_sentence(frames, out)

    def _moved(self, until: float) -> bool:
        """True if the wrists travelled clearly more than 'still' before the given time."""
        w = [r[:, 0, :2] for t, r in self.segment if t <= until]
        if len(w) < 5:
            return False
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            spread = np.nanmax(np.stack(w), axis=0) - np.nanmin(np.stack(w), axis=0)
        spread = spread[np.isfinite(spread)]
        return spread.size > 0 and spread.max() > 4 * C.SENTENCE_STILL_MOVE

    def _still_since(self):
        """Time from which the wrists have stayed (almost) still for SENTENCE_STILL_SECONDS, else None."""
        if not self.segment:
            return None
        now = self.segment[-1][0]
        recent = [r for t, r in self.segment if now - t <= C.SENTENCE_STILL_SECONDS]
        if self.segment[0][0] > now - C.SENTENCE_STILL_SECONDS or len(recent) < 5:
            return None
        w = np.stack([r[:, 0, :2] for r in recent])                  # (n, 2 hands, xy)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            spread = np.nanmax(w, axis=0) - np.nanmin(w, axis=0)
        spread = spread[np.isfinite(spread)]
        if spread.size == 0 or spread.max() > C.SENTENCE_STILL_MOVE:
            return None
        return now - C.SENTENCE_STILL_SECONDS

    def _finish_sentence(self, frames, out):
        tr = self.translator
        if not frames or frames[-1][0] - frames[0][0] < C.SENTENCE_MIN_SECONDS:
            return
        clip = np.stack([r for _, r in frames])
        if _travel(clip) < C.SENTENCE_MIN_TRAVEL:
            return                                   # small fidgets, not a sign
        probs = tr.predict_sentence(clip)
        best = int(np.argmax(probs))
        if C.is_background(tr.sentence_labels[best]):
            return                                   # the model says "not a sign"
        top = _sentence_top(probs, tr)
        out["top"] = top
        conf = top[0]["confidence"]
        margin = conf - (top[1]["confidence"] if len(top) > 1 else 0.0)
        if conf >= C.SENTENCE_MIN_CONFIDENCE and margin >= C.SENTENCE_MIN_MARGIN:
            out["current"] = top[0]
            self.sentence.add_word(top[0]["display"])
            out["typed"] = top[0]["label"]
            out["typed_text"] = top[0]["display"]
        elif conf >= C.SENTENCE_UNSURE_CONFIDENCE:
            out["current"] = top[0]                  # shown as "Did you mean...?" buttons


# ---------------------------------------------------------------------- uploads
def predict_image(translator: Translator, bgr: np.ndarray) -> dict:
    translator.refresh()
    with HandTracker(mode="image") as tracker:
        raw, hands = detect_image(tracker, bgr)
    res = {"hands": hands, "top": [], "model_ready": translator.static is not None}
    if hands and translator.static is not None:
        res["top"] = _top(translator.predict_static(raw), translator.static_labels, k=5)
    return res


def predict_video(translator: Translator, path: str, mode: str) -> dict:
    """Run a whole video file through the recogniser.

    Returns the typed text, a timeline of recognised signs, and (for words
    mode) a single best guess for the entire clip, which is what you want for
    short one-word clips like the INCLUDE dataset.
    """
    import cv2

    translator.refresh()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / 15)))            # analyse ~15 frames per second
    frames, times = [], []
    with HandTracker(mode="video") as tracker:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                t = i / fps
                raw, _ = tracker.process(enhance(bgr), int(t * 1000))
                frames.append(raw)
                times.append(t)
            i += 1
    cap.release()
    duration = round(i / fps, 2)
    if not frames:
        return {"error": "Could not read any frames from this video. Try an MP4 or WEBM file."}

    has = np.array([not np.isnan(f[:, 0, 0]).all() for f in frames])
    result = {"duration": duration, "frames": len(frames), "hand_frames": int(has.sum()),
              "timeline": [], "text": "", "clip_top": []}

    if mode == "letters":
        if translator.static is None:
            return {**result, "error": "The letters model is not trained yet."}
        run_label, run_start, run_len = None, 0.0, 0
        min_len = 7                                       # ~0.5 s of the same letter at 15 fps
        seq = []
        for f, t, h in zip(frames, times, has):
            lab = None
            if h:
                probs = translator.predict_static(f)
                j = int(probs.argmax())
                if probs[j] >= C.LETTER_MIN_CONFIDENCE:
                    lab = translator.static_labels[j]
            if lab == run_label:
                run_len += 1
            else:
                if run_label and run_len >= min_len:
                    seq.append((run_label, run_start, t))
                run_label, run_start, run_len = lab, t, 1
        if run_label and run_len >= min_len:
            seq.append((run_label, run_start, times[-1]))
        result["timeline"] = [{"label": l, "display": C.display_label(l), "start": round(a, 2),
                               "end": round(b, 2)} for l, a, b in seq if not C.is_background(l)]
        result["text"] = "".join(x["display"] for x in result["timeline"])
        return result

    if mode == "sentences":
        if translator.sentence is None:
            return {**result, "error": "The sentences model is not trained yet."}
        if has.sum() < 5:
            return {**result, "error": "No hands were found in this video."}
        first, last = int(np.argmax(has)), len(has) - int(np.argmax(has[::-1]))
        top = _sentence_top(translator.predict_sentence(np.stack(frames[first:last])), translator, k=5)
        result["clip_top"] = top
        if top[0]["confidence"] >= C.SENTENCE_MIN_CONFIDENCE:
            result["timeline"] = [{**top[0], "start": round(times[first], 2), "end": round(times[last - 1], 2)}]
            result["text"] = top[0]["display"]
        return result

    if translator.dynamic is None:
        return {**result, "error": "The words model is not trained yet."}
    clip = np.stack(frames)
    result["clip_top"] = _top(translator.predict_dynamic(clip), translator.dynamic_labels, k=5)

    win = int(C.WORD_WINDOW_SECONDS * 15)
    step = max(1, win // 4)
    words = []
    for s in range(0, max(1, len(frames) - win + 1), step):
        part = clip[s:s + win]
        if has[s:s + win].mean() < 0.5:
            continue
        probs = translator.predict_dynamic(part)
        j = int(probs.argmax())
        lab = translator.dynamic_labels[j]
        if probs[j] >= C.WORD_MIN_CONFIDENCE and not C.is_background(lab):
            t0, t1 = times[s], times[min(s + win, len(times)) - 1]
            if words and words[-1]["label"] == lab and t0 <= words[-1]["end"]:
                words[-1]["end"] = round(t1, 2)
                words[-1]["confidence"] = max(words[-1]["confidence"], round(float(probs[j]), 4))
            else:
                words.append({"label": lab, "display": C.display_label(lab), "start": round(t0, 2),
                              "end": round(t1, 2), "confidence": round(float(probs[j]), 4)})
    result["timeline"] = words
    result["text"] = " ".join(w["display"] for w in words)
    return result

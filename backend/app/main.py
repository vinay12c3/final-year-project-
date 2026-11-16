"""SignVision backend: REST API + WebSockets + serves the built frontend.

Run from the backend folder:
    python -m uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from signvision import config as C
from signvision import speech
from signvision.conversation import PHRASES, compose
from signvision import data as D
from signvision.engine import LiveSession, Translator, predict_image, predict_video
from signvision.hands import HandTracker, enhance

from signvision.tunnel import Tunnel

from . import db
from .rooms import Rooms
from .training import TrainingManager

translator: Translator
trainer: TrainingManager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global translator, trainer
    db.init_db()
    translator = Translator()
    trainer = TrainingManager(on_finished=translator.reload)
    yield


app = FastAPI(title="SignVision API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rooms = Rooms()
tunnel = Tunnel(port=int(os.environ.get("SIGNVISION_PORT", "8000")))

# What a visitor coming through the public internet link may open: only the room
# page and its chat. Everything else (recordings, history, training) stays private.
_PUBLIC_PREFIXES = ("/room/", "/assets/", "/ws/room/", "/api/rooms/")
_PUBLIC_FILES = ("/favicon.svg", "/favicon.ico", "/manifest.webmanifest")


class TunnelGuard:
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers") or [])
            from_tunnel = b"cf-ray" in headers or b"cf-connecting-ip" in headers
            path = scope.get("path", "")
            if from_tunnel and not (path.startswith(_PUBLIC_PREFIXES) or path in _PUBLIC_FILES):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 4403})
                    return
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
                await send({"type": "http.response.body",
                            "body": b"This SignVision link only opens a conversation room."})
                return
        await self.inner(scope, receive, send)


app.add_middleware(TunnelGuard)


def decode_image(buf: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ status
@app.get("/api/status")
def status():
    return {"models": translator.status(), "hand_model": C.HAND_MODEL.exists(),
            "speech": {"model": speech.MODEL_NAME, "downloaded": speech.is_downloaded()}}


@app.get("/api/vocabulary")
def vocabulary():
    return {
        "letters": {"suggested": C.LETTERS, "recorded": D.recording_counts("static"),
                    "public": D.processed_counts("static")},
        "words": {"suggested": C.WORDS, "recorded": D.recording_counts("dynamic"),
                  "public": D.processed_counts("dynamic")},
        "sentences": {"suggested": sorted(D.sentence_texts()), "texts": D.sentence_texts(),
                      "recorded": D.recording_counts("sentence"), "public": D.processed_counts("sentence")},
    }


@app.get("/api/metrics")
def metrics():
    out = {}
    for kind in C.KINDS:
        f = C.REPORTS_DIR / kind / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            m["figures"] = sorted(f"/files/reports/{kind}/{p.name}"
                                  for p in (C.REPORTS_DIR / kind).glob("*.png"))
            out[kind] = m
    return out


# ------------------------------------------------------------------ uploads
@app.post("/api/predict/image")
async def api_predict_image(file: UploadFile = File(...)):
    img = decode_image(await file.read())
    if img is None:
        raise HTTPException(400, "This file is not an image we can read. Use JPG or PNG.")
    return await run_in_threadpool(predict_image, translator, img)


@app.post("/api/predict/video")
async def api_predict_video(file: UploadFile = File(...), mode: str = Form("words")):
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(await file.read())
        return await run_in_threadpool(predict_video, translator, path, mode)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------------ history
class HistoryIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    mode: str = "letters"
    source: str = "camera"


# ------------------------------------------------------------------ conversation
class ComposeIn(BaseModel):
    tokens: list[dict] = Field(default_factory=list, max_length=200)


@app.post("/api/conversation/compose")
def conversation_compose(body: ComposeIn):
    """Signs so far -> natural sentence, e.g. [hello, how_are_you] -> 'Hello, how are you?'."""
    return {"sentence": compose(body.tokens)}


@app.get("/api/conversation/phrases")
def conversation_phrases():
    return {label: {"text": p.text, "kind": p.kind} for label, p in PHRASES.items()}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Speech from the hearing person -> text, fully offline (faster-whisper)."""
    data = await audio.read()
    if len(data) < 1000:
        raise HTTPException(400, "The recording is empty. Hold the button while you speak.")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "Recording too long. Keep it under a minute.")
    suffix = Path(audio.filename or "speech.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return await run_in_threadpool(speech.transcribe, path)
    except Exception as e:  # noqa: BLE001 - report decoder/model problems to the UI
        raise HTTPException(500, f"Speech recognition failed: {e}") from e
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ rooms & sharing
@app.post("/api/rooms")
def api_create_room():
    return {"code": rooms.create().code}


@app.get("/api/rooms/{code}")
def api_room(code: str):
    room = rooms.get(code)
    if room is None:
        raise HTTPException(404, "This room does not exist or has ended. Ask for a new link.")
    return {"code": room.code, "messages": room.messages, **room.presence()}


@app.post("/api/rooms/{code}/transcribe")
async def api_room_transcribe(code: str, audio: UploadFile = File(...)):
    if rooms.get(code) is None:
        raise HTTPException(404, "This room has ended.")
    return await transcribe(audio)


@app.get("/api/share")
def api_share_status():
    return {**tunnel.status(), "lan": _lan_address()}


@app.post("/api/share/start")
def api_share_start():
    return tunnel.start()


@app.post("/api/share/stop")
def api_share_stop():
    return tunnel.stop()


def _lan_address() -> str | None:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))          # no packet is sent; picks the Wi-Fi address
            return s.getsockname()[0]
    except OSError:
        return None


@app.websocket("/ws/room/{code}")
async def ws_room(ws: WebSocket, code: str, role: str = "guest"):
    room = rooms.get(code)
    await ws.accept()
    if room is None:
        await ws.send_json({"type": "error", "message": "This room does not exist or has ended."})
        await ws.close()
        return
    role = "host" if role == "host" else "guest"
    room.clients[ws] = role
    await ws.send_json({"type": "history", "code": room.code, "messages": room.messages})
    await room.broadcast(room.presence())
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") != "message" or not str(msg.get("text", "")).strip():
                continue
            # the laptop sends Person A's signs; the phone sends Person B's replies
            who = "signer" if role == "host" and msg.get("who") != "speaker" else "speaker"
            m = rooms.add_message(room, who, str(msg["text"]).strip(), msg.get("signs"))
            await ws.send_json({"type": "ack", "client_id": msg.get("client_id"), "message": m})
            await room.broadcast({"type": "message", "message": m}, skip=ws)
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        room.clients.pop(ws, None)
        await room.broadcast(room.presence())


@app.get("/api/history")
def history():
    return db.list_translations()


@app.post("/api/history")
def history_add(item: HistoryIn):
    return db.add_translation(item.text.strip(), item.mode, item.source)


@app.delete("/api/history/{item_id}")
def history_delete(item_id: int):
    return {"deleted": db.delete_translation(item_id)}


@app.delete("/api/history")
def history_clear():
    return {"deleted": db.delete_translation(None)}


# ------------------------------------------------------------------ recordings & training
@app.delete("/api/recordings/{kind}/{label}/last")
def recording_delete_last(kind: str, label: str):
    if kind not in C.KINDS:
        raise HTTPException(404)
    return {"deleted": D.delete_last_recording(kind, label), "recorded": D.recording_counts(kind)}


@app.post("/api/train/{kind}")
def train_start(kind: str):
    if kind not in C.KINDS:
        raise HTTPException(404)
    return trainer.start(kind)


@app.get("/api/train/{kind}")
def train_status(kind: str):
    if kind not in C.KINDS:
        raise HTTPException(404)
    return trainer.status(kind)


# ------------------------------------------------------------------ live translation
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    """Client sends JPEG frames as binary messages and JSON commands as text.

    Commands: {"type": "mode", "mode": "letters"|"words"|"sentences"}, {"type": "space"},
              {"type": "backspace"}, {"type": "clear"}
    """
    await ws.accept()
    session = await run_in_threadpool(LiveSession, translator, ws.query_params.get("mode", "letters"))
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                img = decode_image(msg["bytes"])
                if img is None:
                    continue
                await ws.send_json(await run_in_threadpool(session.process, img))
            elif msg.get("text"):
                cmd = json.loads(msg["text"])
                kind = cmd.get("type")
                if kind == "mode":
                    session.set_mode(cmd.get("mode", "letters"))
                elif kind == "space":
                    session.sentence.space()
                elif kind == "backspace":
                    session.sentence.backspace()
                elif kind == "clear":
                    session.sentence.clear()
                await ws.send_json({"type": "text", "text": session.sentence.text, "mode": session.mode})
    except WebSocketDisconnect:
        pass
    finally:
        session.close()


@app.websocket("/ws/record")
async def ws_record(ws: WebSocket):
    """Record training samples from the browser camera.

    Binary messages are preview frames (answered with detected hands).
    {"type": "start", "kind": "static"|"dynamic", "label": "A", "seconds": 3, "who": "name"}
    starts collecting; the server saves the sample when the time is up.
    """
    await ws.accept()
    tracker = await run_in_threadpool(HandTracker, "video")
    rec = None
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                cmd = json.loads(msg["text"])
                if cmd.get("type") == "start" and cmd.get("kind") in C.KINDS and cmd.get("label"):
                    rec = {"kind": cmd["kind"], "label": cmd["label"], "who": cmd.get("who", ""),
                           "seconds": float(cmd.get("seconds", 3)), "t0": time.monotonic(), "frames": []}
                elif cmd.get("type") == "cancel":
                    rec = None
                continue
            if not msg.get("bytes"):
                continue
            img = decode_image(msg["bytes"])
            if img is None:
                continue
            now = time.monotonic()
            raw, hands = await run_in_threadpool(tracker.process, enhance(img), int(now * 1000))
            reply = {"type": "frame", "hands": hands, "recording": rec is not None}
            if rec:
                if rec["kind"] != "static" or hands:
                    rec["frames"].append(raw)
                elapsed = now - rec["t0"]
                reply["elapsed"] = round(elapsed, 2)
                reply["captured"] = len(rec["frames"])
                if elapsed >= rec["seconds"]:
                    frames = rec["frames"]
                    hand_frames = sum(1 for f in frames if not np.isnan(f[:, 0, 0]).all())
                    if hand_frames < 5:
                        reply = {"type": "rejected", "hands": hands,
                                 "reason": "No hands were seen during the recording. "
                                           "Step back so both hands are in the frame and try again."}
                    else:
                        clip = np.stack(frames)
                        if rec["kind"] == "sentence":          # keep only the part where the hands are up
                            has = ~np.isnan(clip[:, :, 0, 0]).all(axis=1)
                            clip = clip[int(np.argmax(has)):len(has) - int(np.argmax(has[::-1]))]
                        path = D.save_recording(rec["kind"], rec["label"], clip, rec["who"])
                        reply = {"type": "saved", "hands": hands, "label": path.parent.name,
                                 "frames": len(frames), "recorded": D.recording_counts(rec["kind"])}
                    rec = None
            await ws.send_json(reply)
    except WebSocketDisconnect:
        pass
    finally:
        tracker.close()


# ------------------------------------------------------------------ files & frontend
app.mount("/files/reports", StaticFiles(directory=C.REPORTS_DIR), name="reports")

if C.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=C.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        dist = C.FRONTEND_DIST.resolve()
        target = (dist / path).resolve()
        if path and target.is_file() and target.is_relative_to(dist):
            return FileResponse(target)
        return FileResponse(C.FRONTEND_DIST / "index.html")

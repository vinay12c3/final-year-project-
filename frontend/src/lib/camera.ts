import { useCallback, useEffect, useRef, useState } from "react";
import type { Hand } from "./api";

/** Webcam access through the browser. Works on http://localhost without HTTPS. */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [on, setOn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setOn(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser can't use the camera. Open the app in Chrome or Edge at http://localhost:8000.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (v) {
        v.srcObject = stream;
        await v.play().catch(() => undefined);
      }
      setOn(true);
    } catch (e) {
      const name = (e as DOMException)?.name;
      setError(
        name === "NotAllowedError"
          ? "Camera permission was blocked. Click the camera icon in the address bar, allow access, then try again."
          : name === "NotFoundError"
            ? "No camera was found. Plug in a webcam and try again."
            : name === "NotReadableError"
              ? "The camera is busy in another app (Zoom, Teams, Camera). Close it and try again."
              : "The camera could not start. Reload the page and try again.",
      );
    }
  }, []);

  useEffect(() => stop, [stop]);
  return { videoRef, on, error, start, stop };
}

/**
 * Streams webcam frames to a backend WebSocket as JPEG and hands each JSON
 * reply to onMessage. Only one frame is in flight at a time, so a slow laptop
 * simply gets a lower frame rate instead of a growing delay.
 */
export function useFrameSocket<M>(
  url: string,
  videoRef: React.RefObject<HTMLVideoElement | null>,
  active: boolean,
  onMessage: (msg: M) => void,
  maxFps = 20,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const handler = useRef(onMessage);
  handler.current = onMessage;
  const [connected, setConnected] = useState(false);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    if (!active) return;
    let closed = false;
    let inFlight = false;
    let sentAt = 0;
    let timer = 0;
    let frames = 0;
    let fpsWindow = performance.now();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      inFlight = false;
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data as string);
      if (msg.type !== "text") {
        inFlight = false;
        frames++;
        const now = performance.now();
        if (now - fpsWindow > 1000) {
          setFps(Math.round((frames * 1000) / (now - fpsWindow)));
          frames = 0;
          fpsWindow = now;
        }
      }
      handler.current(msg as M);
    };

    const tick = () => {
      if (closed) return;
      const v = videoRef.current;
      const now = performance.now();
      if (inFlight && now - sentAt > 2000) inFlight = false; // recover from a lost reply
      if (ws.readyState === WebSocket.OPEN && !inFlight && v && v.videoWidth && ctx) {
        const scale = Math.min(1, 640 / v.videoWidth);
        canvas.width = Math.round(v.videoWidth * scale);
        canvas.height = Math.round(v.videoHeight * scale);
        ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
        inFlight = true;
        sentAt = now;
        canvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
            else inFlight = false;
          },
          "image/jpeg",
          0.72,
        );
      }
      timer = window.setTimeout(tick, 1000 / maxFps);
    };
    tick();

    return () => {
      closed = true;
      clearTimeout(timer);
      ws.close();
      wsRef.current = null;
      setFps(0);
    };
  }, [url, active, videoRef, maxFps]);

  const send = useCallback((data: object) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
  }, []);

  return { connected, fps, send };
}

// The 21-point MediaPipe hand skeleton.
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

/**
 * Draws hand skeletons on a canvas laid over an image/video of size width x height.
 * Pass the media's natural size when it is shown with object-fit: cover, so the
 * points stay aligned with the cropped picture.
 */
export function drawHands(
  canvas: HTMLCanvasElement | null,
  hands: Hand[],
  width: number,
  height: number,
  natural?: { w: number; h: number },
) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const r = Math.max(2.5, Math.min(width, height) / 130);
  let ox = 0, oy = 0, sw = width, sh = height;
  if (natural && natural.w && natural.h) {
    const k = Math.max(width / natural.w, height / natural.h);   // object-fit: cover
    sw = natural.w * k;
    sh = natural.h * k;
    ox = (width - sw) / 2;
    oy = (height - sh) / 2;
  }
  for (const hand of hands) {
    const p = hand.points.map(([x, y]) => [ox + x * sw, oy + y * sh] as const);
    ctx.lineCap = "round";
    ctx.lineWidth = r * 1.1;
    for (const [a, b] of HAND_CONNECTIONS) {
      if (!p[a] || !p[b]) continue;
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.beginPath();
      ctx.moveTo(p[a][0], p[a][1]);
      ctx.lineTo(p[b][0], p[b][1]);
      ctx.stroke();
    }
    p.forEach(([x, y], i) => {
      const tip = i % 4 === 0 && i > 0;
      ctx.beginPath();
      ctx.arc(x, y, tip ? r * 1.5 : r, 0, Math.PI * 2);
      ctx.fillStyle = "#14B8A6";
      ctx.shadowColor = "#14B8A6";
      ctx.shadowBlur = tip ? 10 : 4;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#FFFFFF";
      ctx.stroke();
    });
  }
}

/** Speech output with the browser's built-in voices (prefers Indian English). */
export function speak(text: string) {
  if (!("speechSynthesis" in window) || !text.trim()) return false;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.trim());
  const voices = window.speechSynthesis.getVoices();
  u.voice = voices.find((v) => v.lang === "en-IN") ?? voices.find((v) => v.lang.startsWith("en")) ?? null;
  u.rate = 0.95;
  window.speechSynthesis.speak(u);
  return true;
}
